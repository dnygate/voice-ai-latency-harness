"""Stage 2 and 3 validation: real UDP sockets and real send/receive timestamping.

Run:  python -m harness.loopback
      python -m harness.loopback --delays 100,300,600 --calls 5 --json results/stage2.json

Stage 3 adds a second host across a real NIC. On the responder host:

      python -m harness.loopback --responder --listen 0.0.0.0:9000

and on the caller host, once with no impairment and once with netem applied on the
responder's egress:

      python -m harness.loopback --peer 10.0.1.5:9000 --json baseline.json
      python -m harness.loopback --peer 10.0.1.5:9000 --json netem.json
      python -m harness.loopback --compare baseline.json netem.json --expect-netem-ms 50

Add --save-captures DIR to any sweep to keep the raw captures alongside the derived
rows, so that a later question (what would playout have been at a deeper target?) is
answered by re-running the analyser locally rather than by restarting two hosts.

The stage 3 gate is differential and that is the point of it. netem on the responder's
egress delays only the return leg, so the measured residual is the inter-host round trip
plus the injected delay, and subtracting the baseline residual cancels the round trip
without anyone needing to know what it was. See `compare_sweeps`.

Stage 1 proved the metric arithmetic against a synthetically constructed capture. It
could not prove that the live path produces trustworthy timestamps, because there was
no live path. This stage sends real RTP over real sockets on the loopback interface and
runs the identical analyser over the result. Loopback transit is tens of microseconds,
which is negligible against the ~2.4 ms instrument precision, so ground truth remains
essentially the programmed delay.

What this stage actually tests is the thing stage 1 could not: whether *this machine*
can pace a 20 ms frame grid accurately enough for the measurement to mean anything.
That is a property of the host, not of the code, and it varies enormously -- a laptop
on battery with an aggressive scheduler can fail where the same code on a tuned server
passes easily. Run it on any machine before trusting a number from it.

Pacing
------
time.sleep() cannot be relied upon for 20 ms grids: granularity and scheduler wake-up
latency both exceed the budget, badly so on macOS and on any machine that is
power-managed. So pacing is hybrid: sleep until shortly before the deadline, then spin
on the monotonic clock. Spinning burns a core, which is the correct trade here -- the
harness must not be the largest source of error in its own measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import g711
from .analyse import analyse_capture
from .record import Capture, CaptureHeader, Packet, write_capture
from .synth import annotated_prompt, speechlike

SPIN_MARGIN_NS = 1_500_000  # start spinning 1.5 ms before the deadline
RTP_HDR = "!BBHII"


def now_ns() -> int:
    return time.monotonic_ns()


def sleep_until(deadline_ns: int) -> None:
    """Hybrid sleep-then-spin. Accurate to microseconds at the cost of a busy core."""
    remaining = deadline_ns - now_ns() - SPIN_MARGIN_NS
    if remaining > 0:
        time.sleep(remaining / 1e9)
    while now_ns() < deadline_ns:
        pass


def pack_rtp(pt: int, seq: int, ts: int, ssrc: int, marker: bool, payload: bytes) -> bytes:
    b0 = 0x80  # version 2, no padding, no extension, no CSRC
    b1 = (0x80 if marker else 0x00) | (pt & 0x7F)
    return struct.pack(RTP_HDR, b0, b1, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc) + payload


def unpack_rtp(datagram: bytes) -> tuple[int, int, int, int, bool, bytes]:
    b0, b1, seq, ts, ssrc = struct.unpack(RTP_HDR, datagram[:12])
    if (b0 >> 6) != 2:
        raise ValueError("not RTP version 2")
    return b1 & 0x7F, seq, ts, ssrc, bool(b1 & 0x80), datagram[12:]


@dataclass
class PacingReport:
    n: int = 0
    p50_ms: float = float("nan")
    p99_ms: float = float("nan")
    max_ms: float = float("nan")

    @staticmethod
    def measure(send_times_ns: list[int], ptime_ms: float) -> "PacingReport":
        if len(send_times_ns) < 3:
            return PacingReport()
        d = np.diff(np.array(send_times_ns, dtype=np.float64)) / 1e6
        dev = np.abs(d - ptime_ms)
        return PacingReport(len(send_times_ns), float(np.percentile(dev, 50)),
                            float(np.percentile(dev, 99)), float(dev.max()))


@dataclass
class ResponderDiag:
    """The reference responder's own timing error, so harness error can be separated
    from responder error rather than the two being conflated in one residual."""
    triggered: bool = False
    intended_emit_ns: int = 0
    actual_emit_ns: int = 0
    frames_received: int = 0
    frames_sent: int = 0
    pacing: PacingReport = field(default_factory=PacingReport)

    @property
    def self_error_ms(self) -> float:
        if not self.triggered:
            return float("nan")
        return (self.actual_emit_ns - self.intended_emit_ns) / 1e6

    @staticmethod
    def from_remote(d: dict) -> "ResponderDiag":
        """Rebuild from a stage 3 control-plane DIAG body.

        Only the self error crosses the wire, not the two absolute instants it came from,
        because those sit on the responder's clock and mean nothing on this one. They are
        reconstituted with their difference preserved so that `self_error_ms` reports the
        value the responder actually measured.
        """
        diag = ResponderDiag(
            triggered=bool(d.get("triggered")),
            frames_received=int(d.get("frames_received") or 0),
            frames_sent=int(d.get("frames_sent") or 0),
        )
        err_ms = d.get("self_error_ms")
        if err_ms is not None:
            diag.actual_emit_ns = int(round(float(err_ms) * 1e6))
        pacing_max = d.get("pacing_max_ms")
        if pacing_max is not None:
            diag.pacing = PacingReport(max_ms=float(pacing_max))
        return diag


class ReferenceResponder(threading.Thread):
    """Oracle SUT: replies exactly `delay_ms` after the caller's final speech sample.

    It is given the prompt's speech_end_sample. That is legitimate because this is a
    calibration source, not a system under test: it must reply at a *known* instant, so
    it compensates for where within the arriving frame the speech boundary fell. A real
    system has to find that boundary itself, and the lag of doing so is precisely what
    the harness exists to measure.
    """

    def __init__(self, sock: socket.socket, peer: tuple[str, int], delay_ms: float,
                 speech_end_sample: int, response: np.ndarray, codec: str,
                 ptime_ms: float, floor_dbov: float = -62.0, seed: int = 0):
        super().__init__(daemon=True)
        self.sock, self.peer, self.delay_ms = sock, peer, delay_ms
        self.speech_end, self.response = speech_end_sample, response
        self.codec, self.ptime_ms = codec, ptime_ms
        self.spf = int(round(g711.SAMPLE_RATE * ptime_ms / 1000.0))
        self.floor_dbov, self.seed = floor_dbov, seed
        self.diag = ResponderDiag()
        self.stop_flag = threading.Event()

    def run(self) -> None:
        from .synth import noise_at

        self.sock.settimeout(0.05)
        seq = 0
        ssrc = 0x2222_2222
        pt = g711.payload_type(self.codec)
        # Comfort noise from the moment the call is up, so the analyser has a noise
        # floor to estimate against. A stream that is digitally silent until the
        # response begins would make onset detection trivially easy and unrealistic.
        cn = noise_at(self.ptime_ms / 1000.0, self.floor_dbov, seed=self.seed + 3)
        cn_payload = g711.encode(cn[: self.spf], self.codec)

        next_cn = now_ns()
        emit_at: int | None = None
        sends: list[int] = []

        while not self.stop_flag.is_set():
            try:
                data, _ = self.sock.recvfrom(4096)
                t_arr = now_ns()
                _, _, ts, _, _, payload = unpack_rtp(data)
                self.diag.frames_received += 1
                n = len(payload) // g711.bytes_per_sample(self.codec)
                if emit_at is None and ts + n > self.speech_end - 1:
                    # The arriving frame contains the final speech sample. Compensate for
                    # its offset within the frame so the emission instant is exactly
                    # t0 + delay, where t0 is the transmission time of that sample.
                    within = (self.speech_end - 1) - ts
                    # t0 is the transmission instant of the final speech sample, which is
                    # the frame's send time PLUS its offset within the frame. Since t_arr
                    # is that send time plus transit, the emission instant is
                    # t_arr + delay + within. Getting this sign wrong produces a residual
                    # of -2*within, i.e. up to -40 ms at 20 ms frames -- exactly the kind
                    # of systematic error a synthetic capture cannot catch.
                    emit_at = t_arr + int(self.delay_ms * 1e6) + int(within * 1e9 / g711.SAMPLE_RATE)
                    self.diag.triggered = True
                    self.diag.intended_emit_ns = emit_at
            except socket.timeout:
                if emit_at is None and self.diag.frames_received:
                    break

            if emit_at is not None:
                # Dedicated emit path. Crucially this does NOT stay inside the receive
                # loop: checking the emit deadline once per received frame quantises
                # emission to the frame period, adding a uniform 0-20 ms error. That
                # error is invisible if every test delay is a multiple of the frame time,
                # because the deadline then lands on a frame boundary. Nothing further
                # needs to be received after the trigger, so the loop is left behind.
                frame_ns = int(self.ptime_ms * 1e6)
                while now_ns() < emit_at - frame_ns:
                    sleep_until(min(next_cn, emit_at - frame_ns))
                    if now_ns() >= emit_at - frame_ns:
                        break
                    self.sock.sendto(pack_rtp(pt, seq, seq * self.spf, ssrc, seq == 0,
                                              cn_payload), self.peer)
                    seq += 1
                    self.diag.frames_sent += 1
                    next_cn += frame_ns
                sleep_until(emit_at)
                self.diag.actual_emit_ns = now_ns()
                deadline = self.diag.actual_emit_ns
                for i in range(0, len(self.response), self.spf):
                    fr = self.response[i : i + self.spf]
                    if len(fr) < self.spf:
                        fr = np.pad(fr, (0, self.spf - len(fr)))
                    sleep_until(deadline)
                    sends.append(now_ns())
                    self.sock.sendto(pack_rtp(pt, seq, seq * self.spf, ssrc, seq == 0,
                                              g711.encode(fr, self.codec)), self.peer)
                    seq += 1
                    self.diag.frames_sent += 1
                    deadline += frame_ns
                break

            t = now_ns()
            if t >= next_cn:
                self.sock.sendto(pack_rtp(pt, seq, seq * self.spf, ssrc, seq == 0,
                                          cn_payload), self.peer)
                seq += 1
                self.diag.frames_sent += 1
                next_cn += int(self.ptime_ms * 1e6)

        self.diag.pacing = PacingReport.measure(sends, self.ptime_ms)


# ---------------------------------------------------------------------------
# Stage 3 control plane
#
# The responder needs the prompt's speech_end_sample to reply at a known instant, which
# in stage 2 it simply read from the caller's own annotation object. Across two hosts
# that value has to be carried explicitly, so there is a small control protocol beside
# the media stream. It is deliberately separate from the RTP port: mixing JSON into the
# media socket would put non-RTP datagrams in front of the analyser's reassembly.
# ---------------------------------------------------------------------------

CONTROL_MAGIC = "vlh-ctl/1"
CTL_ATTEMPTS = 5
CTL_TIMEOUT_S = 0.6


class ControlPlaneError(RuntimeError):
    """The responder could not be reached or did not answer.

    Raised rather than returning an empty capture, so that a control failure can never be
    counted as a call in which the system under test declined to respond.
    """


def _response_signal(seed: int) -> np.ndarray:
    """The response the reference responder emits.

    Defined here once because both hosts must generate an identical signal from the seed
    alone. Duplicating the expression in the caller and the responder would let them
    drift silently, and a mismatch would show up as an onset-detection anomaly rather
    than as the configuration error it actually was.
    """
    return speechlike(1.2, seed=seed + 500, onset="hard")


def _ctl_encode(kind: str, body: dict) -> bytes:
    return json.dumps({"magic": CONTROL_MAGIC, "kind": kind, **body}).encode("utf-8")


def _ctl_decode(data: bytes) -> dict | None:
    try:
        msg = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return msg if isinstance(msg, dict) and msg.get("magic") == CONTROL_MAGIC else None


def _ctl_request(sock: socket.socket, addr: tuple[str, int], kind: str, body: dict,
                 expect: str, attempts: int = CTL_ATTEMPTS,
                 timeout_s: float = CTL_TIMEOUT_S) -> dict | None:
    """Send a control datagram and wait for `expect`, retransmitting on loss.

    Retransmission matters here for a reason beyond ordinary reliability. The control
    channel shares the impaired path with the media, so under `netem loss 1%` an
    unacknowledged datagram is an expected event rather than a fault. A lost HELLO would
    otherwise produce a call in which the responder never speaks, and at the analyser
    that is indistinguishable from a system under test which failed to answer. Control
    failures and measurement failures must not be able to masquerade as each other.
    """
    sock.settimeout(timeout_s)
    for _ in range(attempts):
        sock.sendto(_ctl_encode(kind, body), addr)
        try:
            data, _ = sock.recvfrom(8192)
        except socket.timeout:
            continue
        msg = _ctl_decode(data)
        if msg and msg.get("kind") == expect:
            return msg
    return None


def serve_responder(listen_host: str, listen_port: int, floor_dbov: float = -62.0) -> int:
    """Run the reference responder as a standalone process for stage 3.

    One call at a time and no concurrency, deliberately. This is a calibration source
    whose whole job is to reply at an exactly known instant, and a queue of overlapping
    calls would introduce a scheduling term into the quantity it exists to hold constant.
    """
    ctl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctl.bind((listen_host, listen_port))
    # Flushed explicitly throughout. This process runs for the length of a sweep, usually
    # watched over SSH with output redirected to a file, and Python buffers stdout when it
    # is not a terminal, so without this the operator sees nothing until the process ends.
    print(f"host: {platform.platform()}  python {platform.python_version()}", flush=True)
    print(f"reference responder listening on {listen_host}:{listen_port}", flush=True)
    print("one call at a time; ctrl-c to stop\n", flush=True)

    last: dict = {}
    while True:
        data, src = ctl.recvfrom(8192)
        msg = _ctl_decode(data)
        if not msg:
            continue

        if msg.get("kind") == "DIAG?":
            ctl.sendto(_ctl_encode("DIAG", {"diag": last}), src)
            continue
        if msg.get("kind") != "HELLO":
            continue

        spec = msg.get("spec") or {}
        media = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        media.bind((listen_host, 0))
        media_port = media.getsockname()[1]
        # Acknowledge before starting work so the caller can begin transmitting. The
        # caller's media port comes from the HELLO body while its address comes from the
        # control datagram's source, which keeps the two consistent without either side
        # having to be told its own externally visible address.
        ctl.sendto(_ctl_encode("READY", {"media_port": media_port}), src)

        peer = (src[0], int(spec["media_port"]))
        responder = ReferenceResponder(
            media, peer, float(spec["delay_ms"]), int(spec["speech_end_sample"]),
            _response_signal(int(spec["seed"])), spec["codec"], float(spec["ptime_ms"]),
            floor_dbov=floor_dbov, seed=int(spec["seed"]))
        responder.start()

        # Bounded wait, so a caller that never transmits (its READY was the datagram
        # that got lost) does not strand this process. Returning to the control loop lets
        # the caller's retransmitted HELLO be served on a fresh media port.
        limit = time.monotonic() + float(spec["delay_ms"]) / 1000.0 + 3.0
        while responder.is_alive() and time.monotonic() < limit:
            time.sleep(0.02)
        responder.stop_flag.set()
        responder.join(timeout=1.0)
        media.close()

        last = {
            "call_id": spec.get("call_id"),
            "triggered": responder.diag.triggered,
            "frames_received": responder.diag.frames_received,
            "frames_sent": responder.diag.frames_sent,
            "self_error_ms": (None if not responder.diag.triggered
                              else round(responder.diag.self_error_ms, 4)),
            "pacing_max_ms": (None if not np.isfinite(responder.diag.pacing.max_ms)
                              else round(responder.diag.pacing.max_ms, 4)),
        }
        print(f"  {spec.get('call_id')}: delay {spec.get('delay_ms')} ms  "
              f"rx {last['frames_received']}  tx {last['frames_sent']}  "
              f"triggered {last['triggered']}  self-err {last['self_error_ms']}",
              flush=True)


def run_call(delay_ms: float, codec: str = "pcmu", ptime_ms: float = 20.0,
             seed: int = 0, host: str = "127.0.0.1",
             call_id: str = "loopback",
             peer: tuple[str, int] | None = None) -> tuple[Capture, ResponderDiag]:
    """Place one call and return the capture plus responder diagnostics.

    With `peer` unset this is stage 2 and both endpoints live in this process on loopback.
    With `peer` set to a remote responder's control address it is stage 3, where the only
    changes are that the responder runs on another host and learns speech_end_sample over
    the control channel. The transmit path, the receive path and the analyser are shared
    unchanged between the two, which is what makes their results comparable at all.
    """
    spf = int(round(g711.SAMPLE_RATE * ptime_ms / 1000.0))
    pt = g711.payload_type(codec)

    prompt, _, ann = annotated_prompt(seed=seed)

    caller = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # A remote responder replies to whatever address it saw us from, so the receive
    # socket cannot be pinned to loopback in stage 3.
    caller.bind(("0.0.0.0" if peer else host, 0))
    caller_addr = caller.getsockname()

    responder: ReferenceResponder | None = None
    sut: socket.socket | None = None
    ctl: socket.socket | None = None

    if peer is None:
        sut = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sut.bind((host, 0))
        sut_addr = sut.getsockname()
        responder = ReferenceResponder(sut, caller_addr, delay_ms, ann.speech_end_sample,
                                       _response_signal(seed), codec, ptime_ms, seed=seed)
    else:
        ctl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ctl.bind(("0.0.0.0", 0))
        ready = _ctl_request(ctl, peer, "HELLO", {"spec": {
            "call_id": call_id,
            "delay_ms": float(delay_ms),
            "speech_end_sample": int(ann.speech_end_sample),
            "codec": codec,
            "ptime_ms": float(ptime_ms),
            "seed": int(seed),
            "media_port": caller_addr[1],
        }}, expect="READY")
        if ready is None:
            caller.close()
            ctl.close()
            raise ControlPlaneError(
                f"responder at {peer[0]}:{peer[1]} did not acknowledge after "
                f"{CTL_ATTEMPTS} attempts")
        sut_addr = (peer[0], int(ready["media_port"]))

    packets: list[Packet] = []
    lock = threading.Lock()
    stop_rx = threading.Event()

    def rx_loop() -> None:
        caller.settimeout(0.2)
        while not stop_rx.is_set():
            try:
                data, _ = caller.recvfrom(4096)
            except socket.timeout:
                continue
            t = now_ns()
            try:
                p_pt, seq, ts, ssrc, marker, payload = unpack_rtp(data)
            except ValueError:
                continue
            with lock:
                packets.append(Packet("rx", t, time.time_ns(), seq, ts, ssrc,
                                      p_pt, marker, payload))

    rx_thread = threading.Thread(target=rx_loop, daemon=True)
    # Receive before transmit. In stage 3 the remote responder begins emitting comfort
    # noise as soon as it has acknowledged, so starting the receive loop late would drop
    # the frames the analyser needs in order to estimate a noise floor.
    rx_thread.start()
    if responder is not None:
        responder.start()

    frames = [prompt[i : i + spf] for i in range(0, len(prompt), spf)]
    ssrc_tx = 0x1111_1111
    deadline = now_ns()
    sends: list[int] = []
    for i, fr in enumerate(frames):
        if len(fr) < spf:
            fr = np.pad(fr, (0, spf - len(fr)))
        sleep_until(deadline)
        t = now_ns()
        caller.sendto(pack_rtp(pt, i, i * spf, ssrc_tx, i == 0,
                               g711.encode(fr, codec)), sut_addr)
        sends.append(t)
        with lock:
            packets.append(Packet("tx", t, time.time_ns(), i, i * spf, ssrc_tx,
                                  pt, i == 0, g711.encode(fr, codec)))
        deadline += int(ptime_ms * 1e6)

    # Let the response arrive, then wind down.
    time.sleep(max(0.5, delay_ms / 1000.0 + 0.8))
    if responder is not None:
        responder.stop_flag.set()
    stop_rx.set()
    if responder is not None:
        responder.join(timeout=1.0)
    rx_thread.join(timeout=1.0)

    if responder is not None:
        diag = responder.diag
    else:
        # Retrieve the responder's own timing error from the far host. Without it a
        # stage 3 residual conflates harness error with responder error, and those two
        # have to stay separable for the result to say anything about the instrument.
        got = _ctl_request(ctl, peer, "DIAG?", {}, expect="DIAG")
        diag = ResponderDiag.from_remote((got or {}).get("diag") or {})

    caller.close()
    if sut is not None:
        sut.close()
    if ctl is not None:
        ctl.close()

    tx_pacing = PacingReport.measure(sends, ptime_ms)
    notes = {"platform": platform.platform(), "python": platform.python_version(),
             "tx_pacing_p50_ms": tx_pacing.p50_ms, "tx_pacing_p99_ms": tx_pacing.p99_ms,
             "tx_pacing_max_ms": tx_pacing.max_ms,
             "responder_self_error_ms": diag.self_error_ms}
    if peer is not None:
        notes["peer"] = f"{peer[0]}:{peer[1]}"
    hdr = CaptureHeader(
        call_id=call_id, prompt_id="synthetic", speech_end_sample=ann.speech_end_sample,
        codec=codec, ptime_ms=ptime_ms,
        sut_label="remote-reference-responder" if peer else "loopback-reference-responder",
        run_id="stage3" if peer else "stage2", ground_truth_ms=float(delay_ms),
        notes=notes,
    )
    with lock:
        return Capture(header=hdr, packets=list(packets)), diag


def _save_capture(cap: Capture, directory: str) -> tuple[str, str]:
    """Write the raw capture and return (path, sha256 of the file).

    Until 2026-09-03 the stage 2 and 3 sweeps kept only derived rows and discarded the
    audio, which broke the project's first rule for precisely the runs that cost money to
    repeat: the netem sweep could not be asked what playout MRL would have been at a
    deeper buffer target because the samples were gone. With the capture kept, that is a
    local command. The hash is recorded in the row so the JSON vouches for the file it
    points at.
    """
    path = Path(directory) / f"{cap.header.call_id}.jsonl.gz"
    write_capture(path, cap)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _status(row: dict) -> str:
    """One call's QC status, for the console.

    Advisory flags are shown even when the call passes. They do not invalidate a
    measurement, but they turn it from a point estimate into an upper bound, and the flag
    has to travel with the number. Printing flags only on failure produced a real
    incident: a stage 3 sweep reported forty clean-looking calls while thirty-eight of
    them carried a late-discard advisory that existed in the JSON and nowhere a reader
    would see it.
    """
    flags = row.get("qc_flags") or []
    if not row.get("qc_ok"):
        return "QC FAIL " + str(flags)
    return "ok" if not flags else "ok, advisory " + ",".join(flags)


def compare_sweeps(baseline_path: str, netem_path: str, expect_netem_ms: float,
                   tol_ms: float = 5.0) -> int:
    """Stage 3 gate: the difference between two sweeps must equal the injected delay.

    netem on the responder's egress delays the return leg only, so a measured residual is
    the inter-host round trip, plus the injected delay, plus whatever error the instrument
    contributes. Differencing two sweeps over the same host pair cancels the round trip
    and the responder's own emission error, since both are common to the two runs, and
    what remains is the injected delay. The gate is therefore differential and needs no
    independent measurement of the path, which is the whole reason it is trustworthy on a
    virtualised host where the path is not under our control.

    The tolerance is not a free parameter. What survives the subtraction is the
    instrument's own bias, validated in stage 1 at -0.40 ms with p95 |error| 2.38 ms, so
    the default sits at roughly twice that p95. Do not raise it to obtain a pass: if the
    difference misses by more than this, either netem is not doing what was declared or
    the instrument is wrong, and the discriminators below distinguish those cases.
    """
    def load(path: str) -> tuple[np.ndarray, np.ndarray, int, dict]:
        doc = json.loads(Path(path).read_text())
        all_rows = doc.get("rows", [])
        rows = [r for r in all_rows
                if r.get("qc_ok") and r.get("residual_ms") is not None]
        resid = np.array([r["residual_ms"] for r in rows], dtype=float)
        self_err = np.array([r.get("responder_self_error_ms") or 0.0 for r in rows],
                            dtype=float)
        return resid, self_err, len(all_rows), doc

    b_resid, b_self, b_total, b_doc = load(baseline_path)
    n_resid, n_self, n_total, n_doc = load(netem_path)

    print(f"baseline : {baseline_path}")
    print(f"           {b_resid.size}/{b_total} usable, host {b_doc.get('host')}")
    print(f"netem    : {netem_path}")
    print(f"           {n_resid.size}/{n_total} usable, host {n_doc.get('host')}")
    print(f"declared injected delay: {expect_netem_ms:g} ms")
    b_pt, n_pt = b_doc.get("playout_target_ms"), n_doc.get("playout_target_ms")
    if b_pt is not None and n_pt is not None and b_pt != n_pt:
        # The gate is on ingress, which the buffer target cannot touch, so this is a
        # note rather than a failure. It is still a difference in conditions and a
        # reader of the two files deserves to be told.
        print(f"note: playout targets differ ({b_pt:g} vs {n_pt:g} ms); ingress gate "
              f"is unaffected, playout figures are not comparable across the two")
    print()

    if b_resid.size < 3 or n_resid.size < 3:
        print("STAGE 3 NOT PASSED: fewer than three usable calls in one of the sweeps, so")
        print("  no difference can be estimated. Check caller pacing and the control")
        print("  plane before re-running.")
        return 1

    diff = float(n_resid.mean() - b_resid.mean())
    err = diff - expect_netem_ms
    se = float(np.sqrt(b_resid.var(ddof=1) / b_resid.size
                       + n_resid.var(ddof=1) / n_resid.size))

    print(f"baseline residual : {b_resid.mean():+8.2f} ms  sd {b_resid.std(ddof=1):6.2f} ms")
    print(f"netem residual    : {n_resid.mean():+8.2f} ms  sd {n_resid.std(ddof=1):6.2f} ms")
    print(f"difference        : {diff:+8.2f} ms  (standard error {se:.2f} ms)")
    print(f"error vs declared : {err:+8.2f} ms  (tolerance {tol_ms:g} ms)")
    print(f"responder self-error, baseline {b_self.mean():+.3f} ms, "
          f"netem {n_self.mean():+.3f} ms  (cancels in the difference)")

    # Underpowered is a distinct outcome from failed. With paretonormal jitter the netem
    # sweep's spread is large by design, so a small number of calls can leave the
    # standard error wider than the tolerance, at which point the run cannot support
    # either verdict and saying "passed" would be dishonest.
    underpowered = 2.0 * se > tol_ms

    if abs(err) <= tol_ms and not underpowered:
        print(f"\nstage 3 passed: the instrument recovers a {expect_netem_ms:g} ms "
              f"injected delay to within {abs(err):.2f} ms across a real NIC and two "
              f"hosts")
        return 0

    if abs(err) <= tol_ms and underpowered:
        print(f"\nSTAGE 3 INCONCLUSIVE: the difference is within tolerance but its "
              f"standard error ({se:.2f} ms) is too wide for the {tol_ms:g} ms gate to "
              f"mean anything.")
        needed = int(np.ceil((2.0 * se / tol_ms) ** 2 * min(b_resid.size, n_resid.size)))
        print(f"  Re-run both sweeps with roughly {needed} usable calls each. Do not "
              f"widen the tolerance instead.")
        return 1

    print("\nSTAGE 3 NOT PASSED")
    if b_resid.mean() < 0:
        print(f"  The baseline residual is negative ({b_resid.mean():+.2f} ms). A response "
              f"cannot arrive before the programmed delay plus minimum transit, so this is "
              f"a definitional or arithmetic error rather than a path effect. Check the "
              f"sign of the frame-offset term in the responder and in t0 derivation.")
    elif abs(diff) < tol_ms:
        print(f"  The two sweeps are indistinguishable ({diff:+.2f} ms apart), so netem "
              f"was almost certainly not in force during the second run. Confirm with "
              f"`tc qdisc show dev eth0` on the responder host, and check the qdisc is on "
              f"the interface the media actually traverses.")
    elif abs(diff - 2.0 * expect_netem_ms) < tol_ms:
        print(f"  The difference is close to twice the declared delay ({diff:+.2f} ms). "
              f"netem is being applied to both directions, most likely an ingress qdisc "
              f"or ifb redirect alongside the egress one, so the delay is counted on each "
              f"leg. Remove one.")
    else:
        print(f"  The difference misses the declared delay by {err:+.2f} ms, which is "
              f"neither zero nor a doubling. Check that the declared value matches what "
              f"`tc` was actually given, and that no other qdisc is queued on the same "
              f"interface.")
    if underpowered:
        print(f"  Note also that the standard error is {se:.2f} ms, wide relative to the "
              f"{tol_ms:g} ms gate, so more calls are needed before this verdict is firm.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delays", default="137,213,353,457,806",
                    help="comma-separated programmed delays in ms. Defaults are "
                         "deliberately NOT multiples of the frame time: commensurate "
                         "delays put the deadline on a frame boundary and hide "
                         "frame-quantisation errors entirely.")
    ap.add_argument("--calls", type=int, default=4, help="calls per delay")
    ap.add_argument("--codec", default="pcmu")
    ap.add_argument("--ptime", type=float, default=20.0)
    ap.add_argument("--json")
    ap.add_argument("--playout-target-ms", type=float, default=40.0,
                    help="de-jitter buffer target for playout MRL. A parameter of the "
                         "analysis, not of the capture: the same saved capture can be "
                         "re-analysed under another value.")
    ap.add_argument("--save-captures", metavar="DIR",
                    help="keep every raw capture in DIR as <call_id>.jsonl.gz, hashed "
                         "into the JSON rows")
    ap.add_argument("--gate-ms", type=float, default=6.0,
                    help="max acceptable |residual bias| on this host")
    ap.add_argument("--responder", action="store_true",
                    help="stage 3: run as the reference responder and serve calls")
    ap.add_argument("--listen", default="0.0.0.0:9000",
                    help="stage 3 responder control address, HOST:PORT")
    ap.add_argument("--peer",
                    help="stage 3: remote responder control address, HOST:PORT. Without "
                         "this the sweep runs stage 2 on loopback.")
    ap.add_argument("--compare", nargs=2, metavar=("BASELINE", "NETEM"),
                    help="stage 3 gate: difference two sweep JSON files and check it "
                         "against the injected delay")
    ap.add_argument("--expect-netem-ms", type=float, default=50.0,
                    help="delay declared to netem on the responder's egress")
    ap.add_argument("--stage3-tol-ms", type=float, default=5.0,
                    help="stage 3 differential tolerance, roughly twice the instrument's "
                         "stage 1 p95 |error|. Do not raise it to obtain a pass.")
    args = ap.parse_args()

    if args.compare:
        return compare_sweeps(args.compare[0], args.compare[1],
                              args.expect_netem_ms, args.stage3_tol_ms)

    if args.responder:
        lhost, _, lport = args.listen.rpartition(":")
        return serve_responder(lhost or "0.0.0.0", int(lport))

    peer: tuple[str, int] | None = None
    if args.peer:
        phost, _, pport = args.peer.rpartition(":")
        peer = (phost, int(pport))

    delays = [float(x) for x in args.delays.split(",")]
    print(f"host: {platform.platform()}  python {platform.python_version()}")
    if peer:
        print(f"peer: {peer[0]}:{peer[1]}")
    print(f"stage {'3 (remote peer)' if peer else '2 (loopback)'}: {len(delays)} delays "
          f"x {args.calls} calls, {args.codec} @ {args.ptime:g} ms, "
          f"playout target {args.playout_target_ms:g} ms\n")

    rows = []
    for d in delays:
        for c in range(args.calls):
            try:
                cap, diag = run_call(d, codec=args.codec, ptime_ms=args.ptime,
                                     seed=int(d) + c, call_id=f"lb-{int(d)}-{c}",
                                     peer=peer)
            except ControlPlaneError as exc:
                # Recorded rather than fatal, so a single lost handshake does not destroy
                # a sweep, and counted under its own flag so it can never be tallied as a
                # call in which the responder declined to answer.
                rows.append({"delay_ms": d, "call": c, "control_failed": True,
                             "mrl_ingress_ms": None, "mrl_playout_ms": None,
                             "residual_ms": None, "responder_self_error_ms": None,
                             "qc_ok": False, "qc_flags": ["control_plane"]})
                print(f"  delay {d:>6.0f} ms  call {c}  CONTROL FAILURE: {exc}")
                continue
            res = analyse_capture(cap, playout_target_ms=args.playout_target_ms)
            v = res.variants["headline"]
            saved = _save_capture(cap, args.save_captures) if args.save_captures else None
            self_err = (None if not np.isfinite(diag.self_error_ms)
                        else round(diag.self_error_ms, 3))
            rows.append({
                "delay_ms": d, "call": c,
                "mrl_ingress_ms": None if not np.isfinite(v.mrl_ms) else round(v.mrl_ms, 3),
                # Both MRLs travel together, always. Ingress alone understates what a
                # caller waits and playout alone confounds the responder with our buffer.
                "mrl_playout_ms": (None if not np.isfinite(v.mrl_playout_ms)
                                   else round(v.mrl_playout_ms, 3)),
                "residual_ms": None if not np.isfinite(v.mrl_ms) else round(v.mrl_ms - d, 3),
                "responder_self_error_ms": self_err,
                "tx_pacing_p99_ms": round(res.qc.tx_pacing_p99_ms, 3),
                "tx_pacing_max_ms": round(res.qc.tx_pacing_max_dev_ms, 3),
                "rx_loss_pct": round(res.qc.rx_loss_pct, 3),
                "playout_target_ms": res.qc.playout_target_ms,
                "capture_path": saved[0] if saved else None,
                "capture_sha256": saved[1] if saved else None,
                "qc_ok": res.qc.ok(), "qc_flags": res.qc.flags,
                "tx_packets": res.qc.tx_packets, "rx_packets": res.qc.rx_packets,
            })
            print(f"  delay {d:>6.0f} ms  call {c}  "
                  f"ingress {rows[-1]['mrl_ingress_ms']}  "
                  f"playout {rows[-1]['mrl_playout_ms']}  "
                  f"residual {rows[-1]['residual_ms']}  "
                  f"self-err {'n/a' if self_err is None else f'{self_err:+.2f}'}  "
                  f"tx pacing max {rows[-1]['tx_pacing_max_ms']:.2f}  "
                  f"{_status(rows[-1])}")

    control_failures = sum(1 for r in rows if r.get("control_failed"))
    usable = [r for r in rows if r["qc_ok"] and r["residual_ms"] is not None]
    resid = np.array([r["residual_ms"] for r in usable], dtype=float)
    selferr = np.array([r["responder_self_error_ms"] or 0.0 for r in usable], dtype=float)
    pac = np.array([r.get("tx_pacing_max_ms", float("nan")) for r in rows], dtype=float)

    print(f"\nusable {len(usable)}/{len(rows)} calls")
    if control_failures:
        print(f"control-plane failures: {control_failures} (handshake never acknowledged; "
              f"these are not calls the responder declined to answer)")
    if resid.size:
        print(f"residual: bias {resid.mean():+.2f} ms  sd {resid.std(ddof=1):.2f} ms  "
              f"p95|e| {np.percentile(np.abs(resid), 95):.2f} ms  "
              f"max|e| {np.abs(resid).max():.2f} ms")
        print(f"of which responder self-error: bias {selferr.mean():+.2f} ms  "
              f"max {np.abs(selferr).max():.2f} ms  (subtract to isolate harness error)")
    if np.isfinite(pac).any():
        print(f"tx pacing worst deviation across all calls: {np.nanmax(pac):.2f} ms "
              f"(QC blocks above 5 ms)")

    # Advisory flags are counted and stated at the end as well as per call, because a
    # summary that reports only the distribution invites the reader to quote a figure
    # while leaving behind the caveat that makes it an upper bound.
    advisory: dict[str, int] = {}
    for r in rows:
        if r.get("qc_ok"):
            for f in r.get("qc_flags") or []:
                advisory[f] = advisory.get(f, 0) + 1
    if advisory:
        print("advisory flags on usable calls: "
              + ", ".join(f"{k} on {v}/{len(usable)}" for k, v in sorted(advisory.items())))
        print("  These remain valid measurements of a degraded channel. Where the onset "
              "frame\n  was itself deferred, the figure is an upper bound rather than a "
              "point estimate,\n  so the flag belongs with any number quoted from this "
              "run.")
        if "high_late_discard" in advisory and advisory["high_late_discard"] > len(usable) / 2:
            print("  Late discard on most calls means the playout target is too shallow "
                  "for this\n  channel's jitter. Ingress figures are unaffected; playout "
                  "figures from this run\n  are upper bounds and a deeper target should be "
                  "stated before quoting them.")

    verdict: list[str] = []
    # Systematic means large AND consistent. Both conditions matter: a 0.3 ms bias with
    # 0.03 ms spread is consistent but far too small to be a definitional error, and
    # flagging it would train the reader to ignore this message.
    systematic = bool(resid.size > 2 and abs(resid.mean()) > 2.0
                      and abs(resid.mean()) > 3 * max(resid.std(ddof=1), 0.05))
    if not usable:
        verdict.append("no usable calls: this host cannot pace a frame grid accurately "
                       "enough, or the media path is broken")
    else:
        # The residual-bias gate belongs to stage 2 alone. It encodes the prediction that
        # loopback transit is negligible, so a residual should sit near zero. Across two
        # hosts that prediction is simply false: the residual legitimately carries the
        # inter-host round trip and any injected delay. Applying it here would fail every
        # stage 3 sweep for being correct, so the gate is replaced rather than relaxed,
        # and the replacement is the differential check in `compare_sweeps`.
        if peer is None and abs(resid.mean()) > args.gate_ms:
            verdict.append(f"residual bias {resid.mean():+.2f} ms exceeds "
                           f"{args.gate_ms} ms")
        if len(usable) < 0.8 * len(rows):
            verdict.append(
                f"only {len(usable)}/{len(rows)} calls passed QC -- residual accuracy is "
                f"fine ({resid.mean():+.2f} ms) but this host cannot reliably pace a "
                f"{args.ptime:g} ms grid, so some calls must be discarded")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"host": platform.platform(), "python": platform.python_version(),
             "stage": 3 if peer else 2,
             "playout_target_ms": args.playout_target_ms,
             "captures_dir": args.save_captures,
             "peer": f"{peer[0]}:{peer[1]}" if peer else None,
             "control_failures": control_failures,
             "advisory_flags": advisory,
             "rows": rows, "verdict": verdict}, indent=2))
        print(f"wrote {args.json}")

    stage_name = "STAGE 3 SWEEP" if peer else "STAGE 2"
    if verdict:
        print(f"\n{stage_name} NOT PASSED on this host:")
        for v in verdict:
            print("  -", v)
        if peer is None and systematic:
            print(f"\nThe residual is systematic, not noisy (bias {resid.mean():+.2f} ms vs "
                  f"sd {resid.std(ddof=1):.2f} ms). That is a definitional or arithmetic "
                  f"error somewhere in the chain, not host jitter. Check the sign and "
                  f"frame-offset handling in the responder and in t0 derivation. "
                  f"A residual near a whole or double frame time is a strong hint.")
        elif peer is None:
            print("\nThe residual is noisy rather than systematic, which on a laptop "
                  "usually means power management rather than a code defect. Try: mains "
                  "power, Low Power Mode off, `caffeinate -di` alongside the run.")
        else:
            print("\nOn a cloud instance this usually means a burstable instance type "
                  "whose CPU credits have run down. Check with `python -m "
                  "harness.preflight` on this host before blaming the code.")
        return 1

    if peer is None:
        print("\nstage 2 passed on this host: live capture path produces trustworthy "
              "timestamps here")
        return 0

    # A single sweep cannot pass stage 3. The residual here still contains the inter-host
    # round trip, which nothing in this run measures independently, so the claim only
    # becomes testable once a second sweep under netem is differenced against this one.
    print(f"\nstage 3 sweep collected: {len(usable)}/{len(rows)} calls usable, residual "
          f"bias {resid.mean():+.2f} ms (this includes the inter-host round trip).")
    print("This sweep on its own proves nothing about accuracy. Apply netem on the "
          "responder's egress, run a second sweep, then difference them:")
    print("  python -m harness.loopback --compare baseline.json netem.json "
          f"--expect-netem-ms {args.expect_netem_ms:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
