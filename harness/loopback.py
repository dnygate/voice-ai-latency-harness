"""Stage 2 validation: real UDP sockets, real send/receive timestamping, localhost only.

Run:  python -m harness.loopback
      python -m harness.loopback --delays 100,300,600 --calls 5 --json results/stage2.json

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
from .record import Capture, CaptureHeader, Packet
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


def run_call(delay_ms: float, codec: str = "pcmu", ptime_ms: float = 20.0,
             seed: int = 0, host: str = "127.0.0.1",
             call_id: str = "loopback") -> tuple[Capture, ResponderDiag]:
    """Place one loopback call and return the capture plus responder diagnostics."""
    spf = int(round(g711.SAMPLE_RATE * ptime_ms / 1000.0))
    pt = g711.payload_type(codec)

    prompt, _, ann = annotated_prompt(seed=seed)
    response = speechlike(1.2, seed=seed + 500, onset="hard")

    caller = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sut = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    caller.bind((host, 0))
    sut.bind((host, 0))
    caller_addr, sut_addr = caller.getsockname(), sut.getsockname()

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

    responder = ReferenceResponder(sut, caller_addr, delay_ms, ann.speech_end_sample,
                                   response, codec, ptime_ms, seed=seed)
    rx_thread = threading.Thread(target=rx_loop, daemon=True)
    responder.start()
    rx_thread.start()

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
    responder.stop_flag.set()
    stop_rx.set()
    responder.join(timeout=1.0)
    rx_thread.join(timeout=1.0)
    caller.close()
    sut.close()

    tx_pacing = PacingReport.measure(sends, ptime_ms)
    hdr = CaptureHeader(
        call_id=call_id, prompt_id="synthetic", speech_end_sample=ann.speech_end_sample,
        codec=codec, ptime_ms=ptime_ms, sut_label="loopback-reference-responder",
        run_id="stage2", ground_truth_ms=float(delay_ms),
        notes={"platform": platform.platform(), "python": platform.python_version(),
               "tx_pacing_p50_ms": tx_pacing.p50_ms, "tx_pacing_p99_ms": tx_pacing.p99_ms,
               "tx_pacing_max_ms": tx_pacing.max_ms,
               "responder_self_error_ms": responder.diag.self_error_ms},
    )
    with lock:
        return Capture(header=hdr, packets=list(packets)), responder.diag


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
    ap.add_argument("--gate-ms", type=float, default=6.0,
                    help="max acceptable |residual bias| on this host")
    args = ap.parse_args()

    delays = [float(x) for x in args.delays.split(",")]
    print(f"host: {platform.platform()}  python {platform.python_version()}")
    print(f"loopback stage 2: {len(delays)} delays x {args.calls} calls, "
          f"{args.codec} @ {args.ptime:g} ms\n")

    rows = []
    for d in delays:
        for c in range(args.calls):
            cap, diag = run_call(d, codec=args.codec, ptime_ms=args.ptime,
                                 seed=int(d) + c, call_id=f"lb-{int(d)}-{c}")
            res = analyse_capture(cap)
            v = res.variants["headline"]
            rows.append({
                "delay_ms": d, "call": c,
                "mrl_ingress_ms": None if not np.isfinite(v.mrl_ms) else round(v.mrl_ms, 3),
                "residual_ms": None if not np.isfinite(v.mrl_ms) else round(v.mrl_ms - d, 3),
                "responder_self_error_ms": round(diag.self_error_ms, 3),
                "tx_pacing_p99_ms": round(res.qc.tx_pacing_p99_ms, 3),
                "tx_pacing_max_ms": round(res.qc.tx_pacing_max_dev_ms, 3),
                "qc_ok": res.qc.ok(), "qc_flags": res.qc.flags,
                "tx_packets": res.qc.tx_packets, "rx_packets": res.qc.rx_packets,
            })
            print(f"  delay {d:>6.0f} ms  call {c}  "
                  f"measured {rows[-1]['mrl_ingress_ms']}  "
                  f"residual {rows[-1]['residual_ms']}  "
                  f"responder self-err {rows[-1]['responder_self_error_ms']:+.2f}  "
                  f"tx pacing p99 {rows[-1]['tx_pacing_p99_ms']:.2f} max "
                  f"{rows[-1]['tx_pacing_max_ms']:.2f}  "
                  f"{'ok' if rows[-1]['qc_ok'] else 'QC FAIL ' + str(rows[-1]['qc_flags'])}")

    usable = [r for r in rows if r["qc_ok"] and r["residual_ms"] is not None]
    resid = np.array([r["residual_ms"] for r in usable], dtype=float)
    selferr = np.array([r["responder_self_error_ms"] for r in usable], dtype=float)
    pac = np.array([r["tx_pacing_max_ms"] for r in rows], dtype=float)

    print(f"\nusable {len(usable)}/{len(rows)} calls")
    if resid.size:
        print(f"residual: bias {resid.mean():+.2f} ms  sd {resid.std(ddof=1):.2f} ms  "
              f"p95|e| {np.percentile(np.abs(resid), 95):.2f} ms  "
              f"max|e| {np.abs(resid).max():.2f} ms")
        print(f"of which responder self-error: bias {selferr.mean():+.2f} ms  "
              f"max {np.abs(selferr).max():.2f} ms  (subtract to isolate harness error)")
    print(f"tx pacing worst deviation across all calls: {np.nanmax(pac):.2f} ms "
          f"(QC blocks above 5 ms)")

    verdict: list[str] = []
    # Systematic means large AND consistent. Both conditions matter: a 0.3 ms bias with
    # 0.03 ms spread is consistent but far too small to be a definitional error, and
    # flagging it would train the reader to ignore this message.
    systematic = bool(resid.size > 2 and abs(resid.mean()) > 2.0
                      and abs(resid.mean()) > 3 * max(resid.std(ddof=1), 0.05))
    if not usable:
        verdict.append("no usable calls: this host cannot pace a frame grid accurately "
                       "enough, or the loopback path is broken")
    else:
        if abs(resid.mean()) > args.gate_ms:
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
             "rows": rows, "verdict": verdict}, indent=2))
        print(f"wrote {args.json}")

    if verdict:
        print("\nSTAGE 2 NOT PASSED on this host:")
        for v in verdict:
            print("  -", v)
        if systematic:
            print(f"\nThe residual is systematic, not noisy (bias {resid.mean():+.2f} ms vs "
                  f"sd {resid.std(ddof=1):.2f} ms). That is a definitional or arithmetic "
                  f"error somewhere in the chain, not host jitter. Check the sign and "
                  f"frame-offset handling in the responder and in t0 derivation. "
                  f"A residual near a whole or double frame time is a strong hint.")
        else:
            print("\nThe residual is noisy rather than systematic, which on a laptop "
                  "usually means power management rather than a code defect. Try: mains "
                  "power, Low Power Mode off, `caffeinate -di` alongside the run.")
        return 1
    print("\nstage 2 passed on this host: live capture path produces trustworthy "
          "timestamps here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
