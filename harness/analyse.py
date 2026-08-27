"""Offline derivation of mouth-to-ear response latency (MRL) from a capture.

Definitions
-----------
t0  The instant the final speech sample of the caller's utterance is transmitted.
    Located by mapping the prompt's annotated speech_end_sample onto the tx packet
    that carried it, via RTP timestamps (robust to reordering), then interpolating
    within the frame at the sample rate.

t1  The instant the first sample of the system's response arrives. Located by
    reassembling the rx stream in RTP-timestamp order, detecting response onset,
    then mapping the onset sample back to the arrival time of the packet that
    carried it, with within-frame interpolation.

MRL = t1 - t0, both from CLOCK_MONOTONIC on the measuring host.

MRL is what the caller waits, and it is the sum of jitter buffer depth, endpointing
decision lag, ASR finalisation, LLM time-to-first-token, TTS time-to-first-audio,
encode and relay. Vendor-published "first audio latency" typically covers only the
last two or three of those terms.

Onset is definitionally fuzzy, so it is computed under several named variants and
the whole set is reported. The headline figure is one variant; the spread across
variants is the measurement uncertainty and belongs in any published result.

MRL is signed. A negative value means the system emitted audio before the caller
stopped speaking, which is a real and interesting behaviour (aggressive endpointing,
or backchannel), not an error.

Two MRLs, not one
-----------------
mrl_ingress  t1 taken at packet arrival at the measuring host. Isolates what the
             system under test contributed, independent of the caller's playout
             policy. Its variance under network jitter equals the channel jitter --
             that is correct behaviour, not instrument error.

mrl_playout  t1 taken at the instant the onset sample would be played out of a
             de-jitter buffer of a stated target depth. This is what a caller
             actually waits, and it is the quantity that should be compared against
             conversational turn-taking norms. De-jitter buffer depth is a real term
             in the latency budget; published vendor "first audio" figures omit it
             entirely, along with everything else upstream of TTS.

Report both. Quoting only ingress understates what callers experience; quoting only
playout confounds the system with the client's buffering policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np

from . import g711
from .prompts import frame_energy_dbov
from .record import Capture, Packet


@dataclass(frozen=True)
class OnsetVariant:
    name: str
    margin_db: float      # required level above the measured rx noise floor
    abs_dbov: float       # absolute level floor, whichever is higher wins
    sustain_ms: float     # level must persist this long, to reject clicks and breaths


DEFAULT_VARIANTS: tuple[OnsetVariant, ...] = (
    OnsetVariant("sensitive", 6.0, -55.0, 10.0),
    OnsetVariant("headline", 10.0, -50.0, 20.0),
    OnsetVariant("strict", 12.0, -45.0, 30.0),
)

HEADLINE_VARIANT = "headline"


@dataclass
class QC:
    tx_packets: int = 0
    rx_late_discard_frames: int = 0
    playout_target_ms: float = float("nan")
    rx_packets: int = 0
    tx_pacing_p50_ms: float = float("nan")
    tx_pacing_p99_ms: float = float("nan")
    tx_pacing_max_dev_ms: float = float("nan")
    rx_lost_frames: int = 0
    rx_loss_pct: float = 0.0
    rx_max_gap_ms: float = float("nan")
    rx_jitter_ms: float = float("nan")  # RFC 3550 interarrival jitter, converted to ms
    rx_noise_floor_dbov: float = float("nan")
    noise_window_ms: float = float("nan")
    flags: list[str] = field(default_factory=list)

    def ok(self, max_tx_pacing_dev_ms: float = 5.0) -> bool:
        """QC verdict. Channel-degradation flags are advisory, not blocking: a lossy
        call is still a valid measurement of a lossy call. Only flags that invalidate
        the measurement itself block, plus failure of the harness to pace its own
        transmission, which corrupts t0 and therefore everything downstream."""
        blocking = {"no_tx_speech_end", "no_rx_packets", "onset_not_found", "onset_in_first_frame"}
        if blocking & set(self.flags):
            return False
        if np.isfinite(self.tx_pacing_max_dev_ms) and self.tx_pacing_max_dev_ms > max_tx_pacing_dev_ms:
            return False
        return True


@dataclass
class VariantResult:
    name: str
    onset_sample: int
    t1_ns: int          # ingress: arrival of the onset sample
    mrl_ms: float       # ingress MRL, kept as `mrl_ms` for brevity in aggregation
    t1_playout_ns: int = -1
    mrl_playout_ms: float = float("nan")


@dataclass
class CallResult:
    call_id: str
    run_id: str
    concurrency: int
    codec: str
    prompt_id: str
    t0_ns: int
    variants: dict[str, VariantResult]
    qc: QC
    ground_truth_ms: float | None = None

    @property
    def mrl_ms(self) -> float:
        return self.variants[HEADLINE_VARIANT].mrl_ms

    def to_json(self) -> dict:
        d = asdict(self)
        d["mrl_ms"] = self.mrl_ms
        return d


def _stream(packets: list[Packet], codec: str, samples_per_frame: int):
    """Reassemble a packet list into a contiguous sample array in RTP-timestamp order.

    Returns (samples, owner, base_rtp_ts, lost_frames) where owner[i] is the index into
    `packets` of the packet that carried sample i, or -1 for samples synthesised to fill
    a loss gap. Losses are filled with silence and counted; they are never interpolated,
    because a concealed gap would let the onset detector fire on invented energy.
    """
    if not packets:
        return np.zeros(0, dtype=np.int16), np.zeros(0, dtype=np.int64), 0, 0
    order = sorted(range(len(packets)), key=lambda i: packets[i].rtp_ts)
    base = packets[order[0]].rtp_ts
    spans = []
    max_end = 0
    for i in order:
        off = packets[i].rtp_ts - base
        pcm = g711.decode(packets[i].payload, codec)
        spans.append((off, pcm, i))
        max_end = max(max_end, off + len(pcm))
    samples = np.zeros(max_end, dtype=np.int16)
    owner = np.full(max_end, -1, dtype=np.int64)
    for off, pcm, i in spans:
        samples[off : off + len(pcm)] = pcm
        owner[off : off + len(pcm)] = i
    lost = int(np.count_nonzero(owner < 0) // max(1, samples_per_frame))
    return samples, owner, base, lost


def _sample_time_ns(sample_idx: int, owner: np.ndarray, packets: list[Packet],
                    base_rtp: int, sr: int) -> int | None:
    """Map a sample index to a wall-of-monotonic instant via its carrying packet."""
    if sample_idx < 0 or sample_idx >= len(owner):
        return None
    pkt_i = int(owner[sample_idx])
    if pkt_i < 0:
        return None
    p = packets[pkt_i]
    within = sample_idx - (p.rtp_ts - base_rtp)
    return int(p.t_mono_ns + round(within * 1e9 / sr))


def _pacing_stats(tx: list[Packet], ptime_ms: float) -> tuple[float, float, float]:
    if len(tx) < 3:
        return float("nan"), float("nan"), float("nan")
    t = np.array([p.t_mono_ns for p in sorted(tx, key=lambda p: p.rtp_ts)], dtype=np.float64)
    d = np.diff(t) / 1e6
    dev = np.abs(d - ptime_ms)
    return float(np.percentile(d, 50)), float(np.percentile(dev, 99)), float(dev.max())


def _rfc3550_jitter(rx: list[Packet], sr: int) -> float:
    if len(rx) < 2:
        return float("nan")
    pk = sorted(rx, key=lambda p: p.seq)
    j = 0.0
    prev = None
    for p in pk:
        arrival = p.t_mono_ns * sr / 1e9  # arrival in RTP timestamp units
        d_transit = arrival - p.rtp_ts
        if prev is not None:
            j += (abs(d_transit - prev) - j) / 16.0
        prev = d_transit
    return float(j * 1000.0 / sr)


def _detect_onset(samples: np.ndarray, variant: OnsetVariant, noise_floor_dbov: float,
                  sr: int, search_from: int = 0) -> int | None:
    win = max(1, int(round(sr * 0.005)))   # 5 ms analysis window
    hop = win
    starts, dbov = frame_energy_dbov(samples, win, hop)
    thresh = max(noise_floor_dbov + variant.margin_db, variant.abs_dbov)
    need = max(1, int(round(variant.sustain_ms / (1000.0 * hop / sr))))

    above = dbov > thresh
    valid = starts >= search_from
    run = 0
    for k in range(len(above)):
        if not valid[k]:
            run = 0
            continue
        if above[k]:
            run += 1
            if run >= need:
                first_frame = int(starts[k - need + 1])
                seg = np.abs(samples[first_frame : first_frame + win].astype(np.int32))
                # refine to the first sample in that frame above the noise floor amplitude
                amp_thresh = max(1.0, 32768.0 * 10 ** (thresh / 20.0) * 0.5)
                hit = np.flatnonzero(seg >= amp_thresh)
                return first_frame + (int(hit[0]) if hit.size else 0)
        else:
            run = 0
    return None


def _playout_anchor(rx: list[Packet], base_rtp: int, sr: int, target_ms: float,
                    window: int = 25) -> tuple[float, int]:
    """Fixed de-jitter buffer anchor, tracked on minimum observed transit delay.

    Real adaptive buffers converge toward the minimum transit delay seen so far, so the
    anchor is min(arrival - rtp_offset) over an initial window rather than simply the
    first packet's arrival. That makes the playout clock robust to a single early or
    late packet at the start of the stream. Returns (anchor_ns, late_discarded_frames).
    """
    if not rx:
        return float("nan"), 0
    by_arrival = sorted(rx, key=lambda p: p.t_mono_ns)
    transit = [p.t_mono_ns - (p.rtp_ts - base_rtp) * 1e9 / sr for p in by_arrival[:window]]
    anchor = min(transit) + target_ms * 1e6
    late = sum(1 for p in rx if p.t_mono_ns > anchor + (p.rtp_ts - base_rtp) * 1e9 / sr)
    return anchor, late


def analyse_capture(cap: Capture, variants: Iterable[OnsetVariant] = DEFAULT_VARIANTS,
                    noise_guard_ms: float = 100.0,
                    playout_target_ms: float = 40.0) -> CallResult:
    h = cap.header
    sr, spf = h.sample_rate, h.samples_per_frame
    tx, rx = cap.tx(), cap.rx()
    qc = QC(tx_packets=len(tx), rx_packets=len(rx))
    qc.tx_pacing_p50_ms, qc.tx_pacing_p99_ms, qc.tx_pacing_max_dev_ms = _pacing_stats(tx, h.ptime_ms)

    # ---- t0 -------------------------------------------------------------------
    tx_s, tx_owner, tx_base, _ = _stream(tx, h.codec, spf)
    t0 = None
    if h.speech_end_sample >= 0 and len(tx_owner):
        # speech_end_sample is exclusive; the final speech sample is the one before it
        t0 = _sample_time_ns(min(h.speech_end_sample - 1, len(tx_owner) - 1), tx_owner, tx, tx_base, sr)
    if t0 is None:
        qc.flags.append("no_tx_speech_end")
        t0 = 0

    # ---- rx stream and noise floor -------------------------------------------
    rx_s, rx_owner, rx_base, lost = _stream(rx, h.codec, spf)
    qc.rx_lost_frames = lost
    total_frames = max(1, len(rx_s) // max(1, spf))
    qc.rx_loss_pct = 100.0 * lost / total_frames
    qc.rx_jitter_ms = _rfc3550_jitter(rx, sr)
    if len(rx) >= 2:
        arr = np.array(sorted(p.t_mono_ns for p in rx), dtype=np.float64)
        qc.rx_max_gap_ms = float(np.diff(arr).max() / 1e6)
    if not rx:
        qc.flags.append("no_rx_packets")
    anchor_ns, qc.rx_late_discard_frames = _playout_anchor(rx, rx_base, sr, playout_target_ms)
    qc.playout_target_ms = playout_target_ms

    # Noise floor is estimated from rx audio that arrived before t0 - guard, i.e. while
    # the caller was still speaking and the system had not yet answered. Capped at 1 s.
    noise_end = 0
    if len(rx_owner):
        cutoff = t0 - int(noise_guard_ms * 1e6)
        for i, p in enumerate(sorted(rx, key=lambda p: p.rtp_ts)):
            if p.t_mono_ns >= cutoff:
                break
            noise_end = max(noise_end, (p.rtp_ts - rx_base) + spf)
        noise_end = min(noise_end, sr)  # 1 s cap
    if noise_end < int(0.05 * sr):
        noise_end = min(len(rx_s), int(0.1 * sr))
        if len(rx_s):
            qc.flags.append("short_noise_window")
    qc.noise_window_ms = 1000.0 * noise_end / sr
    if noise_end > 0:
        _, nf = frame_energy_dbov(rx_s[:noise_end], max(1, int(0.005 * sr)),
                                  max(1, int(0.005 * sr)))
        qc.rx_noise_floor_dbov = float(np.percentile(nf, 10.0))
    else:
        qc.rx_noise_floor_dbov = -90.0

    # ---- t1 per variant -------------------------------------------------------
    results: dict[str, VariantResult] = {}
    for v in variants:
        onset = _detect_onset(rx_s, v, qc.rx_noise_floor_dbov, sr) if len(rx_s) else None
        if onset is None:
            results[v.name] = VariantResult(v.name, -1, -1, float("nan"))
            continue
        t1 = _sample_time_ns(onset, rx_owner, rx, rx_base, sr)
        if t1 is None:
            results[v.name] = VariantResult(v.name, onset, -1, float("nan"))
            continue
        t1_play = anchor_ns + onset * 1e9 / sr
        results[v.name] = VariantResult(
            v.name, int(onset), int(t1), (t1 - t0) / 1e6,
            t1_playout_ns=int(t1_play), mrl_playout_ms=(t1_play - t0) / 1e6,
        )

    head = results.get(HEADLINE_VARIANT)
    if head is None or not np.isfinite(head.mrl_ms):
        qc.flags.append("onset_not_found")
    elif head.onset_sample < spf:
        qc.flags.append("onset_in_first_frame")
    if head is not None and np.isfinite(head.mrl_ms) and head.mrl_ms < 0:
        qc.flags.append("response_before_speech_end")
    # Advisory channel-quality flags. When either fires, onset localisation is deferred
    # by whole frames and the resulting MRL is an upper bound, not a point estimate.
    if (np.isfinite(qc.tx_pacing_max_dev_ms) and qc.tx_pacing_max_dev_ms > 5.0):
        qc.flags.append("tx_pacing_out_of_spec")
    if qc.rx_loss_pct > 2.0:
        qc.flags.append("high_loss")
    if len(rx) and qc.rx_late_discard_frames > 0.01 * len(rx):
        qc.flags.append("high_late_discard")

    return CallResult(
        call_id=h.call_id, run_id=h.run_id, concurrency=h.concurrency, codec=h.codec,
        prompt_id=h.prompt_id, t0_ns=int(t0), variants=results, qc=qc,
        ground_truth_ms=h.ground_truth_ms,
    )


def summarise(results: list[CallResult], variant: str = HEADLINE_VARIANT) -> dict:
    """Aggregate a run. Only QC-passing calls contribute to the latency distribution.

    n_dropped is reported alongside every distribution and belongs in any published
    table: a run that silently discards a third of its calls is not comparable to one
    that discards none, however good the surviving percentiles look.
    """
    kept = [r for r in results if r.qc.ok() and np.isfinite(r.variants[variant].mrl_ms)]
    out = {
        "variant": variant,
        "n_total": len(results),
        "n_kept": len(kept),
        "n_dropped": len(results) - len(kept),
    }
    for kind, attr in (("ingress", "mrl_ms"), ("playout", "mrl_playout_ms")):
        vals = np.array([getattr(r.variants[variant], attr) for r in kept], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            continue
        out[kind] = {
            "mean_ms": round(float(vals.mean()), 2),
            "p50_ms": round(float(np.percentile(vals, 50)), 2),
            "p90_ms": round(float(np.percentile(vals, 90)), 2),
            "p95_ms": round(float(np.percentile(vals, 95)), 2),
            "p99_ms": round(float(np.percentile(vals, 99)), 2),
            "max_ms": round(float(vals.max()), 2),
        }
    return out
