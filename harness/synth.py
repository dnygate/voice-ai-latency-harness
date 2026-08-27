"""Synthetic signals and a reference responder with programmable, known latency.

Purpose: establish the harness's own accuracy before any real system is measured.
The reference responder replies exactly `delay_ms` after the final speech sample of
the prompt is transmitted. Ground truth is therefore known to the nanosecond, and
the analyser's recovered MRL can be compared against it. Any bias or spread seen
here is instrument error and must be subtracted from, or reported alongside, every
measurement taken of a real system.

Three stages of validation use this module:
  1. in-process (this file)      proves the metric arithmetic and onset detection
  2. UDP loopback                proves the live capture path and its timestamping
  3. remote host with netem      proves end-to-end against a known injected delay

Only stage 1 runs without a network. Stages 2 and 3 reuse the same analyser.
"""

from __future__ import annotations

import numpy as np

from . import g711
from .prompts import PromptAnnotation, annotate_speech_bounds
from .record import Capture, CaptureHeader, Packet

SR = g711.SAMPLE_RATE


# --------------------------------------------------------------------------- signals

def speechlike(duration_s: float, seed: int = 0, f0: float = 120.0,
               level_dbov: float = -20.0, onset: str = "hard",
               onset_ms: float = 0.0) -> np.ndarray:
    """A voiced-speech-like signal: harmonic stack under formant-ish shaping, with a
    syllabic amplitude envelope. Not intelligible, and not meant to be: it exercises
    the energy-based onset detector with realistic spectral tilt and dynamics.

    onset="hard" starts at full envelope on sample 0, so a reference responder built
    from it has an unambiguous true onset. onset="fade" applies a raised-cosine ramp
    of onset_ms, which is what real TTS tends to do, and which is the main source of
    onset-detection bias.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * SR))
    t = np.arange(n) / SR

    jitter = 1.0 + 0.02 * np.sin(2 * np.pi * 4.7 * t + rng.uniform(0, 6.28))
    phase = 2 * np.pi * f0 * np.cumsum(jitter) / SR
    sig = np.zeros(n)
    for k in range(1, 26):
        fk = f0 * k
        if fk > 3400.0:  # telephony band limit
            break
        # crude two-formant emphasis around 700 Hz and 1600 Hz plus -6 dB/oct tilt
        shape = (1.0 / (1.0 + ((fk - 700.0) / 350.0) ** 2)
                 + 0.6 / (1.0 + ((fk - 1600.0) / 500.0) ** 2))
        sig += shape / k * np.sin(k * phase + rng.uniform(0, 6.28))

    sig += 0.03 * rng.standard_normal(n)  # aspiration
    syll = 0.55 + 0.45 * np.sin(2 * np.pi * 3.3 * t - np.pi / 2) ** 2
    sig *= syll

    rms = np.sqrt(np.mean(sig**2)) or 1.0
    sig *= (32768.0 * 10 ** (level_dbov / 20.0)) / rms

    if onset == "fade" and onset_ms > 0:
        m = min(n, int(round(onset_ms * SR / 1000.0)))
        sig[:m] *= 0.5 * (1 - np.cos(np.pi * np.arange(m) / max(1, m)))
    elif onset != "hard":
        raise ValueError("onset must be 'hard' or 'fade'")

    return np.clip(np.round(sig), -32768, 32767).astype(np.int16)


def noise_at(duration_s: float, level_dbov: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * SR))
    x = rng.standard_normal(n)
    x *= (32768.0 * 10 ** (level_dbov / 20.0)) / (np.sqrt(np.mean(x**2)) or 1.0)
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def make_prompt(speech_s: float = 2.0, lead_silence_ms: float = 300.0,
                trail_silence_ms: float = 700.0, floor_dbov: float = -62.0,
                seed: int = 0) -> tuple[np.ndarray, int]:
    """Build a prompt and return (pcm, true_speech_end_sample).

    Trailing silence is essential and realistic: a caller stops speaking and the
    channel keeps carrying frames. Without it there is no interval in which to
    observe endpointing lag.
    """
    lead = noise_at(lead_silence_ms / 1000.0, floor_dbov, seed=seed + 1)
    body = speechlike(speech_s, seed=seed, onset="hard")
    trail = noise_at(trail_silence_ms / 1000.0, floor_dbov, seed=seed + 2)
    pcm = np.concatenate([lead, body, trail])
    return pcm, len(lead) + len(body)


# ------------------------------------------------------------------ reference capture

def jitter_excess(rng: np.random.Generator, n: int, mean_ms: float,
                  shape: float = 2.0) -> np.ndarray:
    """One-sided excess delay in ns, gamma-distributed with the given mean.

    Network jitter is not symmetric. A packet can be queued and delivered late; it
    cannot arrive before its minimum transit time. Modelling jitter as zero-mean
    Gaussian is a common shortcut and it is wrong in a way that matters here: it lets
    a minimum-transit-tracking de-jitter buffer anchor earlier than physically
    possible, which shows up as an apparent negative latency bias. Gamma with shape 2
    gives a realistic right-skewed distribution with a hard floor at zero.
    """
    if mean_ms <= 0:
        return np.zeros(n)
    return rng.gamma(shape=shape, scale=mean_ms / shape, size=n) * 1e6


def _frames(pcm: np.ndarray, spf: int) -> list[np.ndarray]:
    pad = (-len(pcm)) % spf
    if pad:
        pcm = np.concatenate([pcm, np.zeros(pad, dtype=np.int16)])
    return [pcm[i : i + spf] for i in range(0, len(pcm), spf)]


def reference_capture(
    delay_ms: float,
    prompt: np.ndarray | None = None,
    speech_end_sample: int | None = None,
    response: np.ndarray | None = None,
    codec: str = "pcmu",
    ptime_ms: float = 20.0,
    tx_jitter_ms: float = 0.0,
    rx_jitter_ms: float = 0.0,
    rx_base_transit_ms: float = 0.0,
    rx_loss_rate: float = 0.0,
    rx_floor_dbov: float = -62.0,
    seed: int = 0,
    call_id: str = "ref",
    run_id: str = "validation",
    concurrency: int = 1,
    t_start_ns: int = 1_000_000_000_000,
) -> Capture:
    """Construct a capture whose true MRL is exactly `delay_ms`.

    The rx stream carries channel noise from call start, then switches to the response
    at precisely t0 + delay_ms, where t0 is the transmission instant of the prompt's
    final speech sample. The response is aligned so that its first sample lands on that
    instant, which requires the responder to start its frame mid-grid: rx frame
    boundaries are deliberately not aligned to tx frame boundaries, exactly as they
    would not be in a real system.
    """
    rng = np.random.default_rng(seed)
    spf = int(round(SR * ptime_ms / 1000.0))
    frame_ns = int(round(ptime_ms * 1e6))

    if prompt is None:
        prompt, true_end = make_prompt(seed=seed)
        speech_end_sample = true_end
    if speech_end_sample is None:
        raise ValueError("speech_end_sample required when prompt is supplied")
    if response is None:
        response = speechlike(1.5, seed=seed + 100, onset="hard")

    # --- tx: paced frames, optional pacing jitter
    tx_frames = _frames(prompt, spf)
    # Sender pacing slop is symmetric: a timer-driven sender can fire marginally early
    # or late relative to the nominal grid. This is a different physical mechanism from
    # network jitter, which is queueing and therefore strictly one-sided.
    tx_excess = (rng.normal(0.0, tx_jitter_ms, len(tx_frames)) * 1e6
                 if tx_jitter_ms else np.zeros(len(tx_frames)))
    tx_pkts: list[Packet] = []
    ssrc_tx = 0x1111_1111
    for i, fr in enumerate(tx_frames):
        j = int(round(tx_excess[i]))
        tx_pkts.append(Packet(
            dir="tx", t_mono_ns=t_start_ns + i * frame_ns + j,
            t_wall_ns=t_start_ns + i * frame_ns + j, seq=i, rtp_ts=i * spf,
            ssrc=ssrc_tx, pt=g711.payload_type(codec), marker=(i == 0),
            payload=g711.encode(fr, codec),
        ))

    # t0: transmission instant of the final speech sample, matching analyse.py exactly
    last_speech = speech_end_sample - 1
    fi, within = divmod(last_speech, spf)
    fi = min(fi, len(tx_pkts) - 1)
    t0_ns = tx_pkts[fi].t_mono_ns + round(within * 1e9 / SR)
    t_resp_ns = t0_ns + int(round(delay_ms * 1e6))

    # --- rx: noise from call start, response beginning exactly at t_resp_ns.
    # rx frame grid is offset from tx by a deliberate half-frame plus a random slew.
    rx_grid_start = t_start_ns + frame_ns // 2 + int(rng.integers(0, frame_ns // 4))
    n_rx_frames = int(np.ceil((t_resp_ns - rx_grid_start + len(response) * 1e9 / SR) / frame_ns)) + 2
    n_rx_frames = max(n_rx_frames, 1)

    total_rx_samples = n_rx_frames * spf
    rx_pcm = noise_at(total_rx_samples / SR, rx_floor_dbov, seed=seed + 3)[:total_rx_samples]

    # Sample index in the rx grid at which the response must begin.
    resp_start = int(round((t_resp_ns - rx_grid_start) * SR / 1e9))
    if resp_start < 0:
        raise ValueError("delay_ms places the response before the rx stream starts")
    end = min(total_rx_samples, resp_start + len(response))
    if end > resp_start:
        rx_pcm[resp_start:end] = response[: end - resp_start]

    rx_pkts: list[Packet] = []
    ssrc_rx = 0x2222_2222
    rx_excess = jitter_excess(rng, n_rx_frames, rx_jitter_ms)
    base_transit_ns = int(round(rx_base_transit_ms * 1e6))
    for i in range(n_rx_frames):
        if rx_loss_rate and rng.random() < rx_loss_rate:
            continue
        j = base_transit_ns + int(round(rx_excess[i]))
        fr = rx_pcm[i * spf : (i + 1) * spf]
        rx_pkts.append(Packet(
            dir="rx", t_mono_ns=rx_grid_start + i * frame_ns + j,
            t_wall_ns=rx_grid_start + i * frame_ns + j, seq=i, rtp_ts=i * spf,
            ssrc=ssrc_rx, pt=g711.payload_type(codec), marker=(i == 0),
            payload=g711.encode(fr, codec),
        ))

    hdr = CaptureHeader(
        call_id=call_id, prompt_id="synthetic", speech_end_sample=int(speech_end_sample),
        codec=codec, ptime_ms=ptime_ms, sut_label="reference-responder",
        concurrency=concurrency, run_id=run_id, ground_truth_ms=float(delay_ms),
        notes={"tx_jitter_ms": tx_jitter_ms, "rx_jitter_ms": rx_jitter_ms,
               "rx_base_transit_ms": rx_base_transit_ms,
               "rx_loss_rate": rx_loss_rate, "rx_floor_dbov": rx_floor_dbov, "seed": seed},
    )
    return Capture(header=hdr, packets=tx_pkts + rx_pkts)


def annotated_prompt(**kw) -> tuple[np.ndarray, int, PromptAnnotation]:
    """Build a prompt, then run the production annotator on it. Returns
    (pcm, true_speech_end, detected_annotation) so annotator error can be measured."""
    pcm, true_end = make_prompt(**kw)
    ann = annotate_speech_bounds(pcm, prompt_id="synthetic")
    return pcm, true_end, ann
