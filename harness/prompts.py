"""Prompt audio loading and speech-end annotation.

t0 (the "mouth" end of mouth-to-ear) is defined as the instant the final speech
sample of the caller's utterance leaves the harness. That requires knowing, to
sample precision, where speech ends in the clean prompt file. It is deliberately
NOT derived from live VAD: live VAD introduces its own decision lag, which is one
of the quantities under measurement, so using it to define t0 would hide it.

Annotations are computed once, stored as JSON next to the prompt, and treated as
frozen. Every prompt in a published eval set should be hand-verified and its
annotation committed, so the same t0 is used by anyone reproducing the work.
"""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .g711 import SAMPLE_RATE

ANNOTATION_VERSION = "vlh-prompt/1"


@dataclass
class PromptAnnotation:
    version: str
    prompt_id: str
    sha256: str
    sample_rate: int
    n_samples: int
    speech_start_sample: int
    speech_end_sample: int
    method: str
    params: dict
    verified_by: str = ""  # set to a human name once the boundary has been eyeballed

    @property
    def speech_end_s(self) -> float:
        return self.speech_end_sample / self.sample_rate

    @property
    def trailing_silence_ms(self) -> float:
        return 1000.0 * (self.n_samples - self.speech_end_sample) / self.sample_rate


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_wav_8k_mono(path: str | Path) -> np.ndarray:
    """Load a 16-bit mono 8 kHz WAV. Resamples only if scipy is present."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("prompt must be 16-bit PCM")
        n_ch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    pcm = np.frombuffer(raw, dtype="<i2")
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
    if sr != SAMPLE_RATE:
        try:
            from scipy.signal import resample_poly
        except ImportError as e:  # pragma: no cover
            raise ValueError(f"prompt is {sr} Hz, need {SAMPLE_RATE} Hz (scipy absent)") from e
        from math import gcd

        g = gcd(sr, SAMPLE_RATE)
        pcm = resample_poly(pcm.astype(np.float64), SAMPLE_RATE // g, sr // g)
        pcm = np.clip(np.round(pcm), -32768, 32767).astype(np.int16)
    return np.ascontiguousarray(pcm.astype(np.int16))


def write_wav_8k_mono(path: str | Path, pcm: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
    return path


def frame_energy_dbov(pcm: np.ndarray, win: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """Short-term RMS in dBov (0 dBov == full-scale square wave, i.e. RMS 32768)."""
    x = np.asarray(pcm, dtype=np.float64)
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    starts = np.arange(0, len(x) - win + 1, hop)
    idx = starts[:, None] + np.arange(win)[None, :]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    dbov = 20.0 * np.log10(np.maximum(rms, 1e-9) / 32768.0)
    return starts, dbov


def annotate_speech_bounds(
    pcm: np.ndarray,
    prompt_id: str,
    sha256: str = "",
    win_ms: float = 10.0,
    hop_ms: float = 2.5,
    margin_db: float = 15.0,
    abs_floor_dbov: float = -55.0,
    secondary_margin_db: float = 6.0,
    max_extension_ms: float = 120.0,
) -> PromptAnnotation:
    """Locate speech start/end in a clean prompt.

    Threshold is the louder of (measured noise floor + margin) and an absolute floor,
    so a studio-clean prompt with a -90 dBov floor does not end up with an absurdly
    low threshold that latches onto dither.

    The end boundary uses decay-following hysteresis rather than a fixed hangover.
    A fixed forward extension would push t0 later by a constant amount, and since
    MRL = t1 - t0, that biases every reported latency low by exactly that constant --
    the worst possible shape of error for this measurement. Instead, from the last
    supra-threshold frame the boundary follows the energy decay while it stays above a
    secondary threshold, so genuine low-energy final phonemes (unvoiced fricatives,
    stop releases) are captured while a hard cut yields no extension at all.
    """
    win = max(1, int(round(SAMPLE_RATE * win_ms / 1000.0)))
    hop = max(1, int(round(SAMPLE_RATE * hop_ms / 1000.0)))
    starts, dbov = frame_energy_dbov(pcm, win, hop)

    noise_floor = float(np.percentile(dbov, 10.0))
    thresh = max(noise_floor + margin_db, abs_floor_dbov)
    active = np.flatnonzero(dbov > thresh)
    if active.size == 0:
        raise ValueError(f"no speech found in prompt {prompt_id!r} (floor {noise_floor:.1f} dBov)")

    start = int(starts[active[0]])

    # Decay-following end boundary with hysteresis.
    secondary = max(noise_floor + secondary_margin_db, abs_floor_dbov - 10.0)
    k = int(active[-1])
    max_ext_frames = int(round(max_extension_ms / hop_ms))
    limit = min(len(dbov) - 1, k + max_ext_frames)
    while k < limit and dbov[k + 1] > secondary:
        k += 1
    end = min(len(pcm), int(starts[k]) + win)

    # Sub-frame refinement: last short-RMS window in the final frame still above the
    # secondary threshold. A short RMS rather than instantaneous |x| is required --
    # instantaneous peaks in channel noise cross the threshold several percent of the
    # time and drag the boundary several ms late, which biases MRL low.
    fine_win = max(4, int(round(SAMPLE_RATE * 0.002)))
    seg = np.asarray(pcm[int(starts[k]) : end], dtype=np.float64)
    if seg.size >= fine_win:
        idx = np.arange(seg.size - fine_win + 1)[:, None] + np.arange(fine_win)[None, :]
        r = np.sqrt(np.mean(seg[idx] ** 2, axis=1))
        db = 20.0 * np.log10(np.maximum(r, 1e-9) / 32768.0)
        hit = np.flatnonzero(db > secondary)
        if hit.size:
            # Convention: the boundary is the first sample AFTER the last window that
            # still contains signal. Because even one full-level sample lifts a 2 ms RMS
            # window far above threshold, the last supra-threshold window begins one
            # sample before the true end -- so +1 is exact on a hard cut and this
            # convention has zero bias on the reference signal. On naturally decaying
            # speech it sits up to fine_win early; that is the t0 convention uncertainty
            # (~2 ms) and it is reported, not hidden.
            end = int(starts[k]) + int(hit[-1]) + 1

    return PromptAnnotation(
        version=ANNOTATION_VERSION,
        prompt_id=prompt_id,
        sha256=sha256,
        sample_rate=SAMPLE_RATE,
        n_samples=int(len(pcm)),
        speech_start_sample=start,
        speech_end_sample=int(end),
        method="energy+decay-hysteresis",
        params={
            "win_ms": win_ms,
            "hop_ms": hop_ms,
            "margin_db": margin_db,
            "abs_floor_dbov": abs_floor_dbov,
            "secondary_margin_db": secondary_margin_db,
            "max_extension_ms": max_extension_ms,
            "secondary_threshold_dbov": round(max(noise_floor + secondary_margin_db,
                                                  abs_floor_dbov - 10.0), 2),
            "measured_noise_floor_dbov": round(noise_floor, 2),
            "threshold_dbov": round(thresh, 2),
        },
    )


def annotate_file(path: str | Path, **kw) -> PromptAnnotation:
    path = Path(path)
    pcm = load_wav_8k_mono(path)
    return annotate_speech_bounds(pcm, prompt_id=path.stem, sha256=sha256_file(path), **kw)


def annotation_path(wav_path: str | Path) -> Path:
    return Path(wav_path).with_suffix(".annotation.json")


def save_annotation(wav_path: str | Path, ann: PromptAnnotation) -> Path:
    p = annotation_path(wav_path)
    p.write_text(json.dumps(asdict(ann), indent=2) + "\n", encoding="utf-8")
    return p


def load_annotation(wav_path: str | Path) -> PromptAnnotation:
    ann = PromptAnnotation(**json.loads(annotation_path(wav_path).read_text(encoding="utf-8")))
    actual = sha256_file(wav_path)
    if ann.sha256 and ann.sha256 != actual:
        raise ValueError(
            f"prompt {wav_path} has changed since annotation "
            f"({ann.sha256[:12]} != {actual[:12]}); re-annotate and re-verify"
        )
    return ann
