"""Generate the frozen reference signal set. Run once; the artifacts are then the oracle.

Freezing bytes rather than a seed is the lesson of the G.711 golden vectors: a signal
regenerated from a seed changes the day the generator does, and every figure derived from
it moves silently. Here the true boundaries are known by construction rather than by
judgement, which is stronger than hand-verification: the generator places speech and then
silence, so the last speech sample is an index it chose, not a boundary anyone had to
adjudicate.

The manifest keeps truth and current detector output in separate fields, deliberately.
Truth is a property of the signal. Detector output is a property of this revision of the
code, and conflating them is how a regression becomes the new baseline.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from harness import prompts as P
from harness.analyse import DEFAULT_VARIANTS, _onset_threshold, _detect_onset
from harness.synth import make_prompt, noise_at, speechlike

OUT = Path("tests/reference")
SR = P.SAMPLE_RATE
FLOOR = -62.0

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

entries: dict[str, dict] = {}

# ---- stimulus: a prompt whose speech end is exact by construction
pcm, true_end = make_prompt(speech_s=2.0, lead_silence_ms=300.0, trail_silence_ms=700.0,
                            floor_dbov=FLOOR, seed=20260904)
p = P.write_wav_8k_mono(OUT / "stimulus-2s.wav", pcm)
ann = P.annotate_speech_bounds(pcm, prompt_id="stimulus-2s", sha256=sha(p))
entries["stimulus-2s.wav"] = {
    "role": "caller stimulus; the t0 conformance signal",
    "sha256": sha(p), "n_samples": int(len(pcm)),
    "truth": {"speech_end_sample": int(true_end),
              "note": "index the generator stopped writing speech at; exact, not judged"},
    "annotator_now": {"speech_end_sample": int(ann.speech_end_sample),
                      "speech_start_sample": int(ann.speech_start_sample),
                      "method": ann.method, "version": ann.version},
}

# ---- greeting: a system that speaks first, then falls silent
greet = np.concatenate([
    noise_at(0.20, FLOOR, seed=11),
    speechlike(1.10, seed=22, onset="hard"),
    noise_at(1.20, FLOOR, seed=33),
]).astype(np.int16)
g_end = int(0.20 * SR) + int(1.10 * SR)
p = P.write_wav_8k_mono(OUT / "greeting-1p1s.wav", greet)
entries["greeting-1p1s.wav"] = {
    "role": "unprompted greeting; the METHOD.md 2.3 conformance signal",
    "sha256": sha(p), "n_samples": int(len(greet)),
    "truth": {"greeting_start_sample": int(0.20 * SR), "greeting_end_sample": g_end,
              "trailing_silence_ms": 1200.0,
              "note": "detection must land at or after greeting_end_sample; early is "
                      "the -2985 ms failure of finding seven, late costs nothing"},
}

# ---- responses: one hard onset and two ramps, for the onset-variant spread
for label, onset_ms in (("hard", 0.0), ("ramp60", 60.0), ("ramp150", 150.0)):
    body = speechlike(1.0, seed=44, onset="hard" if onset_ms == 0 else "fade",
                      onset_ms=onset_ms)
    lead = int(0.30 * SR)
    pcm = np.concatenate([noise_at(0.30, FLOOR, seed=55), body,
                          noise_at(0.30, FLOOR, seed=66)]).astype(np.int16)
    name = f"response-{label}.wav"
    p = P.write_wav_8k_mono(OUT / name, pcm)
    floor_est = float(np.percentile(
        P.frame_energy_dbov(pcm[:lead], int(0.005 * SR), int(0.005 * SR))[1], 10.0))
    detected = {}
    for v in DEFAULT_VARIANTS:
        o = _detect_onset(pcm, v, floor_est, SR)
        detected[v.name] = None if o is None else int(o)
    entries[name] = {
        "role": f"system response, {'hard onset' if onset_ms == 0 else f'{onset_ms:g} ms ramp'}",
        "sha256": sha(p), "n_samples": int(len(pcm)),
        "truth": {"first_response_sample": lead, "onset_ramp_ms": onset_ms,
                  "note": "sample at which the response begins. On a ramp the detected "
                          "onset legitimately trails it, and the spread across variants "
                          "is the onset-definition uncertainty METHOD.md 2.2 publishes"},
        "annotator_now": {"onset_by_variant": detected,
                          "noise_floor_dbov": round(floor_est, 2)},
    }

manifest = {
    "schema": "vlh-reference/1",
    "generated": "2026-09-04",
    "sample_rate": SR,
    "purpose": (
        "Frozen reference signals. Bytes are the oracle, not the generator: a signal "
        "regenerated from a seed changes when the generator changes, which is how the "
        "G.711 vectors came to be frozen. `truth` is a property of each signal and never "
        "changes. `annotator_now` records what this revision of the code produces, so a "
        "regression is visible as a difference rather than becoming the new baseline."),
    "conformance": (
        "An independent implementation should reproduce `truth.speech_end_sample` on the "
        "stimulus to within a stated tolerance, and should publish its onset figures for "
        "the three responses so its detector can be compared with ours. This is what the "
        "IETF draft needs in order to specify a conformance test rather than an "
        "algorithm."),
    "files": entries,
}
(OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
