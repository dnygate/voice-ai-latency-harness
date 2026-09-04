"""The frozen reference signals are an oracle, and these tests are what makes them one.

Bytes are frozen rather than regenerated from a seed, for the reason the G.711 vectors
are: a signal rebuilt by the generator changes the day the generator changes, and every
figure derived from it moves without anyone noticing. So the hashes are checked first, and
nothing else in this file means anything if they fail.

`truth` in the manifest is a property of each signal and is exact by construction: the
generator wrote speech up to an index it chose, so the boundary was never adjudicated.
`annotator_now` is a property of this revision of the code. Keeping them apart is what
stops a regression quietly becoming the new baseline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from harness.analyse import DEFAULT_VARIANTS, _detect_onset
from harness.prompts import (SAMPLE_RATE, annotate_speech_bounds, frame_energy_dbov,
                             load_wav_8k_mono)

REF = Path(__file__).parent / "reference"
MANIFEST = json.loads((REF / "MANIFEST.json").read_text())
FILES = MANIFEST["files"]


def _floor_dbov(pcm: np.ndarray, upto: int) -> float:
    w = max(1, int(0.005 * SAMPLE_RATE))
    return float(np.percentile(frame_energy_dbov(pcm[:upto], w, w)[1], 10.0))


@pytest.mark.parametrize("name", sorted(FILES))
def test_reference_signal_is_unchanged(name):
    """First and most important. Every other test here is a claim about these exact bytes,
    and a signal that has drifted invalidates all of them at once."""
    path = REF / name
    assert path.exists(), f"{name} is missing from the reference set"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FILES[name]["sha256"]
    assert len(load_wav_8k_mono(path)) == FILES[name]["n_samples"]


def test_annotator_reproduces_the_stimulus_truth_exactly():
    """t0 is the origin of every latency figure, and a bias here propagates into all of
    them with the wrong sign: a late t0 makes the system under test look faster than it
    is. On a hard cut the boundary is exact by construction, so the annotator has no
    licence to be even one sample out."""
    name = "stimulus-2s.wav"
    pcm = load_wav_8k_mono(REF / name)
    ann = annotate_speech_bounds(pcm, prompt_id="stimulus-2s")
    assert ann.speech_end_sample == FILES[name]["truth"]["speech_end_sample"]


def test_hard_onset_is_found_at_truth_by_every_variant():
    """The onset variants differ in how much evidence they need, which matters on audio
    that ramps in and should matter not at all on a signal that starts at full level.
    Before finding nine the three disagreed by 0.99 ms here, which was the t1 refinement
    firing on channel noise rather than any genuine ambiguity."""
    name = "response-hard.wav"
    pcm = load_wav_8k_mono(REF / name)
    truth = FILES[name]["truth"]["first_response_sample"]
    floor = _floor_dbov(pcm, truth)
    found = {v.name: _detect_onset(pcm, v, floor, SAMPLE_RATE) for v in DEFAULT_VARIANTS}
    assert all(o is not None for o in found.values()), found
    assert max(found.values()) - min(found.values()) == 0, found
    assert abs(found["headline"] - truth) <= 1, (found, truth)


@pytest.mark.parametrize("name,lo_ms,hi_ms", [
    ("response-ramp60.wav", 1.0, 8.0),
    ("response-ramp150.wav", 4.0, 16.0),
])
def test_ramped_onset_spread_grows_with_the_ramp(name, lo_ms, hi_ms):
    """The spread across variants is the onset-definition uncertainty that METHOD.md 2.2
    requires be published with every headline figure, and on real synthesised speech it is
    the dominant term. It must grow with the ramp and it must never collapse to zero,
    because a ramp genuinely is ambiguous and a detector claiming otherwise is wrong."""
    pcm = load_wav_8k_mono(REF / name)
    truth = FILES[name]["truth"]["first_response_sample"]
    floor = _floor_dbov(pcm, truth)
    found = {v.name: _detect_onset(pcm, v, floor, SAMPLE_RATE) for v in DEFAULT_VARIANTS}
    spread_ms = (max(found.values()) - min(found.values())) / (SAMPLE_RATE / 1000.0)
    assert lo_ms <= spread_ms <= hi_ms, (name, spread_ms, found)
    # A more sensitive variant needs less evidence, so it can never fire later than a
    # stricter one. Inverted ordering means the thresholds are not doing what they say.
    assert found["sensitive"] <= found["headline"] <= found["strict"], found
    # Detection trails the true start on a ramp, never precedes it: audio below the
    # threshold has not begun as far as any stated definition is concerned.
    assert found["sensitive"] >= truth, (found, truth)


def test_greeting_has_a_detectable_end_with_silence_after_it():
    """METHOD.md 2.3. This is the signal stage 4's greeting detection will be built
    against. Its trailing silence must be long enough that a detector waiting for
    sustained quiet has something to find, and the manifest records the boundary a
    detector must not land before."""
    name = "greeting-1p1s.wav"
    pcm = load_wav_8k_mono(REF / name)
    t = FILES[name]["truth"]
    g_end = t["greeting_end_sample"]
    floor = _floor_dbov(pcm, t["greeting_start_sample"])
    w = max(1, int(0.005 * SAMPLE_RATE))
    _, dbov = frame_energy_dbov(pcm, w, w)
    frames_per_sample = len(dbov) / len(pcm)
    speech = dbov[int(t["greeting_start_sample"] * frames_per_sample) + 2:
                  int(g_end * frames_per_sample) - 2]
    after = dbov[int(g_end * frames_per_sample) + 2:]
    assert speech.mean() > floor + 20.0, (speech.mean(), floor)
    assert after.max() < floor + 10.0, (after.max(), floor)
    assert len(pcm) - g_end >= int(0.5 * SAMPLE_RATE), "need >=500 ms of trailing silence"


def test_manifest_separates_truth_from_current_output_and_they_agree_where_exact():
    """The distinction is the point of the file. `truth` is exact by construction and must
    carry a note saying so, because a truth field nobody can trace the provenance of is
    just a number somebody typed.

    Where the same quantity appears under both, they are being compared rather than
    duplicated, and on a signal whose boundary is exact they must agree. That comparison
    is the conformance claim an independent implementation would check itself against, so
    a divergence here is a finding rather than a tolerance to widen."""
    assert MANIFEST["schema"] == "vlh-reference/1"
    for name, e in FILES.items():
        assert "truth" in e, name
        assert "note" in e["truth"], f"{name}: truth needs to say what it means"
        assert "role" in e, f"{name}: needs to say what it is for"
        for key in set(e["truth"]) & set(e.get("annotator_now", {})):
            assert e["truth"][key] == e["annotator_now"][key], \
                f"{name}: truth and annotator_now disagree on {key}"
