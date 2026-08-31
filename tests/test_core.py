"""Core correctness tests. Run: python -m pytest tests -q"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from harness import g711
from harness.analyse import analyse_capture
from harness.record import Capture, read_capture, write_capture
from harness.synth import annotated_prompt, reference_capture, speechlike

# The codec oracle is a frozen golden-vector file, generated once and verified against
# stdlib audioop on CPython 3.12. audioop was removed in Python 3.13, so depending on it
# would silently skip these tests on any current machine -- exactly the tests that must
# never silently skip, since a wrong codec table corrupts every measurement downstream.
GOLDEN = json.loads((Path(__file__).parent / "g711_golden.json").read_text())


def test_ulaw_encode_matches_golden():
    pcm = np.arange(-32768, 32768, dtype=np.int16)
    got = hashlib.sha256(g711.encode(pcm, "pcmu")).hexdigest()
    assert got == GOLDEN["enc_ulaw_sha256"]


def test_alaw_encode_matches_golden():
    pcm = np.arange(-32768, 32768, dtype=np.int16)
    got = hashlib.sha256(g711.encode(pcm, "pcma")).hexdigest()
    assert got == GOLDEN["enc_alaw_sha256"]


def test_ulaw_decode_matches_golden():
    assert g711.decode(bytes(range(256)), "pcmu").tolist() == GOLDEN["dec_ulaw_table"]


def test_alaw_decode_matches_golden():
    assert g711.decode(bytes(range(256)), "pcma").tolist() == GOLDEN["dec_alaw_table"]


@pytest.mark.skipif(sys.version_info >= (3, 13), reason="audioop removed in 3.13")
def test_golden_still_agrees_with_audioop_where_available():
    """Belt and braces: where the stdlib oracle still exists, confirm the golden file
    was not frozen from a broken implementation."""
    import audioop

    pcm = np.arange(-32768, 32768, dtype=np.int16)
    assert (hashlib.sha256(audioop.lin2ulaw(pcm.tobytes(), 2)).hexdigest()
            == GOLDEN["enc_ulaw_sha256"])
    assert (hashlib.sha256(audioop.lin2alaw(pcm.tobytes(), 2)).hexdigest()
            == GOLDEN["enc_alaw_sha256"])


def test_l16_is_lossless():
    pcm = np.arange(-32768, 32768, dtype=np.int16)
    assert np.array_equal(g711.roundtrip(pcm, "l16"), pcm)


def test_ulaw_roundtrip_snr_is_sane():
    x = speechlike(2.0, seed=1).astype(np.float64)
    y = g711.roundtrip(x.astype(np.int16), "pcmu").astype(np.float64)
    snr = 10 * np.log10(np.sum(x**2) / np.sum((x - y) ** 2))
    assert 30.0 < snr < 45.0, snr  # G.711 gives roughly 38 dB on speech-level input


def test_annotator_is_exact_on_hard_cut_reference():
    """t0 must be exact on the reference signal. Any bias here propagates directly into
    every latency figure the harness will ever produce, with the wrong sign: a late t0
    makes the system under test look faster than it is."""
    for seed in range(16):
        _, true_end, ann = annotated_prompt(seed=seed)
        assert ann.speech_end_sample == true_end, (seed, ann.speech_end_sample, true_end)


@pytest.mark.parametrize("delay", [50, 120, 250, 500, 1000])
def test_recovers_known_delay_clean(delay):
    cap = reference_capture(delay_ms=delay, seed=delay)
    res = analyse_capture(cap)
    assert res.qc.ok(), res.qc.flags
    assert abs(res.mrl_ms - delay) < 5.0, res.mrl_ms


@pytest.mark.parametrize("codec", ["pcmu", "pcma", "l16"])
def test_recovers_known_delay_all_codecs(codec):
    cap = reference_capture(delay_ms=300, codec=codec, seed=7)
    res = analyse_capture(cap)
    assert res.qc.ok(), res.qc.flags
    assert abs(res.mrl_ms - 300) < 5.0, (codec, res.mrl_ms)


def test_ingress_reports_channel_transit_faithfully():
    """A fixed extra transit delay must appear one-for-one in the ingress figure."""
    ing = []
    for seed in range(20):
        r = analyse_capture(reference_capture(delay_ms=400, rx_base_transit_ms=50.0, seed=seed))
        ing.append(r.mrl_ms - 400 - 50.0)
    assert abs(float(np.mean(ing))) < 5.0, np.mean(ing)


def test_playout_buffer_absorbs_jitter_that_ingress_exposes():
    """The two MRLs must differ in exactly the way the physics requires: ingress spread
    tracks the channel's jitter, playout spread does not because the buffer absorbs it."""
    target = 40.0
    ing, pla = [], []
    for seed in range(40):
        r = analyse_capture(reference_capture(delay_ms=400, rx_jitter_ms=30.0, seed=seed),
                            playout_target_ms=target)
        v = r.variants["headline"]
        if np.isfinite(v.mrl_ms):
            ing.append(v.mrl_ms - 400)
            pla.append(v.mrl_playout_ms - 400)
    sd_ing, sd_pla = float(np.std(ing, ddof=1)), float(np.std(pla, ddof=1))
    assert sd_ing > 10.0, sd_ing                       # ingress must expose the jitter
    assert sd_pla < 0.4 * sd_ing, (sd_ing, sd_pla)     # buffer must absorb most of it
    assert abs(float(np.mean(pla)) - target) < 6.0, np.mean(pla)


def test_loss_is_flagged_not_silently_absorbed():
    flagged = 0
    for seed in range(20):
        r = analyse_capture(reference_capture(delay_ms=400, rx_loss_rate=0.10, seed=seed))
        if {"high_loss", "high_late_discard"} & set(r.qc.flags):
            flagged += 1
    assert flagged >= 18, flagged


def test_mispaced_sender_is_rejected_by_qc():
    """The harness must refuse to report a measurement it cannot trust: if its own
    transmit pacing is bad, t0 is unreliable and the call must not be counted."""
    rejected = 0
    for seed in range(20):
        r = analyse_capture(reference_capture(delay_ms=400, tx_jitter_ms=6.0, seed=seed))
        rejected += (not r.qc.ok())
    assert rejected == 20, rejected


def test_onset_definition_uncertainty_grows_with_tts_ramp():
    """The spread across onset variants is the measurement's dominant uncertainty term
    on real systems. It must grow with the ramp, and be near zero on a hard onset."""
    from harness.synth import speechlike as _sl

    def spread(onset_ms):
        biases = []
        for name in ("sensitive", "headline", "strict"):
            errs = []
            for seed in range(6):
                resp = _sl(1.5, seed=seed + 90,
                           onset="fade" if onset_ms else "hard", onset_ms=onset_ms)
                r = analyse_capture(reference_capture(delay_ms=400, response=resp, seed=seed))
                errs.append(r.variants[name].mrl_ms - 400)
            biases.append(float(np.mean(errs)))
        return max(biases) - min(biases)

    hard, ramp60, ramp150 = spread(0.0), spread(60.0), spread(150.0)
    assert hard < 2.0, hard
    assert ramp60 > hard, (hard, ramp60)
    assert ramp150 > ramp60, (ramp60, ramp150)


def test_negative_mrl_is_reported_not_swallowed():
    """A system that answers before the caller finishes must produce a signed result."""
    pcm, _, ann = annotated_prompt(seed=11, trail_silence_ms=900.0)
    cap = reference_capture(delay_ms=-200.0, prompt=pcm,
                            speech_end_sample=ann.speech_end_sample, seed=11)
    res = analyse_capture(cap)
    assert res.mrl_ms < 0
    assert "response_before_speech_end" in res.qc.flags


def test_capture_roundtrips_through_disk(tmp_path):
    cap = reference_capture(delay_ms=250, seed=5)
    p = write_capture(tmp_path / "c.jsonl.gz", cap)
    back = read_capture(p)
    assert isinstance(back, Capture)
    assert back.header.ground_truth_ms == 250.0
    assert len(back.packets) == len(cap.packets)
    assert analyse_capture(back).mrl_ms == pytest.approx(analyse_capture(cap).mrl_ms)


def test_tampered_prompt_is_rejected(tmp_path):
    from harness.prompts import (annotate_file, load_annotation, save_annotation,
                                 write_wav_8k_mono)
    from harness.synth import make_prompt

    pcm, _ = make_prompt(seed=2)
    w = write_wav_8k_mono(tmp_path / "p.wav", pcm)
    save_annotation(w, annotate_file(w))
    load_annotation(w)  # ok
    write_wav_8k_mono(w, np.concatenate([pcm, pcm[:80]]))
    with pytest.raises(ValueError, match="has changed since annotation"):
        load_annotation(w)


# --------------------------------------------------------------- stage 2, live sockets

def test_loopback_recovers_known_delay_over_real_sockets():
    """Stage 2 in miniature. Marked slow-ish but kept in the default suite because it is
    the only test that exercises real send/receive timestamping; a synthetic capture
    cannot catch a sign error in the live path, and one was found here exactly that way.

    Tolerance is loose and QC-gated because pacing is a host property: on a
    power-managed laptop some calls legitimately fail QC and are discarded."""
    from harness.loopback import run_call

    residuals = []
    for delay in (150.0, 350.0):
        cap, diag = run_call(delay, seed=int(delay))
        res = analyse_capture(cap)
        if not res.qc.ok():
            pytest.skip(f"host pacing unfit for stage 2: {res.qc.flags}")
        assert res.qc.rx_packets > 0
        residuals.append(res.mrl_ms - delay)
    assert abs(float(np.mean(residuals))) < 5.0, residuals


# ------------------------------------------------------------ stage 3, two hosts

def _sweep(path: Path, residuals: list[float], self_err: float = 0.01) -> Path:
    """Write a sweep file of the shape `loopback.main` emits.

    Residual sets are given explicitly rather than drawn randomly, because a gate test
    that is itself stochastic tells you nothing on the run where it fails.
    """
    rows = [{"delay_ms": 137.0, "call": i, "residual_ms": r,
             "mrl_ingress_ms": 137.0 + r, "mrl_playout_ms": 177.0 + r,
             "responder_self_error_ms": self_err, "qc_ok": True, "qc_flags": []}
            for i, r in enumerate(residuals)]
    path.write_text(json.dumps({"host": "test-host", "python": "3.12.13", "stage": 3,
                                "peer": "10.0.0.2:9000", "control_failures": 0,
                                "rows": rows, "verdict": []}))
    return path


def _spread(mean: float, sd: float, n: int) -> list[float]:
    """n values with exactly this mean and approximately this standard deviation."""
    half = n // 2
    return [mean - sd] * half + [mean + sd] * half


def test_stage3_gate_recovers_injected_delay(tmp_path, capsys):
    from harness.loopback import compare_sweeps

    base = _sweep(tmp_path / "b.json", _spread(2.0, 0.5, 20))
    netem = _sweep(tmp_path / "n.json", _spread(52.0, 0.5, 20))
    assert compare_sweeps(str(base), str(netem), 50.0) == 0
    assert "stage 3 passed" in capsys.readouterr().out


def test_stage3_gate_detects_netem_not_applied(tmp_path, capsys):
    """The most likely operator error: tc applied to the wrong interface, or not at all."""
    from harness.loopback import compare_sweeps

    base = _sweep(tmp_path / "b.json", _spread(2.0, 0.5, 20))
    netem = _sweep(tmp_path / "n.json", _spread(2.1, 0.5, 20))
    assert compare_sweeps(str(base), str(netem), 50.0) == 1
    assert "not in force" in capsys.readouterr().out


def test_stage3_gate_detects_delay_applied_to_both_legs(tmp_path, capsys):
    from harness.loopback import compare_sweeps

    base = _sweep(tmp_path / "b.json", _spread(2.0, 0.5, 20))
    netem = _sweep(tmp_path / "n.json", _spread(102.0, 0.5, 20))
    assert compare_sweeps(str(base), str(netem), 50.0) == 1
    assert "twice the declared delay" in capsys.readouterr().out


def test_stage3_gate_rejects_negative_baseline_residual(tmp_path, capsys):
    """A response cannot arrive before the programmed delay plus minimum transit, so a
    negative baseline is an arithmetic error rather than anything the path did."""
    from harness.loopback import compare_sweeps

    base = _sweep(tmp_path / "b.json", _spread(-39.5, 0.03, 20))
    netem = _sweep(tmp_path / "n.json", _spread(-39.5, 0.03, 20))
    assert compare_sweeps(str(base), str(netem), 50.0) == 1
    assert "definitional or arithmetic error" in capsys.readouterr().out


def test_stage3_gate_refuses_to_pass_an_underpowered_run(tmp_path, capsys):
    """The difference lands on the declared delay exactly, but the spread is so wide that
    the run cannot support the claim. Reporting a pass here would be dishonest, and the
    fix is more calls rather than a wider tolerance."""
    from harness.loopback import compare_sweeps

    base = _sweep(tmp_path / "b.json", _spread(2.0, 8.0, 6))
    netem = _sweep(tmp_path / "n.json", _spread(52.0, 8.0, 6))
    assert compare_sweeps(str(base), str(netem), 50.0) == 1
    out = capsys.readouterr().out
    assert "INCONCLUSIVE" in out
    assert "Do not widen the tolerance" in out


def test_stage3_gate_needs_a_minimum_number_of_calls(tmp_path):
    from harness.loopback import compare_sweeps

    base = _sweep(tmp_path / "b.json", [2.0, 2.1])
    netem = _sweep(tmp_path / "n.json", [52.0, 52.1])
    assert compare_sweeps(str(base), str(netem), 50.0) == 1


def test_advisory_flags_are_visible_on_a_passing_call():
    """Regression. A stage 3 sweep once printed forty clean-looking calls while
    thirty-eight carried a late-discard advisory, because flags were shown only on QC
    failure. An advisory turns a point estimate into an upper bound, so it has to be
    visible wherever the number is."""
    from harness.loopback import _status

    assert _status({"qc_ok": True, "qc_flags": []}) == "ok"
    assert _status({"qc_ok": True, "qc_flags": ["high_late_discard"]}) == \
        "ok, advisory high_late_discard"
    assert "high_loss" in _status({"qc_ok": True, "qc_flags": ["high_loss", "high_late_discard"]})
    assert _status({"qc_ok": False, "qc_flags": ["tx_pacing_out_of_spec"]}).startswith("QC FAIL")


def test_control_datagrams_round_trip_and_reject_foreign_traffic():
    from harness.loopback import _ctl_decode, _ctl_encode

    msg = _ctl_decode(_ctl_encode("HELLO", {"spec": {"delay_ms": 137.0}}))
    assert msg is not None and msg["kind"] == "HELLO"
    assert msg["spec"]["delay_ms"] == 137.0
    # Anything not carrying our magic is dropped rather than parsed. The control port will
    # receive scans and stray datagrams on any reachable host.
    assert _ctl_decode(b'{"kind": "HELLO"}') is None
    assert _ctl_decode(b"not json at all") is None
    assert _ctl_decode(b"\x80\x00\x00\x01") is None


def test_remote_diag_preserves_responder_self_error():
    """Only the self error crosses the wire; the absolute instants behind it are on the
    responder's clock. The reconstruction must report the same number regardless."""
    from harness.loopback import ResponderDiag

    diag = ResponderDiag.from_remote({"triggered": True, "self_error_ms": 0.0074,
                                      "frames_received": 150, "frames_sent": 60,
                                      "pacing_max_ms": 0.31})
    assert diag.triggered
    assert abs(diag.self_error_ms - 0.0074) < 1e-9
    assert diag.frames_received == 150
    assert abs(diag.pacing.max_ms - 0.31) < 1e-12
    # An untriggered responder has no self error to report, and nan is the honest answer
    # rather than zero, which would understate the residual it feeds into.
    assert not np.isfinite(ResponderDiag.from_remote({"triggered": False}).self_error_ms)
