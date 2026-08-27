"""Stage 1 validation: establish the harness's instrument error before any GPU is booked.

Run:  python -m harness.validate
      python -m harness.validate --json results/stage1.json

Method
------
The reference responder replies exactly `delay_ms` after the prompt's final speech
sample is transmitted, so ground truth is known to the nanosecond. Under each channel
condition the *expected* offset from ground truth is predictable analytically:

  ingress:  E = rx_base_transit_ms + mean(jitter excess)
  playout:  E = rx_base_transit_ms + playout_target_ms

so the test is not merely "is the answer close to the truth" but "does the instrument
reproduce the offset that physics says it must". A harness that passed the first test
and failed the second would be right by accident.

Residual bias is what remains after subtracting E. That is instrument error, and it is
the figure to quote as accuracy in any published result.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .analyse import DEFAULT_VARIANTS, analyse_capture
from .synth import annotated_prompt, reference_capture, speechlike

DELAYS_MS = (50, 100, 150, 200, 300, 450, 600, 800, 1000, 1500)
SEEDS = range(8)
PLAYOUT_TARGET_MS = 40.0

# Acceptance gates on residual bias and precision, applied to the headline variant.
# Rationale: one G.711 frame is 20 ms, and onset can only be localised to the frame
# that carried it plus within-frame interpolation, so a few ms of spread is structural.
# A residual bias above a quarter-frame would be large enough to distort comparisons
# between systems whose true difference is tens of ms.
GATE_RESIDUAL_BIAS_MS = 5.0
GATE_RESIDUAL_P95_MS = 12.0
# Under jitter, ingress spread SHOULD equal the channel's jitter spread -- that is the
# instrument reporting the channel faithfully, not imprecision. So jittered conditions
# are gated on whether observed ingress sd matches the analytically predicted sd, and on
# whether the de-jitter buffer absorbs the jitter as it must. Gamma(shape=k) with mean m
# has sd = m / sqrt(k), and k = 2 here.
JITTER_SHAPE = 2.0
GATE_SD_TOL_MS = 3.0
GATE_SD_TOL_FRAC = 0.35

# What each condition is entitled to be tested for.
#   absolute  : hard onset, undegraded channel -> gate bias AND precision, both paths
#   jitter    : gate ingress sd against prediction, and gate playout bias + precision
#   flagged   : degraded channel -> require the advisory QC flags to fire; do not gate
#               accuracy, because losing or late-discarding the onset frame legitimately
#               defers detection
#   report    : onset-definition sensitivity -> reported as uncertainty, never gated
GATES = {
    "clean_pcmu": "absolute", "clean_pcma": "absolute", "clean_l16": "absolute",
    "ptime10": "absolute", "ptime30": "absolute", "transit_50ms": "absolute",
    "noisy_channel": "absolute",
    "jitter_10ms": "jitter", "jitter_30ms": "jitter", "transit_50_jitter_20": "jitter",
    "loss_3pct": "flagged", "loss_10pct_jitter_30ms": "flagged",
    "bad_tx_pacing": "must_fail_qc",
    "tts_soft_onset_60ms": "report", "tts_soft_onset_150ms": "report",
}


@dataclass
class Condition:
    name: str
    codec: str = "pcmu"
    ptime_ms: float = 20.0
    tx_jitter_ms: float = 0.0
    rx_jitter_ms: float = 0.0          # MEAN of one-sided gamma excess delay
    rx_base_transit_ms: float = 0.0
    rx_loss_rate: float = 0.0
    rx_floor_dbov: float = -62.0
    response_onset: str = "hard"
    response_onset_ms: float = 0.0
    note: str = ""

    def expected_ingress_ms(self) -> float:
        return self.rx_base_transit_ms + self.rx_jitter_ms

    def expected_playout_ms(self) -> float:
        return self.rx_base_transit_ms + PLAYOUT_TARGET_MS

    def expected_ingress_sd_ms(self) -> float:
        return self.rx_jitter_ms / np.sqrt(JITTER_SHAPE)

    @property
    def gate(self) -> str:
        return GATES.get(self.name, "report")

    def capture_kwargs(self) -> dict:
        return {
            "codec": self.codec, "ptime_ms": self.ptime_ms,
            "tx_jitter_ms": self.tx_jitter_ms, "rx_jitter_ms": self.rx_jitter_ms,
            "rx_base_transit_ms": self.rx_base_transit_ms,
            "rx_loss_rate": self.rx_loss_rate, "rx_floor_dbov": self.rx_floor_dbov,
        }


CONDITIONS: tuple[Condition, ...] = (
    Condition("clean_pcmu", note="baseline; G.711 mu-law, the North American PSTN default"),
    Condition("clean_pcma", codec="pcma", note="G.711 A-law, the European PSTN default"),
    Condition("clean_l16", codec="l16", note="unquantised control: isolates codec effects"),
    Condition("ptime10", ptime_ms=10.0, note="finer frame grid should improve onset precision"),
    Condition("ptime30", ptime_ms=30.0, note="coarser grid should degrade it"),
    Condition("jitter_10ms", rx_jitter_ms=10.0, tx_jitter_ms=0.3, note="typical managed network"),
    Condition("jitter_30ms", rx_jitter_ms=30.0, tx_jitter_ms=0.3, note="congested access"),
    Condition("transit_50ms", rx_base_transit_ms=50.0, note="cross-continent leg"),
    Condition("transit_50_jitter_20", rx_base_transit_ms=50.0, rx_jitter_ms=20.0,
              note="realistic long-haul SIP trunk"),
    Condition("loss_3pct", rx_loss_rate=0.03, note="tolerable loss"),
    Condition("loss_10pct_jitter_30ms", rx_loss_rate=0.10, rx_jitter_ms=30.0,
              note="degraded; onset frame itself may be lost"),
    Condition("noisy_channel", rx_floor_dbov=-45.0,
              note="high channel noise stresses the onset threshold"),
    Condition("bad_tx_pacing", tx_jitter_ms=6.0,
              note="deliberately mispaced sender; QC must reject every call"),
    Condition("tts_soft_onset_60ms", response_onset="fade", response_onset_ms=60.0,
              note="realistic TTS ramp-in; quantifies onset-definition uncertainty"),
    Condition("tts_soft_onset_150ms", response_onset="fade", response_onset_ms=150.0,
              note="slow ramp-in; upper bound on onset-definition uncertainty"),
)


def run_condition(cond: Condition, delays=DELAYS_MS, seeds=SEEDS) -> list[dict]:
    rows: list[dict] = []
    for delay in delays:
        for seed in seeds:
            pcm, true_end, ann = annotated_prompt(seed=seed)
            resp = speechlike(1.5, seed=seed + 500, onset=cond.response_onset,
                              onset_ms=cond.response_onset_ms)
            cap = reference_capture(
                delay_ms=delay, prompt=pcm, speech_end_sample=ann.speech_end_sample,
                response=resp, seed=seed, call_id=f"{cond.name}-{delay}-{seed}",
                **cond.capture_kwargs(),
            )
            res = analyse_capture(cap, playout_target_ms=PLAYOUT_TARGET_MS)
            row = {
                "condition": cond.name, "delay_ms": delay, "seed": seed,
                "annotator_err_ms": (ann.speech_end_sample - true_end) * 1000.0 / 8000.0,
                "qc_ok": res.qc.ok(), "qc_flags": list(res.qc.flags),
                "rx_loss_pct": round(res.qc.rx_loss_pct, 2),
                "rx_late_discard_frames": res.qc.rx_late_discard_frames,
                "noise_floor_dbov": round(res.qc.rx_noise_floor_dbov, 1),
            }
            for v in DEFAULT_VARIANTS:
                vr = res.variants[v.name]
                ing, pla = vr.mrl_ms, vr.mrl_playout_ms
                row[f"resid_ingress_{v.name}"] = (
                    None if not np.isfinite(ing) else ing - delay - cond.expected_ingress_ms())
                row[f"resid_playout_{v.name}"] = (
                    None if not np.isfinite(pla) else pla - delay - cond.expected_playout_ms())
            rows.append(row)
    return rows


def _stats(vals: list[float | None]) -> dict:
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "bias_ms": round(float(a.mean()), 2),
        "sd_ms": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 2),
        "p95_abs_ms": round(float(np.percentile(np.abs(a), 95)), 2),
        "max_abs_ms": round(float(np.abs(a).max()), 2),
    }


def summarise(rows: list[dict]) -> dict:
    out: dict = {"annotator": {}, "conditions": {}, "playout_target_ms": PLAYOUT_TARGET_MS}
    ae = np.array([r["annotator_err_ms"] for r in rows], dtype=float)
    out["annotator"] = {"n": int(ae.size), "bias_ms": round(float(ae.mean()), 4),
                        "max_abs_ms": round(float(np.abs(ae).max()), 4)}
    for cond in CONDITIONS:
        sub = [r for r in rows if r["condition"] == cond.name and r["qc_ok"]]
        allsub = [r for r in rows if r["condition"] == cond.name]
        if not allsub:
            continue
        flagset: set[str] = set()
        for r in allsub:
            flagset |= set(r["qc_flags"])
        entry: dict = {
            "note": cond.note, "gate": cond.gate,
            "n": len(allsub), "n_qc_fail": len(allsub) - len(sub),
            "flags_seen": sorted(flagset),
            "expected_ingress_sd_ms": round(cond.expected_ingress_sd_ms(), 2),
            "expected_ingress_ms": cond.expected_ingress_ms(),
            "expected_playout_ms": cond.expected_playout_ms(),
            "variants": {},
        }
        for v in DEFAULT_VARIANTS:
            entry["variants"][v.name] = {
                "ingress": _stats([r[f"resid_ingress_{v.name}"] for r in sub]),
                "playout": _stats([r[f"resid_playout_{v.name}"] for r in sub]),
            }
        out["conditions"][cond.name] = entry
    return out


def check_gates(summary: dict, variant: str = "headline") -> list[str]:
    fails: list[str] = []
    for name, e in summary["conditions"].items():
        gate = e["gate"]
        ing = e["variants"][variant]["ingress"]
        pla = e["variants"][variant]["playout"]

        if gate == "must_fail_qc":
            if e["n_qc_fail"] < e["n"]:
                fails.append(f"{name}: QC accepted {e['n'] - e['n_qc_fail']}/{e['n']} calls "
                             f"from a deliberately mispaced sender; the pacing gate is not working")
            continue
        if gate == "report":
            continue

        if ing.get("n", 0) == 0 or pla.get("n", 0) == 0:
            fails.append(f"{name}: no usable measurements")
            continue

        if gate == "flagged":
            required = {"high_loss", "high_late_discard"}
            if not required & set(e["flags_seen"]):
                fails.append(f"{name}: degraded channel produced none of {sorted(required)}; "
                             f"a lossy call would be reported as if it were clean")
            continue

        if gate == "absolute":
            for kind, st in (("ingress", ing), ("playout", pla)):
                if abs(st["bias_ms"]) > GATE_RESIDUAL_BIAS_MS:
                    fails.append(f"{name}/{kind}: residual bias {st['bias_ms']:+.2f} ms "
                                 f"exceeds {GATE_RESIDUAL_BIAS_MS} ms")
                if st["p95_abs_ms"] > GATE_RESIDUAL_P95_MS:
                    fails.append(f"{name}/{kind}: residual p95 {st['p95_abs_ms']:.2f} ms "
                                 f"exceeds {GATE_RESIDUAL_P95_MS} ms")
        elif gate == "jitter":
            exp_sd = e["expected_ingress_sd_ms"]
            tol = max(GATE_SD_TOL_MS, GATE_SD_TOL_FRAC * exp_sd)
            if abs(ing["sd_ms"] - exp_sd) > tol:
                fails.append(f"{name}/ingress: sd {ing['sd_ms']:.2f} ms does not match the "
                             f"predicted channel jitter sd {exp_sd:.2f} ms (tol {tol:.2f})")
            if abs(pla["bias_ms"]) > GATE_RESIDUAL_BIAS_MS:
                fails.append(f"{name}/playout: residual bias {pla['bias_ms']:+.2f} ms "
                             f"exceeds {GATE_RESIDUAL_BIAS_MS} ms")
            if pla["p95_abs_ms"] > GATE_RESIDUAL_P95_MS:
                fails.append(f"{name}/playout: residual p95 {pla['p95_abs_ms']:.2f} ms "
                             f"exceeds {GATE_RESIDUAL_P95_MS} ms -- the de-jitter buffer "
                             f"is not absorbing the jitter it should")
    return fails


def onset_uncertainty(summary: dict) -> dict:
    """Spread of residual bias across onset variants: the onset-definition uncertainty.

    This is the dominant term in the measurement's uncertainty budget on real systems,
    because real TTS ramps in rather than starting at full level. It must be published
    alongside any headline latency figure.
    """
    out = {}
    for name, e in summary["conditions"].items():
        biases = [e["variants"][v]["ingress"].get("bias_ms") for v in
                  ("sensitive", "headline", "strict")]
        biases = [b for b in biases if b is not None]
        if len(biases) >= 2:
            out[name] = {"per_variant_bias_ms": biases,
                         "spread_ms": round(max(biases) - min(biases), 2)}
    return out


def _table(summary: dict, variant: str) -> str:
    head = (f"{'condition':<24}{'qcf':>4}{'expI':>6}{'expP':>6}"
            f"{'  | ingress: bias    sd  p95|e|':<32}{'  | playout: bias    sd  p95|e|':<32}")
    lines = [head, "-" * len(head)]
    for name, e in summary["conditions"].items():
        i = e["variants"][variant]["ingress"]
        p = e["variants"][variant]["playout"]
        if i.get("n", 0) == 0:
            lines.append(f"{name:<24}{e['n_qc_fail']:>4}  -- no usable measurements --")
            continue
        lines.append(
            f"{name:<24}{e['n_qc_fail']:>4}{e['expected_ingress_ms']:>6.0f}"
            f"{e['expected_playout_ms']:>6.0f}"
            f"  |{i['bias_ms']:>+9.2f}{i['sd_ms']:>7.2f}{i['p95_abs_ms']:>8.2f}"
            f"       |{p['bias_ms']:>+9.2f}{p['sd_ms']:>7.2f}{p['p95_abs_ms']:>8.2f}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="write full rows and summary to this path")
    ap.add_argument("--variant", default="headline")
    ap.add_argument("--quick", action="store_true", help="fewer delays and seeds")
    args = ap.parse_args()

    delays = (100, 300, 800) if args.quick else DELAYS_MS
    seeds = range(3) if args.quick else SEEDS

    rows: list[dict] = []
    for cond in CONDITIONS:
        rows += run_condition(cond, delays=delays, seeds=seeds)
    summary = summarise(rows)

    a = summary["annotator"]
    print(f"\nprompt annotator (t0 definition): n={a['n']} bias={a['bias_ms']:+.4f} ms "
          f"max|err|={a['max_abs_ms']:.4f} ms")
    print(f"de-jitter playout target: {PLAYOUT_TARGET_MS:.0f} ms")
    print(f"residuals below are AFTER subtracting the analytically expected offset "
          f"(expI / expP columns, ms)\n")
    for v in DEFAULT_VARIANTS:
        print(f"== onset variant {v.name!r}: >{v.margin_db} dB over floor, "
              f">{v.abs_dbov} dBov absolute, sustained {v.sustain_ms} ms")
        print(_table(summary, v.name), "\n")

    unc = onset_uncertainty(summary)
    print("onset-definition uncertainty (spread of ingress bias across the three variants):")
    for name in ("clean_pcmu", "tts_soft_onset_60ms", "tts_soft_onset_150ms"):
        if name in unc:
            print(f"  {name:<24} {unc[name]['spread_ms']:>6.2f} ms  "
                  f"(per-variant {unc[name]['per_variant_bias_ms']})")
    print()

    fails = check_gates(summary, args.variant)
    summary["onset_uncertainty"] = unc
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"host": platform.platform(), "python": platform.python_version(),
             "rows": rows, "summary": summary, "gate_failures": fails}, indent=2))
        print(f"wrote {args.json}")
    if fails:
        print("GATE FAILURES:")
        for f in fails:
            print("  -", f)
        return 1
    print(f"all gates passed for variant {args.variant!r}: "
          f"|residual bias| <= {GATE_RESIDUAL_BIAS_MS} ms, "
          f"residual p95 <= {GATE_RESIDUAL_P95_MS} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
