# voice-ai-latency-harness

[![tests](https://github.com/dnygate/voice-ai-latency-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/dnygate/voice-ai-latency-harness/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/1341998349.svg)](https://doi.org/10.5281/zenodo.22124823)

Measures **mouth-to-ear response latency** for voice AI systems on a real SIP/RTP media
path: end of the caller's speech to arrival of the system's first response audio.

Published voice AI latency figures quote model-side first-audio time, which is one or
two terms of an eight-term sum and omits the largest one, endpointing. This measures
the sum.

See [METHOD.md](METHOD.md) for definitions, the uncertainty budget, and validated
instrument accuracy. Read it before quoting any number this produces.
[VALIDATION.md](VALIDATION.md) records per-machine validation results; a figure is
only as trustworthy as the row of the machine that produced it.

## Quick start

    pip install numpy scipy pytest
    python -m pytest tests -q          # 33 tests, ~6 s
    python -m harness.validate        # instrument accuracy sweep, ~40 s

`validate` replies to each prompt at a known delay and reports how accurately that
delay is recovered, per channel condition and per onset definition. All gates must pass
before the harness is pointed at a real system.

## Current status

| stage | what it proves | status |
|---|---|---|
| 1. in-process | metric arithmetic, onset detection, QC gates | **passing** |
| 2. UDP loopback | live capture path and its timestamping | **passing**, per machine ([VALIDATION.md](VALIDATION.md)) |
| 3. remote + netem | end to end across a real NIC, two clocks | not implemented |
| 4. SIP caller | real calls against a real system | not implemented |

Instrument accuracy, stage 1: **bias −0.40 ms, p95 |error| 2.38 ms** on a clean G.711
channel with 20 ms frames, t0 annotation exact with zero variance. Stage 2 over real
sockets, on the reference host documented in [METHOD.md](METHOD.md) §5.1: **bias
+0.35 ms, sd 0.06 ms** across 15 calls. That run predates the per-machine ledger, so
its raw JSON is not in the repository. Stage 2 is a property of each machine: the
stage 2 artifact that *is* committed shows a development laptop being refused by QC
(0/20 usable calls), which is the designed behaviour for a host that cannot pace a
frame grid. [VALIDATION.md](VALIDATION.md) holds the per-machine record.

Run [`python -m harness.preflight`](harness/preflight.py) on any new machine first: it
measures whether that host can pace a frame grid accurately enough to be trusted.

## Layout

    harness/g711.py     mu-law / A-law / L16, ITU reference algorithm, table-driven
    harness/prompts.py  prompt loading and sample-precise speech-end annotation
    harness/record.py   capture schema: raw payloads and timestamps, nothing derived
    harness/analyse.py  offline metric derivation, both MRLs, QC gates
    harness/synth.py    signal generators and the known-delay reference responder
    harness/validate.py stage 1 sweep with analytically predicted gates
    harness/loopback.py stage 2 over real UDP sockets, hybrid sleep+spin pacing
    harness/preflight.py host capability check: can this machine hold a frame grid
    tests/              correctness, including G.711 against frozen golden vectors

## Design rules

**Captures store nothing derived.** Raw payloads, raw timestamps. Metric definitions
can be revised, or a reviewer's alternative applied, without re-running a test that
costs GPU hours.

**Both MRLs, always.** Ingress isolates the system; playout is what the caller waits.
Neither alone is publishable.

**Onset is a distribution, not a number.** Three onset definitions; the spread is the
uncertainty and it travels with the headline figure.

**The harness refuses measurements it cannot trust.** Bad transmit pacing means an
unreliable t0, so the call is dropped rather than reported.

## Not in here

Endpointing strategy, barge-in handling, model configuration. The harness measures; it
does not implement the thing being measured. That boundary is deliberate.
