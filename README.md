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
    python -m pytest tests -q          # 42 tests, ~7 s
    python -m harness.validate        # instrument accuracy sweep, ~40 s

`validate` replies to each prompt at a known delay and reports how accurately that
delay is recovered, per channel condition and per onset definition. All gates must pass
before the harness is pointed at a real system.

## Current status

| stage | what it proves | status |
|---|---|---|
| 1. in-process | metric arithmetic, onset detection, QC gates | **passing** |
| 2. UDP loopback | live capture path and its timestamping | **passing**, per machine ([VALIDATION.md](VALIDATION.md)) |
| 3. remote + netem | end to end across a real NIC, two clocks | **passing**, per host pair ([VALIDATION.md](VALIDATION.md)) |
| 4. SIP caller | real calls against a real system | not implemented |

Instrument accuracy, stage 1: **bias −0.40 ms, p95 |error| 2.38 ms** on a clean G.711
channel with 20 ms frames, t0 annotation exact with zero variance. Those figures have
been reproduced to four significant figures on two unrelated machines, which is what a
host-independent stage should do.

Stage 2 over real sockets, on a qualified host: **bias +0.21 ms, sd 0.02 ms** across
20 of 20 usable calls. Stage 3 across a real NIC and two hosts recovers a `netem`-injected
50 ms delay to **+0.08 ms**, with a standard error of 0.03 ms, using a differential gate
against a no-impairment baseline over the same host pair.

Both figures are properties of a particular machine or pair rather than of the code, which
is why they carry ledger rows. [VALIDATION.md](VALIDATION.md) holds that record, including
refusals: one committed stage 2 artifact shows a development laptop being turned away by QC
with 0 of 20 usable calls, which is the designed behaviour for a host that cannot pace a
frame grid.

Run [`python -m harness.preflight`](harness/preflight.py) on any new machine first: it
measures whether that host can pace a frame grid accurately enough to be trusted.

## Layout

    harness/g711.py     mu-law / A-law / L16, ITU reference algorithm, table-driven
    harness/prompts.py  prompt loading and sample-precise speech-end annotation
    harness/record.py   capture schema: raw payloads and timestamps, nothing derived
    harness/analyse.py  offline metric derivation, both MRLs, QC gates
    harness/synth.py    signal generators and the known-delay reference responder
    harness/validate.py stage 1 sweep with analytically predicted gates
    harness/loopback.py stage 2 over real UDP and stage 3 across two hosts, with a
                        control channel and a differential gate against a baseline
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

**A greeting is not a response.** Deployed systems speak first, before the caller has
said anything, so onset detection begins after the greeting rather than at the start of
the stream. Scanning from zero measured −2985 ms of error and passed quality control.

**Filler is separated from the answer without interpreting it.** A system that plays a
noise while it thinks is distinguished by the silence that follows, not by recognising
what was said, so the figure stays reproducible from a published capture.

## Not in here

Endpointing strategy, barge-in handling, model configuration. The harness measures; it
does not implement the thing being measured. That boundary is deliberate.
