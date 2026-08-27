# Validation ledger

Per-machine validation results for this harness. Pacing is a property of the host, so
the instrument's accuracy has to be established on every machine that produces a
figure, and this file is the record. A number should be trusted only from a machine
holding a passing row for the stages that number depends on.

Read every row as dated evidence taken under stated conditions. The same laptop can
pass on a quiet morning and refuse an hour later when a background indexer wakes up,
and an operating system update resets the picture entirely. Refusals are recorded
deliberately, since an instrument that visibly disqualifies unfit hosts, including
its own author's, is the quality-control machinery doing its job.

## What each stage qualifies

| stage | command | qualifies | host-dependent? |
|---|---|---|---|
| 1 | `python -m harness.validate` | the metric arithmetic, onset detection and QC gates | no; a failure anywhere means the code is wrong |
| 2 | `python -m harness.loopback` | one host's live capture path, timestamping and pacing | yes |
| 3 | not yet built | a host pair across a real NIC with `netem`-injected delay | yes, both hosts |
| 4 | not yet built | the full SIP calling rig against a real system | yes |

Stages are cumulative for trusting a real measurement (`METHOD.md` §7 requires stage
3 to pass before any real-system figure is published), but each stage's result is a
complete fact about a machine on its own, which is why rows are published per stage
rather than held back until all four exist.

## Adding a row

1. `python -m harness.preflight` first; it is quick and predicts the stage 2 outcome.
2. `python -m harness.validate --json results/validation/<label>-<date>-stage1.json`
3. `python -m harness.loopback --json results/validation/<label>-<date>-stage2.json`

Commit the JSON alongside the row. The date in the filename and the row is the date
the run executed, which may precede the commit. Record the operating system, Python
version, power state and any notable competing load, because those conditions are
part of the result. Choose a short stable label for the machine; never a hostname or a serial
number.

## Machines

| label | hardware | role |
|---|---|---|
| `method-reference` | not recorded; predates this ledger | produced the validated accuracy in `METHOD.md` §5 |
| `mbp-m1pro` | MacBook Pro 18,3, Apple M1 Pro, 10 cores | development machine |

## Results

| machine | stage | date | python | conditions | outcome | key figures | evidence |
|---|---|---|---|---|---|---|---|
| `method-reference` | 1 | 2026-08 | — | — | **passed** | clean-channel headline bias −0.40 ms, sd 0.70 ms, p95 \|e\| 2.38 ms | `METHOD.md` §5 |
| `method-reference` | 2 | 2026-08 | — | — | **passed** | residual bias +0.35 ms, sd 0.06 ms, worst tx pacing 4.91 ms | `METHOD.md` §5.1 |
| `mbp-m1pro` | 1 | 2026-08-24 | 3.12.13 | macOS 26.5.1, mains | **passed**, all gates | clean-channel headline bias −0.40 ms, sd 0.70 ms, p95 \|e\| 2.38 ms; annotator exact over 1200 calls | [json](results/validation/mbp-m1pro-2026-08-24-stage1.json) |
| `mbp-m1pro` | 2 | 2026-08-24 | 3.12.13 | macOS 26.5.1, mains, `caffeinate -di`, heavy background media indexing (load 8 to 12) | **refused**: host pacing | 0/20 calls usable, worst tx pacing 48.0 ms | [json](results/validation/mbp-m1pro-2026-08-24-stage2.json) |

### Notes

**`method-reference`.** The hardware identity of the host behind `METHOD.md` §5 was
not recorded at the time. Its figures stand as the instrument's reference validation;
re-running on that machine with `--json` would upgrade these two rows to
ledger-standard evidence.

**`mbp-m1pro` stage 2.** Five attempts across 21 to 24 August 2026 all ended in
refusal, with usable calls ranging from 0/20 (during background media indexing) to
9/20 (mains, quiet), against the 16/20 the verdict requires. On the calls that did
survive QC, residual accuracy was consistently around +1 ms after subtracting the
responder's own emission error, and the verdict attributed the failures to power
management rather than a code defect on every run. The machine is therefore fully
valid for development, the test suite and stage 1, and unfit as a measurement host,
which is the expected classification for a power-managed laptop. It is also a worked
example of why conditions belong in every row: plugging into mains tripled the yield,
and a background indexer later took the same configuration to zero.
