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
| 3 | `--responder` / `--peer`, then `--compare` | a host pair across a real NIC with `netem`-injected delay | yes, both hosts |
| 4 | not yet built | the full SIP calling rig against a real system | yes |

Stages are cumulative for trusting a real measurement (`METHOD.md` §7 requires stage
3 to pass before any real-system figure is published), but each stage's result is a
complete fact about a machine on its own, which is why rows are published per stage
rather than held back until all four exist.

## Adding a row

1. `python -m harness.preflight` first; it is quick and predicts the stage 2 outcome.
   Keep its output as `results/validation/<label>-<date>-preflight.txt`, because it is
   the cheapest available record of what the host's timing was like on the day and it
   explains a later refusal without anyone having to re-run anything.
2. `python -m harness.validate --json results/validation/<label>-<date>-stage1.json`
3. `python -m harness.loopback --json results/validation/<label>-<date>-stage2.json`
4. For stage 3, a pair of sweeps over the same host pair and the differential result:
   `--peer H:P --json <label>-<date>-stage3-baseline.json` with `netem delay 0ms`, the
   same again as `-stage3-netem.json` with the delay applied, then `--compare` over the
   two. Record the interface name, the exact `tc` invocation and the `tc qdisc show`
   output for both sweeps, since the qdisc is part of the conditions in the same way that
   mains power was for a laptop.

   Five things learned qualifying the first pair, each of which cost a sweep or would
   have. Use fixed-performance instances and never burstable ones, because CPU credits
   are the cloud equivalent of laptop power management. Put `netem` on one host only:
   applying it to both counts the delay on each leg, and `compare_sweeps` reports that as
   a doubling. Run `netem` in *both* sweeps, at `delay 0ms` for the baseline, so the
   queuing discipline is identical and only the delay differs. Allow all UDP between the
   hosts rather than the control port alone, since media sockets bind ephemerally and a
   port-9000-only rule completes the handshake and then receives no audio. And size the
   sweeps for the jitter: at 15 ms of jitter, 25 calls leaves the standard error wider
   than the 5 ms gate and `compare_sweeps` correctly refuses to pass an underpowered run,
   where 40 calls does not.

   Always pass `--save-captures`. Two of the nine findings in `METHOD.md` §6 were found
   only by re-analysing kept captures, and every figure in the stage 3 rows below was
   re-derived from them without restarting a host.

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
| `ec2-c7i-caller` | AWS `c7i.large`, 2 vCPU, eu-west-2b, Ubuntu, Python 3.14.4 | stage 3 caller, and the measuring host |
| `ec2-c7i-responder` | AWS `c7i.large`, 2 vCPU, eu-west-2b, Ubuntu, Python 3.14.4 | stage 3 reference responder, carries `netem` |

## Results

Stage 1 rows dated before 2026-09-04 were derived with the t1 refinement described under
finding nine in `METHOD.md` §6; the analyser was corrected that day and the row of that date
carries the current figures. Earlier rows stand as dated evidence of the analyser they ran.

| machine | stage | date | python | conditions | outcome | key figures | evidence |
|---|---|---|---|---|---|---|---|
| `method-reference` | 1 | 2026-08 | — | — | **passed** | clean-channel headline bias −0.40 ms, sd 0.70 ms, p95 \|e\| 2.38 ms | `METHOD.md` §5 |
| `method-reference` | 2 | 2026-08 | — | — | **passed** | residual bias +0.35 ms, sd 0.06 ms, worst tx pacing 4.91 ms | `METHOD.md` §5.1 |
| `mbp-m1pro` | 1 | 2026-08-24 | 3.12.13 | macOS 26.5.1, mains | **passed**, all gates | clean-channel headline bias −0.40 ms, sd 0.70 ms, p95 \|e\| 2.38 ms; annotator exact over 1200 calls | [json](results/validation/mbp-m1pro-2026-08-24-stage1.json) |
| `mbp-m1pro` | 2 | 2026-08-24 | 3.12.13 | macOS 26.5.1, mains, `caffeinate -di`, heavy background media indexing (load 8 to 12) | **refused**: host pacing | 0/20 calls usable, worst tx pacing 48.0 ms | [json](results/validation/mbp-m1pro-2026-08-24-stage2.json) |
| `ec2-c7i-responder` | 1 | 2026-08-31 | 3.14.4 | c7i.large, eu-west-2b, Ubuntu, Linux 7.0.0-1006-aws, numpy 2.5.2 | **passed**, all gates | reproduces `METHOD.md` §5 to four significant figures: clean-channel headline bias −0.40 ms, sd 0.70 ms, p95 \|e\| 2.38 ms; annotator exact over 1200 calls; mispaced sender rejected on all 80 | [json](results/validation/ec2-c7i-responder-2026-08-31-stage1.json) |
| `ec2-c7i-responder` | 2 | 2026-08-31 | 3.14.4 | as above | **passed** | 20/20 usable, residual bias +0.21 ms, sd 0.02 ms, p95 \|e\| 0.25 ms, worst tx pacing 0.14 ms, responder self-error +0.00 ms on every call | [json](results/validation/ec2-c7i-responder-2026-08-31-stage2.json) |
| `ec2-c7i-caller` | 1 | 2026-08-31 | 3.14.4 | c7i.large, eu-west-2b, Ubuntu, Linux 7.0.0-1006-aws, numpy 2.5.2 | **passed**, all gates | identical to the responder's stage 1 figures, as a host-independent stage should be | [json](results/validation/ec2-c7i-caller-2026-08-31-stage1.json) |
| `ec2-c7i-caller` | 2 | 2026-08-31 | 3.14.4 | as above | **passed** | 20/20 usable, residual bias +0.23 ms, sd 0.04 ms, p95 \|e\| 0.30 ms, worst tx pacing 3.82 ms on one call and 0.15 ms elsewhere | [json](results/validation/ec2-c7i-caller-2026-08-31-stage2.json) |
| `ec2-c7i-caller` + `ec2-c7i-responder` | 3 | 2026-08-31 | 3.14.4 | pair as above, same AZ, private addressing; `netem delay 50ms 15ms distribution paretonormal` on the responder's `enp39s0` egress only, caller qdisc untouched | **passed** | 50 ms declared, difference **+47.75 ms**, error **−2.25 ms** against a 5 ms tolerance, standard error 1.81 ms; 40/40 usable in both sweeps; advisory `high_late_discard` on 38/40 of the netem sweep, so its playout figures are upper bounds | [baseline](results/validation/ec2-c7i-caller-2026-08-31-stage3-baseline.json), [netem](results/validation/ec2-c7i-caller-2026-08-31-stage3-netem.json) |
| `ec2-c7i-caller` + `ec2-c7i-responder` | 3 | 2026-08-31 | 3.14.4 | as above but `netem delay 50ms` with no jitter, isolating the instrument from netem's distribution | **passed** | 50 ms declared, difference **+50.08 ms**, error **+0.08 ms**, standard error 0.03 ms, residual sd 0.11 ms; 20/20 usable, no advisory flags | [baseline](results/validation/ec2-c7i-caller-2026-08-31-stage3-baseline.json), [no-jitter](results/validation/ec2-c7i-caller-2026-08-31-stage3-netem-nojitter.json) |
| `ec2-c7i-caller` + `ec2-c7i-responder` | 3 | 2026-08-31 | 3.14.4 | as above plus `loss 1%`; run to exercise the advisory flags, not to measure accuracy | **advisory exercised** | 20/20 usable, `high_loss` on 3 calls and `high_late_discard` on 19, measured loss mean 1.24% against 1% declared | [json](results/validation/ec2-c7i-caller-2026-08-31-stage3-loss.json) |
| `ec2-c7i-caller` + `ec2-c7i-responder` | 3 | 2026-09-03 | 3.14.4 | fresh baseline, `netem delay 0ms`, captures kept | **passed** (baseline) | 40/40 usable, residual +0.65 ms, sd 0.04 ms; Sunday's was +0.60, so the pair and path are unchanged | [json](results/validation/ec2-c7i-caller-2026-09-03-stage3-baseline.json), [captures](results/captures/ec2-c7i-2026-09-03/stage3-baseline/) |
| `ec2-c7i-caller` + `ec2-c7i-responder` | 3 | 2026-09-03 | 3.14.4 | `netem delay 137ms`, no jitter, a second injected value off the frame grid | **passed** | 137 ms declared, difference **+137.01 ms**, error **+0.01 ms**, standard error 0.02 ms; 20/20 usable. `high_late_discard` on 20/20 is a responder artefact, see notes | [json](results/validation/ec2-c7i-caller-2026-09-03-stage3-netem137.json), [captures](results/captures/ec2-c7i-2026-09-03/stage3-netem137/) |
| `ec2-c7i-caller` + `ec2-c7i-responder` | 3 | 2026-09-03 | 3.14.4 | `netem delay 50ms 15ms distribution paretonormal`, captures kept for re-analysis at any playout target | **collected** | 40/40 usable; late-discard rate 3.2% at a 40 ms target, 1.1% at 60, zero from 80 ms. Playout spread does not fall with target, for the responder reason in the notes | [json](results/validation/ec2-c7i-caller-2026-09-03-stage3-netem50j15.json), [captures](results/captures/ec2-c7i-2026-09-03/stage3-netem50j15/) |
| `ec2-c7i-caller` | 2 | 2026-09-03 | 3.14.4 | `--ptime 10`, 10 ms packetisation | **passed** | 20/20 usable, residual +0.23 ms, sd 0.05 ms, worst tx pacing 0.14 ms; first ledger row at 10 ms | [json](results/validation/ec2-c7i-caller-2026-09-03-stage2-ptime10.json), [captures](results/captures/ec2-c7i-2026-09-03/stage2-ptime10/) |
| `ec2-c7i-caller` + `ec2-c7i-responder` | 3 | 2026-09-04 | 3.14.4 | corrected responder (`ef39ddf`), `netem delay 0ms`, captures kept | **passed** (baseline) | 40/40 usable; as analysed residual −0.74 ms, sd 1.63, which the notes trace to the t1 onset refinement rather than the path; re-deriving t1 from response-packet arrival gives +0.60 ms, sd 0.02, matching both earlier baselines | [json](results/validation/ec2-c7i-caller-2026-09-04-stage3-baseline.json), [captures](results/captures/ec2-c7i-2026-09-04/stage3-baseline/) |
| `ec2-c7i-caller` + `ec2-c7i-responder` | 3 | 2026-09-04 | 3.14.4 | corrected responder, `netem delay 137ms`, no jitter | **passed** | 137 declared, difference **+137.38 ms**, error **+0.38 ms**, SE 0.31; 20/20 usable; **no advisory flags**, against `high_late_discard` on 20/20 with the old responder; comfort-noise gap 50.0 to 20.9 ms and transit phase −87 to −0.09 ms on the same call | [json](results/validation/ec2-c7i-caller-2026-09-04-stage3-netem137.json), [captures](results/captures/ec2-c7i-2026-09-04/stage3-netem137/) |
| `ec2-c7i-caller` + `ec2-c7i-responder` | 3 | 2026-09-04 | 3.14.4 | corrected responder, `netem delay 50ms 15ms distribution paretonormal`, captures kept | **collected, stage 3 closed** | 40/40 usable; re-analysed at every target from 40 to 150 ms: ingress residual sd 11.60, **playout residual sd 2.68** (old responder 9.54), late discard 4.4% at 40 ms, 1.4% at 60, zero from 80. The buffer absorbs jitter over a real path as stage 1 predicted at 3.05 | [json](results/validation/ec2-c7i-caller-2026-09-04-stage3-netem50j15.json), [captures](results/captures/ec2-c7i-2026-09-04/stage3-netem50j15/) |
| `mbp-m1pro` | 1 | 2026-09-04 | 3.12.13 | macOS 26.5.1, mains; analyser corrected for finding nine | **passed**, all gates | clean-channel headline bias +0.01 ms, sd 0.05 ms, p95 \|e\| 0.10 ms; hard-onset variant spread 0.02 ms (was 0.99); annotator exact over 1200 calls | [json](results/validation/mbp-m1pro-2026-09-04-stage1.json) |

### Notes

**`method-reference`.** The hardware identity of the host behind `METHOD.md` §5 was
not recorded at the time. Its figures stand as the instrument's reference validation;
re-running on that machine with `--json` would upgrade these two rows to
ledger-standard evidence.

**The `ec2-c7i` pair.** Preflight on 2026-08-31 returned a worst pacing deviation of
0.002 ms on both hosts, on the 20 ms grid and on the 10 ms grid alike, against the 5 ms
QC threshold. That is roughly four orders of magnitude better than the same measurement
on `mbp-m1pro`, whose worst was 16.49 ms, and it is the clearest illustration in this
file of why pacing is recorded per machine rather than assumed from the code. Monotonic
clock resolution is 1 ns on both. Plain `time.sleep()` also passes on these hosts, unlike
on macOS, and the harness continues to spin regardless, because the margin costs one core
and the alternative is trusting a scheduler.

Two properties worth noting for later. The 10 ms grid passing means these hosts can
support measurement at a 10 ms packetisation time, which no machine in this ledger has
previously been able to claim. And running on Python 3.14 means the `audioop` cross-check
is skipped and the frozen golden vectors are the only codec oracle in play, which is
precisely the situation those vectors were frozen for.

**The `ec2-c7i` pair, stage 3.** The differential recovered the declared 50 ms to within
2.25 ms. That error sits inside the instrument's own stage 1 precision of 2.38 ms p95, so
it needs no separate explanation, and this run cannot distinguish residual instrument bias
from `netem`'s `paretonormal` table carrying a slightly non-zero mean. The measured
residual sd of 11.43 ms against a nominal 15 ms jitter parameter is weak evidence for the
second reading, and a no-jitter repeat at the same base delay would separate them.

A no-jitter repeat the same day settled it. With `netem delay 50ms` and no distribution,
the difference came to **+50.08 ms, an error of +0.08 ms** with a standard error of
0.03 ms and residual sd 0.11 ms over 20 calls. The −2.25 ms therefore belonged to netem's
`paretonormal` table rather than to the instrument, and the instrument's accuracy across a
real NIC and two hosts is better than a tenth of a millisecond, which is tighter than
stage 2 achieved on loopback.

**The `ec2-c7i` pair, playout under jitter.** The jittered sweep raised
`high_late_discard` on 38 of its 40 calls with zero packet loss, so its playout figures are
upper bounds rather than point estimates and the flag has to travel with them. Playout did
not absorb the jitter except at the shortest delay: per programmed delay the ingress and
playout standard deviations were 19.03 against 3.06 ms at 137 ms, then 8.08 against 9.49,
7.37 against 6.58, 9.76 against 10.61, and 9.25 against 7.59. A 40 ms playout target is
simply too shallow for 15 ms of jitter, so frames arrived after their deadline and onset
detection was deferred by whole frames, which adds variance rather than removing it. That
is the behaviour `METHOD.md` §4 specifies, and the ingress-based stage 3 gate is unaffected.
The loss sweep separately raised `high_loss` on 3 calls with measured loss averaging 1.24%
against netem's declared 1%, so both advisory mechanisms have now been exercised on real
data rather than only in simulation.

One instrument defect was found by auditing these files rather than by running them: the
console printed QC flags only on failure, so a sweep could report forty passing calls while
38 carried an advisory recorded in the JSON and nowhere a reader would look. Fixed the same
day, with a regression test.

The reference responder's self-error stayed at +0.00 ms across all 120 remote calls, so the
frame-quantisation defect recorded in `METHOD.md` §6 remains fixed when the responder is a
separate process on a separate machine.

**The `ec2-c7i` pair, 2026-09-03, and two reference-responder defects found in the
captures.** The 137 ms run recovered its declared delay to +0.01 ms, so the instrument has
now recovered two different injected values, 50 and 137 ms, each to within a tenth of a
millisecond. Ingress figures and the differential gate are unaffected by everything that
follows.

Two things in these runs did not match expectation, and because the raw captures were kept
both could be diagnosed locally rather than argued about. First, the 137 ms run with no
jitter at all raised `high_late_discard` on every call. Per-packet transit in the captures
shows why: the responder's first comfort-noise frame goes out on time and the next two go
out 30 and 15 ms late, after which the stream is regular but permanently offset. The
responder paces comfort noise off a 50 ms `recvfrom` timeout, so while no caller packets are
arriving it can only emit a frame every 50 ms and slips 30 ms per frame. The handshake
window before caller media arrives is longer under a larger netem delay, so the accumulated
slip was 17.6 ms at baseline and 44.7 ms at 137 ms, and only the latter exceeds a 40 ms
playout target. The first frame, sent before any slip, becomes the minimum-transit anchor,
and every later frame is then "late" by the slip. Second, re-analysing the jittered captures
at targets from 40 to 150 ms shows late discards falling from 3.2% to zero by 80 ms, which
is the useful half of the finding, while the playout residual standard deviation stays at
9.54 ms at every target. The responder stamps response frames with RTP timestamps that
continue its comfort-noise count, while their emission instant jumps to the programmed
delay, so the timestamps carry a phase error that varies call to call once the handshake
timing is jittered. Playout MRL is derived through the RTP timestamp and inherits that
error; ingress is derived from arrival and does not.

Both defects are in the calibration source and neither is in the instrument, but they mean
these captures cannot demonstrate the playout buffer absorbing jitter over a real path, and
that claim rests on stage 1 alone until the responder is corrected and the jittered sweep
repeated. The corrections are to pace comfort noise on its own deadline regardless of
receive activity, and to derive RTP timestamps from elapsed media time rather than from the
frame count. Both landed in `loopback.py` on 2026-09-04 with regression tests that
reproduce each signature on loopback; the jittered sweep over the pair is still to be
repeated, and until it is the playout rows above stand as recorded.

**The `ec2-c7i` pair, 2026-09-04: stage 3 closed, and a ninth finding.** The corrected
responder was run over the same pair. Both defects are gone on the real path: the 137 ms
sweep raised no advisory at all where the old responder raised late discard on every call,
the comfort-noise gap during the handshake fell from 50.0 to 20.9 ms, the comfort-noise-to-
response transit phase on the same call went from −87 to −0.09 ms, and under jitter the
playout residual spread fell from 9.54 to 2.68 ms while ingress stayed near 11.6. That last
figure is the demonstration the jittered rows above could not provide, and it lands within
a millisecond of the 3.05 ms stage 1 predicted from simulation. Stage 3 is closed.

The same run exposed a defect in the analyser that the old responder had been hiding. The
baseline residual came out at −0.74 ms with sd 1.63, against +0.60 and sd 0.04 on both
earlier baselines, with responder self-error at zero. Per call, every 213 ms call has its
onset at the response packet's first sample and a residual of +0.60; seven of eight 137 ms
calls have the onset 55 to 113 samples inside the final comfort-noise frame and a residual
2 to 4 ms low. Re-deriving t1 from the response packet's arrival gives +0.60 ms, sd 0.02,
across all forty. The cause is that `_detect_onset` refines t1 within its first
above-threshold 5 ms window using instantaneous sample amplitude, the mechanism finding
two removed from t0 and which was never applied to t1. Frame-count stamping had placed
every response on the 5 ms analysis grid, so no noise ever preceded an onset inside its
window; a media clock places the response where it truly began, which for the 137 ms
group was 14 to 20 ms into the final comfort-noise frame, so the first window to clear the
threshold was mostly noise and a noise peak fired the refinement up to a frame early. The
effect is present in stage 1 too, where the current refinement's bias runs from −0.03 ms on
the strict variant to −2.26 on the sensitive, the threshold dependence that noise-chasing
produces, and it contributed the whole of the published −0.40 ms headline bias and the
0.99 ms variant spread on a hard onset, which on a hard onset should be near zero. The
remedy, finding two's sliding RMS mirrored for a start boundary, landed the same day. Stage
1 now reads +0.01 ms bias, p95 0.10 ms, spread 0.02 ms, and every kept stage 3 capture
re-derived under it without an instance:

| sweep, re-derived 2026-09-04 | ingress residual | playout residual sd |
|---|---|---|
| 09-03 baseline, old responder | +0.67 ms, sd 0.07 | 4.04 (stamping artefact, finding eight) |
| 09-04 baseline, corrected responder | **+0.63 ms, sd 0.07** (as run: −0.74, sd 1.63) | **0.10** |
| 09-03 netem 137 ms, old responder | +137.67 ms, sd 0.07 (differential 0.00 ms) | 4.11 |
| 09-04 netem 137 ms, corrected | **+137.58 ms, sd 0.10** (as run: +136.64, sd 0.80; differential −0.05 ms) | **0.13** |
| 09-04 netem 50 ms + 15 ms jitter, corrected | +48.22 ms, sd 9.17 | **2.20** |

Two things follow. The as-run rows above stand as evidence of the analyser that produced
them, and the re-derivation is the demonstration that keeping raw captures is worth the
disk. And the old responder's playout artefact shows even on its jitter-free runs, at 4 ms,
which the as-run figures had never prompted anyone to look at.

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
