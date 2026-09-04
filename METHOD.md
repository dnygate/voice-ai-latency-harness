# Measuring mouth-to-ear response latency in SIP/RTP voice AI

Method specification, v0.1. This document defines the measured quantities and states
the instrument's validated accuracy. It is the normative reference: where code and
document disagree, the document is the specification and the code is the bug.

## 1. The quantity under measurement

The voice AI market quotes **model-side first-audio latency**: the interval from a
request reaching a speech synthesis endpoint to the first audio byte returned. What a
caller experiences is **mouth-to-ear response latency (MRL)**: the interval from the
end of their own speech to the arrival of the system's first response audio.

MRL is the sum of, in order:

| Term | Typically published? |
|---|---|
| de-jitter buffer depth on the inbound leg | no |
| voice activity detection and endpointing decision | no |
| ASR finalisation after endpoint | no |
| orchestration, retrieval, tool calls | no |
| LLM time-to-first-token | sometimes |
| TTS time-to-first-audio | yes |
| encode, packetisation, media relay | no |
| de-jitter buffer depth on the outbound leg | no |

Endpointing is frequently the largest single term and is absent from published
figures entirely. A comparison of published first-audio numbers therefore compares
one or two terms of an eight-term sum, and does not predict which system feels
faster on a call.

## 2. Definitions

**t0** — the instant the final speech sample of the caller's utterance is transmitted.

Located by annotating the clean prompt file offline to sample precision, then mapping
that sample onto the transmitted RTP packet that carried it, via RTP timestamps, with
within-frame interpolation at the sample rate.

t0 is deliberately *not* derived from live VAD. Live VAD has its own decision lag, and
that lag is one of the quantities under measurement; using it to define t0 would hide
the term the method exists to expose.

**t1** — the instant the first sample of the system's response reaches the caller.

Located by reassembling the received stream in RTP-timestamp order, detecting response
onset, then mapping the onset sample back through the packet that carried it.

**MRL = t1 − t0**, both taken from `CLOCK_MONOTONIC` on the measuring host. Because
both timestamps come from one clock on one machine, the headline metric requires no
inter-host clock synchronisation. `CLOCK_REALTIME` is recorded in parallel solely to
join captures to server-side traces for the component decomposition; any skew there
affects the decomposition and never the headline figure.

MRL is **signed**. A negative value means the system emitted audio before the caller
stopped speaking — aggressive endpointing, or a backchannel — which is a real
behaviour to be reported, not an error to be clamped.

### 2.1 Two MRLs, both required

**Ingress MRL** takes t1 at packet arrival. It isolates what the system contributed,
independent of the caller's buffering policy. Its variance under network jitter equals
the channel's jitter: that is the instrument reporting the channel faithfully, not
imprecision.

**Playout MRL** takes t1 at the instant the onset sample would leave a de-jitter buffer
of stated target depth. This is what the caller actually waits, and it is the quantity
to compare against conversational turn-taking norms.

Report both, always, with the playout target depth stated. Quoting ingress alone
understates what callers experience. Quoting playout alone confounds the system under
test with the client's buffering policy.

The de-jitter anchor tracks minimum observed transit delay over an initial window,
which is what adaptive buffers converge toward, rather than anchoring on the first
packet's arrival.

### 2.2 Onset is not a single number

"When did audio start" has no unique answer, because real TTS ramps in rather than
starting at full level. Onset is therefore computed under three named variants:

| variant | above noise floor | absolute floor | sustained |
|---|---|---|---|
| sensitive | 6 dB | −55 dBov | 10 ms |
| headline | 10 dB | −50 dBov | 20 ms |
| strict | 12 dB | −45 dBov | 30 ms |

`headline` is the reported figure. **The spread across all three is the measurement's
onset-definition uncertainty and must be published alongside any headline number.** On
a hard onset the spread is under 1 ms; on a 150 ms TTS ramp it approaches 10 ms.

Levels are in dBov, referenced to full-scale RMS 32768.

### 2.3 The greeting is not a response

Nearly every deployed conversational system speaks first, usually within a few hundred
milliseconds of the call being answered. That audio answers nothing, because the caller has
not yet said anything for it to answer.

Response-onset detection must therefore begin after the greeting has finished, and the
capture has to record where that was. Detection over the whole received stream finds the
greeting rather than the response, and because the greeting precedes t0 the resulting MRL is
large and negative. A signed metric has no natural floor to catch that, and §2 explicitly
licenses negative values as real behaviour, so the wrong number looks like a legitimate one.
On a synthetic capture with an 800 ms greeting beginning at 200 ms and a true MRL of 900 ms,
the measured value was −2085 ms, an error of −2985 ms, and the call passed quality control.

The greeting also contaminates the noise floor that §2.2 needs in order to set onset
thresholds. That floor is estimated from received audio arriving before t0, on the
assumption that the system is silent while the caller is still speaking, and a greeting
inside that window raises the estimate. On the same capture it moved by 1.35 dB, which the
tenth-percentile estimator limited without preventing.

`greeting_end_sample` is therefore carried in the capture, in the received stream's own
sample domain, and both onset detection and noise-floor estimation begin after it. A system
with no greeting records zero, and nothing about the measurement changes.

**Time to greeting** is a separate quantity worth reporting wherever the call flow permits
it, being the interval from the call being answered to the onset of the system's first
audio. It shares no terms with MRL, since no caller utterance is involved, and it describes
something callers experience directly. A system that holds the line silent for three seconds
after answering is unpleasant to use whatever its MRL turns out to be, and nobody publishes
that figure either.

The harness must not begin transmitting while the greeting is still playing. A system with
barge-in detection stops speaking when it hears the caller, which truncates the greeting and
alters the very interaction being measured.

### 2.4 Response continuity, and filler audio

A system that emits an earcon, a breath or a filled pause while its model is still
generating delivers audio quickly and information slowly. Under §2 alone it records a low
MRL and so compares favourably with a system that stays quiet and then answers, which
rewards the wrong behaviour and can be gamed in an afternoon by adding a beep.

Resolving this by identifying which audio carries meaning would require recognising content,
and a metric containing a judgement about meaning cannot be re-derived by a reviewer from a
published capture. The discriminator is therefore structural. Filler is followed by silence
before the substantive response begins, and continuous speech is not, so the gap is
measurable in the same sample domain the onset detector already works in.

Within a window of 2000 ms after t1, the longest interval below the onset threshold is
measured, excluding intervals attributable to lost or late-discarded frames, because a
degraded channel produces gaps that resemble deliberate pauses. Where that interval exceeds
150 ms the response is discontiguous, the flag `discontiguous_response` is raised, and a
second onset is reported at the start of the final contiguous segment.

Both onsets are then published together. MRL keeps its existing definition, so no figure
already measured changes, and a reader can see whether the two differ and by how much.

The 2000 ms window and the 150 ms gap are defaults fixed in this specification rather than
free parameters, for the reason given in §2.2: figures whose parameters differ cannot be
compared even when each is internally correct. They must be stated alongside any figure
derived under them, and revising them is a change to this document rather than to a
configuration file.

## 3. Deriving metrics offline

Captures store raw payloads and raw timestamps only. Nothing derived is ever written
into a capture. Consequences:

- A revised onset definition, or a reviewer's alternative definition, can be applied to
  existing data without re-running a test that costs GPU hours.
- The uncertainty analysis in §2.2 is possible at all.
- Prompt files are hashed into their annotations; a changed prompt fails loudly rather
  than silently invalidating a comparison.

## 4. Quality control

Two classes of flag. **Blocking** flags invalidate the measurement itself: no prompt
speech-end, no received packets, onset not found, onset inside the first frame. Also
blocking: transmit pacing deviation above 5 ms, because a harness that cannot pace its
own transmission has an unreliable t0 and therefore an unreliable everything.

**Advisory** flags mark a degraded channel — loss above 2%, late-discard above 1% of
frames. A lossy call is still a valid measurement of a lossy call. But when the onset
frame itself is lost, detection is deferred by whole frames and the resulting MRL is an
upper bound rather than a point estimate, so the flag must travel with the number.

Every published distribution reports `n_dropped` alongside it. A run that silently
discards a third of its calls is not comparable to one that discards none, however good
the surviving percentiles look.

## 5. Validated instrument accuracy

Stage 1 validation replies to each prompt at a programmable delay, so ground truth is
known to the nanosecond. Under each channel condition the expected offset from truth is
predictable analytically — ingress: base transit + mean jitter excess; playout: base
transit + buffer target — so the test is not "is the answer close to truth" but "does
the instrument reproduce the offset physics requires". Residuals below are *after*
subtracting that expected offset. 1200 calls, 10 delays from 50 to 1500 ms, 8 seeds,
`headline` variant, G.711 μ-law, 20 ms frames, 40 ms playout target.

| condition | ingress bias | ingress sd | playout bias | playout sd |
|---|---|---|---|---|
| clean, μ-law | −0.40 ms | 0.70 ms | −0.40 ms | 0.70 ms |
| clean, A-law | −0.43 ms | 0.72 ms | −0.43 ms | 0.72 ms |
| clean, L16 control | −0.35 ms | 0.67 ms | −0.35 ms | 0.67 ms |
| 10 ms frames | −1.26 ms | 1.17 ms | −1.26 ms | 1.17 ms |
| 30 ms frames | −1.56 ms | 1.45 ms | −1.56 ms | 1.45 ms |
| jitter, mean 10 ms | −0.86 ms | 6.98 ms | −0.22 ms | 1.86 ms |
| jitter, mean 30 ms | +0.35 ms | 19.97 ms | +2.27 ms | 3.05 ms |
| transit +50 ms | −0.40 ms | 0.70 ms | −0.40 ms | 0.70 ms |
| transit +50, jitter 20 | −1.52 ms | 12.83 ms | +2.75 ms | 2.16 ms |
| high channel noise | −0.82 ms | 0.84 ms | −0.82 ms | 0.84 ms |

**Headline instrument accuracy: bias −0.40 ms, p95 |error| 2.38 ms on a clean channel
with a 20 ms frame grid.** t0 annotation is exact on the reference signal, with zero
variance. This licenses claims about differences of tens of milliseconds between
systems; it does not license claims about differences of two.

Behaviours confirmed rather than merely tolerated:

- Ingress sd tracks predicted channel jitter sd; the playout buffer absorbs it,
  cutting sd from 19.97 ms to 3.05 ms at 30 ms mean jitter.
- A deliberately mispaced sender (6 ms transmit jitter) is rejected on all 80 calls,
  confirming the pacing gate works rather than assuming it.
- 10% loss raises the advisory flags on ≥18 of 20 calls.
- Onset-definition spread: 0.99 ms hard onset, 4.69 ms at a 60 ms TTS ramp, 9.84 ms at
  150 ms — the dominant uncertainty term on real systems.

### 5.1 Stage 2: live sockets

Real UDP on loopback, identical analyser. 15 calls, delays 137/213/353/457/806 ms.

| quantity | value |
|---|---|
| residual bias | **+0.35 ms** |
| residual sd | 0.06 ms |
| p95 \|error\| | 0.42 ms |
| reference responder self-error | < 0.01 ms |
| worst transmit pacing deviation | 4.91 ms (QC blocks above 5 ms) |

Pacing is a property of the host, not the code. `harness.preflight` measures it in ten
seconds and should be run on any machine before a figure from that machine is trusted.
Stage 1 is insensitive to host timing and is always runnable.

Per-machine outcomes, including refusals, are recorded in `VALIDATION.md` at the
repository root with the raw JSON committed alongside. A power-managed laptop refused
by QC is the gate working as designed, and preflight predicts that outcome before any
call is placed, which is the answer to whether stage 2 timing on such a host is
representative: an unfit host is detected and excluded rather than averaged in.

## 6. Eight findings that changed the method

Recorded because each is a mistake a reader may be making.

**A fixed VAD hangover biases every latency figure low.** Extending the speech-end
boundary by a constant to catch trailing fricatives pushes t0 later, and since
MRL = t1 − t0, that shortens every reported latency by that constant. A 40 ms hangover
made the instrument 47.5 ms optimistic. Replaced with decay-following hysteresis: from
the last supra-threshold frame the boundary follows the energy decay while it stays
above a secondary threshold, so genuine low-energy final phonemes are captured while a
hard cut yields no extension. Bias fell to zero.

**Instantaneous amplitude thresholds chase channel noise.** Sub-frame refinement on
`|x|` crossed threshold on isolated noise peaks several percent of the time, dragging
the boundary ~6 ms late. A 2 ms sliding RMS rejects them.

**A reference responder can quantise its own emission to its receive loop.** The
calibration responder checked its emit deadline once per received frame, so emission was
quantised to the 20 ms frame period, adding a uniform 0-20 ms error on top of the
programmed delay. It was invisible for a long time because every delay under test was a
multiple of 20 ms, which puts the deadline on a frame boundary. **Never validate a
frame-based measurement only at delays commensurate with the frame time.** The default
sweep now uses 137/213/353/457/806 ms for this reason. Responder self-error fell from
~10 ms to under 10 microseconds.

**Sign errors in frame-offset arithmetic look like plausible latencies.** t0 is the
frame's send time *plus* the target sample's offset within the frame, so the calibrated
emission instant is arrival + delay + offset. The wrong sign produced a residual of
-2x offset, i.e. -39.5 ms with sd 0.03 ms. It was caught on the first stage 2 run and
could not have been caught by a synthetic capture, because the synthetic generator and
the analyser shared the same convention and were therefore consistently wrong together.
Tight spread with large bias is the signature: check arithmetic, not the host.

**Modelling network jitter as zero-mean Gaussian is wrong in a way that matters.** A
packet can be queued and delivered late; it cannot arrive before its minimum transit
time. Two-sided jitter let a minimum-tracking de-jitter buffer anchor earlier than
physically possible, appearing as a −27 ms latency bias that looked like a bug in the
playout model. Jitter is now one-sided gamma (shape 2). Sender pacing slop stays
symmetric, because a timer-driven sender genuinely can fire early or late — a different
physical mechanism, correctly modelled differently.

**A greeting is a response to nothing, and a signed metric cannot notice.** Onset
detection scanned the received stream from its first sample, which is correct against a
reference responder that stays quiet until it replies and wrong against every deployed
system, because deployed systems say hello. The greeting arrives before t0, gets detected
as the response, and yields a large negative MRL. A capture carrying an 800 ms greeting
and a true MRL of 900 ms measured −2085 ms, an error of −2985 ms, and passed quality
control on the way through. §2 licenses negative values as genuine behaviour, so there was
no floor for the wrong answer to trip over. The same greeting sat inside the window used to
estimate the noise floor and moved it 1.35 dB, shifting every onset threshold derived from
it. Detection and floor estimation now both begin at `greeting_end_sample`, recorded in the
capture. This one could not have been caught by any test that existed, because the
synthetic generator had no greeting to produce; it was found by asking what a real call
looks like, and confirmed by teaching the generator to make one.

**A reference responder has to be an honest RTP sender.** Two defects in the calibration
source, neither in the instrument, both found by re-analysing kept captures rather than by
any test. The responder paced its comfort noise off a 50 ms receive timeout, so while no
caller media was arriving it emitted one frame per 50 ms and slipped 30 ms each; the
handshake window before media arrives is such a period and it grows with injected delay,
so the slip was 17.6 ms at baseline and 44.7 ms under 137 ms of netem, and because the
first frame anchored the playout minimum every later frame read as late, raising
`high_late_discard` on 20 of 20 calls that carried no jitter whatever. Separately, response
frames were stamped with the next comfort-noise grid slot while being emitted at t0 + delay,
a phase error of up to one frame that varied call to call once handshake timing was
jittered. Ingress never saw it, being derived from arrival, and playout inherited it
entirely: 9.54 ms of spread at every buffer target from 40 to 150 ms, where stage 1 had
predicted a collapse. Comfort noise is now paced on its own deadline, and every RTP
timestamp is derived from scheduled emission time on a media clock, as a real sender's
sample clock would produce. Both were invisible to stage 1, whose synthetic stream is
constructed correctly by definition, and to stage 2 on loopback, where the handshake window
is too short to slip. They needed a real path and captures that were kept.

**An advisory flag nobody can see is not an advisory.** The stage 3 console printed QC
flags only when a call failed, so a jittered sweep in which 38 of 40 calls raised
`high_late_discard` reported forty passing calls and no caveat at all. Every number in
that run was correct. What was missing was the qualification that §4 attaches to them,
namely that a call whose onset frame was deferred yields an upper bound rather than a
point estimate, and a reader working from the console rather than from the committed JSON
would have quoted those playout figures as though they were point estimates. This was
caught by auditing the artifact rather than by running anything, which is the argument for
storing flags in the capture and for publishing the JSON alongside every claim. Flags now
appear on passing calls and are summarised with counts at the end of a run.

## 7. Validation stages

1. **In-process.** Proves the metric arithmetic, onset detection, and QC gates. No
   network, no GPU. Implemented and passing.
2. **UDP loopback.** Same analyser, real sockets. Proves the live capture path and its
   timestamping. Implemented and passing; found two errors stage 1 structurally could
   not (§6).
3. **Remote host, `netem`-injected delay.** Proves end to end against a known injected
   delay across a real NIC, and across two clocks. Implemented and passing. The gate is
   differential against a no-impairment baseline over the same host pair, which cancels
   the inter-host round trip and the responder's own emission error, so no independent
   measurement of the path is required. First passed 2026-08-31: a declared 50 ms
   recovered to within 2.25 ms, inside the instrument's own precision. See
   `VALIDATION.md`.
4. **SIP caller.** Real calls against a real system. Not yet implemented.

Stage 3 must pass before any figure is measured from a real system. The instrument's
accuracy across a real NIC and a second host is not established by loopback alone.

## 8. Out of scope by design

The harness measures. It does not contain endpointing strategy, barge-in handling, or
model tuning. That boundary is deliberate: it keeps the publishable artifact separable
from the implementation being measured.
