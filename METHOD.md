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

## 6. Five findings that changed the method

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

## 7. Validation stages

1. **In-process.** Proves the metric arithmetic, onset detection, and QC gates. No
   network, no GPU. Implemented and passing.
2. **UDP loopback.** Same analyser, real sockets. Proves the live capture path and its
   timestamping. Implemented and passing; found two errors stage 1 structurally could
   not (§6).
3. **Remote host, `netem`-injected delay.** Proves end to end against a known injected
   delay across a real NIC, and across two clocks. Not yet implemented.
4. **SIP caller.** Real calls against a real system. Not yet implemented.

Stage 3 must pass before any figure is measured from a real system. The instrument's
accuracy across a real NIC and a second host is not established by loopback alone.

## 8. Out of scope by design

The harness measures. It does not contain endpointing strategy, barge-in handling, or
model tuning. That boundary is deliberate: it keeps the publishable artifact separable
from the implementation being measured.
