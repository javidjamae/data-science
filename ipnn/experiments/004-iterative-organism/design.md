# Experiment 004 — The Iterative Organism

**Status:** design only, nothing built. Pre-registration document.
**Backlog origin:** [L-012](../../journal/LEARNINGS.md) — the spec-versus-source
audit of 2026-08-21, which found the mechanism the project is *named for* absent
from every implementation.
**Prior art:** searched before this document was written, unlike
[002](../002-grown-substrate/design.md) — see
[related-work.md](../../related-work.md#the-missing-i--iterative-consideration-searched-2026-08-22)
and §9 below. **No novelty is claimed.**

## 1. The question

`abstract.md` §2–§3 specify an organism whose activity **circulates**: output
feeds back to input, a stimulus is considered over multiple internal passes,
and an answer emerges from that process rather than from a single sweep. That
is the "I" in IPNN.

Neither substrate has it. 001's pool neurons index only the sense and never
read each other. 002's grown edges can in principle form loops, but nothing in
its design or gates ever asked whether they do. Both are single-pass
feedforward, every tick.

> Does letting the organism's own output re-enter its input, and giving it time
> to iterate before committing, change what it can do — and does the answer it
> settles on differ from the answer it would have blurted?

## 2. What is already built and does not need building

Unusually for this project, most of the apparatus exists:

| Piece | Where | Status |
|---|---|---|
| Legal silence — the organism may decline to answer | `organism.ts`, silence logit | built |
| `urge` — a drive to answer that rises during silence | both substrates | built |
| Sustained readout — a free-running answer that can change | `readout.ts` | built (L-009) |
| Manual mode — hold a stimulus and watch it think | `demo-m1/sim.ts` | built |
| `OrganismLike` — a third substrate is a swap, not a fork | `types.ts` | built (002's M0) |

What is missing is one thing: **a path from the output register back into the
pool**, and a policy for how long to let it run.

## 3. The mechanism

**Feedback.** Add output→pool synapses `fbW[k*poolSize + j]`, so an output
firing at tick *t* contributes drive to pool neurons at *t+1*. Same learning
rule as everything else — eligibility × reward-modulated update — so no new
learning machinery. Gain `fbGain` scales the whole path.

**`fbGain = 0` reproduces experiment 001 exactly.** As in 002, the control arm
is the previous experiment one parameter away, and the shared harness makes
that a compiler-enforced fact rather than an intention.

**Iteration.** Today "one tick" is "one pass". Here a stimulus is presented and
the organism is allowed to circulate for up to `maxConsider` ticks before the
teacher reads its answer, with the existing sustained readout deciding when it
has committed — ≥`spokenThreshold` fires in a `spokenWindow`. Silence stays
legal throughout, and `urge` is what eventually forces a commitment.

Nothing here learns *when* to stop. That is deliberate: ACT and PonderNet learn
halting by gradient, and this project has no gradients. Commitment emerges from
urge and the readout threshold, which are already in the code.

## 4. The task problem — and it is the hard part

**Iteration cannot be demonstrated on a task solvable in one pass.** The three
M1 patterns are cleanly separable from a single sweep; 001 reaches 0.98 without
recurrence. Adding feedback to that task can only be neutral or harmful, and a
neutral result would prove nothing.

So 004 needs a task where one pass is *insufficient*. Candidates, in order of
preference:

1. **Degraded stimuli.** Flip a fraction of pixels at random each tick. A single
   pass sees noise; evidence has to accumulate over time. Directly connects the
   readout to the drift-diffusion lineage already in
   [related-work.md](../../related-work.md).
2. **Superimposed patterns.** Show two glyphs at once and ask for the dominant
   one — the binocular-rivalry framing the A-track already cites.
3. **Sequential disambiguation.** A stimulus whose first half is ambiguous
   between two classes and whose second half resolves it. Requires holding a
   hypothesis across ticks, which is exactly what feedback should buy.
4. **Structure versus noise** ([H-016](../../journal/HYPOTHESES.md)) — 80 noise,
   20 structured. Not obviously iteration-dependent, but it is the cheapest new
   task and worth having either way.

**Option 1 is the M2 task.** It has a tunable difficulty knob (flip rate), a
trivially computable ground truth, and a clean prediction: recurrence should
help more as noise rises, and not at all at zero noise.

## 5. Milestones and pre-registered gates

**004-M0 — feedback exists and changes nothing when off.**
Gate: with `fbGain = 0`, 001's M1 curves are **bit-identical** to entry
[2026-08-16-0248](../../journal/entries/2026-08-16-0248-experiment-001-m0-m1-first-build.md),
and the full suite passes. Same gate 002's M0 met; the same discipline applies.

**004-M1 — recurrence does not break what already works.**
Gate: with `fbGain > 0` on the *clean* three-pattern task, tail accuracy stays
≥0.80 on 3 seeds. **This gate can only be lost, not won** — it exists to catch
the failure mode [open-problems §2](../../open-problems.md) has warned about
since the beginning: a recurrent stochastic loop that oscillates or saturates
instead of settling. Standing measurement: activity mean and variance per tick,
and whether either diverges.

**004-M2 — iteration buys something. The critical gate.**
Degraded-stimulus task at flip rates {0, 0.1, 0.2, 0.3}. Arms: `fbGain = 0`
(= 001) versus `fbGain > 0`, 3 seeds each, scored in **trials-to-criterion**
([H-011](../../journal/HYPOTHESES.md)) and in tail accuracy.
**Gate:** the recurrent arm beats the feedforward arm at flip rate ≥0.2 by more
than the seed spread, *and* the two are indistinguishable at flip rate 0.
**If M2 fails, 004 stops** and the finding is "output→input feedback under
reward-only learning does not improve evidence accumulation at toy scale" —
written up with the same rigour as a pass, per journal rule 5.

**004-M3 — changes of mind are observable.**
Using the sustained readout under manual mode: hold a degraded stimulus and
record how often the answer revises before settling, feedforward versus
recurrent. Not a gate — a measurement, and the one L-009 built the instrument
for. Pre-registered expectation: revisions should be *more* frequent and
*earlier* with feedback, and the settled answer more often correct.

## 6. Risks, stated in advance

1. **Instability is the headline risk**, and unlike 002's cold-start fear this
   one has a documented basis: open-problems §2 has flagged the recurrent
   stochastic loop since before any code existed. Mitigation ladder,
   pre-registered in order: cap `fbGain`; keep the homeostat on (it is now
   known to be a strong activity regulator — [L-016](../../journal/LEARNINGS.md));
   add a refractory tick after an output fires; delay feedback by more than one
   tick so the loop cannot ring at the tick frequency.
2. **The eligibility horizon and the loop length are coupled**, exactly as
   §5 of 002's design found for path depth. λ=0.97 gives ~33 ticks; a
   consideration window longer than that means the early passes are
   uncreditable. λ and `maxConsider` must be reported together.
3. **A neutral M2 result is the likely one** if the task is too easy, which is
   why §4 exists and why flip rate 0 is an explicit arm rather than an
   afterthought.
4. **Feedback weights may simply not earn their keep**, as 001's pool weights
   did not ([L-010](../../journal/LEARNINGS.md)). The ablation is obvious and
   pre-registered: freeze `fbW` at initialisation and re-run M2. If frozen
   feedback does as well as learned feedback, the recurrence is a fixed
   perturbation and not a learned one — and that must be reported.

## 7. What this experiment is not

Not an attempt to beat 001 on the clean task. Not adaptive computation time —
nothing learns when to halt. Not a claim about consciousness, deliberation or
reasoning; it is a claim about whether circulating activity improves evidence
accumulation on a degraded input, measured in trials-to-criterion.

## 8. Relationship to the rest of the project

- **001 is the control arm**, again, at `fbGain = 0`.
- **Does not depend on 002.** 002 is blocked at its own M1 and this does not
  wait for it.
- **Ordering.** [Experiment 003](../../journal/entries/2026-08-22-1030-what-counts-as-learning.md)
  (transfer and retention) is still first: it tests the project's own
  definition of intelligence, it is cheaper, and it de-risks
  [backlog track H](../../experiment-ideas.md). 004 is second, and it closes
  the older and more embarrassing gap.

## 9. Prior art and novelty

Searched **before** writing this, which is the correction 002 earned the hard
way. Summary — the components are all occupied:

- **Reward-only learning in recurrent nets:** Miconi (eLife 2017) trains
  recurrent networks with reward-modulated Hebbian updates on delayed phasic
  reward alone. Essentially this learning rule, in a recurrent net, in 2017.
  **This de-risks 004 rather than deflating it:** the thing is known to work,
  so a failure here is ours.
- **Deciding how long to think:** ACT, PonderNet, deep equilibrium models.
- **Revising a decision mid-flight:** *Changes of Mind in an Attractor Network
  of Decision-Making*.

**Smallest honest claim:** four searches did not turn up the specific triple —
reward-only local learning **+** output→input iteration **+** the organism
deciding when to commit under a *rising urge with legal silence*. That is a
weak claim about an unclaimed combination of well-covered parts, marked
*unverified against literature*, and **nothing in this document depends on it
being true.**
