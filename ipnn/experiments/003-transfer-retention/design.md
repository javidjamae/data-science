# Experiment 003 — Transfer, Retention and Savings

**Status:** pre-registration. Written before the harness was run.
**Substrate:** experiment 001, unchanged. **Does not depend on 002**, which is
blocked at its own M1.
**Origin:** [L-020](../../journal/LEARNINGS.md) — the project has defined
intelligence as transfer-plus-retention since `vision.md` was written and has
never once tested it.

## 1. The question

[vision.md](../../vision.md) pillar 4: the same core network should learn one
thing, then another, "then return to the first and *still function*."
Success-criterion 4: "you teach it a second task, return to the first, and it
has not forgotten."

Every gate this project has run — 001's M1, 001-M1b, 002's M0 and M1 — has
measured accuracy on one fixed task. The definition was correct and dormant.

> Does experiment 001 transfer anything from one task to a second, does it
> still perform the first afterwards, and if it has forgotten, is the first
> task any easier to relearn than it was to learn?

Those are rungs 5, 6 and 4 of the [ladder of evidence](../../vision.md#how-we-judge-whether-it-is-learning),
and this is the first use of the doctrine adopted on 2026-08-22.

## 2. Method

**Task A** — the three M1 glyphs (vertical bars, horizontal bars, diagonal X).
**Task B** — three new glyphs (plus, ring, diagonal band) on **the same three
output neurons**. Same outputs is the deliberate choice: it is maximum
interference, and it is what the success criterion describes.

Task B is built to be as learnable as task A so that a difference in
trials-to-criterion means transfer rather than difficulty. All three B glyphs
are 28 pixels, so pixel count cannot be the within-task cue; mutual overlap is
0.17–0.27 by intersection-over-union against A's 0.22–0.23; no cross-task pair
exceeds 0.33. `patterns.test.ts` pins all of it.

**One organism's life, four phases:**

1. learn A to criterion → `T_A`
2. learn B to criterion → `T_B`
3. test A **frozen**, learning off → `R_A`
4. relearn A to criterion → `T_A2`

**Control arms.** Without these the numbers have no referent:

| Arm | What it controls for |
|---|---|
| **naive-B** — a fresh organism learning B first | Transfer. `T_B_naive − T_B` is the transfer effect. Without it, "B took 140 trials" means nothing |
| **A→A** — learn A, then run `T_B` further A trials, then test | Retention's **ceiling**. Controls for elapsed time and for drift under continued reward, so a drop after B is attributable to B and not to the clock |
| **naive-A** — a fresh organism learning A | The savings baseline. Relearning must beat learning-from-scratch, not merely be fast |

**Currency: trials-to-criterion** ([H-011](../../journal/HYPOTHESES.md)), not
accuracy — comparable across learners of different speeds, and it does not
punish a slow learner for being slow. Criterion: rolling-100 ≥ 0.85, cap 4,000
trials. Never reaching criterion is reported as "none", never silently as the
cap.

**Seeds** 1–5. **Frozen probe** 150 trials with learning off, so the probe
cannot itself teach.

## 3. Pre-registered gates

- **Transfer.** `T_B` < `T_B_naive` on ≥3 of 5 seeds.
- **Retention.** Mean `R_A` is within **0.10** of the A→A ceiling. (Chance is
  0.333.)
- **Savings.** `T_A2` < `T_A_naive` on ≥3 of 5 seeds.

## 4. Prediction, stated before running

**All three gates fail except savings.**

[L-010](../../journal/LEARNINGS.md) established that 001 is a 480-weight
readout on a *fixed random projection* — 89% of its learnable synapses can be
frozen at no cost. **A single learnable layer has nowhere to put task-general
structure.** So:

- **Transfer: none.** There is no shared representation to reuse; the random
  projection is identical for both tasks and was never learned in the first
  place.
- **Retention: catastrophic.** The same 480 weights encode A and are then
  overwritten by B. Expect `R_A` near chance.
- **Savings: large, and this is the interesting one.** Even with total
  behavioural forgetting, the projection is unchanged and only the readout has
  moved. Relearning should be much faster than learning from scratch — a
  latent trace that behaviour does not express, which is exactly what the
  savings instrument exists to detect.

If that pattern holds it is a clean result: **001 forgets completely and
relearns cheaply**, which is a precise statement about what a one-layer
reward-only learner can and cannot carry across tasks.

## 5. What each outcome would mean

| Result | Reading |
|---|---|
| Transfer fails, retention fails, savings passes | The predicted result. Converts "the shape should change" from an aesthetic motivation for 002 into a demonstrated deficiency: reward-only learning at one layer does not compose across tasks, though it leaves a trace |
| All three fail | Worse than expected — not even a latent trace survives. Would make the random projection, not the readout, the thing to attack |
| Transfer or retention passes | A single layer carries more than L-010 implies, and the ladder's upper rungs are reachable on the substrate we already have. Would materially raise the value of [backlog track H](../../experiment-ideas.md) |

**Every outcome is informative**, which is the property [H-008](../../journal/HYPOTHESES.md)
demands of anything entering the record.

## 6. Threats to validity, anticipated

1. **Task B may simply be harder than task A** despite the matching. `T_B_naive`
   versus `T_A_naive` is the check and is reported.
2. **Same-outputs is the hardest case.** A negative result here does not rule
   out transfer under a gentler mapping (separate output neurons per task), and
   that arm is not run.
3. **Five seeds**, with trials-to-criterion known to vary widely on this
   substrate (001's naive A has ranged 153–287).
4. **Savings is confounded by criterion choice.** A low criterion makes
   relearning trivially fast. 0.85 is well above the 0.80 gate used elsewhere,
   which mitigates but does not remove this.
5. **One substrate.** Nothing here generalises to 002, which cannot currently
   learn task A at all.

## 7. Prior art

Standard continual-learning territory, and no novelty is claimed. Catastrophic
forgetting, backward transfer and forgetting measures are established metrics;
savings is Ebbinghaus (1885). See
[related-work.md](../../related-work.md#measuring-learning-without-assuming-a-machine-added-2026-08-22).
What is unusual here is only that these are the *primary* gates rather than a
follow-up to an accuracy number.
