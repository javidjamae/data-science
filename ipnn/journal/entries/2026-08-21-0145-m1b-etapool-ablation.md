# M1b: the pool's plasticity earns nothing

- **Entry:** `2026-08-21-0145-m1b-etapool-ablation`
- **When:** 2026-08-21 01:45–01:55 CDT
- **Who:** Javid + Claude (session)
- **Experiment:** [001 — MNIST living demo](../../experiments/001-mnist-living-demo/design.md), ablation of the M1 gate
- **Code state:** uncommitted at time of writing — hash of next commit: _fill in after committing_
- **Re-run:** `cd ipnn/experiments/001-mnist-living-demo/app && npx vite-node tools/m1b-ablation.ts`

## In plain words

The organism has two sets of connections it can adjust: 3,840 running from its
eye into its "brain" pool, and 480 running from the pool to its three answer
neurons. We froze the first set — 3,840 connections locked at the random
values they were born with, never allowed to learn again — and re-ran the
whole experiment.

It made no difference. 0.964 accuracy with the eye→pool connections learning,
0.960 with them frozen, across five seeds. The gap is smaller than the
variation between seeds.

So the part we have been calling the brain is not learning anything useful.
All the learning that matters is happening in the last 480 connections. The
pool is a fixed random scramble of the image, and the organism is a simple
readout sitting on top of it.

## Objective

Settle the pre-registered question: is the pool contributing, or is the M1
result carried entirely by the output weights?

## Gate (pre-registered)

From [entry 2026-08-16-0248](./2026-08-16-0248-experiment-001-m0-m1-first-build.md)
§Next, and indexed in [experiment-ideas §E](../../experiment-ideas.md):

> **M1b — etaPool=0 ablation:** is the pool contributing anything yet?

No pass/fail threshold was pre-registered, correctly — this is a measurement,
not a gate. The decision rule stated before running: *a difference smaller
than the seed-to-seed spread within either arm counts as "contributes
nothing".*

## Hypotheses

- **H1:** freezing sense→pool weights (`etaPool = 0`) costs measurable
  accuracy — the pool is learning useful features.
- **H2 (the alternative):** it costs nothing — the pool is a fixed random
  projection and learning lives in the 480 pool→output weights.

## Method

- **Arms:** `etaPool = 0.01` (the certified M1 configuration) versus
  `etaPool = 0` (sense→pool weights frozen at their random initialization).
  Nothing else differs; `etaOut = 0.08` in both.
- Identical procedure to `m1-sanity.test.ts`: same seed→schedule formula,
  800 trials, rolling-100 accuracy, tail = last 100 trials.
- **Seeds:** 1–5 (the gate uses 1–3; extended to 5 for a slightly better
  view of the spread).
- **Configuration snapshot:** `senseSize 64, poolSize 160, outputSize 3,
  poolFanIn 24, poolGain 2.0, poolBias −1.0, targetPoolSparsity 0.15,
  inhibitionRate 0.02, outputGain 1.5, epsilonExplore 0.05, silenceBias 0.5,
  urgeRate 0.05, urgeMax 3.0, traceDecay 0.97, etaOut 0.08,
  etaPool {0.01 | 0}, wMax 3.0, consolidation true, consolidationN0 1000`;
  teacher `maxTicks 60, blankTicks 15, spokenWindow 20, spokenThreshold 6,
  schedule 'ignore', rewardMagnitude 1.0, correctionMagnitude 0.2,
  baselineRate 0.05`.
- **New tool:** `tools/m1b-ablation.ts`.

## Results

Verbatim tail accuracies (last 100 of 800 trials):

| seed | baseline `etaPool=0.01` | ablated `etaPool=0` |
|---|---|---|
| 1 | 0.98 | 0.93 |
| 2 | 0.98 | 0.98 |
| 3 | 0.99 | 0.98 |
| 4 | 0.94 | 0.96 |
| 5 | 0.93 | 0.95 |

- **mean tail** — baseline **0.964** (min 0.93, max 0.99); ablated **0.960**
  (min 0.93, max 0.98).
- **difference (baseline − ablated): +0.004.**
- Within-arm spread: 0.06 (baseline), 0.05 (ablated). The between-arm
  difference is roughly an order of magnitude smaller than the within-arm
  spread.
- Learning curves are indistinguishable in shape; both arms cross 0.80 in the
  second 100-trial block on most seeds. Example, seed 3 — baseline
  `0.42 → 0.74 → 0.85 → 0.89 → 0.95 → 0.96 → 0.98 → 0.99`, ablated
  `0.43 → 0.74 → 0.88 → 0.99 → 0.95 → 0.99 → 0.94 → 0.98`.
- Weight norms |w|²: pool 329–337 (baseline) vs 321–325 (ablated, i.e.
  unchanged from initialization); out 451–526 vs 483–557. So pool weights
  *did* move under learning — roughly 3% in norm — and that movement bought
  nothing.

## Analysis

H1 is dead; H2 stands. Freezing 3,840 of the organism's 4,320 learnable
synapses — 89% of them — costs 0.004 accuracy, comfortably inside seed noise.

The precise claim, stated carefully: **the pool's *plasticity* earns nothing.**
This is not the same as "the pool is useless." The random projection itself may
well be load-bearing — 64 binary pixels expanded to 160 sparse stochastic units
may be what makes the three patterns cleanly separable for a linear readout.
M1b does not test that; it holds the projection fixed and removes only its
learning. The follow-up that would settle it is a direct sense→output organism
with no pool at all (**M1c**), registered below.

What the M1 result therefore is, stripped of flattering language: a fixed
random expansion of the input, with a 480-weight policy readout trained by
REINFORCE. That is a real and working reward-only learner, and the M1 gate
stands — but the depth is decorative. It also explains why 001 never hit the
credit-assignment problem the abstract worries about (open-problems §1): with
a single learnable layer feeding the outputs, there is barely any credit to
assign. The variance problem that kills this rule family at scale was never
exercised.

This lands squarely on experiment 002. The case for growing a substrate was
previously aesthetic — "the shape should change." It is now empirical: the
designed hidden structure in 001 is provably not earning its keep, so
"design a better hidden structure by hand" and "let it grow one" are now
competing answers to a demonstrated deficiency rather than a matter of taste.
It also sets the bar 002 must clear honestly — beating a random projection is
the real baseline, not beating chance.

## Prior art & novelty

Nothing novel; this is a standard ablation, and the result is the expected one
for a shallow randomly-projected network. It is, in effect, a rediscovery that
random-feature expansions plus a trained linear readout are strong at toy
scale — the extreme-learning-machine / random-kitchen-sink observation, and
the same reason reservoir computing freezes its reservoir on purpose. Worth
adding to [related-work.md](../../related-work.md) when the C/F tracks touch
it. No claim of any kind is made here.

## Learnings

- **L-010:** In experiment 001, the hidden pool's plasticity contributes
  nothing measurable: freezing 89% of the learnable synapses changes tail
  accuracy by 0.004, an order of magnitude below seed spread. The M1 result is
  carried by a 480-weight readout on a *fixed random projection*. Corollary
  for every future comparison: **the baseline to beat is a random projection,
  not chance.** *Evidence:* this entry, 5 seeds × 2 arms.

## Decisions

1. **Experiment 002 proceeds, with its baseline re-stated.** Its M1 gate is
   unchanged in threshold but changed in meaning: matching 001 now means
   matching a random projection, and that comparison must be made explicitly
   rather than against the 0.33 chance line.
2. **M1c registered** — a no-pool control (sense→output directly) to
   determine whether the random *projection* matters even though its learning
   does not. Cheap, and it completes the ablation properly.
3. **Not changing 001's configuration.** `etaPool` stays 0.01: the demo is a
   published record of the certified gate, and tuning it now would invalidate
   the artifact for no gain. The finding is documented, not patched.
4. Docs updated: `README.md` (experiment table, naming), `experiment-ideas.md`
   ranking, `open-problems.md` §1 status.

## Deviations

Ran 5 seeds rather than the gate's 3 — a strengthening, recorded for
transparency. No other departure.

## Threats to validity

1. **The task may be too easy to discriminate the arms.** Three fixed,
   noiseless, well-separated patterns might be solvable from almost any
   expansion, hiding a pool contribution that a harder task would reveal.
   This weakens the generalization, not the M1-specific conclusion.
2. **`etaPool = 0.01` may simply be too small a learning rate** rather than
   pool learning being useless in principle. A rate sweep would separate
   "does not learn" from "cannot help." Not run.
3. **n = 5 seeds.** Enough to see that the difference sits well inside the
   spread; not enough for a confidence interval, and none is claimed.
4. Tail accuracy is one summary of a curve; a difference in *speed* of
   learning rather than final accuracy would be partly masked, though the
   per-100 curves show no obvious separation either.

## Next

Experiment 002 M0 — extract the `OrganismLike` interface from 001 with all 17
tests passing bit-identically, then build the substrate. Pre-registered gate
for M0: *no behavioral change to experiment 001 whatsoever* — the M1 gate
curves must remain identical to those recorded in entry 2026-08-16-0248.
