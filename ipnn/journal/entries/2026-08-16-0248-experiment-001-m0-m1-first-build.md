# Experiment 001 — M0 scaffold + M1 sanity gate (first build)

- **Entry:** `2026-08-16-0248-experiment-001-m0-m1-first-build`
- **When:** 2026-08-16 02:48–02:58 CDT (design phase same night, 02:25–02:48)
- **Who:** Javid + Claude (session)
- **Experiment:** [001 — MNIST living demo](../../experiments/001-mnist-living-demo/design.md)
- **Code state:** uncommitted at time of entry (app scaffold + engine are new files; commit hash to be added at first commit)
- **Re-run:** `cd ipnn/experiments/001-mnist-living-demo/app && npm install && npm test`

## In plain words

We built a tiny artificial "brain" from scratch — a few thousand simulated
brain cells and their connections, written in plain code with no AI
libraries. Then we tried to teach it the way you'd teach a toddler or a pet:
we showed it three simple pictures (vertical stripes, horizontal bars, an X)
and, whenever it "called out" the right name, we gave it praise — a single
"good job" signal. Wrong answers were simply ignored. It was never shown an
answer key and never told the rules; the occasional praise was the only
teaching it ever got.

The first attempt failed in a very human way. The brain latched onto one
answer and started shouting it at everything — and the more confident it
became, the *less able it was to hear feedback at all*. Its certainty had
sealed it off from learning. It answered "the second one!" to every picture,
forever, and being ignored couldn't snap it out of it.

The fix was to build in two character traits: humility (its confidence is
never allowed to reach 100%) and curiosity (about 5% of the time it tries
something other than its favorite answer). With just those two changes, it
went from pure guessing (33% right) to 98% right within a few hundred
showings — learning entirely from praise, while we watched.

Then the real test of the "living model" idea: we stopped all praise and
feedback completely. It kept getting 97% right. The skill is genuinely in
it now — it doesn't need ongoing rewards to keep performing, just like you
don't forget how to read when people stop complimenting your reading.

The biggest lesson: **confidence and learning pull against each other.**
The same mechanism that makes a memory stable is the one that makes a
mistake permanent. A little built-in humility and curiosity aren't nice-to-
haves — they're what keep the system teachable. That tension will follow
this project everywhere.

## Objective

Stand up the from-scratch engine (M0) and clear the M1 gate: demonstrate
that pure reward-modulated local learning — no backprop, no labels, no ML
libraries — can learn to discriminate 3 patterns. This is the go/no-go for
the entire experiment: if the learning rule can't do 3 trivial patterns,
nothing downstream matters.

## Gate (pre-registered)

From design.md §6, written before any code existed:

> **M1 — Sanity learning, no UI. The critical gate.** Headless: 3 fixed 8×8
> patterns → 3 outputs, reward-only teacher. Success: sustained above-chance
> accuracy (>80% over a rolling window) from pure reward-modulated learning,
> reproducible across seeds. **If this fails, everything stops until the
> learning rule works.**

Operationalized as: accuracy ≥ 0.80 over the last 100 of 800 trials, on each
of seeds {1, 2, 3} (chance = 0.33).

## Hypotheses

- **H1:** The three-factor rule (score-function eligibility × broadcast
  advantage) reaches the gate within 800 trials.
- **H2:** Baseline subtraction alone makes the 'ignore' schedule work: wrong
  answers need no explicit punishment, because advantage = 0 − baseline goes
  negative once the organism has tasted success.
- **H3** (added mid-iteration, before the retention run): learned behavior
  persists when reward stops entirely.

## Method

**Built (M0):** Vite + TypeScript + Vitest scaffold with zero runtime
dependencies (`app/`). Headless engine (`app/src/engine/`): `Organism`
(stochastic binary neurons, per-synapse eligibility traces and evidence
counts, urge, reward broadcast), `AutoTeacher` (presents stimuli, judges
"spoken" outputs from winner counts in a sliding window, delivers reward,
owns the learning toggle), `mulberry32` seeded RNG throughout — `Math.random`
is banned from the engine. The organism never sees labels or trial
boundaries; only the reward scalar.

**Task:** three 8×8 binary glyphs with similar pixel counts (vertical bars,
horizontal bars, X), balanced-shuffled presentation, 800 trials per seed.

**Learning rule as implemented:**

- per synapse, per tick: `e ← λ·e + (fired − p)·pre` (REINFORCE score
  function, not the design doc's plain `pre·post`)
- on trial end: `Δw = η·(R − baseline)·e / (1 + n/n0)`; evidence `n`
  accumulates on rewarded synapses (consolidation)

**Configuration snapshot (final, passing):** senseSize 64 · poolSize 160 ·
outputSize 3 · poolFanIn 24 · poolGain 2.0 · poolBias −1.0 ·
targetPoolSparsity 0.15 · inhibitionRate 0.02 · outputGain 1.5 ·
**epsilonExplore 0.05** · silenceBias 0.5 · urgeRate 0.05 · urgeMax 3.0 ·
traceDecay 0.97 · etaOut 0.08 · etaPool 0.01 · wMax 3.0 · consolidation on ·
consolidationN0 1000 · **output drive normalized by active-pool count**.
Teacher: maxTicks 60 · blankTicks 15 · spokenWindow 20 · spokenThreshold 6 ·
schedule 'ignore' · reward 1.0 · baselineRate 0.05. Seeds 1–3 (gate),
42 (retention).

**Attempt 1 differed in three ways:** output drive was an un-normalized sum
over active pool neurons; no ε-exploration; silenceBias 1.0,
spokenThreshold 8.

## Results

**Attempt 1 (02:53 CDT): FAILED.** All seeds pinned at exactly chance:

```
seed 1: last-100 accuracy 0.33   seed 2: 0.33   seed 3: 0.33
```

Instrumented diagnostic run (seed 1, 300 trials, every-50 stats):

```
t=50  acc=0.40 spoken=[9,33,5]  silent=3 meanLat=11.1 baseline=0.358 outP=[0.01,0.83,0.00] max|outW|=0.582
t=100 acc=0.32 spoken=[0,50,0]  silent=0 meanLat=8.7  baseline=0.319 outP=[0.13,0.25,0.06]
t=150 acc=0.34 spoken=[0,50,0]  silent=0 meanLat=8.3  baseline=0.332 outP=[0.00,0.97,0.00]
t=200 acc=0.34 spoken=[0,50,0]  silent=0 meanLat=9.0  baseline=0.348 outP=[0.00,0.93,0.00]
t=250 acc=0.32 spoken=[0,49,0]  silent=1 meanLat=9.1  baseline=0.318 outP=[0.03,0.62,0.05]
t=300 acc=0.34 spoken=[0,46,0]  silent=4 meanLat=13.0 baseline=0.336 outP=[0.09,0.47,0.08]
```

From trial ~100 on, the organism answered "1" to every stimulus.

**Attempt 2 (02:56 CDT, after fix): PASSED.** Rolling accuracy per 100
trials:

```
seed 1: 0.53 → 0.80 → 0.91 → 0.98 → 0.94 → 0.95 → 0.97 → 0.98
seed 2: 0.54 → 0.80 → 0.87 → 0.94 → 0.96 → 0.95 → 0.97 → 0.98
seed 3: 0.42 → 0.74 → 0.85 → 0.89 → 0.95 → 0.96 → 0.98 → 0.99
```

Spoken answers stayed balanced across the three outputs throughout (e.g.
seed-1 diagnostic: [16,18,16] at t=300). Baseline converged to ~0.96.

**Retention run (seed 42):** 500 rewarded trials, then learning switched
off; 100 further unrewarded trials:

```
frozen accuracy over 100 unrewarded trials: 0.97
```

All 4 tests pass; full suite runs in ~1.9 s.

## Analysis

**The failure was answer collapse via softmax saturation.** The evidence
chain: exactly-chance accuracy with a *balanced* stimulus stream +
spoken=[0,50,0] + outP for output 1 at 0.93–0.97 + short latency (~9 ticks,
i.e. answering instantly every trial). Output drive was a sum over ~20
active pool neurons, so logits reached ±15+; the softmax saturated; and
because the learning signal is `(fired − p)`, a neuron at p ≈ 1 generates
≈ 0 eligibility. Punishment requires eligibility too — so the wrong dominant
answer was *unpunishable*. A self-locking failure: confidence destroyed the
very signal that could reduce confidence.

**The fix followed from the diagnosis, not from tuning:** (a) normalize
output drive by active-presynaptic count so logits stay O(weight) and the
softmax stays soft; (b) mix ε = 0.05 uniform exploration into output
sampling, with the mixed probability used in the eligibility so it matches
the actual policy. Retunes (silenceBias, spokenThreshold) only adapted trial
mechanics to the softer early dynamics.

**Hypotheses:** H1 supported (after the fix; the rule as first implemented
failed for a mechanism-level, not hyperparameter-level, reason). H2
supported — the 'ignore' schedule never needed explicit punishment; baseline
rose with competence and wrong answers became self-discouraging. H3
supported at toy scale.

**Surprise worth flagging:** the collapse mechanism (high confidence → zero
plasticity) is *identical* to the consolidation mechanism we deliberately
built in via evidence counts. They are one phenomenon with different
valence.

## Learnings

Added to [LEARNINGS.md](../LEARNINGS.md):

- **L-001:** Three-factor reward learning suffices for 3-pattern
  discrimination at toy scale. *Evidence:* Attempt 2 curves, 3 seeds.
- **L-002:** Softmax saturation kills score-function learning; normalize
  drives by active count. *Evidence:* Attempt 1 diagnostics + fix outcome.
- **L-003:** Persistent ε-exploration is load-bearing for recovery from
  dominance. *Evidence:* part of the minimal fix; collapse without it.
- **L-004:** Confidence–plasticity tension: consolidation and frozen-wrong
  are the same mechanism. *Evidence:* Analysis above.
- **L-005:** Frozen retention holds at toy scale (0.97 over 100 unrewarded
  trials). *Evidence:* retention run.
- **L-006:** Exactly-chance accuracy across seeds signals deterministic
  collapse, not noise. *Evidence:* Attempt 1 signature vs diagnosis.
- **L-007:** Gate-first + instrument-then-diagnose caught a paradigm-level
  bug in one cycle. *Evidence:* this entry's timeline (10 minutes from fail
  to pass).

## Decisions

1. **Rule refinements adopted as the standard engine** (see Deviations) —
   design.md §3 remains the conceptual spec; the engine is the concrete one.
2. **'ignore' stays the default teacher schedule** (H2). The correction
   schedule remains an M4 ablation, not a necessity.
3. **Journal established as the notebook of record**; the earlier
   per-experiment `log.md` convention is retired and its content migrated
   into this entry (docs-rule: claims live in one place).
4. **open-problems.md §1 (credit assignment)** updated with the M1 status.

## Deviations

From design.md §3, all adopted deliberately during the build:

1. **Score-function eligibility** `(fired − p)·pre` instead of plain
   `pre·post` — signed, variance-reduced, provably gradient-following.
2. **Softmax winner-take-all output register with an explicit silence
   option** instead of independent Bernoulli outputs + lateral inhibition —
   cleaner semantics, exact per-option probabilities, and the urge plugs in
   naturally as a bias against silence.
3. **Homeostatic global inhibition** instead of hard k-WTA in the pool —
   Bernoulli firing keeps a well-defined per-neuron p, which the score
   function requires.
4. Retunes during the fix: silenceBias 1.0 → 0.5, spokenThreshold 8 → 6.

## Threats to validity

1. **The pool may be doing nothing.** The 3 patterns are near-orthogonal and
   linearly separable from raw pixels; frozen random features could explain
   all learning. Sense→pool plasticity (etaPool 0.01) is untested — an
   etaPool=0 ablation is needed before crediting the full architecture.
2. **Tuned on the test.** The fix was developed against seeds 1–3 and the
   same 3 patterns that then "passed." Seed 42 (retention) was fresh, but
   the gate seeds were part of the tuning loop. MNIST classes in M2 are the
   real out-of-sample test.
3. **ε puts a floor on error.** 5% exploration is harmless at 3 classes;
   at 10 classes it may slow spoken decisions or cap accuracy.
4. **Consolidation is effectively untested.** n0 = 1000 barely engages in
   800 trials; L-004's tension is anticipated, not yet observed.
5. **Retention test is narrow:** same stimulus distribution, one seed, no
   perturbation, only 100 trials — says nothing about drift over hours
   (that's M4).

## Next

**M2 — minimal living demo.** Pre-registered gate (design.md §6): digit
classes {0, 1, 2} from real MNIST on the 64×64 sense, auto-teacher, live
activity display, speed control, learning toggle in the UI. Success:
*watchably climbing accuracy in ≤ a few thousand stimuli; accuracy persists
with learning off.*

**Candidate quick pre-step (M1b):** etaPool = 0 ablation on the M1 task to
answer threat #1 cheaply — does pool plasticity contribute anything yet?
