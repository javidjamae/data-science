# Open Problems

The honest list. Each entry says what the problem is, IPNN's current working
answer, and how we'll find out if the answer is wrong. Update as experiments
land — this file is a living document.

## 1. Credit assignment (the big one)

**Problem.** When reward arrives, which of the thousands of synapses that
recently fired deserve it? Backprop answers this with gradients; IPNN gives
gradients up.

**Working answer.** Three-factor rule: per-synapse eligibility traces ×
global broadcast reward. Equivalent to REINFORCE, which provably follows the
reward gradient *in expectation* — but with variance that grows with the
number of stochastic units, so learning may be impractically slow beyond toy
scale.

**How we'll know.** Experiment 001, Milestone 1 gate: if the rule can't
reliably learn 3 trivially distinct patterns, iterate on the rule before
building anything else. Variance mitigations to try in order: reward baseline
subtraction, sparser activity (k-winner-take-all), curriculum (fewer classes
first), population/rate readouts instead of single-neuron readouts.

**Status 2026-08-16:** M1 gate CLEARED — 98–99% on 3 patterns across seeds,
reward-only (journal
[2026-08-16-0248](./journal/entries/2026-08-16-0248-experiment-001-m0-m1-first-build.md),
L-001). One mechanism-level fix was required: softmax saturation had made
the dominant answer unpunishable (L-002, L-003). Variance concerns remain
untested beyond toy scale.

## 2. Stability of the recurrent stochastic loop

**Problem.** Output-to-input feedback with stochastic units may oscillate or
saturate instead of settling. Hopfield/Boltzmann networks guaranteed settling
with an energy function; IPNN currently has no equivalent.

**Working answer.** Start with feedforward core + output-feedback only (no
dense pool recurrence), keep activity sparse via inhibition/k-WTA, and add
recurrence incrementally while watching activity statistics.

**How we'll know.** Telemetry: firing-rate time series and output-register
dwell times. Runaway rates or flip-flopping outputs = instability.

## 3. Catastrophic forgetting

**Problem.** Train sense/task A, then B, return to A — standard networks have
overwritten A. The multi-sense vision dies here if unsolved.

**Working answer.** Beta-confidence consolidation: synapses with accumulated
rewarded evidence become low-plasticity and resist overwriting (local,
evidence-count cousin of EWC).

**How we'll know.** Experiment 001, Milestone 4: train digits 0–4, then only
5–9, then re-test 0–4. Compare retention with confidence-consolidation on vs
off.

## 4. Reward sparsity and the cold start

**Problem.** Early on, the network is random; correct outputs (and therefore
rewards) may almost never occur, so learning never starts. Related: nothing
compels the network to respond at all.

**Working answer.** (a) A homeostatic "drive": output excitability rises
during silence, forcing exploration. (b) Teacher options beyond
reward-on-correct: mild negative signal on wrong answers, shaping rewards
(reward near-misses early), curriculum starting with 2 classes. Which
combination works is explicitly an experimental variable, not a settled
design choice.

**How we'll know.** Time-to-first-reward and time-to-above-chance metrics
across teacher-schedule variants.

## 5. What counts as "an output"?

**Problem.** If we read the output register every tick, noise counts as
answers. If we require sustained activity, we've chosen a window and
threshold — hidden hyperparameters of the whole paradigm.

**Working answer.** An output is a register neuron whose firing rate over a
short window exceeds threshold while the others stay below it; "no response"
is a legal and expected state, especially pre-learning.

## 6. Scaling

**Problem.** Every mechanism above may work at 64×64/few-thousand-neuron
scale and stall at the next order of magnitude — the graveyard of most
backprop alternatives.

**Working answer.** None yet. Deliberately deferred: the near-term goal is
demonstrable real-time learning at toy scale, which is falsifiable and
valuable on its own.

## 7. Evaluating a living system

**Problem.** Batch benchmarks don't fit a system whose claim is about the
learning process. There's no standard for "learned well while interacting."

**Working answer.** Longitudinal curves over the exposure stream: rolling
accuracy vs stimuli-seen, retention after reward withdrawal, retention after
task switches, robustness under perturbation, all logged and replayable.
Define these once in experiment 001 and reuse everywhere.

## 8. Recognizing intelligence when we see it

**Problem.** Once the organism can act on its world
([experiment-ideas.md §B](./experiment-ideas.md)), the temptation is to
watch it and judge. But with a stochastic always-on system the observer is
the weakest instrument: humans attribute agency to noise (Heider & Simmel's
subjects saw intent in moving triangles — see related-work). "It looked
deliberate" is pareidolia until proven otherwise, and a solo project has no
second observer to catch it.

**Working answer.** Behavior earns the word "intelligent" only against a
criterion written down *before* watching. A ladder of pre-registered,
measurable criteria, climbed in order: (1) operant — a contingently
rewarded action's rate rises; (2) contingency sensitivity — the rate falls
when the action→reward link is broken while reward rate is held constant;
(3) instrumental use — acting on the world measurably improves task
performance (e.g. the external-memory gate in experiment-ideas §B1);
(4) transfer — an acquired action speeds learning of a new task. Anything
surprising noticed ad hoc is journaled as anecdote and believed only after
a pre-registered replication.

**How we'll know.** If pre-registered criteria keep failing while observers
keep "seeing" intelligence anyway, either the criteria are aimed at the
wrong behaviors or the observers are fooling themselves — both findings.
The ladder gets its first use in the §B1 motor experiments.
