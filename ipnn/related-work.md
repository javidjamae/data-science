# Related Work

What already exists, and what IPNN takes from (or claims beyond) each. Keep
entries short; this is a map, not a survey.

## Cortical theory

- **Mountcastle — uniform cortical column hypothesis (1978).** The neocortex
  runs one repeated circuit regardless of modality; what a region does is
  determined by what it's wired to, not by specialized machinery.
  *Relation:* this is the theoretical license for "one intelligence, many
  senses."
- **Hawkins / Numenta — Thousand Brains theory, HTM.** A general-purpose
  cortical learning algorithm, sparse distributed representations, continuous
  online learning, no train/inference split. *Relation:* closest in spirit to
  the whole IPNN program. IPNN differs in mechanism (Beta-confidence
  stochastic neurons + reward-modulated three-factor learning rather than
  HTM's dendritic prediction model); we should be able to say precisely what
  IPNN adds or simplifies.

## Learning without backprop

- **Three-factor learning rules / reward-modulated Hebbian plasticity
  (Frémaux & Gerstner 2016 review).** Local eligibility traces gated by a
  global neuromodulatory (dopamine-like) signal. *Relation:* this is
  literally IPNN's learning rule family; the "emotion/reward" idea maps onto
  the third factor.
- **Williams 1992 — REINFORCE.** Stochastic units updated by reward ×
  eligibility is equivalent to policy-gradient learning; known to work and
  known to have variance that grows badly with network size. *Relation:*
  tells us both that the rule is sound and where it will hurt (see
  open-problems: credit assignment).
- **Hinton 2022 — Forward-Forward algorithm.** Layer-local learning without
  backprop, motivated by online/biological learning and low-power ("mortal")
  computation. *Relation:* same motivation, different rule; a benchmark for
  what backprop-free methods can reach.
- **BCPNN — Bayesian Confidence Propagation Neural Networks (Lansner et
  al.).** Neurons as Bayesian evidence integrators with Hebbian-Bayesian
  updates; has been run on neuromorphic hardware. *Relation:* nearest
  neighbor to "Beta-confidence" synapses; worth reading closely before
  claiming novelty there.

## Stochastic and recurrent settling networks

- **Hopfield networks / Boltzmann machines.** Recurrent stochastic networks
  that settle via an energy function; known variables are *clamped* and the
  network fills in the rest. *Relation:* the precedent for IPNN's
  labels-as-input-neurons idea and the cautionary tale for stability — they
  needed an energy function to guarantee settling.
- **Deep equilibrium models, PonderNet / Adaptive Computation Time,
  test-time compute in reasoning LLMs.** Iterating a network on its own
  output to "think longer" improves answers. *Relation:* independent
  vindication of IPNN's iterative-consideration loop.
- **Predictive coding.** Cortex as a hierarchy minimizing prediction error
  through recurrent loops; approximates backprop under some conditions.
  *Relation:* alternative account of what the recurrent iterations could be
  computing.

## Uncertainty and continual learning

- **MC dropout (Gal & Ghahramani 2016), deep ensembles.** Multiple stochastic
  passes → a distribution over outputs → calibrated uncertainty. *Relation:*
  IPNN gets this for free from stochastic firing across iterations; these are
  the baselines for the uncertainty claim.
- **Elastic Weight Consolidation (Kirkpatrick et al. 2017) and the continual
  learning literature.** Protect important weights from being overwritten by
  later tasks. *Relation:* IPNN's Beta-confidence tightening is a *local,
  evidence-count* version of the same idea; the forgetting experiments should
  compare against doing nothing and against an EWC-style penalty.

## Hardware

- **Neuromorphic computing (Intel Loihi, spiking neural networks).**
  Event-driven sparse spiking at milliwatt power budgets; on-chip local
  plasticity rules. *Relation:* the existence proof for IPNN's low-power
  pillar — and a constraint worth honoring in the design (local state only,
  no global passes), so a future hardware port stays plausible.
