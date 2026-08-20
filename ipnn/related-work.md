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

## The old online-learning tradition (the lineage IPNN actually belongs to)

Real-time, no-training-phase learning is not a new idea — it is an old
program that lost to batch training on capability, not on vision. IPNN's
positioning: everybody wanted this; it remains unsolved at scale.

- **Barto, Sutton & Anderson 1983 — actor-critic pole balancer.** Neuron-like
  adaptive elements learning in real time from a scalar reward signal, with
  eligibility traces. *Relation:* the closest structural ancestor of the M1
  engine — reward-only, trace-based, always-on. What it never faced: many
  tasks, many senses, retention, scale.
- **Grossberg 1976–87 — Adaptive Resonance Theory (ART).** Continuous online
  learning with no train/test split, built explicitly around the
  **stability–plasticity dilemma** — Grossberg's name for exactly the
  tension we rediscovered as L-004 (consolidation vs frozen-wrong-answer).
  *Relation:* required reading before designing the M4 forgetting
  experiments; ART is a candidate baseline/idea-source for
  confidence-gated plasticity.
- **Rosenblatt 1958 (perceptron), Widrow 1960 (ADALINE/LMS).**
  Sample-by-sample online weight updates; LMS descendants ran in deployed
  real-time adaptive filters. *Relation:* proof that always-learning systems
  were normal engineering before the batch era.
- **Samuel 1959 (checkers), Ashby 1948 (homeostat), cybernetics era.**
  Systems that improved through interaction while operating. *Relation:* the
  "living machine" framing predates AI as a field.

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

## Structural plasticity and pruning

- **LeCun, Denker & Solla 1990 — Optimal Brain Damage; Han et al. 2015 —
  magnitude pruning.** Trained networks are massively over-parameterized;
  most synapses can go. Importance must be *estimated* (Hessian terms,
  weight magnitude). *Relation:* baselines for the pruning track
  (experiment-ideas §C). IPNN's testable angle: evidence counts are a
  native, locally-computed importance signal these methods approximate.
- **Frankle & Carbin 2019 — lottery ticket hypothesis.** Small
  subnetworks suffice if well-initialized. *Relation:* existence proof for
  "same competence, smaller body" — but the result is about
  retrainability-from-init under SGD, which has no analogue in an
  always-on regime; comparisons must respect the disanalogy.
- **Bellec et al. 2018 — DEEP R.** Rewiring during training in sparse
  (spiking) networks — connections die and are resampled online.
  *Relation:* closest algorithmic precedent for structural plasticity in
  IPNN's regime.
- **Biological development: synaptic overproduction-then-pruning
  (Huttenlocher); adult neurogenesis (dentate gyrus).** Cortex overshoots
  synapse counts in childhood then cuts back hard; new neurons arrive
  hyper-plastic. *Relation:* the blueprint for grow-then-prune
  (experiment-ideas §C2–C3).

## Agency and sensorimotor learning

- **Skinner — operant conditioning; Rescorla 1968 — contingency vs
  contiguity.** Learned agency has a minimal experimental definition:
  contingent reinforcement raises response rate, and degrading the
  contingency (same reward rate, non-contingent) lowers it. *Relation:*
  the motor-access gates (experiment-ideas §B1) are lifted directly from
  this playbook.
- **Intrinsic motivation / curiosity (Schmidhuber; Oudeyer; Pathak et al.
  2017).** Agents that generate their own exploration signal. *Relation:*
  kin to the urge, and the frame for "play" (experiment-ideas §B3).
- **Friston — active inference.** Perception and action as one loop
  minimizing surprise. *Relation:* a rival account of what a closed-loop
  IPNN would be computing; worth a precise comparison if §B1 works.
- **Heider & Simmel 1944.** Humans attribute intent to moving geometric
  shapes. *Relation:* the observer-bias evidence behind open-problems §8 —
  why "it looked deliberate" is inadmissible without a pre-registered
  criterion.

## Hardware

- **Neuromorphic computing (Intel Loihi, spiking neural networks).**
  Event-driven sparse spiking at milliwatt power budgets; on-chip local
  plasticity rules. *Relation:* the existence proof for IPNN's low-power
  pillar — and a constraint worth honoring in the design (local state only,
  no global passes), so a future hardware port stays plausible.
