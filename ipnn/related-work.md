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

## Perception as a process in time (the A-track's lineage)

- **Binocular rivalry / bistable perception (Levelt 1965; Blake & Logothetis
  2002).** Hold a constant ambiguous stimulus in front of a subject and the
  percept *alternates on its own*, with characteristic dwell-time
  distributions (roughly gamma/lognormal, not exponential) that shift
  systematically with stimulus strength. *Relation:* the closest experimental
  precedent for experiment-ideas §A1 — "changing its mind" under sustained
  exposure is a rediscovery of this paradigm, and the dwell-time distribution
  is the established measure. It also supplies the null worth beating: an
  alternation rate that varies with stimulus ambiguity is the signature, mere
  alternation is not.
- **Drift-diffusion / sequential-sampling models (Ratcliff 1978; Gold &
  Shadlen 2007).** Decisions modeled as noisy evidence accumulated to a
  bound, predicting the joint distribution of choice and reaction time.
  *Relation:* the framework for reading answer latency as a measurement
  rather than a timeout, and the reference account for "does thinking longer
  help" (§A3). IPNN's sliding-window readout is a crude bound-crossing rule;
  the comparison is worth making explicit before claiming novelty in §A3.

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
- **Kappel, Habenschuss, Legenstein & Maass 2015–2018 — synaptic sampling /
  the dynamic connectome.** Synapses appear and disappear stochastically
  (modeling spine motility); plasticity is recast as sampling from a posterior
  over network *structures*, and the reward-based version shows stable
  computational function emerging from a connectome that never stops
  rewiring. *Relation:* the closest prior art to "connections form and die
  probabilistically under reward" (experiment-ideas §F). Must be read closely
  before any novelty claim there — this is the paper most likely to have got
  there first.
- **Stanley & Miikkulainen 2002 — NEAT.** Evolves network topology starting
  from a minimal structure and complexifying. *Relation:* precedent for
  "start minimal and grow," but the search is an evolutionary outer loop over
  a population, not an online local rule inside one always-on organism.
- **Biological development: synaptic overproduction-then-pruning
  (Huttenlocher); adult neurogenesis (dentate gyrus).** Cortex overshoots
  synapse counts in childhood then cuts back hard; new neurons arrive
  hyper-plastic. *Relation:* the blueprint for grow-then-prune
  (experiment-ideas §C2–C3).

## Space, distance and conduction delay (the §F lineage)

Where the network sits in space, and how long a signal takes to cross it, are
treated here as computational variables rather than implementation details.

- **Watts & Strogatz 1998 — small-world networks.** Mostly-local connectivity
  plus a few long-range shortcuts collapses path length while preserving
  clustering. *Relation:* this is the topology a distance-decaying growth rule
  produces, and cortical connectomes are famously small-world — so "our grown
  network is small-world" is a property of the rule, not a discovery.
- **Exponential distance rule (Ercsey-Ravasz et al. 2013; Braitenberg &
  Schüz).** Cortical connection probability falls off exponentially with
  distance, and that single rule reproduces much of the observed connectome
  structure. *Relation:* the empirical justification for a distance-decaying
  wiring kernel — and the reason such a kernel is a *citation*, not an idea.
- **Generative network models of connectomes (Vértes et al. 2012; Betzel et
  al. 2016; Akarca et al. 2021).** Brain-like graphs generated from a wiring
  rule trading a distance penalty against a topological term (homophily).
  *Relation:* these already generate brain-like topology from growth rules.
  Any §F claim of the form "it self-organizes into brain-like structure" is
  pre-empted here; these models just aren't *functional learners*, which is
  where §F would have to differ.
- **Achterberg, Akarca et al. 2023 — spatially embedded RNNs (seRNN).**
  Adding a spatial/wiring-cost constraint to a trained RNN yields brain-like
  modularity, energy-efficient topology, and functional signatures resembling
  neuroscience findings. *Relation:* the strongest recent demonstration that
  spatial embedding alone buys brain-like structure. Differs from §F in that
  the network is trained by gradient descent with fixed topology, not grown.
- **Izhikevich 2006 — polychronization.** Spiking networks with *axonal
  conduction delays* plus STDP spontaneously form "polychronous groups":
  time-locked, non-synchronous firing patterns in which a specific
  spatiotemporal input converges on a target because the path delays line up.
  Izhikevich argues delays vastly increase representational capacity.
  *Relation:* the direct precedent for §F's delay claim, and the one to beat.
  Crucially, his delays are *fixed* and his topology is random — §F's
  distinguishing move is that structure grows so as to exploit delay, and
  that delay is a function of the topology being grown.
- **Abeles — synfire chains.** Stable propagation of precisely-timed activity
  through chained pools. *Relation:* the classical account of what a grown
  multi-hop path would be computing, and a stability reference point.

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
