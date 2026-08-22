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

## Measuring learning without assuming a machine (added 2026-08-22)

Added before any claim rests on it, per journal rule 12. These are the sources
behind [vision.md](./vision.md#how-we-judge-whether-it-is-learning)'s ladder
and comparative battery. Every one predates machine learning, and every one
has been applied across species — which is the admissibility test (H-009).

- **Ebbinghaus (1885), *Über das Gedächtnis*** — the savings method: measure
  retention by how much *faster* something is relearned, not by whether it can
  be recalled. The original demonstration that behaviour and stored knowledge
  are separable. Rung 4.
- **Pavlov (1927), *Conditioned Reflexes*** — extinction and **spontaneous
  recovery**: a response that has been extinguished returns after a rest with
  no further training, so extinction suppresses rather than erases. A direct
  probe for a trace behaviour is not expressing.
- **Tolman & Honzik (1930), "Introduction and removal of reward and maze
  performance in rats"** — **latent learning**. Unrewarded rats looked like
  non-learners for ten days, then matched the rewarded group almost
  immediately once food was introduced. The canonical answer to "how would you
  know if it were learning invisibly?", and the model for H-010.
- **Harlow (1949), "The formation of learning sets"** — learning-to-learn:
  performance on the *n*-th novel discrimination improves with *n*, until
  rhesus monkeys solve new problems in one trial. Rung 7, and the strongest
  claim this project could eventually support.
- **Comparative reversal-learning literature** (bees, fish, rodents, primates)
  — cognitive flexibility as trials-to-recover after a contingency swap.
- **Guttman & Kalish (1956), generalisation gradients in pigeons** — orderly
  interpolation to stimuli never trained on, which a lookup table does not
  produce.
- **Kandel and colleagues on *Aplysia*** — habituation, sensitisation and
  their synaptic substrates in an animal with ~20,000 neurons; the existence
  proof that these measures do not require a big brain, or a brain we
  designed.
- **In-vivo synaptic turnover imaging** (spine formation/elimination in mouse
  cortex under learning) — the biological version of rungs 2 and 3, and the
  reason "structural persistence" is a measurement rather than a metaphor.

**What this project takes:** the measures themselves, essentially unmodified,
as the primary gates rather than as commentary on an accuracy number.
**What it does not claim:** any novelty in them. The only unusual choice is
the refusal to let a benchmark-style accuracy figure be the gate.

## The three "open seams" — all three checked 2026-08-22, all three occupied

The [2026-08-21 novelty audit](./journal/entries/2026-08-21-0050-novelty-audit-and-the-missing-i.md)
identified three seams that looked unexamined and marked all of them
*unverified*. They were searched on 2026-08-22. **None of them is open**, and
the one the audit singled out as most promising is the most thoroughly closed.

### 1. Credit assignment by diffusion geometry — pre-empted, and recently

The audit called this "the one worth pursuing, because it is a different *kind*
of answer to open-problems §1". It is experiment 002's design §4: reward as a
substance emitted from a locus, read locally at each synapse, so `Δw = η·R(x)·e`
rather than `η·R_global·e`.

- **"Diffusion of Neuromodulators for Temporal Credit Assignment"**
  ([arXiv 2603.08949](https://arxiv.org/html/2603.08949), March 2026) —
  credit assignment "determined by the local concentration of a modulatory
  particle rather than by its point of origin," in recurrent spiking networks
  on temporal tasks. This is 002 §4's central idea, published five months
  before 002 was designed.
- **Cell-type-specific neuromodulation guides synaptic credit assignment in a
  spiking neural network** (Liu et al., [PNAS 2021](https://www.pnas.org/doi/full/10.1073/pnas.2111821118))
  — and it already ran **the ablation 002 pre-registered as its control arm**:
  performance degrades when spatial specificity is removed so modulatory
  signals reach all cells without attenuation, though nonspecific modulation
  still beats none. That is 002's `rewardField: 'uniform'` versus `'diffuse'`
  comparison, already reported.

**Consequence for this project:** 002's headline mechanism is not novel, and
its M2 gate would be re-running a published ablation. This does not make 002
worthless — the combination with *grown* structure is still untested, and 002
failed at M1 for unrelated reasons — but no novelty claim can rest on §4.

### 2. Metabolic rent, and death as failure to pay — well established

Energy-constrained structural plasticity is a developed literature: overgrowth
followed by pruning maximises memory performance under metabolic constraints
([synaptic pruning and synapse efficiency](https://arxiv.org/pdf/cond-mat/0207545);
[Concurrence of form and function in developing networks](https://www.nature.com/articles/s41467-018-04537-6)),
alongside [energy-efficient synaptic plasticity](https://ncbi.nlm.nih.gov/pmc/articles/PMC7082127),
[competitive plasticity to reduce the energetic costs of learning](https://www.biorxiv.org/content/10.1101/2023.04.04.535544v1.full.pdf),
and [the write-cost bottleneck](https://link.springer.com/article/10.1007/s11571-026-10508-1)
on energy limits for continual learning. 002's rent-and-death is a
straightforward instance.

### 3. Latency as a plastic, use-tuned per-edge parameter — occupied

The audit's remaining seam, and never built in 002 either (latency is set from
span and then frozen). Delay learning in spiking networks is an active area:
[co-learning delays, weights and adaptation](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1360300/full),
[learnable axonal delay](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10665570/),
[DelRec](https://arxiv.org/html/2509.24852),
[delays via dilated convolutions with learnable spacings](https://arxiv.org/pdf/2306.17670),
and delay-domain extensions of STDP including three-factor rules for online
adaptation. Also relevant: [delay selection by STDP](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3567188/).

### What this leaves

No mechanism in this project is novel. What remains unexamined is the thing the
project is *named for* and has never built ([L-012](./journal/LEARNINGS.md)):
iterative consideration — output→input feedback, multiple internal passes per
stimulus, with reward-only learning and legal silence. Whether that is
occupied has **not** been searched.

## The missing "I" — iterative consideration (searched 2026-08-22)

The project is named for a mechanism it has never built ([L-012](./journal/LEARNINGS.md)):
`abstract.md` §2–§3 specify output→input feedback and multiple internal passes
per stimulus, and both substrates are single-pass feedforward. This was
identified on 2026-08-22 as the last place novelty might live. **It was then
searched, and it is largely occupied too.** Recorded before any design rests on
it.

**Reward-only learning in recurrent networks — done, and it works.**
- Miconi, [*Biologically plausible learning in recurrent neural networks
  reproduces neural dynamics observed during cognitive tasks*](https://elifesciences.org/articles/20899)
  (eLife 2017) — node-perturbation with reward-modulated Hebbian updates,
  guided **solely by delayed phasic reward at the end of each trial**, learning
  flexible associations and memory maintenance. This is essentially IPNN's
  learning rule inside a recurrent network, nine years ago.
- Song, Yang & Wang, [*Reward-based training of recurrent neural networks for
  cognitive and value-based tasks*](https://elifesciences.org/articles/21492)
  (eLife 2017).
- [Local online learning in recurrent networks with random feedback](https://elifesciences.org/articles/43299)
  (RFLO); [noise-based reward-modulated learning](https://arxiv.org/html/2503.23972v1).

**Deciding how long to think — done.** Adaptive Computation Time (Graves
2016), [PonderNet](https://www.alphaxiv.org/abs/2107.05407) (halting as a
latent-variable model with a geometric prior), [probabilistic ACT](https://arxiv.org/pdf/1712.00386),
deep equilibrium and fixed-point-iteration layers, and a large 2025–26
test-time-compute literature.

**Revising a decision mid-deliberation — done.** [*Changes of Mind in an
Attractor Network of Decision-Making*](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1002086)
(PLOS Comput Biol) models exactly the phenomenon the sustained readout was
built to observe ([L-009](./journal/LEARNINGS.md)). Also
[recurrence for hard inputs](https://arxiv.org/pdf/2106.04537) (easy-to-hard
generalisation) and the [debate over whether iterative convergent computation
is a useful inductive bias](https://www.biorxiv.org/content/10.1101/2023.10.13.562196.full.pdf).

### What four searches did not turn up

The specific **triple**: reward-only local three-factor learning **and**
output→input iteration **and** the organism deciding for itself when to commit,
where **silence is a legal un-punished state and the drive to answer builds
over time** (IPNN's `urge`). Halting in the ACT/PonderNet line is learned by
gradient; thresholds in the drift-diffusion line are fixed; Miconi's networks
answer every trial.

**This is a weak claim and should be treated as one.** Four searches is not a
survey, all three components are individually well covered, and an unclaimed
combination of well-covered parts is a low bar. Marked *unverified*.

### Why to build it anyway, novelty aside

1. **The spec demands it.** `abstract.md` has specified it since the beginning
   and no implementation has had it (L-012). Closing a spec-versus-source gap
   needs no novelty argument.
2. **Miconi de-risks it.** Reward-only learning is *known* to work in recurrent
   networks, so a failure here would be ours, not the paradigm's — and that
   makes it a much better-posed experiment than 002 was.
3. **The instrument already exists.** The sustained readout was built for
   precisely this (L-009: "a commit-once readout cannot measure a mind
   changing"), and manual mode already holds a stimulus while the answer
   free-runs.
