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

## The "living AI" program — everyone who has wanted this (added 2026-08-27)

**Honesty banner:** written from model knowledge on 2026-08-27 in answer to
Javid's question "it just lives in its world … always on and learning like a
human — is that not novel?"
([2026-08-27-0000](./journal/entries/2026-08-27-0000-is-a-living-ai-novel.md)).
**Not from a live search.** Every entry is *unverified against literature*.

The [old online-learning tradition](#the-old-online-learning-tradition-the-lineage-ipnn-actually-belongs-to)
section above covers the *learning rules*. This section covers the *framing* —
an agent that has no training phase, lives in a world, and develops through its
own history. That framing is not a gap in the field. It is a named research
program with a fifty-year lineage and several active flagships.

| Program | The claim, in their words | Where it sits relative to IPNN |
|---|---|---|
| **Developmental robotics / epigenetic robotics** — Weng et al., *Autonomous Mental Development by Robots and Animals*, **Science 2001**; Lungarella, Metta, Pfeifer & Sandini survey 2003; the iCub platform | A robot whose competence is not programmed or trained but **develops through open-ended lifelong interaction with its environment**, with no task specified in advance | **The closest named program to Javid's framing, and it is 25 years old.** Differs in substrate (conventional learning algorithms inside a robot) rather than in aspiration |
| **Continual / lifelong learning** — Ring, *Continual Learning in Reinforcement Learning Domains* (PhD, 1994); Thrun & Mitchell, *Lifelong Robot Learning* (1995) | An agent that learns continually, never resets, and uses everything learned so far to learn the next thing faster | Ring's 1994 thesis is essentially this project's goal statement, written 32 years earlier |
| **Never-Ending Learning** — NELL (Carlson et al. 2010; Mitchell et al., *CACM* 2018) | A system deliberately left **running continuously for years**, learning and self-correcting the whole time | Same "no stopping point" commitment; symbolic/text domain, not a sensorimotor organism |
| **The Alberta Plan** — Sutton, Bowling & Pilarski (2022); Sutton's OaK architecture | AI research should target a **continually-learning agent in an unending stream of experience** — explicitly rejecting the train-then-deploy framing | The most direct contemporary statement of "there is no stop-train-retry", from the person who wrote the RL textbook |
| **Open-endedness** — Stanley, Lehman & Clune (2017); POET (Wang et al. 2019) | Systems that generate their own ever-harder challenges rather than being handed a fixed benchmark | Javid's "life involves learning and playing games" is this, at organism scale rather than population scale |
| **Artificial curiosity / intrinsic motivation** — Schmidhuber (1991, PowerPlay 2011); Oudeyer & Kaplan | The agent supplies its own reward and its own curriculum, so it keeps learning with no external teacher | Already cited in this file for the reward question; belongs here too, as the answer to "who sets the challenges" |
| **Hawkins / Numenta (HTM)** | Continuous online learning, **no train/inference split**, one cortical algorithm across senses | Cited at the top of this file as closest-in-spirit. Worth restating here: the no-split claim is *theirs*, published, and predates this project |
| **Grossberg — ART (1976–87)** | Continuous learning with no train/test split, built around the **stability–plasticity dilemma** | The dilemma has a name because it was identified fifty years ago and never closed |
| **Neuromorphic on-chip learning** — Intel Loihi, SpiNNaker | Always-on local plasticity in hardware, at power budgets that make continuous learning affordable | The hardware half of the same bet; already cited under Hardware |

### Within-lifetime structural change vs evolutionary topology search [J]

Javid's sharpening, 2026-08-27: *"It's not that we're picking a hardcoded
wiring and evolving or culling offspring, like some evolutionary algorithms
might do. It's actually ADAPTING its neural structure to learn and adapt to its
learnings and environment."* This is a real boundary and it moves the claim, so
the prior art splits in two.

**Across-lifetime (what this is NOT):** NEAT (Stanley & Miikkulainen 2002) and
neuroevolution generally, Neural Architecture Search, Neural Developmental
Programs (Najarro, Sudhakaran & Risi 2023). Topology is searched by a
population under a fitness function; the individual's structure is fixed once
built.

**Within-lifetime (the actual neighbourhood), and it is populated:**

- **Growing Neural Gas** (Fritzke 1995) and the self-organising-map family —
  adds and removes nodes and edges online, driven by accumulated local error.
  The oldest close analogue: topology grown from the data stream itself.
- **Cascade-Correlation** (Fahlman & Lebiere 1990) — recruits new hidden units
  when learning stalls, during training.
- **Dynamic sparse training** — SET (Mocanu et al. 2018), RigL (Evci et al.
  2020): prune-and-regrow connections *throughout* training rather than after
  it. A large and active area.
- **DEEP R** (Bellec et al. 2018) and **synaptic sampling / the dynamic
  connectome** (Kappel, Habenschuss, Legenstein & Maass) — already cited under
  Structural plasticity; both rewire online in a running network.
- **Progressive Networks** (Rusu et al. 2016), **Dynamically Expandable
  Networks** (Yoon et al. 2018) — add capacity for a new task, but are *told*
  where the task boundary is.

**What separates this project from that list**, stated as the conjunction
because no element separates it alone: every system above is steered by a
**gradient or a loss inside a training loop**, and most are given **task
boundaries**. Here structural change is steered by **local reward and a
metabolic economy in which failure to pay is death**, in an organism that is
never not running, with no task boundary supplied and no global objective to
differentiate. *Unverified* — dynamic sparse training is a large enough
literature that a reward-driven, boundary-free variant plausibly exists.

**Reference frames** — Javid's *"it learns reference frames over time
(hopefully)"* — is **Hawkins' central claim**, cited at the top of this file:
cortical columns learning reference frames is the Thousand Brains theory. This
project may not claim the idea. What would differ is the **origin**: Numenta
posits grid-cell-like modules as machinery; backlog
[§G](./experiment-ideas.md) asks whether a reference frame can be *grown* by an
organism that starts with none. Nobody should read "learns reference frames" as
novel; "grows the machinery that represents them, from zero edges" is the
claim, and it is unbuilt.

### What this does *not* settle

Prior art of an aspiration is not prior art of a mechanism. Every program above
wanted the same thing and each is limited by a specific mechanism failure —
credit assignment that degrades with scale, stability-versus-plasticity,
capacity that must be rationed. The right reading is
[L-011](./journal/LEARNINGS.md): **prior art is a check on claims, not a source
of designs.** It should change what this project *claims* and not one line of
what it *builds*.

**What this project may not claim, as of this entry:** that always-on learning,
the absence of a train/deploy split, an agent living in a world, or
self-directed play are new ideas. They are not, and several are older than
deep learning.

**The smallest claim that survives the above** — *unverified*, and stated as a
conjunction because no single element survives alone: an always-on,
reward-only, gradient-free organism **whose wiring is grown rather than given,
under a metabolic economy in which failure to pay is death**, receiving no task
boundaries, and evaluated on measures that would apply to an animal. Progressive
Networks (Rusu et al. 2016) add capacity per task but are *told* where the task
boundary is; DEEP R and synaptic sampling rewire online but on a given
topology; developmental robotics develops behaviour on a fixed network. The
conjunction is what has not turned up.
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
- **Aitchison, Jegminat, Menendez, Pfister, Pouget & Latham 2021 — *Synaptic
  plasticity as Bayesian inference* (Nature Neuroscience)** (added 2026-08-27,
  *unverified — not searched*). Each synapse represents a **posterior
  distribution** over its own weight, and the **uncertainty modulates the
  learning rate**: uncertain synapses move fast, confident ones move slowly.
  *Relation:* this is IPNN's Beta-confidence gating, published, with
  experimental support. Combined with BCPNN above, the "Bayesian synapse"
  half of the architecture must be treated as **occupied**, and the honest
  novelty question moves to whether *structure* is driven by the same
  posterior — see the dynamic-connectome entry under Structural plasticity,
  which is where that too was got to first.

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

## Adaptation and fast transfer in RL (added 2026-08-26)

**Honesty banner:** this section was written from model knowledge on
2026-08-26, **not from a live literature search**, in answer to Javid's
question "do RL models already do well at adaptability?"
([2026-08-26-2341](./journal/entries/2026-08-26-2341-adaptability-the-measure.md)).
Every claim here is *unverified against literature* until someone runs the
search. It is filed anyway because its absence was itself the finding: the map
had a hole exactly where the project's own goal statement lives.

The project's goal — [rung 7](./vision.md#the-ladder-of-evidence), "adapt,
don't relearn" — has a large and active ML literature under four different
names, none of which appeared anywhere in this file until now.

### The names it goes by

| Name | Core claim | Canonical references |
|---|---|---|
| **Meta-learning / "learning to learn"** | Train across a *distribution of tasks* so a new draw from it is solved in few gradient steps or few episodes | MAML (Finn, Abbeel, Levine 2017); Reptile (Nichol 2018); RL² (Duan 2016) and *Learning to reinforcement learn* (Wang 2016); PEARL (Rakelly 2019) |
| **Zero-shot / few-shot policy generalisation** | Measure performance on *held-out levels*, not training levels — and RL agents turned out to memorise levels badly | Procgen / CoinRun (Cobbe 2019–20); Kirk et al., *A Survey of Zero-shot Generalisation in Deep RL* (JAIR 2023) |
| **Domain randomisation / sim-to-real** | Randomise the physics during training so the real object is *in distribution* on first contact | Tobin 2017; OpenAI Dactyl + Automatic Domain Randomization (2019) |
| **Continual / lifelong RL** | Keep the earlier tasks while learning later ones — and keep the *ability* to learn at all | EWC (Kirkpatrick 2017); GEM and the BWT/FWT metrics (Lopez-Paz & Ranzato 2017); primacy bias (Nikishin 2022); dormant neurons (Sokar 2023); **loss of plasticity** (Dohare et al., *Nature* 2024) |

Two results are the closest analogues of what this project says it wants:

- **AdA — *Human-Timescale Adaptation in an Open-Ended Task Space*
  (DeepMind, 2023).** An agent in XLand 2.0 meets genuinely held-out tasks and
  improves within a handful of episodes, at roughly the rate a human tester
  does. This is the strongest published "comes in new and is decent
  immediately" result. *Relation:* it is the existence proof that the
  behaviour is achievable — and the honest reading of its cost is below.
- **In-context learning in large models.** Adaptation with **no weight change
  at all**: the new task is absorbed through the input channel. *Relation:*
  this is structurally the same claim as [H-018](./journal/HYPOTHESES.md)'s
  "rule as a sense" — condition the output on a cue rather than re-fit the
  mapping — arrived at from the opposite direction and at nine orders of
  magnitude more scale.

### The distinction that actually matters: amortised vs earned

Nearly all of the above buys adaptation speed by **paying for it up front, in
task coverage.** Domain randomisation makes the new ball fast to handle by
having trained on ten thousand balls; meta-RL adapts fast *within its
meta-training distribution* and degrades sharply outside it; AdA's minutes-long
adaptation sits on top of an enormous procedurally generated task space and a
correspondingly enormous number of environment steps. The adaptation is real;
it is **amortised**, not free, and it is bounded by the family it was amortised
over.

Two things follow for this project:

1. **A fast-adapting entrant is not evidence of the mechanism this project is
   after** unless the game is outside the family it was trained on. That makes
   *held-out game family*, not *held-out game*, the load-bearing control in
   backlog [§H](./experiment-ideas.md#h-the-games--an-evaluation-ecosystem-not-another-benchmark).
   Registered as [H-027](./journal/HYPOTHESES.md).
2. **The retention half is genuinely unsolved, and is where the field is
   weakest.** Catastrophic forgetting is old news; the sharper 2024 result is
   *loss of plasticity* — networks trained through long task sequences
   progressively lose the ability to learn anything new, which is
   [L-034](./journal/LEARNINGS.md) arrived at independently in a completely
   different substrate. Meta-RL benchmarks almost universally score
   first-exposure efficiency on a fresh agent and do not score what the agent
   still knows from task 1.

**What this project takes:** the vocabulary (few-shot, zero-shot, meta-train
distribution, BWT/forgetting), and the amortisation control as a hard
requirement on §H. **What it does not claim:** that fast adaptation is novel,
or that anything here beats meta-RL at adapting. On the measured record it
does not — see [L-030](./journal/LEARNINGS.md), [L-033](./journal/LEARNINGS.md),
[L-039](./journal/LEARNINGS.md). **What it claims may be different:** the
*source* of the adaptation — a factoring built into the substrate rather than
coverage bought with experience — and the insistence on scoring acquisition
and retention on the same organism, which the RL adaptation literature and the
continual-learning literature currently do separately.

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

## How real brains wire themselves (searched 2026-08-22, for the convergence program)

Added when Javid redirected the 002 convergence work to biology **[J]**: how do
actual nervous systems emerge and grow, and what does that imply for a
substrate that must grow its own connections?

**The developmental sequence, neocortex:**

1. **Neurogenesis and radial migration** — neurons are born in the ventricular
   zone and migrate along radial glia into *layers*, inside-out (deep layers
   first). The laminar skeleton exists before learning does.
   ([Molecular pathways of projection-neuron production and migration](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4682034/))
2. **A protomap, then molecular routing.** Transcription-factor gradients
   pattern the cortical primordium before activity matters
   ([the protomap propagates through an Eomes-dependent intermediate map](https://www.pnas.org/doi/10.1073/pnas.1209076110)),
   and growth cones are steered to their approximate **target region** by
   graded guidance molecules — ephrins, netrins, semaphorins, slits
   ([Eph/ephrin in cortical development](https://esmed.org/MRA/mra/article/view/1694),
   [netrin-4 and thalamocortical branching](https://www.pnas.org/doi/10.1073/pnas.1402095111)).
   Sperry's chemoaffinity hypothesis, vindicated in gradient form. **Coarse
   topology is inherited; activity never routes it.**
3. **Structured spontaneous activity.** Before the eyes open, the retina
   generates *waves* — spatially correlated, travelling fronts, not white
   noise. Neighbouring cells co-fire, so the correlations themselves carry
   topographic information, and disrupting the wave *pattern* (not the
   activity level) breaks map refinement — the waves are **instructive**
   ([retinal waves likely instruct eye-specific projections](https://link.springer.com/article/10.1186/1749-8104-4-24),
   [waves can generate long-range horizontal connectivity in V1](https://www.jneurosci.org/content/40/34/6584),
   [waves simulate future optic flow](https://www.science.org/doi/10.1126/science.abd0830)).
4. **Overproduction, then activity-dependent pruning.** Exuberant synapse
   production followed by large-scale elimination, Hebbian refinement deciding
   what survives ([ephrin-A2 and experience-dependent pruning](https://pmc.ncbi.nlm.nih.gov/articles/PMC3792401/);
   the two-component summary — **graded guidance cues + activity-dependent
   refinement** — in [Cang et al., superior colliculus maps](https://pubmed.ncbi.nlm.nih.gov/18945909/)).
   Huttenlocher's classic counts (textbook) put peak synapse density far above
   adult levels.
5. **Critical periods.** Plasticity is deliberately high early and then gated
   down (inhibitory maturation, perineuronal nets — Hensch, textbook).
   Closure is *regulated and reopenable*, unlike L-034's accidental terminal
   freeze.

**Simpler animals bracket the axis.** *C. elegans* (302 neurons) is essentially
genetically wired — the locked-at-birth end. Insects are largely genetic with
activity refinement. The neocortex sits far toward grown-under-guidance —
but **no animal occupies "no inherited routing, unstructured noise, purely
local growth."** That point in design space is uninhabited, and it is exactly
where 002 stands.

**Model lineage for the refinement half:** Willshaw & von der Malsburg's
self-organizing maps from correlated activity; Linsker; Miller's
correlation-based development of ocular dominance and orientation (all
textbook); Katz & Shatz's synthesis of activity in circuit construction.

**What this project takes:** the *sequence* — inherited coarse scaffold →
structured spontaneous activity → correlation refinement → overproduce/prune
→ staged plasticity. **What it does not claim:** any novelty; this is the
most-trodden ground in developmental neuroscience, which is precisely its
value here.
