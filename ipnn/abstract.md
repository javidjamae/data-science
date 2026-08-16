# IPNN: An Iterative, Probabilistic Neural Network for Always-On, Real-Time Learning

## Abstract

We propose the Iterative Probabilistic Neural Network (IPNN): an always-on
learning system with no separation between training and inference. The system
perceives its environment through sensory surfaces (e.g., a 64×64 grid of
input neurons acting as a "visual" sense), iterates internally on what it
perceives via recurrent feedback from output to input, and adjusts its
synapses continuously through a reward-modulated, local learning rule —
interaction with the environment *is* the learning process. Neurons fire
probabilistically, with per-synapse Beta-distributed confidence that tightens
as evidence accumulates; sparse stochastic firing makes the architecture a
natural fit for low-power, event-driven (neuromorphic) computation. The
long-term goal is a single general-purpose learner that can be pointed at
different senses in sequence — or in combination — and continue functioning
on earlier ones, analogous to the uniform learning algorithm hypothesized for
the neocortex.

## 1. Motivation: interaction is the learning

Conventional neural networks split life into two phases: a training phase
(expensive, offline, batch-driven, backpropagation) and an inference phase
(frozen weights). Humans do neither. We learn and act simultaneously; every
interaction with the world is also a weight update. IPNN takes this as its
design axiom rather than as an afterthought:

- There is no training phase. The system is switched on and begins learning
  from its sensory stream immediately.
- Learning can be *disabled* (a frozen mode), but enabled is the default.
- Supervision arrives the way it does for a child: through interaction and
  reward, not through per-sample loss gradients.

This reframes the standard online-learning goal ("adapt to a data stream")
into a stronger claim: the train/inference distinction itself is the artifact
we are removing.

## 2. The system

An IPNN instance ("an organism") consists of:

- **Sensory surfaces.** Sheets of input neurons wired to the environment. The
  first sense is visual: a 64×64 binary grid. Additional senses attach to the
  same core network later.
- **A recurrent core.** A pool of stochastically-firing neurons with sparse
  connectivity, including feedback connections from the output layer back
  toward the input side. Absent external drive, activity can circulate: the
  network "considers" its own output over multiple iterations before (or
  while) responding. Time is explicit — the system runs in ticks,
  continuously, whether or not anything is happening on its senses.
- **An output register.** A small set of neurons whose sustained activity
  constitutes the system's overt response (e.g., ten neurons for digit
  identity). Critically, the system is not *forced* to produce an output on
  every stimulus; "no response" is a legal state.
- **A drive toward expression.** A homeostatic mechanism (an "urge" or
  proto-emotion) that slowly raises output-layer excitability when the system
  has been silent, compelling it to try responses that reward can then shape.
  Reward satisfies the drive; silence builds it.

## 3. Mechanism

**Probabilistic firing.** A neuron fires stochastically as a function of its
weighted input drive. Firing is binary and sparse: most neurons are silent on
most ticks, which is both the biological regime and the low-power regime.

**Beta-distributed confidence.** Each synapse carries, in addition to its
weight, a Beta-flavored confidence (an evidence count in the spirit of the
Beta distribution's α+β). Early on, confidence is low and weights are highly
plastic. As rewarded experience accumulates, confidence tightens and the
synapse becomes resistant to change. This is conjugate-Bayesian in spirit:
learning is evidence accumulation, not gradient descent.

**Reward-modulated local learning.** Learning follows a three-factor rule:

1. *Local eligibility* — each synapse maintains a decaying trace of recent
   pre/post co-activity (a record of "I participated in what just happened").
2. *Global reward* — a scalar broadcast signal R (positive for reward,
   zero for being ignored, optionally negative for correction), delivered by
   the environment or a teacher.
3. *Update* — Δw ∝ R × eligibility / confidence.

No gradients are backpropagated. This is the same family as reward-modulated
Hebbian plasticity and REINFORCE with stochastic units (see
[related-work.md](./related-work.md)); the global reward signal is our
working answer to the credit-assignment problem, and its adequacy is an
empirical question ([open-problems.md](./open-problems.md)).

**Iterative consideration.** Because output feeds back into the network, a
single stimulus triggers multiple internal iterations. Across iterations the
stochastic firing yields a *distribution* over responses rather than a point
answer — inherent uncertainty quantification — and the network can settle
toward a response rather than being read out after one pass.

## 4. Continual learning across senses

The multi-sense goal makes catastrophic forgetting a first-class concern:
train on sense A, then sense B, then return to A — the system must still
function. Our working hypothesis is that Beta-confidence tightening acts as
*consolidation*: synapses that repeatedly earned reward become low-plasticity
and are not overwritten by later learning, in the same spirit as elastic
weight consolidation but arising from the local evidence counts rather than a
separate penalty term. This is a testable claim and one of the project's core
experiments.

## 5. Multimodal combination

When two senses feed the core simultaneously, reward-modulated Hebbian
learning should strengthen synapses that link correlated cross-modal
activity — the mechanism by which the system learns that a sight and a sound
co-occur. This mirrors the neocortical picture (Mountcastle's uniform
cortical circuit; Hawkins' Thousand Brains theory) in which one repeated
learning algorithm serves every modality, and association arises from wiring,
not from modality-specific machinery.

## 6. Low power

The power story follows from the mechanism rather than being bolted on:

- Activity is sparse and event-driven; silent neurons cost (nearly) nothing.
- Learning is local (per-synapse traces plus one broadcast scalar); there is
  no global backward pass and no optimizer state.
- The always-on regime is mostly quiescent: an idle organism ticks cheaply.

This profile is precisely what neuromorphic hardware (spiking chips such as
Intel's Loihi line) is built to exploit, making a hardware-efficient
implementation a plausible long-term path.

## 7. Open problems

Stated plainly (expanded in [open-problems.md](./open-problems.md)):

- **Credit assignment.** Can a single broadcast reward scalar train hidden
  structure of useful depth? Reward-modulated rules are known to suffer high
  variance as networks grow.
- **Stability.** A stochastic recurrent loop has no a-priori guarantee of
  settling rather than oscillating; Hopfield/Boltzmann networks solved this
  with an energy function, and IPNN needs an equivalent story or empirical
  evidence of stability.
- **Forgetting.** The consolidation-via-confidence hypothesis (§4) may fail;
  it must be tested, not assumed.
- **Scaling.** Everything above may work at toy scale and stall beyond it.

## 8. Evaluation: living benchmarks

Because the claim is about *how* learning happens, not terminal accuracy,
evaluation is interactive and longitudinal rather than batch: expose the
organism to stimuli in real time, reward it through a teacher (automated or
human), and measure how performance evolves over the exposure stream —
including after rewards stop, after task switches, and under input
perturbation (translation, rotation, scale, hand-drawn input). The first such
experiment is an interactive MNIST environment
([experiments/001-mnist-living-demo/design.md](./experiments/001-mnist-living-demo/design.md)).
Success is *demonstrable real-time learning under interaction*, not
state-of-the-art digit accuracy.

## 9. Conclusion

IPNN inverts the usual ordering: the always-on, interactive, low-power
learning regime is the goal, and the architecture — probabilistic neurons,
Beta confidence, recurrent consideration, reward-modulated local updates —
is the means. Most individual ingredients have respectable lineages; the bet
is on the combination, and the open problems (credit assignment above all)
are where the research actually lives.
