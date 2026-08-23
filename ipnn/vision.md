# IPNN Vision — The North Star

This is the plain-language statement of what IPNN is for. It should change
rarely. The theory ([abstract.md](./abstract.md)) and the experiments evolve
in service of this.

## Thesis

IPNN is a **living model**. It is not trained in a training phase and then
deployed for inference. It is switched on, and from that moment it perceives,
acts, and learns — continuously and simultaneously. **The interaction IS the
learning.** Humans do not run a pretraining job and then freeze their weights;
they learn by living. IPNN should learn the same way.

## Pillars

1. **Always-on.** The system runs continuously — a loop of perceiving,
   internal iteration ("thinking"), and output. Learning is the default state.
   There is a switch to turn learning *off*; there is no switch needed to turn
   it *on*.

2. **Senses, not datasets.** Input arrives through sensory surfaces. A 64×64
   grid of input neurons is a "visual" sense. Later senses might be an audio
   stream, a text stream, proprioceptive signals from a simulated body. The
   system is exposed to an environment through its senses; it is not fed
   batches.

3. **Taught, not optimized.** Learning is driven by interaction and reward —
   the way a parent teaches a child by showing excitement when they get
   something right — not by a loss function and backpropagation. A global
   reward signal (an "emotion") shapes local synaptic changes.

4. **One intelligence, many senses.** The same core network can be pointed at
   one sense and learn, then at a different sense and learn, then return to
   the first and *still function*. Eventually it should handle two senses at
   once and learn to combine them, the way the neocortex fuses vision and
   hearing. The learning algorithm is general-purpose; the senses are
   interchangeable peripherals. (See Mountcastle's uniform cortical column
   hypothesis and Jeff Hawkins' Thousand Brains theory in
   [related-work.md](./related-work.md).)

5. **Low power.** The system should be able to just *sit there*, interacting
   with its environment and adjusting weights, at a small fraction of the
   energy cost of gradient-descent training. Sparse, probabilistic,
   event-driven firing is the intended route: most neurons silent most of the
   time, computation happening only where activity is.

## What success looks like

Observable behaviors, roughly in order of difficulty:

- You flash it patterns and reward correct responses, and its accuracy
  visibly improves *while you watch* — no training job, no epochs.
- You turn rewards off, and it keeps performing.
- You perturb the input (move it, rotate it, draw it by hand) and it adapts
  in real time.
- You teach it a second task, return to the first, and it has not forgotten.
- You give it two senses at once and it learns correlations between them.

## How we judge whether it is learning

The pillars above say what success *looks like*. They say nothing about how to
tell, on a Tuesday, whether the thing in front of us is learning slowly or not
learning at all. That gap is not academic: it is the difference between
abandoning a system that needed more time and pouring months into one that was
never going to work.

**Task accuracy is a bad instrument here, and we should stop treating it as
the instrument.** [J] It is borrowed from benchmark ML, which this project
explicitly is not ("What this is not", above), and it fails in three specific
ways for a living system:

- **It lags.** Learning machinery can be running long before behaviour moves.
  A flat accuracy curve cannot distinguish "no machinery" from "machinery that
  has not yet reached behaviour".
- **It is confounded by the readout.** In our own teacher, silence scores as
  wrong and an answer requires 6 fires in a 20-tick window. Accuracy therefore
  conflates *does not know* with *knows but will not commit* — a distinction
  that matters enormously and that one number destroys.
- **It has no notion of development.** A growing organism has a trajectory:
  first connected path, first reward, first structure that survives. Accuracy
  collapses all of it into a scalar about one task at one moment.

### What makes a measure admissible [J]

Borrowed measures are fine — better than invented ones, usually — but they
have to be **substrate-independent**. The test:

> Could this measure be applied to an ant, an ape, a human, and a machine?

If yes, it is measuring *learning*, and we can use it. If it only makes sense
for a system with a loss function, a train/test split and an epoch counter,
then it is measuring *a training procedure*, and we should be sceptical and
very deliberate about whether it transfers to what we are building.

This rules out most of the ML evaluation battery and rules in most of
comparative psychology, which has spent a century solving exactly our problem:
how to compare learning across organisms with wildly different bodies, senses,
neuron counts and lifespans. Their answer, broadly, is to define learning over
**behaviour in time** — how many exposures to reach a standard, what survives
an interruption, what transfers — rather than over anything internal to the
learner. Those definitions work on a sea slug with 302 neurons and on a
graduate student, which is exactly the property we need.

One consequence worth stating: **trials-to-criterion is a better currency than
accuracy.** "How many exposures to reach a fixed standard" is comparable
across systems with different asymptotes and different speeds; "accuracy at
trial 2,000" is not comparable to anything. Our gates should be written in it.

### The ladder of evidence [J→C]

So we look for evidence at every level, from structure upward. Each rung can
be true while the rung above it is false, and each rung has its own null.

| | Rung | The question | Null it is measured against |
|---|---|---|---|
| 0 | **Behaviour** | Does it get the task right? | chance |
| 1 | **Decodability** | Is the task recoverable from internal state, whether or not behaviour uses it? | a decoder on the same population, labels shuffled |
| 2 | **Structural differentiation** | Does experience leave a *structural* trace — do a taught and an untaught organism diverge? | two identically-taught organisms |
| 3 | **Persistence** | Is structure surviving longer over time, i.e. is anything accumulating? | the same measure early in the run |
| 4 | **Savings** | Is re-learning faster than first learning? | a never-taught control |
| 5 | **Transfer** | Does task B come faster having learned A? | B learned from scratch |
| 6 | **Retention** | After learning B, does A survive? | A immediately after A |
| 7 | **Learning to learn** | Does the Nth task come faster than the first? | task 1 |

Every rung is annotated below with where else it has been used, because a
measure that has been applied to a slug, a pigeon and an undergraduate is a
measure of learning rather than a measure of us:

| Rung | Also used on | Origin |
|---|---|---|
| 0 Behaviour | everything | — |
| 1 Decodability | monkey motor cortex, rodent hippocampus | systems neuroscience, not ML |
| 2 Structural differentiation | dendritic-spine imaging in trained vs untrained mice | in-vivo microscopy |
| 3 Persistence | synaptic turnover imaging | in-vivo microscopy |
| 4 Savings | humans, rats, *Aplysia*, *C. elegans* | Ebbinghaus 1885 |
| 5 Transfer | monkeys, children, pigeons | comparative psychology |
| 6 Retention under interference | humans, rodents, *Aplysia* | interference studies, ~1900 |
| 7 Learning to learn | rhesus monkeys, children, rats, pigeons | Harlow 1949, "learning set" |

Rungs 4 through 7 are the ones this project actually cares about, and they are
the operational form of pillar 4 and of "teach it a second task, return to the
first, and it has not forgotten". **Rung 4 is the sharpest tool we have for
slow learning**, because savings can detect a latent trace inside a curve that
never left chance — a system can *know* something it cannot yet *do*. It is
Ebbinghaus's method from 1885 and it is not an ML measure at all.

### The comparative battery [J→C]

The rungs say what to look for. These are the protocols that look, all of them
older than machine learning and all of them run on animals that cannot be
asked what they know:

- **Habituation / dishabituation.** Repeat a stimulus; does the response
  decline? Change it; does the response return? This is the most primitive
  learning there is — *Aplysia* does it, single cells do it — and a system
  that cannot habituate is not learning at the floor, let alone above it.
- **Extinction and spontaneous recovery.** Stop rewarding until the behaviour
  disappears, then wait, then test again. If it partly returns *without any
  further teaching*, the trace was never erased, only suppressed. Pavlov's
  result, and one of the cleanest demonstrations anywhere that behaviour and
  knowledge are different things.
- **Latent learning.** Expose the system with no reward at all, then switch
  reward on. If the pre-exposed system reaches criterion faster than a naive
  one, it was learning the whole time with nothing to show for it. **This is
  the single best instrument for "maybe it is learning invisibly"** — Tolman's
  1930 rats wandered a maze unrewarded, looked exactly like non-learners, and
  then solved it almost immediately once food appeared. It turns an
  unfalsifiable worry into a controlled experiment with a naive control group.
- **Reversal learning.** Swap the contingency — the pattern that meant A now
  means B. How many trials to recover? Run on fish, bees, rats and humans, and
  a direct probe of whether a system is flexible or merely fitted.
- **Generalisation gradient.** Present something *between* two trained
  stimuli. Does the response interpolate smoothly, or is it undefined? Pigeons,
  bees and humans all produce orderly gradients; a lookup table does not.
- **Learning set.** Run many different tasks in sequence and ask whether the
  Nth is acquired faster than the first. Harlow's monkeys eventually solved
  novel discriminations in one trial. This is rung 7, and it is the strongest
  claim any of this could support.

Savings, extinction-with-recovery and latent learning share one property that
makes them worth more to us than anything on a benchmark leaderboard: **they
can detect knowledge that behaviour is not expressing.** That is precisely the
regime a slow or blocked learner sits in.

### Slow learning is legitimate, and it makes a demand [J], [C]

We should be willing to run something for a month. [J] A system that took
weeks to become competent and then transferred what it learned to a new
problem would be far more interesting than one that converges in 800 trials
and can do nothing else.

But patience is not a free move, and it comes with a mechanical requirement:
**slow learning requires a slow variable.** [C] A system in which every state
variable has a short time constant cannot learn slowly. It reaches steady
state and then stays there, and further time buys only more steady state. So
"maybe it needs longer" is not a question about our patience — it is a
question about the design, and it is answerable: *what is the longest-lived
quantity in this system, and is it accumulating?* If nothing in the mechanism
has a long time constant, more time will not help, and the fix is to give it
one rather than to wait.

### The hazard this creates, stated plainly [C]

"It is learning, just slowly and in a way you cannot see yet" is unanswerable.
It can absorb any negative result forever, and a project that accepts it stops
being research. The whole ladder above is worth building *and* is exactly the
shape of belief that curdles into faith.

The defence is not scepticism, it is procedure, and it is one rule: **no
instrument enters the record without stating what result would count as
no-learning.** Every rung above names its null for that reason. A measure that
cannot come back negative is not a measure.

## What this is not

- **Not experiment 001.** A hand-designed, fixed architecture — even one that
  learns its weights from reward alone — is not an IPNN. It is a control arm
  and a test harness. The organism's defining property is that **the neurons
  themselves learn to connect to other neurons and create neural pathways**:
  structure is grown, not given. Every claim about *the organism*, every new
  experiment, and every demo of "it" must be demonstrated on a grown
  substrate; 001 appears only as an explicitly-labelled baseline. **[J,
  2026-08-22, stated as binding.]**
- Not an attempt to beat state-of-the-art benchmark accuracy.
- Not built on existing ML training frameworks — the neuron model, learning
  rule, and simulation loop are built from scratch, because the point is a
  different learning paradigm, not a different architecture inside the same
  paradigm.
