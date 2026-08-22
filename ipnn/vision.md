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

Rungs 4 through 7 are the ones this project actually cares about, and they are
the operational form of pillar 4 and of "teach it a second task, return to the
first, and it has not forgotten". **Rung 4 is the sharpest tool we have for
slow learning**, because savings can detect a latent trace inside a curve that
never left chance — a system can *know* something it cannot yet *do*. It is
Ebbinghaus's method from 1885 and it is not an ML measure at all.

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

- Not an attempt to beat state-of-the-art benchmark accuracy.
- Not built on existing ML training frameworks — the neuron model, learning
  rule, and simulation loop are built from scratch, because the point is a
  different learning paradigm, not a different architecture inside the same
  paradigm.
