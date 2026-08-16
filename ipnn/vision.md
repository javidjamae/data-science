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

## What this is not

- Not an attempt to beat state-of-the-art benchmark accuracy.
- Not built on existing ML training frameworks — the neuron model, learning
  rule, and simulation loop are built from scratch, because the point is a
  different learning paradigm, not a different architecture inside the same
  paradigm.
