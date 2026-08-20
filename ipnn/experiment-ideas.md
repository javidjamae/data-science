# Experiment Ideas

The backlog. Raw ideas land here with a date, get shaped into falsifiable
hypotheses, and graduate to a numbered `experiments/NNN/design.md` when
they're next up. Nothing here is a claim — claims live in `abstract.md`,
gates in design docs, evidence in the journal.

**Maturity labels:** `raw` (an itch, unshaped) → `shaped` (hypothesis + gate
sketched) → `designed` (has a design.md) → `done` / `parked` (with a link to
why). Ideas are never deleted, only parked.

## How a raw idea becomes an experiment

The reframing recipe, applied to every entry below:

1. **Name the capability** in one sentence, plain words.
2. **State the falsifiable claim** — what observation would prove it *wrong*?
   If nothing could, it isn't an experiment yet (it may still be a good
   design principle — label it as such).
3. **Find the cheapest measurable gate** — the smallest, preferably headless,
   test that would move the claim. Watchable demos come after measurable
   gates, not instead of them.
4. **List the mechanism gaps** — what the engine can't do yet that the
   experiment needs.
5. **Name the prior art** — add it to [related-work.md](./related-work.md)
   first, then cite it (journal rule: prior art before novelty claims).

---

## A. Temporal dynamics — is it a process or a lookup table?

The load-bearing word in this project is *living*. A frozen network computes
f(input) once; a living one inhabits time — its answer is a trajectory that
can be revised, and its behavior bends under reward *while it happens*.
Nothing has tested this yet. Known engine facts that frame all of A: the M1
organism is **feedforward** (no output→pool feedback — designed in
design.md §3, not yet implemented), within-exposure state is only urge +
sampling noise, reward is delivered once per trial by the teacher, and the
"spoken" answer is a 20-tick sliding-window vote. At ~100 ticks/sec, that
window is ~200 ms and today's 60-tick exposure is ~0.6 s — a "changes its
mind after 3–4 seconds" regime needs 300–400-tick exposures. All cheap to
change; none changed yet.

### A1 — Changing its mind `shaped` ★ recommended next

*Raw idea (Javid, 2026-08-20):* ask it what number it sees; it answers 6,
then revises to 7 a few seconds later. It must not be a fixed output per
input.

- **Capability:** under sustained exposure, the spoken answer is a trajectory
  with structure — dwell, revision, settling — not an i.i.d. draw per query.
- **Falsifiable claim:** on ambiguous stimuli (morphs between two learned
  glyphs), revision statistics (switch rate, dwell-time distribution) differ
  measurably from an i.i.d. null model built by shuffling the same organism's
  per-tick output samples; on clean learned stimuli, revisions are rare
  (stability *is* expressed confidence). If trajectories are
  indistinguishable from shuffled noise, the claim dies.
- **Gate sketch:** learning **off** (isolate dynamics from plasticity), a
  morph continuum between two learned patterns, 300+-tick exposures, many
  repeats × seeds. Report answer-entropy and revision rate vs morph
  position, against the shuffle null.
- **Mechanism gaps:** morph-stimulus generator; longer exposures; a
  tick↔wall-clock convention. No engine changes.
- **Honest expectation:** a feedforward organism may show only window-noise
  revision. That *negative is informative* — it measures how much temporal
  depth exists today and becomes the pre-registered baseline that makes A3's
  feedback loop a before/after measurement instead of a vibe.

### A2 — Live coaching: reward as a real-time knob `shaped` ★ pairs with A1

*Raw idea (Javid, 2026-08-20):* adjust the encouragement/reward while it's
answering, and see whether behavior bends.

- **Capability:** mid-exposure reward contingent on the *current* spoken
  answer shifts the answer trajectory within the same exposure — coaching,
  not training. It says 6, you withhold; it tries 7.
- **Falsifiable claim:** P(switch within T ticks | current answer
  unrewarded) > P(switch | current answer rewarded), same organism,
  counterbalanced, by a pre-registered margin over A1's spontaneous switch
  rate. Secondary dose–response: switch probability varies monotonically
  with reward magnitude.
- **Gate sketch:** learning **on**. Teacher gains a *continuous mode*:
  reward delivered at tick granularity the moment a spoken answer forms,
  contingent on what it currently is. The mechanism is already built for
  this — eligibility traces (λ=0.97, ~30-tick credit window) exist precisely
  so reward can arrive asynchronously; the trial was always the teacher's
  fiction, never the organism's.
- **Mechanism gaps:** continuous-mode AutoTeacher (reward at arbitrary
  ticks); within- vs across-exposure bookkeeping.
- **Threat to pre-empt:** "within-exposure adaptation" could just be many
  tiny training steps. Separate the timescales: measure behavior shift
  within the exposure *and* weight drift across exposures; run a
  plasticity-frozen arm (does coaching still bend behavior via urge alone?).

### A3 — Does thinking longer help? `shaped` (the 2023 claim)

The oldest unpaid debt: the 2023 abstract's one hypothesis — iterative
consideration improves accuracy — flagged *untested* in the journal's
[retrospective entry](./journal/entries/2023-07-12-2228-original-abstract-published.md).

- **Falsifiable claim:** accuracy improves with readout time beyond what
  window-averaging noise reduction explains.
- **Two pre-registered stages:** (i) measure accuracy vs readout ticks on
  the *current feedforward* engine — prediction: **flat** after ~1 window
  (a negative result, pre-registered as such); (ii) implement output→pool
  feedback (design.md §3 already specs it) and re-measure. Stage (i) is the
  control that makes stage (ii) mean something.
- **Mechanism gaps:** stage (ii) is the first real recurrence — triggers
  open-problems §2 (stability telemetry required).

### A4 — Tracking vs perseveration `shaped`

- **Capability:** swap the stimulus mid-exposure (the 6 becomes a 7 on the
  grid); the organism revises within a bounded latency instead of
  perseverating on its first answer.
- **Falsifiable claim:** median revision latency is finite and scales with
  stimulus similarity; a confidence-locked organism (high consolidation)
  perseverates longer — L-004's confidence–plasticity tension, measured
  behaviorally.
- **Gate sketch:** clean swaps at fixed offsets; latency distributions vs
  swap similarity and consolidation level. Cheap; headless; needs only
  longer exposures.

---

## B. Agency — closing the loop on the world

*Raw idea (Javid, 2026-08-20):* give it access to its own grid — read what
we wrote, but also write, erase a pixel, clear. Watch for emergent signs of
intelligence. And: how do we even determine a response is "intelligent"?

### B1 — Motor access to the grid `shaped`

- **Capability:** action outputs (write pixel / erase / clear — start with
  one action, not a toolkit) that modify the sense surface. The organism's
  input now depends on its own behavior: a closed sensorimotor loop.
- **Falsifiable claims, as a ladder** (each rung is its own gate; climb in
  order — see open-problems §8 for why the ladder exists):
  1. **Operant:** make an arbitrary motor action contingently rewarded; its
     rate rises above the un-rewarded baseline. The Skinner-box minimum —
     if this fails, nothing above it is worth running.
  2. **Contingency sensitivity:** break the action→reward link (same reward
     rate, delivered non-contingently); the action rate falls back. This
     separates control from coincidence.
  3. **Instrumental use (external memory):** with a distractor interval
     between stimulus and answer, an organism allowed to write on its grid
     outperforms the same organism with writing disabled — it used the
     world as memory. This is the first result that would deserve the word
     "intelligent" out loud.
- **Mechanism gaps:** motor semantics in the engine; teacher contingency
  modes; reward design that survives an agent (see risk).
- **Risk worth wanting:** reward hacking — e.g. it learns to blank the grid
  to shortcut the teacher. Design the teacher to be robust to it, but log
  it eagerly: gaming the contingency *is* evidence of agency.

### B2 — Recognizing intelligence: pre-registered criteria, not vibes
`shaped, methods` — this is the answer to "how do we determine a response is
intelligent?", and it's a methods commitment rather than an experiment: with
a stochastic always-on system, the observer is the weakest instrument, and
humans attribute agency to noise (we see intent in moving triangles). So no
behavior gets called intelligent unless its criterion was written down
*before* watching. The criteria ladder is B1's; anything surprising noticed
ad hoc is journaled as anecdote and believed only after a pre-registered
replication. Promoted to [open-problems.md §8](./open-problems.md).

### B3 — Play: self-generated curriculum `raw`

Between lessons, let it act freely on its grid. Does free interaction
("play") improve subsequent taught learning vs an idle control? Kin to the
urge, and to intrinsic-motivation work (related-work: agency). Far out;
needs B1 rungs 1–2 first.

---

## C. Structural plasticity — the body changes

*Raw ideas (Javid, 2026-08-20):* can it grow new neurons, not just new
connections? Can it prune itself, killing unused neurons? And separately:
prune a trained network down — same intelligence, smaller body.

### C1 — Evidence-guided pruning `shaped` ★ cheapest C-track entry

- **Capability:** kill synapses that never earned their keep, without
  hurting behavior.
- **The IPNN-specific angle:** classic pruning must *estimate* importance
  (weight magnitude, Hessian terms). IPNN synapses already carry evidence
  counts `n` — a native, locally-computed record of having participated in
  rewarded behavior. The interesting claim is not "pruning works" (known
  since 1990) but that **evidence is a better knife than magnitude**.
- **Falsifiable claim:** at matched sparsity on the M1 task, pruning by
  lowest evidence×|w| degrades accuracy less than pruning by |w| alone or
  at random. If magnitude alone does just as well, the Beta-confidence
  machinery earns nothing here.
- **Gate sketch:** accuracy-vs-sparsity curves, three criteria, multiple
  seeds. Fully headless, engine-ready today — a strong candidate for the
  slot after the A-track.

### C2 — Neurogenesis: growing itself `raw`

- **Capability:** spawn neurons on demand instead of fixing pool size.
- **Toward a claim:** trigger (persistent non-reward? high urge? pool
  saturation?), placement (wire newborns into currently-active
  neighborhoods?), and newborn plasticity (biology says new neurons arrive
  hyper-plastic) are all open design axes. Falsifiable form once shaped:
  a grow-on-demand organism matches a fixed organism *of its final size*
  trained on the same task sequence, while spending fewer neuron-ticks
  total.
- **Prerequisite:** C1 — growth without pruning is a ratchet.

### C3 — Overproduce, then prune: development `raw`

Biology's ordering: childhood overproduces synapses ~2×, then adolescence
cuts them back, hard. Claim once shaped: start oversized → learn → prune by
evidence, ends *better* than training at the final size from the start,
at equal final cost. Composes C1+C2 into a developmental trajectory.

### C4 — Compression: how small can competence get? `raw`

The floor-finding version of C1: prune until the M1/M2 gate fails; report
the smallest passing organism. Feeds the low-power story (abstract §6) —
sparse *and* small is the neuromorphic regime. Baselines: magnitude
pruning; note the lottery-ticket disanalogy honestly (their result is about
retrainability-from-init under SGD; there is no analogue of our always-on
regime, so comparisons need care — see related-work).

---

## D. Architecture origin — designed scaffold vs emergent structure

*Raw idea (Javid, 2026-08-20):* fully self-emergent architecture, vs us
defining components ("grid", "cortical columns").

### D1 — The minimal designed scaffold `shaped, program not experiment`

Sharpen the dichotomy: the sense grid and output register are the
organism's **body** — the I/O contract with the world — and even biology
gets its retina and muscles from the genome. Those stay designed. The real
question is everything *between*: pool topology, modularity, columns.
Reframed as an ablation program: for each piece of designed internal
structure, replace it with structural plasticity (C-track) plus a local
growth rule, and measure what the deletion costs. "Fully self-emergent" is
the limit of this program, not a separate experiment.

### D2 — Emergent modularity `raw, needs multi-sense + C-track`

- **Falsifiable claim:** under multi-task/multi-sense exposure with local
  growth/prune rules, graph-community structure emerges that aligns with
  tasks/senses (modularity score vs a degree-matched random rewiring null),
  without modules being drawn in by hand.
- This is the emergent answer to "cortical columns": columns as an outcome,
  not a blueprint.

### D3 — Designed columns as the comparison arm `raw`

Build the Mountcastle-style columnar variant *only* as a baseline against
D2 — resist making it the default, because hand-drawing the modules would
quietly bake in the answer to D1's question.

---

## E. Already registered (index only — claims live in their own docs)

- **M1b — etaPool=0 ablation:** is the pool contributing anything yet?
  Pre-registered in [entry 1 §Next](./journal/entries/2026-08-16-0248-experiment-001-m0-m1-first-build.md).
- **M2 — minimal living demo** (MNIST {0,1,2}, UI): pre-registered gate in
  [design.md §6](./experiments/001-mnist-living-demo/design.md). The
  A-track's measures are what the M2 UI should make *watchable*.
- **M3 — full experience; M4 — teacher-schedule ablation, forgetting probe,
  reward-withdrawal drift:** design.md §6.
- **Multi-sense / multimodal:** experiments 002+ per design.md §8.

---

## Current ranking (2026-08-20)

1. **A1 + A2 as one experiment** — "changing its mind / live coaching."
   Attacks the word *living* directly; nearly engine-ready; produces the
   baseline A3 needs; its measures become M2's most watchable content.
   Note: entry 1 pre-registered M2 as next — running this first is a
   sequencing change to record in that experiment's journal entry.
2. **A3 stage (i)** — free to run alongside A1 (same harness), settles the
   2023 claim's null baseline.
3. **C1** — cheapest test that the Beta-confidence machinery pays rent.
4. **M1b** — still the cheap standing pre-step for M2.
