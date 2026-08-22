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

**Superseded in ambition by [§F](#f-the-grown-substrate--nothing-wired-but-the-interfaces)**,
which takes this to its limit and adds what D never had: nodes with positions,
edges with latency, and a substrate where credit arrives by diffusion. D
remains the right frame for *incremental* deletions of designed structure; F
is the clean-sheet version.

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

## F. The grown substrate — nothing wired but the interfaces
`designed` — graduated 2026-08-21 to
[experiments/002-grown-substrate/design.md](./experiments/002-grown-substrate/design.md),
which holds the pre-registered gates and control arms. This section remains the
*reasoning*: the limit case of [D1](#d1--the-minimal-designed-scaffold-shaped-program-not-experiment),
with one ingredient D never had — **time-of-flight**.

*Raw idea (Javid, 2026-08-21, thinking out loud):* define no internal
connections at all. Pin only the interfaces — an input cortex, an output
cortex, a reward cortex, each just big enough to carry what it must. Interior
nodes start **completely disconnected**. As the system runs, nodes
probabilistically form connections with neighbours, and can also "jump"
further, with jump probability falling off with distance. Active and rewarded
pairs are more likely to connect and to strengthen; weak connections die. And
critically: **a jump arrives sooner than the chain of short hops it
replaces** — three nodes crossed in one leap beats three sequential relays —
so long-range edges are the fast path and local chains are the slow path.
Fast and slow thinking fall out of the geometry.

### First principles: seven things nervous tissue does that this engine doesn't

Read off biology, then asked "what would that force here?" — deliberately
*not* derived from any ML method. (Prior art is checked at the end of this
section, after the design, not before it.)

1. **Wire costs money, continuously.** An axon occupies volume and burns ATP
   whether or not it is useful, and a large fraction of brain architecture is
   explained by that bill. → An edge should pay **rent** every tick it exists.
   Deletion is then not a tidying pass, it is *failure to pay*. Pruning stops
   being a maintenance phase and becomes the selection pressure that shapes
   the organism. This is a different thing from C1 and supersedes its framing.

2. **Signals take time — and the brain tunes that time.** Oligodendrocytes
   preferentially myelinate axons that are being used, which changes their
   conduction velocity. The brain does not merely *have* delays, it *adjusts*
   them based on activity. → Latency should be a **plastic parameter of the
   edge**, not a fixed function of its span. A path that needs to be fast
   becomes fast. This is strictly stronger than the original idea (where
   latency follows from distance alone) and is the piece with the least
   obvious owner anywhere.

3. **Growing axons climb gradients.** A growth cone does not sample its
   neighbours with a distance prior; it follows chemical fields laid down by
   targets (netrins, ephrins, semaphorins). → Growth should be **directed by
   a field**, not drawn from a distance kernel. Distance-decay is then an
   *emergent consequence* of diffusion, not an assumption.

4. **Neuromodulation is a diffusing field from a source, not a broadcast.**
   Dopamine, ACh and noradrenaline are released from loci and spread by
   volume transmission; concentration falls off with distance from the
   source, different modulators diffuse at different rates, and they mean
   different things — reward-prediction error, surprise, "gate plasticity
   now". → Javid's **reward cortex is right**, and the concentration gradient
   it creates is not a defect to be engineered away. It is information.

5. **Structure is built before experience.** The developing visual system
   generates its own structured activity — retinal waves — and wires itself
   *before* the eyes can see. → Spontaneous firing is not a cold-start hack
   bolted on to get things moving; it is the developmental mechanism, and it
   should be a first-class phase with its own dynamics.

6. **The brain goes offline to change itself.** Sleep is when consolidation
   and much structural change happen. → Gate structural edits to an **offline
   phase**: compute while awake, grow and prune while asleep. This also
   dissolves the stability worry that has blocked recurrence since day one —
   you never rewire mid-thought.

7. **Dendrites compute, so *where* a connection lands matters.** Synapses
   clustered on the same dendritic branch interact superlinearly; a neuron is
   a tree of coincidence detectors, not a point summing unit. → An edge
   should target a **branch**, and co-active inputs clustering onto the same
   branch is a purely local mechanism with no global bookkeeping at all.

### The synthesis — the idea actually worth having

Put 2, 3 and 4 together and **credit assignment stops being a separate
algorithm.**

If reward diffuses outward from a source, and axons grow up gradients, then
the network *physically grows toward the things that pay*. Structure formation
and credit assignment become one mechanism instead of two. A synapse is
credited because of **where it is**, and it is where it is because it grew
there following the reward field.

That is a different kind of answer to [open-problems §1](./open-problems.md)
than anything currently in this repo. Today's answer — eligibility traces ×
broadcast reward — is the standard one, and it carries the standard variance
problem that gets worse with network size. A geometric answer degrades
differently: what limits it is diffusion distance and growth time, not the
number of stochastic units. Whether that is *better* is an empirical question,
and it is a genuinely interesting one.

It also reframes something we previously mislabelled: a reward field means
distal synapses are harder to credit. That was written down as a risk. Under
this reading it is the **mechanism** — the gradient is what makes growth
directional in the first place. Without a falloff there is nothing to climb.

### The sharpest falsifiable claim: does it grow a shortcut when earliness pays?

Not "architecture emerges" — unfalsifiable, and any distance-decaying rule
produces a brain-shaped graph by construction. Not even "delay helps." The
claim with teeth is a **contingency** claim in the structural domain, and it
is the same logic as the operant ladder in §B: a structure must appear
*because* it pays, and fail to appear when it doesn't.

- **Setup:** the same task under two reward schedules. **EARLY** pays more
  for an answer delivered inside the first N ticks; **LATE** pays the same
  amount whenever the answer arrives.
- **Claim:** organisms raised under EARLY grow measurably more long-span,
  low-hop-count input→output paths than organisms under LATE; their
  early-window accuracy is higher; and lesioning the longest *k* edges costs
  EARLY organisms more than it costs LATE organisms.
- **What kills it:** indistinguishable topology statistics under both
  schedules. Then latency is merely being *suffered* rather than exploited,
  and the whole delay premise is dead.

This is what "fast and slow thinking" has to mean if it is to be more than a
metaphor: not that two speeds exist, but that the organism **builds** the fast
path when speed is worth something.

### Staging (each gate kills the next if it fails)

- **F0 — does it wire itself at all?** Spatial substrate, growth + death,
  rent, spontaneous activity, uniform latency, ordinary global reward. Gate:
  reach the M1 gate (>80% on 3 patterns) starting from **zero** internal
  edges. If a network that must grow its own connectivity cannot learn what
  the fixed one learned in 800 trials, nothing below matters.
- **F1 — credit by geometry.** Replace broadcast reward with a diffusing
  field from a reward cortex, growth following the gradient. Gate: it still
  reaches competence, *and* grown structure is measurably oriented toward the
  reward source versus a diffusion-free control. This is the credit-assignment
  claim and the most interesting single experiment in the program.
- **F2 — the shortcut contingency** (EARLY vs LATE, above).
- **F3 — plastic latency.** Activity-dependent "myelination": edges that
  carry useful traffic get faster. Gate: beats fixed-by-span latency on the
  F2 measures.
- **F4 — describe the topology** against degree- and distance-matched nulls.
  Reported as *description*, never as the headline (see prior art below).

### Mechanism gaps — this is a new engine, not a patch

Named honestly, because this is a much larger lift than anything built so far:

1. **Cold start is the hard problem.** Zero edges means no activity, so
   nothing to be Hebbian about, so no output, so no reward, ever. Needs
   spontaneous firing to bootstrap — for which biology offers a good story
   (retinal waves: the developing visual system generates its own structured
   activity *before* it can see). The urge is the existing hook.
2. **Sparse + delayed changes the compute model.** Today's dense typed-array
   loops must become a sparse adjacency with per-edge delay lines or an event
   queue. The 24k-ticks/s browser property probably does not survive.
3. **Candidate-edge combinatorics** — N² proposals must be held down by the
   distance kernel and by proposing only between co-active pairs.
4. **Eligibility traces must span the delays.** λ=0.97 gives ~30 ticks; a
   multi-hop delayed path can easily exceed that, and credit would silently
   never reach the far end. Trace horizon becomes a coupled parameter.
5. **Reward as a *cortex*, not a broadcast.** Interesting and risky: reward
   arriving at a location (like neuromodulator release from a locus) means
   credit becomes distance-dependent, and distal synapses may be structurally
   unlearnable. Worth testing *as its own variable* against the existing
   global broadcast, not adopted silently.
6. **Stability** — growing recurrence with delays is open-problems §2 with the
   difficulty raised. Requires activity homeostasis and telemetry from day one.

### The honest weakness

"No predefined architecture" is partly an illusion: the geometry,
dimensionality, node count, diffusion constants, latency function, rent rate
and death threshold are all design — the design just moved from *the topology*
to *the generative rule*. That is a real and defensible move (it is precisely
what a genome does, per D1) but it should be stated plainly rather than sold
as having removed the designer. The honest claim is "we specify the physics
and the interfaces; the wiring is an outcome," not "nothing is designed."

Second weakness, sharper: **rent, diffusion and growth are three new free
parameter families**, and a system with enough knobs can be tuned into
producing almost any result. Every gate above therefore needs its control arm
specified *before* the parameters are tuned, or F becomes an exercise in
fitting knobs until something brain-shaped appears.

### Checked against prior art — after the design, not before it

Recorded so no claim is made in ignorance. None of this drove the design
above; see [related-work.md](./related-work.md) for the entries.

- Probabilistic formation and death of synapses under reward → **synaptic
  sampling** (Kappel, Habenschuss, Legenstein & Maass). Nearest neighbour and
  likeliest pre-emption of F0.
- Distance-decaying wiring and growing brain-like graphs from a rule → the
  **exponential distance rule**; **generative connectome models** (Vértes,
  Betzel, Akarca). This is why F4 can never be a headline: "it self-organized
  into a brain-like small-world graph" is what such a rule *does*, not a
  finding.
- Spatial embedding shaping function → **seRNN** (Achterberg, Akarca 2023) —
  but gradient-trained with fixed topology.
- Conduction delays as computation → **polychronization** (Izhikevich 2006),
  **synfire chains** (Abeles) — fixed delays on random wiring.
- Growth from minimal structure → **NEAT** (evolutionary outer loop);
  online rewiring → **DEEP R**.

**Seams that look open** (*all unverified against literature — a real search
is required before any of this is claimed*): (i) latency as a *plastic* edge
parameter tuned by use, rather than fixed by span or by construction;
(ii) credit assignment performed *by diffusion geometry*, with growth climbing
the reward gradient, so that structure formation and credit assignment are the
same mechanism; (iii) structural change gated to an offline phase in an
otherwise always-on organism. (ii) is the one to protect and pursue.

## G. Reference frames — a world you have to move around in
`raw → shaping` — needs a **different world**, not a different network. Likely experiment 003+.

*Raw idea (Javid, 2026-08-21):* a new minimal setup with a different "world"
where we can define reference frames and think about columns. Not to be solved
now — but the roadmap should be inching toward it.

### First principles: why a reference frame is meaningless in today's world

A reference frame is a way of saying *where a feature sits on a thing*. Ask
what it takes for that to be a coherent idea at all, and three requirements
fall out — none of which the current setup has:

1. **The world must be bigger than the sense.** If the whole object lands on
   the sensor at once, there is no "where on the object" to represent —
   position is already implicit in which pixel fired. Today's 8×8 glyph fills
   the entire 8×8 sense. There is literally nowhere for a reference frame to
   live. **This is the load-bearing change, and it is a change to the *world*,
   not the network.**
2. **The organism must move within it.** A frame is only useful if you
   traverse it. That means a sensor that can be *somewhere else next tick* —
   which is the motor capability already sketched in §B1, arriving here for a
   reason rather than as a toy.
3. **It must know how it moved.** Movement has to be available internally
   (efference copy), because "I felt X, then moved right, then felt Y" is what
   binds features into a frame. Without it, movement is indistinguishable from
   the world changing on its own.

And then the consequence that matters most for this project:

4. **Prediction becomes a learning signal that costs nothing.** Once the
   organism knows it moved right, it can *predict* what it is about to feel,
   and compare. That error is generated internally, is available every single
   tick, and needs no teacher. IPNN today has exactly one learning signal —
   external reward, which arrives once per trial and is sparse by
   construction (open-problems §4). A brain generates most of its own
   training signal. **This is the single largest missing mechanism in the
   project**, and reference frames are what make it natural rather than
   bolted on.

### The minimal world: a fingertip on a surface

Smallest world in which all four requirements are real:

- A **surface** larger than the sensor — say a 32×32 field carrying a few
  distinct "objects" (textured patches).
- A **sensor** that sees only a small window of it (say 5×5) — a fingertip,
  or a fovea.
- The sensor has a **position**, and moving it is an action.
- **Task:** say which object you are touching. Deliberately unsolvable from a
  single glimpse — a 5×5 window is ambiguous across objects, so the answer
  requires integrating what was felt *at several places*.

That last property is the whole point: the task is constructed so that
feature-at-location binding is the only route to competence. If a single
glimpse sufficed, a bag-of-features organism would pass and teach us nothing.

### The ladder (each rung is buildable and separately gated)

Written as *inches*, because the ask was how to approach this incrementally
rather than in one jump. Rungs 1–2 need no new mechanism at all:

1. **Bigger world, teacher-driven movement.** The sense window is moved
   *by the teacher* over a larger pattern. No motor, no efference copy. Gate:
   does accuracy require multiple glimpses — i.e. does a one-glimpse control
   fail where a multi-glimpse organism succeeds? Establishes that the world
   is genuinely bigger than the sense.
2. **Efference copy.** The movement vector is fed in as additional input
   alongside the sense. Purely additive. Gate: does having the movement
   signal improve integration over not having it? This is the cheapest
   possible test that movement information is usable at all.
3. **The organism moves itself.** Motor output controls the sensor —
   §B1 rungs 1–2 (operant, then contingency) apply directly, now on a
   capability that *matters* rather than an arbitrary rewarded action.
4. **Prediction as a second learning signal.** A predictive output that must
   forecast the next sensory window given the movement; local prediction
   error drives plasticity alongside reward. The real mechanism addition, and
   valuable independently of reference frames.
5. **Many small sensors that vote.** Several windows on the surface at once,
   each with its own partial view, converging on one answer — the columnar
   question (§D2), now with something for columns to be *about*.

### What this needs from the rest of the roadmap

- **Recurrence is a hard prerequisite.** Maintaining "where my sensor is"
  across movements is path integration, which is *state carried through
  time* — impossible in a feedforward organism. This is another reason
  output→pool feedback (abstract §3, still unbuilt) sits on the critical path.
- **Motor** — §B1.
- **Emergent modularity** — §D2 is the columns question; rung 5 gives it a
  task where modules would have a reason to exist.
- **Not blocked on §F.** The grown substrate and reference frames are
  orthogonal: rungs 1–2 could run on experiment 001's fixed architecture. The
  interesting version combines them (does a *grown* substrate develop a
  location representation?), but that is a later composition, not a
  dependency.

### Honest warning about novelty here

This is the one area where the project has serious, well-funded competition
with a decade's head start: reference frames in cortical columns *is*
Numenta's Thousand Brains program, and they have published the column model,
the grid-cell location signal, and the voting mechanism. Anything built here
starts far behind and should be framed as *learning from* that work, not
racing it. Two things might still be ours: doing it with a **grown** substrate
rather than a designed columnar one (§F), and using **prediction error plus
reward together** as two learning signals in one always-on organism. Neither
is claimable without a real literature search.

### Prior art (checked after, not designed from)

Hawkins / Numenta — Thousand Brains, cortical columns with reference frames,
grid-cell-derived location signals, voting. Moser & Moser — grid cells;
O'Keefe — place cells. O'Regan & Noë — sensorimotor contingency theory
(perception as knowledge of how sensation changes with action), the closest
philosophical statement of rung 4. Hinton — capsule networks, explicitly about
pose and reference frames in an ML setting. Friston — active inference, where
prediction error drives both perception and action. Bajcsy — active
perception. To be added to [related-work.md](./related-work.md) when this
track is shaped further; listed here so the section is not written in
ignorance.

## H. The Games — an evaluation ecosystem, not another benchmark

**Origin:** Javid, 2026-08-22 **[J]**. Raw idea; nothing built.

> "Something like the AI Hunger Games, where we put a bunch of different models
> to the test. Anybody could create a model and plug it into the games and run
> it against the challenges, competing with other people's models. It's all
> open source, open weight, and you have to show the full history of how you
> trained it."

### The shape

A growing suite of tiny games in one fixed world — 8×8 binary pixels, the same
world experiment 001 already uses:

- identify vertical lines · identify horizontal lines · identify blank
- which image has the most white squares
- which has the most *connected* white squares
- find the longest vertical line · more-vs-fewer comparisons · and onward

An entrant meets each game **never having been trained on it**. Two things are
scored, and the second is the whole point:

1. **Acquisition** — how fast and how accurately it learns the new game.
2. **Retention** — how well it still performs on *every game it has already
   entered*, now that it has been through another learning phase.

So a model's standing is not a number on one task. It is a trajectory across a
history that grows every time it competes, and a model that wins a new game by
destroying its own past is scored as having lost ground.

### Scoring — what we must not reward by accident

**Added 2026-08-22 [J].** Design notes, not decisions. Nothing is being built.

#### Speed is a constraint, not the score

> "I want to make sure I'm not only rewarding speed. Learning and ability is not
> all about speed. If you're fast within a certain limit, but not the fastest,
> that might be ok, if you're more accurate."

Ranking by speed alone has a specific failure: it rewards systems with small
hypothesis spaces that snap to an answer, and it penalises systems that are
building something more general and get there later. It also **quietly
reintroduces the exact bias [L-020](./journal/LEARNINGS.md) warns about** — the
project spent 2026-08-22 establishing that accuracy-at-fixed-trials punishes a
slow learner for being slow, and a speed leaderboard punishes it again from the
other direction. An ecosystem built to test living systems must not smuggle
that back in on day one.

Three framings that avoid it, in rough order of preference:

1. **Criterion within a budget, then rank by accuracy.** Speed becomes
   pass/fail — did you reach the standard inside N trials — and among everyone
   who passed, standing is asymptotic accuracy. Speed matters up to a
   threshold and not past it, which is exactly what was asked for.
2. **Report the Pareto frontier, not a rank.** Publish (speed, accuracy) for
   every entrant and mark who is on the frontier. Nobody is "the winner"; some
   entrants are simply not dominated. Honest, harder to headline.
3. **Two scores, never collapsed.** Trials-to-criterion *and* final accuracy,
   published side by side, with the composite left to the reader. The moment
   they are collapsed into one number, that number's weighting becomes the
   thing everyone optimises.

Open question: any budget N is arbitrary, and choosing it decides which kinds
of learner can compete at all. It should be justified against something —
human performance on the same game is the obvious anchor, as
[Animal-AI](https://link.springer.com/article/10.3758/s13428-025-02616-3) did
by testing children.

#### Back-to-back learning, and sleep as a declared condition

> "If you can adapt to learning new games in real time, back to back (maybe
> with sleep)."

This is the difference between **blocked** and **interleaved** practice, and it
is a real distinction with a real literature: interleaving is harder in the
moment and produces better retention afterwards. So "learn A, then B, then C
with no reset" is a *different and harder* regime than "learn each in
isolation", and both are worth running.

Two things follow.

- **Sleep becomes a manipulable condition rather than an implementation
  detail.** Sleep-dependent consolidation is well established in humans and
  animals, and experiment 002's substrate already has an offline phase where
  all structural change happens. So *with sleep between games* versus *without*
  is a legitimate arm, and one this project is unusually well placed to run.
- **Order effects become a confound.** If games are entered back-to-back, the
  order changes the result, and entrants who happened to get a friendly order
  would rank higher for no good reason. Any serious version has to
  counterbalance order across entrants, which multiplies the run cost.

#### Generative outputs, and why the tiny world is what makes them scorable

> "For things that aren't just discrete simple outputs, where the AI may have
> to generate an output — an image, a drawing, speech — we may have to give it
> more time to generate. One AI may quickly draw a picture but another may draw
> a much better picture. Not sure how we'd frame that game."

The difficulty is real and it is not about time: **a classification task has a
ground truth and a generation task usually does not.** "Better picture" is a
judgment, and the moment a judge is introduced, the judge is what everyone
optimises against.

But this is precisely where the 8×8 world stops being a limitation and starts
being the answer. **In a 64-pixel binary world, generation is verifiable.**
The set of valid "three vertical bars" images is finite and enumerable, so
"draw one" has a checkable answer in a way "draw a beautiful landscape" never
will. Some games that only work because the world is small:

- **Produce a valid instance of a category.** Scored against the enumerable
  ground-truth set — no judge required.
- **Produce a valid instance nobody has produced before.** Validity *and*
  novelty, both mechanically checkable. This is a genuinely hard task with no
  subjective component.
- **Complete a partial pattern** so that the intended category is recoverable.
- **Produce an instance another entrant's model misclassifies** — adversarial
  generation, scored objectively, and it makes the ecosystem's own entrants
  into the test set.

For the time problem, the same principle as above: **let the entrant declare a
generation budget and publish (budget, quality) as a pair** rather than
collapsing them. A model that draws slowly and well and a model that draws fast
and adequately are different animals, and a single number would hide that.

Risk to name up front: if quality is ever scored by a fixed discriminator,
entrants will overfit the discriminator rather than the task — the standard
adversarial-example failure. Mitigations would be rotating or ensembled judges,
held-out judges, or sticking to the enumerable-ground-truth games above, which
have no judge at all.

#### The general principle under all three

Every scoring choice here is a claim about what intelligence is, and a
leaderboard makes that claim binding on everyone who enters. Speed-only says
fast is smart. Accuracy-only says slow and careful is free. A single composite
says the weighting is settled when it is not. **The safest default for an
ecosystem that does not yet know what it is measuring is to publish the
components and refuse to collapse them** — and to treat any collapse as a
separate, explicitly argued decision.

### Why this is the right vehicle for this project specifically

It is the same claim [vision.md](./vision.md) has made since it was written —
"teach it a second task, return to the first, and it has not forgotten" — with
the scoring function made public and adversarial. And it operationalises
[H-001](./journal/HYPOTHESES.md) (intelligence is transfer and retention),
[H-011](./journal/HYPOTHESES.md) (trials-to-criterion as the currency) and
[H-017](./journal/HYPOTHESES.md) (test on games it was never trained on) in one
artefact.

### Prior art — searched 2026-08-22, and most of this exists

Recorded honestly, because three of these are close and one is very close.

| What | How close | Where it differs |
|---|---|---|
| **ARC-AGI / ARC Prize** (Chollet, *On the Measure of Intelligence*, 2019) | **Very.** Defines intelligence as *skill-acquisition efficiency on unknown tasks* — the same definition. ARC-AGI-3 (2026, $2M) is interactive: explore novel environments, acquire goals on the fly, learn continuously | Scores **first-exposure efficiency**. A search found no evidence it tests retention on environments already solved — the trajectory-across-your-own-history part is absent |
| **GVGAI** (General Video Game AI) | Close. Agents are submitted without knowing the games and evaluated on unseen ones, explicitly to stop overfitting. Has a learning-agents track | Per-competition novel game sets; no persistent per-entrant history, no retention score |
| **General Game Playing** (Stanford/AAAI) | Close in spirit — rules handed to the agent at runtime | Symbolic board games; no learning-curve or retention measurement |
| **Animal-AI Olympics / Environment** | **Strikingly aligned with [H-009](./journal/HYPOTHESES.md).** 900 configurations drawn from comparative cognition — trap-tube, string-pulling, Y-mazes, Thorndike escape boxes — run against crows, chimpanzees, octopuses and children | Someone already built "measures you could apply to an animal" as a testbed. Spatial/physical cognition in 3D, one-shot competition, no retention scoring |
| **Continual-learning benchmarks** (permuted/split MNIST, CIFAR100, TRACE, CITB, CL-VISTA, Continual Learning Bench 2026) | The retention half is **standard**: Backward Transfer and Forgetting Measure are established metrics | Offline dataset sequences, not a live open ecosystem. No public entry, no provenance requirement, and the task sequence is fixed by the benchmark author rather than grown |
| **Open-weight leaderboards** | The open-weight norm exists | They rank models on fixed benchmarks. "Open weight rarely means fully open source: most models publish weights, not training data" — the provenance requirement is the unusual part |

### The smallest honest claim

Every component exists. What a search did not turn up is the **combination**:

> a public, open-entry ecosystem whose *ranking function* is acquisition speed
> **×** retention over the entrant's own growing history, with open weights and
> **full training provenance** required to enter.

Marked *unverified against literature* — one afternoon of searching is not a
survey, and the continual-learning literature is large enough that a closer
analogue very likely exists.

### The differentiator that actually matters, and it is not novelty

**An 8×8 binary world is sized for a hand-built organism.** ARC uses grids up
to 30×30 of visual reasoning; GVGAI is real-time video games; Animal-AI is 3D
physical cognition. All three are, in practice, arenas for large pretrained
systems. A from-scratch substrate with 1,024 stochastic binary units cannot
enter any of them and do anything but lose.

A 64-pixel world is small enough that a grown substrate, a hand-coded rule
system, a tiny RL agent and a frontier model can all compete on the same board
— which makes it a place to ask *what kind of thing learns efficiently*, rather
than *which large model is best*. That is the gap worth building into, and it
is an ecosystem gap rather than a scientific one.

### Open questions before any of this is worth building

- **Provenance is hard to enforce.** "Show the full history of how you trained
  it" is checkable for a small from-scratch system and close to unfalsifiable
  for a fine-tuned frontier model. Does the rule mean anything without
  re-execution?
- **Retention scoring needs a decay policy.** Is a model re-tested on all past
  games after every new one? That cost grows quadratically.
- **Who writes new games, and how are they kept unseeable?** GVGAI's answer
  was to design a fresh set each year, which is expensive.
- **Does the composite score have a degenerate optimum** — e.g. a model that
  learns nothing new but forgets nothing?
- **Hard requirement #3 (added 2026-08-22, from L-038/L-039):** flexibility
  scoring must use task sequences that never revisit a prior state (or score
  revisits separately), and must never credit recoveries at the measurement
  floor — a rigid entrant otherwise farms the leaderboard by standing still
  while a periodic world swings back to it.
- **This is a platform, not an experiment.** It would be a large build, and it
  competes for time with experiment 003, which tests the same hypothesis
  privately and could run this week.

### Status

Backlog only. Not scheduled, not designed, no `design.md`. **Experiment 003
(transfer and retention) is the same question at 1% of the cost and should run
first** — if 001 cannot transfer between two tasks in private, there is nothing
to enter into a public arena.

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

## Current ranking (2026-08-21)

The F-track changes the picture: it is a new engine, not a patch, so it does
not simply displace the queue. But two cheap things now gate everything.

0. **M1b — the `etaPool=0` ablation.** Elevated to first. One test run. It
   answers whether the pool is load-bearing or whether all learning lives in
   the 480 output weights (a random projection with a policy readout). Nothing
   about growing a better substrate means anything until we know whether the
   current substrate does anything.
1. **Output→pool feedback** (abstract §3, design §3). The *I* in IPNN is
   still unimplemented; the 2023 hypothesis is still untested. F's delay
   claims are also unreadable on a feedforward net.
2. **F0 / G0** — does it wire itself at all, from zero edges. The entry gate
   to the whole F program; now
   [experiment 002 §8](./experiments/002-grown-substrate/design.md).
3. **F1 / G1** — credit by diffusion geometry. The most interesting
   experiment in this file.
4. A1+A2, C1, as previously ranked below.

### Superseded ranking (2026-08-20)

1. **A1 + A2 as one experiment** — "changing its mind / live coaching."
   Attacks the word *living* directly; nearly engine-ready; produces the
   baseline A3 needs; its measures become M2's most watchable content.
   Note: entry 1 pre-registered M2 as next — running this first is a
   sequencing change to record in that experiment's journal entry.
2. **A3 stage (i)** — free to run alongside A1 (same harness), settles the
   2023 claim's null baseline.
3. **C1** — cheapest test that the Beta-confidence machinery pays rent.
4. **M1b** — still the cheap standing pre-step for M2.
