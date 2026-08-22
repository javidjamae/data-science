# Experiment 002 — The Grown Substrate

**Status:** M0 built and gated; **M1 run and FAILED** (2026-08-22). M2–M4
blocked — see the status note in §8 and the amendment at the head of §10.
Everything below §8's status note is the original pre-registration, left
unedited except where an amendment is marked as such.
**Live demo:** <https://claude.ai/code/artifact/122d168d-179c-4004-96b2-102377504da6> — watch it wire itself from zero edges, and
switch arms to see the one change that makes the identical rule learn.
**Depends on:** experiment 001's engine, which becomes this experiment's
**baseline arm** rather than being replaced.
**Backlog origin:** [experiment-ideas.md §F](../../experiment-ideas.md)
(raw idea Javid 2026-08-21; shaped the same night).

## 1. The question

Experiment 001 answered "can a fixed network learn from reward alone?" — yes,
at toy scale. Its architecture, however, is designed: 160 pool neurons, each
handed 24 random synapses from the sense at birth, wired densely to 3 outputs,
and never changed again. Weights move; the *shape* never does.

This experiment asks a different question on the **identical task**:

> Given nothing but the interfaces — where the world comes in, where answers
> go out, and where reward arrives — can an organism **grow its own wiring**
> and reach the same competence?

And then the part that makes it more than a curiosity:

> If reward is a substance that spreads from a source rather than a number
> broadcast everywhere, does **where a synapse sits** become the thing that
> decides whether it gets credit — and does the organism grow toward the pay?

Same 8×8 sense, same 3 patterns, same reward-only teacher, same >80% gate.
Only the plumbing changes. That is deliberate: holding the task fixed is what
makes the comparison against 001 mean anything.

## 2. Can we reuse experiment 001's code? Yes — the surface is nine members

Measured, not assumed. Everything outside the organism touches it only
through:

| Used by | Members |
|---|---|
| `teacher.ts` (AutoTeacher, TrialStepper) | `sense`, `tick()`, `lastWinner`, `applyReward()`, `cfg` |
| `demo-m1/sim.ts` | the above, plus `poolFired`, `clearTraces()` |
| `main.ts` (UI) | `sense`, `urge`, `poolActivity()`, `outputProbs()` |
| tests | `weightNorms()` |

So the teacher, the trial state machine, the sustained readout, the demo
controller, the accuracy chart, manual mode and the whole test harness are
**organism-agnostic already**. Extract that surface as an interface and the
substrate becomes swappable:

```ts
export interface OrganismLike {
  readonly cfg: { outputSize: number; /* …substrate-specific */ }
  readonly sense: Uint8Array
  lastWinner: number
  urge: number
  tick(): void
  applyReward(r: number): void      // see §7 — meaning changes, signature doesn't
  clearTraces(): void
  poolActivity(): number
  outputProbs(): Float32Array       // becomes windowed firing rate, see §9
  weightNorms(): { pool: number; out: number }
}
```

**Reused verbatim:** `rng.ts`, `patterns.ts`, `teacher.ts`, `readout.ts`, the
M1 gate harness, the trial/accuracy plumbing, manual mode, the layout-stability
tool. **Replaced:** `organism.ts` only. **Adapted:** the pool panel in the UI
(§9) and `poolFired`, which becomes a spatial map rather than a flat array.

One honest caveat: `applyReward(r)` keeps its signature but changes meaning —
it *injects reward at the reward cortex* instead of applying it everywhere.
The teacher does not need to know that, which is the point.

### One codebase, two substrates — not a fork

Experiment 002 shares experiment 001's foundation rather than branching from
it. Both substrates live side by side in the same `src/engine/`, implementing
the same interface; the teacher, task, readout and UI are literally the same
objects, and the demo picks a substrate at runtime.

This is not a convenience. **Experiment 001 is 002's control arm** (§4), so if
001's code drifted, the comparison would quietly become meaningless — and a
fork guarantees drift. Shared code means "the same teacher, the same three
patterns, the same 20-tick/6-fire readout" is enforced by the compiler rather
than by good intentions.

Consequence worth stating: the shared app physically lives at
`experiments/001-mnist-living-demo/app/` for historical reasons. It is now the
**shared lab**, not 001's private code. It is deliberately *not* being moved:
the journal is append-only and its entries cite those paths, and breaking the
record to tidy a directory name is a bad trade. New substrates get namespaced
subdirectories (`engine/grown/`), the same convention already used for
per-milestone demos (`demo-m1/`).

## 3. The substrate

A 2D sheet of nodes with real positions. 2D rather than 3D for v1 purely
because it is drawable, and being able to watch the wiring grow is worth a
lot here.

- **Lattice:** 32×32 = 1024 node sites (a knob; see §10 scale).
- **Input cortex:** 64 nodes in an 8×8 block, one per sense pixel, placed on
  one side of the sheet. Their firing is clamped by the sense.
- **Output cortex:** 3 nodes placed on the far side, with lateral inhibition
  among them (§6). "Spoken" is decided exactly as in 001 — ≥6 fires in a
  20-tick window — by the *teacher*, not the organism.
- **Reward cortex:** a small locus (1–4 nodes), placed off the input→output
  axis so that "grow toward reward" and "grow toward the output" are not the
  same instruction by construction. **Its position is an experimental
  variable, not a constant.**
- **Interior:** every other site. **Zero edges at t=0.** Nothing is wired.

Input and output are placed far apart on purpose: a path has to be *built*
across the sheet, and path length is then a real, measurable quantity.

## 4. Two fields — the core of the design

Everything interesting follows from having two diffusing chemical signals
instead of one broadcast scalar. Both are scalar fields over the lattice,
updated each tick by ordinary discrete diffusion (`F ← F + D∇²F − decay·F +
sources`), which is a handful of array operations per tick.

**Activity field `A`** — *where to build.*
Emitted by any node that fires. Diffuses a short distance, decays fast.
Growth cones climb it. Biological reading: activity-dependent neurotrophic
signalling (BDNF and kin) — axons grow toward active tissue.

**Reward field `R`** — *what to keep.*
Emitted by the reward cortex when the teacher delivers reward. Diffuses
further and decays more slowly than `A`. It is the third factor of the
learning rule, read **locally at each synapse** (§7). Biological reading:
neuromodulator volume transmission from a locus.

The division of labour is the whole idea in one line: **activity says where to
build, reward says what to keep.**

### Why this is the interesting part

The existing rule is `Δw = η · R_global · e`. The new rule is
`Δw = η · R(x) · e` — the same three-factor form with the third factor read
from a position. That one substitution means a synapse is credited **because
of where it is**, and it is where it is because growth climbed a field to put
it there. Structure formation and credit assignment stop being two mechanisms.

### The control arm is free, and it is experiment 001

Set `R`'s diffusion to infinite (a flag: uniform field) and `R(x)` is the same
everywhere — which is *exactly* experiment 001's broadcast rule. The null
hypothesis is therefore the previous experiment, running in the same engine,
differing by one parameter. That is an unusually clean control and it should
be run from day one, not bolted on later.

## 5. Edges, and time of flight

An edge carries: presynaptic node, postsynaptic node, weight `w`, eligibility
trace `e`, evidence count `n`, and **latency `d` in ticks**.

`d = max(1, ceil(span / v))` with `v > 1`, so a long edge is *temporally
cheaper than the chain of short hops it replaces*: with `v = 3`, one span-3
edge delivers in 1 tick where three unit hops take 3. Long-range edges are the
fast path; local chains are the slow path. This is why the brain myelinates
long tracts — each synaptic relay costs real time.

**Implementation.** Each node owns a ring buffer of future input, depth
`maxDelay + 1`. When node *i* fires at tick *t*, for each outgoing edge it
adds `w` into `inbuf[j][(t + d) mod D]`. At tick *t* node *j* reads and clears
`inbuf[j][t mod D]`. Cost is O(edges) per tick, no event queue needed.

> **Correctness trap, called out because it will silently ruin everything.**
> With delays, "pre and post were co-active" means the presynaptic spike
> *arrived* at *t* having been emitted at *t − d*. Eligibility must be updated
> against the **arrival**, not against the presynaptic node's current state.
> Deposit an arrival flag alongside the weight in the same ring buffer and
> update `e` from that. Getting this wrong assigns credit to the wrong pairs
> and would look like "growth just doesn't work."

Corollary: the eligibility horizon must exceed realistic path delays. λ=0.97
gives ~30 ticks; a deep multi-hop path can exceed that, and credit would
silently never reach the far end. **λ and maximum path depth are coupled** and
must be reported together.

## 6. Nodes

Deliberately close to 001's neuron, so that differences in outcome are
attributable to the plumbing rather than to a new unit model.

- Stochastic binary firing: `p = σ(gain · (drive − inhibition) + bias)`.
- Global homeostatic inhibition holding mean activity near a target sparsity —
  kept because it is the stability lever we already trust.
- **Spontaneous firing** at rate `p_spont`, so nodes fire with no input at
  all. This is not a cold-start hack: the developing visual system wires
  itself on self-generated retinal waves *before* the eyes work. It is the
  bootstrap that makes growth possible at t=0 (§10 risk 1).
- **Lateral inhibition inside the output cortex** so the three outputs
  compete. 001 got competition free from a softmax; this substrate has no
  softmax, so competition must be structural.
- **Urge** as in 001: excitability of output-cortex nodes rises during
  silence, resets on firing.

**Dropped from 001: the output softmax.** Output nodes are ordinary nodes;
what makes an answer is sustained rate, read by the teacher. This is both
more biological and removes the mechanism behind L-002 — but it removes the
protection that mechanism's fix provided too, so output collapse must be
watched for directly (§10 risk 4).

## 7. Learning, rent, growth, death

**Weight update (wake).** On reward, at each edge:

```
Δw = η · R(x_post) · e / (1 + n/n₀)
```

`R(x_post)` is the reward field sampled at the postsynaptic node. Everything
else is 001's rule, including Beta-confidence consolidation.

> Carried-forward defect, deliberately not fixed here: `n` only ever
> increases, so a synapse can never become *less* confident (see the
> 2026-08-21 discussion and abstract §3's "Beta-flavored" hedge). Fixing it
> means tracking α and β separately. **Not** changed in this experiment,
> because changing two things at once would make the comparison against 001
> unreadable. Registered as the first follow-up.

**Rent (wake).** Every edge pays `ρ` per tick as weight decay toward zero.
Pruning is therefore not a maintenance pass — it is *failure to pay*.
Metabolic cost becomes the selection pressure. An edge that stops earning
dies on its own.

**Sleep.** All structural change is gated to an offline phase (every `K`
trials, the sense goes dark and the organism runs `S` ticks of sleep). Two
reasons, one biological and one engineering: brains restructure offline, and
never rewiring mid-thought removes the instability that has blocked
recurrence since open-problems §2. During sleep:

- **Growth.** Each node gets a small budget of growth attempts. A target is
  sampled from sites within `r_max`, weighted by `A(target) · exp(−span/λ_g)`
  — climbing the activity field, with distance discouraged but not forbidden,
  so long jumps are rare rather than impossible. New edge starts at small `w`,
  `n = 0`, `d` from its span.
- **Death.** Edges with `|w| < θ_death` are removed.

Wake computes; sleep rebuilds. No replay in v1 — that is a later variable.

## 8. Milestones and pre-registered gates

Milestone numbering is per-experiment (see the naming section in
[README.md](../../README.md)); these are experiment 002's M0–M4, and they are
unrelated to experiment 001's.

> **Status 2026-08-22.** **M0 — met** (001 bit-identical, 88 tests, `tsc` and
> build clean). **M1 — FAILED**: flat at chance across 3 seeds × 2,000 trials,
> despite growing structure that connects input to output in 4–6 hops and
> earning reward within 200 ticks. The gate's own clause applies —
> "if M1 fails, everything below is moot" — so **M2, M3 and M4 are blocked**
> until depth is fixed or explicitly abandoned. Diagnosis, the shallow control
> arm (0.883, which is *not* a substitute gate), and the M1c/M1d/M1e
> follow-ups are in journal
> [2026-08-22-0208](../../journal/entries/2026-08-22-0208-exp002-m0-built-m1-fails-on-depth.md).

**M0 — scaffold.** Extract the `OrganismLike` interface (§2) from experiment
001 without changing its behavior — all 17 existing tests must pass unchanged
— then build the substrate, fields, ring-buffer delays, growth and rent, with
headless tests for each. No gate beyond "001 still works, bit-identical."

**M1 — it wires itself at all. The critical gate.** Uniform reward field
(= 001's broadcast), uniform latency (all `d = 1`), rent and growth on, from
zero edges.
**Gate:** ≥0.80 rolling accuracy over the last 100 of 2,000 trials, on ≥3
seeds, *and* a connected input→output path exists at the end.
**If M1 fails, everything below is moot** and the finding is "reward-driven
growth from zero does not reach a competence a fixed random projection reaches
in 800 trials" — a real negative, and the journal's rule 5 says it gets
written up with the same rigor as a success.

**M2 — credit by geometry.** Turn on the diffusing reward field.
**Gate:** (a) competence is retained, and (b) grown structure is measurably
oriented toward the reward source versus the uniform-field control —
pre-registered measure: mean edge-density as a function of distance from the
reward cortex, against the uniform-field arm, with reward-cortex position
counterbalanced across ≥3 placements so the effect cannot be an artifact of
one geometry.
This is the most interesting experiment in the program.

**M3 — the shortcut contingency.** Turn on span-dependent latency. Two reward
schedules on the same task: **EARLY** pays more for answers inside the first
N ticks; **LATE** pays the same whenever the answer arrives.
**Gate:** EARLY organisms grow more long-span, low-hop input→output paths than
LATE; early-window accuracy is higher; lesioning the longest *k* edges costs
EARLY more than LATE. **Kill condition:** indistinguishable topology under
both schedules — latency is then merely suffered, not exploited, and the delay
premise is dead.

**M4 — plastic latency.** Activity-dependent "myelination": edges carrying
useful traffic get faster. **Gate:** beats fixed-by-span latency on M3's
measures.

**Control arms, all pre-registered now, before any parameter is tuned:**
uniform-field (= 001's rule); no-rent (ρ=0, no death); uniform-latency;
no-spontaneous-activity (expected to fail G0 outright — it is the cold-start
control); and **experiment 001 itself** as the fixed-architecture baseline.

**Standing measurements** for every arm: edge count over time, deaths over
time, input→output path length distribution, time-to-first-reward, mean
activity and its variance (stability telemetry, open-problems §2).

## 9. What the UI becomes

Mostly free, and better. The sense panel, output register, accuracy chart,
speed/learning controls and manual mode all work unchanged against the
interface in §2. The one panel that changes is the pool raster — 001's
arbitrary 16×10 grid of 160 neurons becomes a **map with real geometry**:
nodes at their positions, grown edges drawn as lines, edge opacity by weight,
and the reward field as a heat overlay. Watching wiring appear and die, and
watching the reward field pulse out from its locus, is the demo.

`outputProbs()` has no softmax to report, so it returns each output's
windowed firing rate; the bars keep their meaning ("how strongly is it saying
this") and the UI needs no structural change.

## 10. Risks and honest expectations

> **Amended 2026-08-22, after M1 ran** (journal
> [2026-08-22-0208](../../journal/entries/2026-08-22-0208-exp002-m0-built-m1-fails-on-depth.md)).
> **This ranking was wrong, and the original order is left below rather than
> rewritten so the miss stays visible.** Risk 1 (cold start) never happened:
> spontaneous firing and urge bootstrapped from zero edges on every seed, with
> the first reward inside 200 ticks, and not one rung of its mitigation ladder
> was needed for its stated purpose. Risk 5 (credit may not reach far
> synapses) was fatal, and for a reason risk 5 does not actually name — the
> problem is not that *credit* fails to reach distant synapses, it is that
> *information* fails to reach them. Stimulus dependence survives exactly one
> hop (7.7σ → 1.7σ), so the distal synapses have nothing worth crediting in
> the first place. Risk 3 (performance) was comfortably clear: 6,900–8,500
> ticks/s against a ≥1,000 target. Risk 4 (output collapse without the
> softmax) did not occur: lateral inhibition held ambiguous ticks to 2.3–2.6%.
> Risk 2 (free parameters) stands, undiminished.
>
> The corrected ranking, for anyone reading this before the next attempt:
> **1. information does not survive a hop. 2. free parameters.
> 3. the readout is geometrically starved** (new, L-014: a 3-site output
> cortex received 19 of 5,258 edges, because growth has no target-attractiveness
> term). Everything else is noise by comparison.
>
> **Two corrections to this document's own machinery, both load-bearing.**
> (a) §6 credits spontaneous firing as "the bootstrap that makes growth
> possible at t=0". It is not the main one — the homeostat is, and it will
> revive a substrate from total silence on its own (L-016). The
> **no-spontaneous-activity control arm in §8 is therefore invalid as
> written**: removing `pSpont` alone leaves the organism wiring up normally.
> It must also disable the homeostat (`inhibitionRate: 0`) and bury the
> resting bias, and with those it fails outright exactly as §8 predicted.
> (b) §8's standing measurement "input→output path length distribution" means
> the distribution **over every input site**, not the shortest path over the
> cortex. The two disagree badly — an arm with a 1-hop shortest path had a
> median sense pixel 3 hops out — and reading the minimum as the depth
> produced a wrong claim once already (L-017).

1. **Cold start is the likely failure mode.** Zero edges → no activity → no
   output → no reward → forever. Spontaneous firing is the designed answer,
   but whether it produces enough structured activity to bootstrap is exactly
   what G0 tests. Pre-registered mitigation ladder, in order: raise
   `p_spont`; add a developmental schedule (high `p_spont` early, decaying);
   seed a sparse random scaffold at t=0 (**note: this weakens the "from zero"
   claim and must be reported as such**); shorten the input→output distance.
2. **Free parameters are the real threat to this experiment's credibility.**
   Rent, two diffusion constants, growth rate, `r_max`, `λ_g`, `θ_death`,
   `p_spont`, `v`, sleep cadence. A system with this many knobs can be tuned
   into producing almost any result. This is why every gate above has its
   control arm fixed *before* tuning. Any parameter search must be reported,
   including the arms that were tried and discarded.
3. **Performance.** O(edges) per tick with a ring buffer is cheap, but edge
   count is unbounded by construction; rent is the only thing holding it down.
   Cap edges per node and report when the cap binds (no silent truncation).
   The 24k ticks/s of 001 will not survive; the target is ≥1k ticks/s.
4. **Output collapse without the softmax.** Removing the softmax removes
   L-002's failure mode *and* its fix. Lateral inhibition is the replacement;
   watch for all-three-firing (no clean answer) and for one-output domination
   (L-006's signature: accuracy pinned exactly at chance).
5. **Credit may not reach far synapses.** Deep paths plus a decaying reward
   field means distal edges may be structurally unlearnable. That is the
   mechanism, not a bug — but if it is *too* strong, only a thin shell near
   the reward cortex ever learns. The distance-to-reward edge-density measure
   in G1 detects this either way.
6. **This is a new engine.** Realistically a substantial build, not an
   evening. Staging it behind the §2 interface means 001 keeps working
   throughout and the two can be compared in the same UI.

## 11. Out of scope

Multi-sense and multimodal fusion (moves to 003+); neurogenesis (nodes are
fixed sites here — only *edges* are born and die); dendritic branch targeting
(§F principle 7 — real, and deferred: it multiplies the state per node);
replay during sleep; the α/β confidence fix; MNIST (the task stays the three
M1 patterns on purpose).

## 12. Relationship to the rest of the project

- **Experiment 001 is not superseded.** It becomes the fixed-architecture
  baseline arm, and its published demo stays as-is.
- **Supersedes in ambition:** experiment-ideas §C1–C3 (pruning, growth) and
  §D1–D3 (designed vs emergent structure) — rent-and-death subsumes pruning,
  and this is D taken to its limit.
- **Still owed first, and cheap:** **M1b** (`etaPool=0` ablation). It tells us
  whether 001's pool is load-bearing or whether all its learning lives in the
  480 output weights. Since 001 is now this experiment's baseline arm, M1b is
  not a detour — it is baseline characterization, and without it a G0 result
  has nothing trustworthy to be compared against.
