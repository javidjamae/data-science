# Manual mode: holding a stimulus and watching the answer run free

- **Entry:** `2026-08-21-0025-manual-mode-sustained-readout`
- **When:** 2026-08-21 00:25–00:50 CDT
- **Who:** Javid + Claude (session)
- **Experiment:** [001 — MNIST living demo](../../experiments/001-mnist-living-demo/design.md) (infrastructure, plus one unregistered observation)
- **Code state:** git `f220847` (committed 2026-08-21 02:02 CDT; results in this entry were produced by the code as committed there)
- **Re-run:** `cd ipnn/experiments/001-mnist-living-demo/app && npm test` ·
  `npm run build:single` ·
  `PLAYWRIGHT_PKG=<path> node tools/measure-jitter.mjs dist-single/ipnn-m1-demo.html "" 1200 manual`

## In plain words

Until now the demo only had one mode: the teacher runs the lesson, flashing
patterns and rewarding right answers, and you watch. Javid asked for the
obvious other thing — let *me* pick what it looks at, or show it nothing, and
watch what it does.

That turned out to need a new instrument, not just a button. The existing code
asks "what did it answer?" once per lesson and then stops listening, which
means it structurally cannot see the organism change its mind. So the answer
readout had to be rebuilt as something that never stops listening.

The first thing it showed is worth the price of admission. Put a pattern it
knows in front of it and it locks on — one answer, held steadily, for
hundreds of ticks at a stretch. Show it a **blank** screen and it babbles:
cycling through answers every few ticks, hundreds of changes of mind, never
settling. It is restless when there is nothing to see. That restlessness is
the "urge" we built in to stop it sitting silent forever — and this is the
first time anyone has actually watched it work.

## Objective

Add a mode where a human chooses the stimulus (or clears it) and observes the
organism's response, and determine what instrument that requires.

## Gate (pre-registered)

None — a feature request raised in conversation, handled as infrastructure
(same process note as entries
[2026-08-20-0856](./2026-08-20-0856-m1-living-demo-ui.md) and
[2026-08-21-0008](./2026-08-21-0008-demo-layout-jitter-fix.md)). Operating
gate, stated before building: *manual mode must be provably incapable of
modifying the organism, the existing 7 tests must pass unchanged, and layout
stability (L-008) must hold in the new mode.*

## Hypotheses

- **H1:** `TrialStepper` cannot serve as the readout for sustained exposure —
  it commits to the first spoken answer and ends the trial, so it can observe
  at most one answer per exposure.
- **H2:** manual mode cannot change the organism, because `applyReward` is the
  only mutator and nothing calls it without a teacher.
- **H3:** a trained organism under sustained exposure to a learned pattern
  spends most of the exposure on the correct answer.

## Method

- **`engine/readout.ts` — `SustainedReadout` (new).** Same "spoken"
  convention as the teacher (≥6 fires in a 20-tick sliding window), but
  free-running: the answer can form, dissolve to silence, and re-form
  indefinitely. Tracks dwell, switches, per-answer occupancy, dwell-episode
  lengths, and `revisions` — transitions from one spoken answer to a
  *different* spoken answer, ignoring silence between them. Includes
  hysteresis: an answer is claimed when it crosses the threshold and released
  only when it falls back under, because an answer sitting exactly at
  threshold would otherwise flicker every tick and every flicker would be
  miscounted as a change of mind. It lives in `engine/` rather than the demo
  because headless §A1 runs need it with no UI attached.
- **`engine/organism.ts`:** added `clearTraces()` (zero eligibility traces;
  weights and evidence counts untouched) and `weightNorms()` (squared L2 norm
  per weight population — the instrument for "did anything actually move?",
  and for the M4c reward-withdrawal drift probe).
- **`demo-m1/sim.ts`:** `mode: 'auto' | 'manual'`, `setMode`,
  `setManualStimulus(label | null)`. Manual ticking runs the organism and the
  readout only — no trial machinery, no judgment, no reward. Traces are
  cleared on every mode switch, so credit accrued during a free-run cannot be
  paid out by the next lesson's reward.
- **UI:** Auto/Manual segmented control; a stimulus bar with the three
  patterns drawn as glyph chips in their series colors plus "nothing (clear)";
  keyboard `m` / `1,2,3` / `0`. In manual mode the learning checkbox is
  disabled (inert without a teacher — said rather than silently ignored), the
  reward lamp is relabelled "none in manual", the accuracy chart is marked
  paused, and the stats strip switches to the exposure readout: what it is
  saying, for how long, changes of mind, agreement, silence. Entering manual
  drops the speed to ~120 ticks/s and restores the previous speed on exit —
  at 5k ticks/s an answer forms and dies between two frames.
- **Measurement:** 10 new headless tests; a headless probe over seeds 1–2 ×
  {each pattern, cleared} at 3,000 ticks per exposure; a scripted browser run
  driving the real UI; the L-008 jitter harness extended with a `manual` mode
  pass.
- **Configuration snapshot:** unchanged from the M1 gate (organism `seed 1..3,
  senseSize 64, poolSize 160, outputSize 3, poolFanIn 24, poolGain 2.0,
  poolBias −1.0, targetPoolSparsity 0.15, inhibitionRate 0.02, outputGain 1.5,
  epsilonExplore 0.05, silenceBias 0.5, urgeRate 0.05, urgeMax 3.0,
  traceDecay 0.97, etaOut 0.08, etaPool 0.01, wMax 3.0, consolidation true,
  consolidationN0 1000`; readout `window 20, threshold 6`, inherited from
  teacher config).

## Results

- **17/17 tests pass** (was 7/7; the original 7 unchanged). Includes:
  weight norms **bit-identical** (`toBe`, not `toBeCloseTo`) across manual
  exposures totalling 7,500 ticks; a 5,000-tick manual exposure to the *wrong*
  pattern followed by auto mode still scoring ≥0.8 over the next 200 trials.
- **Headless probe**, trained organisms (accuracy 0.98), 3,000 ticks per
  exposure, occupancy of the shown pattern / revisions / mean dwell:

  | shown | seed 1 | seed 2 |
  |---|---|---|
  | vertical bars | 87% · 40 rev · 16.0t | 97% · 26 rev · 50.8t |
  | horizontal bars | 94% · 36 rev · 32.1t | 99% · 22 rev · 77.3t |
  | diagonal X | 98% · 16 rev · 77.6t | 95% · 84 rev · 20.1t |
  | **cleared (blank)** | **13/24/24%, 39% silent · 254 rev · 5.8t** | **27/17/19%, 37% silent · 282 rev · 5.4t** |

- **Browser run:** mode switch, stimulus selection, disabled learning control,
  paused-chart label and speed change all behave; no page errors. Observed
  live: a 213-tick unbroken dwell on "diagonal X" while held, versus 40
  changes of mind over a comparable span once cleared.
- **Layout stability:** 0.0000px worst movement in manual mode as well as
  auto — L-008 holds.

## Analysis

H1 confirmed by construction: `TrialStepper.judge()` runs exactly once and
ends the exposure, so no amount of configuration would let it report a second
answer. This is not a defect — a teacher running a lesson *should* commit —
but it means the trial abstraction and the dynamics question are structurally
incompatible, and §A1 was never going to be measurable without this piece.

H2 confirmed at the strongest available standard: bit-identical weight norms,
which is what "observation mode" has to mean if the demo is going to be used
to inspect a trained organism.

H3 confirmed — 87–99% occupancy on the shown pattern across 6 seed×pattern
combinations.

The cleared-sense behavior is the interesting part, and it is **an anecdote,
not a result** — per [open-problems §8](../../open-problems.md), it was
noticed while building rather than predicted in advance, and it gets believed
only after a pre-registered replication. Stated carefully: with a blank sense,
occupancy spread across all three answers with ~38% silence, ~270 revisions
per 3,000 ticks and mean dwells of ~5.5 ticks, against ~16–78 tick dwells and
~20–40 revisions under a learned pattern. The obvious reading is that the urge
(open-problems §4) forces output when there is nothing to see, and with no
input to favor one answer the ε-mixed policy wanders — restlessness with
nothing to settle on. That reading is untested; the alternative that this is
just the readout threshold behaving badly near-chance has not been excluded.

What it does do is sharpen §A1. The A1 gate was written around morph stimuli
between two learned glyphs, on the expectation that a feedforward organism
might show only window-noise revision. These numbers say there is a large
dynamic range in revision statistics *already* — roughly a 10× spread in mean
dwell between blank and learned — so the blank and clean-pattern conditions
are worth pre-registering as the two anchor points of A1's morph continuum
rather than being left implicit. Note also that these dwell numbers come from
the *urge*, which the morph design never mentions; a confound to control.

## Prior art & novelty

- **Similar:** holding a constant stimulus and recording an alternating
  percept is the binocular-rivalry paradigm (Levelt 1965; Blake & Logothetis
  2002), added to [related-work.md](../../related-work.md) this iteration.
  Dwell-time distributions under constant input are their established measure,
  and A1 is a rediscovery of that paradigm. The free-running windowed vote is
  a crude bound-crossing rule of the sequential-sampling family (Ratcliff
  1978), also added there.
- **Different:** nothing yet. This iteration built an instrument and ran an
  unregistered probe with it.
- **Novel (claimed):** none. Explicitly no novelty claim is made here.

## Learnings

- **L-009:** A commit-once readout cannot measure a mind changing. The
  teacher's "spoken" rule ends the exposure at the first threshold crossing,
  which makes revision unobservable *by construction* — so the temporal-dynamics
  track (§A1–A4) needed a free-running readout before it needed any new
  mechanism. Generalizes: when a measurement question is about a process, check
  first whether the existing instrument can represent more than one outcome per
  trial. *Evidence:* `SustainedReadout` was required to observe any revision at
  all; the same organism under `TrialStepper` reports exactly one answer per
  exposure regardless of how long the exposure runs.

## Decisions

1. **The readout lives in `engine/`, not in the demo.** It is experiment
   apparatus, not UI: headless §A1 runs will use it with no browser involved.
2. **Manual mode delivers no reward, deliberately.** Reward-by-hand is live
   coaching — experiment-ideas §A2 — and it would make manual mode capable of
   modifying the organism, which is exactly what the "cannot change it" test
   forbids today. When A2 is designed, that capability arrives with a
   pre-registered gate rather than as a convenience button.
3. **Traces are cleared on every mode switch.** Cross-contamination between a
   free-run and a lesson would be a silent correctness bug in any future
   experiment that mixes them.
4. Docs updated: `related-work.md` (rivalry, sequential sampling);
   `design.md` status line. No change to `abstract.md` or `open-problems.md` —
   the anecdote does not license one.

## Deviations

None from the operating gate. Scope grew by one engine file beyond "add a
button", which H1 justifies.

## Threats to validity

1. **The probe is unregistered and n=2 seeds.** Everything in the
   cleared-sense row is anecdote; §8 discipline applies and no claim rests on
   it.
2. **Revision counts are threshold artifacts as much as dynamics.** `window`
   and `threshold` (20/6) were chosen for the teacher's purposes, and the
   hysteresis rule is my invention, not a derived one. A different threshold
   would produce different revision counts from identical organism behavior —
   so cross-condition comparisons are meaningful, absolute numbers are not.
3. **The urge is an uncontrolled variable** in every number above; blank-sense
   restlessness may be entirely urge-driven and say nothing about perceptual
   dynamics.
4. **Manual mode's "no learning" guarantee is tested, not proven** — it rests
   on `applyReward` being the sole mutator, which is true today and would be
   silently violated by any future mechanism that writes weights elsewhere.
   The weight-norm test would catch it.
5. Mode switching mid-exposure is untested for interactions with the paused
   chart's shading marks.

## Next

Unchanged and now better equipped: design **experiment 002 — changing its
mind / live coaching** (§A1–A2) with pre-registered gates. Two inputs this
iteration adds: pre-register blank and clean-pattern as A1's anchor
conditions with the urge controlled, and note that A2's continuous-mode
teacher now has both seams it needs — `TrialStepper` for tick-granular reward
and `SustainedReadout` for the contingent-on-current-answer trigger.
