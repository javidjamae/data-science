# The demo shook: instrument panels that resized themselves

- **Entry:** `2026-08-21-0008-demo-layout-jitter-fix`
- **When:** 2026-08-21 00:08–00:45 CDT
- **Who:** Javid + Claude (session)
- **Experiment:** [001 — MNIST living demo](../../experiments/001-mnist-living-demo/design.md) (infrastructure: UI defect fix, no science)
- **Code state:** uncommitted at time of writing — hash of next commit: _fill in after committing_
- **Re-run:** `cd ipnn/experiments/001-mnist-living-demo/app && npm test` (tests) ·
  `npm run build:single` (publishable bundle) ·
  `PLAYWRIGHT_PKG=<path to an installed playwright/index.mjs> node tools/measure-jitter.mjs dist-single/ipnn-m1-demo.html [speed 0-100] [viewport width]`
  (layout-stability check; exits non-zero if anything moved)

## In plain words

Javid opened the live demo and the page visibly shook while the organism ran.
Nothing was wrong with the organism — the *page* was the problem. The first
two cards were sized to fit their own captions, and those captions change
constantly as the organism works ("showing: horizontal bars" → "blank
(between stimuli)", "sparsity 9%" → "sparsity 15%"). Every time the text
changed length, the card resized, which shoved the card next to it sideways.
So the instrument panel was jiggling in time with the thing it was supposed to
be steadily measuring.

The fix was to size those cards by the screens they hold rather than by the
words underneath them. Measured in a real browser: the cards used to move up
to 6.2 pixels while running, and now move zero.

## Objective

Remove the visible layout jitter from the published M1 demo, and find out
whether it was a rendering defect or a symptom of something in the engine.

## Gate (pre-registered)

No formal gate — a UI defect report, handled as infrastructure (same process
note as the [2026-08-20 entry](./2026-08-20-0856-m1-living-demo-ui.md)). That
entry did, however, pre-register this class of finding as *Threats to
validity #1*, quoted verbatim:

> **Interactive behavior is undertested.** Headless screenshots cover ~30
> trials of rendering and two themes; the full 800-trial run, speed extremes,
> resize, and hover were reasoned about, not exercised by a human. First real
> session may surface UI defects (not science defects — those paths are
> tested).

Operating gate, stated before the fix: *measured layout movement of every
page element must reach zero while the organism runs, at both speed extremes
and at the mobile breakpoint, with the 7 existing tests passing unchanged.*

## Hypotheses

- **H1:** the jitter is pure CSS layout — the `.panels` grid's first two
  tracks are `auto`, so they size to their widest content, and the widest
  content is a caption whose text length changes every trial phase.
- **H2:** no engine or simulation code is involved; the organism's behavior is
  identical before and after.
- **H3:** the third panel moves *more* than the first two, because it sits in
  the `1fr` track and absorbs the sum of both neighbors' changes.

## Method

- **Changes** (all in `src/main.ts`, rendering only):
  1. `.panels` tracks `auto auto 1fr` → `var(--panel-w) var(--panel-w) 1fr`.
     `--panel-w` is set once at startup by `lockPanelWidth()`, which reads the
     canvas width and the panel's *computed* padding and border, so the value
     cannot drift from the CSS that declares them.
  2. Captions restructured into two `display: block` spans (label / value)
     with `min-height: 2.9em` reserved, so a 1↔2 line wrap can never shift the
     row vertically the way content-sized tracks shifted it horizontally.
  3. Fixed-width value boxes for things that change digit count: pool
     sparsity padded to 2 chars, `#speedlbl` given `min-width: 12ch`, and each
     stats value given a `--w` reservation. `tabular-nums` equalizes digit
     *widths*; it does nothing about digit *counts*.
- **New:** `tools/build-single.mjs` + `npm run build:single`. The published
  single-file bundle was previously built ad hoc from the command line, so the
  hosted page could not be reproduced from the repo. It now can, in two forms:
  a standalone `.html` and an artifact-body variant (the host supplies its own
  document skeleton).
- **Measurement:** `tools/measure-jitter.mjs`, a Playwright harness that samples
  `getBoundingClientRect()` for 11 elements across 240 consecutive animation
  frames and reports peak-to-peak spread of x/y/width/height for each,
  failing above a 0.01px sub-pixel tolerance. Playwright is intentionally not
  a devDependency (browser binaries, occasional check); the script resolves it
  from `PLAYWRIGHT_PKG`. Run
  against two bundles built from the same source tree — `HEAD` (before) and
  the working tree (after) — at the default speed, at both speed-slider
  extremes, and at a 420px viewport.
- **Configuration snapshot:** unchanged from the M1 gate — organism
  `seed 1, senseSize 64, poolSize 160, outputSize 3, poolFanIn 24,
  poolGain 2.0, poolBias −1.0, targetPoolSparsity 0.15, inhibitionRate 0.02,
  outputGain 1.5, epsilonExplore 0.05, silenceBias 0.5, urgeRate 0.05,
  urgeMax 3.0, traceDecay 0.97, etaOut 0.08, etaPool 0.01, wMax 3.0,
  consolidation true, consolidationN0 1000`; teacher `maxTicks 60,
  blankTicks 15, spokenWindow 20, spokenThreshold 6, schedule 'ignore',
  rewardMagnitude 1.0, correctionMagnitude 0.2, baselineRate 0.05`.

## Results

Peak-to-peak movement over 240 frames, 1200px viewport, default speed
(~242 trials elapsed):

| element | before Δx | before Δw | after Δx | after Δw |
|---|---|---|---|---|
| panel: sense | 0.0 | 3.1 | 0.0 | 0.0 |
| panel: pool | 3.1 | 3.1 | 0.0 | 0.0 |
| panel: output register | 6.2 | 6.2 | 0.0 | 0.0 |
| pool canvas | 3.1 | 0.0 | 0.0 | 0.0 |
| outrows | 6.2 | 6.2 | 0.0 | 0.0 |
| chart panel, stats, controls, buttons, inputs | 0.0 | 0.0 | 0.0 | 0.0 |

- Before, worst movement across all elements: **6.2px**. After: **0.0000px**.
- After, speed slider 0 (28 trials elapsed) and slider 100 (3,471 trials
  elapsed): worst movement **0.0000px** in both.
- At a 420px viewport, *both* builds measure 0.0000px.
- Tests: 7/7 passing, unchanged. `tsc --noEmit` clean.
- Frozen-retention printout unchanged: `frozen accuracy over 100 unrewarded
  trials: 0.97`.
- Bundle 23.8 KB standalone / 23.6 KB artifact body.

## Analysis

H1 confirmed and is the whole story: two `auto` grid tracks sized to captions
that rewrite themselves every trial phase. H3 confirmed with an exact
arithmetic signature — the output panel moved 6.2px, precisely the sum of the
two 3.1px neighbors it inherits through the `1fr` track. That additivity is
what rules out a rendering or timing cause and pins it on track sizing.

H2 confirmed: the diff touches rendering only, the 7 tests pass unchanged, and
the frozen-retention number is identical. No organism or teacher behavior was
altered.

The mobile result is worth stating because it looks like a null but explains
the bug: at 420px the grid collapses to a single `1fr` column, so tracks stop
being content-sized and the jitter disappears **in the unfixed build too**.
The defect only ever existed in the multi-column layout. Anyone who had
checked the demo on a phone would have found nothing.

The honest process finding is that the previous entry *predicted* this and
shipped anyway, correctly: it named interactive behavior as undertested,
scoped the risk to "UI defects, not science defects," and that is exactly what
arrived. The pre-registration did its job — it made a surprise into a
confirmation, and it kept the diagnosis cheap because the science paths were
already known-good and could be ruled out in one test run.

## Prior art & novelty

Nothing novel — web engineering. This is textbook layout instability, the
thing Google's Core Web Vitals measures as Cumulative Layout Shift, and the
standard remedy is the one applied here: reserve space for content whose size
varies rather than letting it size its own container. Not added to
[related-work.md](../../related-work.md), which maps the scientific lineage;
front-end technique does not belong there.

## Learnings

- **L-008:** An instrument must not be dimensioned by its own readings. Any
  live-updating text (a label that changes phrase, a number that gains a
  digit) must be barred from sizing its container — reserve the space
  instead — or the panel physically moves in time with the process it is
  measuring. *Evidence:* content-sized grid tracks driven by per-trial caption
  text produced 3.1–6.2px of continuous movement; pinning the tracks to the
  canvas dimensions took it to zero. Applies to every panel the M2/M3 demos
  inherit from this chassis.

## Decisions

1. **The published bundle is a build artifact, not a hand-run command.**
   `npm run build:single` emits both the standalone and artifact-body forms.
   Rationale: the hosted demo is cited from `design.md` and the journal, so it
   must be reproducible from the repo like any other result.
2. **Layout stability is now a checkable property, not a matter of taste.**
   `tools/measure-jitter.mjs` turns "it looks jittery" into a number with a
   pass condition, and lives in the repo rather than a scratch directory so
   M2's demo can be held to the same bar.
3. No docs updated: `abstract.md`, `open-problems.md` and `design.md` are
   untouched, because nothing about the science changed. The live-demo URL is
   unchanged (republished in place).

## Deviations

None. The fix stayed inside rendering, as scoped.

## Threats to validity

1. **The harness measures one browser** (Chromium via Playwright). Text
   metrics differ across engines, so a caption that fits on one line here
   could wrap elsewhere — mitigated by reserving two lines unconditionally,
   but not verified on WebKit or Gecko.
2. **240 frames is a sample, not a proof.** It covers thousands of trials at
   the top speed setting, but a rare state (a very long silence, a
   three-digit sparsity reading) could still exceed a reserved width. The
   reservations are sized with headroom rather than derived from proven
   bounds.
3. **Zero movement is not the same as zero perceived motion.** The canvases
   still repaint every frame by design; this entry claims only that no element
   changes position or size.
4. The `--w` stats reservations are eyeballed character counts. If a future
   stat outgrows its box the text will overflow its reserved width rather
   than being clipped — visible, but not caught automatically.

## Next

Unchanged from the standing plan, and now genuinely unblocked: design
**experiment 002 — changing its mind / live coaching**
([experiment-ideas.md §A1–A2](../../experiment-ideas.md)) with pre-registered
gates. The `TrialStepper` seam noted in the previous entry is still the place
a continuous-mode teacher plugs in. M2 (MNIST demo) follows the 002 decision
and will reuse this page's chassis — now with L-008 applied from the start.
