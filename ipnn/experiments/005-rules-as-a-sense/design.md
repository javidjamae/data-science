# Experiment 005 — The Rule as a Sense

**Status:** pre-registration. Nothing built.
**Origin:** Javid, 2026-08-22 **[J]** — the "rules cortex" idea, in
[2026-08-22-1917](../../journal/entries/2026-08-22-1917-adapt-not-relearn.md):
show the organism the currently expected output protocol through a sense,
rather than making it rediscover the protocol from reward after every change.
**Tests:** [H-018](../../journal/HYPOTHESES.md).
**Substrate:** experiment 001, config-only changes. Defaults untouched; the
pinned M1 curves cannot move.

## 1. The question

003 established that a protocol flip costs 001 everything: below-chance
collapse, 600–2,500 trials of relearning, no improvement with practice
(L-030, L-033). The goal is an organism that does not relearn at a flip
because it has *already learned* — identity once, binding cheap.

> If the rule in force is **visible** — arriving through a sense like every
> other fact about the world — can the organism learn both protocols
> conditioned on the cue, so that a flip costs approximately nothing?

This is cued task-switching (monkeys, pigeons, humans — admissible per
H-009), and mechanistically it is a test of whether the random pool's
accidental cue×stimulus conjunctions (Rigotti-style mixed selectivity) are
usable by a 480-weight readout. No novelty is claimed anywhere in this
design.

## 2. Method

**Sense:** `senseSize: 80` — the 64 stimulus pixels plus a 16-pixel **cue
strip** (two rows of 8). Cue A: top row lit. Cue B: bottom row lit. The cue
is present for the whole stimulus phase, exactly as if it were part of the
image. Nothing else changes: same organism class, same teacher, same
criterion machinery.

**Protocols:** A = identity mapping, B = the [1,2,0] permutation — 003's two
rules, unchanged.

**Training:** interleaved from the start — every trial draws protocol A or B
at random (seeded), shows the matching cue, rewards per that protocol's
mapping. Interleaving forces the conjunction; blocked training would let the
readout treat the cue as decoration until tested.

**Arms, per house rule (a control must remove the thing it names — L-016,
L-024):**

| arm | cue | what it isolates |
|---|---|---|
| **cued** | matches the active protocol | the hypothesis |
| **uncued** | strip always dark | the 003 baseline: pure inference from reward |
| **scrambled-cue** | cue drawn at random, uncorrelated with protocol | presence of a cue vs *information* in the cue |

Seeds 1–5 each. 4,000 interleaved training trials, then the switch test.

**The switch test — measured without the window floor (L-032):** freeze
learning off. Present 200 trials of pure protocol A (cue A), then flip to
pure protocol B (cue B) with **accuracy scored directly on the first 30
trials after the flip** — no rolling window straddling the boundary, no
floor. Repeat the flip 10 times, alternating. Learning stays off, so this
measures what was already learned, nothing else.

## 3. Pre-registered gates

1. **Concurrent mastery.** The cued arm's tail-100 accuracy over interleaved
   training ≥ 0.80 on ≥ 4 of 5 seeds. (Uncued cannot exceed ~0.5 by
   construction — the two protocols disagree on every stimulus and it has no
   way to know which is in force.)
2. **Instant switching.** Cued arm, first-30-after-flip accuracy, mean over
   the 10 flips ≥ 0.70 — while the uncued arm's first-30 sits below chance
   (the L-030 signature). This is "the flip stops hurting."
3. **The cue's information does the work.** The scrambled-cue arm fails gate 1
   (≤ 0.60). If scrambled matches cued, the cue was a crutch of some other
   kind and the result is void.

**Prediction [C], with reasoning:** pass, with moderate confidence. Each pool
neuron draws 24 of 80 inputs, so ~4.8 cue pixels in expectation — nearly every
pool neuron sees the cue, pool states become protocol-conditioned, and six
conjunction classes (2 protocols × 3 stimuli) onto 3 outputs is within a
480-weight readout's capacity on random features. The genuine risk is
interference during interleaved training, ε-exploration noise on first-30
scoring, and L-010's general lesson that this pool underdelivers.

**What failure means:** one learnable layer cannot *select* between two
mappings even when told which is in force. That is a sharper indictment of
shallow substrates than anything in 002's file, and it becomes the concrete
target 002 (grown structure) and 004 (iteration) must hit.

## 4. Relationship to the rest

- **The inferred variant stays open** as the hard case: same design, cue
  removed, rule changing unannounced — already measured to fail flat (L-033).
  005 first shows the factoring is *expressible*; discovering it untold is
  the later, harder rung.
- **H-019's dual** (rotate/zoom the stimuli, protocol fixed) is the next
  experiment after this one; the far cell — both varying — is the target.
- **Track H:** a "shown rules" game class becomes admissible for the Games,
  with the scrambled-cue arm as its integrity check.
- **004** queues behind, sharpened: iteration must buy adaptation *speed*
  (L-039).

## 5. Threats, anticipated

1. **The cue may dominate the stimulus** (16 always-lit-together pixels vs
   20–24 stimulus pixels): the pool could encode the cue and lose the glyph.
   The concurrent-mastery gate would catch it as a failure to reach 0.80.
2. **Two protocols is the minimum.** Passing with 2 does not show it scales
   to 6; a follow-up sweep over all six mappings with six cues is registered
   but not run.
3. **First-30 scoring includes ε-exploration noise** (~5% forced errors),
   which caps the observable ceiling near 0.93 — the 0.70 gate leaves room.
4. **Frozen switch test only.** Whether cued switching *with learning on*
   stays clean (no drift from the still-flowing rewards) is measured but not
   gated.
