# Nothing crystallises: the core exists, is stationary, and nothing builds on it

- **Entry:** `2026-08-22-1300-crystallization`
- **When:** 2026-08-22 12:25–13:15 CDT
- **Who:** Javid + Claude (session)
- **Experiment:** [002](../../experiments/002-grown-substrate/design.md) — testing a hypothesis Javid posed **[J]**
- **Code state:** git `7f66e04` + `Edge.born` age tracking added this session
- **Re-run:** `npx vite-node tools/exp002-crystallization.ts 16000 1`

## The hypothesis, in Javid's words [J]

> "If the measures aren't moving, doesn't that mean that it's still not 'slow'
> enough. Something in the network has to build slowly and strongly without
> decaying. And then when it gets strong enough, faster things can grow around
> it. But if we're not seeing slow growth even, then the system is not moving
> towards learning."

This is sharper than [L-019](../LEARNINGS.md)'s "slow learning requires a slow
variable", because it names the *shape* the slow variable must have: a core
that accumulates without decaying, reaches a strength threshold, and then acts
as **scaffolding** that faster processes build around. Developmental biology
works this way — pioneer axons and radial glia lay a frame first, and
everything else follows it.

The logical half is simply correct and worth stating plainly: **flat is not
slow.** Slow means small-but-nonzero drift. Persistence at 5% on trial 250 and
5% on trial 32,000 is a stationary distribution, and a system at a fixed point
is not on its way anywhere.

## Why the existing measurement could not answer it

`exp002-longrun.ts` reported the fraction of edges alive at one checkpoint
still alive at the next. That single number is consistent with two opposite
worlds:

- **churn** — a random 5% survive each time, never the same ones
- **crystal** — the *same* 5% survive every time, a stable core under a
  churning surface

Aggregate survival is identical in both. Only edge *identity over time*
separates them, so `Edge.born` was added and age is now measured in **sleeps
survived** — how many structural rewirings an edge has lived through.

## Results

16,000 trials, seed 1, 800 rewirings. "core" = edges that survived ≥20
rewirings. `|w|` and `n` compare old edges against ones aged ≤2.

```
M1 arm
  trials | edges | median age | p90 | oldest | core | |w| old/young |  n old/young
    1000 |  5258 |     2      |  5  |   30   |   8  | 0.221 / 0.123 |  95.6 / 1.02
    2000 |  4303 |     2      |  3  |   42   |   4  | 0.421 / 0.123 | 111.5 / 0.77
    4000 |  6026 |     2      |  5  |   30   |   9  | 0.175 / 0.127 |  98.4 / 1.25
    8000 |  4323 |     2      |  4  |   39   |  10  | 0.204 / 0.122 | 107.1 / 0.75
   16000 |  4593 |     2      |  4  |   38   |   7  | 0.089 / 0.122 |  78.8 / 0.72

shallow arm
    1000 |  8598 |     3      |  9  |   45   |  78  | 0.283 / 0.129 | 180.7 / 2.78
    2000 |  6839 |     2      |  6  |   95   |   7  | 0.389 / 0.129 | 392.8 / 3.85
    4000 |  6872 |     3      |  7  |  195   |  12  | 0.368 / 0.126 | 575.2 / 2.81
    8000 |  5904 |     2      |  4  |   66   |   5  | 0.374 / 0.127 | 453.4 / 3.77
   16000 |  6918 |     3      |  7  |  193   |  33  | 0.200 / 0.128 | 386.1 / 3.43

no-rent control (ρ=0, nothing can die)
    1000 | 31205 |    33      | 47  |   50   | 25462 | 0.319 / 0.151 | 30.6 / 0.00
    2000 | 32567 |    81      | 97  |  100   | 32011 | 0.321 / 0.150 | 31.4 / 0.00
    4000 | 32768 |   181      |197  |  200   | 32757 | 0.322 /  n/a  | 31.4 /  n/a
    8000 | 32767 |   380      |396  |  400   | 32597 | 0.332 / 0.150 | 32.7 /  0.00
   16000 | 32768 |   780      |796  |  800   | 32768 | 0.336 /  n/a  | 33.2 /  n/a
```

## Analysis

### The core exists. It is real, and it is differentiated.

In both live arms a small population of old edges exists and it is **not** like
the rest. M1's old edges carry `n ≈ 80–110` against `≈1` for young ones — two
orders of magnitude more accumulated evidence — and roughly twice the weight
magnitude. The shallow arm's are stronger still (`n ≈ 390–575`). Selection is
doing something: the edges that survive many rewirings are the well-evidenced,
strong ones, exactly as rent-and-death is supposed to produce.

So the substrate is not undifferentiated noise. It has a nucleus.

### The core is stationary. It never grows.

M1's core: 8, 4, 9, 10, 7 edges across a 16× span of training. Roughly
**0.15% of a ~5,000-edge population, and the same size at trial 16,000 as at
trial 1,000.** The shallow arm's is larger and wilder (78, 7, 12, 5, 33) but
equally trendless.

Median age is **2 sleeps** in both — the typical edge survives two rewirings,
about 40 trials, and dies. (This also corrects a sloppy figure used earlier in
this project: "94% of every generation dies" conflated cumulative deaths with
per-generation turnover. Median-age-2 is the honest number.)

**So Javid's diagnosis is confirmed on its own terms.** There is no slow
build. The system reaches a stationary distribution — differentiated, with a
nucleus — within about a thousand trials, and then stays there forever.

### The no-rent control is the result that matters most

Turn rent off and the substrate *does* accumulate a permanent core: median age
780, and by trial 4,000 **every one of 32,768 edges is "old"**. That number is
exactly 1,024 nodes × 32 `maxOutDegree` — the structure has saturated its
capacity ceiling and stopped.

And this arm scores **0.444**, far worse than the 0.869 baseline
([2026-08-22-1215](./2026-08-22-1215-longrun-and-a-correction.md)).

Note what happened to the differentiation: old edges' `n` collapses from ~100
(M1, with rent) to ~31 (no rent). When everything survives, surviving stops
being evidence of anything.

That refines the hypothesis in a way worth keeping: **permanence is not the
missing ingredient — selective permanence is.** Rent is what makes survival
informative. Remove it and you get a large, permanent, undifferentiated core
that saturates the structure and performs worse than churn.

### The actual gap: the core is not scaffolding, because nothing can build on it

Javid's hypothesis has two clauses, and they fail differently. "Something must
build slowly and strongly without decaying" — half-true: it builds, it is
strong, it does not grow. "And then when it gets strong enough, faster things
can grow around it" — **this clause has no mechanism at all in 002.**

Growth samples targets weighted by `A(target)·exp(−span/λ_g)`. The activity
field `A` has a 200-tick time constant and knows nothing about edge age,
weight, or evidence count. **A twenty-generation-old, heavily-evidenced edge
confers no advantage whatsoever on new edges growing near it.** The core is
inert. It persists, and nothing accumulates around it, because no term in the
growth rule can see it.

That is why the core is stationary rather than growing: crystallisation needs
a positive feedback from what has already crystallised to what forms next, and
there isn't one. The design has selection (rent) and it has consolidation
(`n`), but the two never compound into structure.

## Learnings

- **L-026:** The grown substrate *does* form a differentiated nucleus — edges
  surviving ≥20 rewirings carry ~100× the evidence count and ~2× the weight of
  young ones — but the nucleus is **stationary**: ~8 edges out of ~5,000, the
  same size at trial 16,000 as at trial 1,000, with median edge age of 2
  sleeps throughout. Selection produces differentiation without accumulation.
  *Evidence:* this entry.
- **L-027:** Permanence without selection is worse than churn. With ρ=0 the
  substrate accumulates a permanent core of all 32,768 edges — saturating
  1,024 nodes × 32 out-degree — old-edge evidence counts collapse from ~100 to
  ~31, and accuracy falls to 0.444 against 0.869. What must persist is
  *selectively* the useful subset; rent is what makes survival informative.
  *Evidence:* this entry, the no-rent arm.
- **L-028:** 002's consolidated structure is inert: growth is weighted by an
  activity field with a 200-tick time constant and is blind to edge age,
  weight and evidence, so an old well-evidenced edge gives no advantage to
  anything growing near it. **Crystallisation requires positive feedback from
  what has already formed to what forms next, and the design has none** — which
  is why the core persists but never grows. *Evidence:* this entry; the growth
  rule in design §7.

## Decisions

1. **[L-019](../LEARNINGS.md) is refined rather than replaced.** "Slow learning
   requires a slow variable" was necessary but not sufficient. The full
   condition, from **[J]** and now supported: a slow variable that is
   *selectively* retained **and** that biases what forms next.
2. **H-012 registered** — scaffolded growth. This is now 002's most promising
   design change, ahead of H-004's site trace, because it is the missing
   clause rather than a second copy of the one that already half-works.
3. **The "94% of every generation dies" figure used earlier in this project is
   withdrawn** as a conflation of cumulative deaths with per-generation
   turnover. The correct figure is a median edge age of 2 sleeps.

## Threats to validity

1. **One seed.** The core sizes are small integers (4–78) and noisy; the
   *absence of a trend* over five checkpoints on one seed is weak evidence for
   stationarity, even though it is consistent across two arms.
2. **`OLD = 20` rewirings is arbitrary.** A different threshold would give
   different core sizes, though the trendlessness is threshold-independent in
   the data above.
3. **The "survival/sleep" column printed 1.000 throughout and is meaningless** —
   edges are born with age 1 by construction, so "fraction aged ≥1" is
   identically 1. It is left in the tool output but carries no information and
   should be removed or fixed.
4. **The no-rent arm saturates `maxOutDegree`**, so its collapse conflates "no
   selection" with "hit the structural cap". Separating those needs a no-rent
   arm with a much higher cap.

## Next

**H-012, scaffolded growth**, as 002's next design change: make growth's target
weighting see consolidated structure — for instance weight candidates by
`A(target)·exp(−span/λ_g)·(1 + κ·evidence near target)`, so a core that forms
becomes a frame that later growth follows. Pre-registered measure: **the core
must grow across checkpoints**, which it demonstrably does not today. If the
core still fails to grow with that term present, the scaffolding hypothesis is
wrong and not merely unimplemented.
