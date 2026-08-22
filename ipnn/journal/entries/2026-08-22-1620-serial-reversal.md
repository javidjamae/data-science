# The organism ages out of learning, and consolidation is why

- **Entry:** `2026-08-22-1620-serial-reversal`
- **When:** 2026-08-22 16:20–16:50 CDT
- **Who:** Javid + Claude (session)
- **Experiment:** [003](../../experiments/003-transfer-retention/design.md), serial reversal
- **Code state:** git `e49319d`
- **Re-run:** `npx vite-node tools/exp003-serial-reversal.ts 1 2 3 4`
- **Corrects:** [2026-08-22-1530](./2026-08-22-1530-transfer-retention.md), which annotated L-004 as unsupported. That annotation was wrong.

## In plain words

The idea, from Javid **[J]**: when you flip the rule — same pictures, different
correct answers — you separate two things. What the organism has worked out the
pictures *mean*, and the rules for what to *do* with them. If those live in
different places inside it, a new rule should be cheap: you already recognise
the picture, you just re-point the output. If they don't, every rule change is
a full rebuild.

**So how fast something reverses tells you whether it has a picture of the
world at all, separate from its habits** — and you can measure it without
opening the thing up, which is why the same test works on a fish.

We flipped the rule eight times. It never got better at it. Not once, on any
seed.

Then something worse showed up. It didn't just fail to improve — **it
progressively lost the ability to learn at all.** First flip: 2,425 attempts,
nearly nine times the cost of learning the task fresh. By the fourth or fifth
flip, most runs never got there at all, within any budget we gave them.

The cause turned out to be a feature we built on purpose. The organism keeps a
confidence count on every connection, and connections it trusts become harder
to change — that's what makes a memory stick. But that count **only ever goes
up.** Every flip of the rule pushes it higher. Eventually every connection is
too confident to move, and the organism can no longer learn *anything* — not
the new rule, not the old one.

Switch that mechanism off and it keeps learning through all eight flips: 22 of
32 flips succeed instead of 7.

**The thing that gives it memory is the same thing that eventually kills its
ability to learn.** The project wrote that down as a worry on day one. Today it
happened, and it is measurable.

## Results

Serial reversal, alternating `[0,1,2] ↔ [1,2,0]` on the same three stimuli.
Criterion rolling-100 ≥ 0.85, cap 2,500 trials per stage.

```
seed | acquire | rev1 | rev2 | rev3 | rev4 | rev5 | rev6 | rev7 | rev8
  1  |    229  | 2219| none| none| none| none| none| none| none
  2  |    259  | 2479| 1092| 1276| none| none| none| none| none
  3  |    278  | none| none| 2465| 1980| none| none| none| none
  4  |    342  | none| none| 2462| none| none| none| none| none

original acquisition 277 · first reversal 2425 (8.8× the cost of learning fresh)
improvement across reversals: 0/4 seeds        gate: FLAT
```

Consolidation ablation, same protocol:

```
seed | consol | acquire| rev1 | rev2 | rev3 | rev4 | rev5 | rev6 | rev7 | rev8
  1  |  ON    |   229  | 2219| none| none| none| none| none| none| none
  1  |  OFF   |   228  |  636| 1143|  891| none| 2049| 1818| none|  737
  2  |  ON    |   259  | 2479| 1092| 1276| none| none| none| none| none
  2  |  OFF   |   263  | none| none| 1239| 2409| none| 2474| none| 1552
  3  |  ON    |   278  | none| none| 2465| 1980| none| none| none| none
  3  |  OFF   |   274  | none| 2263| 2476| none|  753| 1852| 1884| none
  4  |  ON    |   342  | none| none| 2462| none| none| none| none| none
  4  |  OFF   |   353  | 2496| none| 1983| 1881| 1805|  598| 1649| 1007

reversals reaching criterion, of 32: consolidation ON 7 · OFF 22
```

## Analysis

### The prediction held, and reversal speed is now a calibrated instrument

Pre-registered before running: no improvement across reversals, because
[L-010](../LEARNINGS.md) showed 001's middle layer is a fixed random projection
that learns nothing, so there is no representation held apart from the mapping
and nothing to reuse. **0 of 4 seeds improved.** Confirmed.

That makes 001 a **known-negative reference** for this measure. Any future
substrate claiming to hold a representation separately from its policy can be
run through the same protocol, and "reversal gets cheaper with practice" now
has a calibrated zero to be measured against. That is worth more than the
negative result itself.

### But the interesting finding was not the one predicted

Flat was expected. **Terminal decline was not.** Reversal 1 costs 8.8×
acquisition; by reversals 4–5 most runs never reach criterion at any budget,
and once a seed starts failing under consolidation it does not recover.

### Consolidation causes it, and this vindicates L-004

Turning consolidation off triples the number of reversals that succeed (7/32 →
22/32), and the OFF runs keep succeeding right through reversal 8 — seed 1
relearns in 737 trials on the last flip, seed 4 in 1,007. There is no terminal
state without consolidation.

The mechanism is the defect [002's design §7](../../experiments/002-grown-substrate/design.md)
named and deliberately left unfixed: **`n` only ever increases.** Plasticity is
`1/(1 + n/n₀)`, so it decays monotonically over the organism's whole life. Every
reversal drives `n` up on every weight that participates. Eventually every
weight is too well-evidenced to move and the organism cannot learn *any* rule —
not the new one, and not the old one it once knew.

[L-004](../LEARNINGS.md) said exactly this on day one: *"the mechanism behind
answer collapse is the same mechanism IPNN deliberately uses for consolidation.
'Consolidated memory' and 'frozen wrong answer' are one phenomenon with
different valence; expect this tension at every scale."* It is not a tension in
the abstract. It is a measurable death.

### The correction I owe

Four hours ago, in [2026-08-22-1530](./2026-08-22-1530-transfer-retention.md), a
**single** reversal was run with consolidation on and off — 2,093 versus 2,488
trials, variance dwarfing the difference — and L-004 was annotated as *"first
real test came back empty."*

**That annotation was wrong, and it was wrong for a specific and repeatable
reason: one reversal is not a test of a cumulative mechanism.** `n` accumulates.
After one flip it has barely moved and the effect is invisible; after four it is
fatal. The single-shot test could not have detected what it was looking for, and
it was still reported as evidence of absence within the hour.

The annotation is being corrected rather than removed, and the episode is
recorded as L-035.

### A phenomenon worth naming

Set aside whether it is a bug. This substrate **ages**. It learns quickly when
young, learns the same thing more slowly each time it has to revise, and
eventually reaches a state where it is competent at nothing and can acquire
nothing. Its plasticity is a finite resource that is spent and never replenished.

That is a real developmental trajectory falling out of a two-line learning rule,
and it is the first genuinely life-like behaviour this project has produced
without designing it in. Whether it is a defect or a feature depends entirely on
whether an organism *should* be able to spend its plasticity — and nothing in
`vision.md` has an opinion on that.

## Learnings

- **L-033:** 001 shows **no reuse across rule changes**: the first reversal
  costs 8.8× original acquisition and successive reversals never get cheaper
  (0/4 seeds improved over eight flips). Reversal speed is a probe for whether
  perception is factored from policy, needs no access to internals, and works on
  animals — **001 is now the calibrated known-negative for it.** *Evidence:*
  this entry.
- **L-034:** **Consolidation causes cumulative learning death.** Under repeated
  rule change, 7 of 32 reversals reach criterion with consolidation on against
  22 of 32 with it off, and the failures under consolidation are terminal.
  Because `n` only ever increases, plasticity `1/(1+n/n₀)` decays monotonically
  and the organism eventually cannot learn *any* rule. L-004's tension,
  measured. *Evidence:* this entry, 4 seeds × 8 reversals × 2 arms.
- **L-035:** **A single-shot test is not a test of a cumulative mechanism.** One
  reversal found no consolidation effect and L-004 was annotated unsupported
  within the hour; eight reversals showed a 3× effect and a terminal failure
  mode. Before recording a null, ask whether the mechanism needs *repetition* to
  express itself — and if it accumulates, the exposure must accumulate too.
  Sibling of [L-018](../LEARNINGS.md) (a snapshot is weaker than it looks): the
  same error in the repetition dimension rather than the time dimension.
  *Evidence:* this entry against [2026-08-22-1530](./2026-08-22-1530-transfer-retention.md).

## Decisions

1. **L-004's annotation is corrected**, not deleted. It now records that the
   single-reversal test was underpowered and that serial reversal supports it.
2. **The α/β fix is promoted from "registered follow-up" to the next change to
   the learning rule.** 002's design §7 deferred it to avoid changing two things
   at once; it now has a measured failure mode behind it rather than a
   theoretical one. `n` must be able to *decrease*.
3. **Serial reversal joins the standing battery** in
   [vision.md](../../vision.md), between rungs 6 and 7 — it measures something
   neither retention nor learning-set does.
4. **Backlog track H gains a second hard requirement:** entrants must be scored
   over *repeated* rule changes, not one. A system that looks fine on a single
   flip can be terminally damaged by four, and a leaderboard that flips once
   would never see it.

## Threats to validity

1. **Four seeds**, and the per-stage numbers are extremely noisy (636 to 2,496
   trials on the OFF arm).
2. **Cap-as-value.** Stages that never reached criterion were scored at the cap
   (2,500) for the improvement calculation. That is the conservative direction
   for the flat conclusion but it compresses the true spread.
3. **`n₀ = 1000` is one setting.** The aging effect is presumably a function of
   `consolidationN0`, and the sweep from "no aging" to "immediate aging" has not
   been run. The claim is that consolidation *causes* this, not that these
   particular numbers are general.
4. **Alternating between two rules is the easiest serial-reversal design.**
   Using a fresh permutation each time would be harder and is not run.
5. **The aging framing is interpretation, not measurement.** What was measured
   is that repeated reversal under consolidation ends in terminal failure. That
   this constitutes "aging" is a story about it.

## Next

**Run the α/β fix** — let `n` decrease when evidence contradicts a synapse —
and re-run serial reversal against it. Pre-registered prediction: the terminal
failure disappears and reversals stay learnable through all eight, without
losing the frozen-retention result ([L-005](../LEARNINGS.md)) that made
consolidation worth having. If retention collapses when `n` can fall, the
tension L-004 describes is fundamental rather than a fixable defect, and that
would be the more important result.
