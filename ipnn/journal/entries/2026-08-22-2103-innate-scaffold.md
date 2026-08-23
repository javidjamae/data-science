# The innate scaffold works on contact — and the organism's own economy demolishes it

- **Entry:** `2026-08-22-2103-innate-scaffold`
- **When:** 2026-08-22 21:03–21:40 CDT
- **Who:** Javid (the proposal: random interconnections at birth, then let strengthening and rewiring take over; kill failures, mutate what shows signal) **[J]**; implementation and decode **[J→C]**
- **Experiment:** [002](../../experiments/002-grown-substrate/design.md) — the innate-scaffold sweep
- **Code state:** flags `seedEdges`/`seedSpanMax`, defaults 0/recorded; tool committed with prediction and decision rule before first run
- **Re-run:** `npx vite-node tools/exp002-innate-seed.ts <shard> <nshards>` — density {1k,3k,6k,12k} × span {long 44, local 8} × 3 seeds × 1,500 trials

## In plain words

Javid's proposal: don't make the organism build its first wiring from nothing —
give it a big random set of connections at birth, including long ones, so a
path from eye to answer *can exist*, then let strengthening, rent and rewiring
take over. Kill variants that fail; mutate ones that show signal.

Half the prediction landed perfectly, and the other half failed in the most
instructive way anything has failed all day.

**It works on contact.** Long-range innate wiring manufactures short
eye→answer routes at birth (up to 29 of 64 pixels within two hops at the
highest density), and on one organism this produced **the first real learning
ever observed on this geometry** — accuracy blocks of 0.50 from the very
start, against a chance line of 0.33, where fifty knob variations and every
from-zero run sat dead. Local-span seeding at the same densities produced
nothing, on any seed: it is the **long-range** innateness that matters, not
the edge count. Exactly as predicted.

**Then the organism eats its own inheritance.** Watch the two columns:
`w2birth` (pixels within 2 hops at birth) versus `w2end` (same, at the end):
20/29/18 at birth → **1/0/0** at the end. The learning curves *decay* —
0.50 → 0.39 — as the innate routes are taxed and churned away. Total edges
relax to the same ~5,000 rent-equilibrium regardless of how many were
seeded. The birth advantage is gone within the first few dozen rewirings.

## Results, verbatim

```
ARM long-1000    tails 0.19/0.16/0.19 min 0.16  w2birth 0/1/1 w2end 0/0/0 edges 5170/5364/5801  toOut 20/24/8
    curves 0.17>0.22>0.27>0.16>0.19 | 0.16>0.19>0.21>0.30>0.16 | 0.20>0.17>0.08>0.16>0.19
ARM long-3000    tails 0.15/0.19/0.19 min 0.15  w2birth 1/4/1 w2end 0/0/0 edges 4263/4500/4012  toOut 10/13/12
    curves 0.14>0.18>0.12>0.17>0.15 | 0.18>0.20>0.18>0.20>0.19 | 0.27>0.17>0.13>0.26>0.19
ARM long-6000    tails 0.41/0.17/0.15 min 0.15  w2birth 5/8/5 w2end 1/0/0 edges 6962/5390/4715  toOut 23/8/12
    curves 0.50>0.47>0.39>0.43>0.41 | 0.16>0.22>0.17>0.24>0.17 | 0.12>0.13>0.15>0.25>0.15
ARM long-12000   tails 0.39/0.21/0.17 min 0.17  w2birth 20/29/18 w2end 1/0/0 edges 6488/5244/5237  toOut 15/13/18
    curves 0.49>0.50>0.45>0.43>0.39 | 0.11>0.18>0.23>0.10>0.21 | 0.22>0.16>0.16>0.14>0.17
ARM local-1000   tails 0.24/0.21/0.26 min 0.21  w2birth 0/0/0 w2end 0/0/0 edges 5269/5738/6417  toOut 15/19/14
    curves 0.13>0.22>0.12>0.15>0.24 | 0.14>0.17>0.16>0.13>0.21 | 0.19>0.25>0.16>0.22>0.26
ARM local-3000   tails 0.23/0.27/0.25 min 0.23  w2birth 0/0/0 w2end 0/0/0 edges 5823/5340/4935  toOut 17/13/9
    curves 0.15>0.16>0.16>0.17>0.23 | 0.23>0.13>0.24>0.17>0.27 | 0.18>0.19>0.12>0.20>0.25
ARM local-6000   tails 0.16/0.16/0.10 min 0.10  w2birth 0/0/0 w2end 0/0/0 edges 4786/3991/5578  toOut 24/14/17
    curves 0.27>0.21>0.20>0.13>0.16 | 0.12>0.16>0.21>0.14>0.16 | 0.23>0.19>0.18>0.16>0.10
ARM local-12000  tails 0.24/0.12/0.20 min 0.12  w2birth 0/0/0 w2end 0/0/0 edges 5727/3663/5301  toOut 20/10/15
    curves 0.19>0.20>0.22>0.22>0.24 | 0.20>0.18>0.14>0.14>0.12 | 0.15>0.16>0.19>0.11>0.20
```

## Analysis

**Why it dies, precisely.** Three mechanisms compound:

1. **Innate long edges are irreplaceable.** Growth is bounded by `rMax = 8`,
   and a two-hop eye→answer route needs ~10-unit legs. Once an innate long
   edge dies, *nothing in the organism can ever rebuild it*. The scaffold is
   a non-renewable resource.
2. **The economy taxes it like everything else.** Rent gives an unearning
   edge ~24 trials to live. So every innate route is in a race: be found by
   the readout and start earning within ~24 trials, or be destroyed. With
   random signs (half the seeded edges are inhibitory) and noisy credit,
   that race is usually lost — seed 1 won it, seeds 2 and 3 had *more*
   routes at birth (29 vs 20) and still lost.
3. **Routes are necessary, not sufficient.** Seed 2's 29 birth-routes never
   produced signal even early — a route through a wrong-signed edge carries
   nothing, and there is no grace period to fix the sign before rent ends
   the audition.

**The biological reading is sharp.** Real brains do not tax axon tracts on a
synaptic timescale. The *tract* (the wire) is stable infrastructure —
myelinated, long-lived; the *synapses along it* (the weights) churn and
compete. 002 conflates the two: one object carries both the wire's existence
and its weight, so structural capital is destroyed by weight-level
bookkeeping. Biology separates the lifetime of the wire from the lifetime of
the weight; and where long wiring is regenerated, new axons grow *along
existing tracts* (fasciculation) — which 002 also lacks.

**Javid's evolutionary loop is validated as the right next layer, one fix
early.** There is now real fitness variance between scaffolds (seed 1 vs
seeds 2/3), which is what kill-and-mutate needs. But run today, evolution
would mostly be fighting the rent — selecting for scaffolds that survive the
economy bug rather than scaffolds that compute. Fix the economy first, then
let selection tune the scaffold (H-014's first implementation).

## Learnings

- **L-041:** **An innate long-range scaffold produces immediate signal on the
  geometry that defeated everything else** — 0.50 accuracy blocks from birth
  (chance 0.33) at seed densities ≥6k with whole-sheet spans, while
  local-span seeding at identical densities produces nothing on any seed.
  Long-range innateness, not edge count, is the active ingredient — the
  first learning ever observed on 002's M1 geometry. *Evidence:* this entry.
- **L-042:** **002 taxes its structural capital to death because wires and
  weights are one object.** Innate routes decay from 20–29 (birth) to ~0
  (end) as flat rent + `rMax`-bounded growth make long edges irreplaceable
  non-renewables with a ~24-trial audition; learning curves decay with them
  (0.50 → 0.39). Biology separates tract lifetime from synapse lifetime and
  regrows long wiring along existing tracts (fasciculation). A substrate
  that must *keep* inherited structure needs earned durability or
  tract/weight separation. *Evidence:* this entry, w2birth vs w2end on all
  long arms.

## Decisions

1. Per the pre-registered decision rule (best long arm min-seed 0.15 < 0.5 —
   the *mean* signal on seed 1 does not clear it): random innateness alone is
   **insufficient**, and the 006 program leads with **guided scaffolds
   (H-020) plus the economy fix** — tract/weight separation or
   evidence-scaled rent — with fasciculation-style long-edge regrowth
   registered (H-024).
2. **The kill-and-mutate scaffold loop (H-014) is scheduled after the
   economy fix**, so selection optimizes computation rather than
   rent-survival.
3. `seedEdges` stays available and defaulted off; every use reports the
   from-zero caveat, per 002's own design note.

## Threats to validity

1. 1,500 trials; a longer run could in principle re-grow signal (nothing in
   the mechanism suggests how, given irreplaceability).
2. Sign-balanced seeding is one choice; excitatory-biased scaffolds might
   win the race more often and are untested tonight.
3. Three seeds per arm; seed 1's success is n=1 and its curve was already
   decaying.
