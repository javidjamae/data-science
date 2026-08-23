# 002-M1i: without the scaffold, growth never builds a single route

- **Entry:** `2026-08-23-0258-scaffold-ablation`
- **When:** 2026-08-23 02:58 CDT
- **Who:** Javid (the ablation question: everything Slow does, minus the random birth wiring — will it find pathways on its own?) **[J]**
- **Experiment:** **002-M1i** — the scaffold ablation of the Slow champion; prediction committed before running
- **Re-run:** `npx vite-node tools/exp002-noscaffold.ts`

## Results, verbatim (3 seeds × 6,000 trials)

```
slow, NO scaffold           tails 0.29/0.20/0.27   w2end 0/0/0
slow+strong, NO scaffold    tails 0.32/0.33/0.33   w2end 0/0/0
slow WITH scaffold (ref)    tails 0.99/0.99/0.99   w2end 5/10/11
```

## Analysis

The prediction held exactly. Durability and slow cadence *keep and refine*
routes; they cannot *originate* them: from zero, not one eye→answer route
formed in 6,000 trials on any seed, and accuracy never left chance. With the
scaffold and nothing else changed: 0.99 on every seed (incidentally the
strongest showing of the champion config yet — 0.99 × 3 on this seed set).

Why growth cannot route, three compounding walls: reach (rMax 8 on a 20-unit
gap forces multi-leg chains); blind guidance (the activity scent is
short-range and nothing in it points at three quiet output sites); and — the
decisive one — **no pay for half a bridge**: the middle legs of an unfinished
route carry no signal, earn nothing, gain no durability, and rent removes
them before completion.

The two admissible answers to routing are now sharply framed: **born with
routes** (the scaffold — works, 0.99) or **guided growth** (gradients/beacon,
H-020 — biology's mechanism, registered, unbuilt). Scaffold-free convergence
is exactly the guided-growth experiment, nothing less.

## Learnings

- **L-047:** **The scaffold is the load-bearing birth ingredient of the Slow
  stack, by single-variable ablation.** Identical economy and cadence from
  zero: 0.20–0.33 with zero routes ever created; with the scaffold:
  0.99/0.99/0.99. Growth under a short-range activity scent cannot originate
  multi-leg routes because unfinished routes earn nothing ("no pay for half a
  bridge") — origination requires either innate wiring or guidance
  (H-020). *Evidence:* this entry.

## Decisions

1. The champion configuration's honest description is now precise: **grown
   refinement on an innate scaffold** — matching biology's division of labor
   (H-020), not replacing it.
2. Scaffold-free routing = the guided-growth (beacon/gradient) experiment,
   queued in 006 stage 1; until it exists, no from-zero claim is available.
