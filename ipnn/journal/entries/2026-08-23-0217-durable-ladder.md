# 002-M1h: grow slower — Javid's knob was the right knob

- **Entry:** `2026-08-23-0217-durable-ladder`
- **When:** 2026-08-23 02:17–03:05 CDT
- **Who:** Javid (the hypotheses: slower-but-stronger growth; are we pruning too fast; longer runs are fine) **[J]**
- **Experiment:** **002-M1h** — sixteen deliberate arms on the working Durable configuration, 3 seeds × 6,000 trials
- **Code state:** tool committed with groups, prediction, gate and decision rule before first run
- **Re-run:** `npx vite-node tools/exp002-durable-ladder.ts <shard> <nshards>` · confirmation: `tools/exp002-confirm.ts` (held-out seeds 4–6, launched this session)

## In plain words

With the economy fixed, Javid asked to work the knobs again — specifically:
grow slower, grow stronger, prune less aggressively, and let runs go long.

**Grow slower is the winner, and it is not close.** Rewiring every 80 trials
instead of every 20 put two of three seeds at **0.99** — the first
near-perfect organisms this geometry has ever produced. One growth attempt
per neuron instead of two put a seed at **0.98**. The combination
slow-and-strong (fewer attempts, rarer rewiring, louder newborns) had the
best worst-seed (0.59) and one seed at **1.00**.

Stronger-alone helped mildly. Prune-slower hurt, again — third time the
congestion lesson has appeared (L-027, L-044). And the durability constant
we picked half-blind last hour (`rentN0 = 25`) turns out to sit at a real
optimum: weaker durability (50, 100) reverts to the taxed-to-death collapse;
stronger (10) starts to congest.

**The M1 gate (0.80 on all seeds) was NOT passed** — best worst-seed is
0.59. What remains is not level (the best seeds are at ceiling) but
**variance**: the same knobs give 1.00 on one seed and 0.34 on another,
which is the luck of which innate routes exist and survive. That is
precisely the variance Javid's kill-and-mutate scaffold selection (H-014)
exists to remove, and it is next.

## Results, verbatim (min = worst seed; chance 0.33; gate 0.80)

```
ARM slow+strong+prune  tails 0.33/0.58/0.67 min 0.33  w2end 4/16/7  edges 8453/9576/11012
ARM slow+strong        tails 0.63/0.59/1.00 min 0.59  w2end 1/10/22  edges 6980/7442/14031
ARM sleep80            tails 0.99/0.99/0.36 min 0.36  w2end 5/10/3  edges 5427/5184/5169
ARM sleep40            tails 0.67/0.34/0.34 min 0.34  w2end 1/0/1  edges 5793/5167/5004
ARM seed18k            tails 0.66/0.44/0.57 min 0.44  w2end 1/0/1  edges 6166/6814/6585
ARM rentN0=50          tails 0.16/0.33/0.14 min 0.14  w2end 0/0/0  edges 4404/6460/4482
ARM rentN0=100         tails 0.38/0.20/0.18 min 0.18  w2end 1/0/0  edges 5232/4728/4210
ARM rentN0=10          tails 0.67/0.64/0.33 min 0.33  w2end 13/3/1  edges 7603/9416/9375
ARM rent/2+death.01    tails 0.33/0.27/0.65 min 0.27  w2end 1/7/15  edges 7892/10573/10445
ARM rent/2             tails 0.61/0.33/0.33 min 0.33  w2end 2/1/3  edges 8306/10692/6793
ARM durable-base       tails 0.64/0.66/0.56 min 0.56  w2end 2/2/2  edges 6765/6285/7538
ARM death.01           tails 0.33/0.56/0.34 min 0.33  w2end 1/1/0  edges 9565/7510/8321
ARM bw.5+wmax6         tails 0.66/0.56/0.34 min 0.34  w2end 2/4/3  edges 11784/12208/10887
ARM bw.5               tails 0.66/0.51/0.59 min 0.51  w2end 4/1/4  edges 11263/11640/11655
ARM bw.3               tails 0.33/0.66/0.34 min 0.33  w2end 0/3/1  edges 7908/8755/6337
ARM att1               tails 0.98/0.64/0.34 min 0.34  w2end 5/2/2  edges 6377/6199/3917
```

Curves (selected): sleep80 s1 `0.38 > 0.72 > 0.90 > 0.97 > 0.98 > 0.99`,
s2 similar; slow+strong s3 reached `1.00`; durable-base steady ~0.6.
Full curves in the shard logs regenerable from the tool.

## Against the pre-registered elements

- **Gate (any arm ≥0.80 all seeds): NOT MET** (best min 0.59).
- **Prediction scorecard [C]:** "sleepEvery↑ and birthWeight↑ most likely to
  help" — right on sleepEvery (the strongest effect in the table), half-right
  on birthWeight (mild alone, good in combo). "Prune-slower risks
  congestion" — right. "rentN0 has an optimum" — right (25 ≈ optimal).
- **Decision rule:** best arms promoted to held-out confirmation (seeds
  4–6, 8,000 trials): `slow+strong` and `sleep80`, plus one **post-hoc,
  exploratory-labelled** hybrid (slow+strong at sleep80). Results land in
  the next entry.

## Learnings

- **L-045:** **Rewiring cadence is the dominant knob on the durable
  substrate.** sleepEvery 20→80 lifts seeds from ~0.6 to 0.99; one growth
  attempt instead of two puts a seed at 0.98; slow+strong yields the best
  worst-seed (0.59) with a 1.00. Mechanism reading: with earners protected
  (rentN0), the remaining noise was the churn *under* them — a thousand new
  edges per sleep perturbing the readout's inputs; growing slower quiets the
  construction site. Prune-slower fails again (third congestion result), and
  rentN0=25 sits at a measured optimum (10 congests, ≥50 reverts to L-042).
  *Evidence:* this entry.

## Decisions

1. Held-out confirmation running (pre-registered rule); nothing is claimed
   until it lands.
2. **Seed-variance is now the named bottleneck** — same knobs, 1.00 vs 0.34
   — and the scaffold-selection loop (H-014, Javid's kill-and-mutate) is the
   registered attack on it, to run on the best confirmed configuration.
3. **H-025 registered** (idea, **[J→C]**, from Javid's question "which knobs
   make sense as a bayesian probability within a range?"): promote knobs
   from global constants to **per-unit random variables drawn from
   distributions, heritable under the scaffold-selection loop** — per-neuron
   excitability/growth-vigor/reach, per-edge birth strength and durability.
   Today every parameter is one number shared by 1,024 neurons; biology has
   distributions everywhere, and heritable per-unit variation is how the
   *physics* comes under selection (design §11's noted bound).

## Threats to validity

1. Search seeds only (1–3) for the table; that is what the held-out stage
   exists for.
2. 6,000-trial tails; the 0.99 seeds were still at ceiling, the 0.34 seeds
   flat — bimodality, not noise around a mean.
3. The `sleep80` arm changes trials-per-sleep and total sleeps at once;
   cadence and total-rewirings are confounded here.
FTR
