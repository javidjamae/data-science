# Experiment 002 M0 built; M1 fails, and the failure has an address

- **Entry:** `2026-08-22-0208-exp002-m0-built-m1-fails-on-depth`
- **When:** 2026-08-22 01:45–02:40 CDT
- **Who:** Javid + Claude (session)
- **Experiment:** [002 — the grown substrate](../../experiments/002-grown-substrate/design.md), M0 and M1
- **Code state:** git `3cbe1f8` (the results in this entry were produced by the code as committed there)
- **Re-run:**
  ```
  cd ipnn/experiments/001-mnist-living-demo/app
  npx vitest run                                        # 88 tests, incl. 001 unchanged
  npx vite-node tools/exp002-m1-gate.ts m1      2000 1 2 3
  npx vite-node tools/exp002-m1-gate.ts shallow 2000 1 2 3
  npx vite-node tools/exp002-m1-gate.ts no-spont 300 1
  npx vite-node tools/exp002-m1-diagnose.ts 1000 1
  ```
- **Review:** `/code-review high` was run against this branch before the entry
  was finalised. It found five defects, all in the tooling and none in the
  substrate; three of them had already produced wrong numbers in the first
  draft of this entry. See *Threats* 5 and learnings L-016/L-017.

## In plain words

We built the thing experiment 002 was designed to be: a 32×32 sheet of 1,024
sites with **nothing wired at all**, that has to grow its own connections from
scratch, guided by two spreading chemical signals — one saying *where to
build*, one saying *what to keep* — while every connection pays rent every
tick and dies if it stops earning.

It grew. From zero edges it wired itself across the sheet, reached all three
answer neurons in four to six steps, and started getting rewards within the
first two hundred ticks of its life. The failure everyone expected — that a
sheet with nothing wired would just sit there dead forever — did not happen.

And it learned nothing. Two thousand trials, three seeds, dead flat at chance
the whole way.

So we went looking for where the signal dies, and found it in one place. We
measured how much each node's firing depends on which shape is showing. The
eye: enormous, obviously. One connection away from the eye: still clearly
there. **Two connections away: gone. Pure noise.** And the answer neurons sit
four to six connections away, so they are receiving static, and no amount of
reward can teach anything from static.

To check that this was really about distance and not about something else
being broken, we moved the answer neurons closer and changed nothing else. It
learned: 0.85, 0.91, 0.89 across three seeds.

Then we measured how far each of the sixty-four pixels of the eye actually
sits from an answer, and the story got sharper. In the failing version, **not
one pixel** is closer than five connections. In the working version, nineteen
of them are within two. That is the whole difference. The organism is not
building a rich internal picture and reading it off — it is finding whichever
handful of pixels happen to sit close to an answer and living off those.

That is a useful failure. The growth machinery works, the learning rule works,
reward-from-a-standing-start works. What does not work is getting information
*forward* across more than one connection — which is a different problem from
the one we expected. We expected trouble sending credit *backward*. The signal
never made it forward in the first place, so there was nothing for the credit
to land on.

One more thing turned up on the way, and it is the kind of thing that quietly
ruins experiments. The design had a planned "control" test: switch off the
random background firing and the sheet should be dead, proving that background
firing is what gets everything started. We ran it. The sheet came back to life
anyway. It turns out a different mechanism — the one that keeps the network
from getting too excited — also works in reverse: when nothing at all is
firing, it keeps turning up the sensitivity until something does. So the
control we had planned would have proved nothing, and we would not have known.

## Objective

Two things, in order.

1. **M0:** build the grown substrate — lattice, two fields, ring-buffer
   delays, growth, rent, death — behind experiment 001's existing
   `OrganismLike` interface, without changing 001's behaviour by a bit.
2. **M1:** run the pre-registered critical gate on it.

## Gate (pre-registered)

From [002 design §8](../../experiments/002-grown-substrate/design.md), verbatim:

> **M0 — scaffold.** Extract the `OrganismLike` interface (§2) from experiment
> 001 without changing its behavior — all 17 existing tests must pass
> unchanged — then build the substrate, fields, ring-buffer delays, growth and
> rent, with headless tests for each. No gate beyond "001 still works,
> bit-identical."

> **M1 — it wires itself at all. The critical gate.** Uniform reward field
> (= 001's broadcast), uniform latency (all `d = 1`), rent and growth on, from
> zero edges.
> **Gate:** ≥0.80 rolling accuracy over the last 100 of 2,000 trials, on ≥3
> seeds, *and* a connected input→output path exists at the end.
> **If M1 fails, everything below is moot** and the finding is "reward-driven
> growth from zero does not reach a competence a fixed random projection
> reaches in 800 trials" — a real negative, and the journal's rule 5 says it
> gets written up with the same rigor as a success.

The interface extraction half of M0 landed earlier, in `6578a30`. This entry
covers the substrate itself and M1.

## Hypotheses

Written before running.

- **H1 (M0):** the substrate can be built behind the nine-member interface
  with no change whatsoever to experiment 001's behaviour.
- **H2 (cold start):** spontaneous firing plus urge is enough to bootstrap
  from zero edges — the organism will fire, grow, speak, and be rewarded.
  This is design §10 risk 1, the *expected* failure mode. (Survives, but the
  attribution to spontaneous firing turned out to be wrong — see Analysis.)
- **H3 (M1):** with structure grown and a path in place, the three-factor rule
  reaches ≥0.80 — i.e. growing a substrate is at least as good as being given
  a random one.

## Method

**Changes.** New `src/engine/grown/`, five modules, all behind the unchanged
`OrganismLike` contract:

- `lattice.ts` — 32×32 sites with real positions; input cortex (8×8, one site
  per sense pixel, clamped by the sense), output cortex (3 sites on the far
  side), reward cortex (a locus off the input→output axis). Constructor
  enforces the geometry constraints the design states as prose, including that
  the reward locus is further off-axis than one growth step.
- `fields.ts` — discrete diffusion with reflecting boundaries. Activity field
  `A` runs live (length scale 3 sites, time constant 200 ticks). Reward field
  `R` is solved to steady state once at construction — exact, not an
  approximation, because the source is stationary and reward is instantaneous
  — and **normalised to mean 1 over the lattice**, so the diffusing arm
  redistributes the same credit budget as the uniform arm rather than
  delivering less of it.
- `edges.ts` — per-edge weight, eligibility, evidence, and latency
  `d = max(1, ceil(span/v))`; a ring buffer of pending arrivals; CSR adjacency
  rebuilt at each sleep. The class deliberately publishes **no presynaptic
  firing state at all**, only a per-edge `arrived` flag, so the correctness
  trap design §5 calls out (updating eligibility against emission rather than
  arrival) has no wrong thing available to read.
- `grown-organism.ts` — the substrate. Tick order: deliver → fire → eligibility
  and rent → emit → fields → readout → maybe sleep.
- `config.ts` — every knob in one place, including the two control-arm
  switches (`rewardField: 'uniform'` reproduces 001's broadcast exactly;
  `latency: 'uniform'` ignores span).

**Configuration snapshot** (the M1 arm, verbatim):
`width 32, height 32, poolSize 1024, outputSize 3, inputOrigin (2,12),
outputX 29, outputYs [13,16,19], rewardCortex (16,4), rewardRadius 1,
gain 2.0, bias −1.0, targetSparsity 0.15, inhibitionRate 0.02, pSpont 0.02,
lateralInhibition 2.0, urgeRate 0.05, urgeMax 3.0, readoutWindow 20,
rewardField 'uniform', rewardLambda 8, activityD 0.045, activityDecay 0.005,
latency 'uniform', conductionSpeed 3, traceDecay 0.97, eta 0.08, wMax 3.0,
consolidation true, consolidationN0 1000, rent 0.00009, birthWeight 0.15,
deathThreshold 0.02, growthAttempts 2, rMax 8, lambdaG 4, sleepEvery 20,
maxOutDegree 32`; teacher unchanged from 001
(`maxTicks 60, blankTicks 15, spokenWindow 20, spokenThreshold 6,
schedule 'ignore', rewardMagnitude 1.0, correctionMagnitude 0.2,
baselineRate 0.05`); seeds 1–3; 2,000 trials; tail = last 100.

`gain` and `bias` are 001's values on purpose — design §6 asks for a node
deliberately close to 001's so that any difference in outcome is attributable
to the plumbing rather than to a new unit model.

`rent` is the one parameter that is *derived* rather than chosen:
`(birthWeight − deathThreshold) / (sleepEvery × ticks-per-trial)
= 0.13/(20×75) = 8.7e−5`, so an edge that never earns lasts about one
inter-sleep period and then fails to pay. Much larger and every edge dies
before it is ever judged; much smaller and nothing is selected against.

**Measurement.** The M1 gate harness (`tools/exp002-m1-gate.ts`) uses the same
seed→schedule formula, trial count and rolling-100 windows as
`m1-sanity.test.ts`, so the curves are comparable to 001's line for line. It
additionally reports edge count over time, births, deaths, sleeps,
out-degree-cap binds, input→output connectivity and hop counts,
time-to-first-reward, mean activity and variance, and the clean-vs-ambiguous
tick split — the standing measurements design §8 requires.

## Results

**M0 gate — met.** 001's M1 curves are bit-identical to those recorded in
[2026-08-16-0248](./2026-08-16-0248-experiment-001-m0-m1-first-build.md):

```
seed 1  0.53 → 0.80 → 0.91 → 0.98 → 0.94 → 0.95 → 0.97 → 0.98
seed 2  0.54 → 0.80 → 0.87 → 0.94 → 0.96 → 0.95 → 0.97 → 0.98
seed 3  0.42 → 0.74 → 0.85 → 0.89 → 0.95 → 0.96 → 0.98 → 0.99
frozen retention 0.97
```

88 tests pass (17 pre-existing, unchanged; 71 new for the grown substrate),
`tsc` clean, `vite build` clean. Substrate speed 6,900–8,500 ticks/s against
the design's ≥1,000 target (§10 risk 3).

**M1 — FAIL.** 2,000 trials, seeds 1–3, verbatim rolling-100 curves:

```
seed 1  0.18 0.20 0.21 0.15 0.18 0.18 0.17 0.28 0.18 0.22
        0.14 0.23 0.17 0.28 0.22 0.19 0.21 0.16 0.21 0.09   tail 0.090
seed 2  0.26 0.18 0.21 0.13 0.25 0.23 0.19 0.13 0.25 0.15
        0.20 0.17 0.16 0.19 0.13 0.16 0.25 0.23 0.24 0.19   tail 0.190
seed 3  0.24 0.16 0.12 0.15 0.26 0.16 0.15 0.18 0.17 0.17
        0.16 0.23 0.09 0.12 0.20 0.12 0.22 0.17 0.19 0.19   tail 0.190
```

Mean tail **0.157** against a 0.333 chance line, with 40.7% / 42.0% / 45.4% of
trials ending in silence. Silence scores as incorrect, so conditional on the
organism answering at all, accuracy is 0.19/0.58 ≈ **0.33 — exactly chance**,
flat, for the entire run.

Everything else the gate asks about **passed**:

- input→output path exists on every seed, to all three outputs, at 4–6 hops
  by the shortest route. Per sense pixel — design §8's path-length
  distribution, which is the figure that actually matters — **no pixel is
  closer than 5 hops to an answer**: `unreached 10, 5-hop 20, 6-hop 15,
  7-hop 18, 8-hop 1` out of 64
- first reward arrived at tick 46–167
- structure grew and persisted: 4,300–4,700 live edges at the end
- activity stayed pinned at the 0.158 homeostatic target, variance ~2.5e−4
- output cortex ambiguous (two-outputs-at-once) on only 2.3–2.6% of ticks, so
  design §10 risk 4's collapse mode did **not** occur
- deepest path delay 6 ticks against a 33-tick eligibility horizon, so design
  §5's credit-reach coupling was **not** violated
- out-degree cap bound 0 times

**Diagnosis** (`tools/exp002-m1-diagnose.ts`, seed 1, 1,000 trials). Firing
rate per site measured under each pattern with learning off and rent
suspended, expressed as the spread across patterns in standard errors — a
node carries task information exactly to the extent its firing depends on
which pattern is showing:

Under the null — firing independent of the stimulus — this statistic is the
range of three independent standard normals, so its expected value is exactly
**3/√π = 1.693**. That is the noise floor to read the table against.

```
input cortex (hop 0, clamped)  mean 76.6σ
hop 1   n= 178   mean  7.7σ   best 59.1σ   #{>3σ} 137
hop 2   n= 199   mean  1.7σ   best  6.6σ   #{>3σ}  21
hop 3   n= 192   mean  1.8σ   best  4.7σ   #{>3σ}  17
hop 4   n= 182   mean  1.7σ   best  4.5σ   #{>3σ}  15
hop 5   n= 150   mean  1.8σ   best  4.8σ   #{>3σ}  13
hop 6   n=  47   mean  1.9σ   best  3.8σ   #{>3σ}   8
```

At the output cortex itself:

```
output 0: rates 0.138 / 0.120 / 0.129   2.9σ
output 1: rates 0.149 / 0.155 / 0.145   1.5σ
output 2: rates 0.130 / 0.126 / 0.141   2.4σ
```

Structure at the same point, by cortex pair:

```
live edges 5258
  interior → interior   4789
  input    → interior    433
  interior → output       19      ← the readout layer, in full
  output   → interior     17
mean interior in-degree 5.47
output in-degrees       7, 7, 5
|w| into outputs        n=19  mean 0.1096  max 0.2150   (born at 0.15)
churn                   94,786 born, 89,528 died over 50 sleeps — 94% of
                        births die
```

**Shallow arm** (post-hoc; design §10 risk 1's last rung, "shorten the
input→output distance"). `outputX: 29 → 14`. Nothing else changed. 2,000
trials, seeds 1–3.

**This is not a depth-1 arm, and the first draft of this entry wrongly called
it one.** Measured sense-depth distribution: `unreached 8, 1-hop 6, 2-hop 13,
3-hop 26, 4-hop 10, 5-hop 1` out of 64. So six pixels get a direct edge to an
answer and thirteen more are two hops away; the median pixel is still three
hops back. What the arm changes is not "depth removed" but "**19 of 64 pixels
brought inside the one-hop horizon, where the M1 arm had none**."

```
seed 1  0.19 0.30 0.49 0.44 0.50 0.46 0.35 0.38 0.49 0.62
        0.62 0.68 0.68 0.80 0.94 0.96 0.87 0.92 0.90 0.85   tail 0.850
seed 2  0.19 0.19 0.26 0.36 0.62 0.52 0.87 0.95 0.93 0.95
        0.95 0.72 0.84 0.91 0.90 0.91 0.81 0.93 0.82 0.91   tail 0.910
seed 3  0.23 0.47 0.44 0.45 0.59 0.56 0.47 0.81 0.95 0.93
        0.94 0.94 0.89 0.92 0.91 0.96 0.92 0.86 0.96 0.89   tail 0.890
```

Mean tail **0.883**, every seed ≥0.80, all outputs connected, silence down to
9–16%.

**Cold-start control** (`no-spont`, pre-registered in design §8 as "expected to
fail G0 outright"). As first written this arm set `pSpont: 0` and nothing else,
and it **wired up normally** — 4,771 live edges, all three outputs connected,
mean activity 0.162. It was not a control. The reason is measured below.
Corrected to `pSpont 0, bias −30, urgeMax 0, inhibitionRate 0`, it fails
exactly as the design predicted:

```
tail 0.000   silent trials 100.0%   first reward: never
94 live edges (all from the sense-clamped input cortex)
connected [false, false, false]     sense depth: unreached 64 of 64
```

**The homeostat regenerates activity from silence.** Interior firing rate with
`pSpont 0, bias −30`, measured in three consecutive windows:

```
homeostat on   first 300 ticks 0.0000   next 5k 0.0179   next 20k 0.1499
homeostat off  first 300 ticks 0.0000   next 5k 0.0000   next 20k 0.0000
```

0.1499 is `targetSparsity`. With nothing firing, the homeostatic error term is
negative every tick, so inhibition falls without bound until the interior fires
again — whatever the bias says.

**Forces on a weight** (M1 arm, seed 1, 1,000 trials): 59.1 ticks/trial, rent
0.00531 per edge per trial, |Δw| from reward 0.01170 per edge per rewarded
trial, **ratio 2.20**. An unearning edge born at 0.15 lasts ~24 trials against
a sleep interval of 20.

## Analysis

**H1 survives.** M0's gate is met exactly: two substrates, one nine-member
contract, 001 unchanged to the bit. The interface did its job — the teacher,
the three patterns, the 20-tick/6-fire readout, the trial machinery and manual
mode all drove a completely different substrate with no modification.

**H2 survives, but not for the reason it names.** Design §10 risk 1 predicted
cold start as "the likely failure mode": zero edges → no activity → no output
→ no reward → forever. It did not happen — first reward inside 200 ticks on
all three seeds, structure grown, all three outputs connected, and not one rung
of the mitigation ladder needed. The design's most-feared failure was not the
one that came.

The credit, though, does not belong where the design puts it. §6 names
spontaneous firing as "the bootstrap that makes growth possible at t=0". The
measurements say the **homeostat** is the bootstrap, and a far stronger one:
set `pSpont` to zero and bury the resting bias at −30, and interior firing
climbs from 0.0000 to 0.1499 — dead on `targetSparsity` — within 25k ticks,
because an all-silent sheet drives the homeostatic error negative every tick
and inhibition falls until something fires. Global homeostatic inhibition was
kept, per §6, "because it is the stability lever we already trust". It is also
an activity *source*, and nothing in the design says so.

That is not a curiosity, it is a hole in a pre-registered control. The
`no-spontaneous-activity` arm as specified removes `pSpont` and nothing else,
so it would have wired up normally, connected all three outputs, and been
written down as "the cold-start control did not fail after all" — a false
finding about a control that was never controlling. Corrected to also disable
the homeostat, the arm does exactly what design §8 predicted: 100% silence,
zero connectivity, reward never once delivered. **The design's prediction was
right; its control arm could not have shown it.**

**H3 is dead, and the cause is localised.** The chain from stimulus to reward
has one broken link and the diagnostic points straight at it: **task
information survives exactly one hop.** 7.7σ mean at hop 1 (137 of 178 nodes
carrying real signal) collapses to 1.7σ at hop 2, and hops 3 through 6 sit at
1.8, 1.7, 1.8, 1.9.

Those numbers are not merely "small" — they are the null hypothesis to three
significant figures. The statistic is the range of three per-pattern rate
estimates divided by their standard error, so under "this node's firing does
not depend on the stimulus" it is the range of three independent standard
normals, whose expectation is exactly **3/√π = 1.693**. The measured
hop-2-and-beyond means are 1.7, 1.8, 1.7, 1.8, 1.9. There is no residual
signal being missed at these depths; there is nothing there at all. The output cortex sits at hop 4–6 and reads
1.5–2.9σ, which is nothing. Reward then arrives and modulates eligibility
traces that encode no information about the stimulus, so weights execute a
random walk with a rent-driven pull toward zero. 94% of every generation of
edges dies. That is not a system failing to converge; it is a system with
nothing to converge on.

The shallow arm nails the causal claim down, though not in the way it first
appeared to. It is **not** a depth-0 arm: moving `outputX` to 14 leaves the
median sense pixel three hops from an answer. What it does is bring **19 of 64
pixels inside the one-hop horizon** (6 at one hop, 13 at two), where the M1
arm has none closer than five. Change that and *literally nothing else* — same
rule, same rent, same growth, same fields, same zero-edge start — and the
organism goes from 0.157 to 0.883.

That is a sharper result than "shallow works", because it says what is
carrying the load. Three well-separated 8×8 glyphs are discriminable from a
handful of pixels; nineteen with short access is ample, and the other
forty-five, sitting at three-plus hops in the noise, are presumably
contributing nothing in *either* arm. The substrate is not learning a
distributed representation and then reading it out. It is finding whichever
pixels happen to have a short path and using those. So:

- the growth machinery works
- the rent-and-death selection works
- the three-factor rule works in this substrate
- growing a readout from nothing works
- **carrying stimulus information through more than one grown hop does not**

Note the last bullet's wording, which the corrected measurements forced. The
failure is not that *credit* cannot travel back through layers; it is that
*information* does not travel forward through them. Those are different
problems with different fixes, and the design's §10 risk 5 ("credit may not
reach far synapses") names the wrong one.

Two mechanisms are jointly responsible, and they are separable.

*First, per-hop signal-to-noise.* A node has mean in-degree 5.5, weights born
at 0.15, and homeostatic inhibition holding it at 15% firing. One presynaptic
spike shifts the sigmoid argument by `gain × w = 0.3`, which moves `p` from
about 0.15 to about 0.19 — a four-point rate change against Bernoulli noise.
One stage of that is detectable (hop 1). Two stages is not. Nothing about the
learning rule enters this; it is a property of the units and their sparsity,
and it would bite any rule.

The comparison against 001 makes the size of the gap concrete. A 001 pool
neuron receives **24** synapses at ±0.5; a 002 interior node receives **5.5**
at ±0.15. That is roughly a quarter of the fan-in at a third of the weight, so
the drive a 002 node sees is about an order of magnitude weaker relative to
the same Bernoulli noise, at the same gain and bias. 001 never had to
propagate anything through a second layer, so it never paid for this; 002 does,
twice. Framed that way, the failure is less "reward cannot cross a layer" and
more "these units, wired this sparsely and this weakly, do not transmit". Which
is a hopeful framing and a dangerous one — it is exactly the shape of argument
that leads to turning `birthWeight` up until the number moves, and it is why
decision 2 below exists. The depth ladder (M1d) is what turns it into a claim
that can be wrong.

*Second, and separately: the readout layer is geometrically starved.* Nineteen
edges out of 5,258 terminate in the output cortex. Growth samples a target
among the ~200 sites within `rMax`, weighted by activity and distance, and
**nothing makes a site more attractive for being the site that matters**. The
output cortex is 3 sites out of 1,024, so it gets 3-sites-worth of growth.
001's readout had 480 weights; 002 grew 19, at 0.11 mean magnitude and falling.
Even with a perfect signal arriving, three answer neurons with five to seven
inputs each have very little to work with. This one is a real design defect
rather than a parameter setting, and the design does not currently contain a
mechanism that would fix it.

**What this result actually is.** [L-010](../LEARNINGS.md) observed that 001
"never hit the credit-assignment problem the abstract worries about — with a
single learnable layer feeding the outputs, there is barely any credit to
assign. The variance problem that kills this rule family at scale was never
exercised." Experiment 002's M1 exercised it. It broke at layer **two**, not
at some interesting scale. The abstract's central worry is now a measured
number rather than an anticipation, and the shallow arm confirms 002 is
currently the same one-learnable-layer system 001 was — with the layer grown
instead of given, which is a real result but not a deeper one.

Honest note on the shallow arm's 0.883: per L-010 the baseline to beat is a
random projection, not chance, and the shallow arm has no hidden layer at all.
It should be compared against **M1c** (001 with no pool, sense→output direct),
which is registered but still unrun. Until M1c exists, 0.883 says "the rule
works here", not "grown beats designed".

## Prior art & novelty

- **Similar:** the depth collapse is the textbook variance problem for
  score-function estimators — REINFORCE-style credit degrades with the number
  of stochastic units between the eligibility and the reward, which is the
  standard reason the rule family is confined to shallow settings. The
  per-hop information collapse is also what synfire-chain and polychronization
  work (see [related-work.md](../related-work.md)) addresses with precisely
  timed, strongly convergent connectivity rather than sparse random wiring.
- **Different:** the measurement, mildly. Reporting *where* the signal dies as
  a per-hop discriminability profile over a grown topology, rather than
  reporting only the failed end-to-end score, is not something the entry's
  author has seen presented this way — but no search has been done.
- **Novel (claimed):** nothing. This is a negative result reproducing a
  well-understood limitation in a new substrate. Any temptation to claim the
  hop-profile measurement as novel is marked *unverified against literature*
  and should be dropped unless a real search supports it.

## Learnings

- **L-013:** In a sparse grown substrate of stochastic binary units under
  homeostatic inhibition, **stimulus information survives exactly one hop**:
  7.7σ mean discriminability at hop 1, then 1.7–1.9σ at every hop beyond,
  against a null expectation of exactly 3/√π = 1.693 — so the distal substrate
  carries no signal at all, not a weak one. Competence follows the horizon
  rather than the rule: an arm with no sense pixel closer than 5 hops to an
  answer scores 0.157 (chance), and an arm with 19 of 64 pixels inside 2 hops
  scores 0.883, with growth, rent, death and the three-factor rule identical
  in both. The failure is forward information transmission, **not** backward
  credit assignment — a distinction that changes which fixes are even
  relevant. *Evidence:* this entry, 3 seeds × 2,000 trials per arm, a per-hop
  discriminability profile, and per-pixel path-length distributions.
- **L-014:** Undirected growth starves small targets in proportion to their
  share of sites, not their importance. A 3-site output cortex in a 1,024-site
  lattice received 19 of 5,258 edges, because growth weights candidates by
  activity and distance and nothing makes the site that matters more
  attractive for mattering. Any substrate that grows its own wiring toward a
  small, functionally critical target needs a target-attractiveness term, or
  the readout is starved by geometry before learning is even consulted.
  *Evidence:* this entry, structure census by cortex pair.
- **L-015:** An instrument must suspend the costs the system pays, not just
  the learning it does. The first version of the diagnostic switched rewards
  off but left rent running for a 12,600-tick probe, at which point every
  weight had decayed below the death threshold and it faithfully reported that
  the network was disconnected and carried no information — an artefact of the
  measurement's own duration. Sibling of [L-008](../LEARNINGS.md): there, an
  instrument dimensioned by its own readings; here, an instrument that let the
  system decay while it watched. *Evidence:* first and second runs of
  `exp002-m1-diagnose.ts`, same organism, opposite conclusions.
- **L-016:** A homeostat is an activity *source*, not only a stability lever.
  Global homeostatic inhibition adapting toward a target sparsity will
  regenerate activity from a completely silent substrate — inhibition falls
  without bound while the error term stays negative — reaching the target from
  zero even with spontaneous firing off and the resting bias buried at −30
  (0.0000 → 0.0179 → 0.1499 over 25k ticks; permanently 0.0000 with the
  homeostat off). Consequence for experiment design: the pre-registered
  "no-spontaneous-activity" control removed `pSpont` and nothing else, so it
  would have bootstrapped normally and been recorded as a surviving control
  when it was controlling nothing. **A control arm must be verified to
  actually remove the thing it names**, by measuring the mechanism's absence,
  not by deleting the parameter that appears to own it. *Evidence:* this
  entry, three-window interior firing measurement with the homeostat on and
  off, plus the corrected arm failing outright (100% silence, never rewarded).
- **L-017:** The shortest input→output path is a misleading summary of a grown
  substrate's depth, and reporting it alone produced a wrong claim in the
  first draft of this entry: an arm whose *shortest* path was 1 hop had a
  median sense pixel 3 hops from an answer, and was described as "depth
  removed entirely". Report the per-pixel path-length distribution instead —
  design §8 asked for exactly that and the first implementation quietly
  substituted the minimum. *Evidence:* this entry, shallow arm
  `1-hop 6, 2-hop 13, 3-hop 26, 4-hop 10` against a shortest path of 1.

## Decisions

1. **M1 is recorded as FAILED at the pre-registered configuration.** The gate
   said "if M1 fails, everything below is moot", and M2–M4 are therefore
   **blocked**: a diffusing reward field cannot be shown to orient structure
   toward reward when structure carries no task information to begin with, and
   the M3 shortcut contingency is meaningless without a working multi-hop path.
   No M2 work begins until depth is fixed or explicitly abandoned.
2. **No parameter tuning was performed to rescue M1, and none will be without
   a stated hypothesis.** Design §10 risk 2 is explicit that a system with
   this many knobs can be tuned into producing almost any result. The failure
   is diagnosed, not patched. The one configuration change made (the shallow
   arm) was a pre-registered rung of the §10 risk 1 ladder, run once, reported
   as post-hoc, and is not proposed as the new M1.
3. **The two causes get separate follow-ups**, because they are separable and
   conflating them is how this turns into knob-twiddling:
   - **M1d — the depth ladder.** Sweep output distance across 3 seeds each,
     with the arms defined by the **per-pixel path-length distribution**, not
     by the shortest path — L-017 is what that correction is for. Reported
     measure: tail accuracy against *the fraction of sense pixels within 1 and
     within 2 hops*. Pre-registered prediction: accuracy tracks that fraction,
     and is near chance whenever it is zero. This converts "depth kills it"
     from a two-point comparison into a curve, and it is the honest version of
     the claim.
   - **M1e — the starved readout.** Independently of depth, give growth a
     reason to target the output cortex, and measure whether the readout layer
     reaches a useful size. Candidate mechanism, to be chosen *before* running:
     let the output cortex emit into the activity field at an elevated rate, so
     "grow toward the answer" is expressed in the field that already exists
     rather than as a special case in the growth rule.
4. **M1c is now blocking, not optional.** The shallow arm's 0.883 cannot be
   interpreted without 001's no-pool control. Per L-010 the baseline is a
   random projection, and until M1c is run, no comparison between grown and
   designed structure is admissible.
5. **The design document's risk ranking was wrong and should be updated.**
   §10 lists cold start as risk 1 and "credit may not reach far synapses" as
   risk 5. It was the other way round: cold start never happened, and credit
   reach was fatal. Design §10 to be amended to record this, with the original
   ordering left visible.
6. **Sleep is implemented as a structural-change event, not an S-tick phase.**
   See Deviations.

## Deviations

1. **Sleep is an event, not a phase.** Design §7 specifies "every `K` trials,
   the sense goes dark and the organism runs `S` ticks of sleep". Implemented
   instead as a structural-change event on every K-th blank onset. The reason:
   §7 also specifies "no replay in v1", and with no replay there is nothing for
   those S ticks to compute — growth and death are instantaneous array work.
   The requirement the phase existed to serve is fully met, because rewiring
   still happens only with the sense dark, between trials, never mid-thought.
   The S-tick phase is owed back when replay arrives. Detecting the blank onset
   from the sense itself is also what keeps sleep off `OrganismLike`, which
   matters: adding a `sleep()` member would have broken the drop-in swap that
   M0 exists to establish.
2. **The reward field is solved to steady state once rather than integrated
   per tick.** Exact rather than approximate — the source is stationary and
   reward is instantaneous, so the profile the weight update reads is always
   the steady state. Noted because "we ran a diffusion equation" and "we solved
   one" are different claims and the code does the latter.
3. **The reward profile is normalised to mean 1 over the lattice**, a choice
   the design does not specify. Peak-normalisation would have made the
   diffusing arm deliver strictly less total credit than the uniform control,
   confounding "credit placed by geometry" with "less credit". Recorded because
   it will affect every M2 comparison.
4. **Two seeds, not three, on the first shallow run** (1,000 trials) before
   extending to three seeds at 2,000. Only the 3-seed/2,000-trial numbers are
   reported as the arm's result.
5. **The lattice's off-axis constraint was tightened during the build**, from
   "reward locus is not exactly on the input→output segment" to "is further off
   it than one growth step". The original was vacuous — floating point makes
   exact-zero distance unreachable, so a locus half a site off the axis passed.

## Threats to validity

1. **The discriminability profile is one seed.** The hop-2 collapse is a large
   effect (7.7σ → 1.7σ) and the end-to-end failure replicates on three seeds,
   but the per-hop profile itself was measured on seed 1 only. M1d should
   report it per seed.
2. **σ is not information.** Spread-in-standard-errors is a crude proxy for
   how much a node tells you about the stimulus; a node could carry
   pattern-specific *timing* or correlation structure with a flat mean rate and
   this measure would call it noise. The end-to-end failure is consistent with
   the profile, but the profile alone does not prove the information is absent
   — only that it is absent from mean firing rate.
3. **The shallow arm changes geometry, not depth alone.** Moving the output
   cortex to x=14 shortens the path *and* places the outputs in a differently
   connected part of the sheet. The M1d depth ladder is what separates those;
   until it runs, "depth is the cause" is well-supported but not isolated.
4. **The shallow arm may be solving an easier problem than it looks.** If ~19
   short-access pixels carry it, the arm demonstrates that three well-separated
   glyphs are discriminable from a handful of pixels — which is true and not
   very demanding. A task whose classes are not separable from any small pixel
   subset would test this properly, and the M1 patterns were chosen for
   distinctness. This weakens what the 0.883 licenses, and it does not weaken
   the M1 failure.
5. **Three of this entry's numbers were wrong before review and are corrected
   here**, which is itself a threat to the ones that were not reviewed: the
   reward/rent ratio (1.81 → 2.20, a probe-contaminated tick count), the
   shallow arm's depth (called "depth removed entirely"; actually a median of
   3 hops), and the cold-start control (which did not control). All three were
   caught by running the code rather than reading it. Numbers in this entry
   that were not independently re-run should be read with that in mind.
6. **The parameter set is one point in a large space.** Twenty-five knobs, most
   chosen by derivation or by copying 001 rather than by search. A different
   `gain`/`bias`/`targetSparsity`/`birthWeight` combination could plausibly
   raise per-hop SNR enough to change the result. This is exactly the §10
   risk 2 problem, and the honest statement is that M1 failed *at this
   configuration*, not that it is unreachable.
7. **Silence is scored as incorrect**, inherited from 001. With 40–45% silent
   trials in the M1 arm, the headline 0.157 conflates "wrong" with "did not
   answer". The conditional-on-answering figure (≈0.33) is the one that
   supports the "exactly chance" reading, and it is reported alongside.
8. **The reward-versus-rent force ratio (2.20) is a single aggregate**, from
   one arm and one seed, measured on the trained organism. It says reward moves
   a weight a little over twice what rent takes, with near-random sign, which
   supports the random-walk reading without establishing it.

## Next

**M1d — the depth ladder.** Output cortex at path lengths 1, 2, 3 and 4 hops,
every other parameter fixed at the M1 configuration, 3 seeds each, 2,000
trials, with the per-hop discriminability profile reported per seed.

**Pre-registered gate:** none — this is a measurement, not a gate. **Decision
rule stated before running:** if accuracy at 2 hops is already at chance while
1 hop clears 0.80, the depth diagnosis is confirmed and the F-track question
becomes "what mechanism carries information across a hop", which is a design
question, not a tuning one. If instead accuracy degrades gradually with depth,
the diagnosis is wrong and the failure is a scaling problem rather than a
structural one.

**M1c** (001 with no pool, sense→output direct) runs alongside it, since the
shallow arm cannot be interpreted without it.
