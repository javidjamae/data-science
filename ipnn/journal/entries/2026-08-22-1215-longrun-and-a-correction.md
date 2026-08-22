# The 32,000-trial run, and a positive control that wasn't one

- **Entry:** `2026-08-22-1215-longrun-and-a-correction`
- **When:** 2026-08-22 12:15–12:40 CDT
- **Who:** Javid + Claude (session)
- **Experiment:** [002](../../experiments/002-grown-substrate/design.md) — the H-002 trajectory test
- **Code state:** git `386cb78`
- **Re-run:** `npx vite-node tools/exp002-longrun.ts 32000 1`
- **Follows:** [2026-08-22-1030](./2026-08-22-1030-what-counts-as-learning.md), which reported this run's first five checkpoints as interim. Per journal rule 1 that entry is not edited; this one corrects it.

## In plain words

The long run finished. It answers the question it was built for, and it also
shows that one of the things said about it yesterday afternoon was wrong.

**The answer:** the failing arm did not start learning. Not at four thousand
trials, not at thirty-two thousand — a hundred and twenty-eight times longer
than the first measurement. Accuracy, signal, and structural turnover are all
exactly where they were at trial two hundred and fifty.

**The correction:** to prove the measuring instrument wasn't simply broken, it
was also run on the arm that *does* learn, and at the thousand-trial mark that
arm showed a clear signal where the failing one showed none. That was reported
as proof the instrument works. With the whole run visible, it doesn't hold up.
The working arm's deep-layer signal bounces around with no trend and ends up
near chance — *while its accuracy climbs to 90%.*

Which is a more interesting fact than the one it replaces. **The working arm
gets good without ever using its deep layers.** It succeeds on the short path
and the depth stays noise throughout. So there is no arm anywhere in this
experiment where depth carries the answer — and that means the instrument
built to detect signal-at-depth has never been shown to be able to detect
signal at depth. It has only been shown to detect signal.

The conclusion about the failing arm survives, because it rests on something
else entirely: nothing in that system accumulates. But the confidence
attached to it yesterday was borrowed from a control that was not controlling
what it claimed to.

## Results

M1 arm, seed 1, complete (chance 0.333; per-node null 1.693):

```
 trials |  acc  | hop1 σ | hop2 σ | hop2 decode | hop3 decode | edges | persist
   250  | 0.190 |  7.66  |  1.84  |    0.367    |    0.339    |  6046 |   n/a
   500  | 0.170 |  7.42  |  1.72  |    0.400    |    0.294    |  5945 |    5%
  1000  | 0.195 |  6.10  |  1.82  |    0.356    |    0.339    |  4797 |    5%
  2000  | 0.205 |  9.01  |  1.77  |    0.356    |    0.367    |  6406 |    6%
  4000  | 0.200 |  7.10  |  1.73  |    0.400    |    0.317    |  5211 |    5%
  8000  | 0.170 |  6.16  |  1.66  |    0.322    |    0.272    |  5593 |    5%
 16000  | 0.185 |  6.74  |  1.80  |    0.356    |    0.367    |  4756 |    5%
 32000  | 0.230 |  8.51  |  1.75  |    0.372    |    0.367    |  5414 |    5%
```

Shallow arm, same instrument, same seed:

```
 trials |  acc  | hop1 σ | hop2 σ | hop2 decode | hop3 decode | edges | persist
   250  | 0.300 |  9.01  |  1.95  |    0.439    |    0.383    |  6701 |   n/a
   500  | 0.470 |  9.61  |  1.85  |    0.450    |    0.283    |  7208 |    9%
  1000  | 0.590 |  9.40  |  2.08  |    0.650    |    0.467    |  8513 |    8%
  2000  | 0.890 |  6.79  |  1.76  |    0.361    |    0.328    |  6280 |    6%
  4000  | 0.795 |  9.47  |  1.86  |    0.517    |    0.411    |  7865 |    7%
  8000  | 0.800 |  9.18  |  2.05  |    0.478    |    0.439    |  7083 |    7%
 16000  | 0.850 |  7.86  |  1.82  |    0.361    |    0.311    |  7748 |    7%
 32000  | 0.905 |  7.79  |  1.83  |    0.394    |    0.356    |  6897 |    6%
```

Sweep, concurrently: **327 configurations** drawn from the 24-knob space with
M1 geometry pinned. Best worst-seed tail **0.38**. None ≥ 0.50; none ≥ 0.80.
Seven of 327 got any sense pixel within two hops.

## Analysis

### What the M1 arm did over 128× the original observation window

Nothing. Accuracy 0.190 → 0.230, inside its own noise. Per-node
discriminability at hop 2 sits between 1.66 and 1.84 across every checkpoint,
straddling the 1.693 null. Population decoding at hop 2 moves +0.006 across
the whole run, against a chance line of 0.333. Structural persistence is 5% at
trial 250 and 5% at trial 32,000.

**H-002 gets no support here.** Not "we did not look long enough" — we looked
128 times longer and every measure is where it started.

### The correction: a positive control has to control for the right thing

Yesterday's entry reported that hop-2 decoding "rises 0.367 → 0.650 in the
shallow arm while M1 sits at chance", and concluded the instrument was not
blind. Two things are wrong with that now the run is complete.

First, 0.650 was **one checkpoint**, not a trend. The full shallow series runs
0.439, 0.450, 0.650, 0.361, 0.517, 0.478, 0.361, 0.394 — it wanders. With
~120 held-out windows per checkpoint the standard error near 0.4 is about
0.045, so 0.650 is a real excursion rather than pure noise, but it is an
excursion and not a rise. Reporting the interim trend from five checkpoints
was reporting the shape of an incomplete curve.

Second, and much more importantly: **the shallow arm ends at 0.905 accuracy
with hop-2 decoding at 0.394**, a hair above chance. Its deep layers are not
carrying the answer at the point where it is most competent.

That is a genuinely interesting finding and it strengthens
[L-013](../LEARNINGS.md): the organism does not overcome the one-hop wall, it
*routes around* it. It becomes competent on the short path and leaves the
depth as noise — which is exactly what a substrate with a one-hop information
horizon should do if it can.

But it also breaks the control. The shallow arm was supposed to demonstrate
"this instrument can detect signal at depth when signal at depth exists". It
demonstrates no such thing, because in the shallow arm signal at depth never
exists either. **There is no arm anywhere in experiment 002 where depth
carries the answer**, so the depth-decoding instrument has never been
validated on the case it was built for. It has been validated only in the weak
sense that it reads above chance sometimes.

### What survives, and why

The conclusion about the M1 arm does not depend on the broken control, which
is the only reason it survives. It rests on two things:

1. **Every measure is flat**, including two — accuracy and structural
   persistence — that need no decoder at all.
2. **The mechanistic argument** ([L-019](../LEARNINGS.md)): persistence pinned
   at 5% means ~94% of every generation of edges dies, nothing in the
   substrate has a time constant longer than one sleep interval, and a system
   with no slow variable reaches steady state and stays there. That argument
   is about the design, not about the measurement, and 32,000 trials of flat
   persistence is direct evidence for it.

So the finding stands and the confidence stated yesterday was partly borrowed.
Recording that distinction is the point of this entry.

### The shallow arm was not the hard plateau it was called earlier

Measured at 2,500 trials it looked like a ceiling at ~0.87. Across the full
run it reads 0.890, 0.795, 0.800, 0.850, 0.905 from trial 2,000 on — a
fluctuating band with its highest value at the end and its lowest at trial
4,000. Calling that either "a plateau" or "still climbing" over-reads one
seed; the honest statement is a noisy band around ~0.85 whose endpoint happens
to be the maximum, and settling it needs more seeds.

Worth conceding anyway: **it is not a hard ceiling reached at trial 2,000**,
which is what was said earlier from a shorter run. The instinct that we may be
cutting runs off early has some support — in the arm that can learn at all.

## Learnings

- **L-021:** A positive control must control for the *specific* thing being
  measured, not for a nearby easier thing. The shallow arm was used to show
  that a depth-decoding instrument can detect signal, but the shallow arm
  never carries signal at depth — it becomes competent on its short path, with
  hop-2 decoding at 0.394 while accuracy is 0.905. So the instrument was
  validated for "can detect signal" when the claim needed "can detect signal
  at depth", and no arm in the experiment provides the latter. Before trusting
  a null from an instrument, check that some arm produces a *positive* on the
  exact quantity the null is about. *Evidence:* this entry, complete 8-point
  trajectories on both arms.
- **L-022:** Reporting the shape of an incomplete curve is a distinct failure
  mode from reporting a wrong number. Five of eight checkpoints showed a rise
  that the remaining three erased. Interim results should be reported as
  levels, not as trends. *Evidence:* the shallow hop-2 series, 0.439 → 0.650 →
  0.361 → … → 0.394.
- **L-023:** The grown substrate does not defeat its one-hop horizon, it
  routes around it. Given a short path the organism uses it and leaves depth
  as noise; denied one, it fails rather than building depth that works. Depth
  in this substrate is not a resource it declines to use — it is a cost it
  avoids when it can and drowns in when it cannot. *Evidence:* shallow arm at
  0.905 accuracy with hop-2 decode 0.394.

## Decisions

1. **H-002 marked `refuted (for 002's M1 arm)`**, with the scope stated
   explicitly: refuted for this substrate under 128× the original window, not
   refuted as a general principle. The general claim remains open and is now
   H-002's surviving half.
2. **A depth-carrying positive control is owed before any future null about
   depth.** The obvious candidate: a hand-wired organism with a known
   multi-hop path that provably carries the label, run through the same
   instrument. Until that exists, every depth-decoding null in this project
   carries an asterisk.
3. **The sweep continues to completion** for the record (327/2000 at time of
   writing, best 0.38), but it is not expected to change anything, and
   [L-019](../LEARNINGS.md) is the reason: no setting of 24 knobs adds a slow
   variable that the design does not have.

## Threats to validity

1. **One seed, both arms.** The M1 flatness is corroborated by the three-seed
   gate; the shallow trajectory is not corroborated by anything.
2. **The decoding measure remains rate-based** (threat 3 of the previous
   entry, unchanged): timing or correlation structure would read as chance.
3. **Persistence intervals double**, so the 5% figures are not strictly
   comparable row to row. Flatness across doubling intervals is arguably
   *stronger* than flatness across equal ones, but the measure is still coarse.
4. **This entry's central correction rests on the same single seed** it is
   correcting. A second seed could show the shallow hop-2 series doing
   something else entirely.

## Next

Unchanged: **experiment 003, transfer and retention, on 001.** Plus the
depth-carrying positive control from decision 2, which is small and now
blocking any further claim about depth.
