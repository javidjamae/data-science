# Latent learning: something happens during unrewarded time, but it isn't the task

- **Entry:** `2026-08-22-1240-latent-learning`
- **When:** 2026-08-22 12:40–12:55 CDT
- **Who:** Javid + Claude (session)
- **Experiment:** the H-010 test, on 001 and on 002's shallow arm
- **Code state:** git `dad69ef`
- **Re-run:** `npx vite-node tools/latent-learning.ts 001 400 1 2 3 4 5` · `... shallow 400 1 2 3`

## In plain words

Tolman's 1930 experiment, run on both substrates. Let the organism see the
patterns with **no reward at all**, then switch reward on and count how many
trials it needs to reach a standard. If a pre-exposed organism gets there
faster than one starting cold, it was learning during a period when its
behaviour showed nothing — which is precisely the claim "maybe it's learning
invisibly" makes, and the reason this experiment exists.

There is a third arm, and it turned out to be the one that mattered: an
organism pre-exposed to the same amount of world with the pairings
**scrambled**. It sees as much, for as long, with nothing stable to learn. If
it speeds up too, the speed-up was never about the patterns.

**001: no effect, and none was possible.** Its rule is `Δw = η·R·e`. Set
reward to zero and *nothing moves at all* — so this test is negative by
construction for any purely reward-gated learner. Worth knowing as a limit of
the instrument rather than a fact about the world.

**002's shallow arm: a real speed-up that the control ate.** Pre-exposed
organisms reached the standard in a median of 659 trials against 746 for
naive ones. That looks like latent learning until you see the scrambled arm:
**535** — faster still. So the grown substrate genuinely does gain something
from unrewarded time, and it is **structural warm-up, not knowledge of the
task**.

That is a real difference between the two substrates and it is worth stating
plainly: 002 has a channel for acquiring things without reward that 001
structurally lacks, because growth and rent run whether or not reward ever
arrives. It just isn't acquiring the task through it.

## Results

Trials to criterion (rolling-100 ≥ 0.70, cap 3000), verbatim:

```
001, seeds 1-5              per seed                    median   final acc
  naive                     153  162  171  287  220       171      0.988
  pre-exposed, unrewarded   175  196  111  159  164       164      0.994
  scrambled control         195  167  165  162  176       167      0.990

002 shallow, seeds 1-3      per seed                    median   final acc
  naive                    1318  649  746                 746      0.873
  pre-exposed, unrewarded   659  740  289                 659      0.917
  scrambled control         735  535  374                 535      0.903
```

Pre-exposure length 400 trials in both.

## Analysis

**H-010 is answered in the negative for both substrates, and the scrambled
arm is the whole reason the answer is trustworthy.** On 002, comparing only
naive (746) against pre-exposed (659) would have read as a 12% speed-up and
been written up as latent learning. The scrambled control at 535 kills it:
whatever the benefit is, an organism that saw *unstructured* pairings got more
of it. This is [L-016](../LEARNINGS.md) recurring — a control arm has to
actually remove the thing it names — and it is the second time in two days
that a control has overturned a result that looked positive without it.

**But something real is happening during unrewarded exposure in 002**, and
that is a genuine asymmetry with 001. Both pre-exposed arms beat naive by a
wide margin (659 and 535 against 746, and naive's worst seed took 1318). The
grown substrate arrives at the rewarded phase already wired, already at its
steady-state edge population, already past the cold-start period — and that is
worth several hundred trials. It is warm-up in the most literal sense.

The finding is therefore split, and both halves matter:

- 002 **can** acquire something without reward. 001 cannot, at all, by
  construction. That is the growth machinery doing something the fixed
  substrate has no mechanism for.
- What 002 acquires is **not task-specific**. Scrambled exposure buys the same
  benefit or more.

**On H-002.** This does not rescue it. The unrewarded phase produces structure,
not knowledge, and [L-019](../LEARNINGS.md)'s point stands: the structure it
produces reaches steady state and stops. A warm-up effect is not a slow
learning curve — it is a fast one that has already finished.

## Learnings

- **L-024:** Experiment 002 acquires something from unrewarded exposure and
  experiment 001 cannot — growth and rent run without reward, where
  `Δw = η·R·e` is identically zero — but what 002 acquires is **structural
  warm-up, not task knowledge**: a scrambled-pairing control gets the same
  benefit or more (median 535 vs 659 trials to criterion, against 746 naive).
  Corollary for any latent-learning claim: the naive-vs-pre-exposed
  comparison is not sufficient, because "has been running a while" and "has
  learned something" both make it faster. *Evidence:* this entry.
- **L-025:** A latent-learning test is uninformative by construction for a
  purely reward-gated learner. With `Δw = η·R·e` and R=0, no weight can move
  during pre-exposure, so a null is guaranteed and says nothing. Check that a
  substrate *has* an unrewarded learning channel before using this instrument
  on it. *Evidence:* 001's three arms within 7 trials of each other.

## Threats to validity

1. **Three seeds on 002, with enormous variance** — naive alone spans 649 to
   1318 trials. The medians are separated by less than that spread, so the
   ordering of the three arms is not established, only that both pre-exposed
   arms beat naive.
2. **One pre-exposure length** (400 trials). Latent learning could require
   longer exposure, and Tolman's rats had ten days.
3. **Criterion 0.70** was chosen so 002 could reach it at all; a different
   criterion could reorder the arms.
4. **The scrambled control still shows the patterns**, just mispaired — so it
   controls for warm-up and for stimulus statistics, but not for "having seen
   these three images specifically".

## Next

Unchanged: experiment 003. This entry closes H-010 rather than opening
anything, and the honest summary is that the one instrument built specifically
to detect invisible learning found warm-up instead — which is exactly what it
was supposed to be able to distinguish.
