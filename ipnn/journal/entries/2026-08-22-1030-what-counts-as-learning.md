# What counts as learning: the measurement doctrine we never had

- **Entry:** `2026-08-22-1030-what-counts-as-learning`
- **When:** 2026-08-22 10:30–12:15 CDT
- **Who:** Javid + Claude (session)
- **Experiment:** theory / direction — bears on [002](../../experiments/002-grown-substrate/design.md) and on every experiment after it
- **Code state:** git `1a03b72` + this entry's changes; instruments added in the same session (`tools/exp002-longrun.ts`, `tools/exp002-ceiling.ts`, `tools/exp002-sweep*.ts`)
- **Re-run:** `npx vite-node tools/exp002-longrun.ts 32000 1`
- **Attribution:** this entry uses the `[J]` / `[C]` / `[J→C]` marks defined in [README.md](../README.md#attribution-who-thought-of-it). It is the first entry to do so.

## In plain words

This entry has no experiment in it. It is the day the project noticed it had
been judging itself with the wrong instrument.

Javid ran the demo, saw the failing arm sitting at 10–25% forever, and asked
the obvious hard question: **how do you know it will never learn?** A living
thing might take a very long time. A human takes years. If this were a
genuinely powerful learner that needed three weeks instead of four thousand
trials, everything we have written down would look exactly the same.

That is a completely fair challenge, and the honest answer was that our
evidence was a snapshot — one measurement, at one moment — which cannot tell
"nothing is here" apart from "something is here, arriving slowly."

Then came the bigger point: **stop grading a living system with the report
card built for a benchmark.** Accuracy on one fixed task is what the rest of
the field measures. The thing we actually care about is different: teach it
one thing, then teach it a second thing, then check it can still do the first.
Something that learns slowly but *carries what it learned into a new problem*
is more intelligent than something that scores higher on one task and can do
nothing else.

The striking part is that this was already written down. The vision document's
fourth pillar says the network should learn one sense, then another, "then
return to the first and *still function*." Its success list says "you teach it
a second task, return to the first, and it has not forgotten." **We wrote the
right definition of intelligence two years ago and then never once measured
it.** Every gate since has measured accuracy on one fixed task.

So this session built the missing half: a ladder of evidence that looks for
learning at seven levels from structure upward, a ledger for hypotheses
alongside the one for findings, and a rule that every new instrument must
state in advance what result would count as *not learning* — because "it is
learning, you just cannot see it yet" is an idea that can eat a research
project.

## Objective

Answer a direct challenge to the M1 negative result, and convert the response
into doctrine rather than a one-off rebuttal.

## The challenge, as posed

> "This seems like a foundational part of determining if there is any signal
> developing at all, is that right? How do we know that a signal WON'T
> develop, given that this is a living / adapting system? How do we know that
> the total ticks that we listened to is just not enough. A truly intelligent
> system (e.g. a human) might take years to learn something. How do we know we
> don't have a SUPER powerful learning engine that just takes 2-3 weeks to
> learn, not just 4000 ticks?" **[J]**

And then, decisively:

> "I don't want you to evaluate things only using traditional methods that we
> use on traditional ML models. We have to come up with NEW ways of detecting
> learning and intelligence. And we have to be ok with slow growth / slow
> learning. It might take a month to build a model that learns something, but
> then after that we can take that same intelligent learning system and adapt
> it to a different problem and have it learn that as well. Then try it
> against both the original and the new problem. That is true intelligence."
> **[J]**

## What the challenge was right about

Three things, conceded without reservation.

1. **The M1 diagnosis was a snapshot.** Discriminability was measured once,
   after 1,000 trials. A snapshot cannot distinguish absence from slowness.
   The claim "there is nothing to converge on" was stated more strongly than
   a single time point supports. **[C, conceding]**
2. **The measure was insensitive by construction.** Per-node firing-rate
   spread asks whether *one* node's rate depends on the stimulus. Two hundred
   nodes each carrying a tenth of a sigma would show nothing per-node while
   the population carried plenty.
3. **Task accuracy is the wrong primary instrument for a living system**, and
   it is borrowed from precisely the benchmark culture
   [vision.md](../../vision.md) says this project is not part of. **[J]**

## What was built in response

**A trajectory instrument with a positive control** (`tools/exp002-longrun.ts`).
Runs 32,000 trials — 16× the gate — and at checkpoints across the whole run
measures, with learning off and rent suspended (L-015):

- per-node discriminability at each hop depth
- **population decoding**: nearest-centroid classification of the label from
  the whole population at each depth, on held-out windows. Far more sensitive
  than per-node rates.
- **structural persistence**: what fraction of the edges alive at one
  checkpoint are still alive at the next
- and all of it on the *shallow* arm too, as a positive control. A metric flat
  in both arms is a broken instrument, not evidence. **[C]**

**The null, stated in closed form.** The per-node statistic is the range of
three per-pattern rate estimates in standard errors. Under "this node does not
care which pattern is showing" that is the range of three standard normals,
whose expectation is exactly **3/√π = 1.6926**. Monte Carlo over 200,000 draws
returns 1.6913. The same simulation gives **P(range > 3σ) = 8.5%** under the
null — so "21 of 199 nodes exceeded 3σ" (10.6%) is also what noise does, and
the tail was never evidence either. **[C]**

## Results (interim — the 32,000-trial run is still going)

Null calibration, verbatim:

```
closed form E[range] = 3/√π = 1.6926   Monte Carlo = 1.6913
P(range > 3σ) under the null = 8.5%
population decoding chance = 0.333
```

M1 arm, seed 1, through 4,000 trials:

```
trials   acc     hop2 σ   hop2 decode   edges   persistence
   250   0.190    1.84       0.367       6046      n/a
   500   0.170    1.72       0.400       5945       5%
  1000   0.195    1.82       0.356       4797       5%
  2000   0.205    1.77       0.356       6406       6%
  4000   0.200    1.73       0.400       5211       5%
```

Shallow arm (the positive control), same instrument:

```
trials   acc     hop2 σ   hop2 decode   edges   persistence
   250   0.300    1.95       0.439       6701      n/a
   500   0.470    1.85       0.450       7208       9%
  1000   0.590    2.08       0.650       8513       8%
```

## Analysis

**The instrument is not blind.** Hop-2 population decoding rises 0.367 → 0.650
in the shallow arm while accuracy climbs. The same measure on M1 sits at
0.356–0.400 against a 0.333 chance line, unmoved across a sixteen-fold range
of training. So the flat M1 trajectory is a property of the M1 arm, not of the
measure — which is exactly what the positive control was built to establish,
and it is the reason the M1 flatness now means something the snapshot did not.

**But the mechanical argument is the one that actually settles H-002 for this
substrate, and it is not a measurement argument at all.** Structural
persistence is pinned at 5–6% between checkpoints, from trial 250 to trial
4,000. Roughly 94% of every generation of edges dies. Nothing in the substrate
has a time constant longer than one sleep interval: weights decay under rent
on ~20 trials, edges die on the same schedule, the activity field's time
constant is 200 ticks. The system reaches a steady state within a few sleeps
and then maintains it.

That yields the sharpest thing in this entry: **slow learning requires a slow
variable** (H-003) **[C]**. "Maybe it needs a month" is not a question about
our patience, it is a question about the design — and it is answerable by
asking what the longest-lived quantity in the system is. Here, nothing
accumulates, so more time buys only more steady state. This is not an argument
that H-002 is wrong in general; it is an argument that **002 as currently
designed is structurally incapable of being the slow learner H-002 imagines**,
and that the fix is to give it a slow variable rather than to wait.

Which leads straight to H-004 **[C]**: the substrate does have exactly one
monotonically accumulating quantity — the evidence count `n`, which design §7
notes only ever increases — and it is stored **on edges, 94% of which die
every generation.** The one slow variable in the system is carried by its most
ephemeral object and destroyed with it. Moving it to the *site*, as a
per-location trace that outlives the edges that created it and biases where
regrowth goes, would give the substrate its first genuine long-term memory.
That is a design change with a clear rationale rather than a knob to turn.

**On the reframing.** The demand for new detection methods **[J]** is correct
and is now doctrine: [vision.md](../../vision.md) gains a ladder of evidence
with seven rungs, each with its own null. The rungs this project actually
cares about are 4 through 7 — savings, transfer, retention, learning-to-learn
— and rung 4 deserves special note: **savings can detect a latent trace inside
a curve that never left chance.** A system can know something it cannot yet
do. It is Ebbinghaus's 1885 method, it is not an ML measure, and it is the
correct instrument for exactly the situation H-002 describes. **[J→C]**

**And the discipline that has to come with it. [C]** "It is learning, just
slowly and invisibly" is unanswerable and can absorb every negative result
forever. The ladder is worth building *and* is precisely the shape of belief
that turns research into faith. The defence adopted is one rule: no instrument
enters the record without stating what result would count as no-learning.
Every rung names its null for that reason.

**What this says about the project's history.** Pillar 4 and success-criterion
4 have said "teach it a second task, return to the first, and it has not
forgotten" since the vision was written. Every gate since — 001's M1, 002's M1
— has measured single-task accuracy. The definition of intelligence was
correct and dormant, and the experiments drifted toward the benchmark measure
because the benchmark measure is the easy one to compute. That drift is worth
recording as its own learning.

## Prior art & novelty

- **Similar:** savings is Ebbinghaus (1885). Catastrophic forgetting and
  transfer are the standard continual-learning battery (EWC, progressive nets,
  and the rest of that literature). Population decoding of held-out windows is
  ordinary systems neuroscience. Structural persistence measures are standard
  in synaptic-turnover imaging work. **None of this is new; what is unusual is
  only that it is being applied as the primary gate rather than as a follow-up
  to an accuracy number.**
- **Different:** the ladder combines rungs that normally live in separate
  literatures (structural turnover, population decoding, savings, continual
  learning) into one evidence hierarchy for a single system.
- **Novel (claimed):** none. Marked *unverified against literature*; the
  continual-learning and metaplasticity literatures almost certainly contain
  closer analogues to H-004's site-trace than we have looked for.
  [related-work.md](../related-work.md) is owed entries for savings,
  catastrophic forgetting, and synaptic turnover before anything here is
  claimed.

## Learnings

- **L-018:** A negative result from a snapshot is weaker than it looks. "The
  signal is absent" and "the signal has not arrived yet" are indistinguishable
  at one time point, and the fix is cheap: measure the trajectory, and carry a
  positive control so a flat line can be attributed to the system rather than
  to the instrument. *Evidence:* this entry — the same measure that reads flat
  on M1 across a 16× range of training rises 0.367 → 0.650 on the shallow arm.
- **L-019:** Slow learning requires a slow variable. A system whose every
  state variable has a short time constant reaches steady state and stays
  there, so "give it more time" cannot help it. Before running anything
  longer, ask what the longest-lived quantity in the mechanism is and whether
  it is accumulating. In 002 nothing is: structural persistence is flat at
  5–6% from trial 250 to trial 4,000. *Evidence:* this entry.
- **L-020:** Process — a project drifts toward whichever measure is cheapest
  to compute. IPNN's vision has defined intelligence as transfer-plus-
  retention since it was written, and every gate since has measured
  single-task accuracy instead, because accuracy is one number and transfer is
  a protocol. Stating the definition is not enough; it has to be the thing the
  gate tests. *Evidence:* vision.md pillar 4 and success-criterion 4 against
  the gates in 001 and 002 design §8.

## Decisions

1. **Adopt the ladder of evidence** as the project's measurement doctrine
   ([vision.md](../../vision.md)). Behavioural accuracy is demoted from *the*
   gate to rung 0 of seven.
2. **Every new instrument ships with a null and a pre-registered
   decision rule.** Non-negotiable, and the direct defence against H-008.
3. **Add a hypotheses ledger** ([HYPOTHESES.md](../HYPOTHESES.md)) alongside
   the learnings ledger, with `H-###` IDs, statuses, and attribution. H-001
   through H-008 registered.
4. **Adopt the attribution convention** `[J]` / `[C]` / `[J→C]` / `[C→J]`,
   recorded live and never reconstructed (journal rule 7). **[J]**
5. **Journal rule 6 added: thoughts are first-class.** A realisation or a
   reframing gets an entry the day it happens, with or without code. **[J]**
6. **Experiment 003 — transfer and retention — is now the priority**, and
   notably **it does not need 002 to work.** It should run on 001, which
   demonstrably learns. The project has never once tested its own definition
   of intelligence, and the substrate to do it on already exists. **[J→C]**
7. **002's next design change is a slow variable** (H-004's site trace), not a
   parameter search. The sweep continues to completion for the record, but it
   is no longer expected to be the thing that matters.

## Deviations

The 32,000-trial run is still executing at the time of writing; the table
above is truncated at 4,000 trials and marked interim. The entry is being
filed now rather than held, per rule 6 — the realisations are the content, and
the completed trajectory will be appended in a follow-up entry rather than by
editing this one (rule 1).

## Threats to validity

1. **The persistence measure is coarse.** "Fraction of edges alive at
   checkpoint N still alive at N+1" over intervals that double in length is
   not a survival curve, and a doubling interval makes the numbers not
   directly comparable across rows. A proper per-edge age distribution would
   be better and is not implemented.
2. **One seed.** The trajectory is seed 1 only. The flatness is consistent
   with the three-seed M1 gate, but the trajectory claim itself is n=1.
3. **Nearest-centroid decoding is still a rate measure.** It pools across
   nodes, which is the sensitivity gain, but a substrate carrying information
   in spike *timing* or in pairwise correlation would still read as chance.
   Rung 1 of the ladder is therefore weaker than it sounds, and the honest
   statement is "not decodable from population rates".
4. **The mechanical argument (L-019) assumes the measured timescales are the
   only ones.** If some slow quantity exists that we have not thought to
   measure, the conclusion changes. The argument is only as good as the
   inventory of state variables, which is why design §11's knob table now
   exists.
5. **This entry is mostly reasoning, not results.** Its central claims are a
   doctrine and a mechanism argument. Both could be wrong in ways no
   measurement here would catch.

## Next

**Experiment 003 — transfer, savings and retention.** Design owed before any
code. The protocol Javid stated, formalised **[J→C]**:

1. Teach task A (the three M1 patterns) to criterion.
2. Teach task B (three *new* patterns) to criterion.
3. Test A and B both.

With the control arms that make it mean anything, pre-registered now:

- **B-from-scratch** — a naive organism learning B, so "B came faster" has a
  baseline (rung 5, transfer).
- **A-immediately-after-A** — so "A survived B" has a reference (rung 6,
  retention).
- **A-relearn versus never-taught** — savings (rung 4), the instrument that
  can find a trace inside a flat curve.

**Pre-registered gates.** Transfer: B reaches criterion in fewer trials than
B-from-scratch, on ≥3 seeds. Retention: A after B stays within 0.10 of A
immediately after A. Savings: re-learning A is faster than a never-taught
control by more than the seed spread. **Decision rule if it fails:** 001 is a
single-layer readout on a random projection (L-010), and a single layer has
nowhere to put task-general structure, so a null result would be evidence
about *that architecture* and not about the paradigm — which is itself worth
knowing, and is the first real use the project will have made of its own
definition of intelligence.

Runs on 001. Does not wait for 002.
