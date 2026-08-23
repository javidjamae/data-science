# Five hypotheses from a kitchen-table conversation

- **Entry:** `2026-08-22-1340-kian-conversation`
- **When:** 2026-08-22 13:05 CDT (the conversation) · 13:40 CDT (this entry)
- **Who:** Javid, in conversation with his son Kian
- **Experiment:** theory / direction — bears on 002's open gap and on experiment 003
- **Source:** [full verbatim transcript](../artifacts/2026-08-22-kian-conversation.md), Granola id `81158a24-b85a-465f-a6e0-2e2d4eb6bee0`
- **Attribution:** everything here is **[J]**, per Javid's instruction. The transcript is a single undiarised stream — Granola did not mark speaker turns — so the ideas are recorded under his mark with the source conversation cited, rather than reconstructed turn by turn. Reconstructing attribution after the fact is exactly what [journal rule 7](../README.md#attribution-who-thought-of-it) forbids.

## In plain words

Javid talked through the research over ice cream. No code ran. Five ideas came
out of it that were not in the project before, and two of them land directly on
the wall the last two days of work ran into.

The one worth leading with: **the ultimate reward should be passing on what you
learned.** Today's measurements found that 002 builds a small, strong,
well-evidenced core of connections and then never grows it, because nothing
lets what has already formed influence what forms next
([L-028](../LEARNINGS.md)). This conversation proposes a completely different
answer to the same problem — don't make the core scaffold *within* one
organism's life, make successful organisms *seed the next one*. Same gap,
opposite direction.

The other one that matters: **reward should be unlimited, punishment should be
limited, and running out of punishment budget should kill you.** Not a bad
score — death. That is not a new mechanism for this project so much as a
recognition that it already exists one level down: 002's connections pay rent
every tick and die when they can no longer pay. The proposal is to lift a rule
that already works on connections up to whole organisms.

## The five

**H-013 — Infinite reward, limited punishment, and punishment is death.**
Positive reinforcement can be unbounded. Negative reinforcement accumulates
against a threshold, and crossing it ends that instance rather than nudging a
weight. IPNN currently uses a symmetric advantage (`raw − baseline`), which
treats a wrong answer as a small negative number. This says a wrong answer
should spend from a finite budget, and the budget running out should be fatal.

**H-014 — Inheritance is the accumulator.** *"The ultimate reward should be
that it passes on whatever it learned."* A successful instance seeds the next
with part of what it acquired. Two things make this sharper than it first
sounds. It attacks L-028's gap from the opposite side to H-012 — across
lifetimes rather than within one. And a population under selection puts the
**rules** under selection, which today's knob table explicitly notes they are
not: every parameter in 002 is a constant shared by every node and edge, so
structure can be selected but the physics that generates it cannot.

**H-015 — How much is pre-programmed is a dial, not a binary.** A spider spins
a web untaught. Humans look unprogrammed but arrive with balance, breathing and
blinking built in. 001 is fully given, 002 is fully grown, and the interesting
region is between them — a region the project has been sitting on either side
of without ever naming it as an axis.

This also disposes of a standing worry. If you have to define what the system
should learn, are you smuggling in the answer? No: **an organism's parameters
are built in too. Having a genome is not cheating.** Pre-programmed structure
and learned structure are not opposites.

**H-016 — Structure versus noise is the first test, and it is not a
classification task.** Show it 100 images, 80 of them random dots and 20 with
real structure, and ask whether it can separate them *without being told what
the structure is*. Everything the project has measured so far is "put this in
the right box out of three". This asks something prior and harder: is there
anything here at all. A system that cannot tell signal from noise has not
started; one that can has done something no label taught it.

**H-017 — Learning and measurement are separate phases; test on games it was
never trained on.** Train on one thing, then run a battery of *novel* games —
horizontal versus vertical, which image has more lines, find the longest line —
and score how fast it adapts to each. *"You've learned a bunch of stuff in
school but at the end I'm going to throw you into a whole set of challenges,
and that is the ultimate measure."*

## Two convergences worth recording

Both of these arrived in conversation with no knowledge of what was written
into `vision.md` earlier the same day, which is the only reason they are worth
noting at all.

1. **H-017 is Harlow's learning set** (rung 7 of the ladder) and **H-011's
   trials-to-criterion**, arrived at independently and phrased better: a series
   of escalating novel games, scored by how fast it adapts.
2. **The generalisation request** — two vertical bars should still register as
   "vertical" while being distinguishable from three — is a **generalisation
   gradient**, which went into the comparative battery this morning citing
   Guttman & Kalish's pigeons.

Independent convergence is weak evidence, but it is evidence: the measurement
doctrine adopted this morning is reachable from ordinary reasoning about
animals, which is the whole claim behind
[H-009](../HYPOTHESES.md)'s admissibility test.

## Analysis

The two days before this entry established what is broken in 002: information
does not survive a hop ([L-013](../LEARNINGS.md)), the core does not grow
([L-026](../LEARNINGS.md)), and nothing in the substrate has a long enough
timescale to accumulate ([L-019](../LEARNINGS.md)). The project's own answer,
H-012, was to add scaffolding *inside* one organism's life.

H-014 says the accumulator could sit outside the organism entirely. That is a
genuinely different bet, and it is worth stating which one this project is
better positioned to test: **H-012 is a small change to a substrate that
exists; H-014 requires a population, a selection loop, and an inheritance
mechanism, none of which exist.** H-012 first, on cost. But H-014 is the more
interesting hypothesis, and if H-012's core still refuses to grow, H-014 is the
next thing to try rather than a variation on the same theme.

H-013 is cheap and should be folded into whatever runs next: 002 already has
bounded-cost-then-death at the synapse, so lifting it to the organism is a
harness change, not a substrate change.

H-016 deserves attention out of proportion to its size, because it is the only
proposal here that changes **what the organism is asked to do**. Every gate in
this project so far has been three-way classification. "Is there structure in
this image" is a different question, needs no labels beyond a binary, and — per
[H-009](../HYPOTHESES.md) — is a question you could pose to an ant.

## Decisions

1. **H-013 … H-017 registered**, all marked **[J]**, all `open`, with the
   transcript archived at `journal/artifacts/2026-08-22-kian-conversation.md`
   so the source is inspectable rather than summarised away.
2. **Attribution convention extended in principle, not in notation.** Ideas
   arising in conversation with a third party are recorded as **[J]** with the
   conversation cited. No `[K]` mark is being introduced: the transcript is
   undiarised, turns cannot be recovered, and a reconstructed attribution is
   worse than an honest citation.
3. **Ordering unchanged for now.** Experiment 003 (transfer and retention) is
   still next and still runs on 001. H-013 is cheap enough to ride along.
4. **H-016 is promoted above the remaining 002 parameter work.** A
   structure-versus-noise probe is a day's work and asks a question the project
   has never asked.

## Threats to validity

1. **No attribution is recoverable from the source.** The transcript is one
   undiarised stream; every idea here is recorded under [J] by instruction, and
   that is a *convention*, not a measurement. Anyone reading later should treat
   the marks on H-013…H-017 as "arose in this conversation", not "originated by
   this person".
2. **Speech-to-text corruption.** The transcript contains "aunt" for ant,
   "span sweat" for spin a web, "three horse on a wires" for three horizontal
   wires. Nothing load-bearing appears to be affected, but the source is noisy.
3. **No AI summary was retrievable**, only the raw stream, so this entry's
   reading of it has not been cross-checked against Granola's own.
4. **These are hypotheses, not results.** Nothing here has been run. The
   convergences noted above are suggestive at best and prove nothing.

## Next

Unchanged: experiment 003. Then H-016's structure-versus-noise probe, which is
small and asks something new.
