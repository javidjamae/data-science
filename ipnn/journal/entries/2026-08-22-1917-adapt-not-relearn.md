# Adapt, don't relearn: the goal restated, and the 2×2 that organizes what comes next

- **Entry:** `2026-08-22-1917-adapt-not-relearn`
- **When:** 2026-08-22 19:17 CDT
- **Who:** Javid (the framing) + Claude (organization and operationalization)
- **Kind:** theory / direction (journal rule 6 — no code ran for this entry)
- **Code state:** git `d49345b` (master, post PR #2 merge)

## In plain words

After watching the rule-flip demo and its stopped-clock lesson, Javid restated
what the project is actually after — and the restatement reorganizes the next
several experiments. This entry documents that thinking with attribution,
because it arrived in conversation and would otherwise evaporate.

## The thoughts, organized

**1. The goal is not relearning. It is having-already-learned. [J]**

> "The goal is for it to evolve into a system that isn't 'relearning' but has
> already learned and can adapt quickly to the new rules."

Everything measured today — 600–2,500 trials to reverse, both arms cratering
on novel rules — is *relearning*: dragging the same weights to a new
configuration. The target behaviour is different in kind: the knowledge stays
put, and only a thin binding changes. The organism should not fall to
below-chance on a flip at all; it should re-bind.

**2. A rule flip is a communication-protocol change, not a knowledge change. [J]**

> "If it's smart it'll learn to identify something and then independently
> learn how to communicate what it found with varying rules on how it should
> communicate. I think that's what we're testing here."

This renames what experiment 003 has been testing. The stimuli never changed;
what changed is the *protocol* for reporting them. Perception and
communication are separable competencies, and 001 fails the flip precisely
because it has no such separation — its only learnable layer IS the binding
(L-010, L-033). Registered as [H-018](../HYPOTHESES.md).

**3. Rules can be *shown* or *inferred*, and those are different challenges. [J]**

> "We could have a 'rules cortex' where we show it the order of the outputs we
> expect. But that's a different type of challenge than just having to
> 'figure out' what the rules are."

- **Inferred** — the rule changes unannounced and must be discovered from
  reward alone. This is what every 003 protocol has done, and it is the
  comparative literature's unannounced-shift paradigm (Wisconsin
  Card Sorting-like). Measured: 001 never gets better at it (L-033).
- **Shown** — the current rule arrives through a sense, like everything else
  the organism knows about the world. Cued task-switching in the comparative
  literature; run on monkeys, pigeons and humans, so it passes the
  [H-009](../HYPOTHESES.md) admissibility test.

Both belong in the battery. The shown variant is the *unmeasured* one, and it
is also the architectural probe: an organism that conditions its output on a
rule-cue is exhibiting exactly the factoring the goal statement demands.

**4. The dual axis: vary the inputs, hold the protocol. [J]**

> "In the future the inputs vary (rotate them or zoom out slightly) while what
> it has to communicate stays the same. And then we can do both."

Registered as [H-019](../HYPOTHESES.md). Together with H-018 this gives the
project a 2×2 that organizes the next experiments:

| | **protocol fixed** | **protocol varies** |
|---|---|---|
| **percepts fixed** | 001's M1 (done: passes) | 003's reversal (done: fails, relearns) |
| **percepts vary** | invariance — rotate/zoom, future | the far cell: the real target |

**5. There are many possible next steps, and one was chosen. [J→C]**

Candidates on the table tonight: experiment 004 (iteration), H-012 (scaffolded
growth in 002), the savings re-fix, uncued switching at longer budgets, input
invariance, and the shown-rules test. Decision below.

## Decision: experiment 005 — the rule as a sense

The shown-rules test, pre-registered in
[experiments/005-rules-as-a-sense/design.md](../../experiments/005-rules-as-a-sense/design.md).
Why it wins the queue:

1. **It operationalizes the goal statement directly.** "Already learned,
   adapts quickly" has a concrete signature: after training under both
   protocols *with the rule visible*, a flip should cost approximately
   nothing — the organism reads the cue and re-binds, no relearning.
2. **It is the cheapest architectural experiment available.** No substrate
   change at all: widen the sense (64 stimulus pixels + a cue strip), and the
   existing organism, teacher and harness run as-is. A day, not a build.
3. **Both outcomes are sharp**, given L-010. The pool is a fixed random
   projection; whether its accidental cue×stimulus conjunctions are enough
   for a 480-weight readout to express two protocols is a real open question.
   Pass → instant switching exists on the substrate we already have, and the
   demo grows a "show it the rules" toggle where the flip stops hurting.
   Fail → one learnable layer cannot even *select* between two mappings when
   told which one is in force — the strongest indictment of shallow
   substrates yet, and a concrete target for 002/004.
4. **It leaves the inferred variant as the standing hard case** (L-033's flat
   line), which is the right order: first show the factoring is expressible
   at all, then ask whether it can be discovered without being told.

004 stays queued behind it, sharpened by L-039: iteration is worth building
precisely if it buys adaptation *speed*, which is now the measured bottleneck.

## Threats to validity

This entry is direction, not measurement. The one factual claim it leans on —
that cued switching is unmeasured here and admissible — is checked against
the ledger and the battery; the rest is design rationale that 005's gates can
refute.
