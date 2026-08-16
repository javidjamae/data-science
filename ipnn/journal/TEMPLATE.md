# <Iteration title>

- **Entry:** `YYYY-MM-DD-HHMM-slug`
- **When:** YYYY-MM-DD HH:MM–HH:MM <TZ>
- **Who:** Javid + Claude (session)
- **Experiment:** [NNN — name](../../experiments/NNN-name/design.md) (or "theory" / "infrastructure")
- **Code state:** git `<commit hash>` (or "uncommitted — hash of next commit: <fill in after committing>")
- **Re-run:** `<exact command>`

## In plain words

Non-technical summary, readable by anyone: what we tried, whether it worked,
and what we learned. Analogies welcome. Written last, placed first.

## Objective

What question this iteration attacks. One or two sentences.

## Gate (pre-registered)

The success criterion, quoted verbatim from where it was pre-registered
(design.md §, or the previous entry's *Next*). If no gate was pre-registered,
say so — that is itself a process finding.

## Hypotheses

Numbered, falsifiable, written before running.

- H1: …
- H2: …

## Method

What was built or changed, and how it was measured. Include:

- **Changes:** …
- **Configuration snapshot:** exact values (not a file reference), seeds
  included.
- **Measurement:** metrics, windows, number of trials/seeds.

## Results

Raw numbers and observations only — quote program output verbatim where
practical. No interpretation here.

## Analysis

Interpretation of the results. Which hypotheses survived? What surprised us?
Diagnosis of failures, with the evidence chain.

## Prior art & novelty

How this iteration relates to what others have done. Cite
[related-work.md](../../related-work.md) — and when the iteration surfaced
new prior art, add it there first, then cite it here.

- **Similar:** which prior systems/results this work echoes, stated plainly.
- **Different:** what this iteration did that those did not — mechanism,
  setting, or measurement.
- **Novel (claimed):** the smallest honest novelty claim, if any. Mark each
  claim *unverified against literature* until a real search has been done;
  downgrade or retract in a later entry if prior art turns up.
- If nothing here is novel, say so. Replication has value; pretending
  otherwise rots the record.

## Learnings

Each durable learning, added to [LEARNINGS.md](../LEARNINGS.md) with an ID:

- **L-###:** statement. *Evidence:* …

## Decisions

What we decided going forward, with rationale. Note any docs updated as a
result (abstract.md, open-problems.md, design.md).

## Deviations

Departures from the pre-registered design/plan, and why.

## Threats to validity

What could make this entry's conclusions wrong.

## Next

The next iteration's objective and its pre-registered gate.
