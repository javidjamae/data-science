# IPNN Research Journal

The lab notebook of record for the IPNN project. Every iteration — every
build-measure-learn cycle, successful or failed — gets a timestamped entry
here. If a result isn't in the journal, it didn't happen.

## Structure

```
journal/
  README.md      ← this file: the rules
  TEMPLATE.md    ← copy this to start a new entry
  LEARNINGS.md   ← the learnings ledger: every learning, numbered and citable
  entries/       ← one file per iteration: YYYY-MM-DD-HHMM-slug.md
  artifacts/     ← (when needed) raw data dumps per entry: artifacts/<entry-slug>/
```

## The rules

1. **Append-only.** Entries are never rewritten after the fact (typo fixes
   excepted). If an entry turns out to be wrong, a *later* entry corrects it
   and the learnings ledger is updated. The record of being wrong is part of
   the record.
2. **One entry per iteration.** An iteration is a coherent
   build-measure-learn cycle. Write the entry the same day, while the details
   are fresh. Timestamps are local time with timezone stated.
3. **Pre-registered gates.** Success criteria are written down *before*
   running (in the experiment's `design.md` or the previous entry's *Next*
   section). An entry states the gate it was run against, verbatim. Moving a
   goalpost is allowed but must be recorded as a decision with rationale.
4. **Results and Analysis are separate sections.** Results contain raw
   numbers and observations, quoted verbatim where possible — no
   interpretation. Analysis is where interpretation lives. A reader must be
   able to disagree with the analysis while trusting the results.
5. **Negative results are mandatory content.** A failed attempt is logged
   with the same rigor as a success — usually more, since diagnosis is where
   learnings come from.
6. **Learnings are first-class.** Every durable learning gets an `L-###` ID
   in [LEARNINGS.md](./LEARNINGS.md) with a one-line statement, status, and a
   link to the entry holding the evidence. Later entries cite learnings by ID
   instead of restating them. Learnings are never deleted — they are marked
   `superseded by L-###` or `refuted (entry link)`.
7. **Reproducibility block.** Every entry records: the exact configuration
   values used (a snapshot, not a reference to a file that may change), RNG
   seeds, the git commit hash of the code that produced the results (commit
   at iteration boundaries so the hash exists; note "uncommitted" honestly if
   so), and the command to re-run.
8. **Threats to validity.** Every entry with a positive result must state
   what could make its conclusions wrong. If you can't think of one, you
   haven't thought.
9. **Plain words first.** Every entry opens with a non-technical summary —
   what we tried, whether it worked, what we learned — readable by someone
   with no ML background. (This mirrors the plain-language digests required
   by journals like eLife and PLOS.) Written last, placed first.
10. **Prior art & novelty, every entry.** Each entry states how the
    iteration relates to others' prior work: what it echoes (cite
    `../related-work.md`, adding new finds there first), what it does
    differently, and the *smallest honest* novelty claim — marked
    *unverified against literature* until a real search has been done, and
    downgraded in a later entry if prior art turns up. "Nothing novel here,
    this replicates X" is a fully respectable section.
11. **Claims live in one place.** Theory belongs in `../abstract.md`, standing
   doubts in `../open-problems.md`, experiment designs in
   `../experiments/*/design.md`. The journal records *what happened and what
   was learned*, and links outward. When a learning changes the theory, the
   entry's *Decisions* section says which doc was updated.

## Where these rules come from

None of this is invented; each rule is a standard practice borrowed from a
field that learned it the hard way:

- **Append-only, contemporaneous, timestamped entries** — wet-lab electronic
  lab notebook (ELN) standards, where notebooks are patent-grade legal
  records: never erase, correct only in later entries, write the same day.
- **Method / Results / Analysis separation** — the IMRaD structure of
  scientific papers; readers must be able to trust your data while doubting
  your interpretation.
- **Pre-registered gates and hypotheses** — preregistration, adopted across
  psychology and medicine after the replication crisis to stop
  moving-the-goalposts and hindsight bias.
- **Reproducibility block (configs, seeds, commit hash, re-run command)** —
  the NeurIPS reproducibility checklist and standard ML experiment-tracking
  practice (what MLflow/W&B log per run, done here in files).
- **Threats to validity** — required section in empirical software
  engineering and social science; forces the author to attack their own
  result.
- **Learnings ledger with supersede-not-delete statuses** — Architecture
  Decision Records (ADRs) from software engineering.
- **Plain-language summary first** — plain-language digests required by
  journals such as eLife and PLOS.
- **Prior art & novelty section** — the Related Work section of a paper,
  fused with patent practice's prior-art search and claims discipline: state
  the smallest claim that survives what's already known.
- **Negative results logged with full rigor** — the file-drawer problem:
  unpublished failures are how fields fool themselves.

## Entry lifecycle

1. Copy `TEMPLATE.md` → `entries/YYYY-MM-DD-HHMM-slug.md` (HHMM = iteration
   start, local time).
2. Fill *Objective / Gate / Hypotheses* before or at the start of the work.
3. Fill *Method / Results* as you go; *Analysis / Learnings / Decisions /
   Threats / Next* at the end.
4. Add new learnings to `LEARNINGS.md`; update statuses of any superseded
   ones.
5. Update the index below.

## Index (newest first)

| Entry | Iteration | Outcome |
|---|---|---|
| [2026-08-16-0248 — Experiment 001, M0+M1 first build](./entries/2026-08-16-0248-experiment-001-m0-m1-first-build.md) | Engine scaffold + M1 sanity gate | Gate PASSED after one collapse-and-fix cycle; L-001…L-007 |
