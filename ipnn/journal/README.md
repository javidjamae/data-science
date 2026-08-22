# IPNN Research Journal

The lab notebook of record for the IPNN project. Every iteration — every
build-measure-learn cycle, successful or failed — gets a timestamped entry
here. If a result isn't in the journal, it didn't happen.

## Structure

```
journal/
  README.md      ← this file: the rules
  TEMPLATE.md    ← copy this to start a new entry
  LEARNINGS.md   ← the learnings ledger: what we have established (L-###)
  HYPOTHESES.md  ← hypotheses (refutable claims) AND ideas (considerations we carry), H-###
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
6. **Thoughts are first-class, not just iterations.** A realisation, a change
   of direction, a new hypothesis or a reframing of what we are even trying to
   measure is worth recording *the day it happens*, whether or not any code
   ran. File it as a `theory` entry, and register any new hypothesis in
   [HYPOTHESES.md](./HYPOTHESES.md) with an `H-###`. The reason is plain: the
   ideas that change a project's direction are almost never the ones that
   arrive attached to a result, and if they only ever live in a conversation
   they will be re-derived from scratch in three months, usually worse.

7. **Attribution: who thought of it.** See the section below. Every hypothesis,
   and every idea in an entry that materially changed the direction, carries a
   `[J]` / `[C]` / `[J→C]` / `[C→J]` mark.

8. **Learnings are first-class.** Every durable learning gets an `L-###` ID
   in [LEARNINGS.md](./LEARNINGS.md) with a one-line statement, status, and a
   link to the entry holding the evidence. Later entries cite learnings by ID
   instead of restating them. Learnings are never deleted — they are marked
   `superseded by L-###` or `refuted (entry link)`.
9. **Reproducibility block.** Every entry records: the exact configuration
   values used (a snapshot, not a reference to a file that may change), RNG
   seeds, the git commit hash of the code that produced the results (commit
   at iteration boundaries so the hash exists; note "uncommitted" honestly if
   so), and the command to re-run.
10. **Threats to validity.** Every entry with a positive result must state
   what could make its conclusions wrong. If you can't think of one, you
   haven't thought.
11. **Plain words first.** Every entry opens with a non-technical summary —
   what we tried, whether it worked, what we learned — readable by someone
   with no ML background. (This mirrors the plain-language digests required
   by journals like eLife and PLOS.) Written last, placed first.
12. **Prior art & novelty, every entry.** Each entry states how the
    iteration relates to others' prior work: what it echoes (cite
    `../related-work.md`, adding new finds there first), what it does
    differently, and the *smallest honest* novelty claim — marked
    *unverified against literature* until a real search has been done, and
    downgraded in a later entry if prior art turns up. "Nothing novel here,
    this replicates X" is a fully respectable section.
13. **Retrospective entries are allowed, and must say so.** History that
    predates the journal (or that was never written down) may be filed as a
    *retrospective* entry, dated to when the events happened so the index
    reads as a timeline. It must open with a banner giving the date it was
    actually written and separating what is verifiable from the record
    (commits, files, output) from what is reconstructed or inferred. This
    does not weaken rule 1: retrospective entries *add* missing history,
    they never revise an existing entry.
14. **Claims live in one place.** Theory belongs in `../abstract.md`, standing
   doubts in `../open-problems.md`, experiment designs in
   `../experiments/*/design.md`. The journal records *what happened and what
   was learned*, and links outward. When a learning changes the theory, the
   entry's *Decisions* section says which doc was updated.


## Attribution: who thought of it

This project is run by a person and a model working together, over long
sessions, at speed. Six months from now it will be impossible to reconstruct
which of them an idea came from — and that is worth knowing, both for honesty
about the record and because the two have noticeably different failure modes.
So it gets marked at the time, never afterwards.

| Mark | Meaning |
|---|---|
| **[J]** | Javid's, in substance. He stated it; it was not prompted or suggested by the model. |
| **[C]** | Claude's, in substance. The model originated it. |
| **[J→C]** | Javid seeded it; Claude developed, formalised or operationalised it. |
| **[C→J]** | Claude proposed it; Javid selected it, redirected it, or decided on it. |

Rules that keep this from becoming decorative:

- **Recorded live, never reconstructed.** An attribution added weeks later is
  a guess wearing a costume. If it was not marked at the time, leave it
  unmarked rather than inventing one.
- **When genuinely unclear, it is `[J→C]`.** Most good ideas here arrive in
  conversation and neither party can honestly claim them alone. The ambiguous
  case is the common case, and it has its own mark for that reason.
- **It attaches to ideas and decisions, not to typing.** Claude writes almost
  all of the prose and nearly all of the code; that is not authorship of the
  idea. Conversely a one-line instruction that redirects the whole experiment
  is `[J]`, however short it was.
- **Marks go on hypotheses, on direction-changing ideas in entries, and on
  backlog items.** Not on routine work — there is no value in attributing a
  bug fix.

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
- **Banner-marked retrospective entries** — ELN practice for a late or
  reconstructed entry: never backfill silently; record it, date it, and
  state plainly that it was written after the fact.
- **Negative results logged with full rigor** — the file-drawer problem:
  unpublished failures are how fields fool themselves.

## Entry lifecycle

1. Copy `TEMPLATE.md` → `entries/YYYY-MM-DD-HHMM-slug.md` (HHMM = iteration
   start, local time).
2. Fill *Objective / Gate / Hypotheses* before or at the start of the work.
3. Fill *Method / Results* as you go; *Analysis / Learnings / Decisions /
   Threats / Next* at the end.
4. Add new learnings to `LEARNINGS.md`; update statuses of any superseded
   ones. Register new hypotheses in `HYPOTHESES.md` with their attribution,
   and update the status of any the entry bears on.
5. Update the index below.

## Index (newest first)

| Entry | Iteration | Outcome |
|---|---|---|
| [2026-08-22-1735 — The α/β fix](./entries/2026-08-22-1735-alphabeta.md) | Let confidence fall: evidence for and against tracked separately, behind a flag, gates committed before running | **Aging cured** (480/480 synapses plastic after 8 reversals, survival 7→18/32) — but the ≥20 parity bar **missed**, verdict FAIL as registered. Retention *improved* (0.982 v 0.952): **memory and flexibility are two dials, not one**. Residual failures are the rule's, not consolidation's; L-036, L-037 |
| [2026-08-22-1620 — The organism ages out of learning](./entries/2026-08-22-1620-serial-reversal.md) | Serial reversal: flip the rule eight times, does it get better at changing its mind? | **No reuse** (0/4 seeds improved) — and worse, **terminal decline**: by flip 4–5 most runs never learn again. Cause found: **consolidation**. 7/32 reversals succeed with it on, 22/32 with it off. **L-004 vindicated** and this entry's own earlier annotation corrected — one reversal was too short to see a cumulative mechanism; L-033, L-034, L-035 |
| [2026-08-22-1530 — All three gates passed, and the experiment was wrong](./entries/2026-08-22-1530-transfer-retention.md) | Experiment 003: transfer, retention and savings on 001 | All three pre-registered gates **passed** — and the protocol tested capacity, not retention: same outputs with *different* stimuli is a union, not interference. The corrected test (reversal, same stimuli, permuted labels) shows retention collapsing to **0.069, below the 0.333 chance line**, and relearning **7.7× slower** than learning from scratch. Consolidation and code separation both refuted as explanations; L-029…L-032 |
| [2026-08-22-1340 — Five hypotheses from a kitchen-table conversation](./entries/2026-08-22-1340-kian-conversation.md) | Theory: ideas from a conversation Javid had about the research | H-013…H-017 registered. **The ultimate reward is passing on what you learned** — an across-lifetime accumulator attacking L-028's gap from the opposite side to H-012. Also: unbounded reward with bounded punishment where the budget running out is death; pre-programming as a dial not a binary; structure-vs-noise as a first test that isn't classification |
| [2026-08-22-1300 — Nothing crystallises](./entries/2026-08-22-1300-crystallization.md) | Testing Javid's scaffolding hypothesis: does anything build slowly without decaying? | A differentiated core **exists** (~100× evidence, 2× weight of young edges) but is **stationary** — 8 edges of 5,000, unchanged over 16×. No-rent makes it permanent, undifferentiated and *worse* (0.444). The missing clause: growth cannot see consolidated structure, so nothing builds on the core; L-026, L-027, L-028, H-012 |
| [2026-08-22-1240 — Latent learning](./entries/2026-08-22-1240-latent-learning.md) | Tolman's unrewarded-pre-exposure test on both substrates | **H-010 refuted.** 002 does gain from unrewarded time (659 vs 746 trials to criterion) but a scrambled-pairing control gains more (535) — warm-up, not task knowledge. 001 cannot gain at all: R=0 means nothing moves; L-024, L-025 |
| [2026-08-22-1215 — The 32,000-trial run, and a positive control that wasn't one](./entries/2026-08-22-1215-longrun-and-a-correction.md) | H-002 trajectory test at 128× the original window, both arms | M1 **flat on every measure** — accuracy, σ, decoding, persistence (5% throughout). H-002 refuted for this arm. **Correction to the previous entry:** the shallow arm reaches 0.905 accuracy with hop-2 decoding at 0.394, so it never carries signal at depth and was never a valid positive control for a depth instrument; L-021, L-022, L-023 |
| [2026-08-22-1030 — What counts as learning](./entries/2026-08-22-1030-what-counts-as-learning.md) | Theory/direction: answering "how do you know it will never learn?" and replacing accuracy as the primary gate | Adopted the **ladder of evidence** (7 rungs, each with a null) in vision.md; added the hypotheses ledger (H-001…H-008) and the `[J]`/`[C]` attribution convention. Key argument: **slow learning requires a slow variable**, and 002 has none — persistence flat at 5–6%. Experiment 003 (transfer/retention) promoted, and it runs on 001; L-018, L-019, L-020 |
| [2026-08-22-0208 — exp002 M0 built; M1 fails on depth](./entries/2026-08-22-0208-exp002-m0-built-m1-fails-on-depth.md) | Experiment 002: build the grown substrate, run the M1 critical gate | M0 **met** (001 bit-identical, 88 tests). M1 **FAILED** — flat at chance, 3 seeds × 2,000 trials. Cause localised: stimulus information survives exactly one hop (7.7σ → 1.7σ = the 3/√π noise floor), and no sense pixel is within 5 hops of an answer. An arm with 19 of 64 pixels inside 2 hops scores 0.883, same rule. Also: the homeostat regenerates activity from silence, invalidating a pre-registered control; L-013 … L-017 |
| [2026-08-21-0145 — M1b: the etaPool=0 ablation](./entries/2026-08-21-0145-m1b-etapool-ablation.md) | Ablation: is 001's hidden pool contributing? | **No** — freezing 89% of learnable synapses costs 0.004 accuracy. M1 is a readout on a fixed random projection; L-010 |
| [2026-08-21-0050 — Novelty audit and the missing "I"](./entries/2026-08-21-0050-novelty-audit-and-the-missing-i.md) | Theory/direction: honest novelty audit + spec-vs-code audit (ran 00:50–02:05, overlapping M1b) | Nothing novel yet; the recurrence the project is *named* for is unbuilt. Experiment 002 designed; L-011, L-012 |
| [2026-08-21-0025 — Manual mode + sustained readout](./entries/2026-08-21-0025-manual-mode-sustained-readout.md) | Infrastructure: hold a stimulus and watch the answer run free | 17/17 tests; manual mode provably cannot alter the organism; L-009; blank-sense restlessness logged as anecdote |
| [2026-08-21-0008 — Demo layout jitter fix](./entries/2026-08-21-0008-demo-layout-jitter-fix.md) | Infrastructure: UI defect — panels sized by their own captions | Layout movement 6.2px → 0.0px measured; L-008; predicted by the prior entry's threat #1 |
| [2026-08-20-0856 — M1 living demo UI](./entries/2026-08-20-0856-m1-living-demo-ui.md) | Infrastructure: interactive browser demo of the M1 result | 7/7 tests; stepper refactor bit-identical to recorded curves; demo published |
| [2026-08-16-0248 — Experiment 001, M0+M1 first build](./entries/2026-08-16-0248-experiment-001-m0-m1-first-build.md) | Engine scaffold + M1 sanity gate | Gate PASSED after one collapse-and-fix cycle; L-001…L-007 |
| [2023-07-12-2228 — Original IPNN abstract published](./entries/2023-07-12-2228-original-abstract-published.md) | Theory only; idea published to the public repo (`6ff96d3`) | Documentary (retrospective, written 2026-08-20); starts the priority record; no learnings |
