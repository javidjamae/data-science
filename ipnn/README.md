# IPNN — Iterative Probabilistic Neural Network

A research project exploring a **living** neural network: always-on, learning
in real time through interaction and reward, with no training/inference
split. Built from scratch — no ML training frameworks — in the spirit of
Jeff Hawkins' Thousand Brains program.

Start with [vision.md](./vision.md). It's the north star; everything else
serves it. If you hit a label you don't recognise — `002`, `001-M1b`, `L-013`,
`H-004`, `§F`, `[J→C]` — every one of them is defined in
[Naming](#naming--every-label-used-in-this-project) below.

## Map

| Doc | Role | Changes |
|---|---|---|
| [vision.md](./vision.md) | Plain-language north star: living model, senses, reward-taught, multi-sense, low power | Rarely |
| [abstract.md](./abstract.md) | The paper-style writeup: system, mechanism, hypotheses | As theory evolves |
| [open-problems.md](./open-problems.md) | Honest list of unsolved problems + current working answers + how we'll test them | Every time an experiment lands |
| [related-work.md](./related-work.md) | Prior art map: what exists, what IPNN takes or claims beyond it | As we read |
| [experiment-ideas.md](./experiment-ideas.md) | The backlog: raw ideas → shaped hypotheses → numbered experiments; ranked. Lettered tracks `§A`–`§H` | As ideas arrive |
| [journal/](./journal/README.md) | **The lab notebook of record**: one timestamped entry per iteration, the learnings ledger ([LEARNINGS.md](./journal/LEARNINGS.md), what we know) and the hypotheses ledger ([HYPOTHESES.md](./journal/HYPOTHESES.md), what we believe and intend to test) | Every iteration, and whenever a realisation lands |
| [experiments/](./experiments/) | Numbered experiments (see below) | Continuously |
| [disclosure-policy.md](./disclosure-policy.md) | Open-by-default policy with pre-registered safety tripwires; licensing rationale | Rarely, with journaled rationale |

## How we document (the rules)

- **Experiments are numbered directories** — `experiments/NNN-short-name/`
  containing:
  - `design.md` — written *before* building: question, spec, milestones,
    success gates, risks.
  - `results.md` — written *after*: what we learned, with data. Findings that
    change the theory get folded back into `abstract.md` /
    `open-problems.md`.
- **The iteration-by-iteration record lives in the
  [journal](./journal/README.md)** — one timestamped entry per
  build-measure-learn cycle (multiple per day is normal), following the
  rules and template there. Durable learnings get `L-###` IDs in the
  [learnings ledger](./journal/LEARNINGS.md) and are cited by ID everywhere
  else.
- **Claims live in one place.** Theory in `abstract.md`, doubts in
  `open-problems.md`, evidence in journal entries and
  `experiments/*/results.md`. Don't restate theory inside experiment docs —
  link to it.
- **Decisions get recorded where they bind:** experiment-scoped decisions in
  that experiment's `design.md`/`log.md`; project-wide ones in the relevant
  top-level doc. If we accumulate enough cross-cutting decisions, promote
  them to a `decisions/` directory.
- Future homes, when needed (don't create empty dirs before then): `theory/`
  if `abstract.md` outgrows itself, `glossary.md` when vocabulary stabilizes.

## Naming — every label used in this project

Four families of label, answering four different questions. If a label is not
in this table, it is not a label.

### Where work lives

| Label | Means | Examples |
|---|---|---|
| `NNN` | An **experiment**: its own directory, question, `design.md` and gates | `001` fixed-architecture living demo · `002` grown substrate |
| `§A`–`§G` | A **backlog track** in [experiment-ideas.md](./experiment-ideas.md) — a *theme*, not an experiment. Graduates to a number when it earns a `design.md` | `§F` became experiment 002 |

> **`§E` is not an ID.** The backlog letters are section headings in one file.
> They have nothing to do with `L-###` or `H-###`, and there is no `E-###`.

### Steps inside one experiment

| Label | Means |
|---|---|
| `M#` | A **milestone**, defined in that experiment's `design.md`, gating the next |
| `NNN-M#x` | An **ablation** of that milestone — "which part was load-bearing?" — not progress toward the next one |

> **⚠️ `M#` is per-experiment, never global.** 001's M1 and 002's M1 are
> different gates with different outcomes: **001's M1 passed** (0.98), **002's
> M1 failed** (0.157). Always write and read it as "001's M1".
>
> Ablations therefore carry their experiment: **`001-M1b`** (the `etaPool=0`
> ablation), **`001-M1c`** (no-pool control), **`002-M1d`** (depth ladder),
> **`002-M1e`** (starved readout). Bare `M1c` is ambiguous and should not be
> used — an earlier draft did, and the two experiments' suffixes read as one
> alphabetical sequence when they are not.

### What we think

| Label | Means | Ledger |
|---|---|---|
| `L-###` | A **learning** — established, with evidence behind it | [LEARNINGS.md](./journal/LEARNINGS.md) |
| `H-###` | A **hypothesis** (a claim that could be shown wrong) or an **idea** (a design consideration we're carrying). The ledger's `Kind` column says which | [HYPOTHESES.md](./journal/HYPOTHESES.md) |

`L` is what we know; `H` is what we don't yet. Within `H`, a **hypothesis**
names a result that would refute it and an **idea** does not — *an idea becomes
a hypothesis the moment someone states what would refute it.* A hypothesis that
survives testing becomes a learning; one that dies is marked refuted and stays.
Neither is ever deleted. Whole programmes, rather than single considerations,
go in [experiment-ideas.md](./experiment-ideas.md) as lettered tracks (`§A`–`§H`).

### Who thought of it

| Mark | Means |
|---|---|
| `[J]` | Javid's, in substance |
| `[C]` | Claude's, in substance |
| `[J→C]` | Javid seeded it, Claude developed it |
| `[C→J]` | Claude proposed it, Javid selected or redirected it |

Recorded live, never reconstructed. Full convention in
[journal/README.md](./journal/README.md#attribution-who-thought-of-it).

## License

Code: [Apache-2.0](./LICENSE). Documentation (all `.md` in this directory
tree): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Scope: the
`ipnn/` directory only, not the rest of this repository. Rationale in
[disclosure-policy.md](./disclosure-policy.md).

## Experiments

| # | Name | Status | Question |
|---|---|---|---|
| [001](./experiments/001-mnist-living-demo/design.md) | MNIST living demo | M1 gate passed — reward-only learning works at toy scale · **M1b ablation shows the pool's plasticity earns nothing** (L-010) · **[live demo](https://claude.ai/code/artifact/1318cf33-a77e-4c86-b699-784f0c4f3c24)** | Can it learn digits in real time through reward alone — and keep performing when rewards stop? |
| [003](./experiments/003-transfer-retention/design.md) | Transfer & retention | **Run.** All three gates passed, then the design was found flawed — same outputs with different stimuli is a union, not interference. Corrected test (reversal) shows forgetting **below chance** and relearning 7.7× slower (L-029, L-030) | Does 001 learn a second task without losing the first — the project's own definition of intelligence, tested for the first time |
| [004](./experiments/004-iterative-organism/design.md) | The iterative organism | Design only — pre-registered, nothing built. Closes the L-012 gap: the mechanism the project is *named for* has never been implemented | Does letting output re-enter input, with time to iterate before committing, improve evidence accumulation on a degraded stimulus? |
| [002](./experiments/002-grown-substrate/design.md) | Grown substrate | M0 met; **M1 failed** — it wires itself from zero edges but learns nothing, because information survives only one hop (L-013). M2–M4 blocked · **[live demo](https://claude.ai/code/artifact/122d168d-179c-4004-96b2-102377504da6)** | Given only the interfaces, can an organism grow its own wiring — and does reward-as-a-diffusing-field make *position* the thing that decides credit? |
