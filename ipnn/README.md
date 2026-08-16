# IPNN — Iterative Probabilistic Neural Network

A research project exploring a **living** neural network: always-on, learning
in real time through interaction and reward, with no training/inference
split. Built from scratch — no ML training frameworks — in the spirit of
Jeff Hawkins' Thousand Brains program.

Start with [vision.md](./vision.md). It's the north star; everything else
serves it.

## Map

| Doc | Role | Changes |
|---|---|---|
| [vision.md](./vision.md) | Plain-language north star: living model, senses, reward-taught, multi-sense, low power | Rarely |
| [abstract.md](./abstract.md) | The paper-style writeup: system, mechanism, hypotheses | As theory evolves |
| [open-problems.md](./open-problems.md) | Honest list of unsolved problems + current working answers + how we'll test them | Every time an experiment lands |
| [related-work.md](./related-work.md) | Prior art map: what exists, what IPNN takes or claims beyond it | As we read |
| [journal/](./journal/README.md) | **The lab notebook of record**: one timestamped entry per iteration + the citable learnings ledger ([LEARNINGS.md](./journal/LEARNINGS.md)) | Every iteration |
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

## License

Code: [Apache-2.0](./LICENSE). Documentation (all `.md` in this directory
tree): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Scope: the
`ipnn/` directory only, not the rest of this repository. Rationale in
[disclosure-policy.md](./disclosure-policy.md).

## Experiments

| # | Name | Status | Question |
|---|---|---|---|
| [001](./experiments/001-mnist-living-demo/design.md) | MNIST living demo | M1 gate passed — reward-only learning works at toy scale | Can it learn digits in real time through reward alone — and keep performing when rewards stop? |
