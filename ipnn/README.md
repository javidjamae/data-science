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
| [experiment-ideas.md](./experiment-ideas.md) | The backlog: raw ideas → shaped hypotheses → numbered experiments; ranked | As ideas arrive |
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

## Naming (what M1, M1b, 002 actually mean)

- **`NNN`** — an *experiment*: a numbered directory with its own question,
  design doc and gates. `001` is the fixed-architecture living demo; `002` is
  the grown substrate.
- **`M#`** — a *milestone within one experiment*, defined in that experiment's
  `design.md` and gating the next. Always read as "experiment 001's M1",
  never as a global number: each experiment numbers its own milestones from
  M0. In 001: M0 scaffold, **M1 the critical learning gate**, M2 minimal
  living demo, M3 full experience, M4 science.
- **`M#x`** (letter suffix) — an *ablation or variant* of that milestone,
  answering "which part of M# was load-bearing?" rather than advancing the
  experiment. `M1b` = the `etaPool=0` ablation of 001's M1 gate.
- **`L-###`** — a durable learning in the
  [ledger](./journal/LEARNINGS.md), cited by ID everywhere and never deleted.
- **Backlog letters (`§A`–`§G`)** — thematic tracks in
  [experiment-ideas.md](./experiment-ideas.md), *not* experiments. A track
  graduates into a numbered experiment when it gets a `design.md`; §F became
  experiment 002.

## License

Code: [Apache-2.0](./LICENSE). Documentation (all `.md` in this directory
tree): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Scope: the
`ipnn/` directory only, not the rest of this repository. Rationale in
[disclosure-policy.md](./disclosure-policy.md).

## Experiments

| # | Name | Status | Question |
|---|---|---|---|
| [001](./experiments/001-mnist-living-demo/design.md) | MNIST living demo | M1 gate passed — reward-only learning works at toy scale · **M1b ablation shows the pool's plasticity earns nothing** (L-010) · **[live demo](https://claude.ai/code/artifact/1318cf33-a77e-4c86-b699-784f0c4f3c24)** | Can it learn digits in real time through reward alone — and keep performing when rewards stop? |
| [002](./experiments/002-grown-substrate/design.md) | Grown substrate | M0 met; **M1 failed** — it wires itself from zero edges but learns nothing, because information survives only one hop (L-013). M2–M4 blocked | Given only the interfaces, can an organism grow its own wiring — and does reward-as-a-diffusing-field make *position* the thing that decides credit? |
