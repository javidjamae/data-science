# The novelty audit, the missing "I", and a new direction

- **Entry:** `2026-08-21-0050-novelty-audit-and-the-missing-i`
- **When:** 2026-08-21 00:50–02:05 CDT
- **Who:** Javid + Claude (session)
- **Experiment:** theory / direction. No code changed; three docs and one new experiment design produced.
- **Code state:** no code changed; the docs described here are git `ef67886`
- **Re-run:** n/a — documentary entry. Artifacts: [experiments/002-grown-substrate/design.md](../../experiments/002-grown-substrate/design.md), [experiment-ideas.md §F, §G](../../experiment-ideas.md), [related-work.md](../../related-work.md) additions.

## In plain words

Javid asked a blunt question: have we done anything novel? The honest answer
was no — everything built so far is a correct reimplementation of results that
are between 8 and 40 years old, and the M1 task is easier than a pole-balancer
from 1983.

Following that thread turned up something worse and more useful. The project
is called an **Iterative** Probabilistic Neural Network. The abstract devotes
a whole section to "iterative consideration" — output feeding back so the
network can circulate a thought and settle on an answer. That feedback does
not exist in the code. There is no recurrence of any kind. The one word in the
project's own name that distinguishes it from decades of prior work is the one
mechanism nobody built.

The second half of the night was spent designing something that *could* be
new. Not by picking an ML technique, but by going back to what nervous tissue
actually does — wiring costs energy, signals take time, axons grow by
following chemical trails, reward arrives as a substance that spreads from a
place rather than a number announced everywhere — and asking what each of
those would force. That produced experiment 002, and one idea in it looks
genuinely different: if reward spreads out from a source and connections grow
toward it, then *where a connection is* becomes the thing that decides whether
it gets credit — and building the network and assigning credit stop being two
separate problems.

## Objective

Answer "is any of this novel?" honestly, and if not, find where novelty could
plausibly live and design toward it.

## Gate (pre-registered)

None — a direction-setting iteration, not an experiment. Recorded because the
journal's rule 12 requires decisions that change the theory to be logged, and
this one changes the ranking of everything.

## Method

- Audited every claim in the project against the literature *from memory*
  (explicitly not a live search — see Threats).
- Audited `abstract.md` and `vision.md` against `engine/organism.ts`,
  clause by clause, using grep on the actual source rather than recollection.
- Derived a candidate design from biological constraints rather than from ML
  methods, at Javid's explicit instruction: *"I don't want you applying KNOWN
  solutions."* Prior art was then checked **after** the design and recorded as
  a check, not a template.

## Results

**Audit 1 — novelty.** Every mechanism maps to established work: the learning
rule is REINFORCE with eligibility traces (Williams 1992; Barto/Sutton/Anderson
1983); L-002 is policy saturation; L-004 is Grossberg's stability–plasticity
dilemma, already cited in our own related-work; L-008 is Cumulative Layout
Shift; L-009 is the rivalry paradigm's standard instrument. Two *planned*
claims are closer to being pre-empted than related-work admitted, and both
were uncited: **C1's "evidence beats magnitude"** versus Synaptic Intelligence
(Zenke, Poole & Ganguli 2017), and **Beta-confidence consolidation** versus
cascade/complex synapse models (Fusi, Drew & Abbott 2005; Benna & Fusi 2016)
and metaplasticity (Abraham & Bear 1996).

**Audit 2 — spec versus implementation.** Verified against source:

| `abstract.md` says | In the code |
|---|---|
| §2 recurrent core, output→input feedback, activity circulates | **absent** — `poolPre` indexes only the sense; pool neurons never read each other |
| §3 iterative consideration, multiple internal passes per stimulus | **absent** — one pass, every tick |
| §3 Beta-distributed confidence | partial — `n` is a counter that only ever increases; a synapse can never become *less* certain |
| §2 64×64 visual sense | 8×8 (deliberate M1 toy scale) |
| §2 probabilistic sparse firing, urge, legal silence | present |
| §3 three-factor reward-modulated local learning, no gradients | present |

`design.md` §3 recorded the recurrence deferral deliberately, on stability
grounds (open-problems §2) — so this was a documented decision, not a silent
substitution. But the deferral has now survived every subsequent iteration.

**Audit 3 — the structural-plasticity idea is newer than the project.**
Neither `abstract.md` nor `vision.md` mentions growth, pruning, neurogenesis,
or changing architecture. That idea enters the record on 2026-08-20 in
`experiment-ideas.md` §C. The implementation cannot be faulted for omitting
it; the founding documents describe a fixed-topology network.

**Design produced.** [Experiment 002 — the grown substrate](../../experiments/002-grown-substrate/design.md):
interfaces pinned, interior starting at zero edges, growth guided by an
activity field, credit delivered by a diffusing reward field, edge latency set
by span so long edges are the fast path, rent paid per tick so death is
failure to pay, and all structural change gated to a sleep phase. Milestones
M0–M4 with control arms fixed in advance. Measured, not assumed: the organism
interface is **nine members**, so 002 is a substrate swap behind an interface
and experiment 001 becomes its control arm rather than being replaced.

**Backlog captured.** §F (the grown substrate, now graduated to 002) and §G
(reference frames — a world larger than the sense, movement, efference copy,
and prediction error as a second learning signal).

## Analysis

The audits compound. Individually: "nothing novel yet" is unremarkable for a
young project, and "recurrence deferred for stability" was a defensible call.
Together they say something sharper — **the project has been building
outward (UI, demos, journal discipline) while its distinguishing mechanism
stayed unbuilt**, and the honest reason novelty is absent is that the novel
part was never attempted.

The most useful output is not the 002 design but the reframing of what to
protect: three seams look open (all *unverified*) — latency as a plastic
per-edge parameter tuned by use; credit assignment performed by diffusion
geometry so that growth and credit are one mechanism; and structural change
gated to an offline phase in an always-on organism. The second is the one
worth pursuing, because it is a different *kind* of answer to open-problems §1
rather than a variation on the existing one: what limits it is diffusion
distance, not the number of stochastic units, which is precisely the term that
kills this rule family at scale.

One methodological correction was made mid-session and is worth recording.
The first draft of §F led with prior art and derived the design from it —
effectively rebuilding someone else's paper. Javid rejected that outright.
Rewriting it to derive from biological constraints first, with prior art
demoted to an after-the-fact check, produced materially different and better
ideas (the reward *field*, plastic latency, sleep-gated restructuring) that
the prior-art-first framing had not surfaced. Recorded as L-011.

## Prior art & novelty

The entry *is* a prior-art exercise. Added to
[related-work.md](../../related-work.md) this iteration: synaptic sampling
(Kappel/Habenschuss/Legenstein/Maass), NEAT, the exponential distance rule,
generative connectome models (Vértes, Betzel, Akarca), spatially embedded RNNs
(Achterberg/Akarca 2023), polychronization (Izhikevich 2006), synfire chains
(Abeles), plus binocular rivalry and drift-diffusion for the A-track. No
novelty is claimed anywhere in this entry.

## Learnings

- **L-011:** Deriving a design from prior art produces prior art. Working the
  constraints of the real system first — and consulting the literature only as
  an after-the-fact check — surfaced mechanisms (reward as a diffusing field,
  use-tuned latency, sleep-gated restructuring) that the literature-first pass
  did not. Prior art is a *check on claims*, not a *source of designs*.
  *Evidence:* the two §F drafts written back to back this session, from the
  same raw idea, with visibly different outputs.
- **L-012:** Passing gates does not mean implementing the specification. Every
  M1 gate passed while the mechanism the project is *named for* — iterative
  consideration via output→input feedback — was absent, because the gates
  tested what was built rather than what was specified. Spec-versus-source
  audits belong on the calendar, not on intuition. *Evidence:* audit 2 above,
  verified by grep against `organism.ts`.

## Decisions

1. **Recurrence moves onto the critical path.** Output→pool feedback
   (abstract §3, design.md §3) is now ranked above the F-track: it is the
   project's defining mechanism, its oldest untested hypothesis (2023), and
   §F's latency claims are unreadable on a feedforward organism.
2. **Experiment 002 is the grown substrate**, sharing one codebase with 001
   behind an `OrganismLike` interface. **Not a fork** — 001 is 002's control
   arm and shared code is what stops the comparison rotting.
3. **§G (reference frames) is deliberately not experiment 003 yet.** It needs
   a different *world*, plus motor, prediction and recurrence. Captured with a
   five-rung ladder whose first two rungs need no new mechanism.
4. **Prior art is checked after design, not before.** Adopted as a working
   rule from L-011.
5. Docs updated: `related-work.md`, `experiment-ideas.md` (§F, §G, re-ranked),
   `README.md` (naming conventions, experiment table), `open-problems.md` §1.
6. **Not** updated: `abstract.md`. It currently describes a fixed-topology
   network, and if structural plasticity is genuinely part of what IPNN means,
   that belongs in the abstract as a stated mechanism. Flagged for Javid —
   changing the founding document is his call, not a session's.

## Deviations

The session was asked for a demo bug fix and ended up re-ranking the roadmap.
Recorded plainly: this was driven by Javid's questions, not by a session
deciding on its own to redirect the project.

## Threats to validity

1. **The entire novelty audit is from memory, not a literature search.** Every
   pre-emption claim (Zenke, Fusi, Kappel, Izhikevich) is stated at moderate
   confidence about *existence* and low confidence about *precise scope*. A
   real search is required before any of this is relied on — especially before
   abandoning C1 on the strength of it.
2. **"Three seams look open" is an absence-of-evidence claim** made by one
   session from memory, which is the weakest possible basis for a novelty
   claim. It is recorded as a hypothesis about the literature, not a finding.
3. The spec-versus-code audit is solid — it was grep-verified — but covers
   `abstract.md` and `vision.md` only; other docs may contain further drift.
4. Experiment 002's design has never been run and may be unbuildable as
   specified; its cold-start risk in particular is unquantified.

## Next

Experiment 002 M0: extract the `OrganismLike` interface with all 17 tests
passing bit-identically. Pre-registered gate: *no behavioral change to
experiment 001 whatsoever* — M1 gate curves must remain identical to those
recorded in entry 2026-08-16-0248. Then a real literature search on the three
seams, before any further design work rests on them.
