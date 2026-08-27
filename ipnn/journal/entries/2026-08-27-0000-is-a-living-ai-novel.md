# "It just lives in its world" — is that novel, and what would it actually buy us

- **Entry:** `2026-08-27-0000-is-a-living-ai-novel`
- **When:** 2026-08-27 00:00 CDT (continuing the session that produced [2026-08-26-2341](./2026-08-26-2341-adaptability-the-measure.md))
- **Who:** Javid (the claim and the sharpening) + Claude (the prior-art map and the falsifiable form)
- **Kind:** theory / novelty audit — **no code ran** (journal rule 6)
- **Code state:** git `aaba633`, working tree dirty with the previous entry's doc edits

## In plain words

Javid set the measured record aside deliberately and asked the theoretical
question underneath it: **if this works, is a "living" AI — one with no
stop-train-retry cycle, that just lives in a world being given challenges and
rewiring itself as it goes — not novel?**

The honest answer is in two halves. **The aspiration is not novel and is not
close to novel** — it is a named research program with a fifty-year lineage,
and at least one flagship statement of it (Sutton's Alberta Plan) is three
years old. **The theoretical gain, if the mechanism works, is real and is not
mainly about adaptation** — it is that removing the freeze removes an entire
category of problem, and that adaptation which comes from structure has no
task-family boundary, where adaptation bought with coverage always does.

Javid then sharpened the claim in the way that matters most: this is **not**
evolutionary search over wirings. It is one organism changing its own structure
within its own life. That sharpening narrows the prior art considerably — and
that neighbourhood is still populated, just by a different set of people.

## The questions, as asked [J]

> "Forget what we've proven for a second. All I care about at this moment is
> the theoretical gain if our model works. The idea is that if it works, it's a
> 'living' AI. There is no 'stop. train. retry. stop. train. retry.' It just
> lives in its world and we give it challenges. Its life involves learning and
> playing games and adapting its own neural network — but it is always 'on and
> learning' like a human. Is that not novel?"

> "It's not that we're picking a hardcoded wiring and evolving or culling
> offspring, like some evolutionary algorithms might do. It's actually ADAPTING
> its neural structure to learn and adapt to its learnings and environment. It
> learns reference frames over time (hopefully)."

## 1. The aspiration: not novel, and older than deep learning [C]

Mapped in full in [related-work.md § The "living AI"
program](../../related-work.md) — added this entry, banner-marked as written
from model knowledge rather than a search. The short form:

- **Developmental robotics** (Weng et al., *Science* **2001**) — competence
  that is neither programmed nor trained but *develops through open-ended
  lifelong interaction*. Twenty-five years old, and the closest named program
  to the sentence Javid wrote.
- **Ring 1994**, *Continual Learning in Reinforcement Learning Domains* —
  essentially this project's goal statement, 32 years earlier.
- **NELL** — deliberately left running for years.
- **Sutton's Alberta Plan (2022)** — explicitly rejects train-then-deploy in
  favour of a continually-learning agent in an unending stream of experience.
  This is the RL textbook's author saying the same thing, recently, loudly.
- **Hawkins / HTM** — "no train/inference split" is *their published claim*,
  already cited as closest-in-spirit at the top of `related-work.md`.
- **Grossberg / ART (1976)** — continuous learning with no train/test split.
  The stability–plasticity dilemma has a name because it was identified fifty
  years ago and never closed.

**So the project may not claim** that always-on learning, the absence of a
train/deploy split, an agent living in a world, or self-directed play are new
ideas. This is the same verdict as the [2026-08-21 novelty
audit](./2026-08-21-0050-novelty-audit-and-the-missing-i.md), reached from a
different direction and now with names attached.

## 2. Javid's sharpening moves the claim [J]

*Not* evolutionary topology search — NEAT, NAS, culling offspring — but one
organism restructuring itself from its own experience. This matters, and it
narrows the field to a different and smaller set: Growing Neural Gas (Fritzke
1995), Cascade-Correlation (Fahlman 1990), dynamic sparse training (SET, RigL),
DEEP R, synaptic sampling, Progressive Networks. All within-lifetime. All
listed now in `related-work.md`.

**What still separates this, as a conjunction:** every one of those is steered
by a **gradient or a loss inside a training loop**, and several are **told
where the task boundary is**. Here structural change is steered by local reward
and a metabolic economy in which failure to pay is death, in an organism that
is never not running, with no task boundary supplied and no global objective to
differentiate. *Unverified* — dynamic sparse training is large enough that a
reward-driven boundary-free variant may well exist.

**One boundary to state honestly, because it is a live design tension.**
[L-047](../LEARNINGS.md) found that growth **cannot originate** multi-leg
routes — "no pay for half a bridge" — so on the current substrate, origination
comes from the *innate scaffold*, and the registered attack on the 2/3 shelf is
**scaffold selection**, which is a selection loop over inherited wirings. That
is not a contradiction of Javid's framing (biology does both: evolution sets
coarse routing, within-lifetime plasticity refines it) but the division of
labour should be explicit rather than assumed, or the project will end up
claiming within-lifetime adaptation for work that selection did.

**Reference frames** are Hawkins' central claim and cannot be claimed here.
What would differ is the *origin*: Numenta posits the machinery; backlog §G
asks whether it can be grown from zero edges. "Learns reference frames" is not
novel; "grows the machinery that represents them" is the claim, and it is
unbuilt.

## 3. The theoretical gain, if the mechanism works [J→C]

This is what Javid actually asked for, and it is worth stating cleanly because
the answer is *not* "it adapts better".

1. **Adaptation without a family boundary.** Every fast-adapting system in the
   literature buys speed in advance with coverage of a task family, and
   degrades outside it ([H-027](../HYPOTHESES.md)). Adaptation that comes from
   *structure* has no such boundary — the game it has never seen is the same
   kind of event as the game it has. That is a **categorical** difference, not
   a quantitative one, and it is the only claim here that a bigger meta-RL
   training run cannot answer.
2. **Removing the freeze removes a category of problem.** Distribution shift,
   retraining cadence, drift monitoring, fine-tuning regressions, eval-set
   contamination and staleness are all downstream of the existence of a frozen
   artefact. They are not hard problems in a system that has no frozen artefact
   — they are not *problems*.
3. **Capacity as a response to novelty rather than a fixed budget.**
   Catastrophic forgetting is currently fought by rationing fixed capacity —
   regularisation (EWC) or replay. An organism that grows can give a new
   competence new tissue. Progressive Networks do this and must be *told* where
   the task boundary is; an always-on organism has no boundaries to be told.
4. **A cost structure that makes "always on" affordable.** Local rules, no
   backward pass, no gradient — the reason a frontier model cannot be
   always-learning is that its update is a training run. This is the standing
   neuromorphic argument and it is why the substrate choice is not incidental.
5. **The organism's own history is the curriculum.** No dataset, no task
   designer, no benchmark author deciding what comes next.

**And the honest counterweight: nobody lacks this for want of wanting it.**
Three structural reasons, all of which this project has already hit its head
on. Credit assignment degrades with scale (REINFORCE variance —
`open-problems` §1). Stability-versus-plasticity is a genuine trade-off, not a
bug to engineer away (though [L-037](../LEARNINGS.md) is a small piece of good
news: at this scale memory and flexibility came apart into two dials). And an
always-on system has **no held-out set, no epoch and no checkpoint**, which is
why this project had to build a measurement doctrine before it could claim
anything — a real cost, and arguably where its most defensible contribution
currently sits.

## 4. Making "living" falsifiable [J→C]

"It is alive, therefore it is different" is not yet a claim. It becomes one
when it names its control, and the control is obvious once stated:
**a conventional model retrained from scratch on everything-so-far, on a
schedule.** That control has a freeze; it just has a short one. If it matches
the living organism on the same battery, then "living" is an implementation
convenience and not a property, and every argument in §3 is aesthetic.

Registered as [H-028](../HYPOTHESES.md). This is the entry's most useful
output: it converts the project's founding intuition from a framing into
something that can lose.

## Prior art & novelty

- **Similar:** all of §1 and §2 — the framing is developmental robotics,
  continual/lifelong learning, the Alberta Plan and HTM; the mechanism family
  is dynamic sparse training, DEEP R and synaptic sampling; reference frames
  are Hawkins'.
- **Different:** structural change steered by local reward under a metabolic
  economy with death, with no task boundaries and no differentiable objective.
- **Novel (claimed):** nothing standalone. The conjunction in §2, *unverified
  against literature*, and the honest expectation from
  [2026-08-22](./2026-08-22-2022-001-is-not-the-ipnn.md)'s three-seams result
  is that a closer analogue exists than one evening can find.

## Decisions

1. **`related-work.md` gains two sections** — the "living AI" program map, and
   the within-lifetime-versus-evolutionary split. Both banner-marked
   *unverified, written from model knowledge, not searched*.
2. **H-028 registered.** The periodically-retrained control becomes a required
   arm in any experiment that claims a benefit *from* being always-on.
3. **Novelty is explicitly demoted as a goal.** Per
   [L-011](../LEARNINGS.md) — prior art is a check on claims, not a source of
   designs — none of this changes what gets built. It changes only what may be
   said. The value proposition of this project is not that nobody thought of a
   living AI; it is that everybody who did, failed on mechanism.
4. **No experiment promoted.** Substrate work remains the standing priority.

## Threats to validity

Two, both about this entry rather than the project. **First**, §1 and §2 were
written from model knowledge on a project whose rule 12 requires searched prior
art; treating them as searched would be a rule-12 violation wearing a citation.
**Second**, §3 is advocacy. It is the strongest honest case for the project's
premise, written in one pass, by a participant — the failure mode is that a
persuasive list of theoretical gains becomes a reason not to test them, which
is exactly the shape [H-008](../HYPOTHESES.md) warns about. H-028 exists to
give §3 something to lose against.

## Next

No iteration scheduled. Two reading tasks are now owed and should be done
together: the adaptation/meta-RL search behind
[2026-08-26-2341](./2026-08-26-2341-adaptability-the-measure.md) §4, and the
living-AI/structural-plasticity search behind this entry. Substrate work
(scaffold selection, the 2/3 shelf) is unaffected and remains the priority.
