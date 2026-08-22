# Hypotheses Ledger

Open questions and claims-not-yet-tested, numbered and citable — the
forward-looking twin of [LEARNINGS.md](./LEARNINGS.md), which records what we
have already established.

A learning is something we know. A hypothesis is something we believe, or
suspect, or intend to find out. Both deserve an ID, because an idea that only
exists in a conversation is an idea that will be re-derived from scratch in
three months, usually worse.

**Statuses:** **open** (stated, untested) · **testing** (an experiment is
running against it) · **supported** (evidence for, cite the entry) ·
**refuted** (evidence against, cite the entry) · **superseded by H-###**.

Refuted hypotheses are never deleted. Being wrong in a specific, recorded way
is worth more than being vaguely right.

## Attribution

Every hypothesis carries its origin, because on a project run by a person and
a model together, *who thought of it* is part of the record and cannot be
reconstructed afterwards. See the attribution rule in
[README.md](./README.md#attribution-who-thought-of-it).

| Mark | Meaning |
|---|---|
| **[J]** | Javid's, in substance. Stated by him, not prompted by the model. |
| **[C]** | Claude's, in substance. Originated by the model. |
| **[J→C]** | Javid seeded it; Claude developed, formalised or operationalised it. |
| **[C→J]** | Claude proposed it; Javid selected, redirected or decided on it. |

## Ledger

| ID | Origin | Hypothesis | Status | Test |
|---|---|---|---|---|
| H-001 | **[J]** | **Intelligence is transfer and retention, not task accuracy.** A system that learns task A, then learns task B, and still performs A, is intelligent in a way that a system with higher accuracy on A alone is not. Accuracy on a single fixed task is a benchmark measure, and benchmarks are what this project explicitly is not for. | open | experiment 003 (transfer/retention), design owed |
| H-002 | **[J]** | **A living system may learn on timescales far longer than we have been observing.** A flat learning curve over N trials is not evidence of no learning; it may be evidence of an observation window chosen for convenience. A system that took a month of wall-clock to become competent, and then generalised, would be more interesting than one that converges in 800 trials. | **refuted for 002's M1 arm** (scope: this substrate, 128× the original window — every measure flat, persistence 5% throughout); open as a general principle | [2026-08-22-1215](./entries/2026-08-22-1215-longrun-and-a-correction.md) |
| H-003 | **[C]** | **Slow learning requires a slow variable.** A system in which every state variable has a short time constant cannot learn slowly — it reaches steady state and then stays there, and further time buys only more steady state. So H-002 is not a claim about patience; it is a claim about mechanism, and it is checkable by asking what the longest-lived quantity in the system is. | open | measure the timescale of every state variable in 002; compare against observed persistence |
| H-004 | **[C]** | **002's only genuinely accumulating variable is stored on its most ephemeral object.** The evidence count `n` only ever increases (design §7), which makes it the one monotone quantity in the substrate — and it lives on edges, ~94% of which die every generation. The slow variable is destroyed along with its carrier, so nothing accumulates across sleeps. Moving it to the *site* — a per-location trace that outlives the edges that created it and biases where regrowth goes — would give the substrate its first long-term memory. | open | a site-trace variant of 002, measured by structural persistence |
| H-005 | **[J→C]** | **Savings detects learning that behaviour cannot show.** A substrate may hold a latent trace it does not express. Re-teaching a previously-taught task should then be faster than teaching it to a naive control, *even if accuracy during the first exposure never left chance*. This is Ebbinghaus's method, not an ML method, and it is the right instrument for H-002: it can find learning inside a flat curve. | open | savings protocol on 001 and 002, with a never-taught control arm |
| H-006 | **[C]** | **Growth is activity-seeking but not correlation-seeking.** A node grows toward wherever there is activity, regardless of whether that activity has anything to do with its own. So each interior node accumulates ~5 mutually unrelated inputs and averages them into noise, which is a mechanism for L-013's one-hop wall. Growth that preferred *correlated* targets would build coherent receptive fields instead of random mixtures. | open | a correlation-seeking growth rule, measured by hop-2 decodability |
| H-007 | **[J→C]** | **Detection of learning should be a ladder, not a threshold.** Behaviour is the last thing to change, so a single behavioural gate cannot distinguish "no machinery" from "machinery that has not yet reached behaviour". Evidence should be sought at every level from structure upward, with each level having its own null. | open | the ladder in [vision.md](../vision.md#how-we-judge-whether-it-is-learning) |
| H-008 | **[C]** | **Unfalsifiability is this project's real occupational hazard.** "It is learning, just slowly and invisibly" is unanswerable unless every new instrument ships with a null and a pre-registered decision rule. H-002 and H-005 are worth pursuing *and* are exactly the shape of belief that quietly turns a research project into a faith. The defence is that no instrument enters the record without stating what result would count as no-learning. | open | applies to every entry; enforced by journal rule 3 |
| H-009 | **[J]** | **A measure of learning is admissible only if it is substrate-independent.** If it could be applied to an ant, an ape, a human and a machine, it is measuring learning; if it only makes sense for a system with a loss function, a train/test split and epochs, it is measuring a *training procedure* and may not transfer to a living model. Borrowed measures are fine — most of comparative psychology qualifies, most of the ML evaluation battery does not. | open | applied as an admission test to every rung in [vision.md](../vision.md#what-makes-a-measure-admissible) |
| H-010 | **[J→C]** | **Latent learning is the decisive test for H-002.** Expose the organism with reward switched off, then switch reward on, and compare trials-to-criterion against a naive control. If pre-exposure helps, it was acquiring structure the whole time while behaviour showed nothing — which is exactly the claim "it might be learning invisibly" makes. Tolman's 1930 rats did precisely this. It converts an unfalsifiable worry into a controlled experiment with a control group. | **refuted** — no pattern-specific latent learning on either substrate; the scrambled control matched or beat the pre-exposed arm, so the speed-up is warm-up (L-024) | [2026-08-22-1240](./entries/2026-08-22-1240-latent-learning.md) |
| H-011 | **[C]** | **Trials-to-criterion is the right currency, not accuracy.** "How many exposures to reach a fixed standard" is comparable across systems with different asymptotes and speeds; "accuracy at trial 2,000" is comparable to nothing, and silently penalises a slow learner for being slow rather than for failing. Every gate from 003 onward should be written in trials-to-criterion. | open | rewrite the 003 gates in this currency and check it reproduces 001's and 002's known results |
| H-012 | **[J→C]** | **Scaffolded growth: crystallisation needs positive feedback from what has already formed to what forms next.** 002 builds a strong, well-evidenced core and it never grows, because the growth rule cannot see it (L-028). Make growth's target weighting depend on consolidated structure — `A(target)·exp(−span/λ_g)·(1 + κ·evidence near target)` — so a core that forms becomes a frame later growth follows. Javid's clause "when it gets strong enough, faster things can grow around it" has no mechanism in the current design. | open | pre-registered measure: **the core must grow across checkpoints**, which it demonstrably does not today | 
