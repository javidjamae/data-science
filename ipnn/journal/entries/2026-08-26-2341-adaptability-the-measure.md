# Adaptability: the measure we never named, and the world the games run in

- **Entry:** `2026-08-26-2341-adaptability-the-measure`
- **When:** 2026-08-26 23:41 CDT
- **Who:** Javid (the questions and the framing) + Claude (the vocabulary audit and the prior-art map)
- **Kind:** theory / direction — **no code ran** (journal rule 6)
- **Code state:** git `aaba633` (branch `ipnn/exp005-rules-as-a-sense`, clean at entry open)

## In plain words

Javid came back to the project after three days away and asked, in effect,
"what is the thing we are actually trying to measure, and does anyone else
already have it?" He described a standardised toy world with a set of games
thrown at one agent, and he sharpened what counts as success: **not how fast
it can be trained on the new game, but how good it is on the new game
immediately** — the way a person picking up an unfamiliar ball already knows
how to throw and catch.

Three things came out of it. **One:** the project has been measuring this for
weeks and has never given it a single name — we have four names for
neighbouring things and no name for the thing itself. **Two:** the outside
field has an enormous literature on it under four different names, and
`related-work.md` contained none of it. **Three:** Javid's "good right from the
start" is a *different currency* from the one the ledger has been carrying
(trials-to-criterion, H-011), and the difference is not cosmetic.

## The questions, as asked [J]

Recorded verbatim, because the wording is the content.

> "I want to create a series of games on a standardized 'world' where an agent
> has 'senses' that can perceive and interact with that world. But I want there
> to be a set of challenges that we throw at it, and the same agent has to
> learn how to adapt."

> "The key thing I want to think through is adaptability. The measure isn't how
> fast it can 'train' through thousands of iterations. The key thing is that it
> can come in to a new, yet similar activity and be pretty good right from the
> start. For example, a human can perhaps pick up a new type of ball and know
> how to throw and catch it. It doesn't have to learn throwing and catching
> from scratch."

> "What have we been calling this measure in our research? adaptability?"

> "What is well known in the world of adaptability, and how does our new IPNN
> architecture change things? … do other AI models like reinforcement learning
> already do well at 'adaptability'?"

## 1. The proposal is already registered — as §H and H-017

The standardised world with a growing set of games thrown at one agent is
[backlog track §H, "The Games"](../../experiment-ideas.md#h-the-games--an-evaluation-ecosystem-not-another-benchmark),
raised by Javid on 2026-08-22 and never designed, plus
[H-017](../HYPOTHESES.md) ("test with games it was never trained on"). The
world is already specified: the 8×8 binary grid experiment 001 uses. So this is
not a new direction; it is the same direction arrived at independently four
days later, which is itself mild evidence that it is the real one.

What tonight adds to §H is the **currency** (below) and the
**amortisation control** (H-027).

## 2. The vocabulary audit [C]

**"Adaptability" appears zero times in the IPNN documents.** The word has never
been used. What exists instead is four names for four neighbouring things, and
the gap between them is where tonight's thinking lives:

| What we say | Where it lives | What it actually measures |
|---|---|---|
| **"Adapt, don't relearn"** | Javid's goal statement, [2026-08-22-1917](./2026-08-22-1917-adapt-not-relearn.md) | The *aspiration*. Not operationalised. |
| **Savings** (ladder rung 4) | [vision.md](../../vision.md#the-ladder-of-evidence), H-005 | Is *re*-learning faster than first learning? Same task, second time. |
| **Transfer** (rung 5) | vision.md, H-001 | Does task B come faster **having learned A**? Measured in trials-to-criterion. |
| **Retention** (rung 6) | vision.md, H-001 | After B, does A survive? |
| **Learning to learn** (rung 7) | vision.md; Harlow 1949, "learning set" | Does the **Nth** task come faster than the first? |
| **Flexibility** | [L-037](../LEARNINGS.md), [L-038](../LEARNINGS.md), [L-039](../LEARNINGS.md) | Coined for the *reversal* work: can it change its mind, and is synaptic flexibility the same as behavioural flexibility (no — L-039). |
| **Acquisition × retention** | §H scoring | The proposed leaderboard, not a mechanism claim. |

**The answer to Javid's question is: rung 7, Harlow's learning set.** The new
ball is exactly Harlow's monkeys eventually solving a novel discrimination in
one trial. So the project does have a name for it — it is just buried in a
table in `vision.md`, has never been run, and no entry has ever cited it.

**The naming gap, stated plainly:** *flexibility* was coined for re-binding a
**known** stimulus set under a **new rule** (reversal). *Learning to learn* is
about a **new task**. Those have been used loosely as if they were the same
question, and L-039 is the entry where they came apart — 52% synaptic
plasticity, 0% frozen, and still 1-of-11 on novel rules. They are not the same
question and should stop sharing vocabulary.

## 3. Javid's sharpening is a new currency, not a restatement [J]

[H-011](../HYPOTHESES.md) says trials-to-criterion is the right currency,
because accuracy-at-N punishes a slow learner for being slow. That is correct
*within* a task. But Javid's statement is about a different quantity:

> "The measure isn't how fast it can 'train' through thousands of iterations …
> it can come in to a new, yet similar activity and be pretty good right from
> the start."

**Trials-to-criterion is a rate. "Good right from the start" is an intercept.**
A system can have a superb rate and a terrible intercept — which is precisely
what 001 does under a rule flip: it eventually gets there and it starts *below
chance* ([L-030](../LEARNINGS.md): 0.069 against 0.333). Reporting only
trials-to-criterion would have hidden the most damning number the project has
produced.

Registered as [H-026](../HYPOTHESES.md): first-exposure competence is the
primary adaptability currency; trials-to-criterion is the secondary one; and
they must not be collapsed — which is the same refusal §H's scoring section
already argues for on speed-versus-accuracy.

## 4. Does the outside field already do this? [J→C]

Yes, extensively, under four names none of which were in `related-work.md`
until tonight — the full map is now in
[related-work.md § Adaptation and fast transfer in RL](../../related-work.md),
with an honesty banner that it was written from model knowledge and **not from
a live search**. Summary:

- **Meta-learning / meta-RL** (MAML, RL², PEARL) explicitly optimises for
  fast adaptation, and works — *within its meta-training task distribution*.
- **Domain randomisation / sim-to-real** literally solves the new-ball case,
  by training on ten thousand randomised balls first.
- **AdA** (DeepMind 2023) adapts to held-out tasks on human timescales and is
  the strongest published version of what Javid described.
- **In-context learning** in large models is the same shape as
  [H-018](../HYPOTHESES.md)'s rule-as-a-sense — adaptation with no weight
  change — reached from the other end at nine orders of magnitude more scale.

**The distinction worth carrying:** all of it is **amortised adaptation**. The
speed is bought in advance with coverage of the task family, and it decays
sharply outside that family. A human picking up a new ball is *plausibly* doing
something else, but we cannot assert that from the armchair — it is exactly the
sort of claim H-027 exists to test rather than assume.

**Where the field is genuinely weak, and it is our half:** nobody scores
acquisition against the entrant's *own growing history*. Meta-RL benchmarks
measure first-exposure efficiency on a fresh agent. And the 2024 *loss of
plasticity* result — networks that stop being able to learn anything after long
task sequences — is [L-034](../LEARNINGS.md) rediscovered in a completely
different substrate, which is worth noticing: two unrelated systems, the same
terminal failure, from cumulative confidence that only ever rises.

## 5. Where IPNN actually stands on this, honestly

It stands **behind**. This must be in the record before any framing gets
optimistic:

- 001 under a rule flip: retention **0.069 vs 0.333 chance**, relearning
  **7.7× slower** than from scratch — proactive interference, the exact
  opposite of savings ([L-030](../LEARNINGS.md)).
- 001 over eight serial flips: **no reuse at all**, 0/4 seeds improve
  ([L-033](../LEARNINGS.md)). It is the project's calibrated known-negative
  for adaptability.
- Under never-returning rules, the fixed α/β organism recovers from **1 of 11**
  novel rules ([L-039](../LEARNINGS.md)) — tied with a *frozen* organism.
- 002, the actual IPNN substrate, has only just achieved learning **one**
  three-way task at all, and only with an innate scaffold it cannot originate
  itself ([L-045](../LEARNINGS.md), [L-047](../LEARNINGS.md)).

So the claim "IPNN changes adaptability" is a **hypothesis with no supporting
evidence and substantial contrary evidence**. What is defensible is the
*mechanism story*: if H-018's factoring exists, a protocol change costs a
re-bind rather than a re-fit, and that would be adaptation earned from
structure rather than amortised from coverage. Nothing has tested it —
experiment 005 is the test, and 005 is parked by the
[2026-08-22-2022 doctrine](./2026-08-22-2022-001-is-not-the-ipnn.md) until the
grown substrate converges.

## Decisions

1. **The journal captures Javid's questions, not only his conclusions [J].**
   Journal rule 6 said *thoughts* are first-class; it now says **thoughts and
   questions**, and an open question may be filed with no answer attached.
   Rationale, his: the questions are the part that would otherwise evaporate,
   and a question recorded on the day it was asked is worth more than a
   reconstruction of it three months later. `journal/README.md` rule 6 amended.
2. **`related-work.md` gains an adaptation-and-fast-transfer-in-RL section**,
   marked *unverified — written from model knowledge, not searched*. A real
   search is owed before any entry cites it as settled.
3. **H-026 and H-027 registered.** §H's scoring section and H-011 both need to
   absorb H-026 when §H is next touched; not done tonight, to avoid editing a
   backlog track from a theory entry.
4. **No experiment is promoted.** The doctrine stands: the standing priority is
   a growth model that converges. §H remains backlog, and it is *more* clearly
   backlog after tonight — an ecosystem needs an entrant that can learn game 1,
   and 002 reached that state 72 hours ago on one geometry with a scaffold it
   was handed.

## Prior art & novelty

- **Similar:** everything in §2 above is borrowed — Harlow 1949 for the
  measure, Ebbinghaus for savings, the continual-learning BWT/forgetting
  metrics for retention, and the whole meta-RL literature for the adaptation
  half. Chollet's *On the Measure of Intelligence* (2019) defines intelligence
  as skill-acquisition efficiency, which is H-011 and H-017 with a different
  accent.
- **Different:** the insistence on scoring **first-exposure competence and
  trials-to-criterion separately** (H-026), and on an **amortisation control**
  that makes held-out *family* rather than held-out *task* the unit (H-027).
- **Novel (claimed):** nothing. Both of the above are most likely present
  somewhere in the meta-RL evaluation literature under other names —
  *unverified against literature*, and the honest expectation is that a search
  finds them.

## Threats to validity

This entry is direction and vocabulary, not measurement, and it carries two
specific risks. **First,** the §4 prior-art map was written from model
knowledge on a project whose own rule 12 requires searched prior art; it is
banner-marked, and treating it as searched would be exactly the failure
[L-031](../LEARNINGS.md) warns about in a different dimension. **Second,** the
"amortised versus earned" distinction is *rhetorically* attractive in a way
that should raise suspicion — it is the shape of argument that lets a project
dismiss every competing result as not-really-the-same-thing. H-027 exists to
make it falsifiable; until it is run, it is a framing and should be labelled
one.

## Next

No iteration is scheduled from this entry. The queue is unchanged: scaffold
selection on 002, per [2026-08-23-0247](./2026-08-23-0247-confirmation.md)'s
named bottleneck (per-pattern route coverage, the 2/3 shelf). The owed work
created by this entry is a **real literature search** behind §4, which is a
reading task and not an experiment.
