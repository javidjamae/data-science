# Original IPNN abstract written and published (retrospective entry)

- **Entry:** `2023-07-12-2228-original-abstract-published`
- **When:** 2023-07-12, commit timestamped 22:28 CDT
- **Who:** Javid (solo; the abstract was drafted with LLM assistance)
- **Experiment:** theory — no code, no experiment
- **Code state:** git `6ff96d3a87e1e19b75d275bff1e691b8d6c71eb1` ("added abstract for my idea on ipnn"), adding `ipnn/README.md` and `ipnn/abstract.md`
- **Re-run:** `git show 6ff96d3:ipnn/abstract.md`

> **Retrospective entry — written 2026-08-20, three years after the fact.**
> The journal did not exist in 2023; the only contemporaneous record is the
> git commit itself. This entry is reconstructed *from that commit and the
> surrounding history*, not from memory of the session, and is filed under
> its original date so the timeline reads correctly. Everything stated as
> fact below is verifiable from `git log` / `git show`; anything about
> intent or state of mind at the time is marked as inference. Per the
> append-only rule, this entry is documentary — it adds a record that was
> never written, and does not revise any existing entry.

## In plain words

Three years before any code was written, the idea was written down and put
in public. On the night of 12 July 2023, Javid committed a short paper-style
abstract — *"Rethinking Neural Networks: An Iterative, Probabilistic
Approach"* — to a public GitHub repository. It described a neural network
whose neurons fire by chance rather than by fixed rule, whose output loops
back into its own input so it can "think about" its answer over several
passes, and which learns while it runs rather than being trained once and
frozen.

Nothing was built. No experiment was run. It was an idea, dated and made
public — and then it sat untouched for three years while other things took
priority, until the engine was finally built on 16 August 2026.

Two things about that gap are worth recording. First, the public timestamp
is the whole IP strategy of this project (see
[disclosure-policy.md](../../disclosure-policy.md)), and it starts here, in
2023 — not in 2026. Second, the core mechanism survived the three years
essentially unchanged: probabilistic firing, Beta-distributed confidence,
recurrent iteration, learning during use. What changed was everything
*around* it — the goal, the honesty about what is unsolved, and the
admission of whose prior work this stands on.

## Objective

Get the idea out of Javid's head and into a dated, public, citable form.
(Inferred from the artifact and the commit message; no stated objective
exists from the time.)

## Gate (pre-registered)

**None.** No gate, no hypothesis register, no success criterion existed —
the journal, its rules, and the experiment/`design.md` convention were all
created 2026-08-16. That absence is itself the process finding: for three
years the project had exactly one artifact and no mechanism to move it
forward. The 2026 revival began by building the missing apparatus
(vision, journal, gates, disclosure policy) before writing engine code.

## Hypotheses

None registered at the time. The abstract does state one hypothesis in
prose, quoted verbatim:

> Our hypothesis is that by allowing the network to iteratively process its
> input and "think" about the results over time, the IPNN could potentially
> improve prediction accuracy and robustness.

Status as of this writing: **untested.** Iterative consideration is in the
theory ([abstract.md](../../abstract.md) §3) but the M1 engine answers from
a sliding window over ticks; no experiment has yet isolated iteration count
as a variable, and no accuracy-vs-iterations comparison exists.

## Method

Prose only. Two files added in one commit:

- `ipnn/README.md` — four lines: "my thoughts on a new type of Neural
  Network that I have been theorizing that I call an Iterative,
  Probabalistic Neural Network (IPNN)", linking to the abstract.
- `ipnn/abstract.md` — ~600 words in seven sections (Abstract,
  Introduction, Architecture, Learning Mechanism, Hypothesis and Approach,
  Potential Implications, Conclusion), written in paper voice.

The repository (`github.com/javidjamae/data-science`) has been **public
since its creation on 2020-08-05**, so the commit constituted publication
in the priority-through-publication sense the moment it was pushed. It was
not submitted to arXiv, a venue, or a blog as far as the repository record
shows — if it was posted elsewhere, that link belongs in this entry.

## Results

What the 2023 abstract actually claimed, in its own words:

- **Probabilistic firing.** "Each neuron in the network has a Beta
  function-defined firing probability, controlled by alpha and beta
  parameters."
- **Recurrence for consideration.** "incorporates recurrent connections
  from the output layer to the input layer, enabling the network to
  'consider' its output over multiple iterations."
- **Labels as inputs.** "neurons on the input layer representing the known
  labels" — the supervision channel is a sensory channel.
- **Learning by tightening.** "When labels are provided, the alpha and beta
  parameters of each neuron are adjusted to tighten the probability
  distribution that defines whether a neuron will fire."
- **Real-time learning as the point.** "their rigid nature and inability to
  adapt in real-time have long been areas of concern" → "allowing for
  real-time learning and adaptation."
- **Output as a distribution.** "the output can be treated as a probability
  distribution over possible results" — uncertainty quantification as a
  side effect of stochastic firing.
- **Honest about its own status:** "it also opens up many questions
  regarding its implementation, scalability, efficiency, and the comparison
  of its performance with existing methods."

Timeline as recorded by git:

| Date | Event |
|---|---|
| 2020-08-05 | `data-science` repo created, public |
| **2023-07-12 22:28 CDT** | **`6ff96d3` — IPNN README + abstract published** |
| 2023-08-07 → 2023-10 | ML self-study push (masters-prep: MBL/PAC, MNIST net, NLP course, perceptron, makemore, nanoGPT); `masters-prep/online-learning/` syllabus added 2023-09-28. No IPNN commits. |
| 2023-10-02 → 2026-04-02 | Repo activity trails off; last pre-revival commit `dd9659c` 2026-04-02. No IPNN commits for **~3 years, 1 month** |
| 2026-08-16 02:25–02:48 | Experiment 001 design phase |
| 2026-08-16 02:48–02:58 | M0+M1 build; gate passed after one collapse-and-fix cycle ([entry](./2026-08-16-0248-experiment-001-m0-m1-first-build.md)) |
| 2026-08-16 03:07 | `e42a500` — vision-led docs, journal, rewritten abstract, M1-passing engine |
| 2026-08-16 03:15 | `ed6bed9` — disclosure policy + licensing (Apache-2.0 / CC BY 4.0) |
| 2026-08-16 03:21 | `92af8fa` — old online-learning lineage added to related-work |
| 2026-08-16 03:24 | `8d565e0` — prior-art & novelty made a required journal section |

Note the ordering: the IPNN abstract (2023-07-12) **predates** the
masters-prep ML study push (first commit 2023-08-07) by roughly four weeks.
The idea came before the formal study, not out of it.

Diff of the abstract, 2023 → 2026: 235 lines changed, 79 removed — a
rewrite, not an edit. Title changed from *"Rethinking Neural Networks: An
Iterative, Probabilistic Approach"* to *"IPNN: An Iterative, Probabilistic
Neural Network for Always-On, Real-Time Learning."*

## Analysis

**What survived three years unchanged** (present in both the 2023 and 2026
abstracts): stochastic neuron firing; Beta-distributed per-synapse
confidence that tightens with evidence; output→input recurrence producing
multi-iteration "consideration"; learning during operation rather than in a
training phase; output read as a distribution rather than a point estimate.
The mechanism core is the same object.

**What changed, and why it matters:**

1. **Supervision model.** 2023: labels arrive as input neurons and drive
   the α/β update directly — supervised learning in probabilistic clothing.
   2026: a scalar broadcast reward and a three-factor rule; the organism
   never sees labels at all (entry 1, Method). This is the single largest
   departure from the original.
2. **Goal.** 2023: the target was "improve prediction accuracy and
   robustness" — competing with conventional networks on their own metric.
   2026: the goal is the *regime* (always-on, multi-sense, low-power),
   explicitly not terminal accuracy ([abstract.md](../../abstract.md) §8).
   The 2023 framing would have made every experiment a losing comparison.
3. **Scope added.** Multi-sense/continual learning, catastrophic forgetting
   as a first-class concern, neuromorphic/low-power motivation, the "urge"
   and the legal silence action — none of these are in the 2023 text.
4. **Honesty added.** The 2023 abstract lists open questions vaguely
   ("implementation, scalability, efficiency"). The 2026 material names
   credit assignment, stability, forgetting, and scaling as specific
   possible failure modes ([open-problems.md](../../open-problems.md)) and
   holds a related-work file naming prior art the 2023 text did not cite at
   all.
5. **Novelty language dropped.** 2023 says "novel" five times and cites
   nobody. 2026 says most ingredients have respectable lineages and the bet
   is on the combination. Same idea, calibrated claim — the change that
   `8d565e0` later made mandatory per entry.

**Inference (not fact):** the three-year gap looks less like abandonment
than like a missing apparatus. The idea was recorded in the one form Javid
had available in 2023 — a paper abstract — and a paper abstract has no next
step for a solo researcher without a lab. What unblocked it in 2026 was
reframing it as a buildable experiment with a gate, not any new insight
about the mechanism.

## Prior art & novelty

Assessed retrospectively, against
[related-work.md](../../related-work.md) as it stands in 2026 — the 2023
abstract itself contains **zero citations**, which is the finding.

- **Similar:** essentially every element of the 2023 abstract had prior
  art at the time of writing, uncited. Real-time learning with no training
  phase is the old online-learning tradition (Rosenblatt 1958, Widrow 1960,
  Barto/Sutton/Anderson 1983, Grossberg's ART 1976–87). Stochastic binary
  neurons with recurrent settling are Boltzmann machines. Probabilistic
  firing with Beta/evidence-count semantics is BCPNN's neighborhood.
  Learning as conjugate evidence accumulation is textbook Bayesian.
- **Different:** the 2023 text's own combination — Beta-parameterized
  *firing probability* per neuron, updated by supervision arriving as
  ordinary input activity, with output→input recurrence used as
  deliberation.
- **Novel (claimed):** **retracted for this entry.** The 2023 abstract's
  repeated "novel approach" claim was made without a literature search and
  does not survive one. The current honest position is entry 1's: nothing
  in the results is novel; candidate architectural novelty is narrow
  (silence action + urge; evidence-gated plasticity in a reward-modulated
  rule) and remains *unverified against literature*.
- **Priority, separately:** the 2023 date does establish a public,
  timestamped disclosure of the mechanism core — relevant to
  [disclosure-policy.md](../../disclosure-policy.md)'s
  priority-through-publication strategy, which this entry now dates to
  2023-07-12 rather than 2026-08-16.

## Learnings

No new `L-###` learnings. This entry is documentary — it records what was
claimed and when, and holds no experimental evidence. The substantive
observations above (uncited novelty claims; goal reframing from accuracy to
regime) are already covered by the prior-art rule adopted in `8d565e0` and
by [abstract.md](../../abstract.md) §8.

## Decisions

1. **File this entry under its 2023 date**, at the bottom of the journal
   index, rather than dating it 2026-08-20 — the index is a timeline, and
   the project's public record starts in 2023.
2. **Retrospective entries are permitted but must be banner-marked** as
   this one is, stating what is verifiable versus reconstructed. The
   append-only rule governs *revising* entries, not adding missing history.
3. **Date the disclosure record to 2023-07-12.** The
   priority-through-publication clock started with `6ff96d3`, not with the
   2026 revival.
4. **The 2023 abstract stays in history, unmodified.** It is retrievable at
   `git show 6ff96d3:ipnn/abstract.md`; the rewrite is not a retraction and
   the original's uncited novelty claims are corrected here rather than
   erased there.

## Deviations

The entry template assumes a contemporaneous entry with a pre-registered
gate and a re-runnable command. Neither exists here. *Method* describes
what was written rather than what was built, *Results* quotes the artifact
rather than program output, and *Re-run* points at a git command that
retrieves the artifact.

## Threats to validity

1. **Reconstructed intent.** Everything about *why* the abstract was
   written, and why nothing followed for three years, is inference from the
   commit record. Javid's own recollection may differ and should override
   this entry (in a later entry, per the append-only rule).
2. **Commit dates are not publication dates.** Git records author/commit
   time on Javid's machine; the actual push to GitHub could have been
   later, and clock/timezone integrity is assumed. For a priority claim
   that mattered, GitHub's event record would be the stronger evidence.
3. **"Published" may be understated.** This entry can only see the
   repository. If the abstract was also posted to a blog, forum, or
   preprint server in 2023, that is a materially stronger disclosure and is
   missing from the timeline above.
4. **Retrospective coherence bias.** Reading a 2023 text through the 2026
   framing makes the throughline look cleaner than it was. The section
   "what survived" is a genuine textual comparison; the narrative
   connecting them is a story told after the fact.
5. **Authorship attribution.** The abstract's voice suggests LLM
   assistance ("we term", "we propose"), stated above as fact but inferred
   from style; it does not affect the priority record either way.

## Next

Chronologically, the next entry is the 2026-08-16 M0+M1 build
([2026-08-16-0248](./2026-08-16-0248-experiment-001-m0-m1-first-build.md)),
already filed. This entry registers no forward work of its own beyond one
standing item it makes visible:

**Untested claim carried forward from 2023:** iterative consideration
improves accuracy/robustness (quoted under *Hypotheses*). No experiment
isolates it. Candidate cheap test whenever an experiment is next designed:
vary the number of internal ticks before readout on the M1 task and measure
accuracy — does "thinking longer" do anything at all?
