# All three gates passed, and the experiment was wrong

- **Entry:** `2026-08-22-1530-transfer-retention`
- **When:** 2026-08-22 15:30–16:20 CDT
- **Who:** Javid + Claude (session)
- **Experiment:** [003 — transfer, retention and savings](../../experiments/003-transfer-retention/design.md)
- **Code state:** git `e7727bf` + this session's tools
- **Re-run:**
  ```
  cd ipnn/experiments/001-mnist-living-demo/app
  npx vite-node tools/exp003-transfer.ts 1 2 3 4 5
  npx vite-node tools/exp003-why.ts      1 2 3 4 5
  npx vite-node tools/exp003-reversal.ts 1 2 3 4 5
  ```

## In plain words

The project has claimed since it began that the test of intelligence is
learning a second thing without losing the first. Today it finally ran that
test. All three success criteria passed, written down in advance and untouched.

And the experiment was measuring the wrong thing.

Task B was three *new* shapes wired to the same three answer neurons as task A.
That was called "maximum interference." It is the opposite. Answer neuron 1
simply learns to fire for shape A₁ **and** shape B₁. Nothing anywhere ever
tells the organism that A₁ is no longer answer 1. There is no conflict to
survive, so surviving it proves nothing.

Two follow-up tests confirmed the emptiness: switching off the mechanism
designed to protect old memories changed nothing, and the two tasks turned out
to use the *same* neurons rather than conveniently separate ones. Both came
back blank because there was never anything to protect against.

So we ran the test that does contradict prior learning: same shapes, shuffled
answers. Shape A₁ now means answer 2. **It forgot completely** — 6.9% on the
original mapping, where random guessing scores 33%. Below chance, because it
is not confused, it is confidently giving the new answer. And relearning was
not easier for having learned before: it took **2,093 trials against 273** for
the original, and on one seed it never got there at all.

The catastrophic forgetting we predicted was real. Our first test just could
not see it.

## Results

**Original protocol, 5 seeds. All three pre-registered gates passed.**

```
TRIALS TO CRITERION            RETENTION (frozen)
seed    A    B  B-naive        A after B   A after A (ceiling)
  1   229  197    255            0.947          0.933
  2   259  170    200            0.867          0.953
  3   278  232    256            0.860          0.953
  4   342  238    213            0.840          0.947
  5   259  269    275            0.847          0.953

TRANSFER  B after A vs B naive: +19 trials, faster on 4/5      PASS
RETENTION A after B 0.872 vs ceiling 0.948, drop 0.076         PASS
SAVINGS   relearn A vs naive A: +173 trials, 5/5               PASS
```

**Ablation 1 — is consolidation protecting task A?** `consolidation: false`
removes the `1/(1+n/n₀)` term entirely, so every weight stays fully plastic.

```
mean A-retention: consolidation ON 0.872   OFF 0.889   (chance 0.333)
```

**Ablation 2 — do the tasks use different pool neurons?** Top-24 overlap of the
active pool population (≈ targetPoolSparsity × poolSize):

```
seed 1: within task A 0.38   A vs B 0.34
seed 2: within task A 0.38   A vs B 0.37
seed 3: within task A 0.24   A vs B 0.30
```

**The corrected test — reversal.** Task A's stimuli, labels permuted [1,2,0],
so every stimulus is reassigned and none is left alone.

```
seed | trials A | union 2nd | reversal 2nd | A retained: union / reversal / control
  1  |    229   |     197   |     2219     |   0.947 / 0.020 / 0.913
  2  |    259   |     170   |     2479     |   0.867 / 0.027 / 0.867
  3  |    278   |     232   |     2535     |   0.860 / 0.027 / 0.847
  4  |    342   |     238   |     none     |   0.840 / 0.240 / 0.920
  5  |    259   |     269   |     1138     |   0.847 / 0.033 / 0.840

task A retained after UNION 0.872 · after REVERSAL 0.069 · after MORE A 0.877
chance 0.333
reversal took 2093 trials against 273 to learn A originally
```

**Is consolidation causing the slow reversal?** (L-004's predicted tension.)

```
mean reversal trials: consolidation ON 2093 (4/5 reached)  OFF 2488 (5/5 reached)
per-seed OFF: 636, 2923, 3645, 2496, 2741
```

## Analysis

### The design was wrong, and the gates could not detect it

003's design §2 asserts that task B on the same outputs is "maximum
interference, and it is what the success criterion describes." Both halves are
false.

Task B's glyphs are *different stimuli*. Output 1 learns to fire for A₁ and
also for B₁, and the two demands are compatible — a single weight vector
satisfies both, and nothing in the training signal ever contradicts A. What was
measured was **whether the readout has capacity for six patterns instead of
three.** It does. That is a fact about capacity, not about retention.

This is worth stating carefully because pre-registration did not catch it and
could not have. **Writing gates in advance protects against moving the
goalposts; it does not protect against putting the goalposts in the wrong
field.** All three gates were honest, met, and uninformative.

The two ablations are what exposed it, and only in hindsight. Consolidation off
changed retention by −0.017 — *slightly better without it*. And the tasks turn
out to share pool neurons at the same rate task A's own glyphs share them with
each other (A-vs-B 0.30–0.37 against within-A 0.24–0.38), so there is no
convenient code separation either. Two mechanisms that could have explained the
retention, both absent. The correct reading was not "some third mechanism is
protecting it" but "nothing is attacking it."

### What reversal shows

Permute the labels and the picture inverts completely.

**Retention collapses to 0.069 — far *below* the 0.333 chance line.** That
number matters: a forgotten mapping would score at chance, because the organism
would be guessing. Scoring at 0.069 means it is reliably giving the *new*
answer. This is not decay or confusion, it is **systematic replacement**, and
it is the sharpest possible form of catastrophic forgetting. (Compare
[L-006](../LEARNINGS.md), which reads *pinned at exactly chance* as the
signature of collapse; pinned far below chance is the signature of a mapping
that was cleanly overwritten.)

**And prior learning made relearning harder, not easier.** Reversal took 2,093
trials against 273 to learn task A from nothing — **7.7× slower** — with one
seed failing to reach criterion in 4,000 trials at all. That is *proactive
interference*: the existing weights actively push toward the old answer and
must be driven through zero and out the other side. It is the precise opposite
of savings ([H-005](../HYPOTHESES.md)), which predicts prior exposure makes
relearning cheaper.

### Consolidation is not the culprit, which refutes a standing prediction

[L-004](../LEARNINGS.md) has said since the first entry that "consolidated
memory" and "frozen wrong answer" are one mechanism with different valence, and
predicted this tension would appear at every scale. Reversal is where it should
bite hardest — well-evidenced weights resisting the update they now need.

It does not. Reversal with consolidation on took 2,093 trials; off, 2,488,
with per-seed values from 636 to 3,645. The variance dwarfs the difference and
the sign is the wrong way round. **The slow reversal is a property of the
learning rule's noisy credit assignment, not of the consolidation term.** L-004
stands as a principle but gains no support here, and its first real test came
back empty.

### The savings gate passed on an artifact

`T_A2` reads exactly **100 on all five seeds**, and 100 is `WINDOW` — the
criterion is only checkable once the rolling window is full, so 100 is the
measurement's floor, not a duration. It means "already at criterion the first
moment we could look", which is a restatement of the 0.872 retention rather
than independent evidence of savings. **The savings gate should be treated as
not run.** Registered as a measurement defect below.

### What 003 actually established

- **001 has capacity for six patterns on three outputs**, learning the second
  set without disturbing the first, because the two are compatible.
- **When genuinely contradicted, 001 forgets completely and below chance**, and
  relearning is 7.7× harder than learning from scratch.
- **Neither consolidation nor code separation is doing any protective work** in
  the compatible case, and consolidation is not causing the damage in the
  contradictory one.

The prediction in design §4 — catastrophic forgetting — was **right about the
substrate and wrong about which experiment would show it.** L-010's reasoning
holds: a single learnable layer has nowhere to put task-general structure, and
under real contradiction it demonstrates exactly that.

## Learnings

- **L-029:** "Same outputs, different stimuli" is not interference, it is a
  **union**. Output *k* simply learns to accept two inputs and nothing
  contradicts the first, so retention under that protocol measures readout
  *capacity*, not retention. To test retention, the second task must contradict
  the first — reassign the **same** stimuli (reversal). *Evidence:* this entry
  — union retention 0.872, reversal retention 0.069, same substrate and seeds.
- **L-030:** Under genuine contradiction, 001 forgets **below chance** (0.069
  against 0.333) — systematic replacement rather than decay — and prior
  learning makes reversal **7.7× slower** than learning from scratch (2,093 vs
  273 trials, one seed never reaching criterion). Proactive interference, the
  opposite of savings. *Evidence:* this entry, 5 seeds.
- **L-031:** A pre-registered gate can pass for the wrong reason. All three of
  003's gates were written in advance, honestly met, and uninformative, because
  the protocol did not create the condition the gates were about.
  **Pre-registration protects against moving the goalposts, not against
  standing them in the wrong field.** The check it does not provide is: *state
  what the experiment would look like if the manipulation failed to take* — and
  003's ablations are what eventually supplied it. *Evidence:* this entry.
- **L-032:** A criterion checked only once a rolling window is full has a
  **floor at the window length**. "Reached criterion in 100 trials" with
  `WINDOW = 100` means "was already at criterion when first checkable", not a
  measured duration, and reporting it as the latter turns a censored value into
  a false positive. *Evidence:* 003's savings gate, `T_A2 = 100` on 5/5 seeds.

## Decisions

1. **003's design §2 is corrected in place** with a note, as 002's §4 was —
   the claim that same-outputs is maximum interference is marked false rather
   than quietly dropped.
2. **Reversal becomes 003's primary protocol**, and the union arm is retained
   as the capacity control it turned out to be.
3. **The savings gate is void** pending a criterion measurement without a
   floor. [H-005](../HYPOTHESES.md) stays open and untested.
4. **[L-004](../LEARNINGS.md) is not supported by its first real test** and is
   annotated accordingly. It remains active as a principle.
5. **[Backlog track H](../../experiment-ideas.md) gains a hard requirement:**
   any Games entrant must be scored on *contradictory* task sequences, not
   merely additive ones, or its retention score measures capacity. This is the
   single most useful thing 003 produced for the ecosystem idea.

## Threats to validity

1. **Reversal may be too hard rather than representatively hard.** A full
   3-cycle permutation reassigns every stimulus; a single swap would be gentler
   and is not run.
2. **Five seeds, enormous variance.** Reversal ranged 636–3,645 trials with one
   failure. No confidence interval is claimed.
3. **The consolidation comparison is underpowered** — 4/5 versus 5/5 reaching
   criterion, and the means differ by less than the spread. "Not the culprit"
   is the honest reading; "no effect" would be stronger than the data supports.
4. **Only 001.** 002 cannot learn task A at all, so nothing here generalises.
5. **The union arm's transfer result (+19 trials, 4/5 seeds) is weak** and was
   not re-examined after the design flaw surfaced. Task B is also somewhat
   easier than task A naive (240 vs 273 trials), which is a confound the design
   anticipated and the result does not clear.

## Next

**Re-run savings without the floor** — check the criterion from trial 1 using a
shorter warm-up window, so relearning duration is measurable rather than
censored. Cheap, and it is the one pre-registered gate still genuinely open.

Then **004** (the iterative organism), which is designed and waiting.
