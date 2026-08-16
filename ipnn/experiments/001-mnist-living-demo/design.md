# Experiment 001 — MNIST Living Demo

**Status:** M0 + M1 complete (gate passed 2026-08-16, see journal entry
[2026-08-16-0248](../../journal/entries/2026-08-16-0248-experiment-001-m0-m1-first-build.md))
— M2 next
**Question:** Can an IPNN organism learn digit recognition *in real time,
through interaction and reward alone* — no training phase, no backprop, no ML
libraries?

## 1. The experience we're building

A web app containing a living organism you teach like a child:

1. The teacher (automated, or you) flashes a digit — say a "9" — onto the
   organism's 64×64 visual sense.
2. The organism sits with it: internal activity circulates for some number of
   ticks. Maybe it produces nothing on its output register. Maybe it produces
   a "5".
3. If/when it produces "9", the teacher rewards it — the way a parent shows
   excitement when a child gets it right. If it's wrong: either ignore it or
   give a mild negative signal (**both modes are experiment variables — we
   try both**).
4. Move the "9" around the grid in real time. While it keeps saying "9",
   keep rewarding; when it stops, stop rewarding (or signal negatively).
5. Rotate it, zoom in/out. Same deal.
6. Pause auto-training and draw digits by hand with the mouse; reward
   manually with a button.
7. At any point, switch **learning off**. The organism gets no more rewards
   or penalties — and should keep producing correct outputs anyway.

Run this loop for thousands of stimuli and watch a rolling accuracy chart
climb *while it happens*. That chart climbing is the whole experiment.

## 2. Stack decision

**TypeScript + Vite, everything in the browser. No ML libraries, no backend.**

- Sim engine runs in a **Web Worker** (pure TypeScript, zero DOM
  dependencies), targeting 100+ ticks/sec; the UI thread renders at 30–60fps,
  decoupled from tick rate.
- UI: HTML canvas for the sense grid, activity view, and output register;
  keep frameworks minimal (vanilla TS or Preact at most).
- MNIST loaded client-side from the binary IDX files (a ~10MB fetch, or a
  bundled subset to start).
- Telemetry exportable as JSONL so analysis can happen in Python notebooks in
  this repo — the data science stays in Python; the *organism* lives in the
  browser.
- Brain state (all synapses + confidences) serializable to a file /
  localStorage: **the organism persists between sessions**, which is what
  "living" means operationally.

Why not Python: the experiment is fundamentally interactive — drawing,
dragging a digit in real time, watching activity — and a Python sim behind a
websocket adds latency, deployment friction, and two codebases. Since we're
building neurons from scratch, we lose nothing by skipping the Python ML
ecosystem. The engine stays a dependency-free pure-TS module so it runs
headless in tests (and could be ported to Python/Rust later if we outgrow the
browser).

## 3. Organism spec (v1 — concrete numbers to start iterating from)

Everything here is a starting point, expected to change; the point is to be
concrete enough to build.

**Time.** Discrete ticks. A stimulus presentation ≈ 50–200 ticks; blank
inter-stimulus interval ≈ 20–50 ticks. The organism ticks even when the
sense is blank.

**Layers.**
- *Visual sense:* 64×64 binary neurons (4096). Digits are 28×28 MNIST
  bitmaps (thresholded to binary) placed on the grid — centered at first,
  movable/rotatable/scalable later.
- *Core pool:* ~2,000 stochastic binary neurons. Sparse random connectivity:
  each pool neuron receives ~200 synapses drawn from the sense and from other
  pool neurons. v1 keeps pool recurrence *off* (feedforward + output feedback
  only) per open-problems §2, then adds it.
- *Output register:* 10 neurons (digits 0–9), receiving from the pool, with
  lateral inhibition among themselves, and **feedback connections back into
  the pool** — the iterative-consideration loop.

**Neuron.** Binary state per tick. Drive d = Σ(w·pre) + bias; fire with
probability σ(g·d). Sparsity enforced by k-winner-take-all in the pool
(top ~5% of drives eligible to fire) — both biologically sane and the main
stability lever.

**Synapse.** Three numbers: weight `w`, eligibility trace `e`, evidence count
`n` (the Beta-confidence, α+β in spirit).
- Every tick: `e ← λ·e + pre·post` (λ ≈ 0.95, a few-hundred-tick memory).
- On broadcast reward R: `w ← w + η·R·e / (1 + n/n₀)` and, when |R·e| is
  significant, `n ← n + 1`. High-`n` synapses are consolidated: they've
  earned their weights and barely move. (Consolidation can be toggled off —
  that's the forgetting ablation.)
- Reward baseline: use R − R̄ (running mean reward) to cut REINFORCE
  variance.

**Output semantics.** The organism "says k" when output neuron k's firing
rate over the last ~30 ticks exceeds threshold while all others are below
theirs. Silence is legal.

**Drive ("emotion").** A scalar urge that rises slowly while the organism is
silent and resets on any produced output; it scales output-layer excitability.
This makes the organism *want* to respond without being forced to — the
knob that solves the cold-start problem (open-problems §4).

## 4. The teacher

- **Auto mode:** present stimulus → wait up to K ticks for an output →
  correct: R = +1 for a few ticks; wrong: R = 0 (*ignore* schedule) or
  R = −0.2 (*correction* schedule); no output: R = 0 → blank interval →
  next stimulus. Teacher schedule (ignore vs correction, reward magnitude,
  curriculum) is a config object, logged with every run.
- **Manual mode:** auto-teacher paused. Human draws or places stimuli,
  rewards/corrects via buttons (or keys). Same R pathway — the human is just
  a slow teacher.

## 5. UI (single screen)

- **Sense panel:** the 64×64 grid — shows the current stimulus; drawable
  with the mouse; drag/rotate/zoom controls for the placed digit.
- **Brain panel:** live pool activity raster (which neurons fired this tick),
  firing-rate sparkline.
- **Output register:** 10 meters with the fire-rate threshold marked; the
  current "spoken" digit highlighted.
- **Teacher panel:** auto/manual toggle, **learning on/off toggle**, reward
  schedule selector, reward/correct buttons, speed control
  (ticks/sec ×1 – ×1000 for "run through thousands of stimuli").
- **Telemetry strip:** rolling accuracy, responses-per-minute, reward rate,
  weight/confidence histograms; JSONL export; brain save/load.

## 6. Milestones (each gates the next)

- **M0 — Scaffold.** Vite + TS project under `ipnn/experiments/001-mnist-living-demo/app/`,
  engine package with headless tests, MNIST loader.
- **M1 — Sanity learning, no UI. The critical gate.** Headless: 3 fixed 8×8
  patterns → 3 outputs, reward-only teacher. Success: sustained
  above-chance accuracy (>80% over a rolling window) from pure
  reward-modulated learning, reproducible across seeds. **If this fails,
  everything stops until the learning rule works** — we iterate on the rule
  (baseline, k-WTA %, trace decay, schedules), not the UI.
- **M2 — Minimal living demo.** Digit classes {0, 1, 2} centered on the
  grid, auto-teacher, live activity + accuracy display, speed control,
  learning toggle. Success: watchably climbing accuracy in ≤ a few thousand
  stimuli; accuracy persists with learning off.
- **M3 — Full experience.** All 10 classes; move/rotate/zoom stimuli
  (reward-shaped as in §1 steps 4–5); manual drawing mode; brain
  save/load; full telemetry. Success: the seven-step script in §1 is
  executable end to end.
- **M4 — Science.** (a) Teacher-schedule ablation: ignore vs correction vs
  shaping. (b) Forgetting probe: train {0–4}, then only {5–9}, re-test
  {0–4}, consolidation on vs off. (c) Reward-withdrawal drift: hours of
  learning-off ticks, measure decay. Results → `results.md`, narrative →
  `log.md`.

## 7. Risks and expectations

- **Learning may stall near chance** (REINFORCE variance — open-problems
  §1). Mitigations in order: reward baseline, sparser k-WTA, curriculum,
  shaping rewards, rate-based readout. This is why M1 exists.
- **Recurrence may destabilize** — hence v1 ships with output-feedback only.
- **Performance:** 2k neurons × ~200 synapses ≈ 400k synapses — trivial for
  typed arrays in a worker; headroom to 10× if needed.
- **Set expectations now:** from-scratch, reward-only, local-rule MNIST will
  not hit 99%. **60–80% on 10 classes would be a strong result.** The claim
  being tested is "it demonstrably learns in real time through interaction,"
  not "it rivals backprop."

## 8. Out of scope (this experiment)

Multiple senses, multimodal fusion, pool-recurrence studies beyond basic
stability, neuromorphic ports, any comparison-tuned backprop baseline beyond
a reference number. Those are experiments 002+.
