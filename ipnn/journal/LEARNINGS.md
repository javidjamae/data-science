# Learnings Ledger

Every durable learning, numbered and citable. Cite by ID in journal entries
and project docs. Never delete — supersede or refute, with a link.

Statuses: **active** · **superseded by L-###** · **refuted** (link entry).

| ID | Learning | Status | Evidence |
|---|---|---|---|
| L-001 | Reward-modulated three-factor learning (score-function eligibility × broadcast advantage) is sufficient for 3-pattern discrimination at toy scale — no backprop, no labels. Credit assignment's working answer cleared its first gate. | active | [2026-08-16-0248](./entries/2026-08-16-0248-experiment-001-m0-m1-first-build.md) |
| L-002 | Softmax saturation kills score-function learning: un-normalized summed drives push one output's p→1, making eligibility (fired − p)→0, so the dominant answer can never be punished — irrecoverable answer collapse. Normalizing drive by active-presynaptic count prevents it. | active | [2026-08-16-0248](./entries/2026-08-16-0248-experiment-001-m0-m1-first-build.md) |
| L-003 | Persistent exploration is load-bearing, not a nicety: ε-mixed sampling (ε=0.05, with the mixed probability used in eligibility so it matches the policy) is required for non-dominant outputs to ever claim patterns. | active | [2026-08-16-0248](./entries/2026-08-16-0248-experiment-001-m0-m1-first-build.md) |
| L-004 | Confidence–plasticity tension: the mechanism behind answer collapse (high confidence → zero learning signal) is the same mechanism IPNN deliberately uses for consolidation. "Consolidated memory" and "frozen wrong answer" are one phenomenon with different valence; expect this tension at every scale. | active | [2026-08-16-0248](./entries/2026-08-16-0248-experiment-001-m0-m1-first-build.md) |
| L-005 | Frozen retention holds at toy scale: after 500 rewarded trials, 100 trials with learning off scored 0.97 — behavior persists without ongoing reward (the living-model claim, design §1 step 7, verified small). | active | [2026-08-16-0248](./entries/2026-08-16-0248-experiment-001-m0-m1-first-build.md) |
| L-006 | Diagnostic heuristic: accuracy pinned at *exactly* chance across seeds signals deterministic collapse (always answering, always the same way), not noise. Noise wanders; collapse pins. | active | [2026-08-16-0248](./entries/2026-08-16-0248-experiment-001-m0-m1-first-build.md) |
| L-007 | Process: the headless gate-first sequencing (M1 before any UI) caught a paradigm-level defect in a <2s test loop. Instrument-then-diagnose beat guess-then-tune: one instrumented run identified collapse that hyperparameter twiddling would have chased for hours. | active | [2026-08-16-0248](./entries/2026-08-16-0248-experiment-001-m0-m1-first-build.md) |
