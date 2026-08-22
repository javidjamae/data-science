// Experiment 003 — transfer, retention and savings.
//
// The project's own definition of intelligence, tested for the first time.
// vision.md pillar 4 and success-criterion 4 have said "teach it a second task,
// return to the first, and it has not forgotten" since they were written, and
// every gate since has measured single-task accuracy instead (L-020).
//
// PROTOCOL. One organism's life, four phases:
//   1. learn A to criterion            → trials T_A
//   2. learn B to criterion            → trials T_B      (B on the SAME outputs)
//   3. test A frozen, no learning      → retention R_A
//   4. relearn A to criterion          → trials T_A2     → savings = T_A − T_A2
//
// CONTROL ARMS, without which none of the above means anything:
//   naive-B   a fresh organism learning B first. Transfer = T_B_naive − T_B.
//             Without it, "B was learned in 140 trials" is a number with no
//             referent.
//   A→A       learn A, then run an equal number of further A trials, then test.
//             This is retention's CEILING — it controls for time passing and
//             for drift under continued reward, so a drop after B can be
//             attributed to B rather than to the clock.
//   naive-A   a fresh organism learning A. The savings baseline: relearning
//             must beat learning-from-scratch, not merely be fast.
//
// Everything is scored in TRIALS-TO-CRITERION (H-011), not accuracy: it stays
// comparable across learners of different speeds and does not punish a slow
// learner for being slow.
//
//   npx vite-node tools/exp003-transfer.ts [seeds...]

import { Organism } from '../src/engine/organism'
import { AutoTeacher } from '../src/engine/teacher'
import { TASK_A_PATTERNS, TASK_B_PATTERNS } from '../src/engine/patterns'
import { defaultConfig, defaultTeacherConfig, type OrganismLike } from '../src/engine/types'
import { mulberry32, shuffleInPlace, type Rng } from '../src/engine/rng'

const SEEDS = process.argv.slice(2).length
  ? process.argv.slice(2).map(Number)
  : [1, 2, 3, 4, 5]

const CRITERION = 0.85
const WINDOW = 100
const MAX_TRIALS = 4000
/** frozen test length for retention */
const PROBE = 150

interface Learner {
  org: OrganismLike
  teacher: AutoTeacher
  order: Rng
  block: number[]
}

function makeLearner(seed: number): Learner {
  return {
    org: new Organism({ ...defaultConfig, seed }),
    teacher: new AutoTeacher({ ...defaultTeacherConfig }),
    order: mulberry32(seed * 7919 + 1),
    block: [],
  }
}

function nextLabel(l: Learner): number {
  if (l.block.length === 0) l.block = shuffleInPlace([0, 1, 2], l.order)
  return l.block.pop()!
}

/**
 * Train until the rolling window holds at criterion, or the cap is hit.
 * Returns trials taken, or null if criterion was never reached — reported as
 * "none" rather than silently as the cap, because those are different facts.
 */
function trainToCriterion(l: Learner, patterns: Uint8Array[]): number | null {
  const res: boolean[] = []
  for (let t = 0; t < MAX_TRIALS; t++) {
    const label = nextLabel(l)
    res.push(l.teacher.runTrial(l.org, patterns[label], label).correct)
    if (res.length >= WINDOW) {
      const w = res.slice(-WINDOW)
      if (w.filter(Boolean).length / WINDOW >= CRITERION) return t + 1
    }
  }
  return null
}

/** Run n trials with no criterion check (used for the A→A time control). */
function trainFor(l: Learner, patterns: Uint8Array[], n: number): void {
  for (let t = 0; t < n; t++) {
    const label = nextLabel(l)
    l.teacher.runTrial(l.org, patterns[label], label)
  }
}

/** Test with learning OFF, so the probe cannot itself teach. */
function testFrozen(l: Learner, patterns: Uint8Array[]): number {
  const was = l.teacher.learning
  l.teacher.learning = false
  let correct = 0
  for (let t = 0; t < PROBE; t++) {
    const label = nextLabel(l)
    if (l.teacher.runTrial(l.org, patterns[label], label).correct) correct++
  }
  l.teacher.learning = was
  return correct / PROBE
}

interface Row {
  seed: number
  tA: number | null
  tB: number | null
  tBnaive: number | null
  tAnaive: number | null
  retentionAfterB: number
  retentionAfterA: number
  tA2: number | null
  bImmediatelyAfterB: number
}

function run(seed: number): Row {
  // ── main arm: A → B → test A → relearn A ────────────────────────────────
  const m = makeLearner(seed)
  const tA = trainToCriterion(m, TASK_A_PATTERNS)
  const tB = trainToCriterion(m, TASK_B_PATTERNS)
  const retentionAfterB = testFrozen(m, TASK_A_PATTERNS)
  const bImmediatelyAfterB = testFrozen(m, TASK_B_PATTERNS)
  const tA2 = trainToCriterion(m, TASK_A_PATTERNS)

  // ── control: A → A for the same number of trials B took ─────────────────
  // retention's ceiling. Any drop below this is attributable to B rather than
  // to elapsed time or to drift under continued reward.
  const c = makeLearner(seed)
  trainToCriterion(c, TASK_A_PATTERNS)
  trainFor(c, TASK_A_PATTERNS, tB ?? MAX_TRIALS)
  const retentionAfterA = testFrozen(c, TASK_A_PATTERNS)

  // ── control: naive B, and naive A ───────────────────────────────────────
  const nb = makeLearner(seed)
  const tBnaive = trainToCriterion(nb, TASK_B_PATTERNS)
  const na = makeLearner(seed)
  const tAnaive = trainToCriterion(na, TASK_A_PATTERNS)

  return {
    seed,
    tA,
    tB,
    tBnaive,
    tAnaive,
    retentionAfterB,
    retentionAfterA,
    tA2,
    bImmediatelyAfterB,
  }
}

const f = (v: number | null) => (v === null ? ' none' : String(v).padStart(5))
const rows = SEEDS.map(run)

console.log(
  `Experiment 003 — transfer, retention, savings\n` +
    `criterion = rolling-${WINDOW} >= ${CRITERION}, cap ${MAX_TRIALS}, frozen probe ${PROBE} trials\n` +
    `task A = vertical / horizontal / diagonal-X · task B = plus / ring / band, SAME three outputs\n`
)

console.log('TRIALS TO CRITERION')
console.log('seed |    A |    B | B naive | A relearn | A naive')
console.log('-----|------|------|---------|-----------|--------')
for (const r of rows) {
  console.log(
    `  ${r.seed}  |${f(r.tA)} |${f(r.tB)} |  ${f(r.tBnaive)}  |   ${f(r.tA2)}   | ${f(r.tAnaive)}`
  )
}

console.log('\nRETENTION (frozen, no learning)')
console.log('seed | A after B | A after A (ceiling) | B after B')
console.log('-----|-----------|---------------------|----------')
for (const r of rows) {
  console.log(
    `  ${r.seed}  |   ${r.retentionAfterB.toFixed(3)}   |        ${r.retentionAfterA.toFixed(
      3
    )}        |   ${r.bImmediatelyAfterB.toFixed(3)}`
  )
}

// ── the three pre-registered gates ────────────────────────────────────────
const ok = <T,>(xs: (T | null)[]) => xs.filter((x): x is T => x !== null)
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN)

const transferPairs = rows.filter((r) => r.tB !== null && r.tBnaive !== null)
const transferDelta = transferPairs.map((r) => r.tBnaive! - r.tB!)
const savingsPairs = rows.filter((r) => r.tA2 !== null && r.tAnaive !== null)
const savingsDelta = savingsPairs.map((r) => r.tAnaive! - r.tA2!)
const retDrop = rows.map((r) => r.retentionAfterA - r.retentionAfterB)

console.log('\n══ pre-registered gates ═══════════════════════════════════════')
console.log(
  `TRANSFER  B after A vs B naive: mean ${mean(transferDelta) >= 0 ? '+' : ''}` +
    `${mean(transferDelta).toFixed(0)} trials ` +
    `(faster on ${transferDelta.filter((d) => d > 0).length}/${transferDelta.length} seeds)`
)
console.log(
  `          gate — faster on >=3 seeds: ` +
    `${transferDelta.filter((d) => d > 0).length >= 3 ? 'PASS' : 'FAIL'}`
)
console.log(
  `RETENTION A after B ${mean(rows.map((r) => r.retentionAfterB)).toFixed(3)} vs ceiling ` +
    `${mean(rows.map((r) => r.retentionAfterA)).toFixed(3)}, drop ${mean(retDrop).toFixed(3)}`
)
console.log(
  `          gate — drop <= 0.10: ${mean(retDrop) <= 0.1 ? 'PASS' : 'FAIL'}` +
    `   (chance is 0.333)`
)
console.log(
  `SAVINGS   relearn A vs naive A: mean ${mean(savingsDelta) >= 0 ? '+' : ''}` +
    `${mean(savingsDelta).toFixed(0)} trials ` +
    `(faster on ${savingsDelta.filter((d) => d > 0).length}/${savingsDelta.length} seeds)`
)
console.log(
  `          gate — faster on >=3 seeds: ` +
    `${savingsDelta.filter((d) => d > 0).length >= 3 ? 'PASS' : 'FAIL'}`
)
console.log(
  `\nnote: ${ok(rows.map((r) => r.tA)).length}/${rows.length} seeds reached criterion on A, ` +
    `${ok(rows.map((r) => r.tB)).length}/${rows.length} on B`
)
