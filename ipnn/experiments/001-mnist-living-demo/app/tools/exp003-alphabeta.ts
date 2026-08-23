// The α/β fix, against its three pre-registered gates.
//
// L-034: consolidation causes cumulative learning death, because the evidence
// count only ever grows — 7/32 serial reversals learnable with it on, 22/32
// with it off, and the failures terminal. The fix (`evidenceModel: 'beta'`)
// lets contradicting evidence accumulate separately, so plasticity reads the
// *net* evidence |α−β| and a synapse whose world has changed becomes plastic
// again.
//
// ── PRE-REGISTERED GATES, written and committed before this file was run ───
//
//   GATE 1 — reversal survival. Serial reversal (8 flips, seeds 1–4, the
//     L-034 protocol exactly): the beta arm reaches criterion on ≥20 of 32
//     reversals, matching what consolidation-off achieved (22/32). Fail → the
//     fix does not fix.
//
//   GATE 2 — retention holds. The L-005 protocol (500 rewarded trials, then
//     100 with learning off), seeds 42, 1–4: mean frozen accuracy ≥0.90 under
//     beta. Fail → memory and flexibility are ONE DIAL: consolidation cannot
//     exist without rigidity at this scale, L-004's tension is fundamental
//     rather than fixable — and that is the bigger result, not a failure of
//     the experiment.
//
//   GATE 3 — M1 unharmed. The M1 gate (800 trials, seeds 1–3, tail-100
//     ≥0.80) under beta. Fail → the rule change broke basic learning and is
//     reverted.
//
// The count arm runs alongside everywhere, so every number has its control on
// the same seeds in the same output.
//
//   npx vite-node tools/exp003-alphabeta.ts

import { Organism } from '../src/engine/organism'
import { AutoTeacher } from '../src/engine/teacher'
import { TASK_A_PATTERNS } from '../src/engine/patterns'
import { defaultConfig, defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

type Model = 'count' | 'beta'
const CRITERION = 0.85
const WINDOW = 100
const REV_CAP = 2500
const REVERSALS = 8
const RULE_A = [0, 1, 2]
const RULE_B = [1, 2, 0]

function makeRun(seed: number, evidenceModel: Model) {
  const org = new Organism({ ...defaultConfig, seed, evidenceModel })
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(seed * 7919 + 1)
  let block: number[] = []
  const next = () => {
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    return block.pop()!
  }
  return { org, teacher, next }
}
type Run = ReturnType<typeof makeRun>

function toCriterion(r: Run, rule: number[], cap: number): number | null {
  const res: boolean[] = []
  for (let t = 0; t < cap; t++) {
    const i = r.next()
    res.push(r.teacher.runTrial(r.org, TASK_A_PATTERNS[i], rule[i]).correct)
    if (res.length >= WINDOW) {
      const w = res.slice(-WINDOW)
      if (w.filter(Boolean).length / WINDOW >= CRITERION) return t + 1
    }
  }
  return null
}

const fmt = (v: number | null) => (v === null ? ' none' : String(v).padStart(5))
const mean = (x: number[]) => x.reduce((a, b) => a + b, 0) / x.length

// ── Gate 1: serial reversal ────────────────────────────────────────────────
console.log('GATE 1 — serial reversal, 8 flips, alternating [0,1,2] ↔ [1,2,0]\n')
console.log(
  'seed | model | acquire|' +
    Array.from({ length: REVERSALS }, (_, i) => ` rev${i + 1}`.padStart(6)).join('|')
)
console.log('-----|-------|--------|' + '------|'.repeat(REVERSALS))

const reached: Record<Model, number> = { count: 0, beta: 0 }
const betaEvidence: string[] = []
for (const seed of [1, 2, 3, 4]) {
  for (const model of ['count', 'beta'] as Model[]) {
    const r = makeRun(seed, model)
    const stages: (number | null)[] = [toCriterion(r, RULE_A, REV_CAP)]
    for (let k = 1; k <= REVERSALS; k++) {
      stages.push(toCriterion(r, k % 2 === 1 ? RULE_B : RULE_A, REV_CAP))
    }
    reached[model] += stages.slice(1).filter((x) => x !== null).length
    console.log(
      `  ${seed}  | ${model === 'beta' ? 'beta ' : 'count'} | ${fmt(stages[0])}  |` +
        stages.slice(1).map(fmt).join('|')
    )
    if (model === 'beta' && seed === 1) {
      const ev = (r.org as Organism).evidenceTotals()
      betaEvidence.push(
        `  seed 1 beta arm, end of life: out α ${ev.outAlpha.toFixed(0)}, ` +
          `out β ${ev.outBeta.toFixed(0)} — |α−β| ${(ev.outAlpha - ev.outBeta).toFixed(0)}`
      )
    }
  }
}
const gate1 = reached.beta >= 20
console.log(
  `\n  reversals reaching criterion (of 32): count ${reached.count} · beta ${reached.beta}` +
    `   [historical: count 7, no-consolidation 22]`
)
for (const line of betaEvidence) console.log(line)
console.log(`  GATE 1 (beta ≥ 20/32): ${gate1 ? 'PASS' : 'FAIL'}\n`)

// ── Gate 2: frozen retention ───────────────────────────────────────────────
console.log('GATE 2 — the living-model claim: 500 rewarded trials, then 100 frozen\n')
console.log('seed | count | beta')
console.log('-----|-------|------')
const frozenAcc: Record<Model, number[]> = { count: [], beta: [] }
for (const seed of [42, 1, 2, 3, 4]) {
  const row: Record<Model, number> = { count: 0, beta: 0 }
  for (const model of ['count', 'beta'] as Model[]) {
    const r = makeRun(seed, model)
    for (let t = 0; t < 500; t++) {
      const i = r.next()
      r.teacher.runTrial(r.org, TASK_A_PATTERNS[i], i)
    }
    r.teacher.learning = false
    let correct = 0
    for (let t = 0; t < 100; t++) {
      const i = r.next()
      if (r.teacher.runTrial(r.org, TASK_A_PATTERNS[i], i).correct) correct++
    }
    row[model] = correct / 100
    frozenAcc[model].push(correct / 100)
  }
  console.log(` ${String(seed).padStart(3)} | ${row.count.toFixed(2)}  | ${row.beta.toFixed(2)}`)
}
const gate2 = mean(frozenAcc.beta) >= 0.9
console.log(
  `\n  mean frozen accuracy: count ${mean(frozenAcc.count).toFixed(3)} · ` +
    `beta ${mean(frozenAcc.beta).toFixed(3)}   [L-005 recorded 0.97]`
)
console.log(`  GATE 2 (beta mean ≥ 0.90): ${gate2 ? 'PASS' : 'FAIL — memory and flexibility are one dial'}\n`)

// ── Gate 3: the M1 gate under beta ─────────────────────────────────────────
console.log('GATE 3 — the M1 gate (800 trials, tail-100 ≥ 0.80) under beta\n')
let gate3 = true
for (const seed of [1, 2, 3]) {
  const r = makeRun(seed, 'beta')
  const res: boolean[] = []
  const curve: number[] = []
  for (let t = 0; t < 800; t++) {
    const i = r.next()
    res.push(r.teacher.runTrial(r.org, TASK_A_PATTERNS[i], i).correct)
    if ((t + 1) % 100 === 0) curve.push(res.slice(-100).filter(Boolean).length / 100)
  }
  const tail = res.slice(-100).filter(Boolean).length / 100
  if (tail < 0.8) gate3 = false
  console.log(`  seed ${seed}: ${curve.map((a) => a.toFixed(2)).join(' → ')}   tail ${tail.toFixed(2)}`)
}
console.log(`\n  GATE 3: ${gate3 ? 'PASS' : 'FAIL — revert the rule change'}`)

console.log('\n══ verdict ════════════════════════════════════════════════════')
console.log(`  gate 1 reversal survival : ${gate1 ? 'PASS' : 'FAIL'}`)
console.log(`  gate 2 retention holds   : ${gate2 ? 'PASS' : 'FAIL'}`)
console.log(`  gate 3 M1 unharmed       : ${gate3 ? 'PASS' : 'FAIL'}`)
