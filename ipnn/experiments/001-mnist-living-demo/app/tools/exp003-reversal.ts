// Experiment 003, corrected — the design flaw, and the test that actually
// creates interference.
//
// THE FLAW. 003's design called task B on the same three outputs "the
// maximum-interference case". It is not interference at all. Task B's glyphs
// are *different stimuli* from task A's, so output 1 simply learns to fire for
// A₁ **and** B₁. Nothing ever tells the organism that A₁ is not output 1. The
// two tasks are compatible, the union mapping exists, and there is no conflict
// to resolve — which is why retention held at 0.872 and why both ablations
// (consolidation off, code separation) came back empty. There was nothing to
// protect against.
//
// THE REAL TEST is to contradict what was learned: keep task A's stimuli and
// **permute the labels**. Now A₁ must map to output 2, and the organism's
// existing answer is not merely unhelpful, it is actively wrong. This is
// reversal learning — bees, fish, rodents and humans all get run through it,
// so it passes the substrate-independence test (H-009), and it is already in
// the comparative battery.
//
// Three arms, all on the same seeds:
//   union      A then B (new glyphs, same outputs) — the original arm, kept
//              for comparison
//   reversal   A then A-with-permuted-labels — genuine contradiction
//   control    A then A again — the ceiling
//
//   npx vite-node tools/exp003-reversal.ts [seeds...]

import { Organism } from '../src/engine/organism'
import { AutoTeacher } from '../src/engine/teacher'
import { TASK_A_PATTERNS, TASK_B_PATTERNS } from '../src/engine/patterns'
import { defaultConfig, defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

const SEEDS = process.argv.slice(2).length
  ? process.argv.slice(2).map(Number)
  : [1, 2, 3, 4, 5]
const CRITERION = 0.85
const WINDOW = 100
const MAX_TRIALS = 4000
const PROBE = 150
/** A₀→out1, A₁→out2, A₂→out0: every stimulus reassigned, none left alone */
const PERMUTATION = [1, 2, 0]

function learner(seed: number) {
  return {
    org: new Organism({ ...defaultConfig, seed }),
    teacher: new AutoTeacher({ ...defaultTeacherConfig }),
    order: mulberry32(seed * 7919 + 1),
    block: [] as number[],
  }
}
type L = ReturnType<typeof learner>
const next = (l: L) => {
  if (l.block.length === 0) l.block = shuffleInPlace([0, 1, 2], l.order)
  return l.block.pop()!
}

/** `map` turns a stimulus index into the label the teacher rewards. */
function toCriterion(
  l: L,
  ps: Uint8Array[],
  map: (i: number) => number = (i) => i
): number | null {
  const res: boolean[] = []
  for (let t = 0; t < MAX_TRIALS; t++) {
    const i = next(l)
    res.push(l.teacher.runTrial(l.org, ps[i], map(i)).correct)
    if (res.length >= WINDOW) {
      const w = res.slice(-WINDOW)
      if (w.filter(Boolean).length / WINDOW >= CRITERION) return t + 1
    }
  }
  return null
}

function frozen(l: L, ps: Uint8Array[], map: (i: number) => number = (i) => i): number {
  const was = l.teacher.learning
  l.teacher.learning = false
  let c = 0
  for (let t = 0; t < PROBE; t++) {
    const i = next(l)
    if (l.teacher.runTrial(l.org, ps[i], map(i)).correct) c++
  }
  l.teacher.learning = was
  return c / PROBE
}

const id = (i: number) => i
const perm = (i: number) => PERMUTATION[i]

interface Row {
  seed: number
  tA: number
  union: { second: number | null; origRetained: number }
  reversal: { second: number | null; origRetained: number }
  control: { origRetained: number }
}

const rows: Row[] = SEEDS.map((seed) => {
  const u = learner(seed)
  const tA = toCriterion(u, TASK_A_PATTERNS)!
  const uSecond = toCriterion(u, TASK_B_PATTERNS)
  const uRet = frozen(u, TASK_A_PATTERNS)

  const r = learner(seed)
  toCriterion(r, TASK_A_PATTERNS)
  const rSecond = toCriterion(r, TASK_A_PATTERNS, perm)
  // "retained" here means: does it still give the ORIGINAL answer?
  const rRet = frozen(r, TASK_A_PATTERNS, id)

  const c = learner(seed)
  toCriterion(c, TASK_A_PATTERNS)
  const cRet = frozen(c, TASK_A_PATTERNS)

  return {
    seed,
    tA,
    union: { second: uSecond, origRetained: uRet },
    reversal: { second: rSecond, origRetained: rRet },
    control: { origRetained: cRet },
  }
})

const mean = (x: number[]) => x.reduce((a, b) => a + b, 0) / x.length
const n = (v: number | null) => (v === null ? ' none' : String(v).padStart(5))

console.log('Experiment 003, corrected — union versus genuine reversal\n')
console.log(
  'UNION arm    = A then new glyphs on the same outputs (compatible: out₁ learns A₁ AND B₁)'
)
console.log(
  `REVERSAL arm = A then A with labels permuted ${JSON.stringify(PERMUTATION)} (contradictory)\n`
)
console.log('seed | trials A | union 2nd | reversal 2nd | A retained: union / reversal / control')
console.log('-----|----------|-----------|--------------|---------------------------------------')
for (const r of rows) {
  console.log(
    `  ${r.seed}  |   ${String(r.tA).padStart(4)}   |   ${n(r.union.second)}   |    ` +
      `${n(r.reversal.second)}     |      ${r.union.origRetained.toFixed(3)} / ` +
      `${r.reversal.origRetained.toFixed(3)} / ${r.control.origRetained.toFixed(3)}`
  )
}

console.log('\n══ what this shows ════════════════════════════════════════════')
const uRet = mean(rows.map((r) => r.union.origRetained))
const rRet = mean(rows.map((r) => r.reversal.origRetained))
const cRet = mean(rows.map((r) => r.control.origRetained))
console.log(`  task A retained after UNION    : ${uRet.toFixed(3)}`)
console.log(`  task A retained after REVERSAL : ${rRet.toFixed(3)}`)
console.log(`  task A retained after MORE A   : ${cRet.toFixed(3)}  (ceiling)`)
console.log(`  chance                         : 0.333`)
console.log()
console.log(
  `  Interference is real only in the reversal arm: ${(uRet - rRet).toFixed(3)} accuracy`
)
console.log(
  `  separates the two second-tasks, which is ${
    uRet - rRet > 0.2 ? 'LARGE' : 'small'
  } against a 0.333 floor.`
)
const relearn = rows.map((r) => r.reversal.second).filter((x): x is number => x !== null)
console.log(
  `\n  reversal took ${
    relearn.length ? Math.round(mean(relearn)) : 'n/a'
  } trials vs ${Math.round(mean(rows.map((r) => r.tA)))} to learn A originally —`
)
console.log(
  '  reversal being FASTER than original learning is the classic signature that the\n' +
    '  stimulus representation survived and only the mapping was relearned.'
)
