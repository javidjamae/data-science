// Experiment 003 follow-up — WHY did retention hold?
//
// The headline result contradicted the pre-registered prediction: 001 kept task
// A at 0.872 after learning an interfering task B on the same three outputs,
// against a 0.948 ceiling and a 0.333 chance line. It did NOT catastrophically
// forget. Something is protecting it, and claiming a mechanism without an
// ablation is exactly what this project keeps catching itself doing.
//
// Two candidate mechanisms, both testable, both tested here.
//
// 1. CONSOLIDATION. The learning rule is Δw = η·R·e/(1 + n/n₀), where `n` is an
//    evidence count that only ever increases. Weights that became well-evidenced
//    during A are *resistant to change* during B by construction. This is the
//    project's Beta-confidence consolidation doing precisely what it was
//    designed for — and it is close kin to Synaptic Intelligence (Zenke et al.
//    2017), which the 2026-08-21 audit already flagged as under-cited.
//    Ablation: `consolidation: false`. If retention collapses, this is it.
//
// 2. CODE SEPARATION. The pool is a sparse random projection. If task A and
//    task B activate largely *different* pool neurons, they write to largely
//    different readout weights and never collide — retention would then be a
//    property of the representation, not of the learning rule, and nothing is
//    being "protected" at all.
//    Measurement: overlap of the active pool population between A and B glyphs.
//
// These are not exclusive and the answer may be both.
//
//   npx vite-node tools/exp003-why.ts [seeds...]

import { Organism } from '../src/engine/organism'
import { AutoTeacher } from '../src/engine/teacher'
import { TASK_A_PATTERNS, TASK_B_PATTERNS } from '../src/engine/patterns'
import { defaultConfig, defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace, type Rng } from '../src/engine/rng'

const SEEDS = process.argv.slice(2).length
  ? process.argv.slice(2).map(Number)
  : [1, 2, 3, 4, 5]
const CRITERION = 0.85
const WINDOW = 100
const MAX_TRIALS = 4000
const PROBE = 150

function learner(seed: number, consolidation: boolean) {
  return {
    org: new Organism({ ...defaultConfig, seed, consolidation }),
    teacher: new AutoTeacher({ ...defaultTeacherConfig }),
    order: mulberry32(seed * 7919 + 1),
    block: [] as number[],
  }
}
type L = ReturnType<typeof learner>

function next(l: L): number {
  if (l.block.length === 0) l.block = shuffleInPlace([0, 1, 2], l.order)
  return l.block.pop()!
}

function toCriterion(l: L, ps: Uint8Array[]): number | null {
  const res: boolean[] = []
  for (let t = 0; t < MAX_TRIALS; t++) {
    const label = next(l)
    res.push(l.teacher.runTrial(l.org, ps[label], label).correct)
    if (res.length >= WINDOW) {
      const w = res.slice(-WINDOW)
      if (w.filter(Boolean).length / WINDOW >= CRITERION) return t + 1
    }
  }
  return null
}

function frozen(l: L, ps: Uint8Array[]): number {
  const was = l.teacher.learning
  l.teacher.learning = false
  let c = 0
  for (let t = 0; t < PROBE; t++) {
    const label = next(l)
    if (l.teacher.runTrial(l.org, ps[label], label).correct) c++
  }
  l.teacher.learning = was
  return c / PROBE
}

// ── 1. the consolidation ablation ─────────────────────────────────────────
console.log('1. IS IT CONSOLIDATION?')
console.log('   Δw = η·R·e/(1 + n/n₀). With consolidation off, every weight stays')
console.log('   fully plastic forever and task B can overwrite task A freely.\n')
console.log('   seed | consolidation |  A→B retention | B retention | trials A | trials B')
console.log('   -----|---------------|----------------|-------------|----------|---------')

const results: Record<string, number[]> = { on: [], off: [] }
for (const seed of SEEDS) {
  for (const on of [true, false]) {
    const l = learner(seed, on)
    const tA = toCriterion(l, TASK_A_PATTERNS)
    const tB = toCriterion(l, TASK_B_PATTERNS)
    const rA = frozen(l, TASK_A_PATTERNS)
    const rB = frozen(l, TASK_B_PATTERNS)
    results[on ? 'on' : 'off'].push(rA)
    console.log(
      `     ${seed}  |      ${on ? 'ON ' : 'OFF'}      |     ${rA.toFixed(3)}      |    ` +
        `${rB.toFixed(3)}    |   ${String(tA ?? 0).padStart(4)}   |   ${String(tB ?? 0).padStart(4)}`
    )
  }
}
const mean = (x: number[]) => x.reduce((a, b) => a + b, 0) / x.length
console.log(
  `\n   mean A-retention: consolidation ON ${mean(results.on).toFixed(3)}  ` +
    `OFF ${mean(results.off).toFixed(3)}  (chance 0.333)`
)
console.log(
  `   → ${
    mean(results.on) - mean(results.off) > 0.1
      ? 'CONSOLIDATION IS CARRYING IT'
      : 'consolidation is NOT the explanation — retention survives without it'
  }`
)

// ── 2. code separation ────────────────────────────────────────────────────
// If A and B light up different pool neurons, they write to different readout
// weights and never collide. Retention would then be a property of the random
// projection rather than anything the learning rule does.
console.log('\n2. OR IS IT CODE SEPARATION?')
console.log('   Overlap of the active pool population between task A and task B glyphs.')
console.log('   A sparse random projection can separate two tasks for free.\n')

function activeSet(seed: number, pattern: Uint8Array, ticks = 400): Float32Array {
  const org = new Organism({ ...defaultConfig, seed })
  const rate = new Float32Array(org.cfg.poolSize)
  org.sense.set(pattern)
  for (let t = 0; t < 60; t++) org.tick()
  for (let t = 0; t < ticks; t++) {
    org.tick()
    for (let j = 0; j < rate.length; j++) rate[j] += org.poolFired[j]
  }
  for (let j = 0; j < rate.length; j++) rate[j] /= ticks
  return rate
}

/** overlap of the top-k most active units, as a fraction of k */
function topKOverlap(a: Float32Array, b: Float32Array, k: number): number {
  const top = (r: Float32Array) =>
    new Set(
      Array.from(r.keys())
        .sort((x, y) => r[y] - r[x])
        .slice(0, k)
    )
  const ta = top(a)
  const tb = top(b)
  let n = 0
  for (const x of ta) if (tb.has(x)) n++
  return n / k
}

const K = 24 // ≈ targetPoolSparsity × poolSize
for (const seed of SEEDS.slice(0, 3)) {
  const A = TASK_A_PATTERNS.map((p) => activeSet(seed, p))
  const B = TASK_B_PATTERNS.map((p) => activeSet(seed, p))
  const within: number[] = []
  for (let i = 0; i < 3; i++)
    for (let j = i + 1; j < 3; j++) within.push(topKOverlap(A[i], A[j], K))
  const across: number[] = []
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) across.push(topKOverlap(A[i], B[j], K))
  console.log(
    `   seed ${seed}: top-${K} overlap within task A ${mean(within).toFixed(2)}  ` +
      `A vs B ${mean(across).toFixed(2)}`
  )
}
console.log(
  '\n   Reading: if A-vs-B overlap is well below within-A overlap, the two tasks\n' +
    '   occupy different parts of the pool and never had to compete for the same\n' +
    '   weights — retention would be free, not earned.'
)
