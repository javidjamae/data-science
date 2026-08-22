// Why does the shallow arm stall at 0.85-0.90 instead of reaching 001's 0.98?
//
// Two very different diagnoses produce the same accuracy number:
//   (a) it is getting answers WRONG  → a learning/capacity problem
//   (b) it is not answering at all   → a readout/commitment problem
// A silent trial scores as incorrect (inherited from 001), so a 12% silence
// rate caps accuracy at 0.88 no matter how good the discrimination is.
//
// This splits them, and also measures how much of the wobble is the substrate
// still being rebuilt underneath a competent readout — every sleep replaces
// thousands of edges, and rent is charged at a flat rate that consolidation
// never slows down.
//
//   npx vite-node tools/exp002-plateau.ts [trials] [seeds...]

import { GrownOrganism } from '../src/engine/grown/grown-organism'
import { defaultGrownConfig } from '../src/engine/grown/config'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

const argv = process.argv.slice(2)
const TRIALS = argv.length > 0 ? Number(argv[0]) : 2000
const SEEDS = argv.length > 1 ? argv.slice(1).map(Number) : [1, 2, 3]
const TAIL = 400

function run(seed: number) {
  const org = new GrownOrganism({ ...defaultGrownConfig, outputX: 14, seed })
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(seed * 7919 + 1)

  const rows: { correct: boolean; spoken: number | null; latency: number }[] = []
  const edgeTrace: number[] = []
  let block: number[] = []

  for (let t = 0; t < TRIALS; t++) {
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    const label = block.pop()!
    const r = teacher.runTrial(org, M1_PATTERNS[label], label)
    rows.push({ correct: r.correct, spoken: r.spoken, latency: r.latency })
    if ((t + 1) % 100 === 0) edgeTrace.push(org.edges.count)
  }

  const tail = rows.slice(-TAIL)
  const spoke = tail.filter((r) => r.spoken !== null)
  const silent = tail.length - spoke.length
  const rightWhenSpoke = spoke.filter((r) => r.correct).length

  // how much does the live edge population move between sleeps, once it is
  // already competent? churn under a working readout is the wobble's suspect
  const lateEdges = edgeTrace.slice(-10)
  const meanEdges = lateEdges.reduce((a, b) => a + b, 0) / lateEdges.length
  const swing = (Math.max(...lateEdges) - Math.min(...lateEdges)) / meanEdges

  return {
    seed,
    tailAccuracy: tail.filter((r) => r.correct).length / tail.length,
    silenceRate: silent / tail.length,
    conditionalAccuracy: spoke.length ? rightWhenSpoke / spoke.length : NaN,
    meanLatency: spoke.reduce((a, r) => a + r.latency, 0) / Math.max(1, spoke.length),
    edgeSwing: swing,
    stats: org.stats(),
  }
}

console.log(`shallow arm — ${TRIALS} trials, tail = last ${TAIL}\n`)
console.log(
  'seed |  tail  | silent | right when it spoke | mean latency | edge swing'
)
console.log('-----|--------|--------|---------------------|--------------|-----------')

const all = SEEDS.map(run)
for (const r of all) {
  console.log(
    `  ${r.seed}  | ${r.tailAccuracy.toFixed(3)}  | ${(r.silenceRate * 100)
      .toFixed(1)
      .padStart(5)}% |        ${r.conditionalAccuracy.toFixed(3)}        |     ${r.meanLatency
      .toFixed(1)
      .padStart(5)}    |   ${(r.edgeSwing * 100).toFixed(0)}%`
  )
}

const mean = (f: (r: (typeof all)[0]) => number) =>
  all.reduce((a, r) => a + f(r), 0) / all.length

console.log()
console.log(`mean tail accuracy        : ${mean((r) => r.tailAccuracy).toFixed(3)}`)
console.log(`mean silence rate         : ${(mean((r) => r.silenceRate) * 100).toFixed(1)}%`)
console.log(`mean accuracy when spoken : ${mean((r) => r.conditionalAccuracy).toFixed(3)}`)
console.log(
  `ceiling implied by silence: ${(1 - mean((r) => r.silenceRate)).toFixed(3)}  ` +
    `(what tail accuracy would be if every spoken answer were right)`
)
console.log(
  `edge population swing     : ${(mean((r) => r.edgeSwing) * 100).toFixed(0)}% of mean, ` +
    `late in the run`
)
