// M1b — the etaPool=0 ablation, pre-registered in journal entry
// 2026-08-16-0248 §Next: "is the pool contributing anything yet?"
//
// The M1 organism has two learnable populations:
//   sense→pool   3,840 synapses (160 neurons × 24 fan-in), learning rate etaPool
//   pool→output    480 synapses (3 × 160, dense),          learning rate etaOut
//
// If freezing the sense→pool weights at their random initial values costs
// nothing, then the pool is a fixed random projection and all the learning
// lives in those 480 output weights — i.e. the "brain" is scenery and the
// organism is a policy readout on random features.
//
// Identical procedure to m1-sanity.test.ts (same seeds, same schedule, same
// trial count) so the arms are comparable line for line.
//
//   npx vite-node tools/m1b-ablation.ts

import { Organism } from '../src/engine/organism'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultConfig, defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

const TRIALS = 800
const TAIL = 100
const SEEDS = [1, 2, 3, 4, 5]

function runSeed(seed: number, etaPool: number) {
  const org = new Organism({ ...defaultConfig, seed, etaPool })
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(seed * 7919 + 1)

  const results: boolean[] = []
  const curve: number[] = []
  let block: number[] = []

  for (let t = 0; t < TRIALS; t++) {
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    const label = block.pop()!
    results.push(teacher.runTrial(org, M1_PATTERNS[label], label).correct)
    if ((t + 1) % 100 === 0) {
      const w = results.slice(-100)
      curve.push(w.filter(Boolean).length / w.length)
    }
  }
  const tail = results.slice(-TAIL)
  return {
    tail: tail.filter(Boolean).length / TAIL,
    curve,
    // trials until the rolling-100 window first reaches the gate
    norms: org.weightNorms(),
  }
}

const arms = [
  { name: 'baseline  etaPool=0.01', eta: 0.01 },
  { name: 'ablated   etaPool=0   ', eta: 0 },
]

const summary: Record<string, number[]> = {}

for (const arm of arms) {
  console.log(`\n=== ${arm.name} ===`)
  summary[arm.name] = []
  for (const seed of SEEDS) {
    const r = runSeed(seed, arm.eta)
    summary[arm.name].push(r.tail)
    console.log(
      `seed ${seed}  ${r.curve.map((a) => a.toFixed(2)).join(' → ')}` +
        `   tail ${r.tail.toFixed(2)}   |w|² pool ${r.norms.pool.toFixed(1)} out ${r.norms.out.toFixed(1)}`
    )
  }
}

const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length
console.log('\n=== M1b summary ===')
for (const arm of arms) {
  const v = summary[arm.name]
  console.log(
    `${arm.name}  mean tail ${mean(v).toFixed(3)}  ` +
      `min ${Math.min(...v).toFixed(2)}  max ${Math.max(...v).toFixed(2)}`
  )
}
const delta = mean(summary[arms[0].name]) - mean(summary[arms[1].name])
console.log(
  `\ndifference (baseline − ablated): ${delta >= 0 ? '+' : ''}${delta.toFixed(3)} accuracy`
)
