// 002-M1i — the scaffold ablation of the Slow champion. [J: "take off the
// random generation and just do everything else Slow does — will it
// create/find pathways on its own that reach the output?"]
//
// PRE-REGISTERED PREDICTION [C]: no. Earned durability protects earners and
// slow cadence quiets churn, but neither changes ROUTING: growth reaches
// rMax=8 on a 20-unit gap, guided by a short-range activity scent, and
// nothing rewards the intermediate legs of a route that does not yet carry
// signal (L-013). Expect within-2-hops = 0 throughout and accuracy at
// chance — which would establish, by single-variable ablation, that the
// scaffold is the load-bearing birth ingredient of the Slow stack.
// If it DOES learn, the scaffold is dispensable and L-041 was misread.
import { GrownOrganism } from '../src/engine/grown/grown-organism'
import { defaultGrownConfig, type GrownConfig } from '../src/engine/grown/config'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

const SEEDS = [1, 2, 5]
const TRIALS = 6000
const ARMS: { name: string; over: Partial<GrownConfig> }[] = [
  { name: 'slow, NO scaffold', over: { rentN0: 25, sleepEvery: 80 } },
  { name: 'slow+strong, NO scaffold', over: { rentN0: 25, sleepEvery: 40, growthAttempts: 1, birthWeight: 0.5 } },
  { name: 'slow WITH scaffold (ref)', over: { rentN0: 25, sleepEvery: 80, seedEdges: 12000, seedSpanMax: 44 } },
]
function run(over: Partial<GrownConfig>, seed: number) {
  const org = new GrownOrganism({ ...defaultGrownConfig, ...over, seed })
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(seed * 7919 + 1)
  let block: number[] = []
  const res: boolean[] = []
  const curve: number[] = []
  for (let t = 0; t < TRIALS; t++) {
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    const label = block.pop()!
    res.push(teacher.runTrial(org, M1_PATTERNS[label], label).correct)
    if ((t + 1) % 1500 === 0) curve.push(res.slice(-200).filter(Boolean).length / 200)
  }
  const h = org.stats().inputHops
  return {
    tail: res.slice(-200).filter(Boolean).length / 200,
    curve,
    w2: (h[1] ?? 0) + (h[2] ?? 0),
    edges: org.stats().edges,
  }
}
for (const arm of ARMS) {
  const rs = SEEDS.map((s) => run(arm.over, s))
  console.log(
    `ARM ${arm.name.padEnd(26)} tails ${rs.map((r) => r.tail.toFixed(2)).join('/')}` +
      `  w2end ${rs.map((r) => r.w2).join('/')}  edges ${rs.map((r) => r.edges).join('/')}`
  )
  console.log(`    curves ${rs.map((r) => r.curve.map((c) => c.toFixed(2)).join('>')).join(' | ')}`)
}
