// 002-M1h confirmation stage, per the pre-registered decision rule: the best
// ladder arms re-run on seeds NEVER used in the search (4,5,6), longer
// (8,000 trials). Candidates by worst-seed tail: slow+strong (0.59, one seed
// at 1.00) and sleep80 (0.99/0.99/0.36). A third arm — slow+strong at
// sleep80 — is POST-HOC (not in the registered ladder) and is labelled
// exploratory: its result guides, but cannot confirm, anything.
import { GrownOrganism } from '../src/engine/grown/grown-organism'
import { defaultGrownConfig, type GrownConfig } from '../src/engine/grown/config'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

const [SHARD, NSHARDS] = [Number(process.argv[2] ?? 0), Number(process.argv[3] ?? 1)]
const SEEDS = [4, 5, 6]
const TRIALS = 8000
const BASE: Partial<GrownConfig> = { seedEdges: 12000, seedSpanMax: 44, rentN0: 25 }
const ARMS = [
  { name: 'slow+strong', over: { growthAttempts: 1, sleepEvery: 40, birthWeight: 0.5 } },
  { name: 'sleep80', over: { sleepEvery: 80 } },
  { name: 'EXPLORATORY slow+strong80', over: { growthAttempts: 1, sleepEvery: 80, birthWeight: 0.5 } },
]
function run(over: Partial<GrownConfig>, seed: number) {
  const org = new GrownOrganism({ ...defaultGrownConfig, ...BASE, ...over, seed })
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(seed * 7919 + 1)
  let block: number[] = []
  const res: boolean[] = []
  const curve: number[] = []
  for (let t = 0; t < TRIALS; t++) {
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    const label = block.pop()!
    res.push(teacher.runTrial(org, M1_PATTERNS[label], label).correct)
    if ((t + 1) % 1000 === 0) curve.push(res.slice(-200).filter(Boolean).length / 200)
  }
  return { tail: res.slice(-200).filter(Boolean).length / 200, curve }
}
for (let i = 0; i < ARMS.length; i++) {
  if (i % NSHARDS !== SHARD) continue
  const arm = ARMS[i]
  const rs = SEEDS.map((s) => run(arm.over, s))
  console.log(
    `ARM ${arm.name.padEnd(26)} held-out tails ${rs.map((r) => r.tail.toFixed(2)).join('/')}` +
      ` min ${Math.min(...rs.map((r) => r.tail)).toFixed(2)}`
  )
  for (const [k, r] of rs.entries())
    console.log(`    s${SEEDS[k]} ${r.curve.map((c) => c.toFixed(2)).join(' > ')}`)
}
console.log(`SHARD ${SHARD} DONE`)
