// 002-M1h — Javid's knob hypotheses on the WORKING configuration.
//
//   "I'm curious if growing slower but stronger connections would help.
//    Are we pruning things too fast? Even if it takes longer to run, but it
//    converges on a great outcome, that's fine." [J, 2026-08-23]
//
// Unlike the L-040 ladder (which turned knobs on a dead substrate), this one
// tunes a LIVE one: the Durable configuration (long-12000 scaffold +
// rentN0 25), whose curves were still rising at 2,500 trials (L-043) and at
// 0.65 by 5,285 in the demo. Runs are 6,000 trials — longer, per [J].
//
// ── PRE-REGISTERED, committed before first run ────────────────────────────
// GROUPS:
//   SLOW    grow less often / fewer attempts (growthAttempts 1; sleepEvery
//           40/80) — [J]: less churn under the learners
//   STRONG  born stronger (birthWeight .3/.5; wMax 6) — [J]: newborn routes
//           audible immediately
//   PRUNE   slower pruning (rent/2; death .01) — [J]: longer auditions
//   DURA    tune the new durability constant (rentN0 10/50/100)
//   COMBO   slow+strong, slow+strong+prune-slow, denser birth (18k)
// PREDICTION [C], honest: sleepEvery↑ and birthWeight↑ most likely to help
// (churn under the core is the visible noise source; louder newborns win
// auditions); prune-slower risks L-027/L-044-style congestion; rentN0 has an
// optimum (too low = everything durable = congestion, too high = L-042
// returns). Gate: any arm ≥0.80 tail on all 3 seeds = 002's M1 GATE, first
// pass ever on the hard geometry. DECISION RULE: best arm → held-out confirm
// on fresh seeds (4,5,6) at 8,000 trials before anything is claimed (L-031).
//
//   npx vite-node tools/exp002-durable-ladder.ts <shard> <nshards>

import { GrownOrganism } from '../src/engine/grown/grown-organism'
import { defaultGrownConfig, type GrownConfig } from '../src/engine/grown/config'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

const [SHARD, NSHARDS] = [Number(process.argv[2] ?? 0), Number(process.argv[3] ?? 1)]
const SEEDS = [1, 2, 3]
const TRIALS = 6000
const BASE: Partial<GrownConfig> = { seedEdges: 12000, seedSpanMax: 44, rentN0: 25 }

const ARMS: { name: string; over: Partial<GrownConfig> }[] = [
  { name: 'durable-base', over: {} },
  // SLOW — grow less, rewire less often
  { name: 'att1', over: { growthAttempts: 1 } },
  { name: 'sleep40', over: { sleepEvery: 40 } },
  { name: 'sleep80', over: { sleepEvery: 80 } },
  // STRONG — born louder
  { name: 'bw.3', over: { birthWeight: 0.3 } },
  { name: 'bw.5', over: { birthWeight: 0.5 } },
  { name: 'bw.5+wmax6', over: { birthWeight: 0.5, wMax: 6 } },
  // PRUNE — slower auditions
  { name: 'rent/2', over: { rent: 0.000045 } },
  { name: 'death.01', over: { deathThreshold: 0.01 } },
  { name: 'rent/2+death.01', over: { rent: 0.000045, deathThreshold: 0.01 } },
  // DURA — tune durability itself
  { name: 'rentN0=10', over: { rentN0: 10 } },
  { name: 'rentN0=50', over: { rentN0: 50 } },
  { name: 'rentN0=100', over: { rentN0: 100 } },
  // COMBO
  { name: 'slow+strong', over: { growthAttempts: 1, sleepEvery: 40, birthWeight: 0.5 } },
  { name: 'slow+strong+prune', over: { growthAttempts: 1, sleepEvery: 40, birthWeight: 0.5, rent: 0.000045 } },
  { name: 'seed18k', over: { seedEdges: 18000 } },
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
  const tail = res.slice(-200).filter(Boolean).length / 200
  const h = org.stats().inputHops
  return { tail, curve, w2end: (h[1] ?? 0) + (h[2] ?? 0), edges: org.stats().edges }
}

for (let i = 0; i < ARMS.length; i++) {
  if (i % NSHARDS !== SHARD) continue
  const arm = ARMS[i]
  const rs = SEEDS.map((s) => run(arm.over, s))
  const min = Math.min(...rs.map((r) => r.tail))
  console.log(
    `ARM ${arm.name.padEnd(18)} tails ${rs.map((r) => r.tail.toFixed(2)).join('/')}` +
      ` min ${min.toFixed(2)}  w2end ${rs.map((r) => r.w2end).join('/')}` +
      `  edges ${rs.map((r) => r.edges).join('/')}`
  )
  for (const [k, r] of rs.entries()) {
    console.log(`    s${SEEDS[k]} ${r.curve.map((c) => c.toFixed(2)).join(' > ')}`)
  }
}
console.log(`SHARD ${SHARD} DONE`)
