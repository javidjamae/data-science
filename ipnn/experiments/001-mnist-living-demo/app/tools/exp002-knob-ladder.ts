// 002 knob ladder — fifty deliberate variations, aimed at the diagnosed
// failures, on the REAL M1 geometry. [J: "I want to play around with the 002
// knobs — like 50 variations."]
//
// This is not the random sweep. That ran earlier (327 draws of 24 knobs,
// geometry pinned): best worst-seed tail 0.38, nothing ≥0.5. Random draws
// mostly land in incoherent corners; these arms are hypothesis-grouped, each
// group attacking one diagnosed failure:
//
//   SNR      L-013's mechanism: one arriving spike moves a node's firing
//            probability by ~0.04 (gain·birthWeight ≈ 0.3 of sigmoid drive).
//            Raise birthWeight / gain / wMax so a spike is *audible*.
//   DENSITY  more wiring per node (maxOutDegree, growthAttempts, rMax) so
//            drive sums over more correlated inputs.
//   ECONOMY  slower rent, lower death threshold, longer wake between rewires
//            — let structure live long enough to earn.
//   EXCITE   sparsity, bias, spontaneous rate — where the units sit on the
//            sigmoid.
//   FIELDS   growth reach and activity-field memory.
//   RULE     eligibility horizon and learning rate.
//   COMBO    the promising directions together.
//
// ── PRE-REGISTERED, before first run ──────────────────────────────────────
// PREDICTION [C]: no arm reaches the M1 gate (min-seed ≥0.8). The random
// sweep is prior evidence, and L-013's per-hop information collapse is a
// structural property that knobs shift only quantitatively. Expected: the
// SNR and COMBO groups move tail accuracy meaningfully off the 0.19 floor;
// nothing clears 0.5 on its worst seed.
// DECISION RULE: best arm min-seed ≥0.5 → promote to 3 seeds × 2,000 trials
// and refine around it. All arms <0.5 → knobs alone are insufficient at this
// geometry, and the next build is the diagnosed MECHANISMS (M1e output
// beacon, H-006 correlation-seeking growth, H-012 scaffolding, α/β port).
//
//   npx vite-node tools/exp002-knob-ladder.ts <shard> <nshards>
//
// Screen: 2 seeds × 1,000 trials per arm, tail-100. Reported per arm: tail
// per seed, sense pixels within 2 hops, edges reaching an answer neuron,
// connectivity. Baseline arm included for reference.

import { GrownOrganism } from '../src/engine/grown/grown-organism'
import { defaultGrownConfig, type GrownConfig } from '../src/engine/grown/config'
import { ROLE_OUTPUT } from '../src/engine/grown/lattice'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

const [SHARD, NSHARDS] = [Number(process.argv[2] ?? 0), Number(process.argv[3] ?? 1)]
const SEEDS = [1, 2]
const TRIALS = 1000
const TAIL = 100

type Over = Partial<GrownConfig>
const ARMS: { name: string; over: Over }[] = [
  { name: 'baseline', over: {} },
  // ── SNR: make one spike audible ──
  { name: 'bw.3', over: { birthWeight: 0.3 } },
  { name: 'bw.5', over: { birthWeight: 0.5 } },
  { name: 'bw.8', over: { birthWeight: 0.8 } },
  { name: 'gain3', over: { gain: 3 } },
  { name: 'gain4', over: { gain: 4 } },
  { name: 'bw.5+gain3', over: { birthWeight: 0.5, gain: 3 } },
  { name: 'bw.8+gain3', over: { birthWeight: 0.8, gain: 3 } },
  { name: 'bw.5+wmax6', over: { birthWeight: 0.5, wMax: 6 } },
  // ── DENSITY ──
  { name: 'deg64', over: { maxOutDegree: 64 } },
  { name: 'deg128', over: { maxOutDegree: 128 } },
  { name: 'att4', over: { growthAttempts: 4 } },
  { name: 'att8', over: { growthAttempts: 8 } },
  { name: 'deg64+att4', over: { maxOutDegree: 64, growthAttempts: 4 } },
  { name: 'rmax11', over: { rMax: 11 } },
  { name: 'rmax11+deg64', over: { rMax: 11, maxOutDegree: 64 } },
  { name: 'rmax11+att4', over: { rMax: 11, growthAttempts: 4 } },
  // ── ECONOMY ──
  { name: 'rent/2', over: { rent: 0.000045 } },
  { name: 'rent*2', over: { rent: 0.00018 } },
  { name: 'death.01', over: { deathThreshold: 0.01 } },
  { name: 'sleep40', over: { sleepEvery: 40 } },
  { name: 'sleep80', over: { sleepEvery: 80 } },
  { name: 'sleep40+rent/2', over: { sleepEvery: 40, rent: 0.000045 } },
  // ── EXCITE ──
  { name: 'sparse.08', over: { targetSparsity: 0.08 } },
  { name: 'sparse.25', over: { targetSparsity: 0.25 } },
  { name: 'bias-.5', over: { bias: -0.5 } },
  { name: 'bias-2', over: { bias: -2 } },
  { name: 'spont.05', over: { pSpont: 0.05 } },
  { name: 'spont.005', over: { pSpont: 0.005 } },
  // ── FIELDS ──
  { name: 'lamg8', over: { lambdaG: 8 } },
  { name: 'lamg2', over: { lambdaG: 2 } },
  { name: 'actfast', over: { activityDecay: 0.02 } },
  { name: 'actslow', over: { activityDecay: 0.001 } },
  // ── RULE ──
  { name: 'trace.99', over: { traceDecay: 0.99 } },
  { name: 'eta.2', over: { eta: 0.2 } },
  { name: 'eta.02', over: { eta: 0.02 } },
  { name: 'trace.99+eta.2', over: { traceDecay: 0.99, eta: 0.2 } },
  // ── singles that pair the above across groups ──
  { name: 'bw.5+deg64', over: { birthWeight: 0.5, maxOutDegree: 64 } },
  { name: 'gain3+sleep40', over: { gain: 3, sleepEvery: 40 } },
  { name: 'eta.2+bw.5', over: { eta: 0.2, birthWeight: 0.5 } },
  { name: 'sparse.08+lamg8', over: { targetSparsity: 0.08, lambdaG: 8 } },
  { name: 'actslow+sleep40', over: { activityDecay: 0.001, sleepEvery: 40 } },
  // ── COMBOS ──
  { name: 'hiSNR', over: { birthWeight: 0.5, gain: 3, maxOutDegree: 64, growthAttempts: 4 } },
  { name: 'hiSNR+sleep40', over: { birthWeight: 0.5, gain: 3, maxOutDegree: 64, growthAttempts: 4, sleepEvery: 40 } },
  { name: 'hiSNR+lamg8', over: { birthWeight: 0.5, gain: 3, maxOutDegree: 64, growthAttempts: 4, lambdaG: 8 } },
  { name: 'dense+rich', over: { maxOutDegree: 128, growthAttempts: 8, birthWeight: 0.3 } },
  { name: 'trace99+hiSNR', over: { traceDecay: 0.99, birthWeight: 0.5, gain: 3 } },
  { name: 'sparseHiSNR', over: { targetSparsity: 0.08, birthWeight: 0.8, gain: 3 } },
  { name: 'dense+slow', over: { maxOutDegree: 64, growthAttempts: 4, sleepEvery: 80, rent: 0.000045 } },
  { name: 'kitchen', over: { birthWeight: 0.5, gain: 3, maxOutDegree: 64, growthAttempts: 4, sleepEvery: 40, traceDecay: 0.99, lambdaG: 8 } },
]

function run(over: Over, seed: number) {
  const org = new GrownOrganism({ ...defaultGrownConfig, ...over, seed })
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(seed * 7919 + 1)
  let block: number[] = []
  const res: boolean[] = []
  for (let t = 0; t < TRIALS; t++) {
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    const label = block.pop()!
    res.push(teacher.runTrial(org, M1_PATTERNS[label], label).correct)
  }
  const tail = res.slice(-TAIL).filter(Boolean).length / TAIL
  const stats = org.stats()
  const h = stats.inputHops
  let toOut = 0
  for (let s = 0; s < org.edges.count; s++) {
    if (org.lattice.role[org.edges.post[s]] === ROLE_OUTPUT) toOut++
  }
  return {
    tail,
    within2: (h[1] ?? 0) + (h[2] ?? 0),
    toOut,
    edges: stats.edges,
    connected: stats.connected.every(Boolean),
  }
}

for (let i = 0; i < ARMS.length; i++) {
  if (i % NSHARDS !== SHARD) continue
  const arm = ARMS[i]
  const t0 = Date.now()
  const rs = SEEDS.map((s) => run(arm.over, s))
  const min = Math.min(...rs.map((r) => r.tail))
  console.log(
    `ARM ${arm.name.padEnd(16)} tails ${rs.map((r) => r.tail.toFixed(2)).join('/')}` +
      ` min ${min.toFixed(2)}  within2 ${rs.map((r) => r.within2).join('/')}` +
      ` toOut ${rs.map((r) => r.toOut).join('/')}  edges ${rs[0].edges}` +
      ` conn ${rs.every((r) => r.connected) ? 'y' : 'N'}  ${((Date.now() - t0) / 1000).toFixed(0)}s`
  )
}
console.log(`SHARD ${SHARD} DONE`)
