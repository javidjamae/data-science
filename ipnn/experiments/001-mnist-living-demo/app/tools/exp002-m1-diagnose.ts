// Experiment 002 M1 — where does the chain break?
//
// M1 failed flat at the pre-registered configuration: ~0.19 tail accuracy on
// three seeds, which is chance once the ~42% silent trials are accounted for
// (0.19 / 0.58 = 0.33). The structure is NOT the obvious culprit — it grows,
// it connects input to output in 4-6 hops, and reward arrives within the
// first 200 ticks. So something between "reward arrives" and "weights encode
// the task" is not working.
//
// This walks the causal chain one link at a time and reports where the signal
// dies. Everything here is post-hoc diagnosis, not a pre-registered measure,
// and is labelled as such in the journal.
//
//   1. structure — who is wired to whom, and how many edges reach the outputs
//   2. information — does interior firing depend on the pattern at all, and
//      how far from the input cortex does that dependence survive
//   3. output — do the three answer neurons fire differently per pattern
//   4. forces — does reward move a weight more than rent does, or less
//
//   npx vite-node tools/exp002-m1-diagnose.ts [trials] [seed]

import { GrownOrganism } from '../src/engine/grown/grown-organism'
import { defaultGrownConfig } from '../src/engine/grown/config'
import { ROLE_INPUT, ROLE_OUTPUT, ROLE_INTERIOR } from '../src/engine/grown/lattice'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

const argv = process.argv.slice(2)
const TRIALS = argv.length > 0 ? Number(argv[0]) : 1000
const SEED = argv.length > 1 ? Number(argv[1]) : 1

const cfg = { ...defaultGrownConfig, seed: SEED }
// The organism gets its own copy. `GrownOrganism.cfg` is readonly by reference
// only, so handing it this object would alias it — and the probe below sets
// rent to zero on the organism, which would silently zero the rent this tool
// is trying to measure against.
const org = new GrownOrganism({ ...cfg })
const teacher = new AutoTeacher({ ...defaultTeacherConfig })
const order = mulberry32(SEED * 7919 + 1)

// ── train, while measuring the two forces acting on a weight ────────────────
// rent is deterministic (rent × ticks); reward's contribution is what we want
// to compare it against, so accumulate |Δw| from applyReward directly.
let rewardWork = 0
let rewardEvents = 0
const applyReward = org.applyReward.bind(org)
;(org as unknown as { applyReward: (r: number) => void }).applyReward = (r: number) => {
  const before = Float32Array.from(org.edges.w)
  applyReward(r)
  let d = 0
  for (let s = 0; s < org.edges.count; s++) d += Math.abs(org.edges.w[s] - before[s])
  if (org.edges.count > 0) {
    rewardWork += d / org.edges.count
    rewardEvents++
  }
}

let block: number[] = []
const results: boolean[] = []
for (let t = 0; t < TRIALS; t++) {
  if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
  const label = block.pop()!
  results.push(teacher.runTrial(org, M1_PATTERNS[label], label).correct)
}
const tailWindow = results.slice(-100)
const tail = tailWindow.filter(Boolean).length / tailWindow.length
// captured here, before the probe adds thousands of ticks that belong to no
// trial — section 4's rent baseline is per-trial and would be inflated by them
const trainingTicks = org.stats().ticks

// ── 1. structure ────────────────────────────────────────────────────────────
const lat = org.lattice
const roleName = (r: number) =>
  r === ROLE_INPUT ? 'input' : r === ROLE_OUTPUT ? 'output' : 'interior'
const byPair = new Map<string, number>()
const inDegree = new Int32Array(lat.size)
for (let s = 0; s < org.edges.count; s++) {
  const key = `${roleName(lat.role[org.edges.pre[s]])} → ${roleName(lat.role[org.edges.post[s]])}`
  byPair.set(key, (byPair.get(key) ?? 0) + 1)
  inDegree[org.edges.post[s]]++
}

console.log(`exp002 M1 diagnosis — seed ${SEED}, ${TRIALS} trials, tail accuracy ${tail.toFixed(3)}`)
console.log()
console.log('1. STRUCTURE')
console.log(`   live edges: ${org.edges.count}`)
for (const [k, v] of [...byPair.entries()].sort((a, b) => b[1] - a[1])) {
  console.log(`   ${k.padEnd(22)} ${v}`)
}
let interiorIn = 0
let interiorN = 0
for (let i = 0; i < lat.size; i++) {
  if (lat.role[i] === ROLE_INTERIOR) {
    interiorIn += inDegree[i]
    interiorN++
  }
}
console.log(`   mean interior in-degree: ${(interiorIn / interiorN).toFixed(2)}`)
console.log(
  `   output in-degrees      : ${Array.from(lat.outputNodes, (i) => inDegree[i]).join(', ')}`
)
const outW: number[] = []
for (let s = 0; s < org.edges.count; s++) {
  if (lat.role[org.edges.post[s]] === ROLE_OUTPUT) outW.push(org.edges.w[s])
}
console.log(
  `   |w| into outputs       : n=${outW.length} mean ${
    outW.length ? (outW.reduce((a, b) => a + Math.abs(b), 0) / outW.length).toFixed(4) : 'n/a'
  } max ${outW.length ? Math.max(...outW.map(Math.abs)).toFixed(4) : 'n/a'} (born at ${cfg.birthWeight})`
)

// hop distance from the input cortex, over live edges — measured on the
// structure as trained, BEFORE the probe below touches anything
const hop = new Int32Array(lat.size).fill(-1)
{
  const q = new Int32Array(lat.size)
  let head = 0
  let tail2 = 0
  for (const i of lat.inputNodes) {
    hop[i] = 0
    q[tail2++] = i
  }
  while (head < tail2) {
    const i = q[head++]
    const [a, b] = org.edges.outRange(i)
    for (let s = a; s < b; s++) {
      if (Math.abs(org.edges.w[s]) < cfg.deathThreshold) continue
      const j = org.edges.post[s]
      if (hop[j] !== -1) continue
      hop[j] = hop[i] + 1
      q[tail2++] = j
    }
  }
}

/** How many standard errors apart are this node's per-pattern firing rates?
 * Anything near 0 means the node's firing is independent of the stimulus. */
function discriminability(i: number): number {
  const r = [rate[0][i], rate[1][i], rate[2][i]]
  const mean = (r[0] + r[1] + r[2]) / 3
  const spread = Math.max(...r) - Math.min(...r)
  const se = Math.sqrt(Math.max(1e-9, (mean * (1 - mean)) / PROBE))
  return spread / se
}

// ── 2 & 3. information: hold the substrate still and measure firing per pattern
// Learning is off AND rent is suspended for the probe. Rent matters: the probe
// runs thousands of ticks, and at 9e-5 per tick it would decay every weight
// past the death threshold long before the measurement finished — which is
// exactly the artefact the first version of this tool reported.
teacher.learning = false
org.cfg.rent = 0
const PROBE = 3000
const rate: Float32Array[] = [0, 1, 2].map(() => new Float32Array(lat.size))
for (let label = 0; label < 3; label++) {
  org.sense.set(M1_PATTERNS[label])
  org.clearTraces()
  for (let t = 0; t < 200; t++) org.tick() // settle
  for (let t = 0; t < PROBE; t++) {
    org.tick()
    for (let i = 0; i < lat.size; i++) rate[label][i] += org.poolFired[i]
  }
  for (let i = 0; i < lat.size; i++) rate[label][i] /= PROBE
}

console.log()
console.log('2. INFORMATION — how far from the input does pattern-dependence survive?')
console.log('   (spread of per-pattern firing rate, in standard errors; <3 is noise)')
const buckets = new Map<number, number[]>()
for (let i = 0; i < lat.size; i++) {
  if (lat.role[i] !== ROLE_INTERIOR) continue
  if (hop[i] < 0) continue
  if (!buckets.has(hop[i])) buckets.set(hop[i], [])
  buckets.get(hop[i])!.push(discriminability(i))
}
for (const h of [...buckets.keys()].sort((a, b) => a - b)) {
  const v = buckets.get(h)!.sort((a, b) => b - a)
  const mean = v.reduce((a, b) => a + b, 0) / v.length
  console.log(
    `   hop ${h}: n=${String(v.length).padStart(4)}  mean ${mean.toFixed(1)}  ` +
      `best ${v[0].toFixed(1)}  #{>3σ} ${v.filter((x) => x > 3).length}`
  )
}
const inputD = Array.from(lat.inputNodes, discriminability)
console.log(
  `   input cortex (hop 0, clamped): mean ${(
    inputD.reduce((a, b) => a + b, 0) / inputD.length
  ).toFixed(1)}`
)

console.log()
console.log('3. OUTPUT — does the answer depend on the pattern?')
for (let k = 0; k < 3; k++) {
  const i = lat.outputNodes[k]
  console.log(
    `   output ${k}: rates ${[0, 1, 2].map((l) => rate[l][i].toFixed(3)).join(' / ')}  ` +
      `discriminability ${discriminability(i).toFixed(1)}σ`
  )
}

// ── 4. the two forces on a weight ───────────────────────────────────────────
console.log()
console.log('4. FORCES — reward versus rent, per edge')
const ticksPerTrial = trainingTicks / TRIALS
const rentPerTrial = cfg.rent * ticksPerTrial
const rewardPerEvent = rewardEvents ? rewardWork / rewardEvents : 0
console.log(`   ticks per trial        : ${ticksPerTrial.toFixed(1)}`)
console.log(`   rent paid per trial    : ${rentPerTrial.toFixed(5)} per edge`)
console.log(`   |Δw| from reward       : ${rewardPerEvent.toFixed(5)} per edge per rewarded trial`)
console.log(`   ratio reward/rent      : ${(rewardPerEvent / rentPerTrial).toFixed(2)}`)
console.log(
  `   edge lifetime at birth : ${(cfg.birthWeight - cfg.deathThreshold) / rentPerTrial} trials ` +
    `unearned, vs sleep every ${cfg.sleepEvery}`
)
const s = org.stats()
console.log(
  `   churn                  : ${s.edgesBorn} born, ${s.edgesDied} died over ${s.sleeps} sleeps ` +
    `(${(s.edgesDied / Math.max(1, s.edgesBorn) * 100).toFixed(0)}% of births die)`
)
