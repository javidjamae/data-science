// Experiment 002 M1 — the critical gate (design §8).
//
//   Uniform reward field (= 001's broadcast), uniform latency (all d = 1),
//   rent and growth on, from zero edges.
//
//   Gate: ≥0.80 rolling accuracy over the last 100 of 2,000 trials, on ≥3
//   seeds, AND a connected input→output path exists at the end.
//
// If M1 fails, everything below it is moot and the finding is "reward-driven
// growth from zero does not reach a competence a fixed random projection
// reaches in 800 trials" — a real negative, written up with the same rigour
// as a success (journal rule 5).
//
// The baseline this is measured against is NOT the 0.33 chance line. L-010
// showed 001's competence is carried by a 480-weight readout on a *fixed
// random projection*, so the thing to beat is a random projection.
//
//   npx vite-node tools/exp002-m1-gate.ts [arm] [trials] [seeds...]

import { GrownOrganism } from '../src/engine/grown/grown-organism'
import { defaultGrownConfig, type GrownConfig } from '../src/engine/grown/config'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

/**
 * Arms. `m1` is the pre-registered gate configuration and the only one whose
 * result is the gate; everything else is a diagnostic arm, run *after* m1
 * failed, and reported as post-hoc.
 */
const ARMS: Record<string, { note: string; cfg: Partial<GrownConfig> }> = {
  m1: {
    note: 'the pre-registered gate: uniform field, uniform latency, from zero edges',
    cfg: {},
  },
  shallow: {
    note:
      'design §10 risk 1, last rung: shorten the input→output distance. NOT a ' +
      'depth-0 arm — at outputX 14 only the right-hand columns of the input ' +
      'block are within rMax of an output, so part of the sense gets a direct ' +
      'edge and the rest still needs two hops or more. Read the inputHops ' +
      'distribution below rather than the shortest-path figure.',
    cfg: { outputX: 14 },
  },
  'no-rent': {
    note: 'pre-registered control arm: ρ=0, nothing ever dies',
    cfg: { rent: 0 },
  },
  'no-spont': {
    // Removing pSpont does NOT remove the bootstrap, and neither does also
    // burying the bias. The homeostat is an activity *source*: with nothing
    // firing it drives inhibition negative without bound until the interior
    // fires again, converging on targetSparsity from a completely silent
    // sheet (measured: 0.0000 → 0.0179 → 0.1499 over 25k ticks at bias −30).
    // Run this arm without `inhibitionRate: 0` and it wires up normally, and
    // recording that as "the cold-start control failed outright" would be a
    // false conclusion drawn from a control that never controlled anything.
    note:
      'pre-registered control arm: the cold-start control, expected to fail ' +
      'outright. Removing spontaneous firing is not enough on its own — the ' +
      'homeostat regenerates activity from silence — so this also buries the ' +
      'resting bias and switches the homeostat off.',
    cfg: { pSpont: 0, bias: -30, urgeMax: 0, inhibitionRate: 0 },
  },
}

const argv = process.argv.slice(2)
const ARM = argv.length > 0 ? argv[0] : 'm1'
const TRIALS = argv.length > 1 ? Number(argv[1]) : 2000
const SEEDS = argv.length > 2 ? argv.slice(2).map(Number) : [1, 2, 3]
const TAIL = 100
const GATE = 0.8

if (!ARMS[ARM]) {
  console.error(`unknown arm "${ARM}"; known: ${Object.keys(ARMS).join(', ')}`)
  process.exit(1)
}
const armCfg: GrownConfig = { ...defaultGrownConfig, ...ARMS[ARM].cfg }

function runSeed(seed: number) {
  const org = new GrownOrganism({ ...armCfg, seed })
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(seed * 7919 + 1)

  const results: boolean[] = []
  const curve: number[] = []
  const edgeTrace: number[] = []
  let block: number[] = []
  let silent = 0
  const t0 = Date.now()

  for (let t = 0; t < TRIALS; t++) {
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    const label = block.pop()!
    const res = teacher.runTrial(org, M1_PATTERNS[label], label)
    results.push(res.correct)
    if (res.spoken === null) silent++
    if ((t + 1) % 100 === 0) {
      const w = results.slice(-100)
      curve.push(w.filter(Boolean).length / w.length)
      edgeTrace.push(org.edges.count)
    }
  }

  const tail = results.slice(-TAIL)
  const stats = org.stats()
  return {
    seed,
    // divide by what was actually measured: at TRIALS < TAIL the window is
    // short, and a constant denominator would silently understate the headline
    tail: tail.filter(Boolean).length / tail.length,
    curve,
    edgeTrace,
    silentFraction: silent / TRIALS,
    stats,
    norms: org.weightNorms(),
    seconds: (Date.now() - t0) / 1000,
    ticksPerSecond: stats.ticks / Math.max(0.001, (Date.now() - t0) / 1000),
  }
}

console.log(
  `M1 [arm: ${ARM}] — ${TRIALS} trials, seeds ${SEEDS.join(',')}, ` +
    `rewardField=${armCfg.rewardField}, latency=${armCfg.latency}`
)
console.log(`  ${ARMS[ARM].note}`)
if (ARM !== 'm1') console.log('  NOT the gate — a diagnostic arm, post-hoc.')
console.log()

const runs = SEEDS.map(runSeed)

for (const r of runs) {
  const s = r.stats
  console.log(`── seed ${r.seed} ─────────────────────────────────────────────`)
  console.log(`  accuracy /100 : ${r.curve.map((a) => a.toFixed(2)).join(' → ')}`)
  console.log(`  tail (last ${TAIL}): ${r.tail.toFixed(3)}   silent trials: ${(r.silentFraction * 100).toFixed(1)}%`)
  console.log(`  edges /100    : ${r.edgeTrace.join(' → ')}`)
  console.log(
    `  structure     : ${s.edges} live, ${s.edgesBorn} born, ${s.edgesDied} died, ` +
      `${s.sleeps} sleeps, ${s.growthAttempts} attempts, capBinds ${s.capBinds}`
  )
  console.log(
    `  input→output  : connected ${JSON.stringify(s.connected)}  shortest hops ${JSON.stringify(s.hops)}`
  )
  // the honest depth figure: how far each of the 64 sense pixels is from an
  // answer. The shortest-path number above is one pixel's story, not the
  // substrate's.
  console.log(
    `  sense depth   : ${s.inputHops
      .map((n, h) => `${h === 0 ? 'unreached' : `${h}hop`} ${n}`)
      .filter((_, h) => s.inputHops[h] > 0)
      .join('  ')}  (of 64 pixels)`
  )
  console.log(
    `  credit reach  : trace horizon ${s.traceHorizonTicks.toFixed(1)} ticks vs deepest path ` +
      `${s.deepestPathDelayTicks ?? 'n/a'} ticks`
  )
  console.log(
    `  activity      : mean ${s.meanActivity.toFixed(4)} var ${s.activityVariance.toExponential(2)}`
  )
  console.log(
    `  output cortex : ${s.spokenTicks} clean ticks, ${s.ambiguousTicks} ambiguous ` +
      `(${((100 * s.ambiguousTicks) / Math.max(1, s.spokenTicks + s.ambiguousTicks)).toFixed(1)}%)`
  )
  console.log(`  first reward  : tick ${s.ticksToFirstReward ?? 'never'}`)
  console.log(`  weight norms  : pool ${r.norms.pool.toFixed(1)}  out ${r.norms.out.toFixed(1)}`)
  console.log(`  speed         : ${r.ticksPerSecond.toFixed(0)} ticks/s over ${r.seconds.toFixed(1)}s`)
  console.log()
}

const meanTail = runs.reduce((a, r) => a + r.tail, 0) / runs.length
const accuracyPass = runs.every((r) => r.tail >= GATE)
const pathPass = runs.every((r) => r.stats.connected.some(Boolean))
const fullPathPass = runs.every((r) => r.stats.connected.every(Boolean))

console.log('══ gate ═══════════════════════════════════════════════════════')
console.log(`  mean tail accuracy   : ${meanTail.toFixed(3)}   (chance 0.333)`)
console.log(`  every seed ≥ ${GATE}     : ${accuracyPass ? 'PASS' : 'FAIL'}`)
console.log(`  some output connected: ${pathPass ? 'PASS' : 'FAIL'}`)
console.log(`  all outputs connected: ${fullPathPass ? 'PASS' : 'FAIL'}`)
console.log(`  M1                   : ${accuracyPass && pathPass ? 'PASS' : 'FAIL'}`)
