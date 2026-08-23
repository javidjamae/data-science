// Random search over the grown substrate's knobs, with provenance.
//
// ── Read this before trusting any result it produces ──────────────────────
// Design §10 risk 2, verbatim: "Free parameters are the real threat to this
// experiment's credibility. A system with this many knobs can be tuned into
// producing almost any result. Any parameter search must be reported,
// including the arms that were tried and discarded."
//
// With ~25 free knobs and 3 seeds, a random search WILL find a configuration
// that scores well by luck alone. That is not a discovery, it is the
// multiple-comparisons problem wearing a lab coat. Two defences are built in
// and neither is optional:
//
//   1. SEED SPLIT. Configurations are scored on search seeds only. The
//      reporter re-runs the winners on seeds they have never been evaluated
//      on. A config that wins on search seeds and collapses on held-out seeds
//      was noise, and the split is what exposes it.
//   2. FULL PROVENANCE. Every run — winners and losers alike — appends one
//      JSONL line carrying its complete config, the git commit that produced
//      it, the search-space version, and its per-seed results. Nothing is
//      filtered before it is written down, so "we tried 4,000 configs and
//      report the best" is auditable rather than asserted.
//
// Geometry is deliberately FIXED at the failing M1 arm. Sweeping `outputX`
// would simply rediscover "move the outputs closer", which is already known
// (L-013) and is not what this is asking. The question here is the harder
// one: holding the hard geometry fixed, does any parameter setting make
// information cross more than one hop?
//
//   npx vite-node tools/exp002-sweep.ts --runs=200 --trials=1000 --shard=0/8
//
// Shards write separate files and can run concurrently; the reporter merges.

import { execSync } from 'node:child_process'
import { appendFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { GrownOrganism } from '../src/engine/grown/grown-organism'
import { defaultGrownConfig, type GrownConfig } from '../src/engine/grown/config'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace, type Rng } from '../src/engine/rng'

/** Bump when the space below changes, so old rows are never silently mixed
 * with new ones in a report. */
export const SEARCH_SPACE_VERSION = 'v1'

const arg = (name: string, fallback: string): string => {
  const hit = process.argv.slice(2).find((a) => a.startsWith(`--${name}=`))
  return hit ? hit.split('=').slice(1).join('=') : fallback
}

const RUNS = Number(arg('runs', '200'))
const TRIALS = Number(arg('trials', '1000'))
const SEARCH_SEEDS = arg('seeds', '1,2').split(',').map(Number)
const [SHARD, SHARDS] = arg('shard', '0/1').split('/').map(Number)
const OUT_DIR = join(process.cwd(), 'sweeps')
const OUT = join(OUT_DIR, `exp002-sweep-${SEARCH_SPACE_VERSION}-shard${SHARD}.jsonl`)
/** hard stop: a runaway config must not hang the shard */
const EDGE_ABORT = 80_000

const GIT = (() => {
  try {
    return execSync('git rev-parse --short HEAD', { encoding: 'utf8' }).trim()
  } catch {
    return 'unknown'
  }
})()

// ─────────────────────────────────────────────────────────── the search space

type Sampler = (r: Rng) => number | boolean
const uniform = (lo: number, hi: number): Sampler => (r) => lo + r() * (hi - lo)
/** log-uniform: the right prior for a rate or a scale, where the interesting
 * variation is multiplicative rather than additive */
const logUniform =
  (lo: number, hi: number): Sampler =>
  (r) =>
    Math.exp(Math.log(lo) + r() * (Math.log(hi) - Math.log(lo)))
const intUniform =
  (lo: number, hi: number): Sampler =>
  (r) =>
    Math.floor(lo + r() * (hi - lo + 1))
const choice =
  (...vals: (number | boolean)[]): Sampler =>
  (r) =>
    vals[Math.floor(r() * vals.length)]

/**
 * Every knob that could plausibly change whether information crosses a hop.
 * Grouped by what it governs, which is also how the report reads them back.
 */
export const SPACE: Record<string, Sampler> = {
  // how excitable a node is, and how sparse the sheet stays
  gain: logUniform(0.5, 8),
  bias: uniform(-3.5, 0.5),
  targetSparsity: logUniform(0.02, 0.45),
  inhibitionRate: logUniform(0.001, 0.15),
  pSpont: logUniform(0.0005, 0.12),

  // how the three answers compete, and how hard silence is pushed to break
  lateralInhibition: uniform(0, 6),
  urgeRate: logUniform(0.005, 0.4),
  urgeMax: uniform(0.5, 7),

  // the learning rule
  traceDecay: uniform(0.75, 0.995),
  eta: logUniform(0.004, 0.6),
  wMax: logUniform(0.4, 8),
  consolidation: choice(true, false),
  consolidationN0: logUniform(50, 20_000),

  // metabolism: what it costs to exist, what a new edge is worth, when it dies
  rent: logUniform(5e-6, 2e-3),
  birthWeight: logUniform(0.02, 1.2),
  deathThreshold: logUniform(0.001, 0.3),

  // how it builds
  growthAttempts: intUniform(1, 8),
  // capped at 11: the M1 geometry puts the reward locus 11.7 lattice units
  // off the input→output axis, and the Lattice constructor rejects an rMax
  // that would let one growth cone reach both (design §3). Sampling past that
  // does not explore anything, it just throws.
  rMax: intUniform(3, 11),
  lambdaG: uniform(0.8, 14),
  sleepEvery: intUniform(2, 80),
  maxOutDegree: intUniform(4, 128),

  // the activity field growth climbs
  activityD: uniform(0.005, 0.22),
  activityDecay: logUniform(0.0005, 0.06),
}

/** Draw one configuration, then repair the combinations that are incoherent
 * rather than merely unpromising — an edge born below its own death threshold
 * would die at the first sleep every time, which tests nothing. */
export function sampleConfig(r: Rng, seed: number): GrownConfig {
  const drawn: Record<string, number | boolean> = {}
  for (const [k, s] of Object.entries(SPACE)) drawn[k] = s(r)

  const birthWeight = drawn.birthWeight as number
  let deathThreshold = drawn.deathThreshold as number
  if (deathThreshold >= birthWeight * 0.8) deathThreshold = birthWeight * 0.8 * (0.2 + 0.6 * r())
  drawn.deathThreshold = deathThreshold
  drawn.wMax = Math.max(drawn.wMax as number, birthWeight * 1.5)

  return {
    ...defaultGrownConfig,
    ...drawn,
    // fixed: the M1 arm's geometry and its two control-arm switches. This
    // search asks whether the parameters can rescue the hard geometry, not
    // whether an easier geometry exists.
    seed,
    rewardField: 'uniform',
    latency: 'uniform',
    outputX: defaultGrownConfig.outputX,
  } as GrownConfig
}

// ───────────────────────────────────────────────────────────────── one run

export interface SeedResult {
  seed: number
  tail: number
  conditional: number
  silence: number
  edges: number
  connected: boolean
  withinTwoHops: number
  aborted: boolean
}

function runSeed(cfg: GrownConfig, trials: number): SeedResult {
  const org = new GrownOrganism(cfg)
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(cfg.seed * 7919 + 1)
  const rows: { correct: boolean; spoken: number | null }[] = []
  let block: number[] = []
  let aborted = false

  for (let t = 0; t < trials; t++) {
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    const label = block.pop()!
    const r = teacher.runTrial(org, M1_PATTERNS[label], label)
    rows.push({ correct: r.correct, spoken: r.spoken })
    if ((t & 31) === 31 && org.edges.count > EDGE_ABORT) {
      aborted = true
      break
    }
  }

  const tail = rows.slice(-100)
  const spoke = tail.filter((r) => r.spoken !== null)
  const stats = org.stats()
  const h = stats.inputHops
  return {
    seed: cfg.seed,
    tail: tail.length ? tail.filter((r) => r.correct).length / tail.length : 0,
    conditional: spoke.length ? spoke.filter((r) => r.correct).length / spoke.length : 0,
    silence: tail.length ? 1 - spoke.length / tail.length : 1,
    edges: stats.edges,
    connected: stats.connected.some(Boolean),
    withinTwoHops: (h[1] ?? 0) + (h[2] ?? 0),
    aborted,
  }
}

/** Worst seed, not mean. A configuration that works on one seed and fails on
 * another has not solved anything, and scoring by mean rewards exactly that. */
export function score(seeds: SeedResult[]): number {
  return Math.min(...seeds.map((s) => s.tail))
}

// ─────────────────────────────────────────────────────────────────── driver

mkdirSync(OUT_DIR, { recursive: true })
// each shard draws from its own stream, so shards never duplicate each other
const rng = mulberry32(0xc0ffee + SHARD * 1_000_003)

console.log(
  `sweep ${SEARCH_SPACE_VERSION} shard ${SHARD}/${SHARDS} — ${RUNS} configs × ` +
    `${SEARCH_SEEDS.length} seeds × ${TRIALS} trials → ${OUT}`
)

let best = -1
for (let i = 0; i < RUNS; i++) {
  const base = sampleConfig(rng, SEARCH_SEEDS[0])
  const t0 = Date.now()

  // A draw the substrate refuses to build (an incoherent geometry, say) is
  // still a draw, and writing it down is what keeps "we tried N configs"
  // honest. It must never take the shard down with it.
  let seeds: SeedResult[] | null = null
  let error: string | null = null
  try {
    seeds = SEARCH_SEEDS.map((s) => runSeed({ ...base, seed: s }, TRIALS))
  } catch (e) {
    error = e instanceof Error ? e.message : String(e)
  }
  const sc = seeds ? score(seeds) : -1

  appendFileSync(
    OUT,
    JSON.stringify({
      space: SEARCH_SPACE_VERSION,
      git: GIT,
      shard: SHARD,
      run: i,
      trials: TRIALS,
      searchSeeds: SEARCH_SEEDS,
      score: sc,
      seeds: seeds ?? [],
      rejected: error,
      seconds: (Date.now() - t0) / 1000,
      config: base,
    }) + '\n'
  )
  if (error || !seeds) {
    console.log(`  [${SHARD}] run ${i}: rejected — ${(error ?? '').slice(0, 80)}`)
    continue
  }

  if (sc > best) {
    best = sc
    console.log(
      `  [${SHARD}] run ${i}: new best worst-seed tail ${sc.toFixed(3)} ` +
        `(edges ${seeds[0].edges}, within-2-hops ${seeds[0].withinTwoHops})`
    )
  } else if (i % 25 === 24) {
    console.log(`  [${SHARD}] ${i + 1}/${RUNS} done, best ${best.toFixed(3)}`)
  }
}
console.log(`  [${SHARD}] finished ${RUNS} runs, best worst-seed tail ${best.toFixed(3)}`)
