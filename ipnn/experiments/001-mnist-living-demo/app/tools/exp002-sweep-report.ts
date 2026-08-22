// Read a sweep back, and try hard to disbelieve its winners.
//
// The whole hazard of a 24-knob random search is that the top of the ranking
// is where the luck accumulates: with enough draws, some configuration scores
// well on the search seeds for no reason at all. So this does not report the
// leaderboard. It re-runs the top candidates on **seeds they have never been
// evaluated on**, for longer than the screen ran, and reports the held-out
// number as the result — with the search-seed number beside it purely so the
// size of the shrinkage is visible.
//
// It also reports what was discarded and why, which design §10 risk 2
// requires and which a leaderboard silently omits.
//
//   npx vite-node tools/exp002-sweep-report.ts [--top=12] [--trials=2000]

import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { GrownOrganism } from '../src/engine/grown/grown-organism'
import type { GrownConfig } from '../src/engine/grown/config'
import { defaultGrownConfig } from '../src/engine/grown/config'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'
import { SEARCH_SPACE_VERSION, SPACE } from './exp002-sweep'

const arg = (n: string, d: string) =>
  process.argv.slice(2).find((a) => a.startsWith(`--${n}=`))?.split('=')[1] ?? d

const TOP = Number(arg('top', '12'))
const TRIALS = Number(arg('trials', '2000'))
/** never used during the search — this is the whole point */
const HELDOUT = arg('heldout', '11,12,13,14,15').split(',').map(Number)
const DIR = join(process.cwd(), 'sweeps')

interface Row {
  space: string
  git: string
  score: number
  trials: number
  searchSeeds: number[]
  seeds: { seed: number; tail: number; edges: number; connected: boolean; withinTwoHops: number; aborted: boolean }[]
  config: GrownConfig
}

const rows: Row[] = []
for (const f of readdirSync(DIR).filter((f) => f.endsWith('.jsonl'))) {
  for (const line of readFileSync(join(DIR, f), 'utf8').split('\n')) {
    if (!line.trim()) continue
    const r = JSON.parse(line) as Row
    if (r.space === SEARCH_SPACE_VERSION) rows.push(r)
  }
}

if (rows.length === 0) {
  console.error(`no rows for space ${SEARCH_SPACE_VERSION} in ${DIR}`)
  process.exit(1)
}

// ───────────────────────────────────────────────────── what the search saw

const aborted = rows.filter((r) => r.seeds.some((s) => s.aborted)).length
const disconnected = rows.filter((r) => r.seeds.some((s) => !s.connected)).length
const reachedTwoHops = rows.filter((r) => r.seeds.some((s) => s.withinTwoHops > 0)).length
const sorted = [...rows].sort((a, b) => b.score - a.score)
const scores = rows.map((r) => r.score).sort((a, b) => a - b)
const q = (p: number) => scores[Math.floor(p * (scores.length - 1))]

console.log(`sweep ${SEARCH_SPACE_VERSION} — ${rows.length} configurations`)
console.log(`  commits          : ${[...new Set(rows.map((r) => r.git))].join(', ')}`)
console.log(`  screen length    : ${[...new Set(rows.map((r) => r.trials))].join(', ')} trials`)
console.log(`  search seeds     : ${[...new Set(rows.map((r) => r.searchSeeds.join('+')))].join(', ')}`)
console.log()
console.log('score distribution (worst-seed tail accuracy; chance 0.333)')
console.log(
  `  min ${q(0).toFixed(3)} · p50 ${q(0.5).toFixed(3)} · p90 ${q(0.9).toFixed(3)} · ` +
    `p99 ${q(0.99).toFixed(3)} · max ${q(1).toFixed(3)}`
)
console.log(`  configs ≥ 0.50   : ${rows.filter((r) => r.score >= 0.5).length}`)
console.log(`  configs ≥ 0.80   : ${rows.filter((r) => r.score >= 0.8).length}  ← the gate`)
console.log()
console.log('discarded / degenerate (reported, not hidden)')
console.log(`  hit the edge-count abort : ${aborted}`)
console.log(`  never connected to an output on some seed : ${disconnected}`)
console.log(`  got any sense pixel within 2 hops : ${reachedTwoHops} of ${rows.length}`)
console.log()

// ─────────────────────────────────────────── re-run the winners, honestly

function evaluate(cfg: GrownConfig, seed: number, trials: number): number {
  const org = new GrownOrganism({ ...cfg, seed })
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(seed * 7919 + 1)
  const res: boolean[] = []
  let block: number[] = []
  for (let t = 0; t < trials; t++) {
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    const label = block.pop()!
    res.push(teacher.runTrial(org, M1_PATTERNS[label], label).correct)
  }
  const tail = res.slice(-100)
  return tail.filter(Boolean).length / tail.length
}

console.log(
  `re-running the top ${TOP} on held-out seeds ${HELDOUT.join(',')} at ${TRIALS} trials\n`
)
console.log('rank | search score | held-out (worst / mean) | shrinkage | verdict')
console.log('-----|--------------|-------------------------|-----------|--------')

const validated = sorted.slice(0, TOP).map((r, i) => {
  const held = HELDOUT.map((s) => evaluate(r.config, s, TRIALS))
  const worst = Math.min(...held)
  const mean = held.reduce((a, b) => a + b, 0) / held.length
  const verdict = worst >= 0.8 ? 'PASSES GATE' : mean >= 0.5 ? 'partial' : 'noise'
  console.log(
    `  ${String(i + 1).padStart(2)} |    ${r.score.toFixed(3)}     |    ` +
      `${worst.toFixed(3)} / ${mean.toFixed(3)}       |  ${(r.score - mean >= 0 ? '+' : '')}` +
      `${(r.score - mean).toFixed(3)}   | ${verdict}`
  )
  return { row: r, worst, mean }
})

const survivors = validated.filter((v) => v.worst >= 0.8)
console.log()
if (survivors.length === 0) {
  const bestHeld = Math.max(...validated.map((v) => v.mean))
  console.log(
    `NO configuration cleared the gate on held-out seeds. Best held-out mean: ${bestHeld.toFixed(3)}.`
  )
  console.log(
    'The search-seed scores above are therefore selection noise, and the honest'
  )
  console.log(
    'reading is that no setting of these knobs rescues the M1 geometry — not that'
  )
  console.log('none exists, but that none was found in this many draws of this space.')
} else {
  console.log(`${survivors.length} configuration(s) survived held-out validation:`)
  for (const s of survivors) {
    console.log(`\n  held-out worst ${s.worst.toFixed(3)} mean ${s.mean.toFixed(3)}`)
    const diffs = Object.keys(SPACE)
      .filter(
        (k) =>
          JSON.stringify((s.row.config as never)[k]) !==
          JSON.stringify((defaultGrownConfig as never)[k])
      )
      .map((k) => {
        const v = (s.row.config as never)[k] as number | boolean
        return `${k}=${typeof v === 'number' ? Number(v.toPrecision(3)) : v}`
      })
    console.log(`  differs from the M1 default in: ${diffs.join(', ')}`)
  }
}
