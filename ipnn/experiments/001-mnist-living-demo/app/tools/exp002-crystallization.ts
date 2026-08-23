// Is anything crystallising? [J]
//
// THE HYPOTHESIS THIS TESTS, in Javid's words: "Something in the network has
// to build slowly and strongly without decaying. And then when it gets strong
// enough, faster things can grow around it. But if we're not seeing slow
// growth even, then the system is not moving towards learning."
//
// That is sharper than L-019's "slow learning requires a slow variable",
// because it names the shape the slow variable has to have: a core that
// accumulates monotonically, reaches a strength threshold, and then acts as
// scaffolding for faster processes. Developmental biology works this way —
// pioneer axons and radial glia lay a frame before anything else follows it.
//
// WHY THE EXISTING MEASUREMENT CANNOT ANSWER IT. `exp002-longrun.ts` reports
// the fraction of edges alive at one checkpoint still alive at the next: 5%,
// flat forever. That single number is consistent with two opposite worlds:
//
//   (a) CHURN        a random 5% survive each time, never the same ones.
//                    Age distribution geometric, no core, nothing accumulates.
//   (b) CRYSTAL      the SAME 5% survive every time, a stable core under a
//                    churning surface. Age distribution heavy-tailed, and the
//                    core grows.
//
// Aggregate survival is identical in both. Only edge *identity over time*
// separates them, which is why `Edge.born` now exists.
//
// WHAT WOULD COUNT AS EACH ANSWER, stated before running (H-008):
//   crystallising → the old-edge population grows across checkpoints, and old
//                   edges are stronger (|w|) and better-evidenced (n) than
//                   young ones.
//   churning      → the age histogram is stationary and geometric, oldest-age
//                   grows only logarithmically, and old edges look like young
//                   ones on every other measure.
//
//   npx vite-node tools/exp002-crystallization.ts [trials] [seed]

import { GrownOrganism } from '../src/engine/grown/grown-organism'
import { defaultGrownConfig, type GrownConfig } from '../src/engine/grown/config'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

const argv = process.argv.slice(2)
const TOTAL = Number(argv[0] ?? 16_000)
const SEED = Number(argv[1] ?? 1)
const CHECKPOINTS = (() => {
  const cs: number[] = []
  for (let t = 1000; t < TOTAL; t *= 2) cs.push(t)
  cs.push(TOTAL)
  return cs
})()
/** an edge this old has survived this many structural rewirings */
const OLD = 20

function run(name: string, over: Partial<GrownConfig>) {
  const org = new GrownOrganism({ ...defaultGrownConfig, ...over, seed: SEED })
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(SEED * 7919 + 1)
  let block: number[] = []
  let done = 0

  console.log(`\n${name}`)
  console.log(
    '  trials | edges | median age | p90 age | oldest | core(≥20) | ' +
      '|w| old/young | n old/young | survival/sleep'
  )
  console.log(
    '  -------|-------|------------|---------|--------|-----------|' +
      '---------------|-------------|---------------'
  )

  const coreSizes: number[] = []
  for (const cp of CHECKPOINTS) {
    while (done < cp) {
      if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
      teacher.runTrial(org, M1_PATTERNS[block[block.length - 1]], block.pop()!)
      done++
    }

    const ages = org.edgeAges()
    const n = ages.length
    if (n === 0) continue
    const sorted = Int32Array.from(ages).sort()
    const median = sorted[Math.floor(n / 2)]
    const p90 = sorted[Math.floor(n * 0.9)]
    const oldest = sorted[n - 1]

    let core = 0
    let wOld = 0
    let wYoung = 0
    let nOld = 0
    let nYoung = 0
    let cOld = 0
    let cYoung = 0
    for (let s = 0; s < n; s++) {
      if (ages[s] >= OLD) {
        core++
        wOld += Math.abs(org.edges.w[s])
        nOld += org.edges.n[s]
        cOld++
      } else if (ages[s] <= 2) {
        wYoung += Math.abs(org.edges.w[s])
        nYoung += org.edges.n[s]
        cYoung++
      }
    }
    coreSizes.push(core)

    // per-sleep survival implied by the age histogram: under pure churn with
    // constant hazard, P(age ≥ k) = p^k, so p = (fraction aged ≥ 1)
    let ge1 = 0
    for (let s = 0; s < n; s++) if (ages[s] >= 1) ge1++
    const perSleep = ge1 / n

    const f = (v: number, c: number) => (c ? (v / c).toFixed(3) : ' n/a ')
    console.log(
      `  ${String(cp).padStart(6)} | ${String(n).padStart(5)} | ` +
        `${String(median).padStart(10)} | ${String(p90).padStart(7)} | ` +
        `${String(oldest).padStart(6)} | ${String(core).padStart(9)} | ` +
        `${f(wOld, cOld)} / ${f(wYoung, cYoung)} | ${f(nOld, cOld)} / ${f(nYoung, cYoung)} | ` +
        `${perSleep.toFixed(3)}`
    )
  }

  const growing =
    coreSizes.length > 2 && coreSizes[coreSizes.length - 1] > coreSizes[0] * 1.5
  console.log(
    `  → core (edges surviving ≥${OLD} rewirings) went ${coreSizes[0]} → ` +
      `${coreSizes[coreSizes.length - 1]}: ${growing ? 'GROWING' : 'not growing'}`
  )
  return coreSizes
}

console.log(
  `Crystallisation test — ${TOTAL} trials, seed ${SEED}, ` +
    `sleepEvery ${defaultGrownConfig.sleepEvery} (so ${Math.round(TOTAL / defaultGrownConfig.sleepEvery)} rewirings)\n` +
    `Age is measured in SLEEPS SURVIVED. "core" = edges that have lived through ≥${OLD} rewirings.`
)

run('M1 arm', {})
run('shallow arm', { outputX: 14 })
run('no-rent control (nothing can die)', { rent: 0 })

console.log(
  '\nReading: a geometric age histogram with a stationary core is CHURN — a ' +
    'stationary\ndistribution, not a slow climb. A growing core with stronger, ' +
    'better-evidenced old\nedges is CRYSTALLISATION, and would mean something is ' +
    'building slowly underneath.'
)
