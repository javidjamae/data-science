// The innate-scaffold sweep — Javid's proposal, run the night it was made.
//
//   "What if we randomly create a bunch of interconnections throughout the
//    network as a starting point, and then run it and let it start
//    strengthening and rewiring itself from there. That way there is already
//    a possibility for a pathway." [J, 2026-08-22]
//
// Biology concurs: overproduction-then-pruning, and innate long tracts.
// 002's own design listed "seed a sparse random scaffold" in its mitigation
// ladder with the caveat that it weakens the from-zero claim — reported here.
//
// ── PRE-REGISTERED, committed before first run ────────────────────────────
// ARMS: seed density {1k, 3k, 6k, 12k} × span {long (44 = whole sheet),
// local (8 = growth's own limit)}, 3 seeds × 1,500 trials, M1 geometry.
// PREDICTION [C]: long-span seeding at sufficient density produces the first
// learning ever seen on this geometry — it manufactures the short eye→answer
// routes whose absence killed all 50 knob arms (within2 > 0 at birth), and
// the shallow arm proved the rule learns when routes exist. Local-span
// seeding at any density stays at chance: same edges, no short routes —
// this is the control that shows LONG-RANGE innateness, not edge count, is
// what matters. Gate: the M1 numbers (tail ≥ 0.8, 3 seeds).
// DECISION RULE: best long arm min-seed ≥ 0.5 → this becomes the birth
// condition for the 006 developmental program, and the kill-and-mutate
// evolutionary loop over scaffolds (H-014's first implementation) is the
// registered follow-up. All arms < 0.5 → random innateness is insufficient
// and 006 leads with guided scaffolds (H-020) instead.
//
//   npx vite-node tools/exp002-innate-seed.ts <shard> <nshards>

import { GrownOrganism } from '../src/engine/grown/grown-organism'
import { defaultGrownConfig } from '../src/engine/grown/config'
import { ROLE_OUTPUT } from '../src/engine/grown/lattice'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

const [SHARD, NSHARDS] = [Number(process.argv[2] ?? 0), Number(process.argv[3] ?? 1)]
const SEEDS = [1, 2, 3]
const TRIALS = 1500
const ARMS: { name: string; seedEdges: number; seedSpanMax: number }[] = []
for (const density of [1000, 3000, 6000, 12000]) {
  ARMS.push({ name: `long-${density}`, seedEdges: density, seedSpanMax: 44 })
  ARMS.push({ name: `local-${density}`, seedEdges: density, seedSpanMax: 8 })
}

function run(arm: (typeof ARMS)[0], seed: number) {
  const org = new GrownOrganism({
    ...defaultGrownConfig,
    seed,
    seedEdges: arm.seedEdges,
    seedSpanMax: arm.seedSpanMax,
  })
  const hBirth = org.stats().inputHops
  const within2Birth = (hBirth[1] ?? 0) + (hBirth[2] ?? 0)
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(seed * 7919 + 1)
  let block: number[] = []
  const res: boolean[] = []
  const curve: number[] = []
  for (let t = 0; t < TRIALS; t++) {
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    const label = block.pop()!
    res.push(teacher.runTrial(org, M1_PATTERNS[label], label).correct)
    if ((t + 1) % 300 === 0) curve.push(res.slice(-100).filter(Boolean).length / 100)
  }
  const tail = res.slice(-100).filter(Boolean).length / 100
  const s = org.stats()
  const h = s.inputHops
  let toOut = 0
  for (let i = 0; i < org.edges.count; i++) {
    if (org.lattice.role[org.edges.post[i]] === ROLE_OUTPUT) toOut++
  }
  return { tail, curve, within2Birth, within2End: (h[1] ?? 0) + (h[2] ?? 0), edges: s.edges, toOut }
}

for (let i = 0; i < ARMS.length; i++) {
  if (i % NSHARDS !== SHARD) continue
  const arm = ARMS[i]
  const rs = SEEDS.map((s) => run(arm, s))
  const min = Math.min(...rs.map((r) => r.tail))
  console.log(
    `ARM ${arm.name.padEnd(12)} tails ${rs.map((r) => r.tail.toFixed(2)).join('/')}` +
      ` min ${min.toFixed(2)}  w2birth ${rs.map((r) => r.within2Birth).join('/')}` +
      ` w2end ${rs.map((r) => r.within2End).join('/')}` +
      ` edges ${rs.map((r) => r.edges).join('/')}  toOut ${rs.map((r) => r.toOut).join('/')}`
  )
  console.log(`    curves ${rs.map((r) => r.curve.map((c) => c.toFixed(2)).join('>')).join(' | ')}`)
}
console.log(`SHARD ${SHARD} DONE`)
