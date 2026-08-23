// 002-M1g — the economy fixes, against the scaffold they exist to save.
//
// Sub-experiment of 002's M1, testing two registered hypotheses:
//   H-023 earned durability — rent/(1 + n/rentN0): proven edges become cheap
//         to keep (use-dependent stabilization; the one-line form of
//         wire/weight lifetime separation)
//   H-022 juvenile grace — no rent, no death for an edge's first
//         `graceSleeps` rewirings: a fair audition (the smallest form of a
//         developmental schedule)
//
// Parent result: 2026-08-22-2103 (L-041/L-042) — the innate scaffold learns
// from birth and is then taxed to death: w2 20–29 → ~0, curves 0.50 → 0.39.
//
// ── PRE-REGISTERED, committed before first run ────────────────────────────
// ARMS on the long-12000 scaffold, seeds 1–3 × 2,500 trials:
//   off        the recorded economy (also a live REPRODUCTION CHECK of the
//              2103 entry's long-12000 row — must match at 1,500 trials)
//   rent-only  rentN0=25
//   grace-only graceSleeps=8
//   both       rentN0=25, graceSleeps=8
// MEASURES/GATES:
//   RETENTION  w2end ≥ 50% of w2birth on the `both` arm (off arm: →~0)
//   TRAJECTORY last 100-block ≥ first 100-block on `both` (off arm: decays)
//   STRETCH    the M1 gate itself (tail ≥ 0.8, all seeds)
// PREDICTION [C]: both retains structure and stops the decay; rent-only
// partial; grace-only insufficient once grace expires. Full gate uncertain.
// DECISION RULE: retention+trajectory pass → economy fixed → the
// kill-and-mutate scaffold loop (H-014) is next. Retention passes but
// trajectory flat → structure kept but unexploited → credit/readout is the
// next lever, not economy. Still decays → H-023's full tract/weight
// separation, not the one-line form.
//
//   npx vite-node tools/exp002-economy.ts <shard> <nshards>

import { GrownOrganism } from '../src/engine/grown/grown-organism'
import { defaultGrownConfig, type GrownConfig } from '../src/engine/grown/config'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

const [SHARD, NSHARDS] = [Number(process.argv[2] ?? 0), Number(process.argv[3] ?? 1)]
const SEEDS = [1, 2, 3]
const TRIALS = 2500
const BASE: Partial<GrownConfig> = { seedEdges: 12000, seedSpanMax: 44 }
const ARMS: { name: string; over: Partial<GrownConfig> }[] = [
  { name: 'off', over: {} },
  { name: 'rent-only', over: { rentN0: 25 } },
  { name: 'grace-only', over: { graceSleeps: 8 } },
  { name: 'both', over: { rentN0: 25, graceSleeps: 8 } },
]

function run(over: Partial<GrownConfig>, seed: number) {
  const org = new GrownOrganism({ ...defaultGrownConfig, ...BASE, ...over, seed })
  const w2 = () => {
    const h = org.stats().inputHops
    return (h[1] ?? 0) + (h[2] ?? 0)
  }
  const w2birth = w2()
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(seed * 7919 + 1)
  let block: number[] = []
  const res: boolean[] = []
  const curve: number[] = []
  let w2at1500 = -1
  for (let t = 0; t < TRIALS; t++) {
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    const label = block.pop()!
    res.push(teacher.runTrial(org, M1_PATTERNS[label], label).correct)
    if ((t + 1) % 250 === 0) curve.push(res.slice(-100).filter(Boolean).length / 100)
    if (t + 1 === 1500) w2at1500 = w2()
  }
  const tail = res.slice(-100).filter(Boolean).length / 100
  return { tail, curve, w2birth, w2at1500, w2end: w2(), edges: org.stats().edges }
}

for (let i = 0; i < ARMS.length; i++) {
  if (i % NSHARDS !== SHARD) continue
  const arm = ARMS[i]
  const rs = SEEDS.map((s) => run(arm.over, s))
  console.log(
    `ARM ${arm.name.padEnd(10)} tails ${rs.map((r) => r.tail.toFixed(2)).join('/')}` +
      `  w2 birth ${rs.map((r) => r.w2birth).join('/')}` +
      ` @1500 ${rs.map((r) => r.w2at1500).join('/')}` +
      ` end ${rs.map((r) => r.w2end).join('/')}` +
      `  edges ${rs.map((r) => r.edges).join('/')}`
  )
  for (const [k, r] of rs.entries()) {
    console.log(`    seed${SEEDS[k]} curve ${r.curve.map((c) => c.toFixed(2)).join('>')}`)
  }
}
console.log(`SHARD ${SHARD} DONE`)
