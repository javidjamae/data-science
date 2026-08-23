// Serial reversal — does it get better at *changing its mind*?
//
// THE IDEA (Javid, 2026-08-22) [J]. A reversal separates two things that are
// easy to confuse: what the organism has worked out the stimuli *mean*, and the
// rules for what to *do* with them. If those are genuinely separate inside the
// system, a new rule should be cheap — you already recognise the picture, you
// just re-point the output. If they are not separate, every rule change is a
// full rebuild.
//
// So **how fast a system reverses is a probe for whether it has factored
// perception from policy** — and it needs no access to the internals at all,
// which is why it can be run on an animal (H-009).
//
// This is serial reversal learning, a classic comparative measure: flip the
// rule, let them learn it, flip it back, repeat. Animals with real
// representational machinery get faster at each successive flip — they stop
// rebuilding from scratch. Fish improve a little; primates improve a lot.
//
// ── PRE-REGISTERED, written before running (L-031) ────────────────────────
//
// PREDICTION: **no improvement across successive reversals.** L-010 showed
// 001's middle layer is a fixed random projection that learns nothing — 89% of
// its synapses can be frozen at no cost — so all learning lives in the 480
// readout weights. There is no representation held separately from the mapping,
// therefore nothing to reuse, therefore every reversal should cost about the
// same. L-030 already showed reversal 1 costs 7.7× the original learning.
//
// DECISION RULE: compare the mean of reversals 6–8 against reversals 1–3.
//   ≥25% faster on ≥3 seeds  → improvement: something IS being reused, and
//                              L-010's reading of this substrate is incomplete.
//   otherwise                → flat: confirms no perception/policy split.
//
// A flat result is the expected one and is not a failure — it makes reversal
// speed a *calibrated* instrument, with 001 as the known-negative reference for
// any future substrate that claims to have a representation.
//
//   npx vite-node tools/exp003-serial-reversal.ts [seeds...]

import { Organism } from '../src/engine/organism'
import { AutoTeacher } from '../src/engine/teacher'
import { TASK_A_PATTERNS } from '../src/engine/patterns'
import { defaultConfig, defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

const SEEDS = process.argv.slice(2).length
  ? process.argv.slice(2).map(Number)
  : [1, 2, 3, 4]
const CRITERION = 0.85
const WINDOW = 100
const MAX_TRIALS = 2500
const REVERSALS = 8

/** the two mappings the organism alternates between */
const RULE_A = [0, 1, 2]
const RULE_B = [1, 2, 0]

function run(seed: number): (number | null)[] {
  const org = new Organism({ ...defaultConfig, seed })
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(seed * 7919 + 1)
  let block: number[] = []
  const next = () => {
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    return block.pop()!
  }

  const stage = (rule: number[]): number | null => {
    const res: boolean[] = []
    for (let t = 0; t < MAX_TRIALS; t++) {
      const i = next()
      res.push(teacher.runTrial(org, TASK_A_PATTERNS[i], rule[i]).correct)
      if (res.length >= WINDOW) {
        const w = res.slice(-WINDOW)
        if (w.filter(Boolean).length / WINDOW >= CRITERION) return t + 1
      }
    }
    return null
  }

  // stage 0 is original acquisition; stages 1..REVERSALS alternate the rule
  const out: (number | null)[] = [stage(RULE_A)]
  for (let r = 1; r <= REVERSALS; r++) out.push(stage(r % 2 === 1 ? RULE_B : RULE_A))
  return out
}

const rows = SEEDS.map((s) => ({ seed: s, stages: run(s) }))
const fmt = (v: number | null) => (v === null ? ' none' : String(v).padStart(5))

console.log(
  `Serial reversal — ${REVERSALS} rule flips after acquisition, ` +
    `criterion rolling-${WINDOW} ≥ ${CRITERION}, cap ${MAX_TRIALS}\n` +
    `alternating ${JSON.stringify(RULE_A)} ↔ ${JSON.stringify(RULE_B)} on the same three stimuli\n`
)
console.log(
  'seed | acquire |' + Array.from({ length: REVERSALS }, (_, i) => ` rev${i + 1}`).join(' |')
)
console.log('-----|---------|' + '------|'.repeat(REVERSALS))
for (const r of rows) {
  console.log(
    `  ${r.seed}  |  ${fmt(r.stages[0])}  |` +
      r.stages
        .slice(1)
        .map((v) => fmt(v))
        .join('|')
  )
}

const mean = (x: number[]) => (x.length ? x.reduce((a, b) => a + b, 0) / x.length : NaN)
/** a stage that never reached criterion counts as the cap — the conservative
 * choice, since treating it as missing would flatter a failing arm */
const val = (v: number | null) => v ?? MAX_TRIALS

console.log('\n══ is it getting better at changing its mind? ═════════════════')
let improvedSeeds = 0
for (const r of rows) {
  const early = mean(r.stages.slice(1, 4).map(val))
  const late = mean(r.stages.slice(REVERSALS - 2).map(val))
  const gain = (early - late) / early
  if (gain >= 0.25) improvedSeeds++
  console.log(
    `  seed ${r.seed}: reversals 1–3 mean ${early.toFixed(0)} → last three ${late.toFixed(0)}  ` +
      `(${gain >= 0 ? '−' : '+'}${Math.abs(gain * 100).toFixed(0)}%)`
  )
}
const acq = mean(rows.map((r) => val(r.stages[0])))
const rev1 = mean(rows.map((r) => val(r.stages[1])))
console.log(
  `\n  original acquisition ${acq.toFixed(0)} trials · first reversal ${rev1.toFixed(0)} ` +
    `(${(rev1 / acq).toFixed(1)}× the cost of learning it fresh)`
)
console.log(
  `  gate — ≥25% faster on ≥3 seeds: ${improvedSeeds >= 3 ? 'IMPROVEMENT' : 'FLAT'} ` +
    `(${improvedSeeds}/${rows.length} seeds improved)`
)
console.log(
  improvedSeeds >= 3
    ? '\n  Something is being reused across rule changes — L-010 understates this substrate.'
    : '\n  No reuse across rule changes: each flip is a full rebuild, which is what a\n' +
      '  substrate with no representation held apart from its mapping should do.\n' +
      '  001 is now the calibrated known-negative for this instrument.'
)
