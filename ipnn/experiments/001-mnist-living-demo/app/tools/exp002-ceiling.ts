// What holds the shallow arm at ~0.87 when 001 reached 0.98?
//
// It is not silence: late in the run only 2.6% of trials go unanswered, and
// accuracy *given* it answered is 0.892. It is getting roughly one committed
// answer in nine genuinely wrong, and never getting better.
//
// Three suspects, each with a pre-registered or design-registered arm:
//
//  1. CONSOLIDATION — design §7's "carried-forward defect, deliberately not
//     fixed": the evidence count `n` only ever increases, so plasticity
//     1/(1+n/n₀) decays monotonically and a synapse can never become *less*
//     confident. That predicts exactly this shape: fast early learning, then
//     a hard freeze at whatever error rate happened to exist. It is L-004's
//     confidence-plasticity tension, and "consolidated memory" and "frozen
//     wrong answer" being one mechanism is the whole point of L-004.
//
//  2. RENT — a pre-registered control arm (ρ=0). Rent is charged flat per
//     tick, but consolidation slows *learning* as evidence accumulates.
//     Nothing slows the decay. A well-learned edge therefore keeps paying
//     full price while its ability to be repaid shrinks — an asymmetry that
//     would erode competence continuously.
//
//  3. CHURN — the edge population still swings ~30% between sleeps this late
//     in the run. The readout is being partly rebuilt underneath itself
//     forever, which would cap the ceiling and produce the fluctuation.
//     Isolated here by stopping structural change once competence exists.
//
//   npx vite-node tools/exp002-ceiling.ts [trials] [seeds...]

import { GrownOrganism } from '../src/engine/grown/grown-organism'
import { defaultGrownConfig, type GrownConfig } from '../src/engine/grown/config'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

const argv = process.argv.slice(2)
const TRIALS = argv.length > 0 ? Number(argv[0]) : 2500
const SEEDS = argv.length > 1 ? argv.slice(1).map(Number) : [1, 2, 3]
const TAIL = 400
/** trial at which the "frozen structure" arm stops rewiring */
const FREEZE_AT = 1200

const SHALLOW: Partial<GrownConfig> = { outputX: 14 }

const ARMS: { name: string; cfg: Partial<GrownConfig>; freeze?: boolean }[] = [
  { name: 'shallow (baseline)', cfg: {} },
  { name: 'consolidation off  ', cfg: { consolidation: false } },
  { name: 'no rent (ρ=0)      ', cfg: { rent: 0 } },
  { name: 'structure frozen   ', cfg: {}, freeze: true },
]

function run(seed: number, cfg: Partial<GrownConfig>, freeze: boolean) {
  const org = new GrownOrganism({ ...defaultGrownConfig, ...SHALLOW, ...cfg, seed })
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(seed * 7919 + 1)

  const rows: { correct: boolean; spoken: number | null }[] = []
  let block: number[] = []

  for (let t = 0; t < TRIALS; t++) {
    // freezing means "no more growth and no more death": rewiring stops, the
    // readout is left to settle on the structure it has
    if (freeze && t === FREEZE_AT) org.cfg.sleepEvery = Number.MAX_SAFE_INTEGER
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    const label = block.pop()!
    const r = teacher.runTrial(org, M1_PATTERNS[label], label)
    rows.push({ correct: r.correct, spoken: r.spoken })
  }

  const tail = rows.slice(-TAIL)
  const spoke = tail.filter((r) => r.spoken !== null)
  return {
    tail: tail.filter((r) => r.correct).length / tail.length,
    silence: 1 - spoke.length / tail.length,
    conditional: spoke.length ? spoke.filter((r) => r.correct).length / spoke.length : NaN,
    edges: org.edges.count,
  }
}

console.log(
  `shallow arm ceiling — ${TRIALS} trials, tail = last ${TAIL}, ` +
    `seeds ${SEEDS.join(',')}, structure frozen at trial ${FREEZE_AT}\n`
)
console.log('arm                   |  tail  | silent | right when it spoke | edges')
console.log('----------------------|--------|--------|---------------------|-------')

for (const arm of ARMS) {
  const rs = SEEDS.map((s) => run(s, arm.cfg, arm.freeze ?? false))
  const m = (f: (r: (typeof rs)[0]) => number) => rs.reduce((a, r) => a + f(r), 0) / rs.length
  console.log(
    `${arm.name} | ${m((r) => r.tail).toFixed(3)}  | ${(m((r) => r.silence) * 100)
      .toFixed(1)
      .padStart(5)}% |        ${m((r) => r.conditional).toFixed(3)}        | ${Math.round(
      m((r) => r.edges)
    ).toLocaleString()}`
  )
}
