// Latent learning — Tolman & Honzik (1930), applied to an IPNN organism.
//
// THE QUESTION IT ANSWERS. "How do you know it isn't learning invisibly?" is
// unfalsifiable as stated, and it is also the single most important objection
// to any negative result from a living system. Tolman answered the same
// objection in rats in 1930 and the method transfers unchanged, which is
// exactly why it is admissible here (H-009): it never once refers to weights,
// losses, epochs or gradients. It asks only about behaviour in time.
//
// THE PROTOCOL.
//   pre-exposed : N trials with the patterns presented and reward SWITCHED
//                 OFF, then reward on, then measure trials-to-criterion.
//   naive       : no pre-exposure. Same measurement.
//   scrambled   : N trials of pre-exposure to the SAME number of stimuli in a
//                 shuffled order with no reward — a control for "it just needs
//                 to have been running a while" as distinct from "it learned
//                 something about these patterns". Without this arm, any
//                 speed-up could be mere warm-up.
//
// THE READING. Tolman's rats wandered a maze unrewarded for ten days and
// looked exactly like non-learners. When food appeared they matched the
// rewarded group almost immediately — they had been building a map the whole
// time with nothing to show for it.
//
//   pre-exposed reaches criterion FASTER than naive  → it was learning
//       during a period when behaviour showed nothing. H-002 supported.
//   pre-exposed reaches criterion NO faster          → nothing was acquired.
//       The flat curve was flat all the way down.
//
// Both outcomes are informative, which is the property H-008 demands and the
// reason this is worth running rather than arguing about.
//
// Measured in TRIALS-TO-CRITERION (H-011), not accuracy: it is the currency
// comparative psychology uses precisely because it stays comparable across
// learners of different speeds, and it does not punish a slow learner for
// being slow.
//
//   npx vite-node tools/latent-learning.ts [substrate] [preTrials] [seeds...]
//     substrate: 001 | shallow | m1

import { Organism } from '../src/engine/organism'
import { GrownOrganism } from '../src/engine/grown/grown-organism'
import { defaultGrownConfig } from '../src/engine/grown/config'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultConfig, defaultTeacherConfig, type OrganismLike } from '../src/engine/types'
import { mulberry32, shuffleInPlace } from '../src/engine/rng'

const argv = process.argv.slice(2)
const SUBSTRATE = (argv[0] ?? '001') as '001' | 'shallow' | 'm1'
const PRE = Number(argv[1] ?? 400)
const SEEDS = argv.length > 2 ? argv.slice(2).map(Number) : [1, 2, 3, 4, 5]
const MAX_TRIALS = 3000
const CRITERION = 0.7
const WINDOW = 100

function build(seed: number): OrganismLike {
  if (SUBSTRATE === '001') return new Organism({ ...defaultConfig, seed })
  return new GrownOrganism({
    ...defaultGrownConfig,
    ...(SUBSTRATE === 'shallow' ? { outputX: 14 } : {}),
    seed,
  })
}

type Phase = 'none' | 'patterns' | 'scrambled'

/**
 * One organism's life: an unrewarded pre-exposure phase, then a rewarded
 * phase measured in trials-to-criterion.
 *
 * The pre-exposure phase is genuinely unrewarded — `teacher.learning = false`
 * means `applyReward` is never reached, so no weight can move from reward.
 * Anything the organism gains during it must come from living, not teaching.
 */
function run(seed: number, phase: Phase) {
  const org = build(seed)
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(seed * 7919 + 1)
  let block: number[] = []
  const next = () => {
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    return block.pop()!
  }

  if (phase !== 'none') {
    teacher.learning = false
    for (let t = 0; t < PRE; t++) {
      const label = next()
      // 'scrambled' shows the same stimuli but pairs them with a randomised
      // "label" the teacher judges against — the organism sees the same
      // amount of the world, with no stable structure to acquire
      const shown = phase === 'patterns' ? label : Math.floor(order() * 3)
      teacher.runTrial(org, M1_PATTERNS[shown], label)
    }
    teacher.learning = true
    org.clearTraces()
  }

  const results: boolean[] = []
  let reached: number | null = null
  for (let t = 0; t < MAX_TRIALS; t++) {
    const label = next()
    results.push(teacher.runTrial(org, M1_PATTERNS[label], label).correct)
    if (reached === null && results.length >= WINDOW) {
      const w = results.slice(-WINDOW)
      if (w.filter(Boolean).length / WINDOW >= CRITERION) reached = t + 1
    }
  }
  const tail = results.slice(-WINDOW)
  return { reached, final: tail.filter(Boolean).length / WINDOW }
}

const ARMS: { name: string; phase: Phase }[] = [
  { name: 'naive (no pre-exposure)', phase: 'none' },
  { name: `pre-exposed, unrewarded`, phase: 'patterns' },
  { name: `scrambled control      `, phase: 'scrambled' },
]

console.log(
  `Latent learning — substrate ${SUBSTRATE}, ${PRE} unrewarded pre-exposure trials, ` +
    `criterion = rolling-${WINDOW} ≥ ${CRITERION}, cap ${MAX_TRIALS}\n`
)
console.log('arm                     | trials to criterion (per seed)      | median | final acc')
console.log('------------------------|-------------------------------------|--------|----------')

const table: Record<string, (number | null)[]> = {}
for (const arm of ARMS) {
  const rs = SEEDS.map((s) => run(s, arm.phase))
  table[arm.name] = rs.map((r) => r.reached)
  const got = rs.map((r) => r.reached).filter((x): x is number => x !== null)
  const median = got.length
    ? [...got].sort((a, b) => a - b)[Math.floor(got.length / 2)]
    : null
  const finals = rs.reduce((a, r) => a + r.final, 0) / rs.length
  console.log(
    `${arm.name} | ${rs
      .map((r) => (r.reached === null ? ' none' : String(r.reached).padStart(5)))
      .join(' ')} | ${median === null ? ' none ' : String(median).padStart(6)} |   ${finals.toFixed(3)}`
  )
}

console.log()
const naive = table['naive (no pre-exposure)']
const pre = table['pre-exposed, unrewarded']
const both = SEEDS.map((_, i) => [naive[i], pre[i]] as const).filter(
  ([a, b]) => a !== null && b !== null
)
if (both.length === 0) {
  console.log(
    'Neither arm reached criterion on any seed, so trials-to-criterion is undefined.'
  )
  console.log(
    'Read the final-accuracy column instead — and note that a substrate which never'
  )
  console.log('reaches criterion cannot demonstrate latent learning by this method at all.')
} else {
  const diffs = both.map(([a, b]) => a! - b!)
  const mean = diffs.reduce((x, y) => x + y, 0) / diffs.length
  const faster = diffs.filter((d) => d > 0).length
  console.log(
    `Pre-exposure changed trials-to-criterion by ${mean >= 0 ? '−' : '+'}${Math.abs(mean).toFixed(
      0
    )} trials on average (faster on ${faster}/${both.length} seeds).`
  )
  console.log(
    diffs.every((d) => d > 0)
      ? 'Faster on EVERY seed: evidence of learning during a period when reward never arrived.'
      : 'Not consistent across seeds — no latent-learning effect at this pre-exposure length.'
  )
}
