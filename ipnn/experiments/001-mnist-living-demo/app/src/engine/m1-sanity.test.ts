// M1 — the critical gate (design.md §6).
// Can pure reward-modulated local learning take 3 distinct patterns to
// sustained above-chance accuracy, with no backprop and no labels — only a
// broadcast reward scalar? If this fails, everything stops until the
// learning rule works.

import { describe, it, expect } from 'vitest'
import { Organism } from './organism'
import { AutoTeacher } from './teacher'
import { M1_PATTERNS } from './patterns'
import { defaultConfig, defaultTeacherConfig } from './types'
import { mulberry32, shuffleInPlace } from './rng'

const TRIALS = 800
const TAIL = 100
const GATE = 0.8

// ── The recorded record, asserted ─────────────────────────────────────────
// These are the exact curves journal entry 2026-08-16-0248 recorded and that
// every later entry compares against. Asserting them (not just the ≥0.80
// gate) is what makes "old experiments keep working at HEAD" an enforced
// invariant instead of a habit: any change that nudges the default
// organism's behaviour by one hundredth on one block fails the suite.
//
// If this ever fails with NO code diff — e.g. right after a Node/V8 upgrade —
// the cause is floating-point drift in the engine (Math.exp is not
// spec-pinned), not a regression. Record it in the journal and re-pin.
const RECORDED_CURVES: Record<number, string> = {
  1: '0.53 0.80 0.91 0.98 0.94 0.95 0.97 0.98',
  2: '0.54 0.80 0.87 0.94 0.96 0.95 0.97 0.98',
  3: '0.42 0.74 0.85 0.89 0.95 0.96 0.98 0.99',
}
const RECORDED_FROZEN = 0.97

function runSeed(seed: number): { tailAccuracy: number; curve: number[] } {
  const org = new Organism({ ...defaultConfig, seed })
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(seed * 7919 + 1)

  const results: boolean[] = []
  const curve: number[] = []
  let block: number[] = []

  for (let t = 0; t < TRIALS; t++) {
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    const label = block.pop()!
    const res = teacher.runTrial(org, M1_PATTERNS[label], label)
    results.push(res.correct)
    if ((t + 1) % 100 === 0) {
      const w = results.slice(-100)
      curve.push(w.filter(Boolean).length / w.length)
    }
  }

  const tail = results.slice(-TAIL)
  return { tailAccuracy: tail.filter(Boolean).length / TAIL, curve }
}

describe('M1 sanity gate: 3 patterns, reward-only learning', () => {
  for (const seed of [1, 2, 3]) {
    it(`seed ${seed}: last-${TAIL} accuracy ≥ ${GATE} (chance ≈ 0.33)`, () => {
      const { tailAccuracy, curve } = runSeed(seed)
      console.log(
        `seed ${seed} rolling accuracy per 100 trials: ${curve
          .map((a) => a.toFixed(2))
          .join(' → ')}`
      )
      expect(tailAccuracy).toBeGreaterThanOrEqual(GATE)
      // bit-identical to the record, not merely above the gate
      expect(curve.map((a) => a.toFixed(2)).join(' ')).toBe(RECORDED_CURVES[seed])
    })
  }

  it('keeps performing after learning is switched off (the living-model claim)', () => {
    const org = new Organism({ ...defaultConfig, seed: 42 })
    const teacher = new AutoTeacher({ ...defaultTeacherConfig })
    const order = mulberry32(42 * 7919 + 1)

    let block: number[] = []
    for (let t = 0; t < 500; t++) {
      if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
      const label = block.pop()!
      teacher.runTrial(org, M1_PATTERNS[label], label)
    }

    // rewards stop; the organism keeps living and answering
    teacher.learning = false
    let correct = 0
    for (let t = 0; t < 100; t++) {
      if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
      const label = block.pop()!
      if (teacher.runTrial(org, M1_PATTERNS[label], label).correct) correct++
    }
    console.log(`frozen accuracy over 100 unrewarded trials: ${correct / 100}`)
    expect(correct / 100).toBe(RECORDED_FROZEN)
  })
}, 180_000)
