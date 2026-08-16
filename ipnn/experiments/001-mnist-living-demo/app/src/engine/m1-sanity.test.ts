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
    expect(correct / 100).toBeGreaterThanOrEqual(GATE)
  })
}, 180_000)
