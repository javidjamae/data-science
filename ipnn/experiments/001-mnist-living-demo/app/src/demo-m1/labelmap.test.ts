// The rule flip (experiment 003's reversal), at the DemoSim level.
//
// The default is the identity map, and the identity map must be invisible —
// the recorded demos and the pinned M1 curves depend on it.

import { describe, it, expect } from 'vitest'
import { DemoSim } from './sim'
import { M1_PATTERNS } from '../engine/patterns'

const findShownPattern = (sim: DemoSim): number => {
  for (let i = 0; i < 3; i++) {
    if (M1_PATTERNS[i].every((v, j) => sim.org.sense[j] === v)) return i
  }
  return -1
}

describe('the rule flip', () => {
  it('defaults to identity and resets to identity', () => {
    const sim = new DemoSim(1)
    expect(sim.labelMap).toEqual([0, 1, 2])
    sim.setLabelMap([1, 2, 0])
    sim.reset(1)
    expect(sim.labelMap).toEqual([0, 1, 2])
  })

  it('rewards the mapped label for the shown stimulus, from the next trial on', () => {
    const sim = new DemoSim(3)
    sim.setLabelMap([1, 2, 0])
    // the trial in flight predates the flip and keeps its judgment — a flip
    // never retroactively re-judges the stimulus already on the sense. Check
    // the pairing on trials begun after the flip.
    for (let t = 1; t <= 5; t++) {
      sim.runToTrials(t)
      sim.tick(1) // into the next trial's stimulus phase
      const shown = findShownPattern(sim)
      expect(shown).toBeGreaterThanOrEqual(0)
      expect(sim.stepper.label).toBe([1, 2, 0][shown])
    }
  })

  it('collapses a trained organism, which then relearns the new rule', () => {
    const sim = new DemoSim(1)
    sim.runToTrials(400)
    expect(sim.rollingAccuracy).toBeGreaterThan(0.8)
    sim.setLabelMap([1, 2, 0])
    const at = sim.trials.length
    sim.runToTrials(at + 150)
    // under the new rule the old answers are wrong: accuracy must crash
    // through chance on its way down (L-030's below-chance signature)
    expect(sim.rollingAccuracy).toBeLessThan(0.4)
  })
})
