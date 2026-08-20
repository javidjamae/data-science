// The demo must show the same organism the M1 gate certified. DemoSim uses
// the same seeds and the same schedule formula as m1-sanity.test.ts, driven
// through the tick-granular path a live renderer uses — so this asserts the
// M1 gate (and the frozen-retention claim) through the demo plumbing.

import { describe, it, expect } from 'vitest'
import { DemoSim } from './sim'

describe('DemoSim: the M1 gate through the demo path', () => {
  it('seed 1 reaches the gate tick-by-tick (acc ≥ 0.8, chance ≈ 0.33)', () => {
    const sim = new DemoSim(1)
    sim.runToTrials(800)
    expect(sim.rollingAccuracy).toBeGreaterThanOrEqual(0.8)
    // curve recorded once per trial, monotone bookkeeping intact
    expect(sim.accuracyCurve.length).toBe(800)
    expect(sim.trials.length).toBe(800)
  })

  it('learning toggle: accuracy holds after rewards stop', () => {
    const sim = new DemoSim(42)
    sim.runToTrials(500)
    sim.setLearning(false)
    const before = sim.trials.length
    sim.runToTrials(600)
    const frozen = sim.trials.slice(before)
    const acc = frozen.filter((t) => t.correct).length / frozen.length
    expect(acc).toBeGreaterThanOrEqual(0.8)
    expect(sim.marks).toEqual([{ trial: 500, learning: false }])
  })

  it('reset reproduces the identical trial sequence (determinism)', () => {
    const a = new DemoSim(3)
    a.runToTrials(120)
    const first = a.trials.map((t) => `${t.label}:${t.spoken}:${t.latency}`)
    a.reset(3)
    a.runToTrials(120)
    const second = a.trials.map((t) => `${t.label}:${t.spoken}:${t.latency}`)
    expect(second).toEqual(first)
  })
})
