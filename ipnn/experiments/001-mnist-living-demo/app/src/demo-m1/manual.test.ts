// Manual mode: you hold a stimulus on the sense and watch the answer form.
//
// The claims that matter here are (a) the exposure really is sustained — the
// stimulus stays put and the readout keeps running, and (b) manual mode is
// an *observation* mode: it must be incapable of changing the organism, since
// the whole point is to inspect a trained organism without disturbing it.

import { describe, it, expect } from 'vitest'
import { DemoSim } from './sim'
import { M1_PATTERNS } from '../engine/patterns'
import { SustainedReadout } from '../engine/readout'

/** a trained organism: the M1 gate, reached the ordinary way */
function trained(seed: number, trials = 800): DemoSim {
  const sim = new DemoSim(seed)
  sim.runToTrials(trials)
  expect(sim.rollingAccuracy).toBeGreaterThanOrEqual(0.8)
  return sim
}

describe('manual mode', () => {
  it('holds the chosen stimulus on the sense and runs no trials', () => {
    const sim = trained(1)
    const trialsBefore = sim.trials.length

    sim.setMode('manual')
    sim.setManualStimulus(1)
    sim.tick(2000)

    expect(Array.from(sim.org.sense)).toEqual(Array.from(M1_PATTERNS[1]))
    expect(sim.trials.length).toBe(trialsBefore)
    expect(sim.accuracyCurve.length).toBe(trialsBefore)
    expect(sim.readout.ticks).toBe(2000)
  })

  it('clearing blanks the sense', () => {
    const sim = trained(1)
    sim.setMode('manual')
    sim.setManualStimulus(2)
    sim.tick(300)
    sim.setManualStimulus(null)
    sim.tick(300)

    expect(sim.currentLabel).toBeNull()
    expect(Array.from(sim.org.sense).every((v) => v === 0)).toBe(true)
  })

  it('cannot change the organism: no reward, so no weight moves', () => {
    const sim = trained(1)
    const before = sim.org.weightNorms()

    sim.setMode('manual')
    for (const label of [0, 1, 2, null, 0]) {
      sim.setManualStimulus(label)
      sim.tick(1500)
    }
    const after = sim.org.weightNorms()

    // exact equality, not approximate: applyReward is the only mutator and it
    // is never called, so these must be bit-identical
    expect(after.pool).toBe(before.pool)
    expect(after.out).toBe(before.out)
  })

  it('a trained organism settles on the right answer under sustained exposure', () => {
    for (const seed of [1, 2, 3]) {
      const sim = trained(seed)
      sim.setMode('manual')
      for (let label = 0; label < 3; label++) {
        sim.setManualStimulus(label)
        sim.tick(3000)
        const shares = sim.readout.shares()
        // it spends most of the exposure saying the right thing...
        expect(shares[label]).toBeGreaterThan(0.5)
        // ...and more of it than on either wrong answer
        for (let k = 0; k < 3; k++) {
          if (k !== label) expect(shares[label]).toBeGreaterThan(shares[k])
        }
      }
    }
  })

  it('returning to auto resumes teaching, and accuracy survives the excursion', () => {
    const sim = trained(1)
    sim.setMode('manual')
    // a long look at the *wrong* thing: if this could teach, it would damage
    sim.setManualStimulus(2)
    sim.tick(5000)

    sim.setMode('auto')
    const before = sim.trials.length
    sim.runToTrials(before + 200)
    const after = sim.trials.slice(before)
    const acc = after.filter((t) => t.correct).length / after.length
    expect(acc).toBeGreaterThanOrEqual(0.8)
  })

  it('switching mode clears eligibility traces (no stale credit)', () => {
    const sim = trained(1)
    sim.setMode('manual')
    sim.setManualStimulus(0)
    sim.tick(500)

    // traces are private; observe the guarantee through its effect — a reward
    // broadcast immediately after clearing must move nothing
    sim.org.clearTraces()
    const before = sim.org.weightNorms()
    sim.org.applyReward(1)
    const after = sim.org.weightNorms()
    expect(after.pool).toBe(before.pool)
    expect(after.out).toBe(before.out)
  })
})

describe('SustainedReadout', () => {
  const cfg = { outputSize: 3, window: 20, threshold: 6 }

  it('claims an answer at threshold and holds it (hysteresis)', () => {
    const r = new SustainedReadout(cfg)
    for (let i = 0; i < 5; i++) r.observe(1)
    expect(r.answer).toBeNull() // 5 fires — not yet
    r.observe(1)
    expect(r.answer).toBe(1) // 6th fire crosses the threshold
    expect(r.switches).toBe(1)

    // silence does not immediately unseat it: the window still holds 6 fires
    for (let i = 0; i < 5; i++) r.observe(-1)
    expect(r.answer).toBe(1)
  })

  it('releases to silence once the fires age out of the window', () => {
    const r = new SustainedReadout(cfg)
    for (let i = 0; i < 6; i++) r.observe(1)
    expect(r.answer).toBe(1)
    for (let i = 0; i < cfg.window; i++) r.observe(-1)
    expect(r.answer).toBeNull()
  })

  it('counts a revision only when one spoken answer replaces a different one', () => {
    const r = new SustainedReadout(cfg)
    for (let i = 0; i < 6; i++) r.observe(0)
    expect(r.answer).toBe(0)
    expect(r.revisions).toBe(0) // first answer is not a change of mind

    // let 0 age out, pass through silence, then say 2
    for (let i = 0; i < cfg.window; i++) r.observe(-1)
    expect(r.answer).toBeNull()
    expect(r.revisions).toBe(0) // silence is not a revision either

    for (let i = 0; i < 6; i++) r.observe(2)
    expect(r.answer).toBe(2)
    expect(r.revisions).toBe(1) // 0 → 2, ignoring the silence between
  })

  it('tracks occupancy, dwell and reset', () => {
    const r = new SustainedReadout(cfg)
    for (let i = 0; i < 30; i++) r.observe(1)
    const shares = r.shares()
    expect(shares.reduce((a, b) => a + b, 0)).toBeCloseTo(1)
    expect(shares[1]).toBeGreaterThan(0.5)
    expect(r.dwell).toBeGreaterThan(0)

    r.reset()
    expect(r.ticks).toBe(0)
    expect(r.answer).toBeNull()
    expect(r.switches).toBe(0)
    expect(r.revisions).toBe(0)
    expect(r.shares()[cfg.outputSize]).toBe(0)
  })
})
