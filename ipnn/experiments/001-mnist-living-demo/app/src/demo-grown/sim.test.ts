// The experiment 002 demo path, headless.
//
// The claim under test is M0's, restated where it is easiest to break: the
// demo controller written for experiment 001 drives experiment 002's substrate
// with no modification. If that stops being true, the "one codebase, two
// substrates" arrangement has quietly become a fork.

import { describe, it, expect } from 'vitest'
import { DemoSim } from '../demo-m1/sim'
import { grownSim, grownOrganism, GROWN_ARMS, grownConfig } from './sim'
import { GrownOrganism } from '../engine/grown/grown-organism'
import { Organism } from '../engine/organism'

describe('the demo controller is substrate-agnostic', () => {
  it('defaults to experiment 001, so the published demo is untouched', () => {
    const sim = new DemoSim(1)
    expect(sim.org).toBeInstanceOf(Organism)
    expect(sim.org.cfg.poolSize).toBe(160)
  })

  it('drives the grown substrate through the same controller', () => {
    const sim = grownSim(1, 'm1')
    expect(sim.org).toBeInstanceOf(GrownOrganism)
    expect(sim.org.cfg.poolSize).toBe(1024)

    sim.runToTrials(20)
    expect(sim.trials.length).toBe(20)
    expect(sim.accuracyCurve.length).toBe(20)
    expect(sim.rollingAccuracy).toBeGreaterThanOrEqual(0)
    expect(sim.rollingAccuracy).toBeLessThanOrEqual(1)
    // the raster the renderer draws is the lattice
    expect(sim.poolAccum.length).toBe(1024)
    expect(sim.totalTicks).toBeGreaterThan(0)
  })

  it('grows structure while the demo runs, and reports it', () => {
    const sim = grownSim(2, 'm1')
    sim.runToTrials(60)
    const org = grownOrganism(sim)
    const stats = org.stats()
    expect(stats.sleeps).toBeGreaterThan(0)
    expect(stats.edges).toBeGreaterThan(0)
    expect(stats.inputHops.reduce((a, b) => a + b, 0)).toBe(64)
  })

  it('manual mode still cannot change a grown organism', () => {
    // the guarantee 001's manual mode carries has to survive the swap: no
    // teacher, therefore no reward, therefore no weight may move
    const sim = grownSim(3, 'm1')
    sim.runToTrials(40)
    sim.setMode('manual')
    const before = sim.org.weightNorms()
    sim.setManualStimulus(1)
    sim.tick(400)
    const after = sim.org.weightNorms()
    // rent is not learning: it decays every edge every tick regardless, so
    // norms may fall. What must not happen is a weight *growing*.
    expect(after.pool).toBeLessThanOrEqual(before.pool + 1e-6)
    expect(after.out).toBeLessThanOrEqual(before.out + 1e-6)
    expect(sim.spoken === null || sim.spoken >= 0).toBe(true)
  })

  it('resets deterministically, arm by arm', () => {
    const run = (arm: 'm1' | 'shallow') => {
      const sim = grownSim(5, arm)
      sim.runToTrials(30)
      return {
        trials: sim.trials.map((t) => t.correct),
        edges: grownOrganism(sim).edges.count,
      }
    }
    expect(run('m1')).toEqual(run('m1'))
    expect(run('shallow')).not.toEqual(run('m1'))
  })
})

describe('the arms differ only where they claim to', () => {
  it('changes the output cortex position and nothing else', () => {
    const a = grownConfig('m1', 1)
    const b = grownConfig('shallow', 1)
    const differing = (Object.keys(a) as (keyof typeof a)[]).filter(
      (k) => JSON.stringify(a[k]) !== JSON.stringify(b[k])
    )
    expect(differing).toEqual(['outputX'])
  })

  it('puts more of the sense within reach of an answer in the shallow arm', () => {
    // the measurement the whole entry turns on: depth per sense pixel, not
    // the shortest path over the cortex
    const withinTwoHops = (arm: 'm1' | 'shallow'): number => {
      const sim = grownSim(1, arm)
      sim.runToTrials(200)
      const h = grownOrganism(sim).stats().inputHops
      return (h[1] ?? 0) + (h[2] ?? 0)
    }
    expect(withinTwoHops('m1')).toBe(0)
    expect(withinTwoHops('shallow')).toBeGreaterThan(0)
  })

  it('describes both arms for the UI', () => {
    for (const spec of Object.values(GROWN_ARMS)) {
      expect(spec.label.length).toBeGreaterThan(0)
      expect(spec.note.length).toBeGreaterThan(0)
    }
  })
})
