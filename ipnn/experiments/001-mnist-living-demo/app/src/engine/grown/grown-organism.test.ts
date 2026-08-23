// M0 — the substrate as a whole.
//
// The point of M0 is that experiment 002 is a *swap*, not a fork: the same
// teacher, the same three patterns, the same 20-tick/6-fire readout, the same
// trial machinery, driving a completely different substrate. So the tests
// that matter most here are the ones that show the grown organism is
// interchangeable with 001's behind the nine-member interface.

import { describe, it, expect } from 'vitest'
import { GrownOrganism } from './grown-organism'
import { defaultGrownConfig, type GrownConfig } from './config'
import { Organism } from '../organism'
import { AutoTeacher } from '../teacher'
import { M1_PATTERNS } from '../patterns'
import { defaultConfig, defaultTeacherConfig, type OrganismLike } from '../types'
import { mulberry32, shuffleInPlace } from '../rng'

function organism(over: Partial<GrownConfig> = {}): GrownOrganism {
  return new GrownOrganism({ ...defaultGrownConfig, ...over })
}

describe('the OrganismLike contract', () => {
  it('is satisfied by both substrates, so the teacher cannot tell them apart', () => {
    // this is the M0 deliverable in one assertion: two substrates, one
    // contract, and every consumer written against the contract
    const substrates: OrganismLike[] = [
      new Organism({ ...defaultConfig }),
      organism(),
    ]
    for (const org of substrates) {
      expect(org.sense.length).toBe(64)
      expect(org.poolFired.length).toBe(org.cfg.poolSize)
      expect(org.cfg.outputSize).toBe(3)
      expect(typeof org.lastWinner).toBe('number')
      expect(typeof org.urge).toBe('number')
      org.tick()
      org.applyReward(0.5)
      org.clearTraces()
      expect(org.outputProbs().length).toBe(3)
      expect(org.poolActivity()).toBeGreaterThanOrEqual(0)
      expect(org.poolActivity()).toBeLessThanOrEqual(1)
      const norms = org.weightNorms()
      expect(Number.isFinite(norms.pool)).toBe(true)
      expect(Number.isFinite(norms.out)).toBe(true)
    }
  })

  it('exposes the lattice as the raster the UI draws', () => {
    const org = organism()
    expect(org.cfg.poolSize).toBe(org.cfg.width * org.cfg.height)
    expect(org.poolFired.length).toBe(1024)
  })
})

describe('determinism', () => {
  it('same seed, same life — down to the last edge', () => {
    // the engine may not use Math.random anywhere; experiments have to be
    // reproducible or the journal's re-run blocks are fiction
    const run = (seed: number) => {
      const org = organism({ seed, sleepEvery: 5 })
      const teacher = new AutoTeacher({ ...defaultTeacherConfig })
      for (let t = 0; t < 30; t++) {
        teacher.runTrial(org, M1_PATTERNS[t % 3], t % 3)
      }
      return {
        edges: org.edges.count,
        w: Array.from(org.edges.w.slice(0, 50)),
        post: Array.from(org.edges.post.slice(0, 50)),
        activity: org.poolActivity(),
        stats: org.stats(),
      }
    }
    const a = run(7)
    const b = run(7)
    const c = run(8)
    expect(a).toEqual(b)
    expect(c.w).not.toEqual(a.w)
  })
})

describe('living under the real teacher', () => {
  it('runs full trials, grows wiring, and answers', () => {
    const org = organism({ seed: 1 })
    const teacher = new AutoTeacher({ ...defaultTeacherConfig })
    const order = mulberry32(1)
    let block: number[] = []

    let spoke = 0
    for (let t = 0; t < 120; t++) {
      if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
      const label = block.pop()!
      const res = teacher.runTrial(org, M1_PATTERNS[label], label)
      if (res.spoken !== null) spoke++
    }

    const stats = org.stats()
    expect(stats.sleeps).toBeGreaterThan(0)
    expect(stats.edgesBorn).toBeGreaterThan(0)
    expect(stats.edges).toBeGreaterThan(0)
    // urge is the cold-start bootstrap: with nothing wired, silence has to
    // eventually force an answer, or reward can never arrive at all
    expect(spoke).toBeGreaterThan(0)
    expect(stats.ticksToFirstReward).not.toBeNull()
  })

  it('keeps its activity bounded — the stability telemetry open-problems §2 asks for', () => {
    const org = organism({ seed: 3 })
    const teacher = new AutoTeacher({ ...defaultTeacherConfig })
    for (let t = 0; t < 150; t++) teacher.runTrial(org, M1_PATTERNS[t % 3], t % 3)
    const stats = org.stats()
    expect(stats.meanActivity).toBeGreaterThan(0)
    expect(stats.meanActivity).toBeLessThan(0.6)
    expect(Number.isFinite(stats.activityVariance)).toBe(true)
    for (let s = 0; s < org.edges.count; s++) {
      expect(Number.isFinite(org.edges.w[s])).toBe(true)
      expect(Math.abs(org.edges.w[s])).toBeLessThanOrEqual(defaultGrownConfig.wMax)
    }
  })

  it('reports how far every sense pixel is from an answer, not just the best one', () => {
    // design §8's "input→output path length distribution". The shortest path
    // over the whole cortex is one pixel's story: a substrate can report
    // "1 hop" while 63 of its 64 pixels are four hops back, and reading that
    // as a shallow substrate is how a depth claim goes wrong.
    const org = organism({ seed: 9 })
    const teacher = new AutoTeacher({ ...defaultTeacherConfig })
    for (let t = 0; t < 120; t++) teacher.runTrial(org, M1_PATTERNS[t % 3], t % 3)

    const stats = org.stats()
    expect(stats.inputHops.reduce((a, b) => a + b, 0)).toBe(64)
    const reached = stats.inputHops.slice(1).reduce((a, b) => a + b, 0)
    expect(reached).toBeGreaterThan(0)
    // the shortest cortex-wide path must agree with the shallowest pixel
    const shallowest = stats.inputHops.findIndex((n, h) => h > 0 && n > 0)
    const best = Math.min(...stats.hops.filter((h): h is number => h !== null))
    expect(shallowest).toBe(best)
  })

  it('reports the eligibility horizon alongside path depth, as design §5 requires', () => {
    // λ and maximum path depth are coupled: if a path's delay exceeds the
    // trace horizon, credit silently never reaches its far end
    const org = organism({ seed: 5, latency: 'span' })
    const teacher = new AutoTeacher({ ...defaultTeacherConfig })
    for (let t = 0; t < 100; t++) teacher.runTrial(org, M1_PATTERNS[t % 3], t % 3)
    const stats = org.stats()
    expect(stats.traceHorizonTicks).toBeCloseTo(1 / (1 - defaultGrownConfig.traceDecay), 6)
    if (stats.deepestPathDelayTicks !== null) {
      expect(stats.deepestPathDelayTicks).toBeGreaterThan(0)
    }
  })
})

describe('the output cortex without a softmax', () => {
  it('calls two simultaneous outputs no answer at all, and counts it', () => {
    // design §10 risk 4: removing the softmax removes L-002's failure mode
    // and its fix, so output collapse has to be watched for directly rather
    // than papered over by picking whoever had more drive
    const org = organism({ lateralInhibition: 0, bias: 5, urgeMax: 0 })
    for (let t = 0; t < 50; t++) org.tick()
    const stats = org.stats()
    expect(stats.ambiguousTicks).toBeGreaterThan(0)
    expect(org.lastWinner).toBe(-1)
  })

  it('lateral inhibition is what makes a single answer possible', () => {
    const ambiguousFraction = (lateralInhibition: number): number => {
      const org = organism({ lateralInhibition, bias: 2, urgeMax: 0, seed: 11 })
      for (let t = 0; t < 400; t++) org.tick()
      const s = org.stats()
      return s.ambiguousTicks / Math.max(1, s.ambiguousTicks + s.spokenTicks)
    }
    expect(ambiguousFraction(4)).toBeLessThan(ambiguousFraction(0))
  })

  it('urge rises during silence and resets when an output fires', () => {
    const org = organism({ bias: -30, pSpont: 0 })
    for (let t = 0; t < 10; t++) org.tick()
    expect(org.urge).toBeCloseTo(10 * defaultGrownConfig.urgeRate, 6)
    for (let t = 0; t < 1000; t++) org.tick()
    expect(org.urge).toBeLessThanOrEqual(defaultGrownConfig.urgeMax)
  })

  it('reports a firing rate rather than a policy probability', () => {
    // no softmax to report; the UI's bars keep their meaning (design §9)
    const org = organism({ seed: 2 })
    for (let t = 0; t < 200; t++) org.tick()
    const probs = org.outputProbs()
    for (const p of probs) {
      expect(p).toBeGreaterThanOrEqual(0)
      expect(p).toBeLessThanOrEqual(1)
    }
  })
})

describe('traces, weights and what survives what', () => {
  it('clearTraces forgets what just happened, not what was learned', () => {
    const org = organism({ sleepEvery: 1_000_000 })
    for (let t = 0; t < 400; t++) org.tick()
    org.sleep()
    org.edges.e.fill(0.5)
    const before = org.weightNorms()
    org.clearTraces()
    for (let s = 0; s < org.edges.count; s++) expect(org.edges.e[s]).toBe(0)
    expect(org.weightNorms()).toEqual(before)
  })

  it('reward with a zero advantage changes nothing at all', () => {
    const org = organism({ sleepEvery: 1_000_000 })
    for (let t = 0; t < 400; t++) org.tick()
    org.sleep()
    org.edges.e.fill(1)
    const before = Array.from(org.edges.w)
    org.applyReward(0)
    expect(Array.from(org.edges.w)).toEqual(before)
  })

  it('splits weightNorms the way 001 does: the readout layer, and everything else', () => {
    const org = organism({ sleepEvery: 1_000_000 })
    for (let t = 0; t < 600; t++) org.tick()
    org.sleep()
    const norms = org.weightNorms()
    expect(norms.pool).toBeGreaterThan(0)
    // `out` is every edge terminating in the output cortex — 001's 480
    // readout weights, grown rather than given
    expect(norms.out).toBeGreaterThanOrEqual(0)
    let byHand = 0
    for (let s = 0; s < org.edges.count; s++) {
      if (org.lattice.outputNodes.includes(org.edges.post[s])) {
        byHand += org.edges.w[s] * org.edges.w[s]
      }
    }
    expect(norms.out).toBeCloseTo(byHand, 4)
  })
})
