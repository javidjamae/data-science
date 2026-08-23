// The two economy fixes (L-042's remedy): earned durability and juvenile
// grace. Both default OFF — the recorded organism is untouched.

import { describe, it, expect } from 'vitest'
import { GrownOrganism } from './grown-organism'
import { defaultGrownConfig } from './config'

describe('defaults', () => {
  it('both fixes are off in the recorded configuration', () => {
    expect(defaultGrownConfig.rentN0).toBe(0)
    expect(defaultGrownConfig.graceSleeps).toBe(0)
  })
})

describe('earned durability (rentN0)', () => {
  it('a proven edge pays less rent; an unproven one pays full price', () => {
    const org = new GrownOrganism({
      ...defaultGrownConfig,
      seed: 1,
      seedEdges: 500,
      rentN0: 25,
      sleepEvery: 1_000_000,
    })
    const E = org.edges
    E.w.fill(0.5)
    E.n.fill(0)
    E.n[0] = 250 // ten rentN0s of evidence → pays ~1/11th
    org.sense.fill(0)
    for (let t = 0; t < 200; t++) org.tick()
    const proven = 0.5 - E.w[0]
    const unproven = 0.5 - E.w[1]
    expect(unproven).toBeCloseTo(200 * defaultGrownConfig.rent, 4)
    expect(proven).toBeLessThan(unproven / 8)
    expect(proven).toBeGreaterThan(0) // still taxed — durability is earned, not free
  })
})

describe('juvenile grace (graceSleeps)', () => {
  it('a juvenile pays no rent and cannot die; an adult pays and can', () => {
    const org = new GrownOrganism({
      ...defaultGrownConfig,
      seed: 2,
      seedEdges: 500,
      graceSleeps: 5,
      sleepEvery: 1_000_000,
    })
    const E = org.edges
    E.w.fill(0.01) // below deathThreshold — doomed, if death applied
    org.sense.fill(0)
    for (let t = 0; t < 100; t++) org.tick()
    // juveniles: no rent — weights untouched
    expect(E.w[0]).toBeCloseTo(0.01, 6)
    org.sleep() // sleeps: 0 → 1; all edges age 1 < 5
    expect(org.stats().edgesDied).toBe(0)
    expect(org.edges.count).toBeGreaterThanOrEqual(500)

    // age them out of grace: five sleeps
    for (let k = 0; k < 5; k++) org.sleep()
    org.edges.w.fill(0.01)
    const before = org.edges.count
    org.sleep()
    // adults below threshold now die
    expect(org.edges.count).toBeLessThan(before)
  })

  it('an edge grown later gets its own grace from its own birth sleep', () => {
    const org = new GrownOrganism({
      ...defaultGrownConfig,
      seed: 3,
      graceSleeps: 3,
      sleepEvery: 1_000_000,
    })
    for (let t = 0; t < 400; t++) org.tick()
    org.sleep() // growth happens; newborns have born = 0 (sleep index at birth)
    expect(org.edges.count).toBeGreaterThan(0)
    org.edges.w.fill(0.001) // all doomed if adult
    org.sleep() // age 1 < 3: protected
    expect(org.edges.count).toBeGreaterThan(0)
  })
})
