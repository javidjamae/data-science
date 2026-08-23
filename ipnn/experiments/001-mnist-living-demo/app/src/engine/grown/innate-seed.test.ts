// The innate scaffold (Javid 2026-08-22): random edges present at birth,
// then subject to the same strengthening, rent and rewiring as anything grown.

import { describe, it, expect } from 'vitest'
import { GrownOrganism } from './grown-organism'
import { defaultGrownConfig } from './config'
import { ROLE_INPUT, ROLE_OUTPUT } from './lattice'

describe('the innate scaffold', () => {
  it('defaults to zero — the recorded from-zero organism is untouched', () => {
    expect(defaultGrownConfig.seedEdges).toBe(0)
    const org = new GrownOrganism({ ...defaultGrownConfig, seed: 1 })
    expect(org.edges.count).toBe(0)
  })

  it('is present at birth, legal, and can be long-range', () => {
    const org = new GrownOrganism({ ...defaultGrownConfig, seed: 2, seedEdges: 3000 })
    expect(org.edges.count).toBe(3000)
    const lat = org.lattice
    const seen = new Set<number>()
    let sawLong = false
    for (const e of org.edges.toEdges()) {
      expect(e.pre).not.toBe(e.post)
      expect(lat.role[e.post]).not.toBe(ROLE_INPUT)
      expect(lat.role[e.pre] === ROLE_OUTPUT && lat.role[e.post] === ROLE_OUTPUT).toBe(false)
      const key = e.pre * lat.size + e.post
      expect(seen.has(key)).toBe(false)
      seen.add(key)
      if (lat.span(e.pre, e.post) > defaultGrownConfig.rMax) sawLong = true
    }
    // the point of innateness: tracts longer than any growth cone's reach
    expect(sawLong).toBe(true)
  })

  it('creates short eye→answer routes that from-zero growth never has', () => {
    const org = new GrownOrganism({ ...defaultGrownConfig, seed: 3, seedEdges: 6000 })
    const h = org.stats().inputHops
    expect((h[1] ?? 0) + (h[2] ?? 0)).toBeGreaterThan(0)
  })

  it('remains subject to rent: an unearning scaffold decays', () => {
    const org = new GrownOrganism({
      ...defaultGrownConfig,
      seed: 4,
      seedEdges: 2000,
      sleepEvery: 1_000_000,
    })
    const w0 = org.weightNorms()
    for (let t = 0; t < 300; t++) org.tick()
    const w1 = org.weightNorms()
    expect(w1.pool + w1.out).toBeLessThan(w0.pool + w0.out)
  })
})
