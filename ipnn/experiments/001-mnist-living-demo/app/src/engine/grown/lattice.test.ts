// M0 — the sheet (design §3).

import { describe, it, expect } from 'vitest'
import { Lattice, ROLE_INPUT, ROLE_OUTPUT, ROLE_INTERIOR } from './lattice'
import { defaultGrownConfig, REWARD_PLACEMENTS } from './config'

const cfg = defaultGrownConfig

describe('the lattice', () => {
  it('places three cortices that do not overlap, and leaves the rest interior', () => {
    const lat = new Lattice(cfg)
    expect(lat.size).toBe(1024)

    let input = 0
    let output = 0
    let interior = 0
    for (let i = 0; i < lat.size; i++) {
      if (lat.role[i] === ROLE_INPUT) input++
      else if (lat.role[i] === ROLE_OUTPUT) output++
      else if (lat.role[i] === ROLE_INTERIOR) interior++
    }
    expect(input).toBe(64)
    expect(output).toBe(3)
    expect(interior).toBe(1024 - 67)

    // the sense→site map is a bijection onto the input block
    expect(new Set(Array.from(lat.inputNodes)).size).toBe(64)
    for (const i of lat.inputNodes) expect(lat.role[i]).toBe(ROLE_INPUT)
    for (const i of lat.outputNodes) expect(lat.role[i]).toBe(ROLE_OUTPUT)
  })

  it('round-trips index ↔ coordinates', () => {
    const lat = new Lattice(cfg)
    for (let i = 0; i < lat.size; i++) {
      expect(lat.index(lat.xOf(i), lat.yOf(i))).toBe(i)
    }
  })

  it('keeps input and output too far apart for any single edge to span', () => {
    const lat = new Lattice(cfg)
    expect(lat.inputOutputSpan).toBeGreaterThan(cfg.rMax)
    // so a path needs at least this many hops, and path length is a real
    // measurable rather than a formality
    expect(Math.ceil(lat.inputOutputSpan / cfg.rMax)).toBeGreaterThanOrEqual(3)
  })

  it('puts the reward cortex off the input→output axis, in every M2 placement', () => {
    // design §3: on the axis, "grow toward reward" and "grow toward the
    // output" would be the same instruction by construction
    for (const rewardCortex of REWARD_PLACEMENTS) {
      const lat = new Lattice({ ...cfg, rewardCortex })
      expect(lat.rewardOffAxis).toBeGreaterThan(cfg.rMax)
      expect(lat.rewardNodes.length).toBeGreaterThan(0)
      for (const i of lat.rewardNodes) expect(lat.isReward[i]).toBe(1)
    }
  })

  it('refuses a reward cortex on, or within one growth step of, the axis', () => {
    // the input block is centred on y = 15.5, the outputs on y = 16, so this
    // sits essentially on the line between them
    expect(() => new Lattice({ ...cfg, rewardCortex: { x: 20, y: 16 } })).toThrow(
      /input→output axis/
    )
    // and "barely off it" is still on it for every purpose that matters: a
    // single growth cone could reach both
    expect(() => new Lattice({ ...cfg, rewardCortex: { x: 20, y: 22 } })).toThrow(
      /within one growth step/
    )
  })

  it('refuses a poolSize that is not the lattice', () => {
    // poolSize is the length of poolFired, which the UI draws as the map
    expect(() => new Lattice({ ...cfg, poolSize: 160 })).toThrow(/poolSize/)
  })

  it('refuses cortices that fall off the sheet', () => {
    expect(() => new Lattice({ ...cfg, outputX: 40 })).toThrow(/outside/)
    expect(() => new Lattice({ ...cfg, inputOrigin: { x: 30, y: 12 } })).toThrow(/outside/)
  })

  it('offers only growth targets inside rMax, and never the source itself', () => {
    const lat = new Lattice(cfg)
    expect(lat.growthOffsets.length).toBeGreaterThan(100)
    for (const off of lat.growthOffsets) {
      expect(off.span).toBeGreaterThan(0)
      expect(off.span).toBeLessThanOrEqual(cfg.rMax)
      expect(off.distanceWeight).toBeCloseTo(Math.exp(-off.span / cfg.lambdaG), 10)
    }
    // distance is discouraged but not forbidden: the far ring keeps a real
    // share of the probability, which is what makes long jumps rare rather
    // than impossible
    const near = lat.growthOffsets.find((o) => o.span === 1)!
    const far = lat.growthOffsets.find((o) => o.span === 8)!
    expect(far.distanceWeight / near.distanceWeight).toBeGreaterThan(0.1)
  })

  it('measures span as real Euclidean distance', () => {
    const lat = new Lattice(cfg)
    expect(lat.span(lat.index(3, 4), lat.index(6, 8))).toBeCloseTo(5, 10)
    expect(lat.span(lat.index(0, 0), lat.index(0, 0))).toBe(0)
  })
})
