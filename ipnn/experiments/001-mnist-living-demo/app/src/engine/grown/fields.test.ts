// M0 — the two fields (design §4).

import { describe, it, expect } from 'vitest'
import {
  ActivityField,
  diffuseStep,
  solveRewardProfile,
  uniformRewardProfile,
  MAX_STABLE_D,
} from './fields'
import { Lattice } from './lattice'
import { defaultGrownConfig } from './config'

const W = 32
const H = 32
const cfg = defaultGrownConfig

function sum(f: Float32Array): number {
  let s = 0
  for (let i = 0; i < f.length; i++) s += f[i]
  return s
}

describe('diffusion', () => {
  it('conserves mass when nothing decays — nothing leaks off the sheet', () => {
    // reflecting boundaries mean zero flux across the edge; without this the
    // field quietly drains at the border and every "grow toward" measurement
    // acquires a centre bias
    const f = new Float32Array(W * H)
    const scratch = new Float32Array(W * H)
    f[0] = 100 // a corner, where a leak would show worst
    f[W * H - 1] = 50
    const before = sum(f)
    for (let t = 0; t < 500; t++) diffuseStep(f, scratch, W, H, 0.2, 0)
    expect(sum(f)).toBeCloseTo(before, 2)
  })

  it('spreads a point source outward, monotonically decreasing with distance', () => {
    const f = new Float32Array(W * H)
    const scratch = new Float32Array(W * H)
    const centre = 16 * W + 16
    f[centre] = 1000
    for (let t = 0; t < 200; t++) diffuseStep(f, scratch, W, H, 0.2, 0.001)

    let prev = Infinity
    for (let d = 0; d <= 10; d++) {
      const v = f[16 * W + 16 + d]
      expect(v).toBeLessThan(prev)
      prev = v
    }
  })

  it('stays finite for thousands of steps at the configured constants', () => {
    const a = new ActivityField(W, H, cfg.activityD, cfg.activityDecay)
    for (let t = 0; t < 20_000; t++) {
      a.emit((t * 7919) % (W * H), 1)
      a.step()
    }
    for (let i = 0; i < a.values.length; i++) {
      expect(Number.isFinite(a.values[i])).toBe(true)
      expect(a.values[i]).toBeGreaterThanOrEqual(0)
    }
  })

  it('refuses a diffusion constant above the explicit-scheme stability limit', () => {
    expect(() => new ActivityField(W, H, MAX_STABLE_D, 0.005)).toThrow(/stability/)
  })
})

describe('the activity field — where to build', () => {
  it('separates its length scale from its time constant', () => {
    // the two constants exist precisely so these can be set independently:
    // "how far does nearby reach" and "how long is activity remembered"
    const a = new ActivityField(W, H, cfg.activityD, cfg.activityDecay)
    expect(a.lengthScale).toBeCloseTo(3, 6)
    expect(a.timeConstant).toBeCloseTo(200, 6)
  })

  it('forgets: activity with no source decays away', () => {
    const a = new ActivityField(W, H, cfg.activityD, cfg.activityDecay)
    a.emit(16 * W + 16, 100)
    for (let t = 0; t < 100; t++) a.step()
    const early = sum(a.values)
    for (let t = 0; t < 2000; t++) a.step()
    expect(sum(a.values)).toBeLessThan(early * 0.01)
  })

  it('is highest where the firing was — which is what growth climbs', () => {
    const a = new ActivityField(W, H, cfg.activityD, cfg.activityDecay)
    const busy = 10 * W + 10
    const quiet = 25 * W + 25
    for (let t = 0; t < 500; t++) {
      a.emit(busy, 1)
      a.step()
    }
    expect(a.values[busy]).toBeGreaterThan(a.values[quiet] * 100)
  })
})

describe('the reward field — what to keep', () => {
  it('falls off with distance from the locus', () => {
    const lat = new Lattice(cfg)
    const R = solveRewardProfile(W, H, lat.rewardNodes, cfg.rewardLambda)
    const src = cfg.rewardCortex
    let prev = Infinity
    for (let d = 0; d <= 12; d++) {
      const v = R[lat.index(src.x + d, src.y)]
      expect(v).toBeLessThan(prev)
      prev = v
    }
  })

  it('is normalised to mean 1, so it redistributes credit rather than shrinking it', () => {
    // this is what makes the diffusing arm comparable to the uniform control:
    // both deliver the same total credit budget, placed differently
    const lat = new Lattice(cfg)
    const R = solveRewardProfile(W, H, lat.rewardNodes, cfg.rewardLambda)
    expect(sum(R) / R.length).toBeCloseTo(1, 4)
    const uniform = uniformRewardProfile(W * H)
    expect(sum(uniform) / uniform.length).toBeCloseTo(1, 10)
  })

  it('makes position matter: the output cortex and the input cortex see different R', () => {
    // if these were equal, "credited because of where it is" would be empty
    const lat = new Lattice(cfg)
    const R = solveRewardProfile(W, H, lat.rewardNodes, cfg.rewardLambda)
    const atReward = R[lat.rewardNodes[0]]
    const atOutput = R[lat.outputNodes[1]]
    const atInput = R[lat.inputNodes[32]]
    expect(atReward).toBeGreaterThan(atOutput * 2)
    expect(Math.abs(atOutput - atInput) / Math.max(atOutput, atInput)).toBeGreaterThan(0.05)
  })

  it('a longer length scale flattens the profile toward the uniform arm', () => {
    const lat = new Lattice(cfg)
    const tight = solveRewardProfile(W, H, lat.rewardNodes, 3)
    const broad = solveRewardProfile(W, H, lat.rewardNodes, 40)
    const spread = (f: Float32Array) => Math.max(...f) / Math.min(...f)
    expect(spread(tight)).toBeGreaterThan(spread(broad))
    // the limit of infinite diffusion is exactly 001's broadcast
    expect(spread(broad)).toBeLessThan(3)
  })

  it('the uniform arm is R(x) = 1 everywhere — experiment 001 exactly', () => {
    const R = uniformRewardProfile(W * H)
    for (let i = 0; i < R.length; i++) expect(R[i]).toBe(1)
  })

  it('refuses a field with no source', () => {
    expect(() => solveRewardProfile(W, H, [], 8)).toThrow(/no source/)
  })
})
