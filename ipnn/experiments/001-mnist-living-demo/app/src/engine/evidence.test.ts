// The α/β evidence model (L-034's fix), pinned at the mechanism level.
//
// The claim the config comment makes and these tests enforce: with β ≡ 0 the
// two models are arithmetically identical, so 'count' remains bit-for-bit the
// published rule — and under 'beta', contradiction restores plasticity instead
// of being ignored.

import { describe, it, expect } from 'vitest'
import { Organism } from './organism'
import { AutoTeacher } from './teacher'
import { M1_PATTERNS } from './patterns'
import { defaultConfig, defaultTeacherConfig } from './types'
import { mulberry32, shuffleInPlace } from './rng'

/** run n teacher trials so the organism accumulates real evidence */
function live(org: Organism, trials: number, seed: number): void {
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(seed * 7919 + 1)
  let block: number[] = []
  for (let t = 0; t < trials; t++) {
    if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
    const label = block.pop()!
    teacher.runTrial(org, M1_PATTERNS[label], label)
  }
}

describe('the count model is unchanged', () => {
  it('is the default, and β is never written under it', () => {
    expect(defaultConfig.evidenceModel).toBe('count')
    const org = new Organism({ ...defaultConfig, seed: 1 })
    live(org, 120, 1)
    const ev = org.evidenceTotals()
    // negative advantages certainly occurred (baseline > 0, wrong answers),
    // yet β must not have moved — that is the original rule's blindness,
    // preserved on purpose
    expect(ev.outAlpha).toBeGreaterThan(0)
    expect(ev.outBeta).toBe(0)
    expect(ev.poolBeta).toBe(0)
  })

  it('count and beta organisms are weight-identical while no negative reward arrives', () => {
    // the two models may only diverge at the β increment, which requires r<0.
    // Drive both with identical ticks and positive-only rewards: every weight
    // must match exactly.
    const a = new Organism({ ...defaultConfig, seed: 7, evidenceModel: 'count' })
    const b = new Organism({ ...defaultConfig, seed: 7, evidenceModel: 'beta' })
    for (const org of [a, b]) {
      org.sense.set(M1_PATTERNS[0])
      for (let t = 0; t < 200; t++) {
        org.tick()
        if (t % 5 === 4) org.applyReward(0.7)
      }
    }
    expect(a.weightNorms()).toEqual(b.weightNorms())
  })
})

describe('the beta model: contradiction restores plasticity', () => {
  it('accumulates β on negative reward, and only then', () => {
    const org = new Organism({ ...defaultConfig, seed: 3, evidenceModel: 'beta' })
    org.sense.set(M1_PATTERNS[1])
    for (let t = 0; t < 50; t++) org.tick()
    org.applyReward(1)
    const afterPositive = org.evidenceTotals()
    expect(afterPositive.outAlpha).toBeGreaterThan(0)
    expect(afterPositive.outBeta).toBe(0)

    for (let t = 0; t < 50; t++) org.tick()
    org.applyReward(-0.5)
    const afterNegative = org.evidenceTotals()
    expect(afterNegative.outBeta).toBeGreaterThan(0)
    // and a negative reward never adds confirming evidence
    expect(afterNegative.outAlpha).toBe(afterPositive.outAlpha)
  })

  it('a consolidated synapse moves again once evidence turns mixed', () => {
    // Plasticity is 1/(1 + |α−β|/n0). Consolidate hard, measure the step a
    // fixed (e, r) produces, then contradict until β ≈ α and measure the same
    // step again: it must be near the young-synapse step, not the frozen one.
    const n0 = 200
    const org = new Organism({
      ...defaultConfig,
      seed: 5,
      evidenceModel: 'beta',
      consolidationN0: n0,
    })
    // reach inside deliberately: this test is about the update arithmetic, and
    // driving exact (e, α, β) states through behaviour alone would be
    // hopelessly indirect
    const guts = org as unknown as {
      outE: Float32Array
      outN: Float32Array
      outBeta: Float32Array
      outW: Float32Array
    }

    const step = (): number => {
      guts.outE.fill(0)
      guts.outE[0] = 1
      const before = guts.outW[0]
      org.applyReward(0.5)
      return guts.outW[0] - before
    }

    // young: α = β = 0 → full step
    const young = step()
    // consolidated: α = 5·n0 → 1/(1+5) of the step... (each step() also adds
    // |e|=1 to α, which is negligible against 1000)
    guts.outN[0] = 5 * n0
    const frozen = step()
    expect(frozen).toBeLessThan(young * 0.2)
    // contradicted: β climbs to meet α → |α−β| ≈ 0 → the step returns
    guts.outBeta[0] = guts.outN[0]
    const restored = step()
    expect(restored).toBeGreaterThan(young * 0.9)
    expect(restored).toBeLessThanOrEqual(young * 1.01)
  })

  it('under count, the same contradiction restores nothing — the aging is real', () => {
    const n0 = 200
    const org = new Organism({
      ...defaultConfig,
      seed: 5,
      evidenceModel: 'count',
      consolidationN0: n0,
    })
    const guts = org as unknown as { outE: Float32Array; outN: Float32Array; outW: Float32Array }

    const step = (): number => {
      guts.outE.fill(0)
      guts.outE[0] = 1
      const before = guts.outW[0]
      org.applyReward(0.5)
      return guts.outW[0] - before
    }

    const young = step()
    guts.outN[0] = 5 * n0
    const frozen = step()
    expect(frozen).toBeLessThan(young * 0.2)

    // hammer it with contradiction — the exact barrage that restored the beta
    // synapse. α can only grow, so plasticity must not recover at all.
    for (let i = 0; i < 20; i++) {
      guts.outE.fill(0)
      guts.outE[0] = 1
      org.applyReward(-1)
    }
    const after = step()
    expect(after).toBeLessThan(frozen * 1.05) // no recovery whatsoever
    expect(org.evidenceTotals().outBeta).toBe(0)
  })
})
