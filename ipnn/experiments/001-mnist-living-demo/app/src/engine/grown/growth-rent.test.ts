// M0 — learning, rent, growth, death (design §7).
//
// The claim being pinned here is that pruning is not a maintenance pass: an
// edge dies because it failed to pay rent. Metabolic cost is the selection
// pressure, and nothing else removes an edge.

import { describe, it, expect } from 'vitest'
import { GrownOrganism } from './grown-organism'
import { defaultGrownConfig, type GrownConfig } from './config'
import { ROLE_INPUT, ROLE_INTERIOR, ROLE_OUTPUT } from './lattice'
import { M1_PATTERNS } from '../patterns'

function organism(over: Partial<GrownConfig> = {}): GrownOrganism {
  return new GrownOrganism({ ...defaultGrownConfig, ...over })
}

/**
 * Live for a while, then rewire. Growth is gated on a node's own firing rate,
 * which is zero at construction — so a freshly built organism that sleeps
 * immediately grows nothing at all. It has to have lived first.
 */
function warm(org: GrownOrganism, ticks = 400): void {
  for (let t = 0; t < ticks; t++) org.tick()
  org.sleep()
}

/** one crude trial: a pattern, then darkness */
function trial(org: GrownOrganism, label = 0, ticks = 40, blank = 10): void {
  org.sense.set(M1_PATTERNS[label])
  for (let t = 0; t < ticks; t++) org.tick()
  org.sense.fill(0)
  for (let t = 0; t < blank; t++) org.tick()
}

describe('zero edges at t=0', () => {
  it('starts with nothing wired', () => {
    const org = organism()
    expect(org.edges.count).toBe(0)
    expect(org.stats().connected).toEqual([false, false, false])
    expect(org.stats().hops).toEqual([null, null, null])
  })

  it('grows nothing on a first sleep, because nothing has fired yet', () => {
    const org = organism()
    org.sleep()
    expect(org.edges.count).toBe(0)
  })

  it('still lives: spontaneous firing gives growth something to climb', () => {
    // design §6 — this is the bootstrap, and without it the substrate is a
    // dead sheet forever: no activity, no field, no growth, no reward
    const org = organism()
    for (let t = 0; t < 50; t++) org.tick()
    expect(org.poolActivity()).toBeGreaterThan(0)
    let fieldMass = 0
    for (const v of org.activity.values) fieldMass += v
    expect(fieldMass).toBeGreaterThan(0)
  })

  it('is a dead sheet only once the homeostat is off too — the cold-start control', () => {
    // Pre-registered as the no-spontaneous-activity arm, expected to fail
    // outright (design §8). Getting there takes more than removing pSpont:
    // see the next test for why. The input cortex still fires, because the
    // sense clamps it; nothing else ever does.
    const org = organism({ pSpont: 0, bias: -30, urgeMax: 0, inhibitionRate: 0 })
    let nonInputFired = 0
    for (let t = 0; t < 25_000; t++) {
      org.tick()
      for (let i = 0; i < org.lattice.size; i++) {
        if (org.lattice.role[i] !== ROLE_INPUT && org.poolFired[i]) nonInputFired++
      }
    }
    expect(nonInputFired).toBe(0)
  })

  it('the homeostat is a second, stronger bootstrap: it revives a silenced sheet', () => {
    // Design §6 names spontaneous firing as "the bootstrap that makes growth
    // possible at t=0". It is not the only one, and not the strongest. Global
    // homeostatic inhibition was kept because it is "the stability lever we
    // already trust" — but it is also an activity *source*: with nothing
    // firing, the error term is negative every tick, so inhibition falls
    // without bound until the interior fires again, whatever the bias says.
    //
    // This matters beyond tidiness. A control arm that removes pSpont and
    // nothing else does not remove the bootstrap, and would be reported as
    // "the cold-start control survived" when it was never a control.
    const org = organism({ pSpont: 0, bias: -30, urgeMax: 0, sleepEvery: 1_000_000 })
    const interiorRate = (ticks: number): number => {
      let fired = 0
      let n = 0
      for (let t = 0; t < ticks; t++) {
        org.tick()
        for (let i = 0; i < org.lattice.size; i++) {
          if (org.lattice.role[i] === ROLE_INTERIOR) {
            fired += org.poolFired[i]
            n++
          }
        }
      }
      return fired / n
    }
    expect(interiorRate(300)).toBe(0) // silent at first, which is the trap
    interiorRate(5000)
    // and back at the homeostatic target, from a sheet that should be dead
    expect(interiorRate(20_000)).toBeGreaterThan(defaultGrownConfig.targetSparsity * 0.8)
  })
})

describe('rent', () => {
  it('is paid every tick by every edge, as decay toward zero', () => {
    const org = organism({ sleepEvery: 1_000_000 })
    warm(org)
    expect(org.edges.count).toBeGreaterThan(0)

    org.edges.w.fill(0.5)
    for (let t = 0; t < 100; t++) org.tick()
    expect(org.edges.w[0]).toBeCloseTo(0.5 - 100 * defaultGrownConfig.rent, 5)
    expect(org.edges.w[0]).toBeGreaterThan(0)
  })

  it('decays negative weights toward zero too, and never through it', () => {
    const org = organism({ sleepEvery: 1_000_000, rent: 0.01 })
    warm(org)
    org.edges.w.fill(-0.02)
    for (let t = 0; t < 50; t++) org.tick()
    for (let s = 0; s < org.edges.count; s++) {
      expect(org.edges.w[s]).toBe(0)
    }
  })

  it('kills an edge that stops earning, and only at sleep', () => {
    const cfg = { ...defaultGrownConfig, sleepEvery: 1_000_000 }
    const org = new GrownOrganism(cfg)
    warm(org)
    const born = org.edges.count
    expect(born).toBeGreaterThan(0)

    // starve everything: no reward ever arrives, so rent is the only force
    org.edges.w.fill(cfg.deathThreshold * 0.5)
    for (let t = 0; t < 50; t++) org.tick()
    expect(org.edges.count).toBe(born) // structure is static while awake
    expect(org.stats().edgesDied).toBe(0)

    org.sleep()
    expect(org.stats().edgesDied).toBe(born)
  })

  it('spares an edge that earns more than it owes', () => {
    const cfg = { ...defaultGrownConfig, sleepEvery: 1_000_000 }
    const org = new GrownOrganism(cfg)
    warm(org)

    const idx = 0
    const keep = { pre: org.edges.pre[idx], post: org.edges.post[idx] }
    org.edges.w.fill(0)
    org.edges.e.fill(0)
    org.edges.w[idx] = 0.1
    org.edges.e[idx] = 1.0
    org.applyReward(1.0) // one solidly rewarded synapse; every other earns nothing
    expect(org.edges.w[idx]).toBeGreaterThan(0.1)

    for (let t = 0; t < 200; t++) org.tick()
    org.sleep()

    const survivor = org.edges
      .toEdges()
      .find((e) => e.pre === keep.pre && e.post === keep.post)
    expect(survivor).toBeDefined()
    expect(Math.abs(survivor!.w)).toBeGreaterThanOrEqual(cfg.deathThreshold)
  })
})

describe('growth', () => {
  it('only happens at sleep, never mid-thought', () => {
    const org = organism({ sleepEvery: 1_000_000 })
    for (let t = 0; t < 300; t++) org.tick()
    expect(org.edges.count).toBe(0)
    expect(org.stats().sleeps).toBe(0)
    org.sleep()
    expect(org.edges.count).toBeGreaterThan(0)
  })

  it('triggers itself on every K-th blank onset, without the teacher knowing', () => {
    // the sleep cadence has to stay off OrganismLike, or 002 would not be a
    // drop-in swap for 001 behind the same nine-member interface
    const org = organism({ sleepEvery: 3 })
    for (let i = 0; i < 3; i++) trial(org, i % 3)
    expect(org.stats().sleeps).toBe(1)
    for (let i = 0; i < 3; i++) trial(org, i % 3)
    expect(org.stats().sleeps).toBe(2)
  })

  it('respects rMax, forbids self-edges, and never duplicates an edge', () => {
    const org = organism({ sleepEvery: 1_000_000 })
    for (let s = 0; s < 4; s++) warm(org, 200)
    const seen = new Set<number>()
    for (const e of org.edges.toEdges()) {
      expect(e.pre).not.toBe(e.post)
      expect(org.lattice.span(e.pre, e.post)).toBeLessThanOrEqual(defaultGrownConfig.rMax)
      const key = e.pre * org.lattice.size + e.post
      expect(seen.has(key)).toBe(false)
      seen.add(key)
    }
    expect(seen.size).toBeGreaterThan(0)
  })

  it('never grows into the input cortex, whose nodes the sense clamps', () => {
    const org = organism({ sleepEvery: 1_000_000 })
    for (let s = 0; s < 4; s++) warm(org, 200)
    for (const e of org.edges.toEdges()) {
      expect(org.lattice.role[e.post]).not.toBe(ROLE_INPUT)
    }
  })

  it('never wires the output cortex to itself — lateral inhibition owns that', () => {
    const org = organism({ sleepEvery: 1_000_000 })
    for (let s = 0; s < 4; s++) warm(org, 200)
    for (const e of org.edges.toEdges()) {
      const both =
        org.lattice.role[e.pre] === ROLE_OUTPUT && org.lattice.role[e.post] === ROLE_OUTPUT
      expect(both).toBe(false)
    }
  })

  it('grows nothing from a silent node: only what fires can send a cone', () => {
    // interior and output nodes are held silent — homeostat off, or it would
    // revive them — so the input cortex, which the sense clamps, is the only
    // thing left that can grow
    const org = organism({
      sleepEvery: 1_000_000,
      pSpont: 0,
      bias: -30,
      urgeMax: 0,
      inhibitionRate: 0,
    })
    org.sense.set(M1_PATTERNS[0])
    warm(org, 400)
    expect(org.edges.count).toBeGreaterThan(0)
    for (const e of org.edges.toEdges()) {
      expect(org.lattice.role[e.pre]).toBe(ROLE_INPUT)
    }
  })

  it('climbs the activity field: cones land near activity, not at random', () => {
    // "activity says where to build" is the load-bearing half of the design,
    // so it gets a direct test rather than an inference from behaviour.
    // Two identical organisms; only the field they read differs.
    const lat = new GrownOrganism(defaultGrownConfig).lattice
    const hot = lat.index(20, 20)

    const meanDistanceToHot = (flat: boolean): number => {
      const org = organism({ sleepEvery: 1_000_000 })
      for (let t = 0; t < 400; t++) org.tick()
      if (flat) org.activity.values.fill(1)
      else {
        org.activity.values.fill(0)
        org.activity.values[hot] = 1000
      }
      org.sleep()
      // only sources that *could* reach the hot site are informative
      const relevant = org.edges
        .toEdges()
        .filter((e) => org.lattice.span(e.pre, hot) <= defaultGrownConfig.rMax)
      expect(relevant.length).toBeGreaterThan(10)
      return relevant.reduce((a, e) => a + org.lattice.span(e.post, hot), 0) / relevant.length
    }

    expect(meanDistanceToHot(false)).toBeLessThan(meanDistanceToHot(true) * 0.5)
  })

  it('reports the out-degree cap when it binds, rather than truncating silently', () => {
    // design §10 risk 3: edge count is unbounded by construction, so the cap
    // is real — and a cap that binds invisibly would misreport the experiment
    const org = organism({ sleepEvery: 1_000_000, maxOutDegree: 1, growthAttempts: 4 })
    for (let s = 0; s < 3; s++) warm(org, 300)
    expect(org.stats().capBinds).toBeGreaterThan(0)
    for (let i = 0; i < org.lattice.size; i++) {
      expect(org.edges.outDegree(i)).toBeLessThanOrEqual(1)
    }
  })

  it('sets each new edge’s latency from its span', () => {
    const cfg = { ...defaultGrownConfig, sleepEvery: 1_000_000, latency: 'span' as const }
    const org = new GrownOrganism(cfg)
    warm(org)
    let sawSlow = false
    for (const e of org.edges.toEdges()) {
      const span = org.lattice.span(e.pre, e.post)
      expect(e.d).toBe(Math.max(1, Math.ceil(span / cfg.conductionSpeed)))
      if (e.d > 1) sawSlow = true
    }
    expect(sawSlow).toBe(true)
  })

  it('charges every edge one tick in the uniform-latency control arm', () => {
    const org = organism({ sleepEvery: 1_000_000, latency: 'uniform' })
    warm(org)
    for (const e of org.edges.toEdges()) expect(e.d).toBe(1)
  })
})

describe('the reward field is read locally at each synapse', () => {
  it('uniform R gives 001’s rule exactly: equal credit everywhere', () => {
    const org = organism({ rewardField: 'uniform', sleepEvery: 1_000_000 })
    warm(org)
    org.edges.w.fill(0)
    org.edges.e.fill(1)
    org.edges.n.fill(0)
    org.applyReward(1)
    const first = org.edges.w[0]
    expect(first).toBeGreaterThan(0)
    for (let s = 0; s < org.edges.count; s++) expect(org.edges.w[s]).toBeCloseTo(first, 6)
  })

  it('diffusing R pays by position: near the locus earns more than far from it', () => {
    // Δw = η·R(x_post)·e — the one-symbol change from 001, and the reason a
    // synapse is credited *because of where it is*
    const org = organism({ rewardField: 'diffuse', sleepEvery: 1_000_000 })
    warm(org)
    org.edges.w.fill(0)
    org.edges.e.fill(1)
    org.edges.n.fill(0)
    org.applyReward(1)

    const lat = org.lattice
    const src = lat.rewardNodes[0]
    let near = 0
    let nearN = 0
    let far = 0
    let farN = 0
    for (let s = 0; s < org.edges.count; s++) {
      const d = lat.span(org.edges.post[s], src)
      if (d < 5) {
        near += org.edges.w[s]
        nearN++
      } else if (d > 20) {
        far += org.edges.w[s]
        farN++
      }
    }
    expect(nearN).toBeGreaterThan(0)
    expect(farN).toBeGreaterThan(0)
    expect(near / nearN).toBeGreaterThan((far / farN) * 2)
  })

  it('increments evidence un-modulated, so consolidation matches 001', () => {
    // only the weight step is allowed to differ between the arms; if the
    // evidence count moved too, the comparison would be reading two changes
    for (const rewardField of ['uniform', 'diffuse'] as const) {
      const org = organism({ rewardField, sleepEvery: 1_000_000 })
      warm(org)
      org.edges.n.fill(0)
      org.edges.e.fill(0.5)
      org.applyReward(1)
      expect(org.edges.n[0]).toBeCloseTo(0.5, 6)
    }
  })

  it('a negative advantage never adds evidence — confidence is earned, not taxed', () => {
    const org = organism({ sleepEvery: 1_000_000 })
    warm(org)
    org.edges.n.fill(0)
    org.edges.e.fill(0.5)
    org.applyReward(-1)
    expect(org.edges.n[0]).toBe(0)
  })
})

describe('the interior is where everything happens', () => {
  it('leaves the interior as the overwhelming majority of the sheet', () => {
    const org = organism()
    let interior = 0
    for (let i = 0; i < org.lattice.size; i++) {
      if (org.lattice.role[i] === ROLE_INTERIOR) interior++
    }
    expect(interior / org.lattice.size).toBeGreaterThan(0.9)
  })
})
