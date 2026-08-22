// M0 — edges and time of flight (design §5), including the correctness trap
// the design calls out by name because getting it wrong would silently ruin
// everything and look like "growth just doesn't work".

import { describe, it, expect } from 'vitest'
import { EdgeSet, latencyForSpan, type Edge } from './edges'

function edge(pre: number, post: number, d: number, w = 1): Edge {
  return { pre, post, w, e: 0, n: 0, d }
}

describe('time of flight', () => {
  it('makes a long edge cheaper than the chain of short hops it replaces', () => {
    // v = 3: one span-3 edge delivers in 1 tick where three unit hops take 3.
    // This is the whole premise of M3 — long-range edges are the fast path.
    const v = 3
    expect(latencyForSpan(3, v, 'span')).toBe(1)
    expect(3 * latencyForSpan(1, v, 'span')).toBe(3)

    expect(latencyForSpan(0.5, v, 'span')).toBe(1) // never instantaneous
    expect(latencyForSpan(8, v, 'span')).toBe(3)
    expect(latencyForSpan(9, v, 'span')).toBe(3)
    expect(latencyForSpan(10, v, 'span')).toBe(4)
  })

  it('charges nothing for span in the uniform-latency control arm', () => {
    for (const span of [0.5, 1, 4, 8, 30]) {
      expect(latencyForSpan(span, 3, 'uniform')).toBe(1)
    }
  })
})

describe('the delay ring buffer', () => {
  it('delivers a spike exactly d ticks later — not earlier, not later', () => {
    for (const d of [1, 2, 3]) {
      const E = new EdgeSet(4, 3)
      E.setStructure([edge(0, 1, d, 0.7)])
      const drive = new Float32Array(4)

      let t = 0
      // emit once, at tick 0
      E.deliver(t, drive)
      E.emitFrom(0, t)
      expect(drive[1]).toBe(0)
      t++

      for (let k = 1; k <= 6; k++, t++) {
        drive.fill(0)
        E.deliver(t, drive)
        if (k === d) {
          expect(drive[1]).toBeCloseTo(0.7, 6)
          expect(E.arrived[0]).toBe(1)
        } else {
          expect(drive[1]).toBe(0)
          expect(E.arrived[0]).toBe(0)
        }
      }
    }
  })

  it('keeps several spikes in flight at once without them colliding', () => {
    const E = new EdgeSet(4, 3)
    E.setStructure([edge(0, 1, 3, 1)])
    const drive = new Float32Array(4)
    // fire on three consecutive ticks; three separate arrivals must follow
    for (let t = 0; t < 3; t++) {
      drive.fill(0)
      E.deliver(t, drive)
      E.emitFrom(0, t)
      expect(drive[1]).toBe(0)
    }
    for (let t = 3; t < 6; t++) {
      drive.fill(0)
      E.deliver(t, drive)
      expect(drive[1]).toBeCloseTo(1, 6)
    }
    drive.fill(0)
    E.deliver(6, drive)
    expect(drive[1]).toBe(0)
  })

  it('sums simultaneous arrivals from different edges into the same node', () => {
    const E = new EdgeSet(4, 3)
    E.setStructure([edge(0, 3, 2, 0.5), edge(1, 3, 2, -0.2), edge(2, 3, 1, 1.0)])
    const drive = new Float32Array(4)
    E.deliver(0, drive)
    E.emitFrom(0, 0)
    E.emitFrom(1, 0)
    drive.fill(0)
    E.deliver(1, drive)
    expect(drive[3]).toBe(0)
    E.emitFrom(2, 1) // lands at t=2 alongside the two span-2 spikes
    drive.fill(0)
    E.deliver(2, drive)
    expect(drive[3]).toBeCloseTo(0.5 - 0.2 + 1.0, 6)
  })

  it('refuses a latency the ring was not sized for', () => {
    const E = new EdgeSet(4, 2)
    expect(() => E.setStructure([edge(0, 1, 3)])).toThrow(/ring buffer/)
    expect(() => E.setStructure([edge(0, 1, 0)])).toThrow(/outside/)
  })

  it('drops in-flight spikes on request, without touching weights', () => {
    const E = new EdgeSet(4, 3)
    E.setStructure([edge(0, 1, 3, 0.9)])
    const drive = new Float32Array(4)
    E.emitFrom(0, 0)
    E.clearInFlight()
    for (let t = 1; t < 8; t++) {
      drive.fill(0)
      E.deliver(t, drive)
      expect(drive[1]).toBe(0)
    }
    expect(E.w[0]).toBeCloseTo(0.9, 6)
  })
})

describe('the correctness trap: eligibility follows the arrival, not the emission', () => {
  it('credits the tick the spike landed on, d ticks after the pre fired', () => {
    // Design §5, verbatim: "with delays, 'pre and post were co-active' means
    // the presynaptic spike *arrived* at t having been emitted at t − d."
    // Update eligibility against the presynaptic node's *current* state
    // instead and credit lands on the wrong pairs entirely.
    const d = 3
    const E = new EdgeSet(2, 3)
    E.setStructure([edge(0, 1, d, 1)])
    const drive = new Float32Array(2)

    // pre fires at t=0 only; post is made to fire at every tick so that the
    // only thing distinguishing the ticks is whether a spike arrived
    const lam = 0.9
    const postFired = 1
    const postP = 0.5
    const trace: number[] = []
    for (let t = 0; t < 8; t++) {
      drive.fill(0)
      E.deliver(t, drive)
      E.e[0] = lam * E.e[0] + (E.arrived[0] ? postFired - postP : 0)
      if (t === 0) E.emitFrom(0, t)
      trace.push(E.e[0])
    }

    // nothing before the arrival...
    for (let t = 0; t < d; t++) expect(trace[t]).toBe(0)
    // ...a single deposit exactly on it...
    expect(trace[d]).toBeCloseTo(postFired - postP, 6)
    // ...and pure decay after
    expect(trace[d + 1]).toBeCloseTo((postFired - postP) * lam, 6)
    expect(trace[d + 2]).toBeCloseTo((postFired - postP) * lam * lam, 6)
  })

  it('exposes no presynaptic firing state at all — the wrong thing is unavailable', () => {
    // The defence is structural rather than disciplinary: `arrived` is the
    // only per-edge co-activity signal the class publishes, so an eligibility
    // update has nothing incorrect to read.
    const E = new EdgeSet(2, 2)
    E.setStructure([edge(0, 1, 1)])
    const perEdgeSignals = Object.keys(E).filter((k) =>
      ['arrived', 'preFired', 'preState', 'presynapticFired'].includes(k)
    )
    expect(perEdgeSignals).toEqual(['arrived'])
  })
})

describe('the CSR adjacency', () => {
  it('groups every edge under its presynaptic node', () => {
    const E = new EdgeSet(5, 2)
    E.setStructure([edge(3, 1, 1), edge(0, 2, 1), edge(3, 4, 1), edge(0, 1, 1)])
    expect(E.count).toBe(4)
    expect(E.outDegree(0)).toBe(2)
    expect(E.outDegree(3)).toBe(2)
    expect(E.outDegree(1)).toBe(0)
    for (const node of [0, 3]) {
      const [s, e] = E.outRange(node)
      for (let i = s; i < e; i++) expect(E.pre[i]).toBe(node)
    }
  })

  it('round-trips through toEdges, which is what sleep rewrites', () => {
    const E = new EdgeSet(5, 3)
    const original = [edge(1, 2, 2, 0.4), edge(0, 3, 1, -0.9)]
    E.setStructure(original)
    E.e[0] = 0.25
    E.n[0] = 7
    const out = E.toEdges()
    expect(out.length).toBe(2)
    const restored = out.find((e) => e.pre === 0)!
    expect(restored.post).toBe(3)
    expect(restored.w).toBeCloseTo(-0.9, 6)
    expect(restored.d).toBe(1)
    // traces and evidence survive a rewrite: sleep changes structure, not
    // what was learned
    const other = out.find((e) => e.pre === 1)!
    expect(other.e + restored.e).toBeCloseTo(0.25, 6)
    expect(other.n + restored.n).toBeCloseTo(7, 6)
  })
})
