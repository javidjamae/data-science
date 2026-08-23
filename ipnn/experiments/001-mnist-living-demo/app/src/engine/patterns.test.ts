// The two task sets have to be comparably learnable, or experiment 003's
// central comparison — trials-to-criterion on B versus on A — measures
// difficulty rather than transfer.

import { describe, it, expect } from 'vitest'
import { M1_PATTERNS, TASK_A_PATTERNS, TASK_B_PATTERNS } from './patterns'

const count = (p: Uint8Array) => p.reduce((a, b) => a + b, 0)
const iou = (a: Uint8Array, b: Uint8Array) => {
  let inter = 0
  let uni = 0
  for (let i = 0; i < a.length; i++) {
    if (a[i] && b[i]) inter++
    if (a[i] || b[i]) uni++
  }
  return inter / uni
}

describe('task A and task B are comparable', () => {
  it('task A is untouched — every existing gate cites it', () => {
    expect(TASK_A_PATTERNS).toBe(M1_PATTERNS)
    expect(M1_PATTERNS.length).toBe(3)
    expect(M1_PATTERNS.map(count)).toEqual([24, 24, 20])
  })

  it('task B has three glyphs of identical pixel count', () => {
    // identical counts mean "how much is lit" cannot be the discriminating cue
    expect(TASK_B_PATTERNS.length).toBe(3)
    expect(new Set(TASK_B_PATTERNS.map(count)).size).toBe(1)
  })

  it('task B is as internally separable as task A', () => {
    const spread = (ps: Uint8Array[]) => {
      const o: number[] = []
      for (let i = 0; i < ps.length; i++) {
        for (let j = i + 1; j < ps.length; j++) o.push(iou(ps[i], ps[j]))
      }
      return Math.max(...o)
    }
    // B's worst-case pair must not be meaningfully harder than A's
    expect(spread(TASK_B_PATTERNS)).toBeLessThan(spread(M1_PATTERNS) + 0.12)
  })

  it('no task B glyph is a near-copy of a task A glyph', () => {
    // otherwise "transfer" could just be A's answer still being right
    for (const b of TASK_B_PATTERNS) {
      for (const a of M1_PATTERNS) expect(iou(a, b)).toBeLessThan(0.5)
    }
  })

  it('all six glyphs are distinct', () => {
    const all = [...M1_PATTERNS, ...TASK_B_PATTERNS].map((p) => p.join(''))
    expect(new Set(all).size).toBe(6)
  })
})
