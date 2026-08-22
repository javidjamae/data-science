// The sheet the organism grows on (design §3).
//
// A 2D lattice of node sites with real positions. 2D rather than 3D for v1
// purely because it is drawable, and being able to watch the wiring grow is
// worth a lot here.
//
// Three cortices are placed on it and everything else is interior:
//   input   64 sites in an 8×8 block, one per sense pixel, clamped by the sense
//   output  3 sites on the far side, competing by lateral inhibition
//   reward  a small locus, deliberately OFF the input→output axis
//
// The interior starts with zero edges. Input and output are placed far apart
// on purpose: no single edge can span the gap, so a path has to be built.

import type { GrownConfig, Vec2 } from './config'

export const ROLE_INTERIOR = 0
export const ROLE_INPUT = 1
export const ROLE_OUTPUT = 2

/** one candidate growth target, as an offset from a source site */
export interface GrowthOffset {
  dx: number
  dy: number
  span: number
  /** exp(−span/lambdaG), precomputed: growth's distance penalty */
  distanceWeight: number
}

export class Lattice {
  readonly width: number
  readonly height: number
  readonly size: number

  /** ROLE_* per site */
  readonly role: Uint8Array
  /** 1 for sites inside the reward locus. Reward sites are ordinary nodes;
   * this only marks where the reward field is emitted from. */
  readonly isReward: Uint8Array

  /** sense pixel p → site index */
  readonly inputNodes: Int32Array
  /** output k → site index */
  readonly outputNodes: Int32Array
  readonly rewardNodes: Int32Array

  /** candidate growth targets within rMax, shared by every source site */
  readonly growthOffsets: GrowthOffset[]

  /** how far the reward locus sits off the input→output axis. Design §3
   * requires this to be non-zero; the constructor enforces it. */
  readonly rewardOffAxis: number
  /** straight-line distance from the input block's centre to the output
   * cortex's centre — the gap a grown path has to cross */
  readonly inputOutputSpan: number

  constructor(cfg: GrownConfig) {
    this.width = cfg.width
    this.height = cfg.height
    this.size = cfg.width * cfg.height

    if (cfg.poolSize !== this.size) {
      throw new Error(
        `poolSize ${cfg.poolSize} must equal width×height ${this.size}: it is ` +
          `the length of poolFired, which the UI draws as the lattice`
      )
    }

    this.role = new Uint8Array(this.size)
    this.isReward = new Uint8Array(this.size)

    // --- input cortex: 8×8, sense pixel (r,c) → site (originX+c, originY+r) ---
    this.inputNodes = new Int32Array(64)
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const x = cfg.inputOrigin.x + c
        const y = cfg.inputOrigin.y + r
        this.requireInBounds(x, y, 'input cortex')
        const i = this.index(x, y)
        this.inputNodes[r * 8 + c] = i
        this.role[i] = ROLE_INPUT
      }
    }

    // --- output cortex ---
    if (cfg.outputYs.length !== cfg.outputSize) {
      throw new Error(
        `outputYs has ${cfg.outputYs.length} rows but outputSize is ${cfg.outputSize}`
      )
    }
    this.outputNodes = new Int32Array(cfg.outputSize)
    for (let k = 0; k < cfg.outputSize; k++) {
      const x = cfg.outputX
      const y = cfg.outputYs[k]
      this.requireInBounds(x, y, 'output cortex')
      const i = this.index(x, y)
      if (this.role[i] !== ROLE_INTERIOR) {
        throw new Error(`output node ${k} at (${x},${y}) collides with another cortex`)
      }
      this.outputNodes[k] = i
      this.role[i] = ROLE_OUTPUT
    }

    // --- reward locus ---
    const reward: number[] = []
    const rr = cfg.rewardRadius
    for (let dy = -rr; dy <= rr; dy++) {
      for (let dx = -rr; dx <= rr; dx++) {
        if (Math.hypot(dx, dy) > rr) continue
        const x = cfg.rewardCortex.x + dx
        const y = cfg.rewardCortex.y + dy
        this.requireInBounds(x, y, 'reward cortex')
        const i = this.index(x, y)
        this.isReward[i] = 1
        reward.push(i)
      }
    }
    this.rewardNodes = Int32Array.from(reward)

    // --- geometry checks the design asks for ---
    const inCentre: Vec2 = {
      x: cfg.inputOrigin.x + 3.5,
      y: cfg.inputOrigin.y + 3.5,
    }
    const outCentre: Vec2 = {
      x: cfg.outputX,
      y: cfg.outputYs.reduce((a, b) => a + b, 0) / cfg.outputYs.length,
    }
    this.inputOutputSpan = Math.hypot(outCentre.x - inCentre.x, outCentre.y - inCentre.y)
    this.rewardOffAxis = distanceToSegment(cfg.rewardCortex, inCentre, outCentre)
    // "Off the axis" has to be operational, not symbolic: a locus a fraction
    // of a lattice unit off the line is on it for every purpose that matters.
    // Requiring more than one growth step's reach means no single cone can
    // satisfy "toward reward" and "toward the output" at once, which is the
    // confound design §3 is guarding against.
    if (this.rewardOffAxis <= cfg.rMax) {
      throw new Error(
        `the reward cortex sits ${this.rewardOffAxis.toFixed(1)} from the ` +
          `input→output axis, within one growth step (rMax ${cfg.rMax}), which ` +
          'makes "grow toward reward" and "grow toward the output" the same ' +
          'instruction by construction (design §3)'
      )
    }
    if (this.inputOutputSpan <= cfg.rMax) {
      throw new Error(
        `input and output are ${this.inputOutputSpan.toFixed(1)} apart but rMax ` +
          `is ${cfg.rMax}: a single edge could span the gap, so path length ` +
          `would stop being a real quantity (design §3)`
      )
    }

    // --- growth candidate offsets, computed once ---
    const offsets: GrowthOffset[] = []
    const rMax = cfg.rMax
    for (let dy = -rMax; dy <= rMax; dy++) {
      for (let dx = -rMax; dx <= rMax; dx++) {
        if (dx === 0 && dy === 0) continue
        const span = Math.hypot(dx, dy)
        if (span > rMax) continue
        offsets.push({ dx, dy, span, distanceWeight: Math.exp(-span / cfg.lambdaG) })
      }
    }
    this.growthOffsets = offsets
  }

  index(x: number, y: number): number {
    return y * this.width + x
  }

  xOf(i: number): number {
    return i % this.width
  }

  yOf(i: number): number {
    return (i / this.width) | 0
  }

  /** Euclidean distance between two sites — an edge's span, which sets its
   * time of flight (design §5). */
  span(i: number, j: number): number {
    return Math.hypot(this.xOf(i) - this.xOf(j), this.yOf(i) - this.yOf(j))
  }

  private requireInBounds(x: number, y: number, what: string): void {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
      throw new Error(
        `${what} site (${x},${y}) falls outside the ${this.width}×${this.height} lattice`
      )
    }
  }
}

/** distance from p to the segment ab — "how far off the axis is it" */
function distanceToSegment(p: Vec2, a: Vec2, b: Vec2): number {
  const vx = b.x - a.x
  const vy = b.y - a.y
  const len2 = vx * vx + vy * vy
  if (len2 === 0) return Math.hypot(p.x - a.x, p.y - a.y)
  let t = ((p.x - a.x) * vx + (p.y - a.y) * vy) / len2
  t = Math.max(0, Math.min(1, t))
  return Math.hypot(p.x - (a.x + t * vx), p.y - (a.y + t * vy))
}
