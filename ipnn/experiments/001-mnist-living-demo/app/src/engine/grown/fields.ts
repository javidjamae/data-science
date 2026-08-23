// The two diffusing fields (design §4).
//
//   activity field A — WHERE TO BUILD.  Emitted by any node that fires,
//     short range, and growth cones climb it. Biological reading:
//     activity-dependent neurotrophic signalling — axons grow toward active
//     tissue.
//
//   reward field R — WHAT TO KEEP.  Emitted by the reward cortex, longer
//     range, and read *locally at each synapse* as the third factor of the
//     learning rule. Biological reading: neuromodulator volume transmission
//     from a locus.
//
// The whole idea in one line: activity says where to build, reward says what
// to keep. And the substitution that makes it interesting is one symbol —
// 001's rule is Δw = η·R_global·e, this one is Δw = η·R(x)·e. A synapse is
// credited because of *where it is*, and it is where it is because growth
// climbed a field to put it there.
//
// Both obey the same discrete equation, F ← F + D∇²F − decay·F + sources,
// with a 5-point Laplacian and reflecting (Neumann) boundaries so nothing
// leaks off the edge of the sheet.

/** Explicit 2D diffusion is unstable above this; both fields are checked. */
export const MAX_STABLE_D = 0.25

/**
 * One diffusion step, in place. Reflecting boundaries: a site's missing
 * neighbours are replaced by itself, which is equivalent to zero flux across
 * the edge and keeps the total from quietly draining away at the border.
 */
export function diffuseStep(
  f: Float32Array,
  scratch: Float32Array,
  width: number,
  height: number,
  D: number,
  decay: number
): void {
  for (let y = 0; y < height; y++) {
    const row = y * width
    for (let x = 0; x < width; x++) {
      const i = row + x
      const c = f[i]
      const l = x > 0 ? f[i - 1] : c
      const r = x < width - 1 ? f[i + 1] : c
      const u = y > 0 ? f[i - width] : c
      const d = y < height - 1 ? f[i + width] : c
      scratch[i] = c + D * (l + r + u + d - 4 * c) - decay * c
    }
  }
  f.set(scratch)
}

/**
 * The activity field. Nodes deposit into it when they fire; it spreads a
 * short distance and fades. Its steady-state length scale is
 * sqrt(D/decay) lattice units and its time constant is 1/decay ticks — the
 * two are set independently, which is why both constants exist.
 */
export class ActivityField {
  readonly values: Float32Array
  private readonly scratch: Float32Array
  private readonly width: number
  private readonly height: number
  private readonly D: number
  private readonly decay: number

  constructor(width: number, height: number, D: number, decay: number) {
    if (D >= MAX_STABLE_D) {
      throw new Error(
        `activityD ${D} is at or above the explicit-scheme stability limit ` +
          `${MAX_STABLE_D}; the field would oscillate and diverge`
      )
    }
    this.width = width
    this.height = height
    this.D = D
    this.decay = decay
    this.values = new Float32Array(width * height)
    this.scratch = new Float32Array(width * height)
  }

  /** length scale in lattice units — how far "nearby" reaches */
  get lengthScale(): number {
    return Math.sqrt(this.D / this.decay)
  }

  /** time constant in ticks — how long activity is remembered */
  get timeConstant(): number {
    return 1 / this.decay
  }

  emit(site: number, amount: number): void {
    this.values[site] += amount
  }

  step(): void {
    diffuseStep(this.values, this.scratch, this.width, this.height, this.D, this.decay)
  }

  clear(): void {
    this.values.fill(0)
  }
}

/**
 * The reward field's spatial profile, R(x).
 *
 * Solved once at construction rather than integrated every tick, and that is
 * an exact simplification rather than an approximation: the source is
 * stationary, reward arrives as a single instantaneous event, and the weight
 * update reads R at that instant. The field the update sees is therefore
 * always the steady state of the same equation the activity field obeys,
 * D∇²R − decay·R + S = 0, which depends only on the length scale
 * sqrt(D/decay). Running the transient every tick would change nothing the
 * learning rule can observe.
 *
 * **Normalised to mean 1 over the lattice, deliberately.** The uniform arm
 * (001's broadcast) has R = 1 everywhere, so equal-mean normalisation means
 * the diffusing arm *redistributes* the same total credit rather than
 * delivering less of it. Normalising to a peak of 1 instead would confound
 * "credit is placed by geometry" with "credit is smaller", and the M1/M2
 * comparison against the uniform control would be unreadable.
 */
export function solveRewardProfile(
  width: number,
  height: number,
  sources: ArrayLike<number>,
  lengthScale: number,
  opts: { D?: number; tolerance?: number; maxIterations?: number } = {}
): Float32Array {
  const n = width * height
  const D = opts.D ?? 0.2
  const tol = opts.tolerance ?? 1e-9
  const maxIter = opts.maxIterations ?? 200_000
  if (D >= MAX_STABLE_D) {
    throw new Error(`reward-field D ${D} is at or above the stability limit ${MAX_STABLE_D}`)
  }
  if (lengthScale <= 0) throw new Error('rewardLambda must be positive')
  if (sources.length === 0) throw new Error('the reward field has no source sites')

  const decay = D / (lengthScale * lengthScale)
  const f = new Float32Array(n)
  const scratch = new Float32Array(n)

  // Relax to steady state. Convergence is governed by the slowest mode, which
  // is the decay itself, so this takes a few multiples of 1/decay.
  const checkEvery = 200
  let prevSum = 0
  for (let it = 0; it < maxIter; it++) {
    for (let s = 0; s < sources.length; s++) f[sources[s]] += 1
    diffuseStep(f, scratch, width, height, D, decay)
    if (it % checkEvery === checkEvery - 1) {
      let sum = 0
      for (let i = 0; i < n; i++) sum += f[i]
      if (Math.abs(sum - prevSum) <= tol * Math.max(1, sum)) break
      prevSum = sum
    }
  }

  let sum = 0
  for (let i = 0; i < n; i++) sum += f[i]
  const mean = sum / n
  if (!(mean > 0) || !Number.isFinite(mean)) {
    throw new Error('the reward field failed to converge to a positive steady state')
  }
  for (let i = 0; i < n; i++) f[i] /= mean
  return f
}

/** The uniform arm: R(x) = 1 everywhere, which is experiment 001's broadcast
 * rule exactly. The null hypothesis is the previous experiment, one parameter
 * away (design §4). */
export function uniformRewardProfile(size: number): Float32Array {
  return new Float32Array(size).fill(1)
}
