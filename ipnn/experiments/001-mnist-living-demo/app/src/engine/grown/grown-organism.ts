// Experiment 002's substrate: a grown one.
//
// Experiment 001 answered "can a fixed network learn from reward alone?" —
// yes. Its *shape*, though, was designed and never changed, and M1b showed
// that shape was not earning its keep: freezing 89% of its learnable synapses
// cost 0.004 accuracy. So this substrate is handed nothing but the interfaces
// — where the world comes in, where answers go out, where reward arrives —
// and has to grow its own wiring across a sheet with zero edges at t=0.
//
// It implements the same nine-member OrganismLike contract as 001, so the
// teacher, the three patterns, the sustained readout, manual mode and the
// whole test harness are shared verbatim rather than forked. That is not a
// convenience: experiment 001 is this experiment's control arm, and a fork
// would let the two drift until the comparison meant nothing.
//
// The learning rule differs from 001's by one symbol:
//
//     001:  Δw = η · R_global · e / (1 + n/n₀)
//     002:  Δw = η · R(x_post) · e / (1 + n/n₀)
//
// the third factor read from a *position* rather than broadcast. And with
// `rewardField: 'uniform'` the two are identical, which is why the null
// hypothesis here is the previous experiment one parameter away.
//
// ── Deviation from design §7, recorded rather than buried ─────────────────
// The design specifies sleep as an S-tick offline phase every K trials.
// Here, v1 sleep is a structural-change *event* at every K-th blank onset,
// not an S-tick phase. The reason: design §7 also says "no replay in v1", and
// without replay there is nothing for those S ticks to compute — growth and
// death are instantaneous array work. The requirement the phase existed to
// serve is fully met, because rewiring still happens only with the sense
// dark, between trials, never mid-thought. The S-tick phase comes back when
// replay does, and is registered as owed.
//
// Detecting the blank onset from the sense going dark is what keeps this off
// the interface: the teacher never has to know the organism sleeps.

import { mulberry32, type Rng } from '../rng'
import type { OrganismLike } from '../types'
import { type GrownConfig } from './config'
import { Lattice, ROLE_INPUT, ROLE_OUTPUT } from './lattice'
import { ActivityField, solveRewardProfile, uniformRewardProfile } from './fields'
import { EdgeSet, latencyForSpan, type Edge } from './edges'

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

/** Standing measurements, design §8. Substrate-specific and deliberately not
 * on OrganismLike: the interface is the minimum every substrate must expose,
 * not a ceiling on what one may offer its own experiments. */
export interface GrownStats {
  ticks: number
  edges: number
  edgesBorn: number
  edgesDied: number
  sleeps: number
  growthAttempts: number
  /** times the per-node out-degree cap blocked a growth attempt. Design §10
   * risk 3: this must be reported, never silently truncated. */
  capBinds: number
  /** ticks until the first positive reward ever landed, or null */
  ticksToFirstReward: number | null
  /** is there a path of live edges from the input cortex to each output */
  connected: boolean[]
  /** hops along the shortest such path, or null where none exists */
  hops: (number | null)[]
  /**
   * Design §8's "input→output path length distribution", per sense pixel:
   * `inputHops[h]` is how many of the 64 input sites reach *some* output in
   * exactly h hops, and `inputHops[0]` counts those that reach none.
   *
   * The shortest path over the whole cortex (`hops` above) is a misleading
   * summary on its own — one pixel wired straight to an output reports "1 hop"
   * for a substrate in which the other 63 are four hops back. This is the
   * honest version.
   */
  inputHops: number[]
  meanActivity: number
  activityVariance: number
  /** ticks where exactly one output fired — a clean answer */
  spokenTicks: number
  /** ticks where two or more outputs fired at once. Design §10 risk 4: the
   * softmax is gone, so output collapse must be watched for directly. */
  ambiguousTicks: number
  /** design §5 corollary: credit cannot reach further than the eligibility
   * horizon, so λ and the deepest path delay must be reported together.
   * The delay is measured along the shortest-*hop* input→output path, which
   * is an upper-bound proxy: a longer-hop path could in principle deliver
   * sooner, and that is exactly the shortcut effect M3 goes looking for. */
  traceHorizonTicks: number
  deepestPathDelayTicks: number | null
}

export class GrownOrganism implements OrganismLike {
  readonly cfg: GrownConfig
  readonly lattice: Lattice
  private rng: Rng

  readonly sense: Uint8Array
  /** firing state of every lattice site — the raster, now a real map */
  readonly poolFired: Uint8Array
  lastWinner = -1
  urge = 0

  readonly edges: EdgeSet
  readonly activity: ActivityField
  /** R(x): the reward field's spatial profile, mean-normalised to 1 */
  readonly rewardProfile: Float32Array

  private p: Float32Array
  private drive: Float32Array
  private fireRate: Float32Array
  private inhibition = 0
  private ticks = 0

  // sustained readout: firing rate per output over the last readoutWindow ticks
  private outRing: Uint8Array
  private outCounts: Int32Array
  private outRates: Float32Array

  // sleep bookkeeping
  private wasDark = true
  private blankOnsets = 0
  private growthScratch: Float32Array
  private growthTargets: Int32Array

  // telemetry
  private stats_ = {
    edgesBorn: 0,
    edgesDied: 0,
    sleeps: 0,
    growthAttempts: 0,
    capBinds: 0,
    ticksToFirstReward: null as number | null,
    spokenTicks: 0,
    ambiguousTicks: 0,
    activitySum: 0,
    activitySumSq: 0,
  }

  constructor(cfg: GrownConfig) {
    this.cfg = cfg
    this.rng = mulberry32(cfg.seed)
    this.lattice = new Lattice(cfg)

    const n = this.lattice.size
    this.sense = new Uint8Array(64)
    this.poolFired = new Uint8Array(n)
    this.p = new Float32Array(n)
    this.drive = new Float32Array(n)
    this.fireRate = new Float32Array(n)

    // Zero edges at t=0. Nothing is wired; everything below has to be grown.
    const maxDelay = latencyForSpan(cfg.rMax, cfg.conductionSpeed, cfg.latency)
    this.edges = new EdgeSet(n, maxDelay)
    this.edges.setStructure([])

    this.activity = new ActivityField(cfg.width, cfg.height, cfg.activityD, cfg.activityDecay)
    this.rewardProfile =
      cfg.rewardField === 'uniform'
        ? uniformRewardProfile(n)
        : solveRewardProfile(cfg.width, cfg.height, this.lattice.rewardNodes, cfg.rewardLambda)

    this.outRing = new Uint8Array(cfg.readoutWindow * cfg.outputSize)
    this.outCounts = new Int32Array(cfg.outputSize)
    this.outRates = new Float32Array(cfg.outputSize)

    this.growthScratch = new Float32Array(this.lattice.growthOffsets.length)
    this.growthTargets = new Int32Array(this.lattice.growthOffsets.length)
  }

  // ─────────────────────────────────────────────────────────── one tick of life

  tick(): void {
    const { cfg } = this
    const lat = this.lattice
    const n = lat.size
    const t = this.ticks

    // 1. deliver everything scheduled to arrive now, and publish arrival flags
    this.drive.fill(0)
    this.edges.deliver(t, this.drive)

    // 2. input cortex — clamped by the sense, plus spontaneous firing where
    //    the sense is dark. That second half is not padding: the developing
    //    visual system wires itself on self-generated retinal waves before
    //    the eyes work, and it is what gives growth something to climb at t=0.
    let inputActive = 0
    for (let px = 0; px < 64; px++) {
      const i = lat.inputNodes[px]
      const clamped = this.sense[px] === 1
      this.p[i] = clamped ? 1 : cfg.pSpont
      // draw unconditionally, so the random stream does not depend on which
      // pattern is showing — reproducibility is worth one wasted draw
      const spont = this.rng() < cfg.pSpont
      const fired = clamped || spont ? 1 : 0
      this.poolFired[i] = fired
      inputActive += fired
    }

    // 3. interior — stochastic binary firing under global homeostatic
    //    inhibition, with a spontaneous floor
    let interiorActive = 0
    let interiorCount = 0
    for (let i = 0; i < n; i++) {
      const role = lat.role[i]
      if (role === ROLE_INPUT || role === ROLE_OUTPUT) continue
      interiorCount++
      const base = sigmoid(cfg.gain * (this.drive[i] - this.inhibition) + cfg.bias)
      const pi = base + (1 - base) * cfg.pSpont
      this.p[i] = pi
      const fired = this.rng() < pi ? 1 : 0
      this.poolFired[i] = fired
      interiorActive += fired
    }

    // 4. output cortex — no softmax. Competition is structural: the three
    //    nodes are evaluated in a random order each tick and each one already
    //    firing suppresses the rest. Random order because a fixed one would
    //    hand output 0 a standing advantage.
    const K = cfg.outputSize
    const order = this.shuffledOutputs(K)
    let firedOutputs = 0
    let soleWinner = -1
    for (let oi = 0; oi < K; oi++) {
      const k = order[oi]
      const i = lat.outputNodes[k]
      const suppressed = this.drive[i] - this.inhibition - cfg.lateralInhibition * firedOutputs
      const base = sigmoid(cfg.gain * suppressed + cfg.bias + this.urge)
      const pi = base + (1 - base) * cfg.pSpont
      this.p[i] = pi
      const fired = this.rng() < pi ? 1 : 0
      this.poolFired[i] = fired
      if (fired) {
        soleWinner = firedOutputs === 0 ? k : -1
        firedOutputs++
      }
    }
    // An answer is one output firing alone. Two at once is not a quiet tie to
    // be broken by whoever had more drive — it is the organism failing to say
    // one thing, and it is counted as such (design §10 risk 4).
    this.lastWinner = firedOutputs === 1 ? soleWinner : -1
    if (firedOutputs === 1) this.stats_.spokenTicks++
    else if (firedOutputs > 1) this.stats_.ambiguousTicks++

    // 5. homeostasis, measured on the interior only: the input cortex is
    //    clamped by whatever pattern is showing, and the output cortex has
    //    urge on it, so neither is a reading of the substrate's own excitability
    this.inhibition +=
      cfg.inhibitionRate * (interiorActive / Math.max(1, interiorCount) - cfg.targetSparsity)

    // 6. eligibility, against the ARRIVAL — and rent, paid by every edge every
    //    tick whether it earned anything or not
    const lam = cfg.traceDecay
    const rent = cfg.rent
    const E = this.edges
    for (let s = 0; s < E.count; s++) {
      const j = E.post[s]
      E.e[s] = lam * E.e[s] + (E.arrived[s] ? this.poolFired[j] - this.p[j] : 0)
      const w = E.w[s]
      if (w > 0) E.w[s] = w > rent ? w - rent : 0
      else if (w < 0) E.w[s] = w < -rent ? w + rent : 0
    }

    // 7. emission — every node that fired schedules its outgoing spikes to
    //    land at t + d
    for (let i = 0; i < n; i++) {
      if (this.poolFired[i]) E.emitFrom(i, t)
    }

    // 8. the activity field: fired nodes deposit, the field spreads and fades.
    //    Growth climbs this at the next sleep.
    for (let i = 0; i < n; i++) {
      if (this.poolFired[i]) this.activity.emit(i, 1)
    }
    this.activity.step()

    // 9. per-node firing rate, on the same time constant as the activity
    //    field, so "which nodes may grow" and "where they grow to" are read
    //    over the same window
    const a = cfg.activityDecay
    for (let i = 0; i < n; i++) {
      this.fireRate[i] += a * (this.poolFired[i] - this.fireRate[i])
    }

    // 10. sustained readout and urge
    this.pushReadout()
    if (firedOutputs > 0) this.urge = 0
    else this.urge = Math.min(cfg.urgeMax, this.urge + cfg.urgeRate)

    // telemetry
    const act = (inputActive + interiorActive + firedOutputs) / n
    this.stats_.activitySum += act
    this.stats_.activitySumSq += act * act

    this.ticks++

    // 11. sleep, if the sense has just gone dark for the K-th time
    this.maybeSleep()
  }

  private shuffledOutputs(K: number): Int32Array {
    const order = new Int32Array(K)
    for (let k = 0; k < K; k++) order[k] = k
    for (let i = K - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1))
      const tmp = order[i]
      order[i] = order[j]
      order[j] = tmp
    }
    return order
  }

  private pushReadout(): void {
    const { cfg } = this
    const slot = this.ticks % cfg.readoutWindow
    for (let k = 0; k < cfg.outputSize; k++) {
      const idx = slot * cfg.outputSize + k
      if (this.outRing[idx]) this.outCounts[k]--
      const fired = this.poolFired[this.lattice.outputNodes[k]]
      this.outRing[idx] = fired
      if (fired) this.outCounts[k]++
      this.outRates[k] = this.outCounts[k] / cfg.readoutWindow
    }
  }

  // ───────────────────────────────────────────────────────────────── learning

  /**
   * Reward arrives at the reward cortex and is read locally at each synapse.
   *
   * `applyReward(r)` keeps 001's signature but changes meaning: 001 broadcast
   * r to every synapse, this injects it at a locus and every synapse gets
   * `r · R(x_post)`. The teacher does not need to know that, which is the
   * point of the interface.
   *
   * The evidence count `n` is incremented by |e| un-modulated, exactly as in
   * 001, so consolidation behaves identically in the uniform arm and the only
   * thing R(x) changes is the weight step itself.
   */
  applyReward(r: number): void {
    if (r === 0) return
    const { cfg } = this
    const E = this.edges
    const n0 = cfg.consolidationN0
    const wMax = cfg.wMax

    if (r > 0 && this.stats_.ticksToFirstReward === null) {
      this.stats_.ticksToFirstReward = this.ticks
    }

    for (let s = 0; s < E.count; s++) {
      const e = E.e[s]
      if (e === 0) continue
      const R = this.rewardProfile[E.post[s]]
      const plasticity = cfg.consolidation ? 1 / (1 + E.n[s] / n0) : 1
      let w = E.w[s] + cfg.eta * r * R * e * plasticity
      if (w > wMax) w = wMax
      else if (w < -wMax) w = -wMax
      E.w[s] = w
      if (r > 0) E.n[s] += Math.abs(e)
    }
  }

  /** Forget what just happened, not what was learned (001's semantics). */
  clearTraces(): void {
    this.edges.e.fill(0)
  }

  // ─────────────────────────────────────────────────────── sleep: grow and die

  private maybeSleep(): void {
    let dark = true
    for (let px = 0; px < 64; px++) {
      if (this.sense[px]) {
        dark = false
        break
      }
    }
    const onset = dark && !this.wasDark
    this.wasDark = dark
    if (!onset) return

    this.blankOnsets++
    if (this.blankOnsets % this.cfg.sleepEvery === 0) this.sleep()
  }

  /**
   * Structural change. Death first, then growth, so a newly grown edge gets a
   * full inter-sleep period to earn its keep before it is ever judged.
   */
  sleep(): void {
    const { cfg } = this
    const lat = this.lattice
    const kept: Edge[] = []
    const all = this.edges.toEdges()

    // death: an edge that stopped earning has been decaying under rent since
    // its last reward, and eventually cannot pay
    for (const ed of all) {
      if (Math.abs(ed.w) < cfg.deathThreshold) this.stats_.edgesDied++
      else kept.push(ed)
    }

    const present = new Set<number>()
    const outDeg = new Int32Array(lat.size)
    for (const ed of kept) {
      present.add(ed.pre * lat.size + ed.post)
      outDeg[ed.pre]++
    }

    // growth: active nodes send growth cones up the activity field
    const offsets = lat.growthOffsets
    const A = this.activity.values
    for (let i = 0; i < lat.size; i++) {
      // a silent node grows nothing; a node firing at the target rate grows
      // with certainty. This is the activity gate, and it costs no parameter
      // beyond the attempt budget itself.
      const gate = Math.min(1, this.fireRate[i] / cfg.targetSparsity)
      const xi = lat.xOf(i)
      const yi = lat.yOf(i)

      for (let attempt = 0; attempt < cfg.growthAttempts; attempt++) {
        if (this.rng() >= gate) continue
        this.stats_.growthAttempts++
        if (outDeg[i] >= cfg.maxOutDegree) {
          this.stats_.capBinds++
          continue
        }

        // weight candidates by A(target)·exp(−span/λ_g): climb the activity
        // field, with distance discouraged but not forbidden, so long jumps
        // are rare rather than impossible
        let total = 0
        let m = 0
        for (let o = 0; o < offsets.length; o++) {
          const off = offsets[o]
          const x = xi + off.dx
          const y = yi + off.dy
          if (x < 0 || x >= lat.width || y < 0 || y >= lat.height) continue
          const j = lat.index(x, y)
          // nothing grows into the input cortex: those nodes are clamped by
          // the sense, so an edge into one could never do anything but pay rent
          if (lat.role[j] === ROLE_INPUT) continue
          // and the output cortex does not wire to itself — lateral
          // inhibition already occupies that relationship
          if (lat.role[i] === ROLE_OUTPUT && lat.role[j] === ROLE_OUTPUT) continue
          if (present.has(i * lat.size + j)) continue
          const wgt = (A[j] + 1e-6) * off.distanceWeight
          total += wgt
          this.growthScratch[m] = total
          this.growthTargets[m] = j
          m++
        }
        if (m === 0 || total <= 0) continue

        const pick = this.rng() * total
        let lo = 0
        let hi = m - 1
        while (lo < hi) {
          const mid = (lo + hi) >> 1
          if (this.growthScratch[mid] < pick) lo = mid + 1
          else hi = mid
        }
        const j = this.growthTargets[lo]

        const span = lat.span(i, j)
        kept.push({
          pre: i,
          post: j,
          // random sign: an all-excitatory birth would bias the substrate
          // toward runaway activity, and learning can flip a wrong sign anyway
          w: this.rng() < 0.5 ? cfg.birthWeight : -cfg.birthWeight,
          e: 0,
          n: 0,
          d: latencyForSpan(span, cfg.conductionSpeed, cfg.latency),
        })
        present.add(i * lat.size + j)
        outDeg[i]++
        this.stats_.edgesBorn++
      }
    }

    this.edges.setStructure(kept)
    this.stats_.sleeps++
  }

  // ──────────────────────────────────────────────────────────────── telemetry

  /** fraction of lattice sites active this tick */
  poolActivity(): number {
    let a = 0
    for (let i = 0; i < this.poolFired.length; i++) a += this.poolFired[i]
    return a / this.poolFired.length
  }

  /**
   * There is no softmax to report, so this is each output's firing rate over
   * the readout window. The UI's bars keep their meaning — "how strongly is
   * it saying this" — and need no structural change (design §9).
   */
  outputProbs(): Float32Array {
    return this.outRates
  }

  /**
   * 001 has two weight populations and the interface reports both. The
   * mapping here: `out` is every edge terminating in the output cortex —
   * 001's readout layer — and `pool` is everything else.
   */
  weightNorms(): { pool: number; out: number } {
    const E = this.edges
    let pool = 0
    let out = 0
    for (let s = 0; s < E.count; s++) {
      const w2 = E.w[s] * E.w[s]
      if (this.lattice.role[E.post[s]] === ROLE_OUTPUT) out += w2
      else pool += w2
    }
    return { pool, out }
  }

  stats(): GrownStats {
    const { connected, hops, deepestDelay } = this.tracePaths()
    const inputHops = this.inputHopDistribution()
    const t = Math.max(1, this.ticks)
    const mean = this.stats_.activitySum / t
    return {
      ticks: this.ticks,
      edges: this.edges.count,
      edgesBorn: this.stats_.edgesBorn,
      edgesDied: this.stats_.edgesDied,
      sleeps: this.stats_.sleeps,
      growthAttempts: this.stats_.growthAttempts,
      capBinds: this.stats_.capBinds,
      ticksToFirstReward: this.stats_.ticksToFirstReward,
      connected,
      hops,
      inputHops,
      meanActivity: mean,
      activityVariance: Math.max(0, this.stats_.activitySumSq / t - mean * mean),
      spokenTicks: this.stats_.spokenTicks,
      ambiguousTicks: this.stats_.ambiguousTicks,
      traceHorizonTicks: 1 / (1 - this.cfg.traceDecay),
      deepestPathDelayTicks: deepestDelay,
    }
  }

  /**
   * How far each sense pixel is from *any* output, in hops over live edges.
   *
   * Computed by one breadth-first search backwards from the output cortex,
   * which gives every site its distance-to-an-answer in a single pass. The
   * reverse adjacency is not maintained (structure is CSR by presynaptic node
   * for the wake loop's sake), so each level scans the edge list — O(edges ×
   * depth), which is nothing at telemetry cadence.
   */
  private inputHopDistribution(): number[] {
    const E = this.edges
    const dist = new Int32Array(this.lattice.size).fill(-1)
    let frontier: number[] = []
    for (let k = 0; k < this.cfg.outputSize; k++) {
      const i = this.lattice.outputNodes[k]
      if (dist[i] === -1) {
        dist[i] = 0
        frontier.push(i)
      }
    }

    for (let depth = 1; frontier.length > 0; depth++) {
      const inFrontier = new Uint8Array(this.lattice.size)
      for (const i of frontier) inFrontier[i] = 1
      const next: number[] = []
      for (let s = 0; s < E.count; s++) {
        if (Math.abs(E.w[s]) < this.cfg.deathThreshold) continue
        if (!inFrontier[E.post[s]]) continue
        const pre = E.pre[s]
        if (dist[pre] !== -1) continue
        dist[pre] = depth
        next.push(pre)
      }
      frontier = next
    }

    let maxHop = 0
    for (const i of this.lattice.inputNodes) if (dist[i] > maxHop) maxHop = dist[i]
    // index 0 is "reaches no output at all"; index h is "reaches one in h hops"
    const out = new Array(maxHop + 1).fill(0)
    for (const i of this.lattice.inputNodes) out[dist[i] === -1 ? 0 : dist[i]]++
    return out
  }

  /**
   * Breadth-first search from the input cortex to each output, over edges that
   * are actually alive — an edge already decayed below the death threshold is
   * a path on paper only, and will be gone at the next sleep.
   *
   * This is half of M1's gate: reaching 0.80 accuracy means nothing if no
   * input→output path exists to have carried the information.
   */
  private tracePaths(): {
    connected: boolean[]
    hops: (number | null)[]
    deepestDelay: number | null
  } {
    const lat = this.lattice
    const E = this.edges
    const dist = new Int32Array(lat.size).fill(-1)
    const delay = new Int32Array(lat.size).fill(-1)
    const queue = new Int32Array(lat.size)
    let head = 0
    let tail = 0

    for (let px = 0; px < 64; px++) {
      const i = lat.inputNodes[px]
      if (dist[i] === -1) {
        dist[i] = 0
        delay[i] = 0
        queue[tail++] = i
      }
    }

    while (head < tail) {
      const i = queue[head++]
      const [start, end] = E.outRange(i)
      for (let s = start; s < end; s++) {
        if (Math.abs(E.w[s]) < this.cfg.deathThreshold) continue
        const j = E.post[s]
        if (dist[j] !== -1) continue
        dist[j] = dist[i] + 1
        delay[j] = delay[i] + E.delay[s]
        queue[tail++] = j
      }
    }

    const connected: boolean[] = []
    const hops: (number | null)[] = []
    let deepestDelay: number | null = null
    for (let k = 0; k < this.cfg.outputSize; k++) {
      const i = lat.outputNodes[k]
      const reached = dist[i] !== -1
      connected.push(reached)
      hops.push(reached ? dist[i] : null)
      if (reached && (deepestDelay === null || delay[i] > deepestDelay)) {
        deepestDelay = delay[i]
      }
    }
    return { connected, hops, deepestDelay }
  }
}
