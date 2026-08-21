// The IPNN organism: stochastic binary neurons, local eligibility traces,
// reward-modulated weight updates. No gradients, no ML libraries.
//
// Learning rule (three-factor, REINFORCE-style):
//   per synapse:  e ← λ·e + (post − p(post))·pre        (local eligibility)
//   on reward R:  Δw = η·R·e / (1 + n/n0)               (global broadcast)
// where n is the synapse's accumulated evidence count (Beta-confidence
// consolidation: well-evidenced synapses become resistant to change).
//
// The organism knows nothing about trials, labels, or accuracy. It has a
// sense, it ticks, it sometimes fires an output neuron, and it feels reward.
// Interpretation (what counts as "an answer") lives in the teacher/observer.

import { mulberry32, type Rng } from './rng'
import type { OrganismConfig, OrganismLike } from './types'

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

/** Experiment 001's substrate: fixed wiring, only weights change. Declared
 * `implements OrganismLike` so the contract is enforced at compile time
 * rather than by coincidence — experiment 002 swaps in against it. */
export class Organism implements OrganismLike {
  readonly cfg: OrganismConfig
  private rng: Rng

  /** current sense state (binary), written by the environment */
  readonly sense: Uint8Array

  // sense→pool synapses: for pool neuron j, poolPre[j*fanIn..] are sense
  // indices and poolW the matching weights
  private poolPre: Int32Array
  private poolW: Float32Array
  private poolE: Float32Array
  private poolN: Float32Array

  /** pool firing state this tick (binary) */
  readonly poolFired: Uint8Array
  private poolP: Float32Array
  /** global homeostatic inhibition holding pool activity near target */
  private inhibition = 0

  // pool→output synapses, dense: outW[k*poolSize + j]
  private outW: Float32Array
  private outE: Float32Array
  private outN: Float32Array
  private outP: Float32Array

  /** output neuron that fired this tick, or -1 (silence) */
  lastWinner = -1
  /** the "compelled to respond" drive; rises with silence, resets on output */
  urge = 0

  ticks = 0

  constructor(cfg: OrganismConfig) {
    this.cfg = cfg
    this.rng = mulberry32(cfg.seed)

    this.sense = new Uint8Array(cfg.senseSize)
    this.poolFired = new Uint8Array(cfg.poolSize)
    this.poolP = new Float32Array(cfg.poolSize)

    const nPoolSyn = cfg.poolSize * cfg.poolFanIn
    this.poolPre = new Int32Array(nPoolSyn)
    this.poolW = new Float32Array(nPoolSyn)
    this.poolE = new Float32Array(nPoolSyn)
    this.poolN = new Float32Array(nPoolSyn)
    for (let s = 0; s < nPoolSyn; s++) {
      this.poolPre[s] = Math.floor(this.rng() * cfg.senseSize)
      this.poolW[s] = (this.rng() * 2 - 1) * 0.5
    }

    const nOutSyn = cfg.outputSize * cfg.poolSize
    this.outW = new Float32Array(nOutSyn)
    this.outE = new Float32Array(nOutSyn)
    this.outN = new Float32Array(nOutSyn)
    this.outP = new Float32Array(cfg.outputSize)
    for (let s = 0; s < nOutSyn; s++) {
      this.outW[s] = (this.rng() * 2 - 1) * 0.1
    }
  }

  tick(): void {
    const { cfg } = this
    this.ticks++

    // --- pool: stochastic sigmoid firing under homeostatic inhibition ---
    const fanIn = cfg.poolFanIn
    let active = 0
    for (let j = 0; j < cfg.poolSize; j++) {
      let d = 0
      const base = j * fanIn
      for (let s = base; s < base + fanIn; s++) {
        if (this.sense[this.poolPre[s]]) d += this.poolW[s]
      }
      const p = sigmoid(cfg.poolGain * (d - this.inhibition) + cfg.poolBias)
      this.poolP[j] = p
      const fired = this.rng() < p ? 1 : 0
      this.poolFired[j] = fired
      active += fired
    }
    this.inhibition += cfg.inhibitionRate * (active / cfg.poolSize - cfg.targetPoolSparsity)

    // --- pool eligibility: e ← λe + (post − p)·pre ---
    const lam = cfg.traceDecay
    for (let j = 0; j < cfg.poolSize; j++) {
      const base = j * fanIn
      const g = this.poolFired[j] - this.poolP[j]
      for (let s = base; s < base + fanIn; s++) {
        this.poolE[s] = lam * this.poolE[s] + (this.sense[this.poolPre[s]] ? g : 0)
      }
    }

    // --- output register: softmax winner-take-all with a silence option ---
    // logits: g_out·(mean drive over active pool neurons) per output, and
    // (silenceBias − urge) for staying silent. Drive is normalized by the
    // active count so logits stay O(weight) — an un-normalized sum saturates
    // the softmax, which kills the score-function eligibility (fired − p)
    // and freezes learning (the M1 answer-collapse failure).
    const K = cfg.outputSize
    const norm = 1 / Math.max(1, active)
    const logits = new Float64Array(K + 1)
    let maxLogit = -Infinity
    for (let k = 0; k < K; k++) {
      let d = 0
      const base = k * cfg.poolSize
      for (let j = 0; j < cfg.poolSize; j++) {
        if (this.poolFired[j]) d += this.outW[base + j]
      }
      logits[k] = cfg.outputGain * d * norm
      if (logits[k] > maxLogit) maxLogit = logits[k]
    }
    logits[K] = cfg.silenceBias - this.urge
    if (logits[K] > maxLogit) maxLogit = logits[K]

    let z = 0
    for (let k = 0; k <= K; k++) {
      logits[k] = Math.exp(logits[k] - maxLogit)
      z += logits[k]
    }

    // ε-mixed sampling: mostly softmax, occasionally uniform, so non-dominant
    // outputs keep getting chances to fire (and thus to earn credit). The
    // eligibility below uses the mixed probability so it matches the policy.
    const eps = cfg.epsilonExplore
    let winner: number
    if (this.rng() < eps) {
      winner = Math.floor(this.rng() * (K + 1))
    } else {
      let r = this.rng() * z
      winner = K
      for (let k = 0; k <= K; k++) {
        r -= logits[k]
        if (r <= 0) { winner = k; break }
      }
    }
    this.lastWinner = winner === K ? -1 : winner
    for (let k = 0; k < K; k++) {
      this.outP[k] = (1 - eps) * (logits[k] / z) + eps / (K + 1)
    }

    if (this.lastWinner === -1) {
      this.urge = Math.min(cfg.urgeMax, this.urge + cfg.urgeRate)
    } else {
      this.urge = 0
    }

    // --- output eligibility ---
    for (let k = 0; k < K; k++) {
      const base = k * cfg.poolSize
      const g = (this.lastWinner === k ? 1 : 0) - this.outP[k]
      for (let j = 0; j < cfg.poolSize; j++) {
        this.outE[base + j] =
          lam * this.outE[base + j] + (this.poolFired[j] ? g : 0)
      }
    }
  }

  /**
   * Broadcast a reward (typically an advantage, R − baseline) to every
   * synapse. This is the only learning signal the organism ever receives.
   */
  applyReward(r: number): void {
    if (r === 0) return
    const { cfg } = this
    const n0 = cfg.consolidationN0

    for (let s = 0; s < this.outW.length; s++) {
      const e = this.outE[s]
      if (e === 0) continue
      const plasticity = cfg.consolidation ? 1 / (1 + this.outN[s] / n0) : 1
      let w = this.outW[s] + cfg.etaOut * r * e * plasticity
      if (w > cfg.wMax) w = cfg.wMax
      else if (w < -cfg.wMax) w = -cfg.wMax
      this.outW[s] = w
      if (r > 0) this.outN[s] += Math.abs(e)
    }

    for (let s = 0; s < this.poolW.length; s++) {
      const e = this.poolE[s]
      if (e === 0) continue
      const plasticity = cfg.consolidation ? 1 / (1 + this.poolN[s] / n0) : 1
      let w = this.poolW[s] + cfg.etaPool * r * e * plasticity
      if (w > cfg.wMax) w = cfg.wMax
      else if (w < -cfg.wMax) w = -cfg.wMax
      this.poolW[s] = w
      if (r > 0) this.poolN[s] += Math.abs(e)
    }
  }

  /**
   * Zero every eligibility trace. Credit is carried by traces with a
   * ~30-tick memory (λ=0.97), so an unrewarded excursion — free-running on a
   * held stimulus, say — would otherwise leave stale credit lying around for
   * the next reward to land on. Weights and evidence counts are untouched:
   * this forgets *what just happened*, not *what was learned*.
   */
  clearTraces(): void {
    this.poolE.fill(0)
    this.outE.fill(0)
  }

  /**
   * Squared L2 norms of each weight population. Cheap, and any weight change
   * moves them — which makes this the instrument for "did anything actually
   * learn?" and for the reward-withdrawal drift probe (design §6, M4c).
   */
  weightNorms(): { pool: number; out: number } {
    let pool = 0
    for (let s = 0; s < this.poolW.length; s++) pool += this.poolW[s] * this.poolW[s]
    let out = 0
    for (let s = 0; s < this.outW.length; s++) out += this.outW[s] * this.outW[s]
    return { pool, out }
  }

  /** per-output firing probabilities from the last tick (read-only view;
   * these are the ε-mixed policy probabilities, excluding silence) */
  outputProbs(): Float32Array {
    return this.outP
  }

  /** fraction of pool neurons active this tick (telemetry) */
  poolActivity(): number {
    let a = 0
    for (let j = 0; j < this.poolFired.length; j++) a += this.poolFired[j]
    return a / this.poolFired.length
  }
}
