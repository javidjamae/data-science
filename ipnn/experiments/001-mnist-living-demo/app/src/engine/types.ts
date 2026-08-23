/**
 * The whole surface an organism must expose to the outside world.
 *
 * Measured, not guessed: the teacher, the trial state machine, the sustained
 * readout, the demo controller, the UI and the tests touch an organism only
 * through these members. Everything else about a substrate — how it is wired,
 * whether it has geometry, whether its connections can be born and die — is
 * private to the substrate.
 *
 * This is what lets experiment 002's grown substrate be a *swap* rather than a
 * fork, and it is why experiment 001 can keep working as 002's control arm
 * (002 design §2).
 */
export interface OrganismLike {
  readonly cfg: SubstrateConfig
  /** current sense state (binary), written by the environment */
  readonly sense: Uint8Array
  /** per-unit firing state this tick — the raster the UI draws */
  readonly poolFired: Uint8Array
  /** output unit that fired this tick, or -1 for silence */
  lastWinner: number
  /** the "compelled to respond" drive; rises with silence */
  urge: number

  /** advance one tick of the organism's life */
  tick(): void
  /** deliver reward. How it is *distributed* is the substrate's business:
   * 001 broadcasts it, 002 injects it at the reward cortex to diffuse. */
  applyReward(r: number): void
  /** zero all eligibility traces without touching what was learned */
  clearTraces(): void
  /** fraction of units active this tick (telemetry) */
  poolActivity(): number
  /** per-output strength for display; 001 returns policy probabilities,
   * 002 returns windowed firing rates */
  outputProbs(): Float32Array
  /** squared L2 norms per weight population — "did anything actually move?" */
  weightNorms(): { pool: number; out: number }
}

/** The part of a substrate's config the outside world is allowed to know. */
export interface SubstrateConfig {
  outputSize: number
  /** length of `poolFired` — units the raster displays */
  poolSize: number
}

/** Experiment 001's fixed-architecture substrate. `poolSize` and
 * `outputSize` come from SubstrateConfig — everything below is private to
 * this substrate and invisible to the teacher, the UI and the tests. */
export interface OrganismConfig extends SubstrateConfig {
  seed: number
  senseSize: number
  /** synapses per pool neuron, drawn randomly from the sense */
  poolFanIn: number

  /** sigmoid gain for pool neurons */
  poolGain: number
  /** resting bias for pool neurons (negative → sparse by default) */
  poolBias: number
  /** fraction of pool neurons that should be active per tick */
  targetPoolSparsity: number
  /** homeostatic inhibition adaptation rate */
  inhibitionRate: number

  /** softmax gain for the output register */
  outputGain: number
  /** ε-exploration mixed into output sampling (keeps learning alive) */
  epsilonExplore: number
  /** logit of the "stay silent" option; urge subtracts from it */
  silenceBias: number
  /** urge growth per silent tick */
  urgeRate: number
  urgeMax: number

  /** eligibility trace decay per tick (λ) */
  traceDecay: number
  /** learning rate, pool→output synapses */
  etaOut: number
  /** learning rate, sense→pool synapses */
  etaPool: number
  /** weight clamp */
  wMax: number

  /** Beta-confidence consolidation: plasticity scales as 1/(1 + n/n0) */
  consolidation: boolean
  consolidationN0: number

  /**
   * How evidence accumulates under consolidation.
   *
   * `'count'` — the original rule and the default: `n` grows on positive
   * reward and never shrinks, so plasticity decays monotonically over the
   * organism's whole life. Kept as the default because the published M1 demo
   * and every recorded result are a record of this rule; changing it silently
   * would invalidate them (the M1b entry's decision 3).
   *
   * `'beta'` — the α/β fix, promoted from "registered follow-up" (002 design
   * §7) after serial reversal showed the count model causes *cumulative
   * learning death*: 7/32 reversals learnable against 22/32 without
   * consolidation (L-034). Confirming and contradicting evidence are tracked
   * separately — the α and β the "Beta-confidence" name promised — and
   * plasticity scales as 1/(1 + |α−β|/n0), so a synapse whose evidence turns
   * mixed becomes *less* certain and plastic again. With β ≡ 0 the two models
   * are arithmetically identical, which is what makes this a one-parameter
   * control arm rather than a fork.
   */
  evidenceModel: 'count' | 'beta'
}

export const defaultConfig: OrganismConfig = {
  seed: 1,
  senseSize: 64,
  poolSize: 160,
  outputSize: 3,
  poolFanIn: 24,

  poolGain: 2.0,
  poolBias: -1.0,
  targetPoolSparsity: 0.15,
  inhibitionRate: 0.02,

  outputGain: 1.5,
  epsilonExplore: 0.05,
  silenceBias: 0.5,
  urgeRate: 0.05,
  urgeMax: 3.0,

  traceDecay: 0.97,
  etaOut: 0.08,
  etaPool: 0.01,
  wMax: 3.0,

  consolidation: true,
  consolidationN0: 1000,
  evidenceModel: 'count',
}

export type TeacherSchedule = 'ignore' | 'correction'

export interface TeacherConfig {
  /** max ticks to wait for a spoken output per presentation */
  maxTicks: number
  /** blank-sense ticks between presentations */
  blankTicks: number
  /** sliding window (ticks) for the spoken-output readout */
  spokenWindow: number
  /** fires within the window required to count as "spoken" */
  spokenThreshold: number
  schedule: TeacherSchedule
  rewardMagnitude: number
  /** magnitude of the negative signal under the 'correction' schedule */
  correctionMagnitude: number
  /** running-baseline adaptation rate (REINFORCE variance reduction) */
  baselineRate: number
}

export const defaultTeacherConfig: TeacherConfig = {
  maxTicks: 60,
  blankTicks: 15,
  spokenWindow: 20,
  spokenThreshold: 6,
  schedule: 'ignore',
  rewardMagnitude: 1.0,
  correctionMagnitude: 0.2,
  baselineRate: 0.05,
}

export interface TrialResult {
  label: number
  /** what the organism said, or null if it stayed silent */
  spoken: number | null
  correct: boolean
  /** ticks from stimulus onset to the spoken decision */
  latency: number
}
