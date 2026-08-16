export interface OrganismConfig {
  seed: number
  senseSize: number
  poolSize: number
  outputSize: number
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
