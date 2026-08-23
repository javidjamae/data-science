// Experiment 002's grown substrate: configuration.
//
// Every knob lives here, in one place, on purpose. Design §10 risk 2 names
// free parameters as the main threat to this experiment's credibility — a
// system with this many knobs can be tuned into producing almost any result.
// The defence is that they are enumerable, reported verbatim in the journal,
// and that every gate's control arm was fixed before any of them was touched.
//
// Two of them are the pre-registered control-arm switches:
//   rewardField: 'uniform'  → R(x) = 1 everywhere = experiment 001's broadcast
//   latency:     'uniform'  → every edge delivers in 1 tick, span ignored
// With both set to 'uniform' this substrate's learning rule is 001's rule
// exactly, and the only remaining difference is that the wiring was grown
// rather than given (design §4, "the control arm is free").

import type { SubstrateConfig } from '../types'

/** Where the reward cortex sits. Design §3: its position is an experimental
 * variable, not a constant, and it must lie off the input→output axis so that
 * "grow toward reward" and "grow toward the output" are not the same
 * instruction by construction. M2 counterbalances across these. */
export interface Vec2 {
  x: number
  y: number
}

export interface GrownConfig extends SubstrateConfig {
  seed: number

  // --- lattice (design §3) ---
  width: number
  height: number
  /** top-left corner of the 8×8 input block */
  inputOrigin: Vec2
  /** column the output cortex occupies */
  outputX: number
  /** row of each output node */
  outputYs: number[]
  /** centre of the reward locus */
  rewardCortex: Vec2
  /** locus radius; every site within it emits reward */
  rewardRadius: number

  // --- nodes (design §6) ---
  gain: number
  bias: number
  targetSparsity: number
  inhibitionRate: number
  /** spontaneous firing rate — the bootstrap that makes growth possible at
   * t=0. Not a hack: the developing visual system wires itself on
   * self-generated retinal waves before the eyes work (design §6). */
  pSpont: number
  /** lateral inhibition inside the output cortex; replaces 001's softmax as
   * the source of competition between answers */
  lateralInhibition: number
  urgeRate: number
  urgeMax: number
  /** window (ticks) over which outputProbs() reports firing rate */
  readoutWindow: number

  // --- fields (design §4) ---
  rewardField: 'diffuse' | 'uniform'
  /** length scale of the reward field, in lattice units */
  rewardLambda: number
  /** activity-field diffusion constant (must be < 0.25 for stability) */
  activityD: number
  /** activity-field decay per tick; length scale is sqrt(activityD/decay) */
  activityDecay: number

  // --- edges and time of flight (design §5) ---
  latency: 'span' | 'uniform'
  /** conduction speed in lattice units per tick; v>1 makes a long edge
   * temporally cheaper than the chain of short hops it replaces */
  conductionSpeed: number

  // --- learning (design §7) ---
  traceDecay: number
  eta: number
  wMax: number
  consolidation: boolean
  consolidationN0: number

  // --- rent, growth, death (design §7) ---
  /** paid by every edge every tick, as linear decay toward zero. Pruning is
   * not a maintenance pass — it is failure to pay. */
  rent: number
  /** weight a newly grown edge is born with (sign is random) */
  birthWeight: number
  /** |w| below this and the edge is removed at the next sleep */
  deathThreshold: number
  /** growth attempts per node per sleep, before activity gating */
  growthAttempts: number
  /** maximum span of a newly grown edge */
  rMax: number
  /** growth's distance penalty: weight ∝ A(target)·exp(−span/lambdaG), so
   * long jumps are rare rather than impossible */
  lambdaG: number
  /** structural change happens on every K-th blank onset (see §sleep in
   * grown-organism.ts for the v1 deviation from design §7) */
  sleepEvery: number
  /** cap on outgoing edges per node; design §10 risk 3 requires that binding
   * this cap be reported, never silently truncated */
  maxOutDegree: number

  /**
   * Innate scaffold: random edges present at t=0. `0` is the recorded
   * from-zero organism. Biology's overproduction stage plus innate long
   * tracts (thalamocortical axons are born, not grown to their targets);
   * proposed by Javid 2026-08-22 — random interconnections as a starting
   * point, then strengthening/rewiring/rent take over. **Weakens the
   * "from zero" claim and must be reported wherever used** (design §10
   * risk 1's own caveat).
   */
  seedEdges: number
  /** max span of innate edges — the whole sheet by default, because innate
   * tracts may be long. Growth remains bounded by rMax. */
  seedSpanMax: number

  /**
   * Earned durability (H-023's one-line form): an edge's rent scales as
   * rent/(1 + n/rentN0), so a connection that has proven itself becomes
   * cheap to keep — use-dependent stabilization, the economics of
   * myelination. `0` disables (flat rent — the recorded behavior). Unproven
   * edges still pay full price, so rent stays *informative* (L-027).
   */
  rentN0: number
  /**
   * Juvenile grace (H-022's smallest form): for its first `graceSleeps`
   * rewirings an edge pays no rent and cannot die — a fair audition, long
   * enough to be discovered and have a wrong birth-sign corrected before the
   * landlord arrives. `0` disables (the recorded ~24-trial audition, which
   * L-042 showed bulldozes inherited structure before it can be used).
   */
  graceSleeps: number
}

/** The pre-registered M1 configuration: uniform reward field, uniform
 * latency, rent and growth on, zero edges at t=0 (design §8, M1). */
export const defaultGrownConfig: GrownConfig = {
  seed: 1,

  // 32×32 = 1024 sites. Input on the left, output on the right, ~20 lattice
  // units apart: with rMax = 8 no single edge can span the gap, so a path has
  // to be *built* across the sheet and its length is a real quantity.
  width: 32,
  height: 32,
  poolSize: 32 * 32,
  outputSize: 3,
  inputOrigin: { x: 2, y: 12 },
  outputX: 29,
  outputYs: [13, 16, 19],
  // 12 units off the (horizontal) input→output axis
  rewardCortex: { x: 16, y: 4 },
  rewardRadius: 1,

  gain: 2.0,
  bias: -1.0,
  targetSparsity: 0.15,
  inhibitionRate: 0.02,
  pSpont: 0.02,
  lateralInhibition: 2.0,
  urgeRate: 0.05,
  urgeMax: 3.0,
  readoutWindow: 20,

  rewardField: 'uniform',
  rewardLambda: 8,
  activityD: 0.045,
  activityDecay: 0.005, // length scale sqrt(0.045/0.005) = 3, τ = 200 ticks

  latency: 'uniform',
  conductionSpeed: 3,

  traceDecay: 0.97,
  eta: 0.08,
  wMax: 3.0,
  consolidation: true,
  consolidationN0: 1000,

  // Rent is not a free choice: it has to be commensurate with the sleep
  // interval, or the mechanism is incoherent. An edge that never earns should
  // last about one inter-sleep period and then fail to pay, so
  //   rent ≈ (birthWeight − deathThreshold) / (sleepEvery × ticks per trial)
  //        = (0.15 − 0.02) / (20 × 75) = 8.7e-5
  // Much larger and every edge dies before it is ever judged; much smaller and
  // nothing is ever selected against.
  rent: 0.00009,
  birthWeight: 0.15,
  deathThreshold: 0.02,
  growthAttempts: 2,
  rMax: 8,
  lambdaG: 4,
  sleepEvery: 20,
  maxOutDegree: 32,

  seedEdges: 0,
  seedSpanMax: 44,
  rentN0: 0,
  graceSleeps: 0,
}

/** Reward-cortex placements for M2's counterbalancing (design §8: the
 * orientation effect must survive ≥3 placements, so it cannot be an artifact
 * of one geometry). All lie off the y≈16 input→output axis. */
export const REWARD_PLACEMENTS: Vec2[] = [
  { x: 16, y: 4 }, // above
  { x: 16, y: 27 }, // below
  { x: 27, y: 4 }, // above, output side
]
