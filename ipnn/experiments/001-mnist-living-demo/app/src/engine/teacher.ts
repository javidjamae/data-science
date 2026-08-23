// The teacher is the "parent": it presents stimuli, watches the output
// register, and delivers reward. The organism never sees labels — only the
// scalar reward broadcast.
//
// The teacher (not the organism) decides what counts as "spoken": an output
// neuron that fired ≥ spokenThreshold times within the last spokenWindow
// ticks since stimulus onset.
//
// Trials can be driven two ways: `runTrial` (headless, runs to completion —
// what the M1 tests use) or `beginTrial` + `TrialStepper.step()` (one tick
// at a time — what a live UI uses). runTrial delegates to the stepper, so
// there is exactly one implementation of trial mechanics.

import type { OrganismLike, TrialResult } from './types'
import type { TeacherConfig } from './types'

export type TrialPhase = 'stimulus' | 'blank' | 'done'

/**
 * Tick-granular state machine for one trial: stimulus phase (present, wait
 * for a spoken answer), judgment (reward broadcast the moment the answer
 * forms or the wait times out), blank phase (sense dark, life goes on).
 */
export class TrialStepper {
  readonly label: number
  /** what the organism said, or null (undecided during stimulus / silent) */
  spoken: number | null = null
  /** trial phase; 'done' after the blank interval completes */
  phase: TrialPhase = 'stimulus'
  /** per-output fire counts inside the sliding readout window (observable) */
  readonly counts: Int32Array
  /** raw reward at judgment (before baseline subtraction); 0 until judged */
  rawReward = 0
  /** advantage actually broadcast (0 when learning is off or before judgment) */
  appliedReward = 0
  /** set at judgment time; unchanged by the blank phase */
  result: TrialResult | null = null

  private readonly teacher: AutoTeacher
  private readonly org: OrganismLike
  private readonly win: Int8Array
  private t = 0
  private blankT = 0

  constructor(
    teacher: AutoTeacher,
    org: OrganismLike,
    pattern: Uint8Array,
    label: number
  ) {
    this.teacher = teacher
    this.org = org
    this.label = label
    org.sense.set(pattern)
    this.win = new Int8Array(teacher.cfg.spokenWindow).fill(-1)
    this.counts = new Int32Array(org.cfg.outputSize)
  }

  /** Advance one tick. Returns true once the trial (incl. blank) is done. */
  step(): boolean {
    const { cfg } = this.teacher
    const org = this.org

    if (this.phase === 'stimulus') {
      org.tick()
      const slot = this.t % cfg.spokenWindow
      const old = this.win[slot]
      if (old >= 0) this.counts[old]--
      this.win[slot] = org.lastWinner as any
      if (org.lastWinner >= 0) this.counts[org.lastWinner]++
      this.t++

      if (
        org.lastWinner >= 0 &&
        this.counts[org.lastWinner] >= cfg.spokenThreshold
      ) {
        this.spoken = org.lastWinner
        this.judge(this.t)
      } else if (this.t >= cfg.maxTicks) {
        this.judge(cfg.maxTicks)
      }
      // judge() may have advanced the phase (TS can't see the mutation)
      return (this.phase as TrialPhase) === 'done'
    }

    if (this.phase === 'blank') {
      org.tick()
      this.blankT++
      if (this.blankT >= cfg.blankTicks) this.phase = 'done'
      return this.phase === 'done'
    }

    return true
  }

  /** Reward broadcast + bookkeeping; runs exactly once per trial. */
  private judge(latency: number): void {
    const { cfg } = this.teacher
    const correct = this.spoken === this.label

    let raw = 0
    if (this.spoken !== null) {
      raw = correct
        ? cfg.rewardMagnitude
        : cfg.schedule === 'correction'
          ? -cfg.correctionMagnitude
          : 0
    }
    this.rawReward = raw

    // advantage = raw − baseline: turns "ignored" into mild discouragement
    // once the organism has tasted success, without an explicit punishment
    if (this.teacher.learning) {
      this.appliedReward = raw - this.teacher.baseline
      this.org.applyReward(this.appliedReward)
      this.teacher.baseline += cfg.baselineRate * (raw - this.teacher.baseline)
    }

    this.result = { label: this.label, spoken: this.spoken, correct, latency }

    // blank interval: sense goes dark, traces decay, life goes on
    this.org.sense.fill(0)
    this.phase = cfg.blankTicks > 0 ? 'blank' : 'done'
  }
}

export class AutoTeacher {
  readonly cfg: TeacherConfig
  /** running mean of raw reward — the REINFORCE baseline */
  baseline = 0
  /** the learning toggle: when false, stimuli are presented and judged but
   * no reward is delivered — the organism runs frozen */
  learning = true

  constructor(cfg: TeacherConfig) {
    this.cfg = cfg
  }

  /** Start a trial to be driven one tick at a time (live UIs). */
  beginTrial(org: OrganismLike, pattern: Uint8Array, label: number): TrialStepper {
    return new TrialStepper(this, org, pattern, label)
  }

  /** Present one stimulus, wait for an answer, deliver reward, blank. */
  runTrial(org: OrganismLike, pattern: Uint8Array, label: number): TrialResult {
    const stepper = this.beginTrial(org, pattern, label)
    while (!stepper.step()) {
      /* run to completion */
    }
    return stepper.result!
  }
}
