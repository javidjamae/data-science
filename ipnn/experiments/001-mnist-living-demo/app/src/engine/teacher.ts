// The teacher is the "parent": it presents stimuli, watches the output
// register, and delivers reward. The organism never sees labels — only the
// scalar reward broadcast.
//
// The teacher (not the organism) decides what counts as "spoken": an output
// neuron that fired ≥ spokenThreshold times within the last spokenWindow
// ticks since stimulus onset.

import type { Organism } from './organism'
import type { TeacherConfig, TrialResult } from './types'

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

  /** Present one stimulus, wait for an answer, deliver reward, blank. */
  runTrial(org: Organism, pattern: Uint8Array, label: number): TrialResult {
    const { cfg } = this
    org.sense.set(pattern)

    // ring buffer of winners since onset
    const win = new Int8Array(cfg.spokenWindow).fill(-1)
    const counts = new Int32Array(org.cfg.outputSize)

    let spoken: number | null = null
    let latency = cfg.maxTicks
    for (let t = 0; t < cfg.maxTicks; t++) {
      org.tick()
      const slot = t % cfg.spokenWindow
      const old = win[slot]
      if (old >= 0) counts[old]--
      win[slot] = org.lastWinner as any
      if (org.lastWinner >= 0) counts[org.lastWinner]++

      if (org.lastWinner >= 0 && counts[org.lastWinner] >= cfg.spokenThreshold) {
        spoken = org.lastWinner
        latency = t + 1
        break
      }
    }

    const correct = spoken === label
    let raw = 0
    if (spoken !== null) {
      raw = correct
        ? cfg.rewardMagnitude
        : cfg.schedule === 'correction'
          ? -cfg.correctionMagnitude
          : 0
    }

    // advantage = raw − baseline: turns "ignored" into mild discouragement
    // once the organism has tasted success, without an explicit punishment
    if (this.learning) {
      org.applyReward(raw - this.baseline)
      this.baseline += cfg.baselineRate * (raw - this.baseline)
    }

    // blank interval: sense goes dark, traces decay, life goes on
    org.sense.fill(0)
    for (let t = 0; t < cfg.blankTicks; t++) org.tick()

    return { label, spoken, correct, latency }
  }
}
