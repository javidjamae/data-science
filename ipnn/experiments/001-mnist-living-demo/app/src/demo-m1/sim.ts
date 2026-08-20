// DOM-free controller for the M1 living demo: owns the organism, the
// teacher, and the balanced presentation schedule (identical to the M1
// test harness), and exposes tick-batched stepping plus the observables a
// renderer needs. Nothing in this file touches the DOM — it runs headless
// under vitest, which is how we know the demo shows the same organism the
// gate certified.

import { Organism } from '../engine/organism'
import { AutoTeacher, TrialStepper } from '../engine/teacher'
import { M1_PATTERNS } from '../engine/patterns'
import {
  defaultConfig,
  defaultTeacherConfig,
  type TrialResult,
} from '../engine/types'
import { mulberry32, shuffleInPlace, type Rng } from '../engine/rng'

const ROLLING_WINDOW = 100

/** A learning-toggle event, for shading chart regions. */
export interface ToggleMark {
  trial: number
  learning: boolean
}

export class DemoSim {
  seed: number
  org!: Organism
  teacher!: AutoTeacher
  stepper!: TrialStepper

  /** completed trials, in order */
  trials: TrialResult[] = []
  /** rolling accuracy (last ≤100 trials) recorded after every trial */
  accuracyCurve: number[] = []
  /** learning on/off toggle history (chart shading) */
  marks: ToggleMark[] = []

  /** raw reward from the most recent judgment (renderer decays it) */
  rewardPulse = 0
  /** per-pool-neuron fire counts since last clearAccum() (raster brightness) */
  poolAccum!: Float32Array
  /** ticks accumulated since last clearAccum() */
  accumTicks = 0
  totalTicks = 0

  private order!: Rng
  private block: number[] = []
  private correctInWindow = 0

  constructor(seed: number) {
    this.seed = seed
    this.reset(seed)
  }

  reset(seed: number): void {
    this.seed = seed
    this.org = new Organism({ ...defaultConfig, seed })
    this.teacher = new AutoTeacher({ ...defaultTeacherConfig })
    // schedule identical to the M1 gate harness (m1-sanity.test.ts)
    this.order = mulberry32(seed * 7919 + 1)
    this.block = []
    this.trials = []
    this.accuracyCurve = []
    this.marks = []
    this.rewardPulse = 0
    this.poolAccum = new Float32Array(this.org.cfg.poolSize)
    this.accumTicks = 0
    this.totalTicks = 0
    this.correctInWindow = 0
    this.beginTrial()
  }

  get learning(): boolean {
    return this.teacher.learning
  }

  setLearning(on: boolean): void {
    if (on === this.teacher.learning) return
    this.teacher.learning = on
    this.marks.push({ trial: this.trials.length, learning: on })
  }

  /** the label currently on the sense (what the teacher is showing) */
  get currentLabel(): number {
    return this.stepper.label
  }

  /** rolling accuracy over the last ≤100 completed trials */
  get rollingAccuracy(): number {
    const n = Math.min(this.trials.length, ROLLING_WINDOW)
    return n === 0 ? 0 : this.correctInWindow / n
  }

  /** Advance the world by n ticks (trials begin/end as they may). */
  tick(n: number): void {
    for (let i = 0; i < n; i++) {
      const wasStimulus = this.stepper.phase === 'stimulus'
      const done = this.stepper.step()

      // the moment of judgment: stimulus phase just ended
      if (wasStimulus && this.stepper.phase !== 'stimulus') {
        this.rewardPulse = this.stepper.rawReward
      }

      // raster accumulation
      const fired = this.org.poolFired
      for (let j = 0; j < fired.length; j++) this.poolAccum[j] += fired[j]
      this.accumTicks++
      this.totalTicks++

      if (done) {
        const res = this.stepper.result!
        this.trials.push(res)
        this.correctInWindow += res.correct ? 1 : 0
        const drop = this.trials.length - ROLLING_WINDOW - 1
        if (drop >= 0 && this.trials[drop].correct) this.correctInWindow--
        this.accuracyCurve.push(this.rollingAccuracy)
        this.beginTrial()
      }
    }
  }

  /** Run until exactly `n` total trials have completed (headless testing).
   * Chunks ticks for speed, then single-steps near the boundary — the
   * shortest possible trial is ~21 ticks, so a 500-tick chunk can complete
   * at most ~24 trials. */
  runToTrials(n: number): void {
    while (this.trials.length < n) {
      this.tick(n - this.trials.length > 30 ? 500 : 1)
    }
  }

  /** Reset the raster accumulator (call once per rendered frame). */
  clearAccum(): void {
    this.poolAccum.fill(0)
    this.accumTicks = 0
  }

  private beginTrial(): void {
    if (this.block.length === 0) {
      this.block = shuffleInPlace([0, 1, 2], this.order)
    }
    const label = this.block.pop()!
    this.stepper = this.teacher.beginTrial(this.org, M1_PATTERNS[label], label)
  }
}
