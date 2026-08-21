// DOM-free controller for the M1 living demo: owns the organism, the
// teacher, and the balanced presentation schedule (identical to the M1
// test harness), and exposes tick-batched stepping plus the observables a
// renderer needs. Nothing in this file touches the DOM — it runs headless
// under vitest, which is how we know the demo shows the same organism the
// gate certified.

import { Organism } from '../engine/organism'
import type { OrganismLike } from '../engine/types'
import { AutoTeacher, TrialStepper } from '../engine/teacher'
import { M1_PATTERNS } from '../engine/patterns'
import { SustainedReadout } from '../engine/readout'
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

/**
 * `auto` — the teacher runs lessons: presents, judges, rewards.
 * `manual` — you choose what is on the sense and it stays there. No teacher,
 * therefore no reward, therefore no weight change: manual mode observes the
 * organism, it never trains it. (Delivering reward by hand is live coaching —
 * experiment-ideas §A2 — and is deliberately not in this mode yet.)
 */
export type SimMode = 'auto' | 'manual'

/** blank sense, used when the manual stimulus is cleared */
const BLANK = new Uint8Array(defaultConfig.senseSize)

export class DemoSim {
  seed: number
  org!: OrganismLike
  teacher!: AutoTeacher
  stepper!: TrialStepper

  /** completed trials, in order */
  trials: TrialResult[] = []
  /** rolling accuracy (last ≤100 trials) recorded after every trial */
  accuracyCurve: number[] = []
  /** learning on/off toggle history (chart shading) */
  marks: ToggleMark[] = []

  /** auto = teacher-driven lessons; manual = you hold a stimulus and watch */
  mode: SimMode = 'auto'
  /** in manual mode: which pattern is on the sense, or null for cleared */
  manualLabel: number | null = null
  /** free-running answer readout — only advanced in manual mode */
  readout!: SustainedReadout

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
    this.mode = 'auto'
    this.manualLabel = null
    this.readout = new SustainedReadout({
      outputSize: this.org.cfg.outputSize,
      window: this.teacher.cfg.spokenWindow,
      threshold: this.teacher.cfg.spokenThreshold,
    })
    this.beginTrial()
  }

  /**
   * Switch between teacher-driven lessons and held-stimulus observation.
   * Traces are cleared on every switch: they carry ~30 ticks of credit, and
   * credit earned during a free-run must not be paid out by the next lesson's
   * reward (or vice versa). Weights are untouched.
   */
  setMode(mode: SimMode): void {
    if (mode === this.mode) return
    this.mode = mode
    this.org.clearTraces()
    this.readout.reset()
    if (mode === 'manual') {
      this.manualLabel = null
      this.org.sense.set(BLANK)
    } else {
      this.manualLabel = null
      this.beginTrial()
    }
  }

  /**
   * Put a pattern on the sense and leave it there (null clears it). Resets
   * the readout, so dwell and revision counts describe *this* exposure.
   */
  setManualStimulus(label: number | null): void {
    if (this.mode !== 'manual') return
    this.manualLabel = label
    this.org.sense.set(label === null ? BLANK : M1_PATTERNS[label])
    this.readout.reset()
  }

  get learning(): boolean {
    return this.teacher.learning
  }

  setLearning(on: boolean): void {
    if (on === this.teacher.learning) return
    this.teacher.learning = on
    this.marks.push({ trial: this.trials.length, learning: on })
  }

  /** the pattern currently on the sense, or null when it is blank */
  get currentLabel(): number | null {
    if (this.mode === 'manual') return this.manualLabel
    return this.stepper.phase === 'stimulus' ? this.stepper.label : null
  }

  /** what the organism is saying right now, or null for silence. In auto
   * mode this is the teacher's committed answer for the trial in progress. */
  get spoken(): number | null {
    return this.mode === 'manual' ? this.readout.answer : this.stepper.spoken
  }

  /** windowed per-output fire counts driving the "spoken" decision */
  get spokenCounts(): Int32Array {
    return this.mode === 'manual' ? this.readout.counts : this.stepper.counts
  }

  /** rolling accuracy over the last ≤100 completed trials */
  get rollingAccuracy(): number {
    const n = Math.min(this.trials.length, ROLLING_WINDOW)
    return n === 0 ? 0 : this.correctInWindow / n
  }

  /** Advance the world by n ticks (trials begin/end as they may). */
  tick(n: number): void {
    if (this.mode === 'manual') {
      this.tickManual(n)
      return
    }
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

  /**
   * Manual mode: the sense holds whatever was put on it, the organism ticks,
   * and the readout watches. No trial machinery, no judgment, no reward — so
   * `applyReward` is never reached and no weight can move.
   */
  private tickManual(n: number): void {
    for (let i = 0; i < n; i++) {
      this.org.tick()
      this.readout.observe(this.org.lastWinner)

      const fired = this.org.poolFired
      for (let j = 0; j < fired.length; j++) this.poolAccum[j] += fired[j]
      this.accumTicks++
      this.totalTicks++
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
