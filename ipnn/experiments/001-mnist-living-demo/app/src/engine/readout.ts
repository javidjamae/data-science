// What the organism is saying *right now*, under a stimulus that is simply
// held there.
//
// TrialStepper answers a different question: it waits for the first moment a
// spoken answer forms, commits to it, and ends the trial. That is the right
// semantics for a teacher running a lesson, but it can only ever observe one
// answer per exposure — so it structurally cannot see an organism change its
// mind. Under sustained exposure the answer has to be free-running.
//
// Same convention as the teacher, so the two agree on what "spoken" means:
// an output is spoken when it fired at least `threshold` times within the
// last `window` ticks. The difference is only that this never terminates, and
// that it lets the answer fall back to silence and re-form.
//
// This is the instrument experiment 002 needs (experiment-ideas §A1: dwell,
// revision, settling), which is why it lives in engine/ rather than in the
// demo: headless A1 runs use it with no UI attached.

export interface ReadoutConfig {
  outputSize: number
  /** sliding window (ticks) for the spoken-output readout */
  window: number
  /** fires within the window required to count as "spoken" */
  threshold: number
}

export class SustainedReadout {
  readonly cfg: ReadoutConfig

  /** the currently spoken output, or null for silence */
  answer: number | null = null
  /** ticks the current answer (including silence) has been held */
  dwell = 0
  /** how many times the answer has changed since the last reset, counting
   * transitions into and out of silence */
  switches = 0
  /** how many times it changed its mind: one spoken answer replaced by a
   * *different* spoken answer, ignoring any silence in between. This is the
   * "it said 6, then said 7" counter — §A1's headline measure. */
  revisions = 0
  /** the most recent non-silent answer, or null if it has never spoken */
  lastSpoken: number | null = null
  /** ticks observed since the last reset */
  ticks = 0
  /** per-output fire counts inside the sliding window (observable) */
  readonly counts: Int32Array
  /** ticks spent on each answer; last slot is silence */
  readonly occupancy: Float64Array
  /** dwell lengths of completed answer episodes, oldest first */
  readonly dwells: number[] = []

  private readonly win: Int8Array
  private t = 0

  constructor(cfg: ReadoutConfig) {
    this.cfg = cfg
    this.counts = new Int32Array(cfg.outputSize)
    this.occupancy = new Float64Array(cfg.outputSize + 1)
    this.win = new Int8Array(cfg.window).fill(-1)
  }

  /**
   * Observe one tick. `winner` is the organism's fired output this tick, or
   * -1 for silence — i.e. `org.lastWinner`, read straight after `org.tick()`.
   */
  observe(winner: number): void {
    const { window, threshold, outputSize } = this.cfg

    const slot = this.t % window
    const old = this.win[slot]
    if (old >= 0) this.counts[old]--
    this.win[slot] = winner as any
    if (winner >= 0) this.counts[winner]++
    this.t++
    this.ticks++

    // Hysteresis, and it matters: an answer is claimed the moment some output
    // crosses the threshold, but it is only *released* when it falls back
    // under it. Without that, an answer sitting exactly at threshold would
    // flicker on and off every tick and every one of those flickers would be
    // counted as the organism changing its mind.
    const prev = this.answer
    if (winner >= 0 && this.counts[winner] >= threshold) {
      this.answer = winner
    } else if (this.answer !== null && this.counts[this.answer] < threshold) {
      this.answer = null
    }

    if (this.answer !== prev) {
      if (this.dwell > 0) this.dwells.push(this.dwell)
      this.switches++
      this.dwell = 0
      if (this.answer !== null) {
        if (this.lastSpoken !== null && this.answer !== this.lastSpoken) {
          this.revisions++
        }
        this.lastSpoken = this.answer
      }
    }
    this.dwell++

    this.occupancy[this.answer === null ? outputSize : this.answer]++
  }

  /** Fraction of observed ticks spent on each answer (last slot: silence). */
  shares(): number[] {
    const n = Math.max(1, this.ticks)
    return Array.from(this.occupancy, (v) => v / n)
  }

  /** Mean length of completed answer episodes, in ticks (0 if none yet). */
  meanDwell(): number {
    if (this.dwells.length === 0) return 0
    return this.dwells.reduce((a, b) => a + b, 0) / this.dwells.length
  }

  /** Start a fresh exposure: clears the window and all statistics. */
  reset(): void {
    this.answer = null
    this.dwell = 0
    this.switches = 0
    this.revisions = 0
    this.lastSpoken = null
    this.ticks = 0
    this.t = 0
    this.counts.fill(0)
    this.occupancy.fill(0)
    this.dwells.length = 0
    this.win.fill(-1)
  }
}
