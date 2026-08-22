// Edges, and time of flight (design §5).
//
// An edge carries a weight, an eligibility trace, an evidence count, and a
// **latency in ticks** set by its physical span:
//
//     d = max(1, ceil(span / v)),  v > 1
//
// so a long edge is *temporally cheaper than the chain of short hops it
// replaces*: with v = 3, one span-3 edge delivers in 1 tick where three unit
// hops take 3. Long-range edges are the fast path. This is why the brain
// myelinates long tracts — every synaptic relay costs real time — and it is
// what M3's EARLY/LATE reward schedules are designed to exploit.
//
// ── The correctness trap, called out in the design because it would silently
//    ruin everything ────────────────────────────────────────────────────────
// With delays, "pre and post were co-active" means the presynaptic spike
// *arrived* at t, having been emitted at t − d. Eligibility must be updated
// against the ARRIVAL, not against the presynaptic node's current state.
// Getting this wrong assigns credit to the wrong pairs and would look like
// "growth just doesn't work".
//
// The defence here is structural rather than disciplinary: nothing in this
// class ever exposes the presynaptic node's current firing state. The only
// per-edge signal it publishes is `arrived`, written by deliver() from the
// ring buffer, so the eligibility update in grown-organism.ts has no wrong
// thing available to read. `delays.test.ts` pins the behaviour.
//
// ── Structure is static during wake ───────────────────────────────────────
// All structural change is gated to sleep, so the adjacency can be a plain
// CSR built once per sleep: edges sorted by presynaptic node, with outStart
// indexing into them. Wake is then O(edges) per tick with no bookkeeping.

/** one edge, as growth and death see it */
export interface Edge {
  pre: number
  post: number
  w: number
  e: number
  n: number
  d: number
}

export class EdgeSet {
  readonly nNodes: number
  /** ring-buffer depth: maxDelay + 1 */
  readonly delaySlots: number

  count = 0
  pre = new Int32Array(0)
  post = new Int32Array(0)
  delay = new Int32Array(0)
  w = new Float32Array(0)
  e = new Float32Array(0)
  n = new Float32Array(0)

  /** per-edge: did a presynaptic spike arrive on *this* tick. The only
   * co-activity signal any caller is allowed to see. */
  arrived = new Uint8Array(0)

  /** pending arrivals: ring[edge*delaySlots + slot] */
  private ring = new Uint8Array(0)
  /** CSR over edges sorted by `pre` */
  private outStart: Int32Array

  constructor(nNodes: number, maxDelay: number) {
    if (maxDelay < 1) throw new Error('maxDelay must be at least 1 tick')
    this.nNodes = nNodes
    this.delaySlots = maxDelay + 1
    this.outStart = new Int32Array(nNodes + 1)
  }

  /**
   * Replace the whole structure. Called only from sleep.
   *
   * Every in-flight spike is dropped: edge indices are reassigned here, so a
   * ring keyed by the old indices would deliver to the wrong edges. Dropping
   * them is correct rather than merely convenient — rewiring happens with the
   * sense dark, between trials, so there is no thought to interrupt.
   */
  setStructure(edges: Edge[]): void {
    edges.sort((a, b) => a.pre - b.pre || a.post - b.post)

    const c = edges.length
    this.count = c
    this.pre = new Int32Array(c)
    this.post = new Int32Array(c)
    this.delay = new Int32Array(c)
    this.w = new Float32Array(c)
    this.e = new Float32Array(c)
    this.n = new Float32Array(c)
    this.arrived = new Uint8Array(c)
    this.ring = new Uint8Array(c * this.delaySlots)

    for (let i = 0; i < c; i++) {
      const ed = edges[i]
      if (ed.d < 1 || ed.d >= this.delaySlots) {
        throw new Error(
          `edge latency ${ed.d} is outside [1, ${this.delaySlots - 1}]; the ring ` +
            `buffer was sized for maxDelay ${this.delaySlots - 1}`
        )
      }
      this.pre[i] = ed.pre
      this.post[i] = ed.post
      this.delay[i] = ed.d
      this.w[i] = ed.w
      this.e[i] = ed.e
      this.n[i] = ed.n
    }

    this.outStart.fill(0)
    for (let i = 0; i < c; i++) this.outStart[this.pre[i] + 1]++
    for (let node = 0; node < this.nNodes; node++) {
      this.outStart[node + 1] += this.outStart[node]
    }
  }

  /** read the structure back out — what sleep applies growth and death to */
  toEdges(): Edge[] {
    const out: Edge[] = new Array(this.count)
    for (let i = 0; i < this.count; i++) {
      out[i] = {
        pre: this.pre[i],
        post: this.post[i],
        w: this.w[i],
        e: this.e[i],
        n: this.n[i],
        d: this.delay[i],
      }
    }
    return out
  }

  outDegree(node: number): number {
    return this.outStart[node + 1] - this.outStart[node]
  }

  /** edge indices leaving `node`, as a [start, end) range into the arrays */
  outRange(node: number): [number, number] {
    return [this.outStart[node], this.outStart[node + 1]]
  }

  /**
   * Deliver everything scheduled for this tick into `drive`, and publish the
   * per-edge arrival flags the eligibility update reads.
   */
  deliver(tick: number, drive: Float32Array): void {
    const slots = this.delaySlots
    const slot = tick % slots
    for (let i = 0; i < this.count; i++) {
      const r = i * slots + slot
      if (this.ring[r]) {
        this.ring[r] = 0
        this.arrived[i] = 1
        drive[this.post[i]] += this.w[i]
      } else {
        this.arrived[i] = 0
      }
    }
  }

  /** `node` fired at `tick`: schedule its outgoing spikes to land at t + d */
  emitFrom(node: number, tick: number): void {
    const slots = this.delaySlots
    const end = this.outStart[node + 1]
    for (let i = this.outStart[node]; i < end; i++) {
      this.ring[i * slots + ((tick + this.delay[i]) % slots)] = 1
    }
  }

  /** drop every in-flight spike without touching weights or structure */
  clearInFlight(): void {
    this.ring.fill(0)
    this.arrived.fill(0)
  }
}

/** d = max(1, ceil(span / v)) — or 1 for every edge in the uniform-latency
 * control arm, where span is measured but never charged for. */
export function latencyForSpan(
  span: number,
  conductionSpeed: number,
  mode: 'span' | 'uniform'
): number {
  if (mode === 'uniform') return 1
  return Math.max(1, Math.ceil(span / conductionSpeed))
}
