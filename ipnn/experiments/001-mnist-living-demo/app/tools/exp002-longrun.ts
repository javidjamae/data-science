// "How do you know a signal won't develop, given more time?"
//
// The honest answer is that the M1 diagnosis was a SNAPSHOT — discriminability
// measured once, after 1,000 trials — and a snapshot cannot distinguish
// "there is nothing here" from "there is something here, arriving slowly".
// A system that took 20,000 trials to depart from chance would look identical
// at trial 1,000. So would a system that never departs.
//
// This tool separates them, three ways at once, over a run 16× the length of
// the gate:
//
//  1. TRAJECTORY, not endpoint. Every metric is recorded at checkpoints across
//     the whole run. The question is not "is it at chance" but "is anything
//     moving in the right direction". A slow learner must show *something*
//     rising while accuracy is still flat — otherwise there is no mechanism
//     by which time helps.
//
//  2. POPULATION DECODING, not per-node rates. The original measure asked
//     whether a single node's firing rate depends on the stimulus. That is
//     insensitive by construction: 200 nodes each carrying 0.1σ would show
//     nothing per-node while the population carried plenty. So this also asks
//     the sensitive question — can a nearest-centroid decoder read the label
//     off the whole population at each depth? If a decoder can beat chance,
//     there is signal, whatever the per-node numbers say.
//
//  3. A POSITIVE CONTROL. Every metric is run on the shallow arm too, where
//     learning demonstrably happens. A metric that stays flat in *both* arms
//     is a broken instrument, not evidence. This is what makes a flat M1
//     trajectory mean something.
//
// Also measured: structural persistence — what fraction of the edges alive at
// one checkpoint are still alive at the next. This is the mechanistic half of
// the argument. A system that reaches a steady state and stays there does not
// benefit from more time; a system still accumulating might.
//
//   npx vite-node tools/exp002-longrun.ts [trials] [seed]

import { GrownOrganism } from '../src/engine/grown/grown-organism'
import { defaultGrownConfig, type GrownConfig } from '../src/engine/grown/config'
import { ROLE_INTERIOR } from '../src/engine/grown/lattice'
import { AutoTeacher } from '../src/engine/teacher'
import { M1_PATTERNS } from '../src/engine/patterns'
import { defaultTeacherConfig } from '../src/engine/types'
import { mulberry32, shuffleInPlace, type Rng } from '../src/engine/rng'

const argv = process.argv.slice(2)
const TOTAL = argv.length > 0 ? Number(argv[0]) : 32_000
const SEED = argv.length > 1 ? Number(argv[1]) : 1

const CHECKPOINTS = (() => {
  const cs: number[] = []
  for (let t = 250; t < TOTAL; t *= 2) cs.push(t)
  cs.push(TOTAL)
  return cs
})()

/** ticks per pattern during a probe */
const PROBE = 2400
/** ticks per decoding sample — a population rate vector */
const WINDOW = 20

// ─────────────────────────── the null this whole argument rests on ─────────
//
// The per-node statistic is (max − min) of three per-pattern firing rates,
// divided by the standard error of a rate. Under "this node does not care
// which pattern is showing", those three estimates are three independent
// draws from the same distribution, so the statistic is the RANGE OF THREE
// STANDARD NORMALS. Its expectation is exactly 3/√π ≈ 1.6926.
//
// That closed form is the reason 1.7 is not "a small number" but "the number".
// The tail matters too, though: if 10% of nodes exceed 3σ, is that signal or
// is that what noise does? Monte Carlo answers it rather than intuition.
function nullRangeDistribution(rng: Rng, n = 200_000): { mean: number; pAbove3: number } {
  let sum = 0
  let above = 0
  for (let i = 0; i < n; i++) {
    let lo = Infinity
    let hi = -Infinity
    for (let k = 0; k < 3; k++) {
      // Box-Muller
      const u = Math.max(1e-12, rng())
      const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng())
      if (z < lo) lo = z
      if (z > hi) hi = z
    }
    const r = hi - lo
    sum += r
    if (r > 3) above++
  }
  return { mean: sum / n, pAbove3: above / n }
}

// ────────────────────────────────────────────────────────────── the probe

interface DepthReport {
  depth: number
  nodes: number
  meanSigma: number
  fracAbove3: number
  decode: number
}

interface Checkpoint {
  trial: number
  accuracy: number
  edges: number
  persistence: number
  depths: DepthReport[]
  within2: number
}

function hopMap(org: GrownOrganism): Int32Array {
  const lat = org.lattice
  const E = org.edges
  const dist = new Int32Array(lat.size).fill(-1)
  const q = new Int32Array(lat.size)
  let head = 0
  let tail = 0
  for (const i of lat.inputNodes) {
    dist[i] = 0
    q[tail++] = i
  }
  while (head < tail) {
    const i = q[head++]
    const [a, b] = E.outRange(i)
    for (let s = a; s < b; s++) {
      if (Math.abs(E.w[s]) < org.cfg.deathThreshold) continue
      const j = E.post[s]
      if (dist[j] !== -1) continue
      dist[j] = dist[i] + 1
      q[tail++] = j
    }
  }
  return dist
}

/**
 * Freeze the organism and measure. Learning off AND rent off — rent would
 * decay every weight past the death threshold during a probe this long and
 * the tool would faithfully report a disconnected network (L-015).
 */
function probe(org: GrownOrganism, teacher: AutoTeacher, nullDist: { pAbove3: number }) {
  const savedRent = org.cfg.rent
  const savedLearning = teacher.learning
  const savedSleep = org.cfg.sleepEvery
  org.cfg.rent = 0
  org.cfg.sleepEvery = Number.MAX_SAFE_INTEGER
  teacher.learning = false

  const lat = org.lattice
  const dist = hopMap(org)

  // per-tick firing, per pattern, retained so both measures read the same data
  const samples: Float32Array[][] = [[], [], []]
  const rate: Float32Array[] = [0, 1, 2].map(() => new Float32Array(lat.size))
  for (let label = 0; label < 3; label++) {
    org.sense.set(M1_PATTERNS[label])
    org.clearTraces()
    for (let t = 0; t < 200; t++) org.tick()
    let win = new Float32Array(lat.size)
    for (let t = 0; t < PROBE; t++) {
      org.tick()
      for (let i = 0; i < lat.size; i++) {
        rate[label][i] += org.poolFired[i]
        win[i] += org.poolFired[i]
      }
      if ((t + 1) % WINDOW === 0) {
        for (let i = 0; i < lat.size; i++) win[i] /= WINDOW
        samples[label].push(win)
        win = new Float32Array(lat.size)
      }
    }
    for (let i = 0; i < lat.size; i++) rate[label][i] /= PROBE
  }

  const depths: DepthReport[] = []
  for (let d = 1; d <= 4; d++) {
    const nodes: number[] = []
    for (let i = 0; i < lat.size; i++) {
      if (lat.role[i] !== ROLE_INTERIOR) continue
      if (d < 4 ? dist[i] === d : dist[i] >= 4) nodes.push(i)
    }
    if (nodes.length === 0) {
      depths.push({ depth: d, nodes: 0, meanSigma: NaN, fracAbove3: NaN, decode: NaN })
      continue
    }

    // (a) per-node: range of three rates, in standard errors
    let sum = 0
    let above = 0
    for (const i of nodes) {
      const r = [rate[0][i], rate[1][i], rate[2][i]]
      const mean = (r[0] + r[1] + r[2]) / 3
      const se = Math.sqrt(Math.max(1e-9, (mean * (1 - mean)) / PROBE))
      const sigma = (Math.max(...r) - Math.min(...r)) / se
      sum += sigma
      if (sigma > 3) above++
    }

    // (b) population: nearest-centroid decoding on held-out windows. Far more
    // sensitive than (a) — it pools whatever each node individually carries.
    const half = Math.floor(samples[0].length / 2)
    const centroid = [0, 1, 2].map((l) => {
      const c = new Float64Array(nodes.length)
      for (let s = 0; s < half; s++) {
        for (let k = 0; k < nodes.length; k++) c[k] += samples[l][s][nodes[k]]
      }
      for (let k = 0; k < nodes.length; k++) c[k] /= half
      return c
    })
    let correct = 0
    let total = 0
    for (let l = 0; l < 3; l++) {
      for (let s = half; s < samples[l].length; s++) {
        let bestL = -1
        let bestD = Infinity
        for (let c = 0; c < 3; c++) {
          let dd = 0
          for (let k = 0; k < nodes.length; k++) {
            const diff = samples[l][s][nodes[k]] - centroid[c][k]
            dd += diff * diff
          }
          if (dd < bestD) {
            bestD = dd
            bestL = c
          }
        }
        if (bestL === l) correct++
        total++
      }
    }

    depths.push({
      depth: d,
      nodes: nodes.length,
      meanSigma: sum / nodes.length,
      fracAbove3: above / nodes.length,
      decode: total ? correct / total : NaN,
    })
  }
  void nullDist

  org.cfg.rent = savedRent
  org.cfg.sleepEvery = savedSleep
  teacher.learning = savedLearning
  org.clearTraces()
  return depths
}

// ───────────────────────────────────────────────────────────────── the run

function runArm(name: string, cfgOver: Partial<GrownConfig>, nullDist: { pAbove3: number }) {
  const org = new GrownOrganism({ ...defaultGrownConfig, ...cfgOver, seed: SEED })
  const teacher = new AutoTeacher({ ...defaultTeacherConfig })
  const order = mulberry32(SEED * 7919 + 1)
  const results: boolean[] = []
  let block: number[] = []
  let prevEdges = new Set<number>()
  const out: Checkpoint[] = []

  let done = 0
  for (const cp of CHECKPOINTS) {
    while (done < cp) {
      if (block.length === 0) block = shuffleInPlace([0, 1, 2], order)
      const label = block.pop()!
      results.push(teacher.runTrial(org, M1_PATTERNS[label], label).correct)
      done++
    }

    const live = new Set<number>()
    for (let s = 0; s < org.edges.count; s++) {
      live.add(org.edges.pre[s] * org.lattice.size + org.edges.post[s])
    }
    let survived = 0
    for (const k of prevEdges) if (live.has(k)) survived++
    const persistence = prevEdges.size ? survived / prevEdges.size : NaN
    prevEdges = live

    const depths = probe(org, teacher, nullDist)
    const h = org.stats().inputHops
    const tail = results.slice(-200)
    out.push({
      trial: cp,
      accuracy: tail.filter(Boolean).length / tail.length,
      edges: org.edges.count,
      persistence,
      depths,
      within2: (h[1] ?? 0) + (h[2] ?? 0),
    })
    const d2 = depths.find((d) => d.depth === 2)!
    console.log(
      `  ${name} @${String(cp).padStart(6)} trials: acc ${out[out.length - 1].accuracy.toFixed(3)}` +
        `  hop2 σ ${Number.isNaN(d2.meanSigma) ? ' n/a ' : d2.meanSigma.toFixed(2)}` +
        `  hop2 decode ${Number.isNaN(d2.decode) ? ' n/a ' : d2.decode.toFixed(3)}` +
        `  edges ${org.edges.count}  persist ${Number.isNaN(persistence) ? 'n/a' : (persistence * 100).toFixed(0) + '%'}`
    )
  }
  return out
}

const nullDist = nullRangeDistribution(mulberry32(20260822))
console.log('NULL for the per-node statistic (range of 3 standard normals)')
console.log(
  `  closed form E[range] = 3/√π = ${(3 / Math.sqrt(Math.PI)).toFixed(4)}   ` +
    `Monte Carlo = ${nullDist.mean.toFixed(4)}`
)
console.log(
  `  P(range > 3σ) under the null = ${(nullDist.pAbove3 * 100).toFixed(1)}%  ` +
    `← what fraction of nodes "look significant" with no signal at all`
)
console.log(`  population decoding chance = 0.333\n`)
console.log(`${TOTAL} trials, seed ${SEED}, checkpoints ${CHECKPOINTS.join(', ')}\n`)

const arms = [
  { name: 'M1     ', cfg: {} },
  { name: 'shallow', cfg: { outputX: 14 } },
]
const all = arms.map((a) => ({ name: a.name, cps: runArm(a.name, a.cfg, nullDist) }))

console.log('\n══ trajectories ═══════════════════════════════════════════════')
for (const arm of all) {
  console.log(`\n${arm.name.trim()}`)
  console.log('  trials |  acc  | hop1 σ | hop2 σ | hop2 decode | hop3 decode | edges | persist')
  console.log('  -------|-------|--------|--------|-------------|-------------|-------|--------')
  for (const c of arm.cps) {
    const g = (d: number) => c.depths.find((x) => x.depth === d)!
    const f = (v: number, p = 3) => (Number.isNaN(v) ? ' n/a ' : v.toFixed(p))
    console.log(
      `  ${String(c.trial).padStart(6)} | ${c.accuracy.toFixed(3)} |  ${f(g(1).meanSigma, 2)}  |` +
        `  ${f(g(2).meanSigma, 2)}  |    ${f(g(2).decode)}    |    ${f(g(3).decode)}    |` +
        ` ${String(c.edges).padStart(5)} |  ${Number.isNaN(c.persistence) ? 'n/a' : (c.persistence * 100).toFixed(0) + '%'}`
    )
  }
}

console.log('\n══ is anything moving? ════════════════════════════════════════')
for (const arm of all) {
  const first = arm.cps[0]
  const last = arm.cps[arm.cps.length - 1]
  const d2 = (c: Checkpoint) => c.depths.find((x) => x.depth === 2)!
  const delta = (a: number, b: number) => (b - a >= 0 ? '+' : '') + (b - a).toFixed(3)
  console.log(
    `${arm.name}  accuracy ${delta(first.accuracy, last.accuracy)}   ` +
      `hop2 σ ${delta(d2(first).meanSigma, d2(last).meanSigma)}   ` +
      `hop2 decode ${delta(d2(first).decode, d2(last).decode)}   ` +
      `over ${first.trial} → ${last.trial} trials`
  )
}
