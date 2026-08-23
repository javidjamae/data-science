// Experiment 002 — the grown substrate, watchable.
//
// 001's demo shows an organism learning. This one shows an organism *building
// itself*, and then shows why that was not enough. The sheet starts with zero
// edges; wiring appears at every sleep, most of it dies before the next one,
// and what survives is what earned. That churn is the experiment.
//
// The panel that matters is the sense-depth histogram. M1's gate was failed
// not because the rule is wrong but because no sense pixel ever ends up closer
// than five hops to an answer, and stimulus information survives exactly one.
// Switch to the shallow arm — which changes nothing but where the outputs sit
// — and the histogram fills in at 1 and 2 hops while the accuracy curve lifts
// off the chance line. Same rule, same rent, same seed.

import { DemoSim } from '../demo-m1/sim'
import { grownSim, grownOrganism, GROWN_ARMS, type GrownArm } from './sim'
import type { GrownOrganism, GrownStats } from '../engine/grown/grown-organism'
import { ROLE_INPUT, ROLE_OUTPUT } from '../engine/grown/lattice'
import { M1_PATTERNS } from '../engine/patterns'

const SERIES = ['#3E8EDE', '#C08A18', '#DE5FA5']
const PATTERN_NAMES = ['vertical bars', 'horizontal bars', 'diagonal X']
const SCREEN_BG = '#0E1116'
const EXCITE = '#3FB950'
const INHIBIT = '#E05A7D'
const REPO = 'https://github.com/javidjamae/data-science/blob/master/ipnn'
const CHANCE = 1 / 3
const GATE = 0.8

const STYLE = `
:root {
  --bg: #F5F7FA; --ink: #1A2129; --ink-2: #57626E; --ink-3: #85909C;
  --panel: #FFFFFF; --border: #DCE2E9; --accent: #0969DA; --good: #1A7F37;
  --bad: #C4351C;
  --screen: ${SCREEN_BG}; --screen-border: #2A313C; --screen-ink: #8C99A8;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0D1117; --ink: #E6EDF3; --ink-2: #9BA7B5; --ink-3: #788593;
    --panel: #161B22; --border: #262D37; --accent: #58A6FF; --good: #3FB950;
    --bad: #F85149;
  }
}
:root[data-theme="dark"] {
  --bg: #0D1117; --ink: #E6EDF3; --ink-2: #9BA7B5; --ink-3: #788593;
  --panel: #161B22; --border: #262D37; --accent: #58A6FF; --good: #3FB950;
  --bad: #F85149;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font-family: var(--sans); line-height: 1.45; }
.demo { max-width: 1120px; margin: 0 auto; padding: 20px 20px 40px; }
header h1 { font-family: var(--mono); font-size: 1.15rem; font-weight: 600;
  letter-spacing: 0.02em; margin: 0; }
header .tagline { color: var(--ink-2); margin: 4px 0 0; font-size: 0.92rem; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 14px;
  margin: 18px 0 10px; padding: 10px 14px; background: var(--panel);
  border: 1px solid var(--border); border-radius: 8px;
  font-family: var(--mono); font-size: 0.8rem; }
.controls label { display: flex; align-items: center; gap: 7px; color: var(--ink-2); }
button { font: inherit; font-family: var(--mono); cursor: pointer;
  background: var(--panel); color: var(--ink); border: 1px solid var(--border);
  border-radius: 6px; padding: 5px 14px; }
button:hover { border-color: var(--accent); }
button:focus-visible, input:focus-visible, select:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px; }
#run { min-width: 84px; }
/* reserving the width stops the readout shoving its neighbours around as the
   number gains digits — the layout-stability rule from L-008 */
#speedlbl { display: inline-block; min-width: 11ch; }
.segmented { display: inline-flex; border: 1px solid var(--border); border-radius: 6px; }
.segmented button { border: 0; background: none; padding: 5px 12px; }
.segmented button + button { border-left: 1px solid var(--border); }
.segmented button[aria-pressed="true"] { background: var(--accent); color: #FFF;
  border-radius: 5px; }
.segmented button[aria-pressed="true"] + button { border-left-color: transparent; }
input[type="range"] { width: 120px; accent-color: var(--accent); }
input[type="checkbox"] { width: 16px; height: 16px; accent-color: var(--accent); }
input[type="number"] { font: inherit; width: 58px; padding: 4px 6px;
  background: var(--bg); color: var(--ink); border: 1px solid var(--border);
  border-radius: 6px; }
select { font: inherit; font-family: var(--mono); padding: 4px 6px;
  background: var(--bg); color: var(--ink); border: 1px solid var(--border);
  border-radius: 6px; }
.armnote { margin: 0 0 14px; padding: 9px 13px; border-radius: 8px;
  border: 1px solid var(--border); border-left: 3px solid var(--accent);
  background: var(--panel); color: var(--ink-2); font-size: 0.84rem; }
.layout { display: grid; grid-template-columns: 528px 1fr; gap: 14px;
  align-items: start; }
@media (max-width: 940px) { .layout { grid-template-columns: 1fr; } }
.col { display: grid; gap: 14px; min-width: 0; }
.panel { background: var(--panel); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 14px; min-width: 0; }
.panel h2 { margin: 0 0 8px; font-family: var(--mono); font-size: 0.72rem;
  font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--ink-3); display: flex; justify-content: space-between;
  align-items: center; gap: 10px; }
.screen { background: var(--screen); border: 1px solid var(--screen-border);
  border-radius: 6px; padding: 8px; }
.screen canvas { display: block; width: 100%; height: auto; }
.cap { font-family: var(--mono); font-size: 0.74rem; color: var(--ink-3);
  margin: 8px 0 0; min-height: 1.3em; }
.badge { font-family: var(--mono); font-size: 0.66rem; padding: 2px 8px;
  border-radius: 999px; border: 1px solid var(--border); color: var(--ink-3);
  white-space: nowrap; }
.badge.on { border-color: var(--accent); color: var(--accent); }
.legend { display: flex; flex-wrap: wrap; gap: 10px 14px; margin-top: 9px;
  font-family: var(--mono); font-size: 0.7rem; color: var(--ink-3); }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.swatch { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
.outs { display: grid; gap: 7px; }
.out { display: grid; grid-template-columns: 8.5rem 1fr 3.2rem; align-items: center;
  gap: 9px; font-family: var(--mono); font-size: 0.76rem; }
.bar { height: 9px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; }
.bar i { display: block; height: 100%; width: 0; border-radius: 999px; }
.stats { font-family: var(--mono); font-size: 0.76rem; color: var(--ink-2);
  display: grid; gap: 3px; }
.stats b { color: var(--ink); font-weight: 600; }
.depth { display: grid; gap: 4px; margin-top: 4px; }
.depthrow { display: grid; grid-template-columns: 4.6rem 1fr 2.4rem;
  align-items: center; gap: 8px; font-family: var(--mono); font-size: 0.72rem; }
.depthbar { height: 10px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 3px; overflow: hidden; }
.depthbar i { display: block; height: 100%; width: 0; }
.note { color: var(--ink-3); font-size: 0.78rem; margin: 9px 0 0; }
a { color: var(--accent); }
footer { margin-top: 22px; color: var(--ink-3); font-size: 0.78rem; }
`

const app = document.querySelector<HTMLDivElement>('#app')!
const styleEl = document.createElement('style')
styleEl.textContent = STYLE
document.head.appendChild(styleEl)

app.innerHTML = `
<div class="demo">
  <header>
    <h1>IPNN · experiment 002 — the grown substrate</h1>
    <p class="tagline">
      A 32×32 sheet with <strong>zero connections</strong>. It grows its own
      wiring from spontaneous activity, pays rent on every connection, and
      learns from reward alone. Watch where it wires — and where the signal
      stops.
    </p>
  </header>

  <div class="controls">
    <button id="run">Run</button>
    <label>speed <input id="speed" type="range" min="0" max="100" value="55">
      <span id="speedlbl">≈300 ticks/s</span></label>
    <span class="segmented" role="group" aria-label="arm">
      <button id="armm1" aria-pressed="true">M1</button>
      <button id="armshallow" aria-pressed="false">Shallow</button>
      <button id="arminnate" aria-pressed="false">Innate</button>
    </span>
    <label><input id="learning" type="checkbox" checked> reward on</label>
    <label>overlay <select id="overlay">
      <option value="none">none</option>
      <option value="reward">reward field</option>
      <option value="activity">activity field</option>
    </select></label>
    <label>seed <input id="seed" type="number" value="1" min="1" step="1"></label>
    <button id="reset">Reset</button>
  </div>

  <p class="armnote" id="armnote"></p>

  <div class="layout">
    <div class="col">
      <section class="panel">
        <h2>the sheet <span class="badge" id="sleepbadge">awake</span></h2>
        <div class="screen"><canvas id="sheet"></canvas></div>
        <div class="legend">
          <span><i class="swatch" style="background:#3E8EDE"></i> input cortex</span>
          <span><i class="swatch" style="background:#C08A18"></i> reward locus</span>
          <span><i class="swatch" style="background:#8C99A8"></i> interior</span>
          <span><i class="swatch" style="background:${EXCITE}"></i> excitatory edge</span>
          <span><i class="swatch" style="background:${INHIBIT}"></i> inhibitory edge</span>
        </div>
        <p class="cap" id="sheetcap"></p>
      </section>

      <section class="panel">
        <h2>accuracy <span class="badge" id="accbadge"></span></h2>
        <div class="screen"><canvas id="chart"></canvas></div>
        <p class="cap">chance 0.33 · gate 0.80 · rolling over the last 100 trials</p>
      </section>
    </div>

    <div class="col">
      <section class="panel">
        <h2>what it sees <span class="badge" id="stimbadge"></span></h2>
        <div class="screen" style="width:154px"><canvas id="sense"></canvas></div>
      </section>

      <section class="panel">
        <h2>what it says</h2>
        <div class="outs" id="outs"></div>
        <p class="cap" id="outcap"></p>
      </section>

      <section class="panel">
        <h2>how far the sense is from an answer</h2>
        <div class="depth" id="depth"></div>
        <p class="note" id="depthnote"></p>
      </section>

      <section class="panel">
        <h2>structure</h2>
        <div class="stats" id="stats"></div>
      </section>
    </div>
  </div>

  <footer>
    Substrate, gates and the full negative result:
    <a href="${REPO}/experiments/002-grown-substrate/design.md">design</a> ·
    <a href="${REPO}/journal/entries/2026-08-22-0208-exp002-m0-built-m1-fails-on-depth.md">journal entry</a>
  </footer>
</div>
`

const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T

const runBtn = $<HTMLButtonElement>('run')
const speedInput = $<HTMLInputElement>('speed')
const speedLbl = $('speedlbl')
const armM1 = $<HTMLButtonElement>('armm1')
const armShallow = $<HTMLButtonElement>('armshallow')
const armInnate = $<HTMLButtonElement>('arminnate')
const armNote = $('armnote')
const learningInput = $<HTMLInputElement>('learning')
const overlaySel = $<HTMLSelectElement>('overlay')
const seedInput = $<HTMLInputElement>('seed')
const resetBtn = $<HTMLButtonElement>('reset')
const sleepBadge = $('sleepbadge')
const stimBadge = $('stimbadge')
const accBadge = $('accbadge')
const sheetCap = $('sheetcap')
const outCap = $('outcap')
const outsEl = $('outs')
const depthEl = $('depth')
const depthNote = $('depthnote')
const statsEl = $('stats')

const dpr = Math.min(2, window.devicePixelRatio || 1)

function setupCanvas(c: HTMLCanvasElement, w: number, h: number): CanvasRenderingContext2D {
  c.width = Math.round(w * dpr)
  c.height = Math.round(h * dpr)
  c.style.width = `${w}px`
  c.style.height = `${h}px`
  const ctx = c.getContext('2d')!
  ctx.scale(dpr, dpr)
  return ctx
}

// ------------------------------------------------------------------- state

let arm: GrownArm = 'm1'
let sim: DemoSim = grownSim(1, arm)
let running = false
let last = 0

// stats() walks the edge list twice (BFS forward and backward), so it runs on
// a timer rather than per frame — it is telemetry, not animation
let stats: GrownStats = grownOrganism(sim).stats()
let statsAt = 0
let outEdges = 0
let lastSleeps = 0
let sleepFlashUntil = 0

const SHEET_PX = 496
const sheetCtx = setupCanvas($<HTMLCanvasElement>('sheet'), SHEET_PX, SHEET_PX)
const senseCtx = setupCanvas($<HTMLCanvasElement>('sense'), 136, 136)
const CHART_W = SHEET_PX
const CHART_H = 130
const chartCtx = setupCanvas($<HTMLCanvasElement>('chart'), CHART_W, CHART_H)

function org(): GrownOrganism {
  return grownOrganism(sim)
}

function targetTps(): number {
  // 20 → 4000 ticks/s, geometric: slow enough to watch a single spike, fast
  // enough to reach a sleep in a couple of seconds
  const t = Number(speedInput.value) / 100
  return Math.round(20 * Math.pow(200, t))
}

// -------------------------------------------------------------- the sheet

const outsHtml = PATTERN_NAMES.map(
  (name, k) => `
  <div class="out">
    <span style="color:${SERIES[k]}">${name}</span>
    <span class="bar"><i id="outbar${k}" style="background:${SERIES[k]}"></i></span>
    <span id="outval${k}" style="color:var(--ink-3)">0.00</span>
  </div>`
).join('')
outsEl.innerHTML = outsHtml

function drawSheet(): void {
  const o = org()
  const lat = o.lattice
  const n = lat.width
  const cell = SHEET_PX / n
  const ctx = sheetCtx

  ctx.fillStyle = SCREEN_BG
  ctx.fillRect(0, 0, SHEET_PX, SHEET_PX)

  // --- optional field heat, drawn under everything ---
  const mode = overlaySel.value
  if (mode !== 'none') {
    const field = mode === 'reward' ? o.rewardProfile : o.activity.values
    let max = 0
    for (let i = 0; i < field.length; i++) if (field[i] > max) max = field[i]
    if (max > 0) {
      const tint = mode === 'reward' ? '192,138,24' : '62,142,222'
      for (let i = 0; i < field.length; i++) {
        const v = field[i] / max
        if (v <= 0.01) continue
        ctx.fillStyle = `rgba(${tint},${(v * 0.5).toFixed(3)})`
        ctx.fillRect(lat.xOf(i) * cell, lat.yOf(i) * cell, cell, cell)
      }
    }
  }

  // --- edges, bucketed by strength so the whole sheet is a handful of
  //     strokes rather than thousands of them ---
  const E = o.edges
  const BUCKETS = 4
  const REF = 0.45 // weight at which an edge is drawn at full opacity
  const paths: Path2D[][] = [
    Array.from({ length: BUCKETS }, () => new Path2D()),
    Array.from({ length: BUCKETS }, () => new Path2D()),
  ]
  for (let s = 0; s < E.count; s++) {
    const w = E.w[s]
    const mag = Math.abs(w)
    if (mag < o.cfg.deathThreshold) continue
    const b = Math.min(BUCKETS - 1, Math.floor((mag / REF) * BUCKETS))
    const p = paths[w >= 0 ? 0 : 1][b]
    const a = E.pre[s]
    const c = E.post[s]
    p.moveTo((lat.xOf(a) + 0.5) * cell, (lat.yOf(a) + 0.5) * cell)
    p.lineTo((lat.xOf(c) + 0.5) * cell, (lat.yOf(c) + 0.5) * cell)
  }
  ctx.lineWidth = 1
  for (let sign = 0; sign < 2; sign++) {
    for (let b = 0; b < BUCKETS; b++) {
      ctx.strokeStyle = sign === 0 ? EXCITE : INHIBIT
      ctx.globalAlpha = 0.07 + 0.33 * ((b + 1) / BUCKETS)
      ctx.stroke(paths[sign][b])
    }
  }
  ctx.globalAlpha = 1

  // --- nodes ---
  const accum = sim.poolAccum
  const ticks = Math.max(1, sim.accumTicks)
  const r = Math.max(1.4, cell * 0.19)
  for (let i = 0; i < lat.size; i++) {
    const x = (lat.xOf(i) + 0.5) * cell
    const y = (lat.yOf(i) + 0.5) * cell
    const rate = Math.min(1, accum[i] / ticks / 0.4)
    const role = lat.role[i]

    if (role === ROLE_INPUT) {
      ctx.fillStyle = `rgba(62,142,222,${(0.3 + 0.7 * rate).toFixed(3)})`
      ctx.fillRect(x - cell * 0.34, y - cell * 0.34, cell * 0.68, cell * 0.68)
      continue
    }
    if (role === ROLE_OUTPUT) {
      // drawn larger than any other node, and outlined: three sites out of
      // 1,024 are otherwise very easy to lose in the wiring
      const k = Array.prototype.indexOf.call(lat.outputNodes, i)
      ctx.fillStyle = SERIES[k] ?? '#FFF'
      ctx.globalAlpha = 0.4 + 0.6 * rate
      ctx.fillRect(x - cell * 0.6, y - cell * 0.6, cell * 1.2, cell * 1.2)
      ctx.globalAlpha = 1
      ctx.strokeStyle = SERIES[k] ?? '#FFF'
      ctx.lineWidth = 1
      ctx.strokeRect(x - cell * 0.8, y - cell * 0.8, cell * 1.6, cell * 1.6)
      continue
    }
    // interior: dim grey at rest, pale when it has been firing
    const g = 0.16 + 0.84 * rate
    ctx.fillStyle = `rgba(140,153,168,${g.toFixed(3)})`
    ctx.beginPath()
    ctx.arc(x, y, r, 0, Math.PI * 2)
    ctx.fill()
  }

  // reward locus on top, so it is never buried by wiring
  ctx.strokeStyle = '#C08A18'
  ctx.lineWidth = 1.6
  for (let idx = 0; idx < lat.rewardNodes.length; idx++) {
    const i = lat.rewardNodes[idx]
    ctx.beginPath()
    ctx.arc((lat.xOf(i) + 0.5) * cell, (lat.yOf(i) + 0.5) * cell, cell * 0.5, 0, Math.PI * 2)
    ctx.stroke()
  }
}

// -------------------------------------------------------------- the sense

function drawSense(): void {
  const ctx = senseCtx
  const cell = 136 / 8
  ctx.fillStyle = SCREEN_BG
  ctx.fillRect(0, 0, 136, 136)
  const s = sim.org.sense
  for (let i = 0; i < 64; i++) {
    ctx.fillStyle = s[i] ? '#3E8EDE' : '#1B222B'
    ctx.fillRect((i % 8) * cell + 1, Math.floor(i / 8) * cell + 1, cell - 2, cell - 2)
  }
}

// ---------------------------------------------------------------- accuracy

function drawChart(): void {
  const W = CHART_W
  const H = CHART_H
  const ctx = chartCtx
  ctx.fillStyle = SCREEN_BG
  ctx.fillRect(0, 0, W, H)

  const y = (v: number) => H - 6 - v * (H - 14)

  ctx.setLineDash([3, 3])
  ctx.lineWidth = 1
  ctx.strokeStyle = '#4A5462'
  ctx.beginPath()
  ctx.moveTo(0, y(CHANCE))
  ctx.lineTo(W, y(CHANCE))
  ctx.stroke()
  ctx.strokeStyle = '#2F6F45'
  ctx.beginPath()
  ctx.moveTo(0, y(GATE))
  ctx.lineTo(W, y(GATE))
  ctx.stroke()
  ctx.setLineDash([])

  const curve = sim.accuracyCurve
  if (curve.length > 1) {
    // The window doubles in steps rather than tracking the data continuously.
    // A continuously-growing axis makes the curve crawl backwards under the
    // viewer; a fixed 2,000-trial axis leaves the first few hundred trials
    // squashed into nothing. Stepping is the compromise: rare, legible jumps.
    let span = 250
    while (span < curve.length && span < 4000) span *= 2
    // past the widest step the axis stops growing and scrolls instead, so the
    // line stays inside its box however long this is left running
    const start = Math.max(0, curve.length - span)
    ctx.strokeStyle = '#58A6FF'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    for (let i = start; i < curve.length; i++) {
      const px = ((i - start) / span) * W
      const py = y(curve[i])
      if (i === start) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.stroke()
  }
}

// ------------------------------------------------------------------ panels

function refreshStats(now: number): void {
  if (now - statsAt < 400) return
  statsAt = now
  const o = org()
  stats = o.stats()

  let into = 0
  for (let s = 0; s < o.edges.count; s++) {
    if (o.lattice.role[o.edges.post[s]] === ROLE_OUTPUT) into++
  }
  outEdges = into

  if (stats.sleeps !== lastSleeps) {
    lastSleeps = stats.sleeps
    sleepFlashUntil = now + 700
  }
}

function drawDepth(): void {
  const h = stats.inputHops
  const at = (d: number) => h[d] ?? 0
  let beyond = 0
  for (let d = 4; d < h.length; d++) beyond += h[d]

  // Fixed buckets, always all five rows even when empty. Both because the
  // granular version (5,6,7…13 hops) buried the point, and because a panel
  // whose row count changes as it runs shoves everything below it around —
  // the layout-stability rule from L-008.
  const rows = [
    { label: '1 hop', n: at(1), color: '#3FB950' },
    { label: '2 hops', n: at(2), color: '#C08A18' },
    { label: '3 hops', n: at(3), color: '#5A6673' },
    { label: '4+ hops', n: beyond, color: '#454E5A' },
    { label: 'no path', n: at(0), color: '#333B44' },
  ]

  depthEl.innerHTML = rows
    .map(
      (r) => `<div class="depthrow"><span style="color:var(--ink-3)">${r.label}</span>
        <span class="depthbar"><i style="width:${(r.n / 64) * 100}%;background:${r.color}"></i></span>
        <span style="color:var(--ink-2)">${r.n}</span></div>`
    )
    .join('')

  const within2 = (h[1] ?? 0) + (h[2] ?? 0)
  depthNote.textContent =
    within2 === 0
      ? 'Not one of the 64 sense pixels is within two hops of an answer. Stimulus information survives exactly one hop, so the answer neurons are reading noise — and no amount of reward can teach from noise.'
      : `${within2} of the 64 sense pixels sit within two hops of an answer. That is what the organism is actually learning from; the rest, further out, is noise.`
}

function drawStats(): void {
  const acc = sim.rollingAccuracy
  statsEl.innerHTML = `
    <div>edges live <b>${stats.edges.toLocaleString()}</b> ·
      born <b>${stats.edgesBorn.toLocaleString()}</b> ·
      died <b>${stats.edgesDied.toLocaleString()}</b></div>
    <div>rewirings (sleeps) <b>${stats.sleeps}</b> ·
      trials <b>${sim.trials.length.toLocaleString()}</b></div>
    <div>edges reaching an answer neuron <b>${outEdges}</b>
      <span style="color:var(--ink-3)">of ${stats.edges.toLocaleString()}</span></div>
    <div>rolling accuracy <b>${acc.toFixed(2)}</b>
      <span style="color:var(--ink-3)">chance 0.33</span></div>`
  accBadge.textContent = `${acc.toFixed(2)}`
  accBadge.className = acc >= GATE ? 'badge on' : 'badge'
}

function drawOutputs(): void {
  const probs = sim.org.outputProbs()
  const spoken = sim.spoken
  for (let k = 0; k < 3; k++) {
    const v = probs[k] ?? 0
    ;($(`outbar${k}`) as HTMLElement).style.width = `${Math.min(1, v) * 100}%`
    $(`outval${k}`).textContent = v.toFixed(2)
  }
  outCap.textContent =
    spoken === null
      ? 'saying nothing yet — firing rate over the last 20 ticks'
      : `says: ${PATTERN_NAMES[spoken]}`
}

// -------------------------------------------------------------------- loop

function frame(now: number): void {
  if (running) {
    const dt = Math.min(0.25, (now - last) / 1000)
    const n = Math.max(1, Math.round(targetTps() * dt))
    sim.tick(n)
  }
  last = now

  refreshStats(now)
  drawSheet()
  drawSense()
  drawChart()
  drawOutputs()
  drawDepth()
  drawStats()
  sim.clearAccum()

  const asleep = now < sleepFlashUntil
  sleepBadge.textContent = asleep ? 'rewiring' : 'awake'
  sleepBadge.className = asleep ? 'badge on' : 'badge'

  const label = sim.currentLabel
  stimBadge.textContent = label === null ? 'blank' : PATTERN_NAMES[label]
  sheetCap.textContent =
    `${stats.edges.toLocaleString()} connections · ` +
    `input cortex left, answer neurons ${arm === 'shallow' ? 'centre' : 'far right'}` +
    (arm === 'innate' ? ' · born with 12,000 (long tracts included)' : '')

  requestAnimationFrame(frame)
}

// ----------------------------------------------------------------- controls

function rebuild(): void {
  const seed = Math.max(1, Math.round(Number(seedInput.value) || 1))
  sim = grownSim(seed, arm)
  sim.setLearning(learningInput.checked)
  stats = org().stats()
  statsAt = 0
  lastSleeps = 0
  outEdges = 0
  armNote.textContent = GROWN_ARMS[arm].note
}

function setArm(next: GrownArm): void {
  arm = next
  armM1.setAttribute('aria-pressed', String(next === 'm1'))
  armShallow.setAttribute('aria-pressed', String(next === 'shallow'))
  armInnate.setAttribute('aria-pressed', String(next === 'innate'))
  rebuild()
}

runBtn.onclick = () => {
  running = !running
  runBtn.textContent = running ? 'Pause' : 'Run'
}
speedInput.oninput = () => {
  speedLbl.textContent = `≈${targetTps().toLocaleString()} ticks/s`
}
armM1.onclick = () => setArm('m1')
armShallow.onclick = () => setArm('shallow')
armInnate.onclick = () => setArm('innate')
learningInput.onchange = () => sim.setLearning(learningInput.checked)
resetBtn.onclick = rebuild
seedInput.onchange = rebuild

speedInput.oninput(new Event('input'))
armNote.textContent = GROWN_ARMS[arm].note
requestAnimationFrame(frame)
