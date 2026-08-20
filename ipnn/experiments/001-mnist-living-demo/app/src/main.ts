// The M1 living demo: watch the gate-certified organism learn 3 patterns
// live, from reward alone. Rendering only — all simulation logic lives in
// src/demo-m1/sim.ts (headless-tested) on top of src/engine/ (the M1 gate).
// Demos are namespaced per milestone (demo-m1, demo-m2, …); when M2 lands,
// main.ts becomes a switcher (or per-demo HTML entries).
//
// The page chrome follows the viewer's light/dark theme; the three
// instrument screens are deliberately dark in both, like physical
// oscilloscopes. Series colors were validated (CVD + contrast) against the
// screen surface; identity is never color-alone (every mark has a label).

import { DemoSim } from './demo-m1/sim'
import { M1_PATTERNS } from './engine/patterns'

// categorical trio for the three patterns — dataviz six-checks PASS on dark
const SERIES = ['#3E8EDE', '#C08A18', '#DE5FA5']
const PATTERN_NAMES = ['vertical bars', 'horizontal bars', 'diagonal X']
const SCREEN_BG = '#0E1116'
const REPO = 'https://github.com/javidjamae/data-science/blob/master/ipnn'

const STYLE = `
:root {
  --bg: #F5F7FA;
  --ink: #1A2129;
  --ink-2: #57626E;
  --ink-3: #85909C;
  --panel: #FFFFFF;
  --border: #DCE2E9;
  --accent: #0969DA;
  --good: #1A7F37;
  --screen: ${SCREEN_BG};
  --screen-border: #2A313C;
  --screen-ink: #8C99A8;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0D1117; --ink: #E6EDF3; --ink-2: #9BA7B5; --ink-3: #788593;
    --panel: #161B22; --border: #262D37; --accent: #58A6FF; --good: #3FB950;
  }
}
:root[data-theme="dark"] {
  --bg: #0D1117; --ink: #E6EDF3; --ink-2: #9BA7B5; --ink-3: #788593;
  --panel: #161B22; --border: #262D37; --accent: #58A6FF; --good: #3FB950;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: var(--sans); line-height: 1.45;
}
.demo { max-width: 1060px; margin: 0 auto; padding: 20px 20px 40px; }
header h1 {
  font-family: var(--mono); font-size: 1.15rem; font-weight: 600;
  letter-spacing: 0.02em; margin: 0;
}
header .tagline { color: var(--ink-2); margin: 4px 0 0; font-size: 0.92rem; }
.controls {
  display: flex; flex-wrap: wrap; align-items: center; gap: 14px;
  margin: 18px 0 14px; padding: 10px 14px;
  background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  font-family: var(--mono); font-size: 0.8rem;
}
.controls label { display: flex; align-items: center; gap: 7px; color: var(--ink-2); }
button {
  font: inherit; font-family: var(--mono); cursor: pointer;
  background: var(--panel); color: var(--ink);
  border: 1px solid var(--border); border-radius: 6px; padding: 5px 14px;
}
button:hover { border-color: var(--accent); }
button:focus-visible, input:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px;
}
#run { min-width: 84px; }
input[type="range"] { width: 130px; accent-color: var(--accent); }
input[type="checkbox"] { width: 16px; height: 16px; accent-color: var(--accent); }
input[type="number"] {
  font: inherit; width: 58px; padding: 4px 6px; background: var(--bg);
  color: var(--ink); border: 1px solid var(--border); border-radius: 6px;
}
.frozen-badge {
  display: none; padding: 2px 9px; border-radius: 999px;
  border: 1px solid var(--accent); color: var(--accent); font-size: 0.72rem;
}
.is-frozen .frozen-badge { display: inline-block; }
.panels {
  display: grid; grid-template-columns: auto auto 1fr; gap: 14px;
  align-items: stretch;
}
@media (max-width: 860px) { .panels { grid-template-columns: 1fr; } }
.panel {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 14px; min-width: 0;
}
.panel h2 {
  margin: 0 0 8px; font-family: var(--mono); font-size: 0.72rem;
  font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--ink-3);
}
.panel canvas.screen {
  display: block; background: var(--screen);
  border: 1px solid var(--screen-border); border-radius: 6px;
}
.panel .cap {
  margin: 8px 0 0; font-family: var(--mono); font-size: 0.74rem;
  color: var(--ink-2); font-variant-numeric: tabular-nums;
}
.outrows { display: flex; flex-direction: column; gap: 8px; background: var(--screen);
  border: 1px solid var(--screen-border); border-radius: 6px; padding: 12px; }
.outrow { display: grid; grid-template-columns: 26px 118px 1fr 76px 16px;
  gap: 10px; align-items: center; font-family: var(--mono); font-size: 0.76rem; }
.outrow .name { color: var(--screen-ink); white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.outrow canvas { display: block; }
.pbar { height: 10px; background: #1A212B; border-radius: 3px; overflow: hidden; }
.pbar > i { display: block; height: 100%; width: 0; border-radius: 3px;
  transition: none; }
.pips { display: flex; gap: 3px; }
.pip { width: 8px; height: 8px; border-radius: 2px; background: #1A212B; }
.spoken-dot { width: 10px; height: 10px; border-radius: 50%; background: #1A212B; }
.lamp-row { display: flex; align-items: center; gap: 8px; margin-top: 10px;
  font-family: var(--mono); font-size: 0.74rem; color: var(--screen-ink); }
.lamp { width: 12px; height: 12px; border-radius: 50%; background: #1A212B; }
.chartwrap { position: relative; margin-top: 14px; }
.chartwrap .panel { padding-bottom: 8px; }
#chart { display: block; width: 100%; }
.tooltip {
  position: absolute; display: none; pointer-events: none;
  background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 4px 9px; font-family: var(--mono); font-size: 0.74rem;
  color: var(--ink); white-space: nowrap; box-shadow: 0 2px 8px rgba(0,0,0,0.18);
  font-variant-numeric: tabular-nums;
}
.stats {
  display: flex; flex-wrap: wrap; gap: 8px 26px; margin-top: 12px;
  font-family: var(--mono); font-size: 0.78rem; color: var(--ink-2);
  font-variant-numeric: tabular-nums;
}
.stats b { color: var(--ink); font-weight: 600; }
footer { margin-top: 22px; color: var(--ink-2); font-size: 0.85rem;
  border-top: 1px solid var(--border); padding-top: 14px; }
footer a { color: var(--accent); }
`

// ---------------------------------------------------------------- markup

const app = document.querySelector<HTMLDivElement>('#app')!
const styleEl = document.createElement('style')
styleEl.textContent = STYLE
document.head.appendChild(styleEl)

app.innerHTML = `
<div class="demo" id="root">
  <header>
    <h1>IPNN · experiment 001 · the M1 organism, live</h1>
    <p class="tagline">A few thousand from-scratch synapses learning three
    patterns in real time — no backprop, no labels, no training phase. Only a
    reward signal, like praising a pet. <span class="frozen-badge">frozen — no rewards</span></p>
  </header>

  <div class="controls">
    <button id="run" aria-pressed="true">Pause</button>
    <label>speed <input id="speed" type="range" min="0" max="100" value="57">
      <span id="speedlbl"></span></label>
    <label><input id="learning" type="checkbox" checked> learning</label>
    <label>seed <input id="seed" type="number" value="1" min="1" step="1"></label>
    <button id="reset">Reset</button>
  </div>

  <div class="panels">
    <section class="panel">
      <h2>Sense · 8×8</h2>
      <canvas id="sense" class="screen" role="img" aria-label="the organism's 8 by 8 visual sense"></canvas>
      <p class="cap" id="sensecap"></p>
    </section>
    <section class="panel">
      <h2>Pool · 160 neurons</h2>
      <canvas id="pool" class="screen" role="img" aria-label="live firing raster of the 160 pool neurons"></canvas>
      <p class="cap" id="poolcap"></p>
    </section>
    <section class="panel">
      <h2>Output register</h2>
      <div class="outrows" id="outrows"></div>
      <div class="lamp-row"><span class="lamp" id="lamp"></span>
        <span id="lamplbl">reward</span></div>
      <p class="cap">spoken = 6 fires within a 20-tick window · silence is legal</p>
    </section>
  </div>

  <div class="chartwrap">
    <section class="panel">
      <h2>Rolling accuracy · last 100 trials</h2>
      <canvas id="chart" height="230"></canvas>
    </section>
    <div class="tooltip" id="tooltip"></div>
  </div>

  <div class="stats" id="stats"></div>

  <footer>
    This is the experiment the journal recorded on 2026-08-16: seeds 1–3
    reproduce the logged curves exactly (chance ≈ 0.33 → ~0.98 within 800
    trials). Toggle <em>learning</em> off once it's competent — accuracy
    holding without reward is the living-model claim.
    <a href="${REPO}/journal/entries/2026-08-16-0248-experiment-001-m0-m1-first-build.md">journal entry</a> ·
    <a href="${REPO}/experiments/001-mnist-living-demo/design.md">design</a> ·
    <a href="${REPO}/vision.md">vision</a>
  </footer>
</div>`

// ---------------------------------------------------------------- helpers

const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T

const root = $('root')
const runBtn = $<HTMLButtonElement>('run')
const speedInput = $<HTMLInputElement>('speed')
const speedLbl = $('speedlbl')
const learningInput = $<HTMLInputElement>('learning')
const seedInput = $<HTMLInputElement>('seed')
const resetBtn = $<HTMLButtonElement>('reset')
const senseCap = $('sensecap')
const poolCap = $('poolcap')
const lamp = $('lamp')
const statsEl = $('stats')
const tooltip = $('tooltip')

const dpr = Math.min(2, window.devicePixelRatio || 1)

function setupCanvas(
  c: HTMLCanvasElement,
  wCss: number,
  hCss: number
): CanvasRenderingContext2D {
  c.width = Math.round(wCss * dpr)
  c.height = Math.round(hCss * dpr)
  c.style.width = `${wCss}px`
  c.style.height = `${hCss}px`
  const ctx = c.getContext('2d')!
  ctx.scale(dpr, dpr)
  return ctx
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim()
}

// ---------------------------------------------------------------- sim

const sim = new DemoSim(1)

let running = !window.matchMedia('(prefers-reduced-motion: reduce)').matches
let uiPulse = 0
let measuredTps = 0
let lastT = performance.now()

function targetTps(): number {
  // log scale: 60 ticks/s at 0 → 24,000 at 100
  return 60 * Math.pow(400, Number(speedInput.value) / 100)
}

// ---------------------------------------------------------------- sense

const CELL = 21
const senseCtx = setupCanvas($<HTMLCanvasElement>('sense'), CELL * 8, CELL * 8)

function drawSense(): void {
  senseCtx.fillStyle = SCREEN_BG
  senseCtx.fillRect(0, 0, CELL * 8, CELL * 8)
  const on = sim.stepper.phase === 'stimulus'
  senseCtx.fillStyle = on ? '#E7EDF4' : '#1A212B'
  const s = sim.org.sense
  for (let i = 0; i < 64; i++) {
    if (s[i]) {
      senseCtx.fillRect((i % 8) * CELL + 1, ((i / 8) | 0) * CELL + 1, CELL - 2, CELL - 2)
    }
  }
  senseCap.textContent = on
    ? `showing: ${PATTERN_NAMES[sim.currentLabel]}`
    : 'blank (between stimuli)'
  senseCap.style.color = on ? SERIES[sim.currentLabel] : ''
}

// ---------------------------------------------------------------- pool

const PCOLS = 16
const PROWS = 10
const PCELL = 10.5
const poolCtx = setupCanvas(
  $<HTMLCanvasElement>('pool'),
  PCOLS * PCELL,
  PROWS * PCELL
)

function drawPool(): void {
  poolCtx.fillStyle = SCREEN_BG
  poolCtx.fillRect(0, 0, PCOLS * PCELL, PROWS * PCELL)
  const n = Math.max(1, sim.accumTicks)
  for (let j = 0; j < 160; j++) {
    const rate = sim.poolAccum[j] / n
    if (rate <= 0) continue
    const b = Math.sqrt(rate) // gamma boost so sparse activity reads
    poolCtx.fillStyle = `rgba(158, 205, 255, ${Math.min(1, 0.15 + b)})`
    poolCtx.fillRect(
      (j % PCOLS) * PCELL + 1,
      ((j / PCOLS) | 0) * PCELL + 1,
      PCELL - 2,
      PCELL - 2
    )
  }
  poolCap.textContent = `sparsity ${(sim.org.poolActivity() * 100).toFixed(0)}% · urge ${sim.org.urge.toFixed(2)}`
}

// ---------------------------------------------------------------- outputs

interface OutRow {
  bar: HTMLElement
  pips: HTMLElement[]
  dot: HTMLElement
}
const outRows: OutRow[] = []
{
  const wrap = $('outrows')
  for (let k = 0; k < 3; k++) {
    const row = document.createElement('div')
    row.className = 'outrow'
    row.innerHTML = `
      <canvas width="${24 * dpr}" height="${24 * dpr}" style="width:24px;height:24px"></canvas>
      <span class="name">${PATTERN_NAMES[k]}</span>
      <span class="pbar"><i style="background:${SERIES[k]}"></i></span>
      <span class="pips">${'<span class="pip"></span>'.repeat(6)}</span>
      <span class="spoken-dot"></span>`
    wrap.appendChild(row)
    // mini glyph chip in the series color
    const gc = row.querySelector('canvas')!.getContext('2d')!
    gc.scale(dpr, dpr)
    gc.fillStyle = SERIES[k]
    const pat = M1_PATTERNS[k]
    for (let i = 0; i < 64; i++) {
      if (pat[i]) gc.fillRect((i % 8) * 3, ((i / 8) | 0) * 3, 3, 3)
    }
    outRows.push({
      bar: row.querySelector('.pbar > i') as HTMLElement,
      pips: Array.from(row.querySelectorAll('.pip')),
      dot: row.querySelector('.spoken-dot') as HTMLElement,
    })
  }
  // silence row
  const srow = document.createElement('div')
  srow.className = 'outrow'
  srow.innerHTML = `
    <span></span><span class="name">(silence)</span>
    <span class="pbar"><i style="background:#3B4552"></i></span>
    <span></span><span></span>`
  wrap.appendChild(srow)
  outRows.push({
    bar: srow.querySelector('.pbar > i') as HTMLElement,
    pips: [],
    dot: srow.querySelector('.name') as HTMLElement, // unused
  })
}

function drawOutputs(): void {
  const p = sim.org.outputProbs()
  let sum = 0
  const spoken = sim.stepper.spoken
  for (let k = 0; k < 3; k++) {
    sum += p[k]
    outRows[k].bar.style.width = `${(p[k] * 100).toFixed(1)}%`
    const c = Math.min(6, sim.stepper.counts[k])
    outRows[k].pips.forEach((pip, i) => {
      pip.style.background = i < c ? SERIES[k] : '#1A212B'
    })
    outRows[k].dot.style.background = spoken === k ? SERIES[k] : '#1A212B'
  }
  outRows[3].bar.style.width = `${(Math.max(0, 1 - sum) * 100).toFixed(1)}%`

  uiPulse = Math.max(uiPulse * 0.88, sim.rewardPulse > 0 ? 1 : 0)
  sim.rewardPulse = 0
  lamp.style.background =
    uiPulse > 0.05 ? cssVar('--good') : '#1A212B'
  lamp.style.opacity = String(0.35 + 0.65 * uiPulse)
}

// ---------------------------------------------------------------- chart

const chartCanvas = $<HTMLCanvasElement>('chart')
let chartCtx: CanvasRenderingContext2D
let chartW = 0
const CHART_H = 230
const M = { l: 46, r: 16, t: 12, b: 24 }

function sizeChart(): void {
  chartW = chartCanvas.parentElement!.clientWidth - 28
  chartCtx = setupCanvas(chartCanvas, chartW, CHART_H)
}
sizeChart()
window.addEventListener('resize', () => sizeChart())

let hoverX: number | null = null

function drawChart(): void {
  const ctx = chartCtx
  const w = chartW
  const h = CHART_H
  const pw = w - M.l - M.r
  const ph = h - M.t - M.b
  const ink2 = cssVar('--ink-2')
  const ink3 = cssVar('--ink-3')
  const border = cssVar('--border')
  const accent = cssVar('--accent')

  ctx.clearRect(0, 0, w, h)
  ctx.font = '11px ui-monospace, Menlo, monospace'

  const curve = sim.accuracyCurve
  const n = curve.length
  const xmax = Math.max(100, n)
  const X = (i: number) => M.l + (i / xmax) * pw
  const Y = (v: number) => M.t + (1 - v) * ph

  // learning-off shading (from toggle marks)
  ctx.fillStyle = 'rgba(128,140,153,0.13)'
  let offStart: number | null = null
  for (const m of sim.marks) {
    if (!m.learning) offStart = m.trial
    else if (offStart !== null) {
      ctx.fillRect(X(offStart), M.t, X(m.trial) - X(offStart), ph)
      offStart = null
    }
  }
  if (offStart !== null) ctx.fillRect(X(offStart), M.t, X(n) - X(offStart), ph)

  // gridlines: 0, chance, gate, 1 — numbers in the gutter, words in-plot
  const grid: Array<[number, string, string, number[]]> = [
    [0, '0', '', []],
    [1 / 3, '.33', 'chance', [5, 4]],
    [0.8, '.80', 'gate', [2, 3]],
    [1, '1.0', '', []],
  ]
  for (const [v, num, word, dash] of grid) {
    ctx.strokeStyle = border
    ctx.setLineDash(dash)
    ctx.beginPath()
    ctx.moveTo(M.l, Y(v))
    ctx.lineTo(w - M.r, Y(v))
    ctx.stroke()
    ctx.setLineDash([])
    ctx.fillStyle = ink3
    ctx.textAlign = 'right'
    ctx.fillText(num, M.l - 6, Y(v) + 4)
    if (word) {
      ctx.textAlign = 'left'
      ctx.fillText(word, M.l + 5, Y(v) - 5)
    }
  }

  // x ticks (skip any that would collide with the axis title at right)
  const step = xmax <= 200 ? 50 : xmax <= 1000 ? 200 : xmax <= 5000 ? 1000 : 5000
  ctx.fillStyle = ink3
  ctx.textAlign = 'center'
  for (let x = 0; x <= xmax; x += step) {
    if (X(x) < w - M.r - 56) ctx.fillText(String(x), X(x), h - 8)
  }
  ctx.fillStyle = ink2
  ctx.textAlign = 'right'
  ctx.fillText('trials', w - M.r, h - 8)

  if (n > 1) {
    // the accuracy line (single series — the panel title is its legend)
    ctx.strokeStyle = accent
    ctx.lineWidth = 2
    ctx.lineJoin = 'round'
    ctx.beginPath()
    const dec = Math.max(1, Math.floor(n / pw))
    for (let i = 0; i < n; i += dec) {
      const x = X(i)
      const y = Y(curve[i])
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.lineTo(X(n - 1), Y(curve[n - 1]))
    ctx.stroke()

    // emphasized endpoint + direct value label
    ctx.fillStyle = accent
    ctx.beginPath()
    ctx.arc(X(n - 1), Y(curve[n - 1]), 3.5, 0, Math.PI * 2)
    ctx.fill()
    ctx.textAlign = 'left'
    ctx.fillText(
      curve[n - 1].toFixed(2),
      Math.min(X(n - 1) + 7, w - M.r - 30),
      Y(curve[n - 1]) - 7
    )
  }

  // hover crosshair + tooltip
  if (hoverX !== null && n > 1) {
    const i = Math.max(0, Math.min(n - 1, Math.round(((hoverX - M.l) / pw) * xmax)))
    if (i < n) {
      ctx.strokeStyle = ink3
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(X(i), M.t)
      ctx.lineTo(X(i), M.t + ph)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = accent
      ctx.beginPath()
      ctx.arc(X(i), Y(curve[i]), 4, 0, Math.PI * 2)
      ctx.fill()
      tooltip.style.display = 'block'
      tooltip.textContent = `trial ${i + 1} · accuracy ${curve[i].toFixed(2)}`
      const wrap = chartCanvas.closest('.chartwrap') as HTMLElement
      const rect = chartCanvas.getBoundingClientRect()
      const wr = wrap.getBoundingClientRect()
      tooltip.style.left = `${Math.min(rect.left - wr.left + X(i) + 12, wr.width - 170)}px`
      tooltip.style.top = `${rect.top - wr.top + Y(curve[i]) - 34}px`
    }
  } else {
    tooltip.style.display = 'none'
  }
}

chartCanvas.addEventListener('mousemove', (e) => {
  const r = chartCanvas.getBoundingClientRect()
  hoverX = e.clientX - r.left
})
chartCanvas.addEventListener('mouseleave', () => (hoverX = null))

// ---------------------------------------------------------------- stats

function drawStats(): void {
  const recent = sim.trials.slice(-20)
  const lat = recent.length
    ? recent.reduce((a, t) => a + t.latency, 0) / recent.length
    : 0
  const silent = recent.length
    ? recent.filter((t) => t.spoken === null).length / recent.length
    : 0
  statsEl.innerHTML = `
    <span>trials <b>${sim.trials.length}</b></span>
    <span>accuracy (last 100) <b>${sim.rollingAccuracy.toFixed(2)}</b></span>
    <span>reward baseline <b>${sim.teacher.baseline.toFixed(2)}</b></span>
    <span>answer latency <b>${lat.toFixed(0)} ticks</b></span>
    <span>silent <b>${(silent * 100).toFixed(0)}%</b></span>
    <span>speed <b>${(measuredTps / 1000).toFixed(1)}k ticks/s</b></span>`
}

// ---------------------------------------------------------------- controls

runBtn.addEventListener('click', () => {
  running = !running
  runBtn.textContent = running ? 'Pause' : 'Run'
  runBtn.setAttribute('aria-pressed', String(running))
  lastT = performance.now()
})
learningInput.addEventListener('change', () => {
  sim.setLearning(learningInput.checked)
  root.classList.toggle('is-frozen', !learningInput.checked)
})
resetBtn.addEventListener('click', () => {
  sim.reset(Math.max(1, Number(seedInput.value) | 0))
  learningInput.checked = true
  root.classList.remove('is-frozen')
  uiPulse = 0
})
function speedLabel(): void {
  const t = targetTps()
  speedLbl.textContent =
    t >= 1000 ? `≈${(t / 1000).toFixed(1)}k ticks/s` : `≈${t.toFixed(0)} ticks/s`
}
speedInput.addEventListener('input', speedLabel)
speedLabel()
window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement) return
  if (e.key === ' ') { e.preventDefault(); runBtn.click() }
  if (e.key === 'l') learningInput.click()
})
if (!running) runBtn.textContent = 'Run'

// ---------------------------------------------------------------- loop

function frame(now: number): void {
  const dt = Math.min(0.1, (now - lastT) / 1000)
  lastT = now
  if (running && dt > 0) {
    const n = Math.max(1, Math.min(30000, Math.round(targetTps() * dt)))
    sim.tick(n)
    measuredTps = measuredTps * 0.9 + (n / dt) * 0.1
  }
  drawSense()
  drawPool()
  drawOutputs()
  drawChart()
  drawStats()
  sim.clearAccum()
  requestAnimationFrame(frame)
}
requestAnimationFrame(frame)
