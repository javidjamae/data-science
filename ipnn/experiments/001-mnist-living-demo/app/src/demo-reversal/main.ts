// Experiment 003 — the rule flip, watchable.
//
// Two organisms, identical except for one config value, race through the same
// life: learn the three patterns, then have the RULE flipped — same stimuli,
// reassigned answers — over and over.
//
//   count  the published rule. Evidence only accumulates, so plasticity
//          decays monotonically over its whole life. Watch its "plasticity
//          remaining" gauge drain and never refill: after a few flips it can
//          no longer learn any rule, new or old (L-034 — it ages to death).
//   α/β    the fix. Contradicting evidence is tracked separately, so a
//          synapse whose world has changed becomes uncertain — and plastic —
//          again. It keeps coming back, flip after flip (L-036).
//
// The below-chance dip right after each flip is L-030 live: the organism is
// not confused, it is confidently giving the OLD answer.

import { DemoSim } from '../demo-m1/sim'
import { Organism } from '../engine/organism'
import { defaultConfig } from '../engine/types'

const SERIES = { count: '#C08A18', beta: '#3E8EDE' }
const SCREEN_BG = '#0E1116'
const REPO = 'https://github.com/javidjamae/data-science/blob/master/ipnn'
const CHANCE = 1 / 3
const CRITERION = 0.85
const STIM = ['bars', 'stripes', 'X']
const FLIPPED = [1, 2, 0]

const STYLE = `
:root {
  --bg: #F5F7FA; --ink: #1A2129; --ink-2: #57626E; --ink-3: #85909C;
  --panel: #FFFFFF; --border: #DCE2E9; --accent: #0969DA; --bad: #C4351C;
  --screen: ${SCREEN_BG}; --screen-border: #2A313C;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0D1117; --ink: #E6EDF3; --ink-2: #9BA7B5; --ink-3: #788593;
    --panel: #161B22; --border: #262D37; --accent: #58A6FF; --bad: #F85149;
  }
}
:root[data-theme="dark"] {
  --bg: #0D1117; --ink: #E6EDF3; --ink-2: #9BA7B5; --ink-3: #788593;
  --panel: #161B22; --border: #262D37; --accent: #58A6FF; --bad: #F85149;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font-family: var(--sans); line-height: 1.45; }
.demo { max-width: 980px; margin: 0 auto; padding: 20px 20px 40px; }
header h1 { font-family: var(--mono); font-size: 1.15rem; font-weight: 600;
  letter-spacing: 0.02em; margin: 0; }
header .tagline { color: var(--ink-2); margin: 4px 0 0; font-size: 0.92rem; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 14px;
  margin: 18px 0 14px; padding: 10px 14px; background: var(--panel);
  border: 1px solid var(--border); border-radius: 8px;
  font-family: var(--mono); font-size: 0.8rem; }
.controls label { display: flex; align-items: center; gap: 7px; color: var(--ink-2); }
button { font: inherit; font-family: var(--mono); cursor: pointer;
  background: var(--panel); color: var(--ink); border: 1px solid var(--border);
  border-radius: 6px; padding: 5px 14px; }
button:hover { border-color: var(--accent); }
button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
#run { min-width: 84px; }
#flip { border-color: var(--accent); color: var(--accent); font-weight: 600; }
#speedlbl { display: inline-block; min-width: 11ch; }
input[type="range"] { width: 120px; accent-color: var(--accent); }
input[type="checkbox"] { width: 16px; height: 16px; accent-color: var(--accent); }
input[type="number"] { font: inherit; width: 58px; padding: 4px 6px;
  background: var(--bg); color: var(--ink); border: 1px solid var(--border);
  border-radius: 6px; }
.rulebar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
  margin: 0 0 14px; font-family: var(--mono); font-size: 0.8rem; color: var(--ink-2); }
.rule { display: inline-flex; gap: 8px; padding: 4px 12px; border-radius: 999px;
  border: 1px solid var(--border); background: var(--panel); }
.rule b { color: var(--ink); font-weight: 600; }
.panel { background: var(--panel); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 14px; min-width: 0; }
.panel h2 { margin: 0 0 8px; font-family: var(--mono); font-size: 0.72rem;
  font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--ink-3); display: flex; justify-content: space-between; gap: 10px; }
.screen { background: var(--screen); border: 1px solid var(--screen-border);
  border-radius: 6px; padding: 8px; }
.screen canvas { display: block; width: 100%; height: auto; }
.cap { font-family: var(--mono); font-size: 0.74rem; color: var(--ink-3); margin: 8px 0 0; }
.legend { display: flex; gap: 16px; margin-top: 9px; font-family: var(--mono);
  font-size: 0.72rem; color: var(--ink-3); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
.arms { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
@media (max-width: 700px) { .arms { grid-template-columns: 1fr; } }
.stat { display: grid; grid-template-columns: 11rem 1fr 4.2rem; align-items: center;
  gap: 9px; font-family: var(--mono); font-size: 0.76rem; margin: 6px 0; }
.stat .lbl { color: var(--ink-2); }
.bar { height: 10px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; }
.bar i { display: block; height: 100%; width: 0; border-radius: 999px;
  transition: width 0.2s; }
.big { font-family: var(--mono); font-size: 1.5rem; font-weight: 600; margin: 2px 0 8px; }
.score { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.78rem; }
.score th { text-align: right; font-weight: 600; padding: 4px 8px; }
.score th:first-child { text-align: left; }
.score td { padding: 4px 8px; border-top: 1px solid var(--border); text-align: right;
  font-variant-numeric: tabular-nums; }
.score td:first-child { text-align: left; color: var(--ink-2); }
.big small { font-size: 0.72rem; font-weight: 400; color: var(--ink-3); }
footer { margin-top: 22px; color: var(--ink-3); font-size: 0.78rem; }
a { color: var(--accent); }
`

const app = document.querySelector<HTMLDivElement>('#app')!
const styleEl = document.createElement('style')
styleEl.textContent = STYLE
document.head.appendChild(styleEl)

const armCard = (id: string, name: string, color: string, note: string) => `
  <section class="panel">
    <h2><span style="color:${color}">■</span> ${name}</h2>
    <div class="big"><span id="${id}rules">0</span><small> / <span id="${id}total">0</span> rules learned</small></div>
    <div class="stat"><span class="lbl">rolling accuracy</span>
      <span class="bar"><i id="${id}acc" style="background:${color}"></i></span>
      <span id="${id}accv">0.00</span></div>
    <div class="stat"><span class="lbl">plasticity remaining</span>
      <span class="bar"><i id="${id}plast" style="background:${color}"></i></span>
      <span id="${id}plastv">100%</span></div>
    <div class="stat"><span class="lbl">frozen synapses</span>
      <span class="bar"><i id="${id}frozen" style="background:var(--bad)"></i></span>
      <span id="${id}frozenv">0%</span></div>
    <div class="stat"><span class="lbl">evidence α / β</span>
      <span id="${id}ev" style="grid-column: 2 / 4; text-align: right">0 / 0</span></div>
    <p class="cap">${note}</p>
  </section>`

app.innerHTML = `
<div class="demo">
  <header>
    <h1>IPNN · experiment 003 — the rule flip</h1>
    <p class="tagline">
      Two organisms, identical but for one line: how confidence accumulates.
      Learn the shapes, then <strong>flip the rule</strong> — same shapes, new
      answers — again and again. One of them ages out of learning. The other
      keeps changing its mind.
    </p>
  </header>

  <div class="controls">
    <button id="run">Run</button>
    <label>speed <input id="speed" type="range" min="0" max="100" value="70">
      <span id="speedlbl">≈1,000 t/s</span></label>
    <button id="flip">Flip the rule</button>
    <label><input id="auto" type="checkbox" checked> auto-flip every
      <input id="every" type="number" value="2500" min="150" step="50"> trials</label>
    <label>seed <input id="seed" type="number" value="1" min="1" step="1"></label>
    <button id="reset">Reset</button>
  </div>

  <div class="rulebar">
    <span>current rule:</span>
    <span class="rule" id="rule"></span>
    <span id="flipcount">0 flips so far</span>
  </div>

  <section class="panel">
    <h2>accuracy, flip after flip</h2>
    <div class="screen"><canvas id="chart"></canvas></div>
    <div class="legend">
      <span><i class="swatch" style="background:${SERIES.count}"></i> count — the published rule</span>
      <span><i class="swatch" style="background:${SERIES.beta}"></i> α/β — confidence can fall</span>
      <span>┆ rule flip (marked per arm — the x-axis is that arm's own trial count)</span>
    </div>
    <p class="cap">chance 0.33 · criterion 0.85 · the dip below chance after a flip
      is the organism confidently giving the <em>old</em> answer</p>
  </section>

  <section class="panel" style="margin-top: 14px">
    <h2>scoreboard — the whole life</h2>
    <table class="score">
      <thead><tr><th></th>
        <th style="color:${SERIES.count}">count</th>
        <th style="color:${SERIES.beta}">α/β</th></tr></thead>
      <tbody>
        <tr><td>lifetime accuracy, every trial ever</td><td id="sc_acc_c">–</td><td id="sc_acc_b">–</td></tr>
        <tr><td>share of life spent at criterion (≥0.85)</td><td id="sc_crit_c">–</td><td id="sc_crit_b">–</td></tr>
        <tr><td>first learning (trials to criterion)</td><td id="sc_first_c">–</td><td id="sc_first_b">–</td></tr>
        <tr><td>flips recovered from</td><td id="sc_rec_c">–</td><td id="sc_rec_b">–</td></tr>
        <tr><td>median recovery after a flip (trials)</td><td id="sc_med_c">–</td><td id="sc_med_b">–</td></tr>
      </tbody>
    </table>
    <p class="cap">recovery = trials from the flip until rolling-100 accuracy is back at ≥0.85.
      The 100-trial window means recoveries under 100 cannot be measured; measured ones run 600–2,500.</p>
  </section>

  <div class="arms">
    ${armCard('c', 'count — n only grows', SERIES.count,
      'Every success makes every involved synapse harder to change, forever. Watch plasticity drain — it never refills, and after a few flips this organism cannot learn any rule at all.')}
    ${armCard('b', 'α/β — confidence can fall', SERIES.beta,
      'Contradiction accumulates as β, and plasticity reads |α−β|: a synapse whose world changed becomes uncertain — and movable — again. Same seed, same everything else. Relearning costs 600–2,500 trials, so give each rule a full stage.')}
  </div>

  <footer>
    The findings behind this page:
    <a href="${REPO}/journal/entries/2026-08-22-1620-serial-reversal.md">the organism ages out of learning</a> ·
    <a href="${REPO}/journal/entries/2026-08-22-1735-alphabeta.md">the α/β fix</a>
  </footer>
</div>
`

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const runBtn = $<HTMLButtonElement>('run')
const speedInput = $<HTMLInputElement>('speed')
const speedLbl = $('speedlbl')
const flipBtn = $<HTMLButtonElement>('flip')
const autoInput = $<HTMLInputElement>('auto')
const everyInput = $<HTMLInputElement>('every')
const seedInput = $<HTMLInputElement>('seed')
const resetBtn = $<HTMLButtonElement>('reset')
const ruleEl = $('rule')
const flipCountEl = $('flipcount')

const dpr = Math.min(2, window.devicePixelRatio || 1)
const CHART_W = 932
const CHART_H = 190
const chartCanvas = $<HTMLCanvasElement>('chart')
chartCanvas.width = Math.round(CHART_W * dpr)
chartCanvas.height = Math.round(CHART_H * dpr)
chartCanvas.style.width = '100%'
const ctx = chartCanvas.getContext('2d')!
ctx.scale(dpr, dpr)

// ── state ──────────────────────────────────────────────────────────────────

type ArmId = 'c' | 'b'
interface Arm {
  id: ArmId
  sim: DemoSim
  color: string
  /** one entry per rule stage (acquisition + each flip) */
  stages: { fromTrial: number; recoveredAt: number | null }[]
  flipTrials: number[]
  /** incremental whole-life counters (scanning all trials per frame would
   * be O(life) — these advance only over trials new since the last frame) */
  seen: number
  correctTotal: number
  atCriterion: number
}

function makeArm(id: ArmId, seed: number): Arm {
  const model = id === 'c' ? 'count' : 'beta'
  return {
    id,
    sim: new DemoSim(seed, (s) => new Organism({ ...defaultConfig, seed: s, evidenceModel: model })),
    color: id === 'c' ? SERIES.count : SERIES.beta,
    stages: [{ fromTrial: 0, recoveredAt: null }],
    flipTrials: [],
    seen: 0,
    correctTotal: 0,
    atCriterion: 0,
  }
}

let arms: Arm[] = [makeArm('c', 1), makeArm('b', 1)]
let flipped = false
let running = false
let last = 0

function currentMap(): number[] {
  return flipped ? FLIPPED : [0, 1, 2]
}

function doFlip(): void {
  flipped = !flipped
  for (const a of arms) {
    a.sim.setLabelMap(currentMap())
    a.flipTrials.push(a.sim.trials.length)
    a.stages.push({ fromTrial: a.sim.trials.length, recoveredAt: null })
  }
  drawRule()
}

function drawRule(): void {
  const map = currentMap()
  ruleEl.innerHTML = STIM.map((s, i) => `${s}→<b>${STIM[map[i]]}</b>`).join(' · ')
  const flips = arms[0].flipTrials.length
  flipCountEl.textContent = `${flips} flip${flips === 1 ? '' : 's'} so far`
}

function rebuild(): void {
  const seed = Math.max(1, Math.round(Number(seedInput.value) || 1))
  arms = [makeArm('c', seed), makeArm('b', seed)]
  flipped = false
  drawRule()
}

function targetTps(): number {
  const t = Number(speedInput.value) / 100
  return Math.round(20 * Math.pow(300, t))
}

// ── loop ───────────────────────────────────────────────────────────────────

function frame(now: number): void {
  if (running) {
    const dt = Math.min(0.25, (now - last) / 1000)
    const n = Math.max(1, Math.round(targetTps() * dt))
    for (const a of arms) a.sim.tick(n)

    // stage bookkeeping: a stage counts as learned once rolling accuracy holds
    // the criterion with the whole window inside the stage — and the trial
    // index where that first happens is the recovery time
    for (const a of arms) {
      const st = a.stages[a.stages.length - 1]
      if (st.recoveredAt === null && a.sim.trials.length - st.fromTrial >= 100 &&
          a.sim.rollingAccuracy >= CRITERION) {
        st.recoveredAt = a.sim.trials.length
      }
      // whole-life counters, advanced only over the trials new this frame
      const t = a.sim.trials
      const c = a.sim.accuracyCurve
      for (let i = a.seen; i < t.length; i++) {
        if (t[i].correct) a.correctTotal++
        if (c[i] >= CRITERION) a.atCriterion++
      }
      a.seen = t.length
    }

    if (autoInput.checked) {
      // 2,500 is the serial-reversal harness's own per-stage budget, and
      // relearning measures at 600–2,500 trials — shorter intervals put BOTH
      // arms into a failure cascade and show nothing but shared collapse
      const every = Math.max(150, Number(everyInput.value) || 2500)
      const since = Math.min(...arms.map((a) => a.sim.trials.length - (a.flipTrials.at(-1) ?? 0)))
      if (since >= every) doFlip()
    }
  }
  last = now

  drawChart()
  for (const a of arms) drawArm(a)
  drawScore()
  requestAnimationFrame(frame)
}

function drawChart(): void {
  ctx.fillStyle = SCREEN_BG
  ctx.fillRect(0, 0, CHART_W, CHART_H)
  const y = (v: number) => CHART_H - 8 - v * (CHART_H - 18)

  ctx.setLineDash([3, 3])
  ctx.lineWidth = 1
  ctx.strokeStyle = '#4A5462'
  ctx.beginPath(); ctx.moveTo(0, y(CHANCE)); ctx.lineTo(CHART_W, y(CHANCE)); ctx.stroke()
  ctx.strokeStyle = '#2F6F45'
  ctx.beginPath(); ctx.moveTo(0, y(CRITERION)); ctx.lineTo(CHART_W, y(CRITERION)); ctx.stroke()
  ctx.setLineDash([])

  const maxTrials = Math.max(...arms.map((a) => a.sim.accuracyCurve.length), 2)
  let span = 500
  while (span < maxTrials && span < 8000) span *= 2
  const start = Math.max(0, maxTrials - span)
  const x = (t: number) => ((t - start) / span) * CHART_W

  // Flip markers, PER ARM, in the arm's colour. The x-axis is trial count and
  // trials have different durations (a confident answer ends one in ~21
  // ticks, a hesitant one in up to 75), so at the same wall-clock flip the
  // two arms sit at different trial numbers — one shared marker would make
  // the other arm's response look like lag that isn't there.
  ctx.setLineDash([2, 4])
  for (const a of arms) {
    ctx.strokeStyle = a.color + '66'
    for (const t of a.flipTrials) {
      if (t < start) continue
      ctx.beginPath(); ctx.moveTo(x(t), 4); ctx.lineTo(x(t), CHART_H - 4); ctx.stroke()
    }
  }
  ctx.setLineDash([])

  for (const a of arms) {
    const c = a.sim.accuracyCurve
    if (c.length < 2) continue
    ctx.strokeStyle = a.color
    ctx.lineWidth = 1.6
    ctx.beginPath()
    for (let i = Math.max(0, start); i < c.length; i++) {
      const px = x(i)
      const py = y(c[i])
      if (i === Math.max(0, start)) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.stroke()
  }
}

function drawScore(): void {
  const set = (id: string, v: string) => { $(id).textContent = v }
  for (const a of arms) {
    const k = a.id === 'c' ? 'c' : 'b'
    const n = a.sim.trials.length
    set(`sc_acc_${k}`, n ? (a.correctTotal / n).toFixed(3) : '–')
    set(`sc_crit_${k}`, n ? `${((a.atCriterion / n) * 100).toFixed(0)}%` : '–')
    const first = a.stages[0].recoveredAt
    set(`sc_first_${k}`, first === null ? '–' : String(first))
    const flips = a.stages.slice(1)
    const rec = flips
      .map((st) => (st.recoveredAt === null ? null : st.recoveredAt - st.fromTrial))
      .filter((x): x is number => x !== null)
    set(`sc_rec_${k}`, flips.length ? `${rec.length} / ${flips.length}` : '–')
    if (rec.length) {
      const sorted = [...rec].sort((x, y) => x - y)
      set(`sc_med_${k}`, String(sorted[Math.floor(sorted.length / 2)]))
    } else set(`sc_med_${k}`, '–')
  }
}

function drawArm(a: Arm): void {
  const org = a.sim.org as Organism
  const acc = a.sim.rollingAccuracy
  const p = org.plasticityStats()
  const ev = org.evidenceTotals()
  const learned = a.stages.filter((s) => s.recoveredAt !== null).length

  $(`${a.id}rules`).textContent = String(learned)
  $(`${a.id}total`).textContent = String(a.stages.length)
  ;($(`${a.id}acc`) as HTMLElement).style.width = `${acc * 100}%`
  $(`${a.id}accv`).textContent = acc.toFixed(2)
  ;($(`${a.id}plast`) as HTMLElement).style.width = `${p.mean * 100}%`
  $(`${a.id}plastv`).textContent = `${(p.mean * 100).toFixed(0)}%`
  ;($(`${a.id}frozen`) as HTMLElement).style.width = `${p.frozenFraction * 100}%`
  $(`${a.id}frozenv`).textContent = `${(p.frozenFraction * 100).toFixed(0)}%`
  const k = (v: number) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(0)}k` : v.toFixed(0))
  $(`${a.id}ev`).textContent = `${k(ev.outAlpha)} / ${k(ev.outBeta)}`
}

// ── controls ───────────────────────────────────────────────────────────────

runBtn.onclick = () => {
  running = !running
  runBtn.textContent = running ? 'Pause' : 'Run'
}
speedInput.oninput = () => {
  speedLbl.textContent = `≈${targetTps().toLocaleString()} t/s`
}
flipBtn.onclick = doFlip
resetBtn.onclick = rebuild
seedInput.onchange = rebuild

speedInput.oninput(new Event('input'))
drawRule()
requestAnimationFrame(frame)
