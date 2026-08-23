// Layout-stability check for the demo page (see journal entry
// 2026-08-21-0008 and L-008): an instrument panel must not change position or
// size while the organism runs.
//
// Samples getBoundingClientRect() for every element that could move, across
// many consecutive animation frames, and reports peak-to-peak spread. Exits
// non-zero if anything moved.
//
//   npm run build:single
//   node tools/measure-jitter.mjs dist-single/ipnn-m1-demo.html [speed] [width]
//
// Playwright is deliberately NOT a devDependency — it pulls browser binaries
// and this is an occasional check, not part of `npm test`. Point PLAYWRIGHT_PKG
// at an existing install, or `npm i -D playwright` temporarily.

import { pathToFileURL } from 'node:url'
import { resolve } from 'node:path'

const PKG = process.env.PLAYWRIGHT_PKG ?? 'playwright'
let chromium
try {
  ;({ chromium } = await import(PKG))
} catch {
  console.error(
    `Could not load Playwright from "${PKG}".\n` +
      `Set PLAYWRIGHT_PKG to an installed copy, e.g.\n` +
      `  PLAYWRIGHT_PKG=/path/to/node_modules/playwright/index.mjs \\\n` +
      `    node tools/measure-jitter.mjs dist-single/ipnn-m1-demo.html`
  )
  process.exit(2)
}

const target = process.argv[2]
if (!target) {
  console.error('usage: node tools/measure-jitter.mjs <html file> [speed 0-100] [viewport width]')
  process.exit(2)
}
const url = /^https?:|^file:/.test(target)
  ? target
  : pathToFileURL(resolve(target)).href
const speed = process.argv[3] ?? null
const width = Number(process.argv[4] ?? 1200)
// 5th arg "manual" switches to manual mode and holds a pattern before
// sampling — that mode swaps the stats strip's content, so it needs its own
// pass rather than being assumed safe because auto mode is.
const manual = process.argv[5] === 'manual'

// Anything below this is sub-pixel rounding, not motion.
const TOLERANCE_PX = 0.01
const FRAMES = 240

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width, height: 900 } })
await page.goto(url)
await page.waitForTimeout(400)

if (manual) {
  // train briefly first, so the readout has a competent organism to report on
  await page.waitForTimeout(1500)
  await page.click('#modemanual')
  await page.click('#stim1')
}

if (speed !== null) {
  await page.evaluate((v) => {
    const el = document.getElementById('speed')
    el.value = v
    el.dispatchEvent(new Event('input'))
  }, speed)
}

const samples = await page.evaluate(async (frames) => {
  const TARGETS = [
    ['panel:sense', '.panels > .panel:nth-child(1)'],
    ['panel:pool', '.panels > .panel:nth-child(2)'],
    ['panel:output', '.panels > .panel:nth-child(3)'],
    ['sense canvas', '#sense'],
    ['pool canvas', '#pool'],
    ['outrows', '#outrows'],
    ['chart panel', '.chartwrap > .panel'],
    ['stats strip', '#stats'],
    ['controls', '.controls'],
    ['reset button', '#reset'],
    ['seed input', '#seed'],
  ]
  const out = {}
  for (const [name] of TARGETS) out[name] = []
  for (let i = 0; i < frames; i++) {
    await new Promise((r) => requestAnimationFrame(r))
    for (const [name, sel] of TARGETS) {
      const el = document.querySelector(sel)
      if (!el) continue
      const r = el.getBoundingClientRect()
      out[name].push([r.x, r.y, r.width, r.height])
    }
  }
  return out
}, FRAMES)

// first value in the stats strip: trial count in auto mode, current answer in
// manual — just a "was it actually running?" sanity line
const firstStat = await page.evaluate(
  () => document.querySelector('#stats b')?.textContent ?? '?'
)
await browser.close()

console.log(
  `\n${url.split('/').pop()}  speed=${speed ?? 'default'}  ` +
    `viewport=${width}px  mode=${manual ? 'manual' : 'auto'}  first stat: ${firstStat}`
)
console.log('element'.padEnd(16), '  Δx   Δy   Δw   Δh   verdict')

let worst = 0
for (const [name, rows] of Object.entries(samples)) {
  if (!rows.length) continue
  const spread = (i) => {
    const v = rows.map((r) => r[i])
    return Math.max(...v) - Math.min(...v)
  }
  const d = [spread(0), spread(1), spread(2), spread(3)]
  const max = Math.max(...d)
  worst = Math.max(worst, max)
  console.log(
    name.padEnd(16),
    d.map((v) => v.toFixed(1).padStart(4)).join(' '),
    max < TOLERANCE_PX ? '  STABLE' : `  MOVES ${max.toFixed(2)}px`
  )
}

const ok = worst < TOLERANCE_PX
console.log(
  `\nworst movement: ${worst.toFixed(4)}px — ${ok ? 'PASS' : 'FAIL (see L-008)'}\n`
)
process.exit(ok ? 0 : 1)
