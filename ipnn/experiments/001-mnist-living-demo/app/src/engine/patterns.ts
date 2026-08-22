// M1 sanity patterns: three clearly distinct 8×8 binary glyphs with similar
// pixel counts (so pixel count alone can't be the cue).

function grid(rows: string[]): Uint8Array {
  const out = new Uint8Array(64)
  rows.forEach((row, r) => {
    for (let c = 0; c < 8; c++) out[r * 8 + c] = row[c] === '#' ? 1 : 0
  })
  return out
}

/** vertical bars */
export const PATTERN_V = grid([
  '.#..#..#',
  '.#..#..#',
  '.#..#..#',
  '.#..#..#',
  '.#..#..#',
  '.#..#..#',
  '.#..#..#',
  '.#..#..#',
])

/** horizontal bars */
export const PATTERN_H = grid([
  '########',
  '........',
  '........',
  '########',
  '........',
  '........',
  '########',
  '........',
])

/** X diagonal */
export const PATTERN_X = grid([
  '#......#',
  '##....##',
  '..#..#..',
  '...##...',
  '...##...',
  '..#..#..',
  '##....##',
  '#......#',
])

export const M1_PATTERNS: Uint8Array[] = [PATTERN_V, PATTERN_H, PATTERN_X]

// ── Task B, for experiment 003 (transfer and retention) ────────────────────
//
// A second set of three glyphs, mapped onto the SAME three output neurons as
// task A. Same outputs is the deliberate choice: it is the maximum-interference
// case, and the one the project's own success criterion describes — "teach it a
// second task, return to the first, and it has not forgotten."
//
// Designed to be as learnable as task A, so a difference in trials-to-criterion
// means something about transfer rather than about difficulty:
//   - all three are 28 pixels, so pixel count cannot be the cue *within* the task
//     (the same discipline task A uses at 20–24)
//   - mutual overlap 0.17–0.27 by intersection-over-union, against task A's
//     0.22–0.23 — comparably separable
//   - no cross-task pair exceeds 0.33 overlap, so no B glyph is a near-copy of
//     an A glyph
// `patterns.test.ts` pins all three properties.

/** plus / crosshair */
export const PATTERN_PLUS = grid([
  '...##...',
  '...##...',
  '...##...',
  '########',
  '########',
  '...##...',
  '...##...',
  '...##...',
])

/** hollow border */
export const PATTERN_RING = grid([
  '########',
  '#......#',
  '#......#',
  '#......#',
  '#......#',
  '#......#',
  '#......#',
  '########',
])

/** thick diagonal band */
export const PATTERN_BAND = grid([
  '###.....',
  '####....',
  '.####...',
  '..####..',
  '...####.',
  '....####',
  '.....###',
  '......##',
])

export const TASK_B_PATTERNS: Uint8Array[] = [PATTERN_PLUS, PATTERN_RING, PATTERN_BAND]

/** Task A under its experiment-003 name. Identical array, different label —
 * `M1_PATTERNS` is what every existing gate and journal entry cites and is not
 * being renamed. */
export const TASK_A_PATTERNS = M1_PATTERNS
