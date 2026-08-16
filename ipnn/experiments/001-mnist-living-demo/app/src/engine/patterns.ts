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
