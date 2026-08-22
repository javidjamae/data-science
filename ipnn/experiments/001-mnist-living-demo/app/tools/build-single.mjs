// Build a demo as ONE self-contained .html file: no network, no imports,
// no dependencies. This is what gets published as the shareable artifact.
//
// The previous single-file build was done ad hoc from the command line, which
// meant the published page could not be reproduced from the repo. It can now:
//   npm run build:single            → both demos
//   npm run build:single -- grown   → just experiment 002's
//
// There are two outputs per demo. The plain .html is a complete document you
// can open from disk; the .artifact.html omits the document skeleton, because
// the Claude artifact host supplies its own <!doctype>/<html>/<head>/<body>
// and rejects ours. The artifact file is the one that gets published, and
// keeping it a build output rather than a hand-edit is what makes the hosted
// page reproducible from the repo.

import { build } from 'esbuild'
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const APP = dirname(dirname(fileURLToPath(import.meta.url)))
const OUT_DIR = join(APP, 'dist-single')

const TARGETS = {
  m1: {
    entry: 'src/main.ts',
    out: 'ipnn-m1-demo',
    title: 'IPNN — M1 Living Demo',
  },
  grown: {
    entry: 'src/demo-grown/main.ts',
    out: 'ipnn-002-grown-demo',
    title: 'IPNN — experiment 002, the grown substrate',
  },
}

const requested = process.argv.slice(2).filter((a) => !a.startsWith('-'))
const names = requested.length > 0 ? requested : Object.keys(TARGETS)

mkdirSync(OUT_DIR, { recursive: true })

for (const name of names) {
  const target = TARGETS[name]
  if (!target) {
    console.error(`unknown target "${name}"; known: ${Object.keys(TARGETS).join(', ')}`)
    process.exitCode = 1
    continue
  }

  const result = await build({
    entryPoints: [join(APP, target.entry)],
    bundle: true,
    minify: true,
    format: 'iife',
    target: 'es2020',
    write: false,
  })
  const js = result.outputFiles[0].text

  // </script> inside the bundle would close this tag early; nothing in the
  // source has one today, but a future string literal could.
  const safeJs = js.replace(/<\/script>/gi, '<\\/script>')

  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${target.title}</title>
</head>
<body>
<div id="app"></div>
<script>${safeJs}</script>
</body>
</html>
`

  const artifactHtml = `<title>${target.title}</title>
<div id="app"></div>
<script>${safeJs}</script>
`

  const full = join(OUT_DIR, `${target.out}.html`)
  const artifact = join(OUT_DIR, `${target.out}.artifact.html`)
  writeFileSync(full, html)
  writeFileSync(artifact, artifactHtml)
  console.log(`${full}  ${(html.length / 1024).toFixed(1)} KB`)
  console.log(`${artifact}  ${(artifactHtml.length / 1024).toFixed(1)} KB`)
}
