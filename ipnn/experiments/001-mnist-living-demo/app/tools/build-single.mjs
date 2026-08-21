// Build the demo as ONE self-contained .html file: no network, no imports,
// no dependencies. This is what gets published as the shareable artifact.
//
// The previous single-file build was done ad hoc from the command line, which
// meant the published page could not be reproduced from the repo. It can now:
//   npm run build:single   →   dist-single/ipnn-m1-demo.html

import { build } from 'esbuild'
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const APP = dirname(dirname(fileURLToPath(import.meta.url)))
const OUT_DIR = join(APP, 'dist-single')
const OUT = join(OUT_DIR, 'ipnn-m1-demo.html')
// Same bundle, minus the document skeleton: the Claude artifact host supplies
// its own <!doctype>/<html>/<head>/<body> and rejects ours. This is the file
// that gets published; keeping it a build output (not a hand-edit) is what
// makes the hosted page reproducible from the repo.
const OUT_ARTIFACT = join(OUT_DIR, 'ipnn-m1-demo.artifact.html')

const result = await build({
  entryPoints: [join(APP, 'src/main.ts')],
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
<title>IPNN — M1 Living Demo</title>
</head>
<body>
<div id="app"></div>
<script>${safeJs}</script>
</body>
</html>
`

const artifactHtml = `<title>IPNN — M1 Living Demo</title>
<div id="app"></div>
<script>${safeJs}</script>
`

mkdirSync(OUT_DIR, { recursive: true })
writeFileSync(OUT, html)
writeFileSync(OUT_ARTIFACT, artifactHtml)
console.log(`${OUT}  ${(html.length / 1024).toFixed(1)} KB`)
console.log(`${OUT_ARTIFACT}  ${(artifactHtml.length / 1024).toFixed(1)} KB`)
