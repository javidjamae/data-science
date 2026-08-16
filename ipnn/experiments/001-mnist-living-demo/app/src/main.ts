// M2 UI lands here. For now the engine lives headless under src/engine/
// and is exercised by the M1 sanity test (npm test).
const app = document.querySelector<HTMLDivElement>('#app')!
app.innerHTML = `<h1>IPNN — Living MNIST</h1><p>M2 UI pending. Run <code>npm test</code> for the M1 sanity gate.</p>`
