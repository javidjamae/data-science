# What biology actually does, and the uninhabited corner 002 is standing in

- **Entry:** `2026-08-22-2057-biology-of-growth`
- **When:** 2026-08-22 20:57–21:30 CDT
- **Who:** Javid (the redirect: ground the growth model in how real neural nets emerge) **[J]**; synthesis **[J→C]**
- **Kind:** theory / direction (rule 6). Sources searched tonight and filed in [related-work.md](../related-work.md#how-real-brains-wire-themselves-searched-2026-08-22).

## In plain words

With the knob phase closed (L-040: fifty variations, zero pixels closer), the
question became: how does biology grow a working nervous system from cells?
The answer reorganizes the convergence program, because it says 002 was never
a simplified animal — it is a design no animal uses.

Real development is a **sequence of different regimes**, not one rule run
forever:

1. **The genome routes the coarse topology.** Layers form by migration before
   any learning; axons are steered to approximately the right *region* by
   inherited molecular gradients (ephrins, netrins, semaphorins). Activity
   never does the routing.
2. **Spontaneous activity is structured.** Retinal waves are travelling,
   spatially correlated fronts — neighbours co-fire, so the correlations
   themselves carry the topographic information. Disrupt the *pattern* and
   maps fail even with activity levels intact: the waves are instructive.
3. **Hebbian growth only refines.** Fire-together-wire-together operates on
   top of the inherited scaffold, choosing among roughly-right options.
4. **Overproduce, then prune.** Synapses are made in huge excess and then
   eliminated by use — selection over an abundance, not careful accretion.
5. **Plasticity is staged.** Critical periods open and close *on purpose*
   (and can reopen) — unlike L-034's accidental, terminal freeze.

Against that sequence, 002's design reads as one stage — (3), the refinement
stage — asked to do the work of all five, with white noise where the
structured waves should be. The knob ladder's uniform failure (`within2 = 0`
on all fifty arms) is what you would expect: **we asked the refinement
mechanism to do the routing job, and gave it noise to refine.** No animal
occupies that corner: *C. elegans* is wired almost entirely by genome; the
neocortex is grown — but grown *under guidance*, out of *structured*
activity, through *staged* regimes.

One more thing falls out, and it dissolves an old worry. H-015 already said
"having a genome is not cheating" **[J]**; biology says the genome's role is
specifically **topology priors** — which region connects to which — while
activity supplies the content. An inherited laminar scaffold in 002 is not a
retreat to 001's hand-wiring: 001 hard-codes every synapse; a scaffold biases
*where growth looks*, and every actual connection is still grown, refined,
rented and killed by the organism's own life.

## How this refits the mechanism program

The four mechanisms queued after L-040 survive, but two get renamed by
biology and two new requirements appear that no diagnosis of ours had found:

| planned mechanism | biological reading |
|---|---|
| M1e output beacon | a **target-derived guidance cue** — the standard way axons find a distant target (stage 1, special case) |
| H-006 correlation-seeking growth | **Hebbian structural refinement** (stage 3) — and it only works if the activity contains correlations, which is new requirement #1 |
| H-012 scaffolding | consolidation-biased refinement (stage 3/4 boundary) |
| α/β port | the substrate's plasticity dial — which stage 5 says should be *scheduled*, not just un-broken |
| **new: structured spontaneous waves** (H-021) | stage 2 — the bootstrap activity must carry topology; iid `pSpont` carries none |
| **new: inherited laminar affinity gradients** (H-020) | stage 1 — the genome routes coarse topology; growth chooses within it |
| **new: developmental schedules** (H-022) | stages as a sequence — overproduce→prune, plastic→consolidated |

## Decisions

1. **The convergence program is re-founded as a developmental program** —
   experiment 006, "the developing substrate", to be pre-registered with the
   five stages as separately ablatable components (waves-without-scaffold,
   scaffold-without-waves, no-schedule, etc.), each arm against 002's
   recorded baseline. H-020 is its central hypothesis.
2. M1e and H-006 are **absorbed into 006** under their biological readings
   rather than run as isolated patches.
3. No novelty is claimed for any of this — it is the most-trodden ground in
   developmental neuroscience, which is exactly why it belongs under a
   substrate that has just proven immune to parameters.

## Threats to validity

This entry is synthesis, not measurement. Its factual claims are sourced in
related-work.md; its central prediction (H-020) is stated so that 006 can
refute it — if scaffold + waves + correlation growth still leaves `within2`
at zero and accuracy at chance, the biological reading of 002's failure is
wrong, and that would itself be worth knowing.
