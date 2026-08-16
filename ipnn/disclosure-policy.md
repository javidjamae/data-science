# IPNN Disclosure Policy

Adopted 2026-08-16, while the project is a 3-pattern toy — deliberately.
Tripwires decided under excitement, sleep deprivation, or a shiny result are
worthless; these were pre-registered when the stakes were zero, for the same
reason experiment gates are pre-registered before running. Changes to this
policy are allowed but must be committed with written rationale.

## Default: open

Everything — vision, theory, the journal (failures included), and code — is
developed in public by default. Rationale, from the discussion that produced
this policy:

- **Priority-through-publication is the IP strategy.** The public timestamped
  record (journal + git history) is the claim to the ideas. Secrecy forfeits
  it; simultaneous independent discovery is the norm in science.
- **Scrutiny is the safety strategy at this scale.** A solo project's blind
  spots stay blind; the M1 answer-collapse failure (L-002) is the standing
  reminder that this system fails in unanticipated ways.
- **Secrecy doesn't buy a solo researcher control** — only delay, at the
  cost of collaborators, feedback, and the credibility that would matter if
  outside help were ever genuinely needed.

## The disclosure spectrum

"Open" is a dial, not a switch. Levels, from least to most enabling:

- **L0 — findings:** results, learnings, plain-language accounts.
- **L1 — mechanism:** full algorithmic detail and source code.
- **L2 — organisms:** trained brain states and exact configurations.
- **L3 — turnkey scale:** anything enabling push-button replication of a
  significantly capable system.

Current default: L0–L2 publish freely (today L3 is not meaningfully distinct
from L1; the level exists for the future).

## Tripwires

If ANY of the following occurs, pause publication of new L1+ material and
follow the procedure below. Private journaling continues — the record must
never stop, only its publication.

- **T1 — Unanticipated generalization:** capability on tasks meaningfully
  beyond training exposure, neither designed for nor predicted in the
  experiment's design.md.
- **T2 — Self-directed improvement:** the system materially improves its own
  learning process absent teacher signal.
- **T3 — Economically significant capability:** the system performs
  real-world tasks at a level a stranger would pay for.
- **T4 — The safety-researcher test:** any result where a competent AI-safety
  researcher would plausibly say "don't publish the recipe."
- **T5 — Credible outside concern:** a qualified third party raises a
  specific safety concern about already-published material.

## Procedure on a tripwire

1. Stop pushing L1+ detail. Keep journaling privately.
2. Within two weeks, get qualified outside eyes under confidentiality — at
   least one person with genuine AI-safety expertise (academic safety
   groups or a frontier-lab safety team; not public forums).
3. Decide jointly: resume open, adopt staged disclosure, or escalate further.
4. Journal the event and the decision (published in redacted form if
   necessary).

The premise throughout: if a tripwire ever genuinely fires, solo stewardship
is not a credible safety plan; the procedure exists to bring in others
early, not to protect a secret indefinitely.

## Licensing

- **Code: Apache-2.0** ([LICENSE](./LICENSE)) — permissive with an explicit
  patent grant, the norm for open ML research; maximizes collaboration and
  adoption.
- **Documentation (this directory's .md files, including the journal):**
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free reuse
  with attribution, which is exactly the priority/credit interest.
- **Deliberately not GPL:** copyleft governs code *expression*, not ideas —
  it would not prevent a closed reimplementation of the mechanism (the only
  thing worth protecting here) but would deter collaborators and adopters.
  Revisit if contributor dynamics ever change the calculus.

## Patents

Default: none — publication establishes priority and the moat in this field
is execution. If commercialization ever matters, file a provisional
*before* publishing the relevant mechanism (publication first destroys
patentability in most jurisdictions).

## Review

Revisit this policy at each experiment's `results.md`, and immediately upon
entering any qualitatively new capability regime.
