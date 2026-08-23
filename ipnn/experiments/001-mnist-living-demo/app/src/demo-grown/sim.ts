// DOM-free wiring for the experiment 002 demo.
//
// There is almost nothing here, and that is the point. The demo controller,
// the teacher, the balanced schedule, the sustained readout, manual mode, the
// accuracy bookkeeping — all of it is `DemoSim`, reused verbatim from
// experiment 001. This file only says *which substrate to build*, because
// `OrganismLike` is the entire surface between them (002 design §2, §9).
//
// The two arms are the ones the journal entry
// `2026-08-22-0208-exp002-m0-built-m1-fails-on-depth` reports, and switching
// between them live is the demo's actual argument: the substrate, the rule,
// the rent, the growth and the seed are identical, and one learns while the
// other never leaves chance.

import { DemoSim, type SubstrateFactory } from '../demo-m1/sim'
import { GrownOrganism } from '../engine/grown/grown-organism'
import { defaultGrownConfig, type GrownConfig } from '../engine/grown/config'

export interface GrownArmSpec {
  label: string
  /** one line, shown under the arm selector */
  note: string
  cfg: Partial<GrownConfig>
}

export const GROWN_ARMS = {
  m1: {
    label: 'M1 · as pre-registered',
    note:
      'Outputs on the far side of the sheet. No sense pixel ends up closer ' +
      'than 5 hops to an answer — and information survives only one hop, so ' +
      'the answer neurons read noise. This arm failed its gate: 0.157.',
    cfg: {},
  },
  shallow: {
    label: 'Shallow · outputs moved in',
    note:
      'The only change is where the output cortex sits. That brings ~19 of ' +
      'the 64 sense pixels within 2 hops of an answer, and the identical rule ' +
      'now learns: 0.883. Depth is the whole difference.',
    cfg: { outputX: 14 },
  },
} satisfies Record<string, GrownArmSpec>

export type GrownArm = keyof typeof GROWN_ARMS

export function grownConfig(arm: GrownArm, seed: number): GrownConfig {
  return { ...defaultGrownConfig, ...GROWN_ARMS[arm].cfg, seed }
}

export function grownSubstrate(arm: GrownArm): SubstrateFactory {
  return (seed) => new GrownOrganism(grownConfig(arm, seed))
}

/** A DemoSim driving the grown substrate. Same controller, different organism. */
export function grownSim(seed: number, arm: GrownArm): DemoSim {
  return new DemoSim(seed, grownSubstrate(arm))
}

/** Narrowing helper for the renderer, which is substrate-specific by nature —
 * it draws a lattice, and only this substrate has one. */
export function grownOrganism(sim: DemoSim): GrownOrganism {
  return sim.org as GrownOrganism
}
