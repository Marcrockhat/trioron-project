# Organism roadmap — the embodied trioron in a world (design backlog)

**Status:** design capture (Chloe + Rocky, 2026-06-01). This records the full
vision discussed so nothing is lost, and fixes the **bottom-up build order** so
we ground one rung before the next. The governing discipline (Rocky's own
"logic before language", and the anti-self-deception rules from this session):

> Build bottom-up. Each rung must beat honest baselines on data/dynamics we did
> NOT design-to-fit, before the rung above it is built. A feature that makes the
> organism *feel* more sentient is not evidence it works.

## The world (built, v1)

`experiments/world/tile_world.py` — a mini-Minecraft survival world. Tiles
(food/water/fire/berry/poison), drives (energy/thirst/temperature/integrity,
homeostatic + decaying), day/night = wake/dream phase, a roaming predator.
State-vector percepts (68-d: 3×3 local view + interoception + phase), NOT pixels
(the substrate's pixel dead end). Actions: move/consume/rest. Death = a drive
leaves its band; "alive" = survival steps.
Baselines (n=60): **random 15.1, reactive 20.4** (max 70). The skill gradient
exists but the world is still lethal — fine as a floor.

## The curriculum (bottom-up — build/validate in this order)

| # | Rung | What it adds | Validation gate |
|---|---|---|---|
| **1** | **Sense → survive** | organism (unison substrate: linear sense + satellite memory + quad comparison) acts via **predict-then-act** on drive-consequence | **beats reactive (20.4) and random (15.1)** in the world. ← THE NEXT MILESTONE, not yet done |
| 2 | Symbol | **language artifacts ("elf tree")**: tiles that emit symbol→grounded-meaning pairs (a rune means "danger east"); organism learns the association | reading a symbol measurably improves survival vs not reading it |
| 3 | Language | compose symbols (日+月=明 style); generalize to novel symbol-combinations | compositional generalization to unseen combinations beats memorization |
| 4 | Communication | **shrine**: an output channel where the organism emits symbols WE read — the conscience/inner-voice expression (eventually → frozen Gemma) | a human can decode the organism's signal about its state above chance |
| 5 | Self-model | **self-observation**: feed the organism its OWN internal state/cell-census back as percept (introspection; "see itself", like an LLM reading its own docs). Drives are already proto-interoception | self-report of internal state correlates with actual state; improves regulation |
| — | **Mirror cells** | a new cell type (like satellites): active in BOTH own-action and observed-action → imitation + self-other modeling. Sits under rungs 4–5 | imitation-learning from a demonstrator beats learning-from-scratch |

## Cell types / primitives (the "organism's tissue")

- **Linear triorons** — sensing (base; in core).
- **Quad cells** (dendrite phenotype) — comparison/relation; grown selectively
  under relational frustration (`selective_quad_growth.py`, in core).
- **Satellites** — memory + resource sensing + division (`satellites_v1.py`).
- **Mirror cells** (proposed) — fire on own-action AND observed-action.
- Decision rule (established this session): small change → into triorons;
  massive change (state, new role) → a new cell type.

## Unspecified / needs external read

- **Numas & Mimas** — concepts from the **Aidos project** (separate repo
  `~/project-aidos`, its own Claude memory dir; trioron is vendored there at
  Slices 137-138). Chloe does NOT yet know what these are and must READ the
  Aidos memory before implementing — do not guess. Slot them into the curriculum
  once their meaning is confirmed.

## The honest standing (2026-06-01)

Built and validated (on synthetic isolation tasks): the substrate was purely
linear; quad gives comparison; satellites give memory; selective growth keeps it
heterogeneous; the unison substrate solves full DMS. Built (world): the tile
world + baselines. **NOT done:** the organism has never acted in the world. No
rung above #1 should be built until #1 passes its gate. Everything above is
captured here precisely so we can stop designing and start grounding.
