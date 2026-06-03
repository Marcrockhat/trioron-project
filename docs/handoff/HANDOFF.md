# Trioron Handoff

**Session date:** 2026-06-02 → 2026-06-03
**Session number:** 013
**Session title:** Mirror cells as a real cell type → emergent pyrophobia →
fire-taming by apprenticeship (BC fails, DAgger works) → Adam's quest for
immortality (multi-master apprenticeship, consolidation, the measurement wall)

> Rewritten in full every session; previous handoffs in git history
> (`git log docs/handoff/HANDOFF.md`). Session 012 (commit `4eac26c`) is the
> embodied-organism arc this builds on (tile_world, Numa/Mima, apprenticing
> verdict). Session 011 is the substrate work (quad, satellites, unison). Read
> 012 first if new — this session completes its open item #1 (mirror cells).

## Summary

Picked up session 012's open item #1 — **wire mirror cells as a real cell type**
(they were only a function-level cross-entropy proxy). That opened into a long,
productive arc that ended at an honest measurement wall:

1. **Mirror cells wired** as a real `MIRROR` gene/cell type (observation channel +
   localized credit gating). Smoke passes.
2. **Master-avatar mechanism** (Rocky's reframe): the obs channel is a *hole* that
   lets a master *pilot the body* (teleoperation). Kept it as a feature + added an
   internalization term so the skill also transfers to solo play.
3. **n=3 mirror-cells verdict**: real cell type does NOT beat the loss-only proxy
   on solo (teacher-quality-limited), but avatar-piloting works (41.1, best).
4. **Autopsy + probe**: the organism dies of **temperature (cold), not the
   predator**. It learned **emergent pyrophobia** — actively avoids fire (3.5×
   less than random when cold) → freezes. Approach-avoidance resolved by phobia.
5. **Fire is untamable at canonical physics** (+0.15 warm vs ~0.01 cool); even a
   perfect-info oracle overheats. Parameterized `TileWorld` physics; tamed world
   (warm 0.04, death band 0.02/0.99) makes fire learnable.
6. **Fire-taming by apprenticeship (n=3, the clean result)**: BC apprentice FAILS
   to internalize (distribution shift). **DAgger WORKS**: cold-deaths −37%,
   fire-occupancy ↑ toward the oracle, survival +28%. Coaching on the student's
   OWN states is the load-bearing difference.
7. **The Quest** ("outlast your masters"): crowned **Adam** (seed 2). Sequential
   DAgger apprenticeship. Chapter 2 (fire) clean: cold 26→14, occupancy 7.4→20.1,
   death-cause flipped cold→thirst. Chapter 3 (water): **no forgetting but water
   didn't take** — the 8 mirror cells are saturated by one skill.
8. **Capacity sweep + consolidate-and-lock**: both **inconclusive — measurement
   noise > effect size** at n=3. The tile-world is too noisy for quantitative
   skill-retention deltas. The one clean-ish hint: even a *locked* skill erodes,
   suggesting interference leaks through the unfrozen shared `interior→output`
   base policy.
9. **Side quests**: measured real NPC RAM (~158 KB/brain — RAM is NOT a
   bottleneck); discussed RPG-NPC application (strong fit; emergent + teachable +
   lineage; trioron "soul" + LLM "mouth").

## Headline numbers

**Mirror-cells teacher-student (n=3, weak teacher 39.8):**
| arm | final solo | avatar-piloted |
|---|---|---|
| scratch | 40.0 | — |
| loss-only proxy | 40.4 | — |
| mirror-cells | 37.4 | **41.1** |

**Fire-taming (n=3, tamed-fire world; the clean win):**
| | survival | cold-deaths | fire-occupancy |
|---|---|---|---|
| fire-oracle (teacher) | 83–90 | 0/40 | 28% |
| scratch | 43.8 | 23.3/40 | 12.9% |
| BC apprentice | — | 24.0/40 | 9.2% (FAIL) |
| **DAgger apprentice** | **56.0** | **14.7/40** | **18.6%** |
| DAgger avatar-piloted | — | — | 80.7 |

**The Quest — Adam (seed 2):** baseline 49.6 → after-fire 55.1 → after-water 55.3.
Masters: fire 90.3 / water 70.6 / food 71.7. **Bar to beat: 90.3.** Chapter 2
fire clean (cold 26→14, occ 7.4→20.1, death flipped to thirst). Chapter 3 water
flat (thirst 21→21, fire retained = no forgetting, but no acquisition).

**Consolidate-and-lock (n=3, capacity-matched 16 cells):** INCONCLUSIVE. PLAIN
fire-occ 12.4→10.7; LOCKED 12.0→12.7. Effect < between-seed spread (4.8–18.9%);
occupancy and cold-death metrics CONTRADICT. Not a result.

**NPC RAM (measured, inference mode):** right-sized (cap 256) = **158 KB resident**
/ brain (matches the int8-compressed 157 KB). 1k NPCs ≈ 150 MB, 10k ≈ 1.5 GB.
Training default (cap 2048) was 6× over-provisioned (~890 KB).

## What was done (files)

New (all `experiments/world/` unless noted):
- **mirror_cells.py** — `MIRROR` cell type: observation channel (master action
  one-hot) + mirror cells (interior efference + obs → output), `keep_only_mirror_grads`
  (localized credit), master-avatar training (avatar CE + internalization CE),
  lesion/obs helpers, `--smoke` (wiring proof) + `--verdict` (n=3).
- **fire_taming.py** — fire oracle (tap-the-stove), tamed-fire physics override,
  BC vs DAgger arms, `train_dagger_student`, avatar-pilot eval.
- **quest.py** — masters (fire/water/food specialists) + `crown_protagonist`.
- **quest_chapter.py** — resume-capable DAgger chapters (`dagger_resume`); Adam
  accumulates skills in `runs/protagonist.pt`.
- **quest_consolidate.py** — consolidate-and-lock (`consolidate` via
  `edge_protected`, `grow_mirror`, `freeze_grads`); PLAIN vs LOCKED n=3.
- **capacity_test.py** — mirror-cell count sweep (8/64/128).
- **measure_npc_ram.py** — real runtime RAM per brain (tensor bytes + RSS delta).
- **diagnose_death.py** — cause-of-death autopsy.
- **probe_temp_conflict.py** — approach-avoidance / pyrophobia probe.
- **render_organism.py** — world GIF + Cytoscape anatomy (MIRROR cells gold).
- **render_fire_taming.py** — side-by-side solo vs avatar-piloted GIF.

Modified:
- **trioron/core/epigenome.py** — added `MIRROR = 10` (marker gene, NOT an
  expression gene; mirror cells dispatch as LINEAR).
- **trioron/core/?** none else.
- **experiments/world/tile_world.py** — parameterized physics as class attrs
  (`WARM_RATE=0.15`, `TEMP_LOW=0.05`, `TEMP_HIGH=0.98`) + per-instance overrides.
  **Defaults unchanged** — existing benches unaffected. fire_taming overrides to
  warm 0.04 / band 0.02–0.99.
- **render_organism.py** `train_solo(..., n_mirror=8)`; **mirror_cells.py**
  `build_mirror(..., capacity=2048)`.

## Key findings

1. **Mirror cells work as a real cell type** (smoke: fires on observed action,
   credit stays on mirror cells with zero leak, lesion-able). Localized-loss design.
2. **The master-avatar "hole" is a feature, not a bug** (Rocky): the obs channel
   is teleoperation — a master pilots the body to near-its-own competence (80.7).
3. **Emergent pyrophobia**: nobody coded fear of fire; the value learner discovers
   it from the burn penalty → avoids fire → freezes. The most "alive" result here.
4. **BC fails, DAgger works** (the clean, defensible win): watching a master
   (behavioural cloning) doesn't transfer a hard skill — distribution shift; the
   student never practises its own mistakes. Coaching on the student's OWN states
   (DAgger) does: −37% cold-deaths, fire-use up toward the oracle.
5. **One skill saturates 8 mirror cells**; a second skill needs growth or
   modularity, not cramming. BUT —
6. **We hit a measurement wall.** The tile-world's death-counts swing ±15 (predator
   luck, spawns); chasing 2–5pp skill-retention deltas at n=3 is hopeless. Capacity
   sweep and consolidate-and-lock are both noise-dominated. Be honest: these are
   QUALITATIVE demos, not quantitative results.
7. **The quest pipeline never invoked Numa/Mima or any consolidation** (Rocky's
   diagnosis) — it was raw SGD. Plus each specialist master is blind outside its
   drive, so a later master actively un-teaches an earlier skill. The real missing
   mechanism is consolidation/integration, and even locking mirror edges leaks
   through the unfrozen shared `interior→output` base policy.
8. **RAM is a non-issue** for the NPC application (~158 KB/brain). Real constraints:
   compute-per-tick (solve by slow/staggered ticking + vectorization) and the
   capacity-vs-skill-richness tradeoff.

## Decisions made

- Mirror cells = real `MIRROR` cell type, **localized-loss** design (credit gated
  to mirror cells).
- Master-avatar: **keep the obs hole** as teleoperation + add an internalization
  term so the skill transfers to solo play.
- **DAgger is the apprenticeship method** (BC is dead for hard skills).
- **Sequential** apprenticeship, not council (council's boundary flip-flops confuse).
- **Adam = seed 2** is the protagonist; persistent identity in `runs/protagonist.pt`.
- Tamed-fire physics is **parameterized, defaults unchanged**.
- **Paused** at the measurement wall — not worth burning hours forcing a number out
  of a noisy metric.

## Open questions / next-up

1. **Lower-noise testbed** is the blocker for everything quantitative. Either a
   controlled fire-probe metric (fixed cold-state battery, not whole-episode
   occupancy) + n=8+, or a quieter world (less predator/spawn variance), or accept
   the quest as a qualitative demo.
2. **Consolidation done right**: freeze/anchor the **base `interior→output` policy**
   too (manifold-anchoring — the substrate has it, we bypassed it), not just mirror
   edges. The leak is there.
3. **Actually invoke Numa/Mima** in the apprenticeship loop (validate a skill is
   real → lock it → only then learn the next). Currently never used in the pipeline.
4. **Master integration**: specialists are drive-blind; a later master un-teaches an
   earlier skill. Need scoped masters or a council-with-context. Food chapter never run.
5. **Strengthen the fire chapter** in sweeps (fresh-net DAgger only reached ~12%
   occupancy vs Adam's warmed 20%) — can't measure retention of a barely-learned skill.
6. **RPG-NPC pilot** (product direction): a teachable survival companion (emergent
   quirk + learn-by-demonstration + persistent across save/load). The quest IS ~80%
   of this. Lineage/culture version depends on solving the multi-skill problem.

## Pointers

- **`experiments/world/`** — the whole arc.
- **`runs/`** (LOCAL, uncommitted — gifs/.pt/.log): `protagonist.pt` (Adam, seed 2,
  skills nominally `['fire','water']` though water didn't take), `mirror_organism.pt`
  (probe organism, OLD harsh-world), `fire_taming_sidebyside.gif`,
  `organism_world.gif`, `organism_anatomy.{html,png}`, `world_legend.png`, and the
  `*_run*.log` / chapter logs.
- **`~/project-aidos/`** — Numa/Mima origin (separate project, own memory dir).
  The real Numa/Mima is richer than our held-out-retest proxy; read before wiring it.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `v2.0-scaffold`. Python 3.10.12,
  torch 2.11.0+cu130, WSL2, 12 cores, 7.4 GiB. `python3`, `OMP_NUM_THREADS=8`.
  matplotlib 3.10.9, imageio 2.37.3, networkx 3.4.2 available.
- **Pre-existing, DO NOT TOUCH**: 3 session-005 carried uncommitted files —
  `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
  `trioron/viz/export.py`. (export.py is why the anatomy MIRROR colour is injected
  into the generated HTML, not the source.)
- Background runs are buffered to file unless launched with `python3 -u` /
  `PYTHONUNBUFFERED=1` — use that for live progress (learned this session).
- tile-world single-seed runs are HIGH variance (±15 on death counts) — n≥8 or a
  controlled metric is required for any quantitative claim.
