# Trioron Handoff

**Session date:** 2026-05-31 → 2026-06-02
**Session number:** 012
**Session title:** Embodied organism arc — mini-Minecraft world, Numa/Mima,
the immortality plateau, population vs solo, and apprenticing (the social-
learning verdict)

> Rewritten in full every session; previous handoffs in git history
> (`git log docs/handoff/HANDOFF.md`). Session 011 (commit `ab72d40` and
> earlier) covers the SUBSTRATE work that precedes this: substrate-was-purely-
> linear → quad nonlinearity in core → selective quad growth → satellites
> (memory) → unison architecture (full DMS). READ that too if you're new.

## Summary

This session pivoted from substrate mechanisms to an **embodied organism in a
world** — triggered by Rocky's reality-check ("are we hallucinating?"). The
honest through-line: we kept chasing a self-deception risk, and Numa/Mima (from
the Aidos project) turned out to be the *intrinsic* answer to it.

Arc:
1. Built a **mini-Minecraft survival world** (`experiments/world/tile_world.py`)
   — tiles, homeostatic drives, day/night, a predator; state-vector percepts
   (no pixels). The "alive" test.
2. **Organism v1/v2/v3** (Q / world-model / model-based planning) all plateaued
   at ~18 survival, below a hand-coded reactive heuristic (~22). Diagnosis
   (Rocky): the **world's lethality ceiling**, not the algorithm.
3. **Numa/Mima** (`numa.py`, from Aidos): Numa = a contrastive loss-drop that
   PERSISTS on re-test (real consolidated learning); Mima = one that REVERTS
   (confabulation). Net Numa is the right "alive" metric, not survival. Organism
   accrues net +1.149 (n=3) — it genuinely learns; a heuristic accrues 0.
4. **Reduced world difficulty** so skill discriminates → the learning organism
   now BEATS reactive (43.8 > 38.5). The plateau *was* the world.
5. **Population** (mortal, selection) + **shared-replay** + **apprenticing**:
   on the discriminating world, **SOLO wins (50.3)**; all social mechanisms
   underperform — apprenticing-to-peers is *worst* (36.0). Social learning HURTS
   when there's no master to copy (copying clueless peers spreads mediocrity).
6. **Apprenticing to a competent master** (teacher-student): a MODEST benefit —
   ~9% faster to competence (51 vs 56 episodes, n=3), limited by a weak teacher
   (39.8). The 1-seed smoke showed ~2× but that was a lucky seed (a near-Mima,
   flagged).

Honest meta-conclusion: **solo individual learning is the workhorse at testable
scale.** Social learning gives a small early boost *with* a competent master and
*hurts without one* — matching the social-learning-strategies finding (copy
selectively, copy the successful). The real win was the difficulty-reduction
revealing the solo organism is genuinely alive (learns to out-survive a
hardcoded heuristic).

## Vocabulary (locked this session — keep distinct)

- **Numa** — learning you EARNED yourself, that consolidates (persists on
  re-test). The substrate's food.
- **Mima** — COUNTERFEIT learning (reverts on re-test). Confabulation. From ANY
  source. (Greek *mîmos* = imposter/mime of real learning — NOT "copy a teacher".)
- **Apprenticing** — genuine SOCIAL learning: skill handed down from a competent
  MASTER (cf. Abbeel & Ng apprenticeship learning). Substrate = **mirror cells**.
  The word encodes the finding: you apprentice under a *master*, never a peer.
- **mirror cells** — the cell type that fires on observed action (substrate of
  apprenticing). NOT YET wired as a real cell type — see open items.
- **shared-replay** — deprecated proxy (pool everyone's transitions); was
  mislabeled "mirror". It HURTS (inconsistent value targets across policies).
- Apprenticed knowledge still gets Numa/Mima-validated: genuine → Numa,
  cargo-cult → Mima.

## Headline numbers

**Survival (difficulty-reduced "discriminating" world; n=60 baselines):**
| agent | greedy survival |
|---|---|
| random | 27.9 |
| reactive (hardcoded) | 38.5 |
| organism v1 (learned Q) | **43.8 ± 2.8** (beats reactive) |

**Numa (organism v2, n=3):** net Numa **+1.149 ± 0.230** (Numa ~2.2, Mima ~1.0
caught), survival ~18.5 on the *old* lethal world. (Learning ≠ survival unless
the model is used to act — v2's world-model wasn't in the action loop.)

**Population (discriminating world, pop=8, n=1):** solo **50.3** > shared-replay
45.9 > pop-selection 41.6 > pop-apprentice 36.0. Social all < solo.

**Apprenticing (teacher-student, n=3):** apprentice 51 eps→competence vs unaided
56; early 29.1 vs 25.5; final ~40 both. Modest, teacher-quality-limited.

## What was done (this session's files, all in `experiments/world/`)

- **tile_world.py** — the world. Drives (energy/thirst/temperature/integrity),
  scent gradients, day/night = wake/dream, predator. DIFFICULTY-REDUCED at the
  end (slower clocks, bigger grid size=12, minor predator) so resource-seeking
  is the binding skill. Baselines `run_random`/`run_reactive`.
- **organism_v1.py** — Q-learner, predict-then-act on discounted drive-value.
  GREEDY eval. The workhorse (beats reactive on the easy world).
- **organism_v2.py** — + a pair-prediction (world-model) head → accrues Numa.
- **organism_v3.py** — model-based planner (rollout through a learned transition
  model). Honest null (15.4); inaccurate model → compounding rollout error.
- **numa.py** — the 5 contrastive pairs from world state + `NumaLedger` with
  dream-validation (train-batch vs held-out-batch per-pair loss).
- **population.py** — mortal population; arms `solo / pop-selection /
  shared-replay / pop-apprentice`. Selection = clone-a-survivor + mutate.
  Apprentice = copy the current fittest's actions (cross-entropy).
- **teacher_student.py** — apprenticing to a FULLY-TRAINED master; `unaided` vs
  `apprentice`; metric = episodes-to-beat-reactive (learning speed).
- **docs/design/organism_roadmap.md** — the bottom-up curriculum (sense→survive→
  symbol→language→communication→self-model) + the backlog (elf-tree/language
  artifacts, shrine, mirror cells, self-observation, Numas/Mimas).

## Key findings

1. **The survival plateau was the world, not the learner.** Q / world-model /
   model-based / population / apprenticing all plateaued ~15-18 on the lethal
   world (random 15, perfect 22 — too narrow to discriminate). Widen the band
   (easier world) → the solo learner climbs to 43.8 > reactive 38.5.
2. **Numa/Mima is the intrinsic anti-hallucination metric.** Real learning
   persists on re-test; fake reverts. Net Numa is what "alive" should mean —
   survival is just the precondition (a reactive heuristic survives but accrues
   0 Numa: it never learns).
3. **"Immortality grants laziness" — partially true but not the binding issue.**
   Costless respawn (Edge-of-Tomorrow loop) removes stakes, BUT at testable
   scale the immortal SOLO learner is the best survivor, not the laziest.
   Mortality+selection+social-learning all *hurt* at pop=8.
4. **Social learning needs a master.** Copying clueless peers (population from
   scratch) spreads mediocrity, collapses exploration diversity, and suppresses
   own RL → worse than solo. Apprenticing to a competent teacher helps (modestly,
   proportional to teacher quality). This is the explore-vs-exploit / innovation-
   vs-imitation tradeoff: independent search when no one knows, copy when someone
   does.
5. **Lineage, not the individual, is what selection should preserve** (Rocky).
   Mortal individual (stakes) + immortal lineage (reproduction). God-as-parent
   (nurture the young) is legitimate; God-as-rescuer (save favorites) recreates
   immortality and manufactures Mima.

## Decisions made

- **Adopt the vocabulary above** (Numa / Mima / apprenticing / mirror cells /
  shared-replay).
- **Solo learner on the discriminating world is the validated "alive" result.**
- **Don't build the 10k vectorized population on the strength of a hypothesis the
  small-scale data pushes back on** — unless we pursue scale + a shared
  competitive world (where apprenticing should shine and altruism fails).
- **Keep "alive" in scare-quotes** until results keep surviving re-test.

## Open questions / next-up

1. **Mirror cells as a real cell type** (still only a function-level proxy) —
   the substrate of apprenticing, and groundwork for the self/other rungs
   (self-observation, the shrine). Build like satellites/quad if we go there.
2. **Strong-teacher apprenticing** — re-run teacher-student with the ~50 solo
   learner as master (not the weak 39.8 one) and a harder task, where solo
   exploration is costly. That's where apprenticing should show a big effect.
3. **Close the Numa→survival loop properly** — v2's world-model isn't used to
   act. A planner that *uses* the Numa-validated model (and a better one than
   v3's) could make learning translate to survival.
4. **Shared competitive world** (organisms in ONE world, competing for the same
   tiles) — Rocky's intuition that apprenticing wins under competition while
   altruism fails. Needs vectorizing for scale (10k = ~100s MB vectorized vs
   ~500 GB as full substrates — see below).
5. **Numas/Mimas from Aidos are richer than our stub** — real version uses
   Trioron's per-pair contrastive loss + dreaming replay; ours uses a held-out
   re-test proxy. Aidos blueprint at `~/project-aidos/docs/blueprint_v3.md` §7.

## 10k-individuals feasibility (measured)

One full-trioron organism = **~50 MB** (capacity-2048 arena) + ~0.55 ms/forward.
**10k as-is ≈ 500 GB / weeks of compute — infeasible** on the 7.4 GiB box
(realistic ceiling ~50-100 full individuals). **10k IS feasible vectorized**
(each organism is a tiny 74→32→6 net ≈ 10 KB; stack into batched tensors, one
`bmm`/step → ~100-200 MB). Tradeoff: vectorized = plain nets, loses per-individual
trioron machinery (growth/satellites/quad). Population dynamics → vectorize;
per-individual substrate studies → cap at tens.

## Pointers

- **`experiments/world/`** — the whole arc (see What-was-done).
- **`~/project-aidos/`** — Numa/Mima/apprenticing concepts originate here
  (`aidos/numa.py`, `aidos/sim/contrast.py`, `docs/blueprint_v3.md` §7). SEPARATE
  project, own Claude memory dir. Read before extending Numa/Mima.
- Logs (uncommitted, /tmp): `run_organism_easyworld.log`,
  `run_population_imitate.log`, `run_teacher_student.log`,
  `run_organism_v2.log`.

## Memories saved this session (cross-PC)

- `numa-mima-implemented` — Numa/Mima mechanism + the reframe (Numa is the metric).
- (011) `substrate-was-purely-linear`, `quad-dendrite-comparison-result`,
  `selective-quad-growth`, `satellites`, `temporal-gate-pass`.
- TODO next session: save an `apprenticing-vocabulary` + `social-learning-verdict`
  memory (solo wins, apprenticing needs a master, the vocab).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `v2.0-scaffold`, all pushed
  (HEAD `b83e60d`). Python 3.10.12, torch 2.11.0+cu130, WSL2, 12 cores, 7.4 GiB.
  `python3`, `OMP_NUM_THREADS=8`. sklearn/scipy/numpy available.
- Full-trioron population runs are SLOW (~70 min for pop=8 ×2seed ×12k steps) —
  the 50 MB/organism cost. Reduce scale or vectorize for more.
- Pre-existing: 4 test failures on clean HEAD (not ours); 3 session-005 carried
  uncommitted files (do not touch).
