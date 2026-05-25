# Trioron Handoff

**Session date:** 2026-05-25
**Session number:** 005
**Session title:** CL machinery debug + developmental program + GA optimization

> This file is **rewritten in full** every session. The previous
> session's handoff is preserved in git history
> (`git log docs/handoff/HANDOFF.md`). Read this file first when
> starting any new session; rewrite it before ending one. See
> `docs/handoff/README.md` for the convention.

## Summary

Major session: fixed 4 CL bugs, implemented 8 lifecycle modules,
built the developmental program (spec + code), ran GA hyperparameter
optimization to 0.704 full accuracy (exceeds v1's 0.601), and
identified the remaining forgetting bottleneck (shared output cells).

## Headline numbers

| Config | Full | Task-Aware | Notes |
|---|---|---|---|
| Session start (broken) | 0.199 | 0.911 | No locking, no growth |
| GA v3 best (seeded) | **0.704** | — | Evolved hyperparams, seeded base |
| Best genome rerun | **0.585** | **0.955** | With forgetting analysis |
| Developmental (256 stem) | 0.208 | 0.859 | Self-organized, no pre-wiring |
| v1 manifold-grown (n=10) | 0.601 | 0.961 | Previous best to beat |

Mean forgetting: **34.6%** (each task loses ~1/3 of peak by end).
Worst: fashion_4_5 at 73%. Best non-last: fashion_8_9 at 18%.

## What was done

### Bug fixes (4)
1. Activation tracking — scheduler stores `_last_activations`
2. Param cap — 32KB → 200KB (initial substrate is 107KB)
3. Edge buffer — `_EDGE_FAN_FACTOR` 16 → 64
4. Dream replay exclusion — `current_classes` list, not single int
   (was destroying just-learned task by ~95%)

### Replay architecture (3 structural changes)
1. **Interleaved replay** — manifold pseudo-samples mixed into
   training batches alongside real current-task data
2. **Dream replays ALL classes** — including current (prevents 95%
   destruction)
3. **Astrocyte-gated replay** — top-K most at-risk classes by
   manifold log-likelihood. O(K) cost regardless of class count.

### Bidirectional gradient freeze
`zero_dormant_grads` now freezes edges BOTH into AND from dormant
cells. Previous unidirectional freeze let new tasks corrupt locked
pathways via their output-facing edge weights.

### Lifecycle modules (8 implemented)
saliency, compact, ship, wake, extend, graft + bases/frozen,
bases/compose. Full pipeline verified end-to-end.

### Developmental program
- **Spec §5.7** inserted into `paper/v3/spec.md` (347 lines)
- **Implementation**: stem cells, morphogen field (adaptive 16-param),
  axon guidance (proximity-based), positional differentiation,
  division orientation (symmetric=width, asymmetric=depth)
- **Band structure**: stems in z-bands (4 bands × n/4 cells)
- All phenotypes registered with linear forward (graceful degradation)

### GA hyperparameter optimization
- `experiments/evolve_hyperparams.py` — 29-gene genome
- GA v3: 8 generations × 8 population, seeded base + interleaved replay
- Best: 0.704 full at h=55, lr=6.68e-4, batch=30, theta_e=0.516,
  consecutive_tasks=4
- **Defaults updated** in all config dataclasses

### Visualization
- Cytoscape.js network viewer (`trioron/viz/export.py`)
- Perception cells filtered at export for readable layout
- `experiments/viz_developmental.py` — captures development stages

### Key architectural findings
1. **Interleaved replay** was the biggest single accuracy fix
2. **Bidirectional gradient freeze** is essential for locking
3. **Shared output cells** are the remaining forgetting bottleneck —
   locked pathways get corrupted when new tasks adjust output edges
4. **Independent pathways** (perc→dedicated→output) need per-task
   output cells to work; shared outputs leak
5. **Developmental base** with sparse connectivity can match seeded
   on task-aware (0.859 vs 0.793) at 6× fewer edges

## State of the build

- **Branch:** `v2.0-scaffold` (up to date with remote)
- **Key commits this session:**
  - `f6cad25` — fix: activation tracking, param cap, growth print
  - `1984a31` — lifecycle: ship/wake/extend/graft/compact/saliency
  - `30b407c` — fix: interleaved replay + dream replays all
  - `03de08a` — feat: astrocyte-gated replay
  - `c79c529` — spec: §5.7 developmental program
  - `34a8e92` — feat: developmental base implementation
  - `205fd60` — feat: division orientation + band structure
  - `68fee5a` — fix: bidirectional gradient freeze
  - `781b4a3` — defaults: evolved hyperparameters from GA v3
- **Working tree:** clean (untracked: `runs/`)

## Evolved default hyperparameters (GA v3 best)

```
h_init=55  batch=30  lr=6.68e-4  epochs=3  n_grow=9
theta_e=0.516  consecutive_tasks=4  g_min=3.92e-6
engagement_decay=0.1  lock_base_rate=0.078
frust_window=86  frust_hinge=2  frust_gain=1.11  frust_ceiling=5.04
dream_replay_bs=36  dream_replay_lr=0.00389
dream_reju_rate=0.084  dream_recycle_rate=0.462
manifold_replay_steps=14
growth_inherit_frac=0.246  growth_new_edges=12
```

## Forgetting profile (best genome, single seed)

```
Mean peak full:  0.931
Mean final full: 0.585
Mean forgetting: 0.346
Worst: fashion_4_5 (0.73 forgotten)
Best:  fashion_8_9 (0.18 forgotten)
Task-aware: 0.955 (near-perfect)
```

## Next-up tasks

1. **Per-task output cells** — the remaining forgetting bottleneck.
   Each task should have dedicated output cells; final prediction
   aggregates via routing. This is the path to 85%+ full accuracy.
2. **Developmental base + evolved params** — rerun developmental
   base with the GA-evolved hyperparameters (currently only tested
   with hand-picked defaults).
3. **CONV/ATTENTION phenotype implementations** — currently all
   phenotypes fall through to linear. Implementing real conv/attention
   forward functions would give the developmental base actual
   architectural diversity.
4. **Visualization fix** — Rocky reports black canvas on some views.
   The Cytoscape.js CDN approach works but layout may need tuning
   for larger graphs.
5. **Weight sharing for CONV** — spec'd in §5.7.4a but not
   implemented. Needs `weight_root` tensor in arena.

## Pointers

- **`experiments/evolve_hyperparams.py`** — GA optimization
- **`runs/evolve_v3/best_genome.json`** — winning genome
- **`trioron/lifecycle/developmental.py`** — stem cells, morphogen,
  axon guidance, differentiation
- **`trioron/bases/developmental.py`** — developmental base
- **`trioron/core/scheduler.py:192`** — bidirectional gradient freeze
- **`paper/v3/spec.md` §5.7** — developmental program spec

## Environment notes

- Working directory: `/home/marcrockhat/trioron-project/`
- Branch: `v2.0-scaffold` (commit `781b4a3`)
- Python: `/usr/bin/python3` (3.10.12) — `torch 2.11.0+cu130`
- Platform: Linux (WSL2)
