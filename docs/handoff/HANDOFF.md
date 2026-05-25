# Trioron Handoff

**Session date:** 2026-05-25
**Session number:** 004
**Session title:** CL machinery fixes + developmental program spec

> This file is **rewritten in full** every session. The previous
> session's handoff is preserved in git history
> (`git log docs/handoff/HANDOFF.md`). Read this file first when
> starting any new session; rewrite it before ending one. See
> `docs/handoff/README.md` for the convention.

## Summary

Fixed four bugs in v2 CL machinery, implemented 8 lifecycle/base
modules (ship/wake/extend/graft/compact/saliency/frozen/compose),
built astrocyte-gated replay, ran GA hyperparameter optimization,
and drafted + inserted §5.7 (Developmental Program) into the spec.

## What was done

### Bug fixes (4)
1. **Activation tracking** — scheduler now stores `_last_activations`
   for credit engagement tracking (was fake tensor).
2. **Param cap** — raised from 32KB to 200KB (initial substrate is
   107KB).
3. **Edge buffer** — raised `_EDGE_FAN_FACTOR` from 16 to 64
   (h_init>35 crashed).
4. **Dream replay exclusion** — `dream_cycle` now accepts
   `current_classes` list (was single `current_class`, excluded only
   1 of 2 classes per task → destroyed just-learned task by ~95%).

### Replay architecture (2 structural changes)
1. **Interleaved replay** — manifold pseudo-samples from past tasks
   mixed into training batches alongside real current-task data.
2. **Astrocyte-gated replay** — `interleaved_replay_batch` evaluates
   per-class manifold log-likelihood on current input; only top-K
   most at-risk classes fire protective replay per batch. Cost is
   O(K) regardless of total class count.
3. **Dream cycle replays all** — post-task dream includes current
   classes too (prevents the 95% destruction).

### Lifecycle modules (8, all were stubs)
- `saliency.py`, `compact.py`, `ship.py`, `wake.py`, `extend.py`,
  `graft.py` + `bases/frozen.py`, `bases/compose.py`
- Full pipeline verified: ship→wake bit-exact, graft with perception
  remapping, compose(seeded, frozen) works.

### GA hyperparameter optimization
- `experiments/evolve_hyperparams.py` — 29-gene genome, fitness =
  mean full-softmax accuracy.
- Best result (gen 1, pre-gated-replay): 0.4012 full with h_init=19,
  lr=6.7e-3, consecutive_tasks=5.
- GA v3 (with interleaved replay, pre-gated) still running in
  background. GA with gated replay not yet run.

### Spec §5.7 — Developmental Program
Inserted 347 lines into `paper/v3/spec.md` covering:
- Stem cells (undifferentiated progenitors)
- Adaptive morphogen field (learnable 16-param position→phenotype)
- Positional differentiation (Phase B: CONV near input, ATTENTION deep)
- Axon guidance (proximity-based sparse wiring)
- Lineage-based weight sharing for CONV (explains optical illusions)
- Lateral signaling (differentiation coordination, shortest-path
  wiring, density regulation)
- Signal-driven redifferentiation (Phase A: plant-like auxin override)
- Developmental base (`bases.developmental`)

## State of the build

- **Branch:** `v2.0-scaffold` (up to date with remote)
- **Commits this session (6):**
  - `f6cad25` — fix: activation tracking, param cap, growth print
  - `bcec436` — handoff: session 003
  - `1984a31` — lifecycle: ship/wake/extend/graft/compact/saliency +
    frozen/compose bases
  - `30b407c` — fix: interleaved replay + dream replays all classes
  - `03de08a` — feat: astrocyte-gated replay
  - `c79c529` — spec: §5.7 developmental program
- **Working tree:** clean (untracked: `runs/`, `experiments/evolve_hyperparams.py` already committed)
- **Background:** GA v3 may still be running (`runs/evolve_v3/`)

## Chained-15 accuracy history

| Config | Full | Task-Aware | Notes |
|---|---|---|---|
| Pre-fix (broken machinery) | 0.199 | 0.911 | No locking, no growth |
| Fix + default params | 0.065 | 0.690 | Locking too aggressive |
| GA best (all-past replay) | 0.401 | — | h=19, lr=6.7e-3, consec=5 |
| Astrocyte-gated (defaults) | 0.243 | 0.793 | Untouched params, smoke |
| v1 manifold-grown (n=10) | 0.601 | 0.961 | Target to match |

## Key findings

1. **Dream destroying current task** was root cause of poor full
   accuracy. Fixed by including current classes in replay + interleaving.
2. **Astrocyte-gated replay** bounds compute at O(K) per batch,
   solves the scalability concern for lifetime deployment.
3. **`seeded` base is a v1-compat shortcut** — the spec prescribes
   `minimal` or `developmental` for cell-first architecture.
4. **Weight sharing needed for CONV** — Rocky's insight that optical
   illusions arise from shared filters.
5. **Lateral signaling needed** — cells must know neighbor positions
   for efficient wiring and differentiation coordination.
6. **Adaptive morphogen** — static z-gradient is insufficient; field
   must learn from task signal.

## Next-up tasks

1. **Implement §5.7** — the developmental program. Priority order:
   a. `spawn_stem()` + `bases/developmental.py`
   b. Adaptive morphogen field
   c. Axon guidance
   d. Positional differentiation
   e. Lateral signaling
   f. CONV + ATTENTION phenotype implementations
   g. Lineage-based weight sharing
   h. Redifferentiation
2. **Run developmental base on chained-15** — compare vs seeded.
3. **GA on gated replay** — re-evolve with astrocyte-gated replay.

## Pointers

- **`paper/v3/spec.md` §5.7** — developmental program spec
- **`trioron/learning/dream.py`** — interleaved + gated replay
- **`trioron/learning/manifold.py:log_likelihood()`** — astrocyte
  gating signal
- **`experiments/evolve_hyperparams.py`** — GA optimization
- **`trioron/phenotype/`** — CONV, ATTENTION stubs need implementing
- **`trioron/bases/`** — `developmental.py` needs creating

## Environment notes

- Working directory: `/home/marcrockhat/trioron-project/`
- Branch: `v2.0-scaffold` (commit `c79c529`)
- Python: `/usr/bin/python3` (3.10.12) — `torch 2.11.0+cu130`
- Platform: Linux (WSL2)
