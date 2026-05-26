# Trioron Handoff

**Session date:** 2026-05-26
**Session number:** 006
**Session title:** Anti-forgetting stack — cluster replay + KIBRA + soft sparsity

> This file is **rewritten in full** every session. The previous
> session's handoff is preserved in git history
> (`git log docs/handoff/HANDOFF.md`). Read this file first when
> starting any new session; rewrite it before ending one. See
> `docs/handoff/README.md` for the convention.

## Summary

Built and validated a three-layer anti-forgetting stack for the v2.0
substrate. Task-aware accuracy matches the session-005 baseline
(0.954 vs 0.955) while adding structural protection against
catastrophic forgetting. Also fixed a 58× replay learning rate bug
and a rejuvenation bug affecting astrocyte lifecycle.

## Headline numbers

| Config | Epochs | Full | Task-Aware | Notes |
|---|---|---|---|---|
| Session 005 baseline | 4 | 0.585 | 0.955 | Previous best |
| **Full stack + routing** | **3** | **0.406** | **0.954** | This session |
| Full stack + routing | 2 | 0.405 | 0.952 | Smoke test |
| Baseline (no stack, 2ep) | 2 | 0.307 | 0.861 | Control |

Full accuracy gap (0.406 vs 0.585) is routing quality, not
forgetting — tasks with near-zero full have 0.88+ task-aware.

## What was done

### Three-layer anti-forgetting stack

1. **Online taxonomic clustering** (`trioron/learning/manifold.py`)
   - `ManifoldCluster` class — centroid, member list, incremental update
   - Classes assigned to clusters during training via code-space proximity
   - `cluster_threshold=5.0`, `cluster_min_samples=50`
   - Discovers ~9-10 clusters for 30 classes (threshold needs tuning
     for broader grouping)

2. **KIBRA edge tagging** (`trioron/learning/dream.py`)
   - One-shot edge importance scoring at each dream cycle
   - `|weight × gradient|` on cluster centroid samples
   - Top-K edges tagged per cluster, K scales with cluster size
     (`KIBRA_EDGES_PER_MEMBER=8`)
   - `arena.edge_protected` tensor — permanent protection bit
   - `zero_dormant_grads` extended to zero tagged edges' gradients
   - ~1920 edges protected by task 14 (3.9% of total)

3. **Soft sparsity** (`experiments/bench_chained_15_v2.py`)
   - L1 penalty on H-cell activations (`SPARSITY_LAMBDA=0.01`)
   - Uses `live_activations` (non-detached) for backprop
   - Encourages sparse coding — reduces cross-task interference
   - Hard top-K masking was tested and rejected (kills learning)

4. **Manifold-gated routing at inference**
   - Per-sample task gates from code-astrocyte log-likelihood
   - `logit_c += log(gate_task(c))` (log-sum-exp composition per §5.3)
   - Lifts full accuracy from ~0.26 (unrouted) to ~0.41 (routed)

### Bug fixes

1. **Astrocyte rejuvenation** (`trioron/learning/rejuvenate.py`)
   — `find_rejuvenation_candidates` now filters by
   `forward_inclusion`, preventing manifold astrocytes from being
   rejuvenated back to ACTIVE and breaking the replay chain.

2. **Dream replay learning rate** (`bench_chained_15_v2.py`)
   — Bench was overriding GA-evolved `replay_lr=0.00389` with
   `LR*0.1=0.0000668` (58× too low). Now uses `DreamConfig()`
   defaults from commit `781b4a3`.

### Experiments run and rejected

- **Per-task output cell locking** — locks output cells to DORMANT
  after each task. REJECTED: removes the replay pathway for
  maintaining output cell calibration. Task-aware dropped to 0.73.
- **Half-logit regularization** — stores H-cell activation
  statistics, penalizes drift during replay. REJECTED: anchors
  H-cells to single-task snapshots, prevents multi-task adaptation.
  Task-aware dropped to 0.77.
- **Hard top-K sparsity** — zeros all but K interior cell
  activations. REJECTED: kills gradient flow (45/55 cells get zero
  gradient). Task-aware dropped to 0.74.

## State of the build

- **Branch:** `v2.0-scaffold`
- **Key commits this session:**
  - `1b016c9` — fix: exclude astrocytes from rejuvenation candidates
  - `752668c` — feat: cluster replay + KIBRA edge tagging + soft
    sparsity + routing
- **Working tree:** pre-existing uncommitted changes in
  `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
  `trioron/viz/export.py` (carried from session 005)
- **Untracked:** `runs/`

## Next-up tasks

1. **Multi-seed validation** — run n=3 or n=5 seeds on the full
   stack at 3 epochs to get σ-confident numbers. Single-seed 0.954
   task-aware needs cross-seed confirmation.

2. **Routing quality** — the full-accuracy gap (0.41 vs 0.59) is
   manifold gate quality. Two tasks (mnist_6_7, fashion_4_5) have
   near-zero full but 0.92+ task-aware = misrouting. Options:
   - Temperature scaling on the task gates
   - Cluster-level routing instead of per-class
   - Calibrated routing via the existing `CalibratedRouter`

3. **Cluster threshold tuning** — current threshold=5.0 yields
   ~10 clusters for 30 classes (~3 per cluster). Lower threshold
   would discover broader groups (digits/fashion/letters) with
   stronger shared protection per KIBRA tag.

4. **KIBRA budget scaling** — currently 8 edges per cluster member.
   May need GA-level sweep for optimal budget.

5. **Per-task output cells (revisit)** — the original hypothesis
   (shared outputs = forgetting bottleneck) was wrong for the
   current architecture, but may become relevant if routing
   quality improves enough to support per-task heads.

## Key architectural findings

1. **Forgetting is a storage organization problem**, not a
   regularization problem. Continuous constraints (HL reg) fight
   against adaptation. One-shot structural tagging (KIBRA) +
   natural sparsity work better.

2. **KIBRA analogy works**: short-lived importance tag at
   consolidation → permanent edge protection. 3.9% of edges
   protected is enough to lift task-aware by 4.6pp.

3. **Soft sparsity >> hard sparsity**: L1 penalty lets the network
   discover its own sparse patterns. Hard top-K destroys gradient
   flow and kills learning.

4. **Replay LR matters enormously**: 58× too low replay LR was
   silently undermining dream cycle effectiveness across all
   session-005 experiments.

5. **Output cells must stay active**: locking them removes the
   gradient pathway that replay uses to maintain calibration.
   Protection should target edges, not cell state.

## Pointers

- **`trioron/learning/manifold.py`** — `ManifoldCluster`,
  `get_interior_ids`, online clustering in `ManifoldArchive`
- **`trioron/learning/dream.py`** — `kibra_tag`,
  `cluster_replay_batch`
- **`trioron/core/arena.py:66`** — `edge_protected` tensor
- **`trioron/core/scheduler.py:217`** — KIBRA protection in
  `zero_dormant_grads`
- **`experiments/bench_chained_15_v2.py`** — full stack integration

## Environment notes

- Working directory: `/home/marcrockhat/trioron-project/`
- Branch: `v2.0-scaffold` (commit `752668c`)
- Python: `/usr/bin/python3` (3.10.12) — `torch 2.11.0+cu130`
- Platform: Linux (WSL2)
