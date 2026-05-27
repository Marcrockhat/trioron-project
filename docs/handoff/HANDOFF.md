# Trioron Handoff

**Session date:** 2026-05-27
**Session number:** 007
**Session title:** Anti-forgetting ablation — cluster replay fix, task-detector prototype

> This file is **rewritten in full** every session. The previous
> session's handoff is preserved in git history
> (`git log docs/handoff/HANDOFF.md`). Read this file first when
> starting any new session; rewrite it before ending one. See
> `docs/handoff/README.md` for the convention.

## Summary

Systematic ablation of the session-006 anti-forgetting stack revealed
cluster replay as the sole source of the full-accuracy regression
(0.55→0.41). Fixed the sampling bug, then discovered the 0.704 GA
baseline was unreproducible — exact GA commit code gives 0.30 full at
15 tasks. The real baseline is ~0.55-0.58. Prototyped per-class
private cells and task-detector routing toward the 85% full target;
detectors route at ~0.50 full but need stronger training and
calibration to close the gap.

## Headline numbers (all seed=42, 2 epochs unless noted)

| Config | Full | Task-Aware | Notes |
|---|---|---|---|
| Baseline (all off) | 0.5544 | 0.9372 | GA hyperparams, no stack |
| Sparsity only | 0.5638 | 0.9482 | +0.009 full, harmless |
| **KIBRA only** | **0.5763** | **0.9511** | **Best single component** |
| Cluster replay (broken) | 0.4608 | 0.9429 | Centroid+delta dilution |
| Cluster replay (fixed) | 0.5429 | 0.9506 | Per-class sampling |
| KIBRA+sparsity | 0.5351 | 0.9521 | Antagonistic combined |
| Full stack (no routing) | 0.5118 | 0.9591 | Components fight |
| No dream replay (4ep) | 0.5618 | 0.9476 | Dream replay neutral |
| KIBRA only (4ep) | 0.5644 | 0.9614 | Epochs don't help full |
| GA commit 781b4a3 (4ep) | 0.3001 | 0.8065 | **0.704 was never real** |
| Task detectors v2 | ~0.50 | 0.9544 | WIP, not converged |

## What was done

### Cluster replay fix

`cluster_replay_batch()` in `trioron/learning/dream.py` was sampling
via cluster centroid+delta+sigma*eps, giving ~1 sample per class when
clusters had many members. Fixed to use clusters for **selection**
(which region is at risk) but sample directly from per-class
`mu+sigma*eps` astrocytes. Recovered from −0.094 to −0.012 vs
baseline.

### GA baseline falsification

Ran the exact code from commit `781b4a3` (the GA hyperparameter
commit claiming 0.704 full). Result: **0.3001 full / 0.8065
task-aware**. All MNIST tasks at 0.0000 full. The 0.704 was either a
mid-curriculum snapshot, a different seed, or a different evaluation
method. The current codebase at ~0.55 is genuinely the best achieved.

### Dream replay investigation

Added `--no-dream-replay` flag. Disabling dream replay entirely
gives 0.5618 full — identical to baseline. The session-006 "58× LR
fix" was a red herring in both directions: the broken LR made replay
ineffective (no harm, no help), and the fixed LR also does nothing
measurable. Dream replay is currently neutral.

### Ablation infrastructure

`bench_chained_15_v2.py` now has toggle flags: `--no-sparsity`,
`--no-kibra`, `--no-routing`, `--legacy-replay`, `--no-dream-replay`,
`--no-private-cells`. Use `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PYTHONUNBUFFERED=1` for solo runs; parallel runs cause 10×+ slowdown
from thread contention.

### Task-detector prototype (WIP)

Three iterations toward 85% full via per-class private cells:

1. **v1 — additive private cells** (100 fan-in, shared edges kept):
   0.5810 full. Private cell's 1-edge contribution drowned by 55
   drifting shared H→output edges.

2. **v2 — severed shared edges** (784 fan-in, private pathway only):
   0.22 full. Private cells immune to forgetting but uncalibrated
   against each other — each trained on only 2 classes.

3. **v3 — task-detector routing** (15 detectors, dedicated BCE
   training with pos/neg examples, argmax routing → task-aware
   logits): ~0.50 full. The dedicated detector training pass works
   (up from 0.10 with piggyback training) but routing accuracy
   degrades to ~50% by task 15.

## Key findings

1. **Components are individually harmless, collectively
   antagonistic.** KIBRA (+0.022) and sparsity (+0.009) help alone
   but combined they drop full by −0.019. Cluster replay hurt because
   of a sampling bug, not the concept.

2. **The forgetting is in the output projection**, not the H-cell
   features. Task-aware stays 0.93-0.96 throughout — the substrate
   CAN discriminate within each 2-class task. The drift is in
   cross-task logit calibration.

3. **Single-neuron task detectors work but are too weak.** A linear
   neuron with 784 inputs can detect "is this MNIST?" vs "is this
   Fashion?" but struggles with 15-way discrimination among similar
   tasks (e.g., EMNIST A-B vs EMNIST C-D).

4. **Thread contention is catastrophic for parallel PyTorch runs.**
   3 concurrent runs on 12 cores: 10-15× slowdown (not 3×). Always
   use `OMP_NUM_THREADS=4` when running multiple bench processes.

## Next-up tasks (toward 85% full)

1. **Multi-neuron task detectors** — give each task 4-8 detector
   cells instead of 1. This provides a nonlinear boundary for 15-way
   task discrimination. Cost: 15 × 8 × 784 = 94K edges (within 300K
   budget).

2. **Detector calibration** — after all tasks trained, run a
   calibration pass across ALL detectors simultaneously using replay
   from all tasks. This normalizes detector scores so they're
   comparable across tasks. Currently each detector is trained in
   isolation.

3. **Detector + shared output hybrid** — instead of pure routing
   (hard argmax), use soft detector scores to weight the task-aware
   logits: `final[c] = Σ_t gate(t) * logit[c] * I(c ∈ task_t)`.
   This is differentiable and less brittle than hard routing.

4. **Output-edge anchoring** — EWC-like quadratic penalty on the
   1650 H→output edges. After each task, anchor current output edge
   weights. During future training, penalize drift. Targets the root
   cause (output projection drift) directly.

5. **Consider if 85% is achievable with the current substrate
   topology.** A 2-layer fully-connected network with 55 shared
   H-cells may have a fundamental capacity limit for 30-class
   continual full-softmax. The v1 manifold replay committee approach
   (separate per-task snapshots) hit higher numbers.

## State of the build

- **Branch:** `v2.0-scaffold`
- **Key commits this session:**
  - `3a3cf1e` — fix: cluster replay per-class sampling + ablation
    flags + task-detector prototype
- **Working tree:** pre-existing uncommitted changes in
  `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
  `trioron/viz/export.py` (carried from session 005)
- **Untracked:** `runs/` (ablation logs — not committed)
- **PARAM_CAP_BYTES**: bumped to 300K in bench for private-cell
  experiments. Revert to 200K if private cells are abandoned.

## Decisions made

- **Cluster replay kept but fixed.** The concept (cluster-level
  selection) is sound; only the sampling was wrong. The fixed version
  is −0.012 vs baseline (within noise).

- **Dream replay left enabled but neutral.** Neither helps nor hurts.
  Not worth removing since it's specced in §4.3.

- **KIBRA is the best single addition.** +0.022 full, +0.014
  task-aware. Keep it on for future experiments.

- **Sparsity dropped from combined configs.** Individually +0.009
  but antagonistic with KIBRA. Not worth the complexity.

## Open questions

1. The 0.704 GA number in commit `781b4a3`'s message — was this
   measured at fewer tasks, a different seed, or a different eval
   method? Should the commit message be corrected?

2. Should the v2 substrate grow deeper (add rank-2 H-cells between
   current rank-1 and output) to provide more capacity for
   multi-task separation?

3. Is the 85% target achievable with pure routing (identify task →
   use task-aware logits), or does the substrate itself need to
   produce calibrated 30-class logits?

## Pointers

- **`experiments/bench_chained_15_v2.py`** — all ablation flags,
  `add_task_detectors()`, `train_detector()`, `evaluate_all_tasks()`
  with detector routing
- **`trioron/learning/dream.py`** — fixed `cluster_replay_batch()`
- **Ablation logs:** `runs/ablation_*.log` (not committed)
- **Spec §4.3** — dream cycle spec (replay stage)

## Environment notes

- Working directory: `/home/marcrockhat/trioron-project/`
- Branch: `v2.0-scaffold` (commit `3a3cf1e`)
- Python: `/usr/bin/python3` (3.10.12) — `torch 2.11.0+cu130`
- Platform: Linux (WSL2), 12 cores, 7.6 GB RAM
- **Critical:** use `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  PYTHONUNBUFFERED=1` prefix for bench runs. Parallel runs without
  thread limits cause catastrophic contention.
