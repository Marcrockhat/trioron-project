# Trioron Handoff

**Session date:** 2026-05-28
**Session number:** 008
**Session title:** Dual-manifold H-space routing — full 0.55 → 0.69 (storage-free) / 0.76 (oracle)

> This file is **rewritten in full** every session. The previous
> session's handoff is preserved in git history
> (`git log docs/handoff/HANDOFF.md`). Read this file first when
> starting any new session; rewrite it before ending one. See
> `docs/handoff/README.md` for the convention.

## Summary

Reframed the full-accuracy problem. Task-aware sits at ~0.96 throughout
the chained-15 curriculum, so the substrate **can** discriminate within
each task — the loss is cross-task **routing**, not within-task
forgetting. The manifold archive was doing routing in 784-d raw pixel
space with a diagonal Gaussian, a poor model for images. Moving the
routing manifold into the 55-d **learned H-cell representation** (dual
manifold: perception-space archive kept for replay, new H-space archive
for routing) plus a **full-covariance Mahalanobis** likelihood lifted
full accuracy from the 0.5544 baseline to **0.6903 storage-free** /
**0.7597 oracle**. The headline class-level variant is a pure QDA
classifier in H-space that **bypasses the output projection entirely** —
the edges that catastrophically forget are not used for the final
prediction.

## Headline numbers (seed=42, smoke = 2 epochs)

| Config | Full | Task-Aware | Note |
|---|---|---|---|
| Baseline (all off) | 0.5544 | 0.9372 | session-007 best reference |
| KIBRA only | 0.5763 | 0.9511 | prior best single component |
| Output-edge anchor λ=100 + det + cal | 0.4274 | 0.9564 | EWC toxic — overwhelmed by EMNIST Fisher |
| Perception-space hard routing | 0.3956 | 0.9597 | pixel Gaussian is a bad task oracle |
| Multi-cell detectors + calibration | 0.4534 | 0.9519 | linear-on-pixels ≈47% routing |
| H-routing diagonal + refresh + replay | 0.6844 | 0.9595 | ≈71% routing |
| Full-cov **task-level** + refresh + replay | 0.7503 | 0.9613 | H picks task, logits pick class |
| **Full-cov class-level QDA (oracle refresh)** | **0.7597** | 0.9585 | ceiling; needs past data |
| **Full-cov class-level QDA (storage-free)** | **0.6903** | 0.9591 | **honest CL result, no past data** |

Effective routing = full / task-aware. Pixel-space ≈42%, diagonal H ≈71%,
full-cov H ≈79%.

## What was done

### Dual manifold (the unlock)
- Perception-space `archive` is unchanged and still drives replay
  (generates 784-d pixel samples).
- New H-space `h_archive` collects 55-d interior-cell activations per
  class during training and is used for routing only.
- Routing modes: `task` (H-manifold picks the task, output logits pick
  the class within it) and `class` (pure QDA over all 30 classes,
  output projection unused). Class-level wins slightly with full cov.

### Stale-statistics fix (refresh)
- H-cell activations drift as later tasks train, so stats collected
  during task *t* are stale by task 14. Stale H-routing collapses to
  ~0.19. `refresh_h_archive()` rebuilds per-class (μ, Σ) from a fresh
  pass through the **final** substrate.
- `--refresh-source real` = oracle (forwards real past data, ceiling).
- `--refresh-source manifold` = storage-free (samples synthetic inputs
  from the perception sketch, forwards them, collects H-stats). This is
  the legitimate continual-learning path.

### Full-covariance Mahalanobis (`trioron/learning/manifold.py`)
- `ManifoldAstrocyte` gained opt-in `full_cov`: batched outer-product
  accumulation, diagonal-shrinkage (s=0.1) + 1e-4·I regularization,
  cached Σ⁻¹ and log|Σ|, and `log_likelihood_full()`. Off by default —
  perception-space replay path is byte-for-byte unchanged. 6 manifold
  tests still pass.

### Negative results (implemented, off by default)
- **Output-edge anchoring** (Online EWC on the 1650 H→output edges,
  γ=0.9). At λ=100 it is toxic (0.4274): EMNIST Fisher (~6) dwarfs MNIST
  Fisher (~0.6), so the penalty is dominated by late tasks while γ-decay
  erodes early-task protection. Did not pursue lower λ — H-routing is the
  better lever.
- **Multi-cell detectors + joint calibration** (4 cells/task, mean-pool,
  BCE with contrastive negatives, post-hoc joint calibration pass).
  Calibration helps (+0.05) but linear-on-784-pixels can't do 15-way
  task ID; ≈0.45 full. Same failure mode as perception-space manifold.

## Key findings

1. **Full accuracy is a routing problem, not a forgetting problem.**
   Task-aware ~0.96 throughout ⇒ perfect routing ⇒ full ≈ 0.96. The
   entire gap is cross-task task identification.
2. **Routing space matters enormously.** Pixel-space ≈42% routing,
   H-space diagonal ≈71%, H-space full-cov ≈79%. Same data, same
   substrate — only the representation the manifold lives in changed.
3. **The QDA-in-H-space classifier bypasses output-edge forgetting.**
   Class-level routing never touches the output projection; the
   forgetting that the whole session-006/007 anti-forgetting stack
   fought is simply routed around.
4. **Refresh is mandatory.** Without it, H-stats go stale and routing
   collapses (~0.19). Storage-free refresh from the perception sketch
   recovers most of the oracle gain (0.69 vs 0.76).

## State of the build

- **Branch:** `v2.0-scaffold`
- **Commits this session (all pushed? see below):**
  - `62aa57e` — dual-manifold H-space routing (0.55→0.68)
  - `7e561e4` — full-covariance Mahalanobis (0.68→0.76)
  - `e9d0634` — storage-free refresh from perception sketch (0.69 honest)
- **Modified & committed:** `experiments/bench_chained_15_v2.py`,
  `trioron/learning/manifold.py`.
- **Pre-existing uncommitted (carried from session 005, NOT touched):**
  `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
  `trioron/viz/export.py`.
- **Untracked:** `runs/` (logs, not committed).
- New bench flags: `--h-routing`, `--refresh-h`,
  `--refresh-source {real,manifold}`, `--full-cov`,
  `--h-route-mode {task,class}`, `--no-task-replay`, `--anchor-lambda`,
  `--no-anchor`, `--det-cells`, `--no-calibrate`.

## Decisions made

- **H-space routing is the direction; anchoring/detectors are parked.**
  Both attack the wrong layer (output) or wrong space (pixels). Left in
  the code (off by default) for the paper's ablation table.
- **Class-level full-cov QDA is the headline mechanism.** It bypasses
  the output projection — the cleanest story and the best number.
- **Storage-free (source=manifold) is the result we stand behind;**
  real-data refresh is reported only as an oracle ceiling.
- **Fixed interior-cell set for H-routing.** `h_interior_ids` is
  captured once before growth; grown cells are excluded so the H-vector
  dimension is stable (otherwise mid-task division breaks the astrocyte
  dimension — that bug bit once and is fixed).

## Open questions

1. **Close the 0.07 storage-free↔oracle gap.** The perception sketch is
   itself a crude 784-d diagonal Gaussian; better synthetic samples
   (or sampling from the cluster structure) should lift 0.69 toward 0.76.
2. **Selective full-cov storage (Rocky's idea).** Full Σ is 55×55≈3K
   floats/class (~363 KB for 30) vs 6.6 KB diagonal. Store full cov
   only for classes/tasks with high misroute rate; diagonal elsewhere.
   Pareto-storage, not an accuracy gain — build only after confirming
   where misroutes concentrate.
3. **Legitimacy at scale.** All numbers are single-seed smoke (2 ep).
   Need multi-seed (n≥3) and full-epoch reruns before this goes in the
   paper. Watch the per-task variance — EMNIST pairs are the weak spot.
4. **Does the QDA-bypass generalize** to the 50-task / extension benches,
   or is 15 tasks where 55-d H-space stops separating cleanly?

## Next-up tasks (priority order)

1. **Multi-seed + full-epoch** rerun of the storage-free full-cov QDA
   config to get a defensible headline (`--seed {1,2,3}`, drop `--smoke`).
2. **Improve storage-free refresh quality** — sample from perception
   *cluster* structure or raise `samples_per_class`; target closing the
   gap to 0.76.
3. **Selective full-cov** — instrument per-task misroute rate during the
   refresh eval, upgrade only the worst tasks to full Σ, measure
   accuracy-vs-storage Pareto.
4. **Port H-routing to the 50-task bench** to test whether the advantage
   holds or 55-d H-space saturates.

## Pointers

- **`experiments/bench_chained_15_v2.py`** — `refresh_h_archive()`
  (real vs manifold), `evaluate_all_tasks()` H-routing block (task vs
  class, diagonal vs full-cov), `add_task_detectors()` /
  `calibrate_detectors()` (parked), `OutputAnchor` / `anchor_penalty()`
  (parked).
- **`trioron/learning/manifold.py`** — `ManifoldAstrocyte.full_cov`,
  `_ensure_precision()`, `log_likelihood_full()`; `ManifoldArchive`
  `full_cov` ctor arg.
- **Logs:** `/tmp/run_*.log` (this session's runs, not committed).
- **Spec §4.5** — manifold replay; the H-space routing extension is not
  yet specced — a spec gap to write up if this becomes the main result.

## Environment notes

- Working directory: `/home/marcrockhat/trioron-project/`
- Branch: `v2.0-scaffold` (HEAD `e9d0634`)
- Python: `/usr/bin/python3` (3.10.12), torch 2.11.0+cu130
- Platform: Linux (WSL2), 12 cores, 7.6 GB RAM
- **Threads:** single solo run → `OMP_NUM_THREADS=8`; two parallel runs
  → `OMP_NUM_THREADS=4` each. Three+ concurrent without limits =
  catastrophic contention (10-15× slowdown). Use `PYTHONUNBUFFERED=1`.
- Smoke run timings: no-replay ≈15s/task, +replay ≈45s/task,
  +full-cov ≈62s/task; refresh adds ~1-2 min.
