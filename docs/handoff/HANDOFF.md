# Trioron Handoff

**Session date:** 2026-05-30 → 2026-05-31
**Session number:** 009
**Session title:** Discriminative H-router lifts storage-free 0.72 → 0.832; substrate-is-bipartite-MLP discovered

> This file is **rewritten in full** every session. The previous
> session's handoff is preserved in git history
> (`git log docs/handoff/HANDOFF.md`). Read this file first when
> starting any new session; rewrite it before ending one. See
> `docs/handoff/README.md` for the convention.

## Summary

Built a discriminative MLP router over forwarded H-vectors that replaces
generative QDA's per-class (μ, Σ) summary with the full nonlinear sample
distribution. Storage-free chained-15 full accuracy lifts from QDA's
0.778 to **0.832 ± 0.009 (n=3 full-epoch)**. The oracle (real-data
refresh) reaches 0.886 — confirming the H-space has more discriminative
headroom than QDA was extracting.

Then probed five levers trying to push toward 0.85: **mixture-of-Gaussians
perception (A)**, **pixel jitter (B)**, **2× LR**, **train-time input
noise**, **H_INIT 55 → 80**. **All flat or regressed.** Investigation of
why H_INIT=80 didn't help exposed a load-bearing architectural finding:
**the substrate is structurally a 1-hidden-layer MLP**. `seeded()`
creates only bipartite perception→interior→output edges, and `divide()`'s
`(a.rank < child_rank)` policy prevents interior cells from ever
connecting to other interior cells, in seed *or* growth. Everything we've
built — Axes 1-5, KIBRA, manifold routing, anchor, dream — sits atop a
single-hidden-layer MLP, not a true graph network. The MLP class-router
beats QDA by exactly +6pp because the router *is* the missing second
hidden layer.

Hand-crafted `--interior-layers 2` smoke probe regressed (task-aware
0.96→0.89, under-training) but doesn't decide depth — the fair test is a
trioron-native one: relax `divide()`'s rank policy so the substrate can
self-organize depth via growth, per the design principle Rocky reasserted
("pure trioron self-adapting network"). Deferred to next session.

Three foundational design memories saved this session: logic-before-language
(reasoning is substrate-level symbolic CL, not LLM-mediated), temporal-
cognition-gap (working memory via dream-phase signal not per-event
timestamps), substrate-is-bipartite-mlp (the structural finding above),
and substrate-self-organizes-architecture (the path-forward principle).

## Headline numbers

Chained-15, storage-free, full-epoch (4ep), clean machinery (no anchor /
no private-cells / no calibrate), n=3 seeds 1/2/3:

| Config | full | σ | task-aware |
|---|---|---|---|
| QDA (class) | 0.7781 | 0.0041 | 0.9508 |
| MLP class-router | **0.8319** | **0.0085** | 0.9508 |
| MLP task-router | 0.8029 | 0.015 | 0.9508 |
| Oracle MLP class (smoke seed=42, real-data refresh) | 0.886 | (n=1) | 0.96 |

**vs v0.2.x published headline 0.601: +23pp.**
**vs hippo K=50 stored-exemplar baseline 0.637: +20pp** (storage-free).

## What was done

### Discriminative H-router (the lift)

- `collect_h_samples`: gathers (H-vector, class, task) triples from
  synthetic (storage-free) or real (oracle) inputs forwarded through the
  current substrate. Mirrors `refresh_h_archive`'s sampling so QDA and
  the router see identical data.
- `HRouter` (`nn.Module`): small MLP (or logistic) over standardized
  H-vectors. Configurable hidden width + epochs.
- `train_h_router`: standardize, Adam + CE, weight decay.
- `evaluate_router`: task granularity (router picks 15-way task, output
  head picks class within) and class granularity (router picks 30-way
  class directly, bypasses output head).
- `--h-router {qda,logistic,mlp}`, `--router-samples`, `--router-hidden`,
  `--router-epochs` flags. One probe run prints POST-REFRESH (QDA) +
  POST-ROUTER (task) + POST-ROUTER (class) on identical trained substrate
  — clean internal comparison.

### Regularization-probe knobs (the saturation story)

- `--perc-mixture-k K`: per-class K-component diagonal-Gaussian mixture
  in the perception archive via streaming online k-means
  (`StreamingMixture` class in manifold.py). Online k-means seeding with
  first K samples; cluster weights well-balanced empirically.
- `--perc-jitter σ`: Gaussian noise added to synthetic pixel samples
  before forwarding (downstream regularization).
- `--lr`: substrate optimizer LR override (default 6.68e-4).
- `--train-input-noise σ`: Gaussian noise on training x_batch only
  (eval/replay unaffected; upstream regularization).

### Substrate-capacity knobs

- `--h-init N`: interior cell count at substrate construction (default 55).
- `--interior-layers N`: number of stacked bipartite hidden layers
  (1 = current behaviour; 2+ adds compositional depth). Hand-crafted;
  exploratory only — the trioron-native path is `divide()`-policy work.

## Key findings

1. **Discriminative > generative for H-space routing.** MLP router beats
   QDA by +6.3pp smoke / +5pp full-epoch n=3. The router exploits
   nonlinear structure that QDA's (μ, Σ) summary discards. *Why it
   matters:* the per-class H-distributions are non-Gaussian enough that
   a discriminative model can extract significantly more signal.
2. **Class-level > task-level routing.** Class router (0.832) beats task
   router (0.803) by ~3pp full-epoch. Bypassing the forgetting-prone
   output head is the stronger play, consistent with session-008 QDA
   finding.
3. **All cheap regularization levers saturated at 0.83.**
   - Mixture-K=4: router REGRESSED -10pp (diagonal sub-cluster cov
     loses cross-pixel correlations rank-40 full-cov preserves).
   - Jitter σ ∈ {0.05, 0.10, 0.20}: router flat (0.831-0.835). Substrate
     absorbs post-hoc input noise before it reaches the router.
   - LR×2 (1.336e-3): router -0.010.
   - Train-time input noise σ=0.10: router flat, task-aware -0.009.
4. **H_INIT 55→80 also flat.** Bigger hidden layer = wider, not deeper.
   QDA regressed -4pp (curse of dimensionality on Σ); router flat;
   task-aware +0.4pp.
5. **The substrate is structurally a 1-hidden-layer MLP.** Loaded
   finding — see `substrate-is-bipartite-mlp` memory. `seeded()` lines
   59-67 create only bipartite layers; `divide()` lines 93-100 enforce
   `(a.rank < child_rank)` so grown cells can only see lower-rank
   sources (always perception for interior cells). Width grows via
   division but depth never does. The arena's `add_edges` accepts any
   src/dst pair; only the connectivity *policies* enforce the bipartite
   shape. *Why it matters:* The 0.83 storage-free ceiling has a clean
   structural explanation, and the path to 0.85+ is concrete — relax
   the rank policy, let the substrate self-organize depth.
6. **`recompute_ranks()` is Kahn's BFS topological sort.** Rank is
   *derived* from edge topology, not set arbitrarily. So adding a single
   inter-interior edge automatically promotes the receiving cell to
   rank+1 next compile — depth emerges from topology.
7. **Hand-crafted 2-layer regressed at smoke training budget.** Same
   2-epoch budget, 2× substrate params, task-aware crashed 0.96 → 0.89.
   Under-training, not a refutation of depth. The trioron-native test
   (`divide()` policy relaxation + growth) is the fair experiment;
   deferred to next session.

## State of the build

- **Branch:** `v2.0-scaffold`
- **Commits this session (all pushed):**
  - `53faea8` — feat: discriminative H-router + regularization probes —
    storage-free 0.72→0.83 (HRouter, collect_h_samples, train_h_router,
    evaluate_router; mixture/jitter/lr/train-noise knobs; StreamingMixture
    in manifold.py)
  - `74d0491` — feat: --interior-layers exploratory probe + --h-init flag
    (multi-layer Seeded; hand-crafted depth ceiling check)
- **Pre-existing uncommitted (carried from session 005, NOT touched):**
  `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
  `trioron/viz/export.py`.
- **Untracked:** `.claude/`, `runs/` (logs, gitignored).
- New bench flags: `--h-router {qda,logistic,mlp}`, `--router-samples`,
  `--router-hidden`, `--router-epochs`, `--perc-mixture-k`,
  `--perc-jitter`, `--lr`, `--train-input-noise`, `--h-init`,
  `--interior-layers`.

## Decisions made

- **Ship 0.832 ± 0.009 as the storage-free chained-15 headline** (n=3
  full-epoch, +23pp over v0.2.x). Discriminative router replaces QDA as
  the headline mechanism; QDA stays available as the baseline.
- **Bipartite-MLP finding is the architectural lever for the next push.**
  Modify `divide()`'s rank policy (`a.rank < child_rank` →
  `a.rank <= child_rank` with cycle check) so the substrate
  self-organizes depth via growth. Per Rocky: "pure trioron
  self-adapting network" — do NOT hand-craft.
- **`--interior-layers` flag is exploratory only**, not a deployment
  path. Useful as a hand-crafted ceiling check if needed.
- **The parked anchor/detector/calibrate machinery was hurting QDA by
  ~6pp.** Clean-config QDA full-epoch n=3 is 0.778 vs n=10 with
  machinery 0.718. The "POST-REFRESH (QDA)" line is depressed by the
  same calibration that clobbers FINAL.

## Open questions

1. **Does self-organized depth help?** Modify `divide()` rank policy
   (`<=` not `<`), full-epoch n=3, measure router-class. Decisive test
   of whether 0.85+ is reachable via the trioron-native path. ~3.5h
   compute.
2. **Or alternately: 2-layer hand-crafted at full-epoch n=3.** Same
   ceiling check at proper training budget. If even hand-crafted depth
   doesn't help, the bipartite ceiling isn't structural and the answer
   is elsewhere. ~3h compute.
3. **Mixture-K with per-sub-cluster full covariance.** Diagnosed but not
   tested — could rescue mixture but K × full_cov memory is ~290MB for
   K=4. Probably wrong direction; flagging for completeness.
4. **Same-seed determinism gap.** Same-seed reruns on CPU diverge mildly
   (seed-1 task-aware 0.952 → 0.940 across runs). Some of the n=3 σ is
   non-determinism, not seed variance. Worth noting in the paper.

## Next-up tasks (priority order)

1. **`divide()` rank-policy relaxation** — `trioron/lifecycle/grow.py`
   lines 93-100, change `(a.rank < child_rank)` to
   `(a.rank <= child_rank)`, rely on existing `_creates_forbidden_cycle`
   to filter cyclic edges. Run smoke seed=42; if depth emerges
   (count grown cells with rank > 1), full-epoch n=3. **This is the
   trioron-native path Rocky reasserted; the work continues here.**
2. **If (1) lifts router toward 0.85+, update headline to that number.**
   Otherwise document that depth alone wasn't sufficient and the
   bipartite ceiling is set by something else (training dynamics, loss
   structure, output-head forgetting).
3. **Pivot to symbolic-CL reasoning track** per Rocky's roadmap (see
   `logic-before-language-principle` memory). Curriculum order
   (2)→(4)→(1): compositional feature-binding → multi-step inference →
   abstract math. The (2)→(4) jump introduces temporal cognition; see
   `temporal-cognition-gap` for the wake/dream-phase framing.
4. **Update paper** (`paper/paper.tex` headline tables) to replace
   v0.2.x manifold-grown 0.601 with the new MLP-router 0.832 ± 0.009
   (n=3). The +23pp over the v0.2.x headline and +20pp over hippo K=50
   stored-exemplar baseline is the new paper story.

## Pointers

- **`experiments/bench_chained_15_v2.py`** — `HRouter`,
  `collect_h_samples`, `train_h_router`, `evaluate_router` (after
  `refresh_h_archive`, before `main`). `--h-router` probe wired into
  `main()` after the QDA POST-REFRESH eval. `build_substrate(h_init,
  interior_layers)`.
- **`trioron/learning/manifold.py`** — `StreamingMixture` class (after
  `ManifoldAstrocyte`); `ManifoldArchive.__init__` accepts `mixture_k`;
  `update_class` also updates the per-class mixture; `sample_mixture`.
- **`trioron/bases/seeded.py`** — `Seeded.interior_layers` for the
  multi-layer hand-crafted probe.
- **`trioron/lifecycle/grow.py:93-100`** — the `(a.rank < child_rank)`
  policy that's the next session's first edit.
- **`trioron/core/graph.py:80-170`** — `recompute_ranks` Kahn's BFS;
  understand this before changing the divide policy.
- **`trioron/core/graph.py:174-185`** — `_creates_forbidden_cycle`;
  relied on as the safety net when we relax the rank constraint.
- **Logs:** `/tmp/run_router_val_s*.log`, `/tmp/run_jitter_*.log`,
  `/tmp/run_mix_k4*.log`, `/tmp/run_lr2x_s42.log`,
  `/tmp/run_tin0p10_s42.log`, `/tmp/run_hinit80_s*.log`,
  `/tmp/run_layers2_s42.log` (this session, not committed).
- **Spec §4.5** — manifold replay; the discriminative H-router is not
  yet specced (spec gap). The bipartite-vs-cell-to-cell architectural
  finding is also a spec gap — spec implies cell-to-cell, code
  enforces bipartite. Worth a spec edit pass.

## Memories saved this session (cross-PC persistent)

- `logic-before-language-principle` — reasoning benches = substrate-level
  symbolic CL, not LLM-mediated; LLM is downstream "storyteller."
- `temporal-cognition-gap` — substrate has no working memory; (4)
  reasoning needs time; reuse wake/dream cycle as phase signal not
  per-event logging.
- `substrate-is-bipartite-mlp` — seeded() + divide() enforce strict rank
  hierarchy; structurally a 1-hidden-layer MLP, not cell-to-cell.
- `substrate-self-organizes-architecture` — design intent: substrate
  forms own depth via growth+training, not hand-crafted; "pure trioron
  self-adapting network."

## Environment notes

- Working directory: `/home/marcrockhat/trioron-project/`
- Branch: `v2.0-scaffold` (HEAD `74d0491` at session end pre-handoff)
- Python: `/usr/bin/python3` (3.10.12), torch 2.11.0+cu130
- Platform: Linux (WSL2), 12 cores, **7.4 GiB RAM**
- **OOM warning:** two concurrent full-epoch router-training runs each
  peak ~3 GiB at the router-train phase; the second push over 7.4 GiB
  triggers the OOM-killer. Stagger or run solo OMP=8 for full-epoch
  router probes. Smoke probes (~hundreds of MB) fit fine alongside.
- **Threading:** solo → `OMP_NUM_THREADS=8`; two parallel → OMP=4 each.
  Use `PYTHONUNBUFFERED=1`.
- Smoke run timings: full-cov + router-100ep ≈ 15-20 min total;
  refresh ≈ 1-2 min; router training ≈ 3-5 min.
- Full-epoch (4ep): ~50 min total per run with current C config.
- **Other project running:** `python3 -m src.main` from
  `/home/marcrockhat/resto-bot` (PID floats) — Rocky's other project,
  idle at 0% CPU, not interfering.
