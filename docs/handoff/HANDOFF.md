# Trioron Handoff

**Session date:** 2026-06-03 → 2026-06-04
**Session number:** 014
**Session title:** Consolidation done right (anchor the base) → fire-economy probe →
"use trioron's own machinery, not hand-rolls" → native pipeline (replay) → the z2
H-routing transfer test reproduces the chained-15 full-accuracy pump in the world

> Rewritten in full every session; prior handoffs in git history. Session 013
> (`a6fcab4`) is the mirror-cells / fire-taming / Adam's-quest arc. This session
> completes its open #2 (consolidation) and pivots to the unification: the world's
> forgetting IS chained-15's full-softmax forgetting, and the fix is the manifold
> machinery we already have. READ THIS, then the headline numbers.

## Summary

Long, productive session with a clear narrative arc and one strong new result:

1. **Consolidation done right** (Rocky's pick). Session 013 diagnosed the leak; I fixed
   it. The quest's lock froze fire's MIRROR cells but left the shared
   `perception→interior→output` base free, so water's TD drifted it and the "locked"
   fire skill eroded. **FULL-LOCK** anchors the base (freeze every old↔old edge + old
   biases at the boundary; water learns only on fresh mirror cells off a frozen rep).
2. **The measurement fix**: a deterministic **cold/thirsty-state battery** scored by
   policy-agreement — replaces ±15-noise occupancy. (Bug found & fixed: the masters'
   *explore* uses the GLOBAL torch RNG, so the battery wasn't deterministic until
   seeded — `torch.manual_seed` before `build_battery`.)
3. **Fire-economy probe** (Rocky's move-cost hypothesis): pyrophobia is largely a
   trip-cost artifact, BUT cheap fire backfires; only a cold-gated shaping reward
   lifts survival.
4. **The pivot** (Rocky: "use all the trioron advantages instead of re-inventing
   wheels"). I had been about to hand-roll soft-EWC. Stopped. Audited the native
   machinery, then wired it: **native_pipeline.py** (CreditTracker + ManifoldArchive
   replay + dream_cycle + frustration→growth).
5. **The unification** (Rocky): the world's forgetting is the *same* forgetting we beat
   on chained-15 (unanchored shared head drifts to last task = the epigenetic-lock read).
   The world is the *full-softmax* regime (no task selector). And the chained-15
   full-softmax **pump came from a manifold MODIFICATION** — dual-manifold **H-space
   routing** ("z2") + **full-covariance Mahalanobis** (0.55→0.69 storage-free / 0.76
   oracle), NOT replay alone.
6. **z2 transfer test** (the session's best result): the core full-cov manifold **routes
   fire-context vs water-context from the 32-d interior code**, reproducing the
   chained-15 pump near-exactly (diagonal 0.640 → full-cov **0.707**, +0.067 vs
   chained-15's +0.08). Routing transfers. **This is the validated next mechanism.**

## Headline numbers

**Consolidation — FULL-LOCK vs MIRROR-LOCK vs PLAIN (n=5, capacity-matched, only the
lock differs).** Retain = policy-agreement on a fixed cold battery (after-water vs
after-fire). `consolidate_base.py`, `runs/consolidate_base_n5.log`:

| arm | fire-retain | fire-comp f→w | water-acq |
|---|---|---|---|
| PLAIN | 0.145 ± 0.064 | 0.197→0.138 | 0.219→0.221 |
| MIRROR-LOCK | 0.162 ± 0.097 | 0.197→0.190 | 0.219→0.232 |
| **FULL-LOCK** | **0.398 ± 0.195** | 0.197→0.211 | 0.219→**0.365** |

Leak confirmed & located: MIRROR-LOCK≈PLAIN (freezing the readout alone buys nothing);
anchoring the base = 2.7× retention AND water acquires *better* (no interference).

**Fire economy — solo TD, no teacher (n=5).** `fire_economy.py`, `runs/fire_economy_n5.log`:

| arm | survival | cold-deaths | fire-occ% |
|---|---|---|---|
| baseline (4, plain) | 43.3 | 24.0 | 10.1 |
| dense (12, plain) | 38.5 | 15.6 | 18.6 |
| **shape (4, shaped)** | **47.4** | 19.4 | 12.7 |
| both (12, shaped) | 42.7 | 14.2 | 19.6 |

Move-economy confirmed (both levers cut cold-deaths, raise fire-use); but DENSE fire
*hurts* survival (trades cold-death for neglect); only the **cold-gated** shaping lifts
survival (+9%) — reward must point at a resource only while its drive is unmet.

**Native pipeline (replay-only) — n=5, four arms.** `native_pipeline.py`,
`runs/native_pipeline_n5.log`:

| arm | fire-retain | fire-comp f→w | water-acq |
|---|---|---|---|
| PLAIN | 0.179 ± 0.147 | 0.215→0.174 ▼ | 0.218→0.237 |
| FULL-LOCK | **0.391 ± 0.169** | 0.215→0.216 – | 0.218→0.350 |
| NATIVE-free | 0.251 ± **0.056** | 0.215→**0.304** ▲ | 0.218→**0.351** |
| NATIVE-gate | 0.203 ± 0.093 | 0.215→0.321 ▲ | 0.218→0.296 |

Mixed/honest: FULL-LOCK wins raw self-consistency, BUT (a) native arms *improve* fire
competence through water (replay reinforces; freeze only holds), (b) NATIVE-free ties
water and is **3× more robust** (±0.056), (c) **ungated > gated** (don't mirror-gate the
water master — it overwrites fire's mirror cells; answers Rocky's question). Self-
consistency is biased toward freezing; on competence, native wins. Replay-only ≈ draw vs
freeze — the missing lever is routing.

**z2 H-routing transfer (n=5, chance 0.50).** `world_routing.py`, `runs/world_routing.log`:

| router | accuracy | per-context recall |
|---|---|---|
| diagonal Gaussian | 0.640 ± 0.068 | fire 0.53 / water 0.74 (biased) |
| **full-cov Mahalanobis** | **0.707 ± 0.021** | fire 0.70 / water 0.71 (balanced) |

**Routing transfers** (0.71 ≫ 0.50, from the *base* organism's interior code, no skill
training). Full-cov **reproduces the chained-15 pump** (+0.067 ≈ +0.08), balances recall,
and cuts variance 3×. The interior code is a usable task-selector. **This is the cleared
next mechanism.**

## What was done (files)

New (all `experiments/world/`):
- **consolidate_base.py** — FULL-LOCK base-anchoring (`make_lock`, vectorized from
  edge_src/edge_dst to survive `compile()` reorders) + deterministic battery probe
  (`build_battery`, `policy_on`, `run_arm`). Commit `41d220a`.
- **fire_economy.py** — 2×2 move-cost probe (`fire_potential` cold-gated shaping,
  `train_solo_eco`). Commit `b7f221f`.
- **native_pipeline.py** — native CL machinery wired into fire→water: CreditTracker +
  ManifoldArchive (z2 mixture, whole-net pseudo-replay) + dream_cycle + frustration→
  divide. `imit_gated` flag (ungated default). Commit `11ff6a8`.
- **world_routing.py** — z2 H-routing transfer test (reuses the IN-CORE full-cov
  astrocyte; only the routing wrapper is new). Commit `11ff6a8`.

Modified (committed):
- **tile_world.py** — `FIRE_N` class attr + per-instance `fire_n` (defaults unchanged,
  `None`→canonical `max(2,s//3)`). Commit `b7f221f`.

**Pre-existing, STILL DO NOT TOUCH**: `trioron/bases/developmental.py`,
`trioron/lifecycle/developmental.py`, `trioron/viz/export.py` (carried session-005).

## Key findings

1. **The leak is the shared base policy.** Freezing a skill's readout is useless; anchor
   the base and it holds — and the next skill learns *better* (no TD interference).
2. **The world's forgetting = chained-15's full-softmax forgetting** (unanchored shared
   head drifts to the last task). The world has NO task selector, so it lives in the
   full-softmax regime (chained-15 peak ~0.60), not task-aware (0.96).
3. **The chained-15 pump was a MANIFOLD MODIFICATION, not replay**: dual-manifold
   **H-space routing** (the "z2") + **full-cov Mahalanobis** (0.55→0.69/0.76). full-cov
   is IN trioron core (`manifold.py`); the routing orchestration is bench-local only
   (`bench_chained_15_v2.py`), never promoted.
4. **Routing transfers to the world** (z2 test: 0.71 full-cov, +0.067 over diagonal).
   The stable interior code separates contexts; full-cov balances + stabilizes. This is
   the lever, validated.
5. **Replay defends weights; routing picks the skill.** Replay-only native ties the
   freeze; the win is **replay + routing**. Don't mirror-gate the water master (ungated
   retains + acquires better).
6. **Native machinery is more ROBUST** (3× tighter cross-seed variance) even where it
   doesn't beat the hand-roll on the mean — matches the full-cov variance reduction.
7. Battery determinism bug: masters' explore uses GLOBAL rng; seed before `build_battery`.

## Decisions made

- **FULL-LOCK** = hard base-anchor; the deterministic battery is the retention metric.
- **Pivot to native machinery** (Rocky): no hand-rolled EWC. Reuse `CreditTracker`,
  `ManifoldArchive`, `dream_cycle`, full-cov astrocyte. Promote routing to core next.
- **Don't mirror-gate** the water-master imitation (ungated default).
- **z2 = the H-space ROUTING manifold + full-cov** (NOT mixture_k=2 — that was my wrong
  first read). full-cov is the load-bearing modification.

## Open questions / next-up (priority order)

1. **Build replay + z2 router** (the cleared win): route the interior code to per-skill
   readouts (fire-mirror vs water-mirror), full-cov. The two levers together should beat
   the freeze on competence AND retention. This is THE next experiment.
2. **Promote routing to trioron core** — `trioron.learning.route()` (or
   `settle_via_manifold`) — so the dual-manifold H-routing stops being bench-local. The
   proper "wire the modification into trioron."
3. **n=10 confirm** of the fire-economy survival deltas (~1σ at n=5).
4. **Soft (EWC) anchoring** of FULL-LOCK — deferred; lower priority than routing.
5. The degenerate-seed problem (some seeds train a bad net across all arms) inflates
   variance; a competence-floor-before-consolidate would tighten means.

## Pointers

- `experiments/world/` — the whole arc. Reproduce: each bench has `--smoke`; full runs are
  `--seeds 5` (~30–100 min; native arms are heavy with per-batch replay).
- `runs/` (LOCAL, uncommitted): `consolidate_base_n5.log`, `fire_economy_n5.log`,
  `native_pipeline_n5.log`, `world_routing.log` — this session's results.
- **The chained-15 routing source to port from**: `experiments/bench_chained_15_v2.py`
  (H-routing logic lines ~267–338; `--full-cov`, `--perc-mixture-k`). Commits `62aa57e`
  (dual-manifold 0.55→0.68), `7e561e4` (full-cov 0.68→0.76), handoff `6d06993` (session 008).
- Core full-cov manifold: `trioron/learning/manifold.py` (`full_cov`, `log_likelihood_full`,
  `sample_full`). Routing has NO core home yet — that's open #2.
- `~/project-aidos/` — Numa/Mima origin (separate project, own memory dir).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `v2.0-scaffold`. Python 3.10.12,
  torch 2.11.0+cu130, WSL2, 12 cores, 7.4 GiB. `python3`, `OMP_NUM_THREADS=8`.
- Long runs: launch python DIRECTLY under the harness (not `nohup &`) for completion
  notification. DON'T wrap a foreground run in `timeout` while another heavy run shares
  the cores — it gets killed (happened to the first routing run).
- Battery/probe determinism requires `torch.manual_seed(...)` before `build_battery`
  (masters' explore uses the global RNG).
- The deterministic battery + (now) the z2 router are the low-noise instruments; do not
  read sub-±15 effects off whole-episode survival/occupancy.
