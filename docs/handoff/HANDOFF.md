# Trioron Handoff

**Session date:** 2026-05-24
**Session number:** 003
**Session title:** chained-15 v2 bench — machinery verification

> This file is **rewritten in full** every session. The previous
> session's handoff is preserved in git history
> (`git log docs/handoff/HANDOFF.md`). Read this file first when
> starting any new session; rewrite it before ending one. See
> `docs/handoff/README.md` for the convention.

## Summary

Ran the chained-15 continual learning benchmark against the v2.0
substrate for the first time. Found and fixed three bugs that
prevented CL machinery from engaging, then verified all subsystems
work end-to-end.

## What was done

1. **First run (broken):** All imports worked, bench ran 15 tasks in
   ~4 min (smoke mode). Results: full=0.1994, task-aware=0.9112.
   But locked=0 on every task, edges flat at 26048, growth never
   fired despite `grown=4` being printed.

2. **Diagnosed three bugs:**
   - **Param cap too small:** `PARAM_CAP_BYTES=32000` but initial
     substrate is 107KB → `divide()` always returned None.
   - **Engagement tracking broken:** bench passed a fake `[1, capacity]`
     activation tensor with only perception cells filled → interior
     cells had zero engagement → locking condition never met.
   - **Print lie:** `grown=N_GROW_PER_TASK - 0` hardcoded, not actual.

3. **Fixed all three:**
   - Raised param cap to 200KB (50K params × 4 bytes, per spec §6).
   - Added `_last_activations` to Scheduler, exposed via
     `Substrate.last_activations` — bench now feeds real per-cell
     activations to `CreditTracker.update_engagement()`.
   - `train_one_task()` returns actual `growth_count`.

4. **Re-ran:** Growth fires (4 on task 0, edges 26048→27632).
   Locking fires (71 total across 15 tasks). Full=0.0653,
   task-aware=0.6902 — worse than pre-fix because locking is too
   aggressive (shared interior cells get frozen early). This is a
   **tuning problem**, not a plumbing problem.

5. **Verified all machinery** with standalone diagnostic:
   - construct, forward, activations, engagement, utility,
     frustration, divide, manifold archive, dream_cycle,
     consolidate — all fire correctly.

## State of the build

- **Branch:** `v2.0-scaffold` (up to date with remote)
- **Commit this session:**
  - `f6cad25` — fix: chained-15 bench — activation tracking, param
    cap, growth accounting
- **Working tree:** clean (untracked: `runs/`)
- **All machinery verified working**

## Decisions made

| Decision | Why |
|---|---|
| Store `_last_activations` on Scheduler (detached) | Cheapest way to expose real activations for credit tracking without changing the forward() return type. Detached to avoid holding the compute graph. |
| Param cap = 200KB | 50K params × 4 bytes aligns with spec §6 Phase 1 ceiling. Initial substrate is ~27K params, leaves room for growth. |

## Known issues (tuning, not bugs)

- **Locking too aggressive:** 71 cells locked in 15 tasks at
  `theta_e=0.3`, `consecutive_tasks=2`. Shared interior cells freeze
  early → stale activations hurt later tasks. Needs higher threshold
  or more consecutive tasks.
- **Growth only on task 0:** Frustration needs multiple 50-step stuck
  windows + 25 sustained steps. Hard to reach in 2 epochs. Full
  epochs (4) or lower thresholds would help.
- **Manifold replay quality unverified:** Replay fires but its
  effect on retention hasn't been isolated.

## Next-up tasks

Rocky indicated the next session will tackle "the real task" — ask
him what that is at session start. Possible directions:

1. **Tune CL hyperparameters** for the chained-15 bench (locking
   threshold, frustration window, growth budget).
2. **Implement remaining subsystems** per spec §9.14 (evolution,
   compact, graft, etc.).
3. **Something else** — Rocky's call.

## Pointers

- **`experiments/bench_chained_15_v2.py`** — the bench script
- **`trioron/core/scheduler.py:180`** — where `_last_activations`
  is captured
- **`trioron/core/construct.py:94`** — `Substrate.last_activations`
  property
- **`trioron/learning/credit.py:59`** — `consolidate()` lock logic
- **`trioron/learning/frustration.py`** — plateau detector config

## Environment notes

- Working directory: `/home/marcrockhat/trioron-project/`
- Branch: `v2.0-scaffold` (commit `f6cad25`)
- Python: `/usr/bin/python3` (3.10.12) — `torch 2.11.0+cu130`
- Platform: Linux (WSL2)
