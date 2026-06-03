# Trioron Handoff

**Session date:** 2026-06-03
**Session number:** 014
**Session title:** Consolidation done right (anchor the base policy, not just the
mirror readout) → the leak located & plugged with a low-noise probe → fire-economy
probe (pyrophobia is a move-cost artifact, but cheap fire backfires)

> Rewritten in full every session; previous handoffs in git history
> (`git log docs/handoff/HANDOFF.md`). Session 013 (commit `a6fcab4`) is the
> mirror-cells / fire-taming / Adam's-quest arc this builds on. This session
> completes its open item #2 (consolidation done right) and #5/move-economy
> diagnosis. Read 013 if new — esp. the measurement-wall note.

## Summary

Two clean results, both born from the same realisation: **you cannot validate a
consolidation/retention mechanism on a metric with ±15 noise** (the wall session 013
hit). The fix in both cases was a deterministic, controlled probe.

1. **Consolidation done right (the session spine, Rocky's pick).** Session 013
   diagnosed the leak but left it unfixed: the quest's consolidate-and-lock froze
   fire's MIRROR cells but left the shared `perception→interior→output` base policy
   free. Every dagger chapter's TD loss reshapes that base (the `td_b/td_w` grads are
   added back AFTER the mirror-gated imitation grads), so learning water drifted the
   base and the "locked" fire skill eroded anyway. **FULL-LOCK** anchors the base: at
   the fire→water boundary it freezes every edge whose BOTH endpoints predate water +
   all old biases, leaving only the fresh water-mirror readout trainable. Water then
   learns as a pure additive readout off a FROZEN representation.
2. **The measurement fix.** Retention is now a FIXED cold-state battery scored by
   deterministic greedy policy-agreement (no episode noise) — what makes the leak
   falsifiable at small n. n=5 separated cleanly where 013's occupancy never could.
3. **Fire-economy probe (Rocky's deferred move-cost hypothesis).** Confirmed that
   pyrophobia is largely a trip-cost artifact AND surfaced a sharper finding: making
   fire cheap/abundant *backfires* on survival; only a cold-gated shaping reward helps.

## Headline numbers

**Consolidation — FULL-LOCK vs MIRROR-LOCK vs PLAIN (n=5, capacity-matched, only the
lock differs).** Retention = greedy policy-agreement on a fixed cold-state battery
(after-water vs after-fire policy). Water-acq = agreement-with-water-master on a fixed
thirsty battery, pre→post water chapter:

| arm | fire-retain | fire-comp f→w | water-acq pre→post |
|---|---|---|---|
| PLAIN | 0.145 ± 0.064 | 0.197→0.138 | 0.219→0.221 |
| MIRROR-LOCK | 0.162 ± 0.097 | 0.197→0.190 | 0.219→0.232 |
| **FULL-LOCK** | **0.398 ± 0.195** | 0.197→**0.211** | 0.219→**0.365** |

- **Leak confirmed & located:** MIRROR-LOCK (0.162) ≈ PLAIN (0.145) — freezing the
  readout alone buys almost nothing; the base drifts underneath. FULL-LOCK = **2.7×
  retention** (~2.7σ over PLAIN). The leak is the unfrozen `interior→output` base.
- **Competence survives only under FULL-LOCK** (PLAIN erodes 0.197→0.138).
- **Anchoring IMPROVED plasticity, didn't cost it:** only FULL-LOCK actually learns
  water (Δ+0.146; all 5 seeds positive). Frozen representation + fresh additive
  readout = a crisp gradient with no interference. **Two skills coexist, and the
  second learns *better* because the first is anchored.** The "wise organism" result.
- Caveats: high seed variance (FULL spans .066–.602; seed2 is a degenerate net across
  ALL arms). Absolute competence stays low (~0.2 — this world is genuinely hard to
  learn). Retention (self-consistency) is the load-bearing signal, not absolute skill.

**Fire economy — solo TD scratch learner, no teacher (n=5, tamed-fire physics):**

| arm | survival | cold-deaths/40 | fire-occ% |
|---|---|---|---|
| baseline (4, plain) | 43.3 | 24.0 | 10.1 |
| dense (12, plain) | 38.5 | **15.6** | **18.6** |
| shape (4, shaped) | **47.4** | 19.4 | 12.7 |
| both (12, shaped) | 42.7 | **14.2** | 19.6 |

- **Move-economy confirmed:** both levers cut cold-deaths (−35%/−41%) and raise
  fire-use (+84% density). Pyrophobia is largely a trip-cost artifact, not a ceiling.
- **But more fire-use ≠ more survival:** DENSE fire maximizes fire-use yet HURTS
  survival (38.5 < 43.3) — cold-death is traded for death-by-neglect elsewhere.
- **Only the cold-gated shaping lifts survival** (47.4, +9%): its `temp<0.5` gate is a
  satiety governor (pull when freezing, release when warm). Raw availability has no
  governor → over-use. Reward must point at a resource only while its drive is unmet.
- Caveat: cold-death/occupancy effects are large; survival deltas (~1σ, n=5) are
  suggestive, not ironclad — n=10 would bank the survival claim.

## What was done (files)

New:
- **experiments/world/consolidate_base.py** — the consolidation experiment. Key parts:
  `make_lock(sub, mode, …)` → per-batch grad-zeroing closure for `none`/`mirror`/`full`
  (vectorized from `edge_src/edge_dst`, NOT positional `edge_protected` — that can
  misalign across `compile()`, a latent bug in 013's quest lock); `dagger(...)` DAgger
  loop taking a `lock` callback; `build_battery(master_fn, which=…)` fixed cold/thirsty
  decision-state battery; `policy_on`, `run_arm`, 3-arm `main`. `--quick` smoke.
- **experiments/world/fire_economy.py** — the 2×2 move-cost probe. `fire_potential(w)`
  cold-gated potential; `train_solo_eco(..., fire_n, shaping)` solo TD with optional
  potential-based shaping; `evaluate_eco(..., fire_n)`; 4-arm `main`. `--quick` smoke.

Modified:
- **experiments/world/tile_world.py** — added `FIRE_N` class attr (default `None`) +
  per-instance `fire_n` __init__ param; `reset()` uses it. **Defaults unchanged**
  (`None` → canonical `max(2, s//3)`). Same approved pattern as 013's WARM_RATE override.

Commits this session: `41d220a` (consolidation), `b7f221f` (fire economy).

**Pre-existing, STILL DO NOT TOUCH** (carried session-005 uncommitted files):
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`. Left modified-unstaged, as in 013.

## Key findings

1. **The leak was the shared base policy, exactly as 013 guessed.** Freezing a skill's
   readout cells is insufficient because the output cell sums readout + base, and TD
   keeps moving the base. Anchor the base (old↔old edges + old biases) and the skill
   holds.
2. **Anchoring is not a plasticity tax — it's a plasticity *aid*.** A fresh readout off
   a frozen representation learns the next skill more cleanly than a net fighting its
   own TD interference. This is the modularity bet (frozen substrate + per-skill
   readout) paying off, consistent with the absorption/branch line in memory.
3. **Controlled probes beat episode metrics for mechanism validation.** A fixed
   decision-state battery + deterministic policy-agreement removed the episode noise
   that blocked 013. This is the general unlock for anything quantitative in the world.
4. **Pyrophobia is substantially a move-cost artifact** (cheaper/closer fire → more
   fire-use, fewer cold-deaths). But the naive fix (more fire) trades one death cause
   for another and lowers survival.
5. **Reward-design lesson:** in a multi-drive agent, the cure for a neglected drive is
   a signal that points to its resource *only while the drive is unmet* (cold-gated
   shaping), NOT cheaper access. Access without a satiety governor → over-commitment.

## Decisions made

- **FULL-LOCK** (anchor the whole old↔old base + old biases) is the consolidation
  mechanism; it is a HARD freeze. The graceful SOFT (EWC-style elastic) version was
  considered and deferred — FULL-LOCK already keeps water-acq rising, so soft anchoring
  is a refinement, not a rescue.
- **Deterministic battery probe** is the retention metric going forward (occupancy
  stays as a noisy deployment cross-check only).
- Freeze grads from `edge_src/edge_dst` each batch (position-independent), NOT from
  positional `edge_protected` — robust to `compile()` reordering.
- `tile_world` fire count parameterized, **defaults unchanged**.

## Open questions / next-up

1. **Soft-anchoring (EWC) refinement** of FULL-LOCK: elastic L2 anchor on the base so
   water can share it where it doesn't hurt fire — likely lifts BOTH retention and
   water-acq further. The "wise" polish; FULL-LOCK is the hard-freeze bound to beat.
2. **n=10 confirm of the fire-economy survival deltas** (currently ~1σ): bank the
   "dense hurts, cold-gated shaping helps survival" claim, or fold it into a broader
   reward-shaping study (gate ALL drives, not just temp).
3. **Chain a 3rd skill** under FULL-LOCK (fire→water→food) to test whether the
   modular freeze-and-grow scales past two skills, or whether the frozen base
   eventually starves the representation new skills need.
4. **Wire Numa/Mima** into the consolidation loop (validate a skill is real → lock it →
   grow → learn next). Still never used in the pipeline; FULL-LOCK currently locks
   unconditionally at the chapter boundary.
5. **The degenerate-seed problem** (seed2 trains a bad net across all arms): training
   fragility, not measurement noise. A reseed-on-collapse or a competence floor before
   consolidating would tighten the means.
6. **RPG-NPC pilot** (product direction, carried from 013): the consolidation result is
   a real ingredient — a companion that keeps old skills while learning new ones.

## Pointers

- **`experiments/world/`** — the whole organism arc.
- **`runs/`** (LOCAL, uncommitted — logs/.pt/.gif): `consolidate_base_n5.log`,
  `fire_economy_n5.log` (this session's results), plus 013's `protagonist.pt`,
  `mirror_organism.pt`, gifs. runs/ is NOT gitignored but has always been treated as
  local — headline numbers live in this handoff, not the logs.
- Reproduce: `python3 experiments/world/consolidate_base.py --seeds 5` (~30 min);
  `python3 experiments/world/fire_economy.py --seeds 5` (~25 min). Both have `--quick`.
- **`~/project-aidos/`** — Numa/Mima origin (separate project, own memory dir). Read
  before wiring Numa/Mima into the consolidation loop (open #4).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `v2.0-scaffold`. Python 3.10.12,
  torch 2.11.0+cu130, WSL2, 12 cores, 7.4 GiB. `python3`, `OMP_NUM_THREADS=8`.
- Long background runs: launch the python DIRECTLY under the harness (not via `nohup &`
  inside a wrapper) so you get a completion notification instead of polling.
- The deterministic battery probe is the template for any future quantitative claim in
  this world — don't try to read sub-±15 effects off whole-episode survival/occupancy.
- tile-world single-seed runs remain HIGH variance on survival/occupancy; cold-death
  *counts* and the battery *agreement* are the lower-noise channels.
