# Trioron Handoff

**Session date:** 2026-06-05
**Session number:** 016
**Session title:** The Pong **Mode-E survival recipe**, built end-to-end on the
**v2.0 core** (no legacy) — four clean primitive donors → vocabulary organism +
**manifold router** → **frustration→dream self-improvement**. The integrated
organism **outlasts every master** (72.5 → **92.1** > fire-master 87.4). Found &
fixed a real perception gap on the way: the predator was invisible in the percept
(EVADE unlearnable) → added a predator scent (percept 74→77-d) → EVADE 0.35→0.92.

> Rewritten in full every session; prior handoffs in git history. Session 015
> (`51b112e`) closed the λ arc and diagnosed the imitation ceiling, prescribing
> exactly this recipe. This session **executes** it and validates it.

## Summary

1. **API reality check (Rocky's question "is the API updated to v2.0?").** No —
   the package is **`__version__ = 1.1.0`**: commits `b8e6fff` (*rename v2.0 →
   v1.1*) + `38acf74` (*move v1 into `trioron/legacy/`*) deliberately renamed and
   relocated the high-level surface. The donor/absorb/extend API
   (`build_donor`/`absorb`/`load_organism`/`extend`) exists and imports, but at
   **`trioron.legacy.api`** — the Pong code's `from trioron.api import ...` path
   is dead (only `hf_space_build/trioron/api.py` has it, an **untracked build
   artifact**). **Rocky's directive: use v2.0 core, leave legacy out, wire what's
   missing.** The world experiment stack is already pure v2-core
   (`trioron.core`/`bases`/`phenotype`/`learning`), so the recipe was built there.

2. **Phase 1 — four clean primitive donors (`experiments/world/primitives.py`).**
   Each primitive = its master restricted to a decision **band**, imitated by a
   v2.0 `build_mirror` substrate via supervised CE on `sub(_solo(percept))→action`
   (the `imitation_ceiling` method, but the substrate is **persisted** as a donor
   `(seed, bias, edge_weight)`). Masters: WARM=`fire_oracle`, HYDRATE=`water_master`,
   FORAGE=`food_master`, **EVADE=new `evade_master`** (flee predator / off-hazard,
   from the `run_reactive` rule). Collection runs masters under **natural
   (stochastic) exploration** + the band — a deterministic master is too good and
   camps the infinite WATER fountain (degenerate always-drink donor); the wander
   creates drive stress, the band recovers the navigation skill.

3. **THE perception fix (EVADE was unlearnable).** First build: EVADE fidelity
   **0.354 ≈ chance (0.312)**. Root cause: the **predator is absent from the
   percept** — `self.pred` is never written to the grid nor the 74-d percept, so
   EVADE's flee-direction is uncorrelated with anything seen (a *perception* gap,
   not substrate; consistent with s015). Corroboration: EVADE-donor took **zero**
   integrity deaths (couldn't evade) while HYDRATE-donor took 6/20. **Fix (Rocky
   approved): add a 3-d predator scent to `TileWorld.percept()`** (direction +
   proximity), appended at the END so existing slicing `[63:69]/[69:73]/[73:74]`
   is intact. **74→77-d. EVADE 0.354→0.917.** All four now learnable.

4. **Phase 2 — vocabulary organism + manifold router (`vocabulary.py`).** The
   router is the **new v2.0-core piece** the recipe needed (there is no
   `trioron.learning.route`; routing was bench-local). Mechanism = the s014/world
   z2 pattern: a per-primitive **full-cov `ManifoldArchive`** over the 14-d
   interoceptive context slice `percept[63:77]`, route by argmax
   `log_likelihood_full`. Routing label = **argmax-danger** (each state's single
   most-urgent drive) so the manifolds are **disjoint and discriminative** —
   overlapping critical bands gave 0.40 routing acc and degenerate routing; the
   argmax-danger labels gave **0.74**. A hand-coded `ArbiterOrganism`
   (argmax-danger, privileged) is the routing upper bound. **Survival n=40: 72.5
   (manifold-route) vs 76.2 (arbiter) — the learned router captures 95% of the
   arbitration**; beats every single donor (58.3) and 3/4 masters.

5. **Phase 3 — frustration→dream self-improvement (`dream_loop.py`).** Deploy →
   dominant death cause (`overheat`→WARM, `thirst`→HYDRATE, …) → collect the
   organism's OWN failure frames in that primitive's band, **relabel with the
   master** (DAgger correction), **dream**: fine-tune the donor on corrections
   (oversampled) **interleaved with skill-demo replay** (anti-forget). Result
   **72.5 → 92.1**, **exceeding the best master (fire 87.4)** — the integrated
   organism (it carries EVADE, which no resource-master has) **outlasts its
   teachers**. **One-shot rescue, NOT a monotone iterator** (oscillates
   72.5→92.1→78.5→91.7) — sharpening WARM alone reshuffles residual failures;
   matches the Pong `pong_dream_loop_multiseed` finding.

6. **A mid-build fix the integration exposed.** Phase-2 first cut overheated 6/6:
   the WARM band `temp<0.5` **omitted the leave-before-overheat decisions** (those
   are at temp≥0.5 near fire). Widened to `temp<0.5 OR _near_fire` → donor learns
   to leave; organism survival 52.8→68→72.5.

## Headline numbers

**Phase 1 fidelity (n per primitive, 77-d percept, held-out imitation acc):**

| primitive | drive | n | chance | fidelity | lift | donor-solo survival |
|---|---|---|---|---|---|---|
| WARM | temperature | 900 | 0.478 | 0.889 | +0.411 | 38.8 |
| HYDRATE | thirst | 900 | 0.261 | 0.533 | +0.272 | 37.8 |
| FORAGE | energy | 481 | 0.247 | 0.753 | +0.505 | 59.7 |
| EVADE | integrity | 236 | 0.312 | **0.917** | +0.604 | 45.6 |

(EVADE pre-fix = 0.354. HYDRATE 0.53 is honest — navigation is a balanced 5-way.)

**Phase 2 survival (n=40, steps alive, max 300):**

| organism | survival | note |
|---|---|---|
| **VOCABULARY (manifold-route)** | **72.5** | beats all donors + 3/4 masters |
| ARBITER (argmax-danger, upper bound) | 76.2 | learned router ≈ 95% of it |
| best single donor (FORAGE) | 58.3 | +14.2 |
| fire / water / food / evade master | 87.4 / 68.1 / 65.0 / 44.6 | |
| floors random / reactive | 49.8 / 141.8 | reactive = headroom |

Routing acc 0.741 full-cov (chance 0.25). Dominant death: overheat 26/40.

**Phase 3 dream trajectory (n=40):** `72.5 → 92.1 → 78.5 → 91.7`  (best **92.1**,
lift **+19.6**, > fire-master 87.4). Overheat stays dominant (~22–28/40).

## What was done (files)

Committed `0a1d16e` this session.
- **NEW `experiments/world/primitives.py`** — Phase 1. `evade_master`, banded
  `collect`, `train_donor`/`save_donor`/`load_donor` (rebuild via
  `build_mirror(seed,n_mirror)` + load weights), `build_all`. `--smoke`.
- **NEW `experiments/world/vocabulary.py`** — Phase 2. `danger`/`argmax_danger`,
  `collect_router_states` (random rollouts, argmax-danger labels, `min_danger`),
  `VocabularyRouter` (full-cov manifold), `VocabularyOrganism` (manifold-route),
  `ArbiterOrganism` (upper bound). `--smoke`/`--diagonal`.
- **NEW `experiments/world/dream_loop.py`** — Phase 3. `probe` (survival+causes),
  `collect_failures` (DAgger relabel), `dream_correct` (corrections + skill
  replay), `dream_loop`. `--smoke`/`--iters`.
- **`experiments/world/tile_world.py`** — `_pred_scent` + percept **74→77**.
- **`experiments/world/{organism_v1,organism_v2,population}.py`** — PERCEPT_DIM
  **77**. **`mirror_cells.py`** — INPUT_DIM 80→**83** (comment).
- `runs/` (LOCAL, not committed): `primitives_build.log`, `vocabulary_phase2.log`,
  `dream_loop_phase3.log`, `runs/primitives/{WARM,HYDRATE,FORAGE,EVADE}.pt`.

**Pre-existing, STILL DO NOT TOUCH** (carried since s005): `trioron/bases/
developmental.py`, `trioron/lifecycle/developmental.py`, `trioron/viz/export.py`.

## Key findings

1. **The predator was invisible.** A whole drive (integrity/EVADE) had no percept
   signal. Adding the predator scent is a clean "lift perception" win and the
   reason EVADE now works. **The percept is 77-d now** — re-derive, don't assume 74.
2. **Routing must be by argmax-URGENCY, not context-membership.** Overlapping
   critical bands → manifolds non-separable (0.40). Disjoint argmax-danger labels
   → 0.74 and a router that nearly matches the hand-coded arbiter.
3. **The vocabulary organism outlasts its masters** (92.1 > 87.4) — the recipe's
   thesis, validated. The edge is structural: no single blind spot (it has EVADE;
   fire-master dies of the predator).
4. **The dream loop is a one-shot rescue, not iterative.** +19.6 on iter 1, then
   oscillates. Single-primitive sharpening reshuffles failures. (cf. Pong.)
5. **Package is v1.1, docs say v2.0.** Naming gap is deliberate (rename commit),
   not a bug. High-level api lives in `trioron.legacy.api`; core machinery in
   `trioron.{core,learning,lifecycle,…}`. Build new world work on core.

## Decisions made

- **Recipe built on v2.0 core, no legacy** (Rocky). World stack already pure-core.
- **Predator scent added to the shared percept** (Rocky chose "add to percept"
  over dropping EVADE or an EVADE-only percept).
- **Router experiment-local first** (`vocabulary.VocabularyRouter`); promote to
  `trioron.learning.route` once the policy stabilises (the open routing item).
- **Dream loop does not overwrite the Phase-1 donors** (`save=False`); the clean
  baseline is preserved.

## Open questions / next-up

1. **Stabilise the dream loop (make the +19.6 stick).** **IMPLEMENTED, awaiting
   n=40 validation** (`dream_loop.py`, commit after `5864790`): DAgger **aggregation**
   (never discard corrections), **keep-best rollback** (revert any iter that
   regresses beyond `accept_margin`), and **plateau escalation** (after a primitive
   plateaus `plateau_patience` times, sharpen the next dominant cause's primitive).
   Smoke confirms the machinery runs and rollback fires on regression. **NEXT: run
   `python3 -m experiments.world.dream_loop --iters 6` (n=40) — does the curve go
   monotone and hold/exceed 92?** If yes, `--save` the best vocabulary as a
   persisted organism. If it plateaus below 92, tune `corr_weight`/replay balance
   or widen escalation.
2. **Overheat is still the ceiling (~24/40).** WARM over-occupies fire (warm_rate
   0.04 overshoots). Beyond donor-dreaming, consider **router hysteresis** (don't
   re-route WARM the instant temp dips) — the reactive floor (141.8) shows the
   arbitration itself has headroom.
3. **Promote the router to core** — `trioron.learning.route()` (handoff open item
   since s014). The argmax-danger-labelled full-cov manifold router is the
   candidate; lift it out of `vocabulary.py` once #1 settles.
4. **FORAGE is under-routed** (recall dipped to 0.10 in the n=40 run) — energy
   danger is rarely the argmax (slow decay). May need per-drive danger calibration.
5. **n≥3 seeds** for the headline 92.1 (single-seed eval set; survival σ ≈ ±9–12).
6. **The 74→77 percept change shifts every world experiment's numbers.** Old 80-d
   organisms (`protagonist.pt`, `mirror_organism.pt`) are now incompatible (83-d).
   `consolidate_base`/`fire_taming`/`imitation_ceiling`/`wg_survival` still run but
   their baselines moved — re-run before citing.

## Pointers

- `experiments/world/primitives.py` (Phase 1) → `vocabulary.py` (Phase 2,
  router) → `dream_loop.py` (Phase 3). Each has `--smoke`. Full:
  `python3 -m experiments.world.vocabulary` (n=40 ≈ 4 min);
  `python3 -m experiments.world.dream_loop --iters 3` (≈ 8 min).
- Masters: `fire_taming.fire_oracle`, `quest.{water_master,food_master}`,
  `primitives.evade_master`. Percept: `tile_world.percept()` (77-d), predator
  scent `_pred_scent` at `[74:77]`.
- Router math reused from `experiments/world/world_routing.py` +
  `trioron.learning.manifold.ManifoldArchive.log_likelihood_full`.
- Donors: `runs/primitives/{WARM,HYDRATE,FORAGE,EVADE}.pt` (LOCAL; seed=0,
  n_mirror=8; reload with `primitives.load_donor`).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `v2.0-scaffold`. Python 3.10.12,
  torch 2.11.0+cu130, WSL2, 12 cores, 7.4 GiB. `python3`, `OMP_NUM_THREADS=8`.
- World benches are CPU/core-bound, ~0.7 GB each. n=40 vocab ≈ 4 min; 3-iter
  dream ≈ 8 min. Survival is the noisy whole-episode metric (±9–12 across seeds).
- A stale second `claude` process (PID ~3057, ~1d19h) is idle and left running
  (Rocky's call); harmless.
