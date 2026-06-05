# Trioron Handoff

**Session date:** 2026-06-05
**Session number:** 018
**Session title:** **Broke the overheat ceiling — split the dual-role WARM donor.**
S017 located the 26/40 overheat ceiling and called it routing-priority. The cheap
probe (lever b: steepen heat-danger so WARM engages earlier) gave only 26→18 and the
*privileged arbiter overheated WORSE* (37/40) — proving the real cause was the WARM
donor's **dual role** (seek-fire-when-cold AND flee-when-hot, default = seek).
**Fix: SPLIT it** — WARM seek-only + a dedicated **FLEE/COOL** donor. Result (n=40,
deployable): **overheat 26→4, survival 82→152 (+85%), the organism now outlasts its
best master 1.74× (151.7 vs 87.4).** A clean, deterministic, principled win that
dwarfs s016's one-shot dream-loop 92.1.

> Rewritten in full every session; prior handoffs in git history. Session 017
> (`85c5b07`) located the overheat ceiling and asked: attack the router. This
> session did the cheap router probe (lever b), found it falsifies the
> routing-priority framing, and split the primitive instead.

## Summary

1. **Cheap probe (lever b) — REAL but MODEST, and it falsifies "routing-priority."**
   Added a `HEAT_GAMMA` knob to `vocabulary.danger()` (gamma<1 = convex-early heat
   ramp → WARM wins argmax sooner) and swept it n=40 (`router_probe.py`). Deployable
   router: overheat 26→18 and survival 82→87 at gamma≈0.25–0.35 — a genuine paired
   reduction (same seed set), but **not a ceiling-break**, and over-steepening
   (gamma 0.15) collapses it (27/40). The tell: the **privileged ARBITER**
   (perfect state, engages WARM whenever heat is argmax) overheats *worse* — 37/40,
   and steepening makes it worse while survival craters (82→63). Earlier selection
   is not the fix.

2. **Diagnosis: the WARM donor's DUAL ROLE is the wall.** WARM had to seek-fire-when-
   cold AND flee-when-hot; its *default tendency is to seek*. Engaging/holding it near
   fire at moderate temp keeps it approaching → overheat (this is why s017's hand-
   coded hysteresis inverted: forcing WARM-commitment cooked 8/8; the flip-flop was
   protective because other drives yanked it away). One donor owning two opposite
   fire-behaviors caps the flee.

3. **Fix: SPLIT (Rocky's pick).** WARM → seek-only (cold regime); new **FLEE** donor
   (`cool_flee_master`, pure away-from-fire, collected under a `fire_oracle`
   *behavior* that soaks/overheats but *labeled* by the flee master — the
   behavior/labeler split, since a pure-flee master never gets hot enough to demo).
   `danger()` now routes **cold→WARM / hot→FLEE** (HEAT_GAMMA steepens FLEE). 5-class
   router.

4. **The split BREAKS the ceiling.** Born-quad WARM+FLEE @800 ep. Standalone donors
   are now decisive: WARM cooks 40/40 alone (seek-only, 82.8% fire-occupancy), FLEE
   freezes 40/40 alone (flee-only, 0.5%). Routed together (n=40, gamma=1.0): **survival
   151.7, overheat 4/40** (was 82.0 / 26). FLEE fidelity **0.973** > old dual WARM
   0.917. Arbiter overheat 37→5, confirming structural.

5. **The ceiling MOVED to the predator.** With overheat solved the organism lives
   151 steps (was 82), long enough that **integrity/predator (EVADE)** is now the
   dominant death (19/40). FORAGE router recall is also weak (0.18 — its manifold
   overlaps HYDRATE's "go-get-resource" context), but energy deaths are only 5/40 so
   misroutes are benign. Both are the next levers.

6. **Bonus: the split dissolved s017's nonlinearity requirement.** The `temp ×
   direction` sign-flip only existed because WARM was dual-role; each split donor is
   now a single-direction nav skill. Born-quad was kept for safety (proven regime)
   but **linear should suffice** — the cheaper untested option.

## Headline numbers

**Deployable vocabulary organism (n=40, manifold-route, full-cov):**

| config | survival | overheat /40 | dominant death |
|---|---|---|---|
| s017 dual-role WARM | 82.0 | **26** | overheat |
| **s018 SPLIT, gamma=1.0** | **151.7** | **4** | integrity (19) |
| s018 split, gamma=0.5 | 157.2 | 2 | integrity (but cold→12) |
| s018 split, gamma=0.25 | 151.6 | 2 | integrity (17) |

**Context (n=40, split @ gamma=1.0):** routing acc **0.813** full-cov (chance 0.20;
recall WARM .84 FLEE .85 HYDRATE .79 FORAGE **.18** EVADE .84). Arbiter (upper bound)
178.7. Floors: random 49.8, reactive 141.8. Best single donor 58.1 (FORAGE). **Best
master 87.4 (fire) → organism 151.7 = 1.74×.** Route usage WARM 15 / FLEE 9 /
HYDRATE 34 / FORAGE 16 / EVADE 26 %.

**Probe (lever b, dual-role, n=40):** deployable 26→18 @gamma 0.25; arbiter 37→29
(survival 82→63). Falsified as a ceiling-break.

## What was done (files)

Committed this session (see git log).
- **`experiments/world/vocabulary.py`** — `HEAT_GAMMA` knob; `danger()` split into
  WARM=cold / FLEE=heat; `PRIM_ORDER` → 5 classes; `route_hist` generalised to
  `len(PRIM_ORDER)`. Everything else iterates PRIM_ORDER so it scaled automatically.
- **`experiments/world/primitives.py`** — `cool_flee_master`; `_band_warm` now
  seek-only (temp<0.5 excl. leave-decisions); new `_band_flee`; `collect()` gains a
  `behavior_fn` (behavior/labeler split); `PRIMITIVES` adds FLEE (behavior=fire_oracle);
  `build_all` + `--only NAME...` to rebuild a subset.
- **NEW `experiments/world/router_probe.py`** — gamma sweep, arbiter + routed arms.
- **`runs/` (LOCAL, NOT committed):** `router_probe_gamma_sweep.log` (dual-role
  probe), `primitives_split_warm_flee_build.log`, `router_probe_split.log`,
  `vocabulary_split_n40.log`. **`runs/primitives/WARM.pt` + `FLEE.pt` are now the
  SPLIT born-quad donors**; old dual-role WARM backed up at
  `runs/primitives/WARM_dualrole_backup.pt`. HYDRATE/FORAGE/EVADE born-quad untouched.

**Pre-existing, STILL DO NOT TOUCH** (carried since s005, left uncommitted):
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`.

## Key findings

1. **A single donor that must do two OPPOSITE things caps the policy.** Splitting it
   into one-primitive-per-drive-DIRECTION (the Mode-E doctrine taken literally) beat
   every router-side fix. Overheat 26→4, survival +85%, 1.74× the best master.
2. **"Routing-priority" was the wrong framing** for the overheat ceiling. The
   privileged arbiter overheating *worse* than the learned router (37 vs 26) was the
   diagnostic: the bottleneck was donor structure, not selection timing.
3. **The behavior/labeler split** lets you demo a primitive whose own master never
   visits its band (pure-flee never overheats) — run a state-generating behavior,
   keep the master's label.
4. **Splitting can dissolve a nonlinearity requirement.** s017 proved WARM needed the
   quad gene; the split removed the sign-flip, so each donor is linearly representable.
5. **HEAT_GAMMA is a minor secondary lever** post-split (overheat 4→2) but trades cold
   deaths; gamma=1.0 is the clean deploy default.

## Decisions made

- **SPLIT WARM into seek-only + dedicated FLEE** — Rocky (chose it over router
  anticipation / bank-and-stop). The session's main result.
- **Deploy at gamma=1.0** (the split does the work; steepening adds cold deaths).
- **Born-quad WARM+FLEE @800 ep** kept for safety; linear flagged as cheaper-next.
- **runs/primitives/WARM.pt overwritten** with seek-only; dual-role preserved at
  `WARM_dualrole_backup.pt`.

## Open questions / next-up

1. **ATTACK THE PREDATOR (the new ceiling).** Integrity/EVADE is now the dominant
   death (19/40). The EVADE donor solo is weak (49.1) and evade-master only 44.6 —
   the master itself may be the wall (echoing the s015 imitation-ceiling story for a
   harder skill). Probe: is EVADE a donor problem, a router problem (EVADE recall is
   high 0.84, so probably donor/master), or does it need anticipation?
2. **FIX FORAGE ROUTING (recall 0.18).** Its manifold overlaps HYDRATE. Cheap probe:
   does a sharper context slice or per-class prior separate them? Low stakes (energy
   deaths 5/40) but it's a clean routing bug.
3. **Try LINEAR WARM/FLEE** — the split predicts linear suffices (no sign-flip). If it
   matches born-quad it's faster to build and a cleaner paper claim.
4. **Re-run the dream loop on the split organism** — does self-improvement now clear
   the predator ceiling, or stack on 152?
5. **Cross-PC rebuild** (`runs/` doesn't sync): rebuild ALL five donors:
   `python3 -m experiments.world.primitives --nonlinear --epochs 800 --collect-seeds 60
   --cap 1500 --no-eval` (now builds WARM, FLEE, HYDRATE, FORAGE, EVADE).
6. **Promote the router to core** (open since s014) — the argmax-danger full-cov
   manifold router into `trioron.learning.route`.

## Pointers

- Probe: `python3 -m experiments.world.router_probe --gammas 1.0 0.5 0.35 0.25`
  (arbiter + routed; `--arbiter-only` for the cheap arm).
- Vocabulary: `python3 -m experiments.world.vocabulary --eval-seeds 40` (5-class).
- Split donors: `primitives.py --only WARM FLEE --nonlinear --epochs 800
  --collect-seeds 60 --cap 1500`. FLEE master = `cool_flee_master`; bands
  `_band_warm`/`_band_flee`; behavior/labeler split in `collect(behavior_fn=...)`.
- Router danger: `vocabulary.danger()` (WARM=cold, FLEE=heat; `HEAT_GAMMA` steepens
  FLEE), `argmax_danger`, `VocabularyRouter` (full-cov manifold over the 14-d ctx slice).
- Temp physics: `fire_taming` sets `WARM_RATE=0.04`, `TEMP_LOW=0.02`, `TEMP_HIGH=0.99`;
  cooling 0.008 day / 0.015 night. Death causes: `fire_taming._cause`.
- Masters: `fire_oracle` (WARM/seek), `cool_flee_master` (FLEE), `water_master`,
  `food_master`, `evade_master`.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `v2.0-scaffold`. Python 3.10.12,
  torch 2.11.0+cu130, WSL2, 12 cores. `python3`, `OMP_NUM_THREADS=8`.
- Born-quad @800 ep × 2 donors ≈ 5 min; the full 5-donor rebuild ≈ 12–15 min.
- Survival eval is single-seed-set n=40; σ ≈ ±9–12, so the 82→152 jump is ~6–7σ and
  the death-cause counts are paired over identical seeds — robust without multi-seed.
- World benches CPU-bound. Uncapped born-quad growth deliberately exceeds the Phase-1
  50K-param contract (exploration only).
