# Trioron Handoff

**Session date:** 2026-06-05
**Session number:** 017
**Session title:** **Located the overheat ceiling.** The vocabulary organism's
dominant death (overheat 26/40) the dream loop could never break is **routing-
priority, not donor quality** — proven by building the *best possible* WARM donor
and watching overheat stay *exactly* 26/40. The arc that got there falsified two
elegant wrong levers (hand-coded hysteresis; a temporal "hysteresis primitive")
and, via Rocky's **staged curriculum**, isolated the real donor wall:
**nonlinearity + saturation**. Born-quad primitives lift survival **72.5 → 82.0**.

> Rewritten in full every session; prior handoffs in git history. Session 016
> (`5864790`) built the Mode-E recipe and asked: stabilise the dream loop. This
> session answered that (it's a one-shot rescue, now made to *stick*), then chased
> the overheat ceiling to its true cause.

## Summary

1. **Dream-loop stabilisation validated (the s016 next-up #1).** Ran
   `dream_loop.py --iters 6` at n=40 (commit `21b061f`'s DAgger-aggregation +
   keep-best-rollback + escalation). Result: `72.5 → 92.1 → 78.5 → 68.3 → 83.4 →
   81.3 → 78.8`, **restored to 92.1**. The keep-best rollback **works** — the
   +19.6 now sticks *deterministically* (every regressing iter is reverted) instead
   of oscillating down. But it is **NOT monotone**: iter 1 captures the whole lift,
   every later dream regresses. **DAgger aggregation actively hurts WARM** (462
   corr → 92.1; 904 → 78.5; 1346 → 68.3 — more accumulated WARM correction
   overshoots). One-shot rescue confirmed, matching Pong. **Overheat stays
   dominant (~22–26/40) through every iter** → the ceiling is not the WARM donor.

2. **Hand-coded router hysteresis — DEAD END, and it inverts** (`hysteresis.py`).
   Hypothesis: hold WARM across the temp-valley so it doesn't thrash. Reality: the
   fire is a **trap** — `fire_taming` sets `WARM_RATE=0.04`/step heating vs
   `~0.008` (day)/`0.015` (night) cooling. Forcing WARM-commitment (danger-
   hysteresis) overheats **8/8**; the memoryless flip-flop was *protective*.
   Routing inertia is the wrong tool.

3. **"Teach hysteresis as a temporal primitive" (Rocky's paradigm) — sound in
   principle, NULL here** (`temporal_warm.py`). Built a recurrent WARM donor on the
   native Axis-7 satellite machinery (`SatelliteOp` leaky trace + BPTT, the
   `bench_temporal_gate` pattern), hysteretic master, ON vs OFF differing only in
   `op.hold`. n=40: **both arms cold-collapse** (OFF 38/40, ON 37/40 cold), gap
   +2.0 = noise. **Confounded:** the donor had to learn navigate-to-random-fire
   AND the latch at once; navigation dominated and failed, so the latch was never
   exercised. (Rocky's diagnostic Q: **yes, the world re-randomises every spawn** —
   `TileWorld.reset` re-scatters fire/water/food/predator; agent always starts
   dead-centre at temp 0.5; train/eval seeds disjoint. So "warm up" is scent-
   following to a *random* location that must generalise — the entanglement.)

4. **Staged curriculum (Rocky's redesign) isolated the real wall**
   (`warm_curriculum.py`). Decouple the skills:
   - **Stage 0 navigate** — linear donor reaches fire **98%** on held-out worlds
     (median 8 steps, peak temp 1.00) on only 0.53 action-fidelity (scent-following
     is error-forgiving). **Navigation is not the wall.**
   - **Stage 1 regulate** — the reactive **master holds the band (3/40 thermal**,
     survives 65, dies of thirst); the hysteretic master 6/40. **Memory is NOT
     needed for the control.** A **LINEAR** donor can't imitate it (34/40 thermal,
     fid 0.515); a **NONLINEAR (quad)** donor halves it (**15/40**, fid 0.590,
     surv 57.3) and dies of thirst like the master. **The wall is NONLINEARITY** —
     regulation is `toward-fire-when-cold / away-when-hot`, a `temp × fire-
     direction` *sign-flip interaction* a linear policy provably cannot represent
     (`selective_quad_growth` / `substrate_was_purely_linear`), not memory.

5. **Native selective-quad GROWTH works but is rate-limited by SATURATION**
   (Rocky's catch). Uncapped adaptive growth (`grow_budget=None`, 50 MB envelope,
   `build_mirror cap_bytes`) self-throttled at **76 cells / 73 quad** — but only
   reached **27/40** (worse than 15). Cause: growth fires ~every 25 steps but a
   **quad cell needs >400 epochs to saturate** (measured: linear saturates ep~50
   at loss 0.93; born-quad still falling at ep800, loss 0.115). Cells churn before
   they learn. **Decision (Rocky): suppress the growth triggers** → born-quad
   (native DENDRITE gene at birth, developmental differentiation) **+ train to
   saturation**. Thermal vs epochs: 100→**20**, 300→18, 500→13, **800→9** (→ the
   master's 3/40 floor).

6. **Payoff — born-quad primitives lift survival but DON'T touch overheat.**
   Rebuilt all four born-quad @ 800 ep (`primitives.py --nonlinear`). Vocabulary
   probe: **survival 72.5 → 82.0 (+9.5), overheat UNCHANGED at exactly 26/40.**
   The best WARM donor makes *zero* difference to overheat → **the ceiling is
   ROUTING-priority**: heat danger ramps linearly in temp while heating is
   0.04/step, so milder drives outrank it and the good WARM donor is never engaged
   until temp is near-lethal (flee-lag then guarantees overshoot). **Donor lever
   exhausted + successful; the router is the remaining lever.**

## Headline numbers

**Dream loop (n=40, iters 6):** `72.5→92.1→78.5→68.3→83.4→81.3→78.8`, best/restored
**92.1**. Keep-best rollback makes +19.6 *stick*; not monotone (one-shot).

**Stage 1 regulation (n=40 thermal deaths /40):**

| arm | fidelity | survival | thermal | note |
|---|---|---|---|---|
| reactive master | — | 65.1 | **3** | holds band; dies of thirst (ceiling) |
| LINEAR donor | 0.515 | 43.6 | 34 | can't imitate the sign-flip |
| NONLINEAR (quad) | 0.590 | 57.3 | **15** | dies of thirst like master |
| native grown (76c/73q) | 0.635 | 49.9 | 27 | growth outpaces saturation |

**Born-quad saturation (thermal /40 vs epochs):** 100→20, 200→18, 300→18, 500→13,
**800→9**. (300-ep "oracle" was undertrained; quad saturates slowly.)

**Born-quad primitive fidelity (@800 ep):** WARM 0.889→**0.917**, FORAGE 0.753→
**0.886**, EVADE 0.917→**0.973** (up); HYDRATE 0.533→**0.477** (down — balanced
nav, not relational, slow saturation cost it).

**Vocabulary (n=40):** linear **72.5 / overheat 26** → born-quad **82.0 / overheat
26**. Route usage WARM 42 / HYDRATE 22 / FORAGE 9 / EVADE 28 %.

## What was done (files)

Committed `fdafe37` this session.
- **NEW `experiments/world/hysteresis.py`** — router-inertia falsification
  (`HysteresisOrganism` / `DangerHysteresisOrganism`, margin×dwell sweep).
- **NEW `experiments/world/temporal_warm.py`** — recurrent WARM via `SatelliteOp`
  (Axis-7), `HystereticWarm` master, ON-vs-OFF BPTT. Null/confounded.
- **NEW `experiments/world/warm_curriculum.py`** — staged curriculum: `stage0`
  (navigate), `stage1` (linear vs nonlinear regulate), `train_donor_native` +
  `stage1_native` (uncapped selective quad growth), `eval_regulate`. `--stage N`.
- **`experiments/world/primitives.py`** — `nonlinear`/born-quad threaded through
  `train_donor` / `save_donor` / `load_donor` (back-compat: missing flag = linear)
  / `build_all` / `--nonlinear` CLI.
- **`experiments/world/mirror_cells.py`** — `build_mirror(cap_bytes=…)` for
  uncapped self-organising growth.
- **`runs/` (LOCAL, NOT committed):** `dream_loop_stabilized_iters6.log`,
  `temporal_warm_n3.log`, `warm_curriculum_stage{0,1,1_nl,2_native_uncapped}.log`,
  `primitives_bornquad_build.log`. **`runs/primitives/*.pt` are now BORN-QUAD**
  (overwrote linear); **linear backup at `runs/primitives_linear_backup/`**.

**Pre-existing, STILL DO NOT TOUCH** (carried since s005, left uncommitted):
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`.

## Key findings

1. **The overheat ceiling is ROUTING, not the donor.** The best WARM donor
   (born-quad, 9/40 thermal standalone) leaves vocabulary overheat at *exactly*
   26/40. Donor-sharpening (incl. the entire dream loop) cannot break it.
2. **The donor wall was NONLINEARITY + SATURATION, not memory.** Regulation is a
   `temp × direction` sign-flip → needs the quad (DENDRITE) gene; quad saturates
   slowly (>400 ep), so born-quad must be trained ~800 ep, not 300.
3. **Memory/anticipation belongs at the ROUTER, not the donor.** Reactive control
   holds the band (master 3/40); the donor never needed memory. The router *does*
   need to anticipate the temp rise to pre-engage WARM. (Rocky's instinct, right
   layer.)
4. **Selective quad growth needs saturation-gating.** Native growth works and
   self-throttles, but firing faster than cells saturate = churn. Either suppress
   it (born-quad) or reset the frustration detector per growth so each cohort
   saturates first.
5. **Born-quad is a principled +9.5 survival** (72.5→82.0) — a cleaner gain than
   the dream loop's one-shot 92.1, and stackable with a router fix.

## Decisions made

- **Born-quad primitives** (native DENDRITE gene at birth, **growth triggers
  suppressed**) + **train to saturation (~800 ep)** — Rocky. The primitive build.
- **Recipe built on v2.0 core** throughout; the recurrent path used the native
  satellite/Axis-7 machinery, not a bolt-on.
- **runs/primitives/ overwritten with born-quad**; linear set preserved at
  `runs/primitives_linear_backup/` for A/B.

## Open questions / next-up

1. **ATTACK THE ROUTER (the overheat lever).** Overheat 26/40 is routing-priority.
   Two candidates: (a) **anticipatory WARM engagement** — router senses temp
   rising fast (the temporal/memory instinct, at the router) and pre-engages WARM
   before heat-danger wins the argmax; (b) **steeper heat-danger ramp** in
   `vocabulary.danger()` (heuristic quick-check of whether earlier engagement alone
   breaks 26/40). Do (b) first as a cheap probe, then (a) natively.
2. **HYDRATE born-quad regressed** (0.533→0.477). Balanced 5-way nav isn't
   relational; quad's slow saturation hurt it at equal epochs. Consider per-
   primitive: born-quad only where relational (WARM/EVADE), linear where nav-like
   (HYDRATE/FORAGE) — or more epochs for HYDRATE.
3. **Stack it:** with the router fix, re-probe + re-run the dream loop on born-quad
   donors — does survival clear 92 *robustly* (vs the one-shot)?
4. **Cross-PC:** `runs/` doesn't sync. On another PC, rebuild born-quad donors:
   `python3 -m experiments.world.primitives --nonlinear --epochs 800
   --collect-seeds 60 --cap 1500 --no-eval` before any vocabulary/dream_loop run.
5. **Promote the router to core** (open since s014) — settle the argmax-danger
   full-cov manifold router (+ any anticipation) into `trioron.learning.route`.

## Pointers

- Curriculum: `warm_curriculum.py --stage {0,1,2}`. Saturation probe was inline
  (see `runs/` logs). Masters: reactive/hysteretic in `warm_curriculum` /
  `temporal_warm`; the registry WARM master is `fire_taming.fire_oracle`.
- Recurrent machinery: `experiments/satellites_v1.py` (`SatelliteOp`, `add_satellite`),
  `experiments/bench_temporal_gate.py` (BPTT). Quad gene: `trioron/phenotype/dendrite.py`
  (σ(z)=z+z²); seeded born-quad via `seeded(..., nonlinear=True)` → DENDRITE bit.
- Native growth: `experiments/selective_quad_growth.py` (the adaptive arm).
- Temp physics: `fire_taming` sets `WARM_RATE=0.04`, `TEMP_LOW=0.02`,
  `TEMP_HIGH=0.99`; cooling 0.008 day / 0.015 night (`tile_world.step`).
- Vocabulary/router: `vocabulary.py` (`build_vocabulary`, `danger`,
  `argmax_danger`); probe: `dream_loop.probe`.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `v2.0-scaffold`. Python 3.10.12,
  torch 2.11.0+cu130, WSL2, 12 cores, 7.4 GiB. `python3`, `OMP_NUM_THREADS=8`.
- World benches CPU-bound. Born-quad @800 ep × 4 primitives ≈ 10–15 min; quad is
  a SLOW learner — budget epochs accordingly. Survival σ ≈ ±9–12 (single-seed eval).
- Born-quad donors strictly dominate linear in capacity (σ(z)=z+z² contains z) but
  cost saturation time; uncapped growth deliberately exceeds the Phase-1 50K-param
  contract (exploration only — a shipped donor compacts back under the envelope).
