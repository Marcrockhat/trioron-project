# Trioron Handoff

**Session date:** 2026-08-15
**Session number:** 048
**Session title:** **Product pivot to GAMING — the survival-world arc revisited on
`conscience-core`. Nested organism (5 trioron leaves + trioron router) VALIDATED:
a consequence-taught trioron router DISCOVERS arbitration (163.2 survival, near
the hand-coded ceiling 167.6, above every master incl. a new "perfect" one at
131); flat distillation FAILS reproducibly (mirror-gate choke + linearity,
diagnosed); cold-start router width collapses by N=20 leaves; flat DQN control
at same budget: 37.9 (4× below nest).**

---

## READ THIS FIRST

1. **The pivot (Rocky, this session):** ArXiv push judged premature ("pushing a
   premature project"); focus moves to PRODUCT, and the chosen application is
   **gaming** — NPCs/organisms that visibly learn and absorb skills at runtime.
   The vehicle is the mini-Minecraft survival world
   (`archive/experiments/world/`, s015–s021 era), now re-validated on the
   current core. The s047 conscience/router line is not abandoned — the nest's
   parent IS that mechanism (see 3).
2. **Everything below is single-seed direction-finding** (40-world evals,
   ±~8 steps noise) unless marked otherwise. n≥3 before quoting anything.
3. **The architecture that works:** skills as separate trioron leaves +
   a trioron router choosing per tick ("nested triorons", Rocky's term).
   Distillation into one flat policy is the architecture that fails — shown
   three independent ways this session.

## WHAT RAN (chronological; all logs in `runs/`, scripts committed `9b664cf`)

1. **`revisit_smoke.py`** — skill absorption on current core. Wiring smoke
   PASSED (mirror cells intact after 2 months of core drift). Chain
   solo→fire→water: 52.4→54.3→56.1, cause-of-death shifts correct, no
   forgetting — complementary skills absorb. Chain +flee: **regression to 41.0**
   (fire-avoidance overwrites warming — antagonistic pair). A hand-built
   **perfect master** (all skills, deterministic, bar 131–135) made the student
   WORSE: 52.4→35.1, integrity deaths 1→17 — arbitrated behavior does not
   distill into a flat policy (`runs/revisit_smoke*.log`).
2. **`diagnose_imitation.py`** — WHY it can't learn fast, 6 supervised arms on
   perfect-master action prediction (`runs/diagnose_imitation.log`):
   - **mirror-gated channel (the DAgger path): 0.476 asymptote, slowest — THE
     choke**; below even the linear ceiling.
   - **linear substrate: 0.600 = linear probe 0.595** — arbiter has linearly
     inexpressible sign-flips (s017 confirmed).
   - **nonlinear full-credit trioron: 0.723 ≈ MLP 0.730 — substrate is FINE.**
   - No perception gap (percept ≥ privileged state).
   - Recipe: leaves train **nonlinear=True + full-credit**, integrate by routing.
3. **Nest re-validation** — `primitives.py` + `vocabulary.py` re-run on current
   core (`runs/{primitives,vocabulary}_recore.log`): 5 leaf donors rebuilt,
   Gaussian-routed organism **148.1** (June result reproduces; routing acc
   0.813 full-cov; HYDRATE leaf weak at 0.533 fidelity — nonlinear retrain is
   the known fix, not yet done).
4. **HONESTY AUDIT (Rocky pushed, was right):** the vocabulary nest is NOT
   100% trioron — router = Gaussian QDA fitted to a HAND-CODED `danger()`
   formula on a hand-picked percept slice. Led to:
5. **`router_trioron.py`** — a real trioron substrate (77→5, nonlinear,
   full-credit) as router (`runs/router_trioron.log`):
   - **(b) consequence-taught (Q over leaf choices, drive-delta reward only):
     163.2** — beats Gaussian 148.1 and imitation arm, ~4 below the hand-coded
     arbiter 167.6; train tail 178.9. **Agreement with the hand-coded danger
     formula only 0.389 — discovery OUT-DESIGNED the formula** (more FLEE, less
     EVADE). 100%-trioron deployed loop achieved.
   - (a) imitation-taught fallback: 152.2 (label acc 0.933).
6. **`router_width_sweep.py`** — how many leaves can cold-start TD discovery
   arbitrate at fixed budget (300 eps)? **5→163.2 (mass on real leaves 1.00),
   10→112.7 (0.56), 20→65.2 (0.40), 40→64.8 (0.10)** — exploration cost is the
   wall, NOT Q-capacity (`runs/router_width_sweep.log`). Fixes, in preference
   order: incremental enrollment (curriculum — how a game works anyway),
   manifold-recognition shortlist + TD choice (hybrid of the two validated
   routers), dreamed exploration.
7. **`dqn_baseline.py`** — the control we owed ourselves: flat 2×128 ReLU DQN,
   identical budget/reward/eval, **n=3: mean 37.9** (34.9/40.0/38.9) — below
   flat linear trioron solo (52.4), 4× below the nest. Dies of the
   fire-exploration trap. Claim earned: **sample-efficiency at game budgets**,
   not asymptote (untuned DQN; more episodes would close it)
   (`runs/dqn_baseline.log`).

8. **`watch_duel.py`** — the first product-shaped artifact: retrains + CHECKPOINTS
   both headline agents (`archive/runs/duel/{router_td,dqn}.pt`; retrained pair
   reproduces held-out means exactly — 163.2 / 34.9), runs both on the same
   unseen map, renders a side-by-side GIF with the router's live leaf choice
   displayed. Map 987654: **nest 144 steps, DQN dead at 13**
   (`runs/duel_map987654.gif`, `runs/watch_duel.log`). Re-render on any map in
   <1 min: `cd archive && python3 experiments/world/watch_duel.py --map-seed N`.
9. **Clarified for Rocky:** the game organisms contain NO sin-wave machinery —
   leaves are linear, router is quad-dendrite σ(z)=z+z² (`bases/seeded.py:33`);
   the phasor/lock-in line (s041–s045) stays parked and unimported here.

## DREAMING DESIGN FOR THE NEST (discussed, not built)

Per-level: leaf dreams = existing `dream_loop.py` (frustration→implicated
primitive→relabel own failures→consolidate w/ replay interleave; broke the
Breakout plateau); router dreams = Dyna-style offline replay oversampling
near-death states; structural dreams = engagement-conditional apoptosis of
unused leaves (validated +0.8σ/-33% std) + int8 archive demotion + NEW-leaf
enrollment triggered by the s047 shift detector (novelty alarm). Day/night in
`tile_world` is the intended wake/dream signal — player-legible learning.
Caveat: Pong dream loop replicated only 2/4 seeds; per-game config sensitivity.

## NEXT (priority)

1. **Wire `dream_loop.py` to the trioron-routed nest** (it currently rebuilds
   the Gaussian organism). Target: push 163.2 past the 167.6 arbiter ceiling.
2. **Incremental-enrollment experiment** — add leaves one at a time to the TD
   router (the width-collapse fix that matches game design).
3. **Retrain leaves nonlinear + full-credit** (HYDRATE 0.533 → expect big lift;
   then the nest number should rise).
4. **n≥3 on the headline arms** (nest 148/163, arbiter, width sweep) before any
   external quote.
5. **NAV leaf design** (Rocky asked): skills emit goals, one shared nav trioron
   walks — router→skill→nav three-level nest. New build.
6. Swap Gaussian `VocabularyRouter` internals for the promoted
   `learning/router.py` `ManifoldRouter` where recognition-routing is used.

## OPEN / unresolved

- 4 pre-existing test failures (test_learning TestCredit ×2, test_lifecycle ×2)
  — predate s047, untouched.
- s047 NEXT items (novelty-alarm probe on tasks 0–9 vs 10–14, conscience API
  bundle for Aidos, router n≥3) are PARKED behind the game arc, not dropped —
  the nest's enrollment trigger (dream design above) is the same mechanism.
- Watcher-command lesson: `pgrep -f X` inside a queued command matches its own
  command line — the DQN queue silently never fired (fixed by direct launch;
  a stale watcher survived ~2.6h and was killed). Don't chain background runs
  that way; launch on completion notification instead.
- `perfect_master` lives in `revisit_smoke.py` (bar 131–135; overheat/thirst
  tradeoffs at the margin remain).

## State of the build / Pointers

- **Scripts (committed `9b664cf`, branch `conscience-core`):**
  `archive/experiments/world/{revisit_smoke,diagnose_imitation,router_trioron,router_width_sweep,dqn_baseline}.py`
- **Logs (untracked per convention):** `runs/revisit_smoke*.log`,
  `runs/diagnose_imitation.log`, `runs/{primitives,vocabulary}_recore.log`,
  `runs/router_trioron.log`, `runs/router_width_sweep.log`,
  `runs/dqn_baseline.log`.
- **Leaf donors:** `runs/primitives/{WARM,FLEE,HYDRATE,FORAGE,EVADE}.pt`
  (rebuilt this session on current core; linear — see NEXT 3).
- s047 state (H-routing promoted, 0.76 era-bound → cite 0.7111, shift detector
  shipped) unchanged and still current — see git history of this file.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)

`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `trioron/legacy/outputs/`,
`notebooks/` checkpoints, output PNGs/logs untracked.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python
  3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`), `OMP_NUM_THREADS=8`
  (4 per process when two run in parallel).
- World experiments run from `archive/` cwd:
  `cd archive && python3 experiments/world/<script>.py` (they insert
  `/repo/archive` on sys.path so `experiments.world.*` resolves there).
- Bench/experiment logs buffer — 0 bytes until exit is normal; check `ps`
  (with a pattern that can't self-match) before assuming a hang.
