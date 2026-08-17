# Trioron Handoff

**Session date:** 2026-08-17
**Session number:** 050
**Session title (updated at close):** **Accessibility session — Rocky's problem: pip users can't
write skill masters. Answer built: the ZERO-MASTER vocabulary (one TD leaf
per declared drive, reward = own drive delta) reaches 112±23 survival vs
master-built 148.5±12.9 (n=3, zero policy code, 3× DQN); band-masked
arbitration from four user-declared thresholds lifts it to 128±27 (paired
+16.2±4.0, seed 2 BEATS its master nest); structural dreaming on top is
flat (+3.6±3.9). Also: `trioron/api.py` public surface created (spec §9.1
row, previously never written), `trioron.pcll` was MISSING from the 0.3.0
wheel (fixed), **v0.3.1 PUBLISHED to PyPI and verified**, README/MANUAL/
QUICKSTART/BRIDGE/TRIORON_MANUAL updated, and **DEPLOYMENT SPEED REALIZED:
new `lifecycle/export.py` dense export — arena forward 485 µs → 50 µs
jit (= a 27 K-param DQN MLP at 1/5 the params), exact 2e-7.**

---

## READ THIS FIRST

1. **Rocky's framing (this session):** the organism was originally
   unsupervised ("blind"); hand-coded masters turned primitives into
   supervised imitation, which no pip user can reproduce. The accessible
   contract is **"declare your drives"** — `Drive(getter, direction,
   stress_threshold)` — and let consequence teach. This session tested
   whether that recovers the vocabulary. Answer: mostly (see numbers), and
   the threshold is load-bearing (arm 2), not decoration.
2. **All numbers n=3 seeds 0–2, 40-map protocol, tamed physics; baseline =
   s049 master-built TD nest per seed 163.2/143.2/139.2 (deterministic —
   reused, not rerun).**
3. **v0.3.1 IS PUBLISHED** (https://pypi.org/project/trioron/0.3.1/,
   Rocky's call; clean `pip install trioron==0.3.1` import-verified).
   PyPI 0.3.0 was broken for pip users (no `trioron.pcll`, no
   `trioron.api`).
4. s049 NEXT items (in-world dream gap, hybrid arbitration, accuracy-vs-
   samples curve, cap-43 pathology, router Σ diet, s048 gaming items) are
   untouched and still stand.

## WHAT RAN (all committed; logs in `outputs/drive_*_s050_seed{0,1,2}.log`)

Scripts (all `archive/experiments/world/`, thin drivers over
`world_dream_newleaf.py` machinery):

1. **`world_drive_vocab.py`** (`fe9f75a`) — (a) zero-master vocabulary.
   DRIVES = temperature (setpoint, −2|temp−0.5|; one leaf covers warm-seek
   + cool), thirst, energy, integrity. Each leaf = `train_drive_leaf`
   (TD over actions, reward = that drive's delta ONLY, no master, no band,
   300 eps, seed+100k). Solo evals, then `train_router_td_n` (consequence-
   taught trioron router, cold start), 40-map eval. Saves every
   `trainable_tensors()` tensor to `runs/drive_vocab/{drive}_seed{s}.pt`
   + `router_seed{s}.pt` (s049 ckpt rule; reload verified bit-exact 100.1).
   - **drive-only nest 100.1 / 97.3 / 138.6 = 112.0±23.1** vs master
     148.5±12.9; **paired −36.5±32.3** (seed 2 ties −0.6).
   - **Leaves are NOT the gap:** solo drive leaves 42–69 (temperature
     63–69) vs best solo master 52.4. Composition lift: masters 52→148
     (2.9×), drives 65→112 (1.7×). Nest cold-deaths 9–12/40 while the
     temperature leaf alone dies cold 2–3/40 → arbitration.
   - Integrity leaf gets least routing share every seed (sparse/terminal
     drive).
2. **`drive_common.py`** — loaders (`load_drive_leaves`, `load_drive_router`,
   `build_router_sub`) rebuilding the same topology/seed and copying
   tensors back.
3. **`world_drive_dream.py`** — arm 1: structural dreaming ITERATED on
   the drive nest (self_reflect → top cause → new drive leaf →
   cold-start N+1 router → eval), 2 rounds.
   - trajectories 100.1→105.8→108.0 / 97.3→94.7→97.7 / 138.6→135.0→141.0
     = **115.6±22.6, paired +3.6±3.9 — FLAT.** Diagnosed cause shrinks
     every round (thirst 13→7, cold 10→6) but deaths migrate; 6-way
     cold-start router bleeds share from good leaves (seed 0
     DRIVE_TEMPERATURE 980→290 routes, cold-deaths 21/40 in round 2 even
     with a SELF_COLD leaf). **Ordering rule: fix arbitration BEFORE
     dreaming** — s049's +10.4 came on an already well-arbitrated nest.
4. **`world_drive_band.py`** — arm 2: band-masked arbitration. Four
   declared thresholds (temperature always eligible; thirst<0.9;
   energy<0.85; integrity = predator within Chebyshev 2 or on FIRE/POISON
   — the one band that is a PERCEPT threshold, stated as such). Router
   logits masked to eligible leaves (all if none); mask applied in TD
   behaviour policy + bootstrap max; router retrained cold, same budget;
   leaves untouched.
   - **112.6 / 113.0 / 159.0 = 128.2±26.7; paired +16.2±4.0 over
     unmasked (positive every seed); vs master −20.3±36.2; seed 2 beats
     its master nest 159.0 vs 139.2.** Remaining wall = integrity
     (14, 11 /40 deaths seeds 0/1).
5. **API + packaging (`8e50baa`):**
   - **`trioron/api.py`** created — spec §9.1 "v2 public API" row had
     never been written; README/MANUAL referenced `trioron.api` 20+ times
     → ModuleNotFoundError for every pip user. Now re-exports
     `trioron.legacy.api.*` (all 23 documented names verified present),
     v2 substrate (`construct`, `seeded`, `minimal`, `frozen`, `compose`,
     `Envelope`, `default_dispatch_table`), Phasecyte (`PhasecyteLeaf/
     Nest`, `dream_distill`, `dreamed_predict`, `PCLLController`).
   - **`trioron.pcll` added to `pyproject` packages** — 0.3.0 wheel on
     PyPI has 0 pcll files (verified by downloading it). Version 0.3.0 →
     0.3.1 (pyproject + `__init__` fallback). Wheel built to `dist/`,
     import-tested from a clean `pip install --target`. pytest: 137 pass,
     the 4 known failures.
   - **README rewritten:** Install; "three ways in" with each path's FEED
     CONTRACT stated (dataset / gradient-by-consequence / label-tolerant
     stream); tested v2 substrate snippet; Phasecyte snippet; "what is
     NOT in the package yet: the embodied organism" with the drive-only
     numbers; v2 layout tree; test section → `pytest tests`; Status
     lines. Disclosure paragraph untouched (still says Opus 4.7).

6. **DENSE EXPORT — deployment speed realized (`d9e15dc`).** Rocky asked
   "how fast is our trioron's response vs DQN?" Measured (1 CPU thread,
   batch 1, world router/leaf 77→32 quad→6, 113 live cells / 2048
   capacity): live arena forward **485 µs** vs DQN QNet (27 K params)
   **48 µs eager / 30 jit** — 10× slower, but that is arena overhead
   (activation buffer over dormant capacity + per-edge gather/scatter),
   not arithmetic (5.5 K live params). Built
   **`trioron/lifecycle/export.py`**: `export_dense(sub)` folds the
   compiled plan bucket-by-bucket into a buffers-only `DenseExport`
   module (per-chunk matmuls over upstream activation blocks, K=1
   stages as 2-D matmul, α folded when 1.0, quad `z+z²` where K≥2,
   tanh) — **exact 2e-7**, jit-traceable: **eager 78 µs / jit 50 µs;
   nest tick (router+leaf) 105 µs.** LINEAR/DENDRITE/TANH only;
   receptor + ATTENTION/CONV/RECURRENT raise NotImplementedError.
   **Does NOT learn** (no grads/arena/λ) — Rocky asked; answer: learning
   stays in the arena checkpoint, re-export after each cycle, no
   export→arena path by design (drops lineage/epigenome/λ/capacity).
   Exported from `trioron.api` and `trioron.lifecycle`;
   `verify_export(sub, module, x)`. Spec rows added: §5.4 Export
   paragraph, §6.4 measured latency table, §9.6 row, §9.14 index,
   §9.11 test list. Tests `tests/test_v2/test_export.py` (6, incl.
   two-layer, tanh, trained+jit, receptor refusal). Full suite 143 pass
   + the 4 known.
7. **Docs pass for 0.3.1:** MANUAL.md new §12.1 (substrate / Phasecyte /
   dense export, feed contracts, gotchas); README deploy note; QUICKSTART
   install + pointer; docs/TRIORON_MANUAL.md §6 bullet; **`trioron.bridge`
   → `trioron.legacy.bridge` in MANUAL/BRIDGE** (the bridge lives only
   under legacy; `from trioron.bridge import ToolDispatcher` was broken —
   only docs referenced it). Disclosure paragraph still says Opus 4.7
   (untouched, Rocky's call).

## GOTCHAS

- 6 background processes at OMP_NUM_THREADS=2 on 12 CPUs ran fine (~40
  min arm 2, ~70 min arm 1 per seed).
- `world_drive_*` scripts import `experiments.world.*` via
  `sys.path.insert(parents[3])` like their siblings — run from anywhere.
- Arm-1 log commit message says "+3.6±5.6"; correct paired σ is ±3.9.

## NEXT (priority)

0. ~~Publish 0.3.1~~ DONE. Merge `conscience-core` → `main` + push (done
   at close if this line is struck through below).
1. **Arm 3 — integrity on threat-distance delta** (reward = Δ predator
   Chebyshev distance, not damage) + band mask; the last identified wall.
   If it lands, drive-only ≈ master-built and the contract is settled.
2. **(b) API shape doc** for the organism: `Body` protocol
   (`observe()`, `act(a)`, `drives: {name: (getter, direction,
   threshold)}`, `done`), `Organism.live(body, ticks)` = drive leaves →
   band-masked router → (then) structural dreaming; `Organism.watch(body,
   demos)` = phasecyte absorb (rung 2); `Stream.from_arrays` for
   classification users; `TileWorld` + Gym adapter as batteries. Spec §9
   partition row BEFORE code (house rule).
3. **(c) code + README** for the above; then s049 NEXT items resume.

## OPEN / unresolved

- 4 pre-existing test failures (test_learning TestCredit ×2,
  test_lifecycle ×2) — untouched.
- Whether the integrity band belongs in a pure-drive contract (it is a
  percept threshold). Arm 3 decides.
- s049 open items unchanged (N_QUANTA ablation, package split, s047 parked).

## State of the build / Pointers

- **Commits (branch `conscience-core`):** `fe9f75a` (drive vocab),
  `8e50baa` (api.py + packaging + README + arm scripts), `797918d` (arm
  logs), `21e34db` (handoff v1), `d9e15dc` (dense export + spec + docs),
  + this handoff; merged to `main` and pushed at close.
- Checkpoints (untracked, `runs/`): `runs/drive_vocab/`,
  `runs/drive_dream/`, `runs/drive_band/`.
- `dist/trioron-0.3.1-py3-none-any.whl` + sdist = what is on PyPI.
- Export latency bench was a scratch script (in this handoff's numbers);
  a committed `bench/bench_export_latency.py` is a cheap follow-up.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)

`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `trioron/legacy/outputs/`,
`notebooks/` checkpoints, output PNGs/uncommitted logs untracked.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python
  3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`), 12 CPUs.
- Drive arms: `OMP_NUM_THREADS=2 python3 archive/experiments/world/
  world_drive_{vocab,dream,band}.py --seed N` (vocab ~45 min, band ~40,
  dream ~70 per seed).
- Bench logs buffer — check mtime before assuming a hang.
