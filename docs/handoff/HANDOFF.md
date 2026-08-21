# Trioron Handoff

**Session date:** 2026-08-21
**Session number:** 057
**Session title:** **KINOPSIS — one organism (Rocky: "focus on 2(a)"): s053 organ + motion-silhouette
leaf + velocity leaf + s056 evidence router folded into a single saveable object
(`Kinopsis`, née `MotionOrganism`) built by the continual schedule old world → moving world → velocity;
scored after every stage, n=3: router arm photo .653 / flat .870 / mixed .766, old
world .849 (forget +.019), velocity .655 acquired with ZERO effect on shape; replay-
settle of the organ heads is not worth it (+.004 photo for +.011 extra forgetting).**

---

## READ THIS FIRST

**NAME: the single organism is `Kinopsis`** (kine- motion + opsis sight; Rocky: "name it for
documentation and search"; intended later use = frozen perceptual AUGMENTATION organ for any
image / frame-packet classifier: shape log-prob + velocity logits + voter H-codes as features).
File `experiments/progenitor/kinopsis.py` (was motion_organism.py; `MotionOrganism` alias kept).

**s057 ORACLE-GAP SWEEP (Rocky: "let's do that" = NEXT-0) — log
`outputs/motion_organism_oracle_s057.log`, `ROUTER_ARMS=... python3 experiments/progenitor/kinopsis.py`
(stage-2-only sweep), n=3, photo / flat / mixed / old / forget / router params:**
router .653/.870/.766/.849/+.019/57 | +H evidence .656/.882/.770/.803/**+.065**/489 |
+per-class .654/.876/.771/.849/+.018/285 | +H+pc .659/.876/.771/.834/+.033/2445 |
+QDA H-voter .656/.872/.766/.846/+.021/84 | +QDA+pc .664/.877/.773/.848/+.019/420 |
+QDA+H+pc .654/.878/.769/.834/+.033/3300.  **NULL: every variant within ±.01 of the 57-param
router on photo; H-codes as router evidence cost old-world forgetting (+.065: router off-
distribution where H shifts); QDA+pc +.011 (1σ) for 7× params — not adopted.**
**The oracle was misread:** on photos the two colour voters sit at chance (.336/.332,
independent of msil .658), so "any voter right" = 1−(1−.658)(1−.33)² ≈ .85 — mostly LUCK,
not complementary skill. Attainable photo ceiling = best voter .658; the router is there.
**Routing lever exhausted; the photo lever is the msil voter itself (grouping / silhouette
quality), not arbitration.** Default stays `router` (57 params, no settle).

**s057 — `experiments/progenitor/kinopsis.py`, log `outputs/motion_organism_s057.log`.**
Rocky's ask this session: first "final arc before the showcase game"; when offered
(a) consolidation / (b) multi-object / (c) oracle gap he said 2(c), then corrected to
**"forget the game, focus on 2(a) — that's what we are chasing from the previous
session"** (s056 NEXT-0 remainder: absorb router + msil + vel into one organism object;
NEXT-2: continual schedule with replay settle). Done:
- `Kinopsis(seed)`: `.stage_old()` (shape_prefeeder.Organ, frozen) →
  `.settle_heads()` (optional, s054 replay recipe) → `.acquire_msil()` → `.fit_router()`
  (57-param Router, frozen replay, hard motion gate; voters frozen) → `.acquire_vel()`.
  Inference: `.shape(streams, evidence, mode)` → routed log-prob [N,5];
  `.velocity(Zmotion)` → [N,17]. `.save(path)` / `Kinopsis.load(path)`; reload
  gives identical predictions (checked each seed). All calibration stats (organ std,
  motion std, evidence μ/σ, empty-mask msil descriptor) travel inside the object.
- Schedule scored after stage 2 and stage 3 (n=3, moving packets):
| arm | mixed | photo | flat | old | forget | vel |
|---|---|---|---|---|---|---|
| uniform vote | .686 | .516 | .844 | .867 | 0 | .655 |
| **router (default)** | **.766** | .653 | .870 | .849 | +.019±.009 | .655 |
| settle+uniform | .739 | .616 | .884 | .847 | +.021 | .655 |
| settle+router | .765 | .657 | .878 | .837 | +.030 | .655 |
  Stage 3 leaves every shape number untouched in every arm (velocity = separate head,
  zero interaction, n=3). Router inside the organism = s056 standalone to 3 decimals.
  Replay-settle: +.008 flat / +.004 photo for +.011 more forgetting → not default.
- **Save size:** 14.2 MB for 115,632 trainable params. Inspected: NO data inside; per
  leaf the arena is ~250 KiB (edges + weights + Fisher/anchor buffers) and the compiled
  **scheduler plan ~1.6 MB** (vel leaf 7.4 MB because 656-d input). Fix = save arena
  tensors only and `compile()` on load (~1–1.5 MB; ~500 KiB without EWC buffers). Not
  done this session (mid-run); it is a format issue, not model size.
- Rocky's framing questions answered in-session (keep consistent): trioron does NOT beat
  CNNs on CIFAR-100 (pure 0.31 full / 0.61 task; cortex arm 0.15/0.63 vs ResNet ~0.75);
  its wins are CL-method comparisons at matched params + zero-forget absorption. For
  "CNN + trioron = good classifier" the missing experiment is frozen CNN + linear head vs
  frozen CNN + trioron under a continual schedule (queued, NEXT-4). "Absorbed" here is
  still paste-and-go leaves under a router, not cell-level `absorb()` (s056: +.09 forget).

## NEXT (priority)
0. ~~Oracle gap~~ CLOSED as NULL (above): oracle was luck-inflated. New 0: **lift the msil
   voter** — photo shape-from-motion .658 is the organism's photo ceiling; levers are grouping
   v4 → v5 (per-pixel motion compensation for interiors, s056 NEXT-6) and the silhouette
   descriptor; the velocity-cluster step from multi-object (item 3) may help here too.
1. Save format: arena-state-only + recompile on load (see above).
2. Cell-level absorption of msil into the organ's shape leaf once 0 is done (s056 graft
   cost .09 old-world).
3. Multi-object common fate (`train_multi`/`test_multi`, maxk=2, unused).
4. Frozen-CNN + linear head vs frozen-CNN + trioron, continual schedule (the "hardware"
   claim).
5. Wagon-wheel demo; emergence control (s019 doctrine) — unchanged from s056.
Rocky's s056 world-check (radius 6–14, defocus) was accepted implicitly; the showcase
game is deferred ("forget about the game atm").

## GOTCHAS (new)
- `Organ._train` leaves grad-bearing `last_activations` on the leaves; `stage_old` runs a
  no-grad forward afterwards or `deepcopy`/pickle fails ("Only Tensors created explicitly
  by the user support deepcopy").
- `voters()` with no `mshape` key = nothing-moves path (empty-mask descriptor, gate off).
- Old gotchas (motion.py build from repo root, 7 GB RAM ≈ 3 leaf procs, python3) apply.
  Run: `python3 experiments/progenitor/kinopsis.py` (~4.5 min/seed, 6 threads).

## State of the build / Pointers
- Commits on `conscience-core` (this session): d31e5d8 (organism) + close commit (Kinopsis
  rename, router sweep, log, handoff).
  `main` NOT advanced. Pushed.
- Saved organisms: `outputs/data/motion/motion_organism_s{0,1,2}.pt` (gitignored, router
  arm after stage 3).
- s056 files unchanged: motion.py / motion_front.py / motion_diag.py / motion_leaf.py /
  motion_absorb.py / motion_router.py; logs `outputs/motion_router_s056.log`, `*_s056b.log`.
- Package (`trioron/`) untouched.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`,
`trioron/legacy/outputs/`, `notebooks/` checkpoints, output PNGs / uncommitted
logs untracked; `outputs/data/**` caches.

## Environment notes
- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python
  3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`), 12 CPUs, 7 GB RAM.
