# Trioron Handoff

**Session date:** 2026-08-20
**Session number:** 054
**Session title:** **s053 recipe carried to split-CIFAR-100 (handoff NEXT-1): full-cov
manifold replay ~doubles diag replay (mono full .112→.204), post-task HEAD-ONLY
settle adds the rest (mono .236, nest 0.330±0.002 / task 0.658 / forget 0.26 —
86 % of its own static 0.383), while the 254 K CNN trained sequentially collapses
to EXACT CHANCE (.010/.100). + Rocky's pre-feeder: the s053 shape-world nest,
frozen, as a peripheral organ on CIFAR — alone 0.221 full (22× chance, 22 K
params); as INPUT to a leaf +2.1 pp continual / +1.1 static; as a 5th VOTER in
the nest: nothing.**

---

## READ THIS FIRST

1. **Rocky's asks this session:** (a) "continue our latest model and test it on
   CIFAR" = handoff s053 NEXT-1 (full-cov replay on CIFAR-continual) — done, it
   transfers; (b) mid-session: "can the previous nest be extended as a
   pre-feeder for the new CIFAR model … not tabula rasa but peripheral
   augmentation" — built (`shape_prefeeder.py`) and measured (below);
   (c) framing he stated: fewer params, mostly linear/fixed math, near-CNN
   performance — "nature has its way"; the hand-coded primitives ≈ retinal
   channels (colour opponency, V1 orientation, spatial-frequency, binocular),
   and he pointed at MOTION (common fate) as the missing grouping cue —
   noted as a future arc, our grouping substitutes colour-distance+Otsu
   because datasets are static.
2. **Bench:** `experiments/progenitor/cifar_continual.py` — split-CIFAR-100,
   10 tasks × 10 fine classes in index order (same split as the archived
   `bench_2_0_cifar_continual`), ONE shared 100-way head, full-softmax, NO task
   id, 8 ep/task Adam 1e-3, n=3 seeds. full = 100-way argmax (chance .01),
   task = argmax within the sample's own 10 classes (chance .10), forget =
   mean(per-task full acc right after training − at end). STREAM=joint = static
   reference. Front-end features cached at `outputs/data/cifar/feat_*.pt`
   (gitignored; rebuild ~12 min, grouping dominates at ~8.5 ms/img).
3. **Readers** (every leaf = `Seeded(d,100,48,nonlinear)`):
   mono = s053 311-d (grouped canon sil+col+frame+flags 106 ⊕ whole bd+col+cn
   205), 32 K; mono-ds = s052 dense⊕stereo⊕colour 900-d, 73 K (OVER the 50 K
   cap — reference); nest = shape 103 + whole 205 + fill (ctex+cstereo+flags)
   292 + ds 900 leaves, log-softmax sum, 147 K total (ds leaf over cap; other
   three 10–19 K); nest3 = the exact s053 nest (74 K, all under cap);
   \*+pre = + the frozen shape-organ stream (below).
4. **Arms:** none | replay (diag sketch) | replay-full (full-cov, boundary-
   cached eigendecomposition rank 32, per-class batch = clamp(256//n_past,
   2, 32)) | full+credit-soft (lock rate .078) | full+settle = full-cov replay
   + post-task HEAD-ONLY re-calibration (200 Adam 1e-3 steps on class-balanced
   full-cov samples of every archived class; soma frozen) — the archived CIFAR
   bench's "load-bearing fix", now per leaf.

## RESULTS (all n=3 except noted; logs `outputs/cifar_*_s054.log`)

### A. Continual split-CIFAR-100
| reader | arm | full | task | forget | acq | t1_end |
|---|---|---|---|---|---|---|
| mono 32 K | none | .066±.002 | .181±.007 | .534 | .600 | .000 |
| mono | replay (diag) | .112±.003 | .429±.008 | .407 | .519 | .069 |
| mono | replay-full | .204±.006 | .567±.001 | .271 | .475 | .192 |
| mono | **full+settle** | **.236±.004** | .568±.004 | **.203** | .440 | .228 |
| mono | full+credit-soft | .160±.009 | .480±.005 | .259 | .419 | .208 |
| mono-ds 73 K | replay-full | .238±.003 | .608±.000 | .287 | .526 | .250 |
| mono-ds | full+settle | .281±.004 | .614±.001 | .204 | .485 | .296 |
| nest 147 K | replay-full | .194±.005 | .653±.002 | .425 | .619 | .171 |
| nest | **full+settle** | **.330±.002** | **.658±.001** | .255 | .584 | .331 |
| mono+pre 43 K | full+settle | .257±.001 | .587±.003 | .204 | .461 | .256 |
| nest+pre 168 K | full+settle | .326±.006 | .654±.001 | .251 | .577 | .333 |
| **cnn-seq 254 K** | none | **.010±.000** | **.100±.000** | .343 | .353 | .000 |

### B. Static (joint, all 100 classes at once — the ceilings)
mono .292/.630 | mono-ds .329/.665 | nest3 .314/.643 (74 K, beats blindman-v2b
.309/.606 at 7.6× fewer params) | nest .383/.711 | pre-only .221/.553 (22 K) |
mono+pre .303/.642 | nest+pre .379/.708 | CNN 242 K .442/.772.

### C. Readings
1. **Full-cov transfers.** Diag→full-cov doubles full-softmax (mono .112→.204,
   mono-ds .116→.238, ~15σ) — same mechanism as shapes E10: diag sketches are
   off-manifold in correlated descriptor spaces. 90 classes × (μ+Σ) per leaf,
   no images.
2. **Head-settle is the second half** (+.03 mono, +.14 nest). It fixes what
   replay-during-training can't: interleaved replay fights the current task;
   the post-task head-only pass on class-BALANCED archive samples re-calibrates
   without touching the soma. This is `settle_head_with_retry` from the
   archived bench / absorption arc, reproduced per leaf.
   It closes 86 % of the continual→static gap for the nest (.330 vs .383).
3. **Nest wins where it should**: task-aware .658 ≈ its static .711 scale;
   nest+settle full .330 vs mono .236. The CNN sequential = exact chance —
   continual is where the architecture pays, statics the CNN still wins
   (.442 vs .383).
4. **Credit-soft REGRESSES here** (mono .160 < .204; nest s1 .163): 9
   boundaries × lock-cap locks 18/48 mono cells (79 across nest leaves) and
   starves acquisition. s053's credit-soft-neutral was 4 boundaries. Lock
   budget must scale with task count — open design point.
5. **Pre-feeder (Rocky's ask):** the s053 nest retrained jointly on shapes
   (deterministic, `PRE_SEED=0`; sanity on shapes .867 shape/.815 fill),
   frozen, native (shape-world) standardization, emits 3×48 H-codes + 14
   logits = 158-d per CIFAR image. Alone: .221/.553 static (22× chance) — the
   organ genuinely reads CIFAR. Concatenated as INPUT: mono+pre continual
   .257 vs .236 (+2.1 pp, ~5σ), static .303 vs .292. As a 5th VOTING leaf:
   nest+pre .326±.006 vs .330±.002, static .379 vs .383 — a weak voter with equal say
   dilutes; augmentation belongs at the INPUT (or needs a router), not the
   vote. "Not tabula rasa" confirmed cheap and positive at +11 K params.

## NEXT (priority)
1. **Weighted/routed arbitration for heterogeneous leaves** — the nest sums
   log-softmax uniformly; nest+pre shows a weak voter dilutes. H-space
   `ManifoldRouter` over leaves, or learned per-leaf temperature. (This is
   also s053 NEXT-4.)
2. **Absorb proper on CIFAR** (`trioron.api.absorb`): graft a NEW task's leaf
   without retraining, vs the settle recipe.
3. Motion/common-fate grouping arc (Rocky): frame-differencing as the grouping
   cue on a moving shape world; replaces the Otsu heuristic with the cue
   biology uses. + discovery control for the hand-coded primitives (s019
   doctrine) — Rocky repeated the germline framing.
4. Lock budget vs task count for credit locking (finding 4).
5. n=5 + capacity push (16 ep) on the operating point if paper-bound;
   batched grouping (8.5 ms/img loop) for anything bigger.

## GOTCHAS (new this session)
- `cifar_continual.py` RUNS ON IMPORT (like diag_shapes*); `shape_prefeeder.py`
  therefore reads the feature caches directly — never import cifar_continual.
- 7 GB RAM fits at most ~3 leaf processes (full-cov archives grow: mono ~1 GB,
  nest ~2.2–2.8 GB); two OOM kills this session — stagger seeds as separate
  processes (`SEEDS=2 READERS=nest`), watch `free -m`.
- READERS=none = CNN bar only. FEATS_ONLY=1 = build caches and exit.
- `ManifoldArchive.replay_batches`/`sample_full` need `finalize_all()` at the
  boundary first (astrocytes must be DORMANT); the bench caches (μ, eigvecs,
  resid) per class at each boundary — sample_full's per-call eigh would be
  ~90 × 800² otherwise.
- Old gotchas from s053 (pkill self-match, ARMS string-typo silently = "none",
  output_ids private) all still apply; ARMS matching here is exact-string in
  `Leaf.__init__` too.

## State of the build / Pointers
- Commits (`conscience-core`): `20dc14a` bench + prefeeder + first results;
  + this close commit (remaining logs + handoff). `main` NOT advanced. Pushed.
- New files: `experiments/progenitor/cifar_continual.py`,
  `experiments/progenitor/shape_prefeeder.py`; logs
  `outputs/cifar_{feats,joint,joint_pre,prefeeder_build}_s054.log`,
  `outputs/cifar_continual_{mono,monods,cnn,mono_settle,monods_settle,monopre}_s054.log`,
  `outputs/cifar_continual_nest{_s0,_s1,_s2,_settle_s0,_settle_s1,_settle_s2,pre_s0,pre_s1,pre_s2}_s054.log`.
- Caches (gitignored): `outputs/data/cifar/feat_{ds,col,bd,cn}_{train,test}.pt`,
  `feat_grp_canon_*.pt`, `feat_pre_*.pt` (~1 GB; rebuild: FEATS_ONLY=1 run then
  `python3 experiments/progenitor/shape_prefeeder.py`; needs shapes splits —
  `python3 experiments/progenitor/shapes.py build` if absent).
- Package (`trioron/`) untouched. Timings: mono arm ~17 min/3 seeds; nest arm
  ~1 h/seed; settle adds ~2 min/run; CNN-seq ~20 min/seed.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`,
`trioron/legacy/outputs/`, `notebooks/` checkpoints, output PNGs /
uncommitted logs untracked; `outputs/data/**` caches.

## Environment notes
- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python
  3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`), 12 CPUs, 7 GB RAM.
