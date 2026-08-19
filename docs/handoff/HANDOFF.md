# Trioron Handoff

**Session date:** 2026-08-19
**Session number:** 053
**Session title:** **Shape world: a fully tagged synthetic recognition dataset
(5 shapes × fill/thickness × colour+iso × zoom 3–18 × skew/rot/flip × crop ×
per-object blur with 3 focus modes × count 1–3; unlimited images/class; held-
out shape×fill combos = compositional "never seen"). First probes: with 8×
CIFAR's data the s052 front end + leaf plateaus at 0.70 and an over-cap MLP
on the same features at 0.73 ⇒ the ceiling is the FRONT END, not the leaf,
not the data. Cepstra read texture not silhouette; a 92-d boundary-
orientation primitive alone matches the 900-d front end; geometric 3-way
stuck at 0.67 for every fixed primitive ⇒ built the GROUPING primitive
(figure/ground + body closing + components + field rule): silhouette-only
stream lifts geo 3-way 0.67→0.76 and held-out combos 0.38→0.53; grouped +
whole-image 311-d + one 50 K leaf = 0.853 fresh ≈ CNN ref 0.882 (242 K,
over cap) and +15 pp over the CNN on the compositional test (CNN 0.33 —
it memorises combos). Count from grouping: 1→0.91, 2→0.37, 3→0 (closing
merged neighbours) → grouping v2 (per-body bridging + second-pass
foreground; multi placement fixed, overlap TAGGED): count no-overlap
0.74 (2: 0.84, 3: 0.75); per-group read + SCALE CANONICALISATION (bbox
crop → 32) = multi set-acc 0.479 vs 0.33 whole-image leaf, small 0.50→
0.57, cropped 0.56→0.60, held-out 0.54→0.58. Rocky: recognition is the deliverable;
"fantasizing" is parked; nest = survivor recipe (SPLIT by factor +
arbitration = grouping).**

---

## READ THIS FIRST

1. **Rocky's ask this session:** train the nests on simple shapes (single/
   multiple, colours, skew, line thickness, zoom in/out, flip, crop, blur
   incl. mixed focus, textures combined on shapes) and identify images never
   seen; "what's important is the primitives to make it able to learn";
   most of the session = dataset build + tagging (done). Fantasizing = new
   term, distinct from trioron dreaming; explained as H-space factor
   recombination / decoder leaf; **not a priority**.
2. **Dataset:** `experiments/progenitor/shapes.py` (generator + `build`),
   doc `docs/design/shape_world_dataset.md` (schema, factors, splits, results
   table, known floors). Splits (fixed seeds, identical on every PC) under
   `outputs/data/shapes/*.pt` — **gitignored, 213 MB; rebuild with
   `python3 experiments/progenitor/shapes.py build` (~5 min)**; only
   `manifest.json` is committed. Feature caches
   `outputs/data/shapes/feat_<front>_<split>.pt` via `shapes_feats.py`
   (ds/col/bd/cn; extraction is seconds — leaf training is the slow part).
   Contact sheet with tags: `python3 experiments/progenitor/shapes_sheet.py
   train 36` → `outputs/shapes_sheet_train.png`.
3. **Front end code moved:** `experiments/progenitor/frontend.py` holds the
   s052 dense⊕stereo (800) lifted out of the exec-chained probes, plus new
   `colour_block` (100), `boundary_block` (92), `corner_block` (13);
   `grouping.py` = the grouping primitive (`groups`, `describe`). Old
   probes untouched.
4. **Probe protocol here:** train split 20 K single-object (≈4 000/shape),
   `Seeded(d,5,48,nonlinear)` leaf, 8 epochs, standardized features, n=3
   seeds (σ shown); tests = fresh (5 K, same dist) / held-out combos (3 K,
   only (triangle,dotted),(square,striped),(circle,outline)) / stress
   subsets (12 K: zoom-out r≤5, zoom-in r≥14, cropped, iso-luminant, outline,
   blur 0/1/2, focus-gradient) / multi-object (BCE 5-way sigmoid head trained
   on train_multi, set-accuracy on test_multi). Chance 0.20.

## WHAT RAN (all committed on `conscience-core`; logs `outputs/shapes_*_s053.log`)

### A. Dataset (`shapes.py`) — see doc for the full factor table
Per-object tags: shape, fill, thick, hue/hue_bin/sat/val, iso, r, rot,
shear, flip, cx/cy, crop, vis (exact in-frame fraction of the solid
silhouette), blur level; per image: count, focus mode (uniform / per-object
depth-of-field / gradient around a focal point), bg, noise. Constraints
added after eyeballing sheets: shape sampled first (balanced 5-way);
textured fills capped r≤13 (a zoomed dotted square *is* a dots field);
crops need r≥6 and ≥30 % silhouette in frame; fields only in single-object
images. Splits: train 20 K / test_fresh 5 K / test_held 3 K / test_stress
12 K / train_multi 20 K / test_multi 4 K.

### B. Recognition probes (`diag_shapes.py`, `diag_shapes2.py`, `diag_shapes3.py`)
| front | d | fresh | geo 3-way | held-out | small r<5 | cropped | iso | blur2 | fill |
|---|---|---|---|---|---|---|---|---|---|
| dense⊕stereo (s052 R) | 800 | 0.686 | — | 0.220 | — | 0.359 | 0.398 | 0.641 | 0.796 |
| dense⊕stereo⊕colour | 900 | 0.705 | 0.583 | 0.227 | 0.367 | 0.356 | 0.519 | 0.677 | 0.798 |
| colour block only | 100 | 0.429 | | 0.301 | | 0.256 | 0.502 | | 0.688 |
| raw pixels (over cap) | 3072 | 0.483 | | 0.279 | | 0.342 | 0.480 | 0.502 | 0.721 |
| **boundary-orientation only** | **92** | **0.709** | 0.655 | 0.354 | 0.465 | 0.406 | 0.391 | 0.648 | |
| boundary⊕colour | 192 | 0.761 | 0.664 | 0.352 | 0.455 | 0.395 | 0.604 | 0.751 | |
| boundary⊕colour⊕corner | 205 | 0.771 | 0.673 | 0.376 | 0.502 | 0.428 | 0.614 | 0.755 | |
| all four | 1005 | 0.766 | 0.674 | 0.349 | 0.500 | 0.432 | 0.546 | 0.733 | |
| refs: MLP 2×1024 on 900-d / on 205-d 2×256 | | 0.729 / 0.792 | | 0.228 / 0.374 | | | | | |
Multi-object set-acc (900-d): 0.345 all / 0.506 single / 0.290 depth-of-field.
Confusion (900-d): fields 0.88; circle 0.59, triangle 0.70, square 0.45;
circle-outline→square 512/1024, square-striped→circle 550/1002, triangle-
dotted→circle 356/974; per scale bin r<5 0.39 … r≥14 0.66; joint shape×fill
20-way leaf does NOT fix held-out (0.155).

**Readings.** (1) Data is not the bottleneck: 8× CIFAR/class, ceiling
0.70–0.73 for any reader on the s052 features. (2) The s052 front end is a
texture reader (fields easy, geometry confused, shape×fill entangled) —
the s052 "layer-1 = texture bands" finding, now on ground truth. (3) A
92-d boundary-orientation block = the whole 900-d front end; +colour 192-d
= best. (4) Corner counting is defeated by textured fills (interior
corners). (5) Blur is cheap here (−3 pp) unlike CIFAR. (6) Geometric 3-way
0.67 for every fixed primitive and the MLP: shear makes ellipse ≈
parallelogram in orientation statistics; the affine-invariant reading needs
grouping (figure/ground, boundary vs interior) before any descriptor —
i.e. the design doc's tokenize→frame→read, with grouping first.

### C. CNN dataset-ceiling reference (`shapes_cnn_ref.py`, OVER CAP, not trioron)
4-block BN CNN, 242 K params, 30 ep one-cycle: **fresh 0.882, geo 3-way
0.809, held-out 0.331, small 0.672, cropped 0.686, iso 0.893, blur2 0.871.**
⇒ readable ceiling ≥ 0.88; the learned monolith ALSO fails the
compositional test (memorises shape×fill combos).

### D. Grouping primitive (`grouping.py`) — figure/ground + object split + boundary/interior split
No labels, no params: bg = border median in (Y, 2·RG, 2·BY); distance map
(3×3 pooled); threshold = max(Otsu, 0.08, 2.5 × 10 %-quantile of border
distances = noise floor); FIELD if ≥ 3 raw components (≥ 4 px) whose joint
bbox spans ≥ 85 % of the frame both ways; else closing (disk r=4) + hole
fill → components (≥ 6 px) → per group: silhouette, boundary ring,
interior (eroded), raw-fg colour mean, second-moment frame (cx, cy, major/
minor scale, orientation, elongation, fill fraction), border-touch; count =
#objects. Sanity vs ground-truth silhouettes (600 draws): IoU 0.74 (solid
0.89 / striped 0.78 / dotted 0.68 / outline 0.59; iso 0.69, blur2 0.69,
small 0.65, cropped 0.68); exactly-one-object 91 %; fields flagged 74 %,
5 % false. `describe(X)` → silhouette-only boundary block (92) / interior-
only dense cepstra (600) / colour (3) / frame (7) / flags (4); cached as
`feat_grp_<split>.pt`. Cost ~10 ms/img.

### E. Grouped streams → leaves (`diag_shapes4.py`, n=3)
| stream | d | fresh | geo 3-way | held-out | small | cropped | iso | blur2 | fill fresh/held |
|---|---|---|---|---|---|---|---|---|---|
| whole-image bd+col+corner | 205 | 0.771 | 0.673 | 0.376 | 0.502 | 0.428 | 0.614 | 0.755 | 0.801 / 0.555 |
| grouped silhouette only | 92 | 0.632 | **0.762** | **0.528** | 0.486 | 0.534 | 0.652 | 0.601 | 0.685 / 0.140 |
| grouped sil+colour+frame+flags | 106 | 0.756 | 0.775 | 0.527 | 0.499 | 0.557 | 0.800 | 0.686 | 0.772 / 0.250 |
| grouped + interior | 706 | 0.801 | 0.783 | 0.464 | 0.541 | 0.582 | 0.767 | 0.757 | 0.828 / 0.527 |
| **grouped 106 + whole 205** | **311** | **0.853** | **0.802** | 0.478 | 0.560 | 0.610 | 0.837 | 0.827 | 0.862 / 0.585 |
| CNN ref (C) | — | 0.882 | 0.809 | 0.331 | 0.672 | 0.686 | 0.893 | 0.871 | — |
Count primitive (grouping alone, test_multi): exact 0.43; 1→0.91, 2→0.37,
3→0.00.

**Readings.** Separating boundary from interior is what factors shape from
fill: silhouette-only geo 3-way +9 pp and held-out +15 pp; fill reads from
the interior stream (held-out 0.53) not the silhouette (0.14). 311-d fixed
primitives + one ≤50 K leaf ≈ CNN on fresh/geo/iso/blur and +15 pp on the
compositional test. Gaps vs CNN = mask quality on small/cropped. Multi-
object counting fails because closing r=4 (needed to bridge dotted fills)
also merges neighbours 8 px apart — bridging must be per body (e.g. close
within each colour-consistent component, or grow bodies from seeds), and
the multi-object splits are the test.

### E2. Grouping v2 + per-group read + scale canonicalisation (`diag_shapes5.py`, logs `shapes_probe4b/5/5c`)
v2: merge raw components only if the smaller is a texture element (≤ 30
px) or both are thin/parallel stripes; close + fill inside each body;
second-pass Otsu on the remainder (≥ 1.1 × noise floor) recovers a low-
contrast (iso) object next to a high-contrast one. Multi splits REBUILT:
centres rejection-sampled (≥ 1.4·(r₁+r₂)+1), failures tagged `y_overlap`
(~19 %; 96 % of the old 3-object images were touching — unsplittable).
Count from grouping alone (test_multi): all 0.60; no-overlap 0.71–0.74
(1: 0.89, 2: 0.84, 3: 0.75); overlap 0.04. Single-object rows unchanged
(311-d 0.853). Per-group read set-accuracy (shape leaf on grouped 106
reads each body): plain 0.357 → **CANON 0.479** (k=2 0.40, k=3 0.27,
no-overlap 0.55, depth-of-field 0.36) vs whole-image BCE 205 0.330 /
whole+grouped BCE 311 0.423; the same CANON leaf on singles: held-out
0.577, small 0.572, cropped 0.601 (fresh 0.749). `grouping.canon_mask`,
`describe(..., canon=True)`, `describe_groups(..., canon=True)`,
`shapes_feats.grouped(split, canon)` cache `feat_grp[_canon]_<split>.pt`.

### F. Rocky's framing (keep)
The nest = the survivor recipe: WARM/FLEE masters were weak per class,
the win was SPLIT + arbitration. Here SPLIT is by factor (silhouette /
interior texture / colour / count) and the arbiter is grouping (which
pixels belong to which object, boundary vs interior) — before any leaf.
Caveat: today's primitives are hand-coded; the s019 doctrine (discover,
don't imitate) means a discovery control must come back once grouping
exists. Every image is multi-labelled (all factors tagged); the probes so
far train shape/fill/set heads and use the rest as test slices.

## NEXT (priority)
1. **Rotation/shear canonicalisation (P_O)** on the canon silhouette
   (second-moment frame → rotate major axis, de-shear) before the
   orientation spectrum; predicted to lift geo 3-way beyond 0.80 and
   held-out beyond 0.58; then interior texture on the canon crop too.
2. Per-body reads at ~0.6–0.75 each cap set-accuracy; multiply out: fix
   the body read first (1), then multi set-acc rows (`diag_shapes5.py`
   CANON=1). Fields in the multi split 77 % flagged — field rule on the
   canon crop / texture leaf fallback.
3. **Per-factor leaves + router (survivor recipe):** shape ← canon
   silhouette; fill ← canon interior; colour; count from grouping; blur/
   focus ← edge width; compose vs single 311-d leaf; then Phasecyte/
   trioron nest per band (data unlimited = clean nest-vs-leaf test).
4. Continual: shapes as tasks with the CL machinery.
5. Discovery control for the primitives (s019 doctrine); fantasizing
   (H-space recombination) once factors separate — parked.
6. Design doc `canonical_frame_primitives.md` §8 decisions still open;
   light-ramp generator not yet in shapes.py; occlusion (overlap tag) is
   now a labelled factor for later.

## GOTCHAS
- `pkill -f`/`pgrep -f` from inside the Bash tool matches the tool's own
  shell (exit 144, kills your own command) — use
  `ps aux | grep "[s]cript" | awk '{print $2}' | xargs -r kill`.
- `torch.multinomial` needs a float tensor (fixed in shapes.py).
- The trioron leaf `construct` hits "Edge buffer full" above hidden≈48 at
  900-d — over-cap references use plain torch MLPs.
- Probe scripts run on import (`diag_shapes*.py`); don't `import` them.
- Sheets: some samples are unreadable by construction (tiny outline +
  strong blur + crop); label-noise floor of a few %.

## State of the build / Pointers
- Commits (`conscience-core`): `f84f74f` dataset + probe 1; `f6dd87b`
  probes 2/3 + boundary/corner blocks; `8db6d0a` grouping + probe 4 + CNN
  ref; `98cdfae` grouping v2 + multi placement; + canon/probe 5 + this
  handoff. `main` NOT advanced. Pushed at close.
- New files: `experiments/progenitor/{shapes,shapes_sheet,shapes_feats,
  frontend,grouping,grouping_eval,diag_shapes,diag_shapes2,diag_shapes3,
  diag_shapes4,diag_shapes5,shapes_cnn_ref}.py`, `docs/design/shape_world_dataset.md` (has the
  results tables), logs `outputs/shapes_{build,probe,probe2,probe3,probe3b,
  probe4,probe4b,probe5,probe5c,cnn_ref,build2}_s053.log`, `grouping_eval_v2_s053.log`, `outputs/data/shapes/manifest.json`.
- No background runs at close.
- Package (`trioron/`) untouched. Memory pointer added
  (`shape_world_dataset.md`); memory is per-PC — this file is the truth.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`,
`trioron/legacy/outputs/`, `notebooks/` checkpoints, output PNGs /
uncommitted logs untracked; `outputs/data/shapes/*.pt` and `feat_*.pt`.

## Environment notes
- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python
  3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`), 12 CPUs, 7 GB RAM.
- Timings: dataset build ~5 min; front-end extraction seconds (cached);
  one trioron leaf on 20 K × 900-d ≈ 2–4 min; `diag_shapes.py` full ≈ 90
  min (raw-pixel leaf dominates); `diag_shapes3.py` ≈ 3 min per front.
