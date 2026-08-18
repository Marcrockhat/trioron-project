# Trioron Handoff

**Session date:** 2026-08-19 (started 2026-08-18)
**Session number:** 052
**Session title:** **The wave-stream arc: Shazam (null) → tokenizer (null) →
stereo (sync is the lever) → dense 2-D ⊕ stereo = new best fixed front end
(0.345 @ 800-d, raw-pixel parity at 30 % width) → what the unsupervised
layer-1 clusters are (texture-style bands, not class, not geometry) → style-
band nest loses to a single leaf (data starvation). Design doc for the next
arc: tokenize→frame→read → canonical-frame primitives (light / scale /
orientation / number) + recogniser trained in canonical frame, off-canonical
test sets. Rocky's frame for the project, restated at close: not chasing
CNNs — asking how nested simple linear elements (genotypic filter bank +
position-specific readouts + routing) get to what convolution gets, and
whether that route composes (crops / zoom / inversion / count) where
monoliths don't. "What we lack is data, not architecture."**

---

## READ THIS FIRST

1. **Everything this session is the s051 probe protocol** (`experiments/
   progenitor/diag_eye4.py` head: CIFAR-100 first 5 superclasses = 25 fine
   classes, 12.5 K train / 2.5 K test, supervised leaf `Seeded(d,100,48,
   nonlinear)`, 8 epochs, standardized; "full" = 100-way argmax over the
   25 present (chance 0.04), "task" = 5-way superclass-restricted (chance
   0.20), "blur" = 2×avg-pool→bilinear-up test set). **Single seed, n=1 —
   a ranking, not σ.** No nest, no Phasecyte, no absorb ran this session
   except the style-band nest probe (§ below).
2. **Design doc for the next arc:** `docs/design/canonical_frame_primitives.md`
   (DRAFT for Rocky's sign-off; §8 lists his open decisions). Read it
   before building anything. Build order §7: `frames.py` generators →
   `diag_number.py` (P_N) → frame primitives L/S/O → arms A/B/B′/C →
   100 classes → continual (arm D, G5).
3. **R's front end going forward:** dense 8-px/stride-2 2-D log-polar
   cepstra (4 radial × 8 orient − gain = 24/patch, 13×13 patches) region-
   pooled to 5×5 (600) ⊕ synchronised stereo (L = horizontal 1-D spectrum
   of the patch, R = vertical, 4+4 bins, pooled 5×5 = 200) = **800-d,
   0.345 / 0.529**; + disparity L−R (900-d) 0.355 = raw pixels 0.356 at
   ¼–⅓ the width. Blur 0.10 (raw 0.36) — accommodation, trainable per
   s051 (i). Code: `diag_stereo.py` (`dense_pooled`, `sync_pooled`,
   `sync_pooled_disp`).
4. **Phase-1 cap reminder:** 50 K params per substrate ⇒ ≤ ~800–1000
   input dims for a hidden-48 leaf. Everything wider is another leaf in a
   nest, not a fatter leaf. Rows marked "(OVER CAP)" in the logs are
   references only.

## WHAT RAN (all committed on `conscience-core`; logs in `outputs/*_s052*.log`)

### A. Proper Shazam probe (`diag_eye7.py`, `diag_eye7b.py`) — NULL for categories
Spectrogram (8-px windows, stride 2, log-polar 32 bins) → 3-D local-max
peaks (top-48) → anchor→target hashes (f1,f2,Δrow,Δcol; zone Z=3) →
inverted index over 12.5 K training images → **offset-consistent vote**
with the 2-D window coordinate as "time" (class-as-song and exemplar-as-
song), + bag-of-hash ablation, + corrected standardized-cosine kNN
baselines, + clean / blur / shift-4 (roll and pad) / fragment-20 (hard,
soft) test sets.

| test | class-vote | bag (no offset) | exemplar 1-NN | raw 1-NN (std) | cep(b) 1-NN |
|---|---|---|---|---|---|
| clean | 0.176 | 0.080 | 0.128 | **0.338** | 0.261 |
| 2×-blur | 0.148 | 0.079 | 0.096 | 0.330 | 0.145 |
| shift 4 px (pad) | 0.086 | 0.071 | 0.092 | 0.174 | 0.118 |
| fragment 20×20 (soft) | 0.068 | 0.066 | 0.070 | 0.209 | 0.103 |

Offset consistency doubles bag matching but the class table is a
**position-locked template** (under shift the true class's vote still
peaks at offset (0,0)); exemplar-level (true Shazam) is worst. Reading:
Shazam = instance retrieval; CIFAR = category generalisation. Keep as an
instance-memory primitive (place / specific-object re-identification)
for the world arc. Note the s051 (k) row was NOT a Shazam probe (spatial
energy peaks, no spectrogram, no vote); this one is.

### B. Wave-stream tokenizer (`diag_tokenizer.py`; env K0,V,WS,ST,NR) — FAILS gate
k-means codebook over per-window cepstral shapes → 2-D BPE (merge most
frequent adjacent right/down pair until V or min-count 50) → tokens =
merged extents; metrics compression / reuse (class entropy) / mosaic
tile-boundary respect / tokens-per-image on 4-tile mosaics; overlapping
n-gram bags (all pairs/triples, top-V vocab); reads via the same leaf
(per-slot centroid, random emb, bag, ⊕ VQ). Gate: tokens ≥ control − 2 pp.

| grid | K0 | V (sat.) | tokens/img | VQ per-slot | tokens per-slot | VQ bag | token bag | overlap pair bag | control |
|---|---|---|---|---|---|---|---|---|---|
| 5×5 (12/5) | 64 | 256 | 21.1/25 | 0.208 | 0.208 | 0.152 | 0.143 | — | (b) 0.304 |
| 5×5 | 64 | 718 | 18.6 | 0.208 | 0.210 | 0.152 | 0.126 | 0.108 | 0.304 |
| 5×5 | 128 | 481 | 22.3 | 0.223 | 0.222 | 0.161 | 0.138 | 0.102 | 0.304 |
| 5×5 | 256 | 432 | 24.0 | 0.236 | 0.232 | 0.163 | 0.149 | 0.082 | 0.304 |
| 13×13 (8/2), pooled 25 | 64 | 1024 | 106/169 | 0.268 | 0.263 | 0.201 | 0.150 | 0.160 | **0.322** |
| 13×13, pooled 25 | 256 | 2048 | 133/169 | 0.276 | 0.270 | **0.242** | 0.145 | 0.145 | 0.322 |

Readings: tokens ≤ VQ ≤ control in every row; quantisation costs 7–10 pp
(≈1.4 pp back per codebook doubling); merges saturate at pairs (max
extent 4) — no phrase structure at 32×32; overlapping n-grams nearly flat
and hurt; the dense grid fixes compression (169→106) not the read; no
number signal from token count. Tokens ARE class-agnostic (4.3–4.5 of
4.64 bits) and respect tile boundaries where measurable. Side findings:
dense 2-D pooled-25 (600-d) 0.322 > (b); dense VQ-256 bag 0.242 at 256-d
position-free (bag-of-visual-words). Rocky's reframing kept in the doc:
the s051 Phasecyte templates were judged by class purity, the wrong
criterion for a tokenizer — but the read still fails.

### C. Stereo spectra (`diag_stereo.py`, `diag_stereo2.py`)
| representation | d | full | task | blur |
|---|---|---|---|---|
| H rows / V cols / H+V unsync / H+V cepstral | 512/512/1024/512 | 0.171 / 0.146 / 0.189 / 0.208 | | |
| synced L/R raster, pooled 25 | 200 | 0.244 | 0.445 | 0.071 |
| synced L/R pooled 49 | 392 | 0.262 | 0.457 | 0.079 |
| synced + disparity, pooled 25 | 300 | 0.254 | | |
| synced full 169 (over cap) | 1352 | 0.243 | | |
| dense 2-D pooled 25 (control) | 600 | 0.322 | 0.511 | 0.132 |
| **dense ⊕ stereo pooled 25** | **800** | **0.345** | **0.529** | 0.095 |
| dense ⊕ stereo ⊕ disparity | 900 | **0.355** | 0.526 | 0.114 |
| dense pooled 49 (over cap) | 1176 | 0.340 | 0.530 | 0.130 |
| dense-49 ⊕ stereo-49 (over cap) | 1568 | 0.344 | 0.530 | 0.125 |
| raw pixels (s051) | 3072 | 0.356 | 0.533 | 0.359 |

Readings: **synchronisation is the lever** (Rocky's objection: row t and
column t are not the same object — fixed by time = shared 13×13 raster,
both spectra of the same patch): 200-d synced > 1024-d unsync. Stereo =
a 1-D projection (8/patch) of the 2-D window spectrum (24/patch); reads
like one; but the incoherent per-row average it carries is not in the
2-D bins, hence the +2.4 pp when combined. Its form (a genuine time-
ordered 169×8 stream) is the natural Axis-7 temporal-leaf input.

### D. What the unsupervised layer-1 clusters are (`diag_cluster_purity.py`, `diag_cluster_what.py`, `diag_cluster_shapes.py`)
k-means (PCA-64, k=25/50/100) as proxy for Phasecyte templates: eye DoG
0/25 clusters ≥34 % one class (s051 reproduced); (b) 3/25; dense 5/25;
**dense⊕stereo 3/25, purity 0.145, NMI 0.111**; raw pixels NMI 0.178
(colour/brightness/background is CIFAR's dominant unsupervised
structure, which gain-removed luminance features discard). Clusters are
not sparse (79–1147). What pushes them (η² by cluster vs by fine class):
**contrast 0.31 vs 0.06, HF energy 0.16 vs 0.06, orientation 0.18 vs
0.15; saturation/colour 0.06 vs 0.31, luminance 0.07 vs 0.16.** NMI vs
superclass 0.066 < vs fine 0.111 — not semantic at any level. **Not
geometry**: synthetic 32×32 circle/triangle/square land in the same
three clusters in the same proportions; polkadots/stripes shift toward
texture clusters; k-means on the shapes alone purity 0.30 (chance 0.20);
a supervised leaf on the same features gets 5 shapes 0.81 (square↔circle
still confused 83/176; polkadots/stripes near-perfect). ⇒ layer-1 =
**texture-style bands**; Phasecyte-first stays the wrong order on CIFAR;
colour is the biggest class-carrying factor the front end lacks.

### E. Style-band nest (`diag_style_nest.py`) — loses to single leaf
Rocky: "the clusters are our first layer; the down layer re-classifies."
k-means bands (dense⊕stereo) → one leaf per band, hard route (nearest
centroid) / soft (softmax −d/T) / uniform-mix control:

| k | hard | soft | uniform-mix | blur (hard/soft) | params |
|---|---|---|---|---|---|
| 1 (single leaf) | 0.345 | | | 0.095 | 45 K |
| 5 | 0.322 | 0.323 | 0.266 | 0.122 / 0.124 | 225 K |
| 10 | 0.292 | 0.318 | 0.260 | 0.111 / 0.107 | 450 K |
| 25 | 0.242 | 0.248 | 0.185 | 0.063 / 0.096 | 1.1 M |

Monotone decline = per-leaf data starvation (500/class → ~100 → 50 →
20); routed ≫ uniform shows real specialists; the only win is blur (a
"blurry band" routes blurred inputs). **Rocky: what we lack is data,
not architecture** (agreed for nesting; Chloe added: the fixed-front-end
+ single-linear-leaf ceiling and missing colour are also real).

## DESIGN DOC — `docs/design/canonical_frame_primitives.md` (DRAFT)
Question: with the image as a wave stream, do *nuisance-frame estimator*
primitives (light direction, scale/distance, orientation, number/
grouping), built as ≤50 K trioron leaves on generator ground truth and
frozen as donors, let a recogniser trained only in canonical frame
survive off-canonical input, and does that beat an augmentation-trained
monolith on anything (accuracy, params, robustness, continual)? Human
illusions (hollow-face, Thatcher, Ames/Ponzo) as the argument that the
recogniser is *not* invariant — it reads through separate frame
estimates with priors. Contents: §0 provenance (all of the above), §1
H1–H3 with prediction (rescue holds; economy ~tie; case rests on
continual + reuse), §2 factor generators on real CIFAR (light ramp 8 dirs
+ top; scale 0.5/0.7/1.4; orientation 90° steps + Thatcher flip; mosaics
2–4 subitizing / 6-8-12 "many"), priors = canonical over-represented, §2b
stage order tokenize→frame→read (tokenizer now marked DROPPED, §2c),
§2c all s052 results, §3 primitives P_L/P_S/P_O/P_N with senses + sanity
bars (plane fit, spectral centroid, connected components; P_N =
subitizer 1–4 exact + log-count Weber head), discovery control, §4 arms
A / B (conditioning) / B′ (canonicalising, the human model) / C
(augmentation, param-matched) / D (nest for continual), §5 gates G0–G5
(+G2b multi-object set-accuracy, G4 prior cost), §6 out of scope (shape
primitives until resolution/discovery is settled), §7 build order, §8
Rocky's decisions: arms to keep; rotation range; priors on/off;
tokenizer knobs (now moot).

## NEXT (priority — "chase the stars")
1. **Rocky signs off the design doc §8** (arms A/B/B′/C or B′ vs C only;
   90° steps incl. 180° vs ±30°; priors on/off).
2. `experiments/progenitor/frames.py` — factor generators + off-canonical
   test-set builders + analytic sanity bars; unit test on synthetic
   shaded blobs.
3. `diag_number.py` — P_N from the spectral-continuity boundary map + eye
   DoG on the mosaic sets; G0-N (subitizer exact 1–4; Weber 6:8 ≥ .7,
   6:12 ≥ .9).
4. `diag_frame_primitives.py` — P_L/P_S/P_O; G0; save under `runs/frames/`.
5. `diag_frame_arms.py` — A/B/B′/C at 25 classes; G1–G4.
6. Cheap ceiling-raisers for R, in parallel: **colour** (per-region
   opponent-colour means/contrast, ~50–100 dims; η² says it's the biggest
   class cue missing); blur augmentation; n=3 seeds on the 25-class
   probe; **100 classes** for dense⊕stereo (the number every arm compares
   to).
7. Later gates: 5-fixation absorb+nest level-2 (s051 part A) on the new
   front end; continual 20-task (G5) vs pure-trioron 0.309/0.606; Axis-7
   temporal leaf on the synced stereo stream; Shazam-style instance
   memory for the world arc; "dream a zoomed/cropped view never seen".
8. s051 items still open: absorb variance n=5 + settle done right; absorb
   the routers; s050/s049 items.

## GOTCHAS
- `pgrep -f "diag_x.py"` inside a `bash -c` waiter matches the waiter's
  own command line → deadlock (bit me once; killed and ran directly).
- Nested `exec` of probe files: set `__file__` to the target path before
  `exec` rather than string-replacing `os.path.abspath(__file__)`.
- Mosaic tile-boundary metric is structurally 0 on the 8/2 grid (clean
  slots either side are 4 steps apart, tokens ≤ 4) — uninformative there.
- The knn() in `diag_eye7.py` main run had a centering bug (row logged);
  corrected baselines are in `diag_eye7b.py` / `shazam_probe_s052_b.log`.
- Probe leaf rows print `[dream ...] ep` progress lines — filter on
  `^  .*full=` when tailing.

## State of the build / Pointers
- **Commits (all `conscience-core`, pushed at close):** `6bfdd02` Shazam
  + design doc; `8825d40` tokenizer sparse; `24b8e11` tokenizer dense +
  stereo; `c60c625` dense⊕stereo; `edd5793` cluster purity; `2df49e0`
  cluster what/shapes + style nest; + this handoff. `main` NOT advanced
  this session (last ff `d5c3607`/v0.3.2 + s051 handoff).
- New files: `experiments/progenitor/diag_eye7{,b}.py`, `diag_tokenizer.py`,
  `diag_stereo{,2}.py`, `diag_cluster_{purity,what,shapes}.py`,
  `diag_style_nest.py`; logs `outputs/{shazam_probe_s052,shazam_probe_s052_b,
  tokenizer_probe_s052_*,stereo_probe_s052,stereo2_probe_s052,cluster_purity_s052,
  cluster_what_s052,cluster_shapes_s052,style_nest_s052}.log`;
  `docs/design/canonical_frame_primitives.md`.
- No background runs at close. No checkpoints written this session.
- Package (`trioron/`) untouched this session.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`,
`trioron/legacy/outputs/`, `notebooks/` checkpoints, output PNGs /
uncommitted logs untracked.

## Environment notes
- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python
  3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`), 12 CPUs, 7 GB RAM.
- Probes: `OMP_NUM_THREADS=6 python3 experiments/progenitor/diag_stereo.py`
  (~10 min); tokenizer dense K0=256/V=2048 ~15 min; Shazam query ~5 min
  per test set. Logs buffer — check mtime before assuming a hang.
