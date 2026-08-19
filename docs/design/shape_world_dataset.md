# Shape world — synthetic recognition dataset (s053)

**Status:** v1 built this session. Generator `experiments/progenitor/shapes.py`;
splits under `outputs/data/shapes/*.pt` (gitignored; rebuild with
`python3 experiments/progenitor/shapes.py build`, deterministic per split seed);
`manifest.json` alongside. Probe: `experiments/progenitor/diag_shapes.py`.

## Why

CIFAR gives 500 images/class and entangles every nuisance factor. Rocky's
frame (s052 close): "what we lack is data, not architecture." This world
gives unlimited images per class with every factor *known and tagged*, so we
can ask the recognition question cleanly: which front-end primitive + which
leaf/nest carries which factor, and does the organism read images it has
never seen — fresh draws, *held-out factor combinations*, extreme zoom, crops,
iso-luminant, blurred, multi-object.

## Classes and factors

| factor | values | tag |
|---|---|---|
| shape | circle, triangle, square, stripes-field, dots-field | `objects[j].shape/name` |
| fill | solid, striped, dotted, outline (fields always solid) | `fill/fill_name` |
| line thickness | 1..3 px (outline only) | `thick` |
| colour | fg hue ∈ [0,1) (bin 0..5), sat, val; bg hue/sat/val; **iso**=1 ⇒ fg luminance == bg (chroma only) | `hue, hue_bin, sat, val, iso`, `bg` |
| scale (zoom) | radius 3..18 px — zoom-out tiny .. zoom-in overflowing the 32-px frame; textured fills capped at 13 (a zoomed dotted square *is* a dots field — boundary must stay in frame) | `r` |
| pose | rot 0..2π, shear −0.6..0.6 (skew), flip | `rot, shear, flip` |
| position / crop | (cx,cy); **crop**=1 ⇒ centre at a border (r ≥ 6, ≥ 30 % of the silhouette in frame); `vis` = exact in-frame fraction of the solid silhouette | `cx, cy, crop, vis` |
| blur (per object) | 0 sharp / 1 mild σ=.7 / 2 strong σ=1.5 | `blur` |
| focus (per image) | 0 uniform; 1 per-object depth-of-field (one sharp, another defocused; multi-object only); 2 gradient — sharp at a focal point, blur grows with distance ("eye focus") | `focus, focal` |
| count | 1..3 objects (fields only when count = 1); multi-hot `y_set` | `count`, `objects[]` |
| noise | Gaussian σ .03 | `noise` |

Image-level label tensors `ys` (object 0 = "the" object): `y_shape, y_fill,
y_iso, y_hue, y_scale, y_rot, y_vis, y_crop, y_blur, y_count, y_set[5],
y_focus, y_blur_img`. Full per-object records in `meta[i]["objects"]`.

## Held-out combinations (compositional "never seen")

`HELD = {(triangle, dotted), (square, striped), (circle, outline)}` never
appear in any *train* split; `test_held` contains only them. A recogniser
that reads shape and fill as separate factors passes; a template matcher
fails.

## Splits (n, seed fixed — same files on every PC)

| split | n | seed | notes |
|---|---|---|---|
| train | 20 000 | 1 | single-object, ~4 000/shape (8× CIFAR), HELD excluded |
| test_fresh | 5 000 | 2 | same distribution, unseen images |
| test_held | 3 000 | 3 | HELD combos only |
| test_stress | 12 000 | 4 | crop .3, iso .3, blur levels ⅓ each — subsets: zoom-out r≤5, zoom-in r≥14, cropped, iso, outline, sharp/mild/strong blur, focus-gradient |
| train_multi | 20 000 | 6 | 1–3 objects, multi-hot targets |
| test_multi | 4 000 | 5 | set-accuracy; depth-of-field subset |

## Intended readings (probe rows)

Fronts: dense⊕stereo 800 (s052 R) | + colour block 100 (per-region Y/RG/BY
mean + Y std) | colour only | raw pixels (over-cap reference). One
`Seeded(d,5,48)` leaf per head; n=3 seeds. Predictions before running (so
the result is a test, not a description): iso-luminant collapses on the
luminance-only front (its whole point); held-out combos ≥ fresh − 10 pp if
shape/fill are separable in the front end; strong blur costs the cepstral
front the most (s052 blur 0.10); zoom-in (overflowing) hurts the
region-pooled read; multi-object set-accuracy is the hard row.

## Results (s053, n=3 seeds unless noted; logs `outputs/shapes_probe*_s053.log`)

| front | d | fresh | geometric 3-way | held-out combos | small r<5 | cropped | iso | blur strong | fill |
|---|---|---|---|---|---|---|---|---|---|
| dense⊕stereo (s052 R) | 800 | 0.686 | — | 0.220 | — | 0.359 | 0.398 | 0.641 | 0.796 |
| dense⊕stereo⊕colour | 900 | 0.705 | 0.583 | 0.227 | 0.367 | 0.356 | 0.519 | 0.677 | 0.798 |
| colour block only | 100 | 0.429 | | 0.301 | | 0.256 | 0.502 | | 0.688 |
| raw pixels (over cap) | 3072 | 0.483 | | 0.279 | | 0.342 | 0.480 | 0.502 | 0.721 |
| **boundary-orientation only** | **92** | **0.709** | 0.655 | 0.354 | 0.465 | 0.406 | 0.391 | 0.648 | |
| boundary ⊕ colour | 192 | 0.761 | 0.664 | 0.352 | 0.455 | 0.395 | 0.604 | 0.751 | |
| boundary ⊕ colour ⊕ corner | 205 | 0.771 | 0.673 | 0.376 | 0.502 | 0.428 | 0.614 | 0.755 | |
| all four blocks | 1005 | 0.766 | 0.674 | 0.349 | 0.500 | 0.432 | 0.546 | 0.733 | |
| refs on 900-d: MLP 2×1024 / 1-NN | | 0.729 / 0.639 | | 0.228 | | | | | |
| refs on 205-d: MLP 2×256 / 1-NN | | 0.792 / 0.659 | | 0.374 | | | | | |

Multi-object set-accuracy (900-d, seed 0): 0.345 all / 0.506 single-object / 0.290 depth-of-field.

Readings: (1) with 8× CIFAR's data per class the fixed front end + leaf
plateaus at 0.70, and an over-cap MLP on the same features at 0.73 — the
ceiling is the **front end**, not the leaf and not the data. (2) The
cepstral front end reads texture (fields 0.88) not silhouette (circle/
triangle/square confuse each other; circle-outline → "square", square-
striped → "circle"); a dotted triangle is read as "dots" ⇒ shape × fill
are not factored ⇒ held-out combos at chance. (3) A 92-d boundary-
orientation block matches the whole 900-d front end and lifts small,
cropped and held-out; corner counting adds ~1 pp because textured fills
create interior corners. (4) Blur is cheap here (−3 pp strong) — opposite
of CIFAR (0.35 → 0.10). (5) The geometric 3-way sits at 0.67 for every
fixed primitive: shear turns circle/square into ellipse/parallelogram with
the same orientation statistics; the affine-invariant reading needs figure/
ground + fill/boundary separation *before* the descriptor — grouping, not
another global block.

### Grouping before describing (`grouping.py`, `diag_shapes4.py`)
Figure/ground (border-median bg, chroma-weighted distance, Otsu + noise
floor) → closing r=4 + hole fill → components → largest object's
silhouette / interior / colour / second-moment frame / flags; field rule =
≥3 components spanning ≥85 % of the frame. Mask IoU vs ground truth
(600-sample check): geometric 0.74 (solid 0.89, striped 0.78, dotted 0.68,
outline 0.59; iso 0.69, blur2 0.69, small 0.65, cropped 0.68); 91 % exactly
one object; fields flagged 74 % with 5 % false.

| stream | d | fresh | geo 3-way | held-out | small | cropped | iso | blur2 | fill fresh/held |
|---|---|---|---|---|---|---|---|---|---|
| whole-image bd+col+corner | 205 | 0.771 | 0.673 | 0.376 | 0.502 | 0.428 | 0.614 | 0.755 | 0.801 / 0.555 |
| grouped silhouette only | 92 | 0.632 | 0.762 | 0.528 | 0.486 | 0.534 | 0.652 | 0.601 | 0.685 / 0.140 |
| grouped sil+colour+frame+flags | 106 | 0.756 | 0.775 | 0.527 | 0.499 | 0.557 | 0.800 | 0.686 | 0.772 / 0.250 |
| grouped + interior | 706 | 0.801 | 0.783 | 0.464 | 0.541 | 0.582 | 0.767 | 0.757 | 0.828 / 0.527 |
| grouped 106 + whole 205 | 311 | **0.853** | **0.802** | 0.478 | 0.560 | 0.610 | 0.837 | 0.827 | 0.862 / 0.585 |
| CNN ref (242 K params, 30 ep, over cap) | — | 0.882 | 0.809 | 0.331 | 0.672 | 0.686 | 0.893 | 0.871 | — |

Readings: silhouette-only lifts geometric 3-way 0.67→0.76 and held-out
0.38→0.53 (shape and fill factor once boundary and interior are separate
streams; fill is read from the interior stream, 0.53 held-out vs 0.14
from the silhouette). 311-d fixed primitives + one 50 K leaf ≈ CNN on
fresh/geo/iso/blur and +15 pp on the compositional test (the CNN
memorises combos: 0.33). Gaps vs CNN: small (0.56 vs 0.67) and cropped
(0.61 vs 0.69) = mask quality. Count from grouping alone on test_multi:
1→0.91, 2→0.37, 3→0.00 — the closing that bridges dotted fills also merges
neighbouring objects (8 px apart): bridging must be per body, next fix.

### Grouping v2 + per-group read + scale canonicalisation (`grouping.py`, `diag_shapes5.py`)
v2 = per-body bridging (components merge only if the smaller is a texture
element ≤ 30 px or both are thin/parallel stripes; closing + hole fill
inside each body) + second-pass low-contrast foreground (re-Otsu on the
remainder, accepted if ≥ 1.1 × noise floor). Multi-object placement fixed
(rejection-sampled ≥ 1.4·(r₁+r₂)+1; failures TAGGED `overlap`, ~19 %).
Count from grouping alone, test_multi no-overlap: exact 0.71–0.74
(1: 0.89, 2: 0.84, 3: 0.75) vs 0.37 (v1); overlap-tagged 0.04 (occlusion).
Single-object masks unchanged (IoU 0.74).

Per-group read (shape leaf trained on single-object grouped 106 stream
reads each body; fields → whole-image 205 leaf), set-accuracy on test_multi:

| | plain | CANON (bbox crop → 32×32) | whole-image BCE 205 | whole+grouped BCE 311 |
|---|---|---|---|---|
| all | 0.357 | **0.479** | 0.330 | 0.423 |
| k=1 / 2 / 3 | 0.77 / 0.20 / 0.12 | 0.78 / 0.40 / 0.27 | 0.55 / 0.21 / 0.24 | 0.70 / 0.29 / 0.28 |
| no-overlap / overlap / depth-of-field | 0.41 / 0.15 / 0.16 | 0.55 / 0.21 / 0.36 | 0.35 / 0.24 / 0.24 | 0.46 / 0.27 / 0.28 |
| single: fresh / held-out / small / cropped | 0.750 / 0.544 / 0.496 / 0.555 | 0.749 / 0.577 / 0.572 / 0.601 | | |

Reading: bodies in multi-object images are small (r 3.5–6); scale
canonicalisation (the fovea's zoom, design-doc P_S) is what makes them
readable — +12 pp set-accuracy, +7 pp small, +5 pp cropped, +3 pp
held-out; the per-group read beats every whole-image multi-label leaf.
Remaining: bodies still read at ~0.6–0.75 each; rotation/shear
canonicalisation (P_O) next; fields in multi split 77 % flagged.

### Rotation/shear canonicalisation (`grouping.canon_affine`, n=3, 311-d = grouped 106 + whole 205)
Whitening the body mask by its second moments (C^-1/2; ellipse→circle,
parallelogram→square, sheared triangle→equilateral), residual rotation left
to the boundary block's angular spectrum.

| | plain | scale-canon | affine-canon | CNN ref |
|---|---|---|---|---|
| fresh | 0.853 | 0.856 | 0.858 | 0.882 |
| geometric 3-way | 0.798 | 0.808 | **0.814** | 0.809 |
| held-out combos | 0.478 | **0.511** | 0.493 | 0.331 |
| small r<5 | 0.556 | **0.627** | 0.597 | 0.672 |
| cropped | 0.619 | 0.654 | 0.656 | 0.686 |
| iso | 0.830 | 0.827 | 0.823 | 0.893 |
| blur strong | 0.828 | 0.834 | 0.831 | 0.871 |
| fill fresh / held-out | 0.861 / 0.582 | 0.865 / 0.591 | 0.872 / 0.607 | — |
| multi set-acc (per-group read) | 0.357 | 0.479 | 0.474 | — |

Reading: canonicalisation is a modest, consistent gain (geo 3-way now
≥ CNN; small +7, cropped +4, held-out +3); affine ≈ scale — whitening's
extra is eaten by mask imperfections (jagged dotted/outline bodies,
cropped bodies' moments) and edge softening on resample. Remaining CNN
margins: iso −6, blur −4, small −4.5, fresh −2.4 pp.

### Mask quality: convex closure per body (`grouping.convex_fill`, CONVEX=True)
Bodies in this world are convex (a cropped convex body is convex) → the
silhouette is the convex hull of the merged body. IoU (600 draws): 0.742 →
**0.789**; outline 0.59 → 0.74, cropped 0.70 → 0.82, iso 0.67 → 0.74,
blur2 0.70 → 0.76; solid 0.90, striped 0.81, dotted 0.71 (hull of dots sits
inside the edge; 1-px dilation didn't help), small 0.65 (≈ 1-px
discretisation floor for ~50-px bodies). Chroma weight 3–4 hurts (keep 2).
Recognition with convex masks (scale-canon, n=3): 106-d held-out 0.569 →
0.604, cropped 0.601 → 0.631; 311-d fresh 0.856 → **0.861**, geo 0.813,
cropped 0.670, held-out 0.515; multi set-acc 0.479 → 0.484. Reader
saturates ≈ 0.86 on this front end — diminishing returns on mask polish.

### Per-body reader streams (a): bcolour 12 / edge 4 / ctex 216 on the canon crop (n=3, scale-canon)
| stream | fresh | geo | held-out | small | cropped | iso | blur2 | fill / held |
|---|---|---|---|---|---|---|---|---|
| 311 | 0.861 | 0.813 | 0.515 | 0.618 | 0.670 | 0.831 | 0.832 | 0.858 / 0.550 |
| +bcolour 323 | 0.863 | 0.816 | 0.503 | 0.605 | 0.672 | 0.835 | 0.838 | 0.866 / 0.559 |
| +edge 315 | 0.863 | 0.816 | 0.517 | 0.614 | 0.666 | 0.834 | 0.837 | 0.861 / 0.576 |
| +ctex 527 | 0.864 | 0.820 | 0.474 | 0.620 | 0.654 | 0.826 | 0.836 | **0.876 / 0.608** |
| +all 543 | **0.866** | 0.820 | 0.473 | 0.629 | 0.655 | 0.832 | 0.836 | **0.883 / 0.621** |
| grouped-only 338 | 0.821 | 0.805 | 0.515 | 0.561 | 0.627 | 0.787 | 0.767 | 0.854 / 0.512 |
Reading: ≤ +0.5 pp on shape; ctex is a fill primitive (+2.5 / +7 held-out)
but costs shape held-out. **Fixed primitives + one 48-cell leaf cap ≈ 0.865
on this world (CNN 0.882); (a) closed — the rest is the reader.**

### (b) Per-factor leaves + router (`diag_shapes6.py`, n=3, leaves on 16 K / router on 4 K)
Leaves: shape ← canon silhouette+frame+flags 103; whole ← 205; fill ← ctex+flags;
hue / iso ← bcolour; blur ← edge+ctex; count ← grouping. Factor leaves fresh/
held-out: fill 0.79/0.57, hue 0.56 (6-way), **iso 0.94/0.92**, blur 0.57 (3-way).

| reader | fresh | geo | held-out | small | cropped | iso | blur2 |
|---|---|---|---|---|---|---|---|
| shape leaf alone | 0.749 | 0.744 | **0.609** | 0.533 | 0.625 | 0.783 | 0.696 |
| whole leaf alone | 0.768 | 0.649 | 0.372 | 0.502 | 0.410 | 0.606 | 0.746 |
| single 311-d leaf | 0.854 | 0.793 | 0.520 | 0.601 | 0.660 | 0.829 | 0.828 |
| uniform mix shape+whole (absorb) | 0.863 | 0.798 | 0.530 | 0.581 | 0.642 | 0.847 | 0.840 |
| ROUTER 16-cell leaf (80 ep) | 0.858 | 0.798 | 0.538 | 0.593 | 0.659 | 0.831 | 0.822 |
| linear router (ref) | 0.864 | 0.804 | 0.524 | 0.589 | 0.657 | 0.835 | 0.834 |
| ROUTER + fill/iso/blur context | 0.862 | 0.805 | 0.430±.06 | 0.620 | 0.643 | 0.827 | 0.825 |
Multi (seed 0): shape-set 0.476, (shape,fill)-pair-set 0.278 (no-overlap 0.548 / 0.343); count 0.83 no-overlap.

Reading: split = +1 pp over the monolithic leaf; learned arbitration ≈
uniform sum (all compositions 0.86 ± 0.5); other factors' logits as router
context re-entangle shape with fill (held-out ↓, unstable). A router at 8
epochs (125 steps) was 0.71 ± 0.05 — training bug, not a verdict. The
nest's value here is structural (factored, reusable reads: iso as its own
0.94 leaf; per-body reads), not raw accuracy — capped by the primitives at
≈ 0.86 vs CNN 0.88. The test the nest exists for is continual / absorb.

### Continual / compositional stream (`shapes_continual.py`, n=3, log `shapes_continual_s053.log`)
5 tasks: T1 solid {c,t,s} → T2 fields → T3 striped {c,t} → T4 dotted {c,s} → T5 outline
{t,s}; shared shape (5) + fill (4) heads, full-softmax, no task id; readers mono (one 311-d
leaf, 9 outputs) vs nest (shape+whole leaves summed, fill leaf); arms none / λ / credit
(lock rate 1.0) / replay / all; bar = CNN 242 K fine-tuned sequentially. Note the design
confounds fill with task (one fill per task) — hardest class-incremental case for the fill head.

| reader / arm | shape | fill | pair | forget | acq | T1@end | held pair | locked |
|---|---|---|---|---|---|---|---|---|
| mono none | 0.356 | 0.135 | 0.121 | 0.679 | 0.858 | 0.000 | 0.000 | 0 |
| mono λ | 0.581 | 0.135 | 0.101 | 0.609 | 0.758 | 0.000 | 0.274* | 0 |
| mono credit | 0.322 | 0.564 | 0.263 | 0.236 | 0.385 | 0.094 | 0.004 | 29 |
| mono replay | 0.636 | 0.456 | **0.399** | 0.478 | 0.839 | 0.295 | 0.042 | 0 |
| mono all | 0.574 | 0.394 | 0.209 | **0.065** | 0.261 | **0.723** | 0.016 | 10 |
| nest none | 0.343 | 0.136 | 0.121 | 0.678 | 0.857 | 0.000 | 0.000 | 0 |
| nest λ | 0.639 | 0.134 | 0.104 | 0.568 | 0.721 | 0.009 | 0.249* | 0 |
| nest credit | 0.418 | 0.140 | 0.114 | 0.550 | 0.716 | 0.014 | 0.017 | 89 |
| nest replay | 0.477 | 0.388 | 0.224 | 0.534 | 0.789 | 0.260 | 0.093 | 0 |
| nest all | 0.484 | 0.298 | 0.198 | 0.143 | 0.370 | 0.679 | 0.184 | 39 |
| CNN sequential | 0.217 | 0.134 | 0.077 | 0.520 | 0.633 | 0.000 | 0.003 | — |
\* λ's fill head collapsed to a constant fill → held pair ≈ shape × ⅓ (artefact).

Readings: (1) unprotected leaf and CNN both forget catastrophically (T1 → 0);
CNN ends worst (0.077). (2) Replay is the mechanism that carries final
accuracy (mono 0.40); λ protects the soma but the shared head drifts to the
last task; credit at rate 1.0 over-locks (mono acq 0.39) or locks interiors
without protecting the head (nest). (3) "all" keeps T1 (0.72) at ~zero
forgetting but stops learning (acq 0.26–0.37) — over-protection (lock rate
1.0 = 13× default), tuning not verdict. (4) Nest does not beat mono under
continual here (replay 0.22 vs 0.40). Follow-up arms (`shapes_continual_b_s053.log`): mono replay+λ shape 0.711 / pair 0.180 /
forget 0.49 / acq 0.68 / T1 0.03; mono all-soft 0.691 / 0.211 / 0.31 / 0.51 / 0.24;
nest replay+λ 0.711 / 0.220 / 0.35 / 0.57 / 0.50; nest all-soft 0.599 / 0.189 / 0.30 /
0.50 / 0.48. λ@1e3 lifts shape retention but strangles fill + acquisition; best final
pair stays mono+replay 0.40 (CNN 0.08). Open: STRENGTH 1e1–1e2 sweep, replay size,
and a stream that doesn't confound fill with task.

### Nest per-leaf diagnosis under continual (replay arm, n=3) and stereo on the crop
Per leaf at stream end: shape leaf T1 0.78 / fields 0.04; whole leaf T1 0.66 /
fields 0.32; their sum T1 **0.73** (mono+replay 0.30) — the nest retains shapes
better; its shape number is dragged by the fields (40 % of test images) which
live only in the whole leaf and are overwritten; fill leaf collapses to
outline (last task) + solid (97 % of predictions) — single-class-per-task.
⇒ "nest < mono" = fields + isolated fill head, both design, not mechanism.
Stereo on the canonical crop (`cstereo` 72 = synced L/R spectra pooled 3×3),
fill leaf n=3: ctex 0.803/0.577 → **ctex+cstereo 0.816/0.597**; cstereo alone
76-d 0.761/0.502; whole-interior 600 0.802/0.594. Same +1–2 pp as on CIFAR.

## Known floors

- Tiny outline + strong blur samples are unreadable by construction (label-noise floor, a few %).
- Blur is applied per object *before* compositing, so a defocused object's edge bleeds over a sharp neighbour — that is what depth of field looks like.

## Open / next

- Balance: shape sampled uniformly then fill; fields have one fill.
- Not yet: gradient/textured backgrounds, occlusion between objects,
  motion (Axis 7), rotation-range restriction (design doc §8).
- Fantasizing (novel-object generation by H-space factor recombination) is
  parked — recognition first (Rocky, s053).
