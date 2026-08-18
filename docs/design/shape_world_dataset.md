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

## Known floors

- Tiny outline + strong blur samples are unreadable by construction (label-noise floor, a few %).
- Blur is applied per object *before* compositing, so a defocused object's edge bleeds over a sharp neighbour — that is what depth of field looks like.

## Open / next

- Balance: shape sampled uniformly then fill; fields have one fill.
- Not yet: gradient/textured backgrounds, occlusion between objects,
  motion (Axis 7), rotation-range restriction (design doc §8).
- Fantasizing (novel-object generation by H-space factor recombination) is
  parked — recognition first (Rocky, s053).
