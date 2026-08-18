# Tokenize → frame → read: a wave-stream tokenizer, canonical-frame
# primitives (light, scale, orientation, number), and a frozen-frame
# recogniser (design, s052)

**Status:** DRAFT for Rocky's sign-off (s052). No code yet.
**Owner:** Rocky (direction) / Chloe (engineering).
**Question:** with the image turned into a *wave stream* (windowed
spectral shapes, s051 `(b)`), does the speech pipeline — **tokenize the
stream first** (fragmenter = tokenizer, unsupervised), then estimate each
token's *frame* (illumination, scale, orientation) with frozen trioron
primitives, then *read* tokens with a recogniser trained only in the
canonical frame — beat the plain alternative (one recogniser over the
untokenized stream, trained with full augmentation) on anything we care
about (accuracy, params, robustness, continual)? Number/grouping falls
out of the tokenizer (fragment count).

**Budget (binding, spec §6):** ≤ 50 K params per substrate ⇒ with the
probe leaf's hidden 48, ≤ ~800–1000 input dims per stream (`(b)` = 800
was built to this). Anything wider is another leaf in the nest, not a
fatter leaf (s051 part A rule).

## 0. Where this comes from

- s051 primitive probes (`outputs/primitive_probe*_s051.log`, table in
  HANDOFF s051 part B): windowed cepstra `(b)` 0.304 is the best fixed
  front end; the **shape-vocabulary primitives (g/h) were a null**
  (0.144 alone, +0 on top of cepstra; detector caps 0.72 on clean
  synthetic 16×16 shapes); fly/saccade and constellation nulls.
- s052 Shazam probe (`diag_eye7*.py`, `outputs/shazam_probe_s052*.log`):
  spectrogram peaks + offset-consistent hash vote — offset consistency
  doubles bag-of-hash matching (0.176 vs 0.080, 25-class) but is far
  below standardized raw-pixel 1-NN (0.338); exemplar-level (true
  Shazam) is the worst row; under a 4-px shift the class vote still
  peaks at offset (0,0) — the class-aggregated table is a position-
  locked template, not an aligner (offset diagnostic, `_b.log`).
  **Reading: Shazam is instance retrieval; CIFAR is category
  generalisation. Null for categories; keep as an instance-memory
  primitive (place / specific-object recognition) for the world arc.**
- Rocky (s052): the first trioron should *fragment the wave into smaller
  fragments* — like a **tokenizer**. This reframes the s051 Phasecyte-
  first CIFAR result: templates were judged by class purity, which is
  the wrong criterion for a tokenizer (a BPE token is not a class
  either); the criterion is compression + reuse + downstream read.
- Rocky (s052): human perception is *tricked* by distance and by
  inversion (hollow-face: light-from-above prior beats shading evidence;
  Thatcher: inverted faces read as normal; Ames/Ponzo: size read
  through assumed distance). **The recogniser is not invariant to
  these factors; it relies on separate estimates — with strong priors
  — of light, scale, orientation, and reads the object in that assumed
  frame.** Being tricked is the cost of the prior. Therefore the
  primitives are canonicalisers, not classifiers.

## 1. The claim, stated so it can fail

H1 (rescue): a recogniser R trained only on canonical-frame data
collapses on off-canonical test sets; the same R conditioned on / fed
through frozen primitive estimates recovers most of the canonical
accuracy.

H2 (economy): the primitive+R organism matches an augmentation-trained
recogniser R_aug of equal total params on off-canonical accuracy, at
lower *task-specific* training (R sees canonical data only; the
primitives are task-agnostic and reusable).

H3 (continual): in the 20-task continual CIFAR run, primitives are frozen
donors shared across tasks; only per-task recognisers grow. Forgetting
and per-task params should be no worse than the pure-trioron bar
(blindman v2b 0.309/0.606) with off-canonical test robustness that v2b
does not have.

Falsification: if H1 fails (no rescue) the primitives do not carry the
frame; if H1 holds but H2 fails by > 2 pp at matched params AND H3 shows
no continual advantage, the design is a null and is recorded as such.

Prediction stated up front (Chloe): H1 holds; H2 is close to a tie —
augmentation is a strong baseline; the case for the design rests on H3
+ reusability, not on raw accuracy.

## 2. Nuisance factors and generators (hand-coded generator = benchmark
## data, allowed; primitives built with our APIs, required)

All factors are applied to *real CIFAR images* (not only synthetic
shapes — that is what killed (g): the shape detectors never saw natural
statistics). A synthetic shaded-blob generator is kept as the *unit test*
of each primitive, not its training set.

| factor | canonical | off-canonical set | how applied to a CIFAR image | ground-truth label |
|---|---|---|---|---|
| **light direction** L | from above (top), ambient 0.5 | 8 directions {N,NE,E,SE,S,SW,W,NW} + top-lit | multiplicative Lambertian ramp: I' = I·(a + (1−a)·max(0, n̂·l̂)) with n̂ a fixed synthetic height field (image-independent hemispherical bump) — the shading pattern is what a light source imposes on a convex object; a = ambient | direction index (8-way) + "top" |
| **scale** S | s = 1.0 (native 32×32) | s ∈ {0.5, 0.7, 1.4} (down-scale + pad with per-image mean; up-scale + centre-crop) | resample about the centre | s bin (4-way) |
| **orientation** O | upright | rotated {90, 180, 270} + flipped-vertical (Thatcher case) | rot90/flip | 5-way |
| **grouping / number** N | one object, centred | k ∈ {2,3,4} objects (subitizing range) and k ∈ {6,8,12} ("many", the approximate-number range) composed as a k-tile mosaic of down-scaled held-out CIFAR images with per-image-mean fill (tile scale = the scale factor S; N and S are entangled by construction, as in nature) — the multi-object frame; the recogniser must know it is looking at several things before it can read any of them | compose from held-out images | count target with the human structure: **exact classes 1/2/3/4, one class "many" (≥5)**, plus a scalar log-count regression head for the ≥5 range (Weber/ratio behaviour) + a per-window "which cluster" map |
| **position** | fixation centre | (already the eye's fixations; not a primitive here) | — | — |

Priors (the "human" part): the primitives are trained with the
*canonical* class over-represented (e.g. 50 % top-lit / upright /
s=1) so that under ambiguity they default to it — the hollow-face
behaviour. Test explicitly whether the prior hurts (§5, gate G4).

## 2b. Stage order: tokenize → frame → read

1. **P_F tokenizer (first trioron).** Input: the wave stream = per-window
   spectral shape (cepstra, 32-d per window; 25 windows of 12 px stride
   5 as `(b)`, or 169 windows of 8 px stride 2 as the Shazam probe —
   knob). Vocabulary learning is **unsupervised, BPE-like**: (i) base
   symbols = k-means codebook over per-window shapes (or the Phasecyte
   lock-in templates we already have), (ii) iteratively merge the most
   frequent *adjacent* symbol pairs (2-D adjacency: right/down) into
   compound tokens up to V ≈ 256; boundaries fall out where no merge
   applies. Tokenizing an image = greedy longest-match over the window
   grid → token ids + extents. Outputs: token id per slot, extents,
   **fragment count** (subitizer 1/2/3/4/many) + log-density head (the
   approximate-number sense) — number is a by-product of tokenizing.
   The mosaic generator (§2) is *evaluation* only (do token boundaries
   respect tile boundaries?), never the tokenizer's training signal.
2. **Frame primitives per token.** P_L / P_S / P_O estimate the frame of
   each token extent (a token can be lit/scaled/rotated differently from
   its neighbours — the multi-object case).
3. **Read.** R reads tokens in canonical frame; the composer combines.

Tokenizer metrics (report before any frame primitive is built):
compression (tokens per image; ≤ 25 slots), reuse (token frequency
across classes — a token that fires for one class only is a class, not
a token), and downstream: R over the token stream (25 slots × 32-d
token embedding = 800, or a V=256 bag, both under the cap) vs R over
untokenized `(b)` 0.304 / raw 0.356 at 25 classes.

### 2c. Tokenizer probe result (s052, `diag_tokenizer.py`, logs `outputs/tokenizer_probe_s052_*.log`)

25-class probe, same leaf; wave stream = 25 windows × 32-d cepstra.

| K0 (codebook) | merges → V | tokens/img (25 slots) | max extent | class-entropy (max 4.64) | tile-boundary cross (chance .19) | VQ per-slot | tokens per-slot | VQ bag | token bag | overlap pair bag |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 192 → 256 | 21.1 | 2 | 4.42 | .064 | 0.208 | 0.208 | 0.152 | 0.143 | — |
| 64 | 654 → 718 (saturated) | 18.6 | 4 | 4.31 | .100 | 0.208 | 0.210 | 0.152 | 0.126 | 0.108 |
| 128 | 353 → 481 (sat.) | 22.3 | 2 | 4.28 | .040 | 0.223 | 0.222 | 0.161 | 0.138 | 0.102 |
| 256 | 176 → 432 (sat.) | 24.0 | 2 | 4.30 | .017 | 0.236 | 0.232 | 0.163 | 0.149 | 0.082 |
| control `(b)` un-quantised | | | | | | **0.304** | | | | |

**Dense stream** (Rocky: "21 tokens for 32×32 is too few" — 8-px windows,
stride 2, 169 slots × 24-d; per-slot reads region-pooled to 5×5 = 600-d
to fit the cap; logs `..._dense8_*.log`):

| K0 | merges → V | tokens/img (169 slots) | max extent | control: dense pooled 25 | VQ per-slot pooled | tokens per-slot pooled | VQ bag | token bag | overlap pair bag |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 960 → 1024 (not saturated) | 106 | 4 | **0.322** | 0.268 | 0.263 | 0.201 | 0.150 | 0.160 |
| 256 | 1792 → 2048 (not sat.) | 133 | 4 | 0.322 | 0.276 | 0.270 | **0.242** | 0.145 | 0.145 |

The dense stream fixes the *tokenizer* metrics (169 → 106 tokens, V no
longer saturates) but not the read: tokens ≤ VQ ≤ control in every row
(gate 0.302: best token read 0.270). Two side findings: the dense
stream region-pooled (600-d) is a **new best fixed front end, 0.322 >
`(b)` 0.304 at ¾ the width**; and the dense VQ-256 *bag* reads 0.242 at
256-d with no position at all (bag-of-visual-words) vs 0.154 for the
s051 pooled-cepstra row.

Readings. (1) **Fails the gate** (≥ control − 2 pp) on both grids: every tokenized read ≤ the
quantised stream it is built on, and quantisation itself costs 7–10 pp
(64→128→256 recovers ~1.4 pp per doubling — the cost is quantisation
per se, not codebook size). (2) **There is no phrase structure to
tokenize at 32×32**: merges saturate (V 718/481/432 at min-count 50),
every merged token is a pair (max extent 2, one run reaches 4),
compression 18–24 of 25 slots; overlapping n-grams are nearly flat
(8052/8192 possible pairs occur; top-1024 cover 41 %) and *hurt* the
read (pair bag 0.10–0.11; pair bag + VQ per-slot 0.159 < VQ per-slot
0.208). Adjacent window shapes are ~independent at this quantisation.
(3) The tokenizer *behaves* like a tokenizer where it can: tokens are
class-agnostic (entropy 4.3–4.4 of 4.64 bits) and merges respect
mosaic tile boundaries (1.7–10 % cross vs 19 % chance) — spectral
continuity is real, but too weak to compress. (4) **No number signal**
from token count (4-tile mosaic 17.8–23.7 vs single 18.6–24.0).
(5) Larger V and overlapping tokens (Rocky's two knobs) do not change
(1)–(4).

**Stereo spectra** (Rocky, s052; `diag_stereo.py`, `outputs/stereo_probe_s052.log`):
two 1-D frequency streams. Unsynchronised (H = each row's spectrum,
V = each column's) 0.171 / 0.146 / H+V 0.189 (1024-d) / cepstral 0.208
(512-d). **Synchronised** (time = shared 13×13 raster of 8-px patches;
L = horizontal spectrum of the patch, R = vertical; both see the same
object at the same time): pooled-25 **0.244 at 200-d**, pooled-49
**0.262 at 392-d**, + disparity L−R 0.254 (300-d), full 169 (1352-d)
0.243. Readings: synchronisation is what matters (200-d synced > 1024-d
unsync); stereo = a 1-D projection (8 numbers/patch) of the 2-D window
spectrum (24/patch), reads like one (0.262 vs 0.322 same grid); its
niche is *form* — a genuine time-ordered stream (169 × 8) for the
Axis-7 temporal leaf, which the static spectrogram is not.

Consequence for the design: **stage 1 (tokenize) is dropped as a
partition/BPE tokenizer.** What survives of the fragmenter idea is the
*continuity boundary map* (adjacent-window spectral discontinuity —
which the mosaic check shows is real) as a P_N input, and the un-
quantised stream `(b)` stays R's input. Build order §7 step 2 becomes:
P_N grouping/number from the boundary map + eye DoG (§3), no tokenizer.

## 3. Primitives = trioron leaves, frozen donors

Each primitive P_k is a supervised trioron leaf built with the same
recipe as the s051 probe leaf (`cifar_eye_nest.train_sub`: `Seeded(d,
n_out, 48, nonlinear)`, Adam, standardized input), input = the sense
the factor is physically carried by (no hand-coded feature *extractor*
beyond the senses we already have; the leaf learns the estimator):

| primitive | sense (input) | d | output | sanity bar (must beat) |
|---|---|---|---|---|
| P_L light | eye DoG signed (`Eye((16,16),rectify=False)`, 412) + 4×4 block luminance means (16) | 428 | 9-way | 2-parameter luminance plane fit → 8-way direction (analytic) |
| P_S scale | log-polar power spectrum, whole image (160) — scale is a radial shift in log-frequency | 160 | 4-way | spectral-centroid threshold |
| P_O orientation | cepstral spectrogram `(b)` (800) — orientation flips window positions + rotates the orientation bins | 800 | 5-way | chance 0.2 (there is no trivial analytic bar; report the leaf) |
| P_N grouping/number | eye DoG signed (412) + per-window DoG energy (25) | 437 | 5-way {1,2,3,4,many} + log-count scalar (≥5) ; optional 25-way per-window cluster id (auxiliary head) | connected components of thresholded DoG energy → count (analytic) |

P_N is the *grouping / number* primitive (Rocky, s052): mono- vs
multi-object, with the human number structure — **subitizing** (exact
and immediate up to ~4) and an **approximate number sense** beyond (a
ratio/Weber estimate, "many"), which is why people hardly count past 5
at a glance. The 5-way head is the subitizer; the log-count head is the
approximate sense. Gate G0-N tests both signatures separately: exact
accuracy on 1–4, and ratio discrimination (6 vs 8 vs 12) on the many
range — a primitive that is exact everywhere or approximate everywhere
has the wrong shape.
Its downstream role differs from the other three — it does not warp the
input, it tells the eye/composer *how many fixations to spend and where*
(one cluster → one canonical read; k clusters → k reads, one per
cluster, each canonicalised by P_L/P_S/P_O and read by R). This is the
point where the primitive set meets the s051 eye design (fixations).

Each is trained once on CIFAR-100 train images under random factor
settings (task-agnostic: labels are the *factor*, never the class),
then frozen (`freeze=True` at graft; λ-locked). Param budget ≈ 25–45 K
each (Phase 1 ceiling per substrate 50 K).

**Discovery variant (Rocky's reflex-vs-wisdom rule — primitives should
be discovered, not imitated):** the factor labels here are *physical
ground truth from the generator*, not a policy to imitate, which is
the accepted use ("generator hand-coded = benchmark data is fine").
Still, log an unsupervised control: k-means (k=9/4/5) on the same
senses; report the cluster–factor alignment (as v3 taxonomy did) so we
know how much of each frame is discoverable without labels.

## 4. Recogniser and the three arms

R = the current best fixed-front-end leaf: cepstral spectrogram `(b)`
(+ eye DoG, `(f)`, as a knob), 25-class probe protocol first, then 100.

| arm | what | trained on |
|---|---|---|
| **A** R alone | s051 `(b)` leaf | canonical only |
| **B** R ⊕ primitives, *conditioning* | R's input = sense ⊕ [P_L, P_S, P_O softmax outputs] (17 extra dims); the recogniser learns to read the frame | canonical + off-canonical (R must see off-canonical to learn the conditioning; params matched to C) |
| **B′** R ∘ primitives, *canonicalising* | P_O/P_S estimates are used to *un-rotate / re-scale the input* before R; P_L output divides out the estimated shading ramp; R itself trained canonical-only | canonical only |
| **C** R_aug | R trained with the full factor augmentation, params matched to B (wider hidden) | canonical + off-canonical |
| **D** nest | B′-style organism as an s051-part-A nest: leaves {P_L, P_S, P_O, R} under a composer / band router (`graft(..., freeze=True)` for primitives, R plastic) | as B′ |

B′ is the human-model arm (recogniser lives in the canonical frame,
primitives carry it there); B is the cheap version; C is the
alternative the design must justify itself against; D is the organism
form needed for H3.

## 5. Gates

Test sets: canonical; each factor off-canonical alone (light-8, scale-3,
orient-4, count-2/4); all jointly; plus the s051 2×-blur set (accommodation
control). Metrics: full (100-way argmax over present classes) and task
(superclass-restricted), as s051. n=3 seeds on the leaf for the
headline arms; primitives single-seed (they are frozen fixtures).

- **G0 primitive quality:** each P_k beats its sanity bar on held-out
  CIFAR test images under its factor (P_L ≥ plane-fit + 10 pp; P_S ≥
  centroid + 10 pp; P_O ≥ 0.6; P_N ≥ connected-components + 10 pp on
  1–4 exact; on the many range, ratio discrimination 6:8 ≥ 0.7 and
  6:12 ≥ 0.9 — the Weber signature). Fail → fix the primitive before any arm.
- **G2b multi-object read:** on the 2-object set, R alone (one read)
  gets one label at best; B′/D with P_N → per-cluster reads scores
  set-accuracy (both labels right) — report; bar = R applied to the two
  known halves (oracle grouping).
- **G1 collapse (H1 premise):** A drops ≥ 15 pp full on the joint
  off-canonical set vs canonical. If it does *not* collapse, the frame
  is already in the features and the design has nothing to rescue.
- **G2 rescue (H1):** B′ recovers ≥ 50 % of the A-canonical − A-off gap;
  B ≥ B′.
- **G3 economy (H2):** B′ (or B) within 2 pp of C at matched params on
  the joint off-canonical set.
- **G4 prior cost:** with the canonical prior (§2) vs uniform training of
  the primitives, off-canonical accuracy under *ambiguous* shading (low
  contrast, ambient 0.8) is reported both ways — the hollow-face test.
  Not a pass/fail; a measurement.
- **G5 continual (H3):** 20-task continual run of D vs the pure-trioron
  bar (0.309/0.606) and vs A-continual; report forgetting, per-task
  params, off-canonical accuracy. This is the arm that decides whether
  the design ships.

## 6. What is NOT in scope (yet)

- Shape primitives: not re-run until the resolution/discovery question
  (HANDOFF s051 (g)) is settled; if re-run, at whole-image 32×32 via the
  eye's fovea and as discovered clusters.
- Position: the eye's fixations already carry it (s051 design).
- Motion / temporal factors (Axis 7): none in CIFAR.
- Shazam-style instance memory: separate primitive for the world arc
  (place / specific-object re-identification), tracked in the s052
  handoff, not here.

## 7. Build order (each step is one probe script + a log line, s051 style)

1. `experiments/progenitor/frames.py` — the factor generators (light
   ramp, scale, orientation, mosaics) + off-canonical test-set builders +
   analytic sanity bars. Unit test on the synthetic shaded-blob set (must
   be near-perfect there).
2. ~~`diag_tokenizer.py` — P_F tokenizer~~ DONE s052, **failed the gate**
   (§2c). Replaced by: `diag_number.py` — P_N from the spectral-
   continuity boundary map + eye DoG on the mosaic sets; G0-N
   (subitizer exact 1–4, Weber ratio on 6/8/12).
3. `diag_frame_primitives.py` — train P_L, P_S, P_O per token extent;
   report G0 and the discovery control; save the leaves under
   `runs/frames/`.
4. `diag_frame_arms.py` — arms A/B/B′/C at 25 classes; G1–G4.
5. If G2+G3 pass (or G2 passes and Rocky wants the continual answer
   regardless): 100 classes, then `cifar_frame_nest.py` for D and G5.

Cost: steps 1–3 are probe-scale (minutes each on CPU); step 4 is a
20-task continual run (hours, s051 scale).

## 8. Decisions needed from Rocky

- Arms: keep all of A/B/B′/C, or drop B (conditioning) and run only the
  human-model B′ vs C?
- Factor ranges: the ones in §2, or narrower (rotation ±30° instead of
  90° steps — CIFAR objects are rarely upside-down; the Thatcher case
  argues for including 180°)?
- Priors on by default (§2, canonical over-represented) or uniform?
- Tokenizer knobs: base codebook k-means vs Phasecyte templates; window
  grid 25 (12 px/5) vs 169 (8 px/2); V = 256.
