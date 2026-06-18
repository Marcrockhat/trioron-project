# Trioron Handoff

**Session date:** 2026-06-17
**Session number:** 042
**Session title:** **"Perspective is depth" — the phasor-optics front-end
decomposed into WHAT / WHERE / DEPTH, and the curved-emitter (Fresnel) channel
that recovers z.** Continuing s041's phasor optics. Five toy scripts (none
promoted). The arc: (1) BUILT the wavelength sweep (NEXT #1) and found it is NOT
the lever — the lens (WHAT) is already ~0.98–0.99 when registered, the sweep
adds ≤+0.02 and high carriers alias/hurt. (2) DECOMPOSED the bench's low 0.645
identity → it is WHERE/registration, almost all on the cramped row axis (WHAT
ceiling 0.98). (3) Rocky's perspective revision (curved 2nd emitter) is ~null on
FLAT 2D — *perspective needs depth* (he called it before the run). (4) On data
WITH depth the curved/Fresnel emitter recovers z near-losslessly (MAE ~0.015)
where the plane wave is provably depth-blind (=chance); depth map → identity
1.00. (5) Volumetric joint (z,r,c): z MAE 0.009 (aggregate tile focus), lateral
r/c ~0.5, identity 0.675. (6) **STEP 2 SOLVED: a field-wide tied-lens conv
response map** registered by min-centroid-distance gives identity **0.975**
(row MAE 0.00, col MAE 0.01) — matching the oracle 0.980 and retiring the stereo
where-channel on this bench.

---

## READ THIS FIRST

1. **Exploration only — nothing promoted into `trioron/`.** All five new files
   are under `experiments/progenitor/`; design captured in
   `docs/design/progenitor_council.md` **§9.1 (s042)**.
2. **The pinned mechanism (s042 result):** plane waves handle **x,y**; a
   **curved/Fresnel wavefront** handles **z** (depth-from-focus) and only bites
   when tiles carry depth; the **wavelength sweep** enriches an already-solved
   **WHAT**. The remaining weak bench number (identity 0.675) is purely lateral
   **row registration** — a *separate* problem from the depth channel.
3. **Regime caveat (carries from s040/s041):** the phasor front-end is
   FIXED-FILTER gradient-free but TOY-only (5×7 dot-matrix font, no real-data
   test yet). Do NOT conflate with the chained-15 headline 0.736/0.446
   (gradient-free PCLL organism, `run_pcll_chained15.py`) or the conv-depth
   stack (Adam, s041).
4. **DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM):**
   `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
   `trioron/viz/export.py`; `.claude/`, `runs/` untracked. Two stalled s039 raw
   logs still untracked. Output PNGs are regenerable and intentionally NOT
   committed. This session's 5 scripts ARE committed.

---

## What was built (and committed)

Commit this session (branch `progenitor-council`), 5 scripts under
`experiments/progenitor/` + design §9.1 + this handoff:

- `spectral_lens.py` — **wavelength sweep** (NEXT #1). Multi-carrier filterbank
  `R_p(w)=mean_j exp(i·w·θ_j)` vs the single carrier. Honest held-out eval, WHAT
  isolated (registered patches). **Finding: not the lever.** single-carrier
  already 0.927 (toy3) / 0.995 (digit10); sweep ≤+0.02 / +0.003; w>~4 aliases
  and HURTS (→0.86/0.95 at w≤32). z-score whitening > the sweep on toy3.
- `where_decompose.py` — **the bottleneck is WHERE.** 10-digit bench identity:
  oracle crop **0.980** (WHAT ceiling), stereo guess 0.450 (row MAE 1.21),
  margin refine 0.645 (row MAE 0.62), full row-slide 0.675 (row MAE 0.52). The
  whole loss is registration, almost all the **row** axis.
- `stereo_perspective.py` — Rocky's **curved 2nd emitter** (Fresnel
  `θ_B=w·idx+κ(idx−idx_c)²`) vs plane-pair vs opposite-curvature, on FLAT 2D
  digits. **~null / non-monotonic in κ** — flat data has no depth to triangulate.
- `lens_depth_focus.py` — **depth-from-focus.** Defocus `φ(a)=μ(1/z−1/f)a²`,
  aperture lock-in peaks at f=z. Curved depth MAE **0.013–0.019** vs plane
  **0.37–0.44** (=chance); depth-map identity **1.000** vs 0.483 (plane=chance).
- `volumetric_localize.py` — **joint (z,r,c).** z MAE **0.009** (aggregate tile
  focus = "center of the tiles"), r MAE 0.52, c MAE 0.51, identity 0.675;
  plane-wave z = 0.335 (=chance, depth-blind).
- `field_conv_register.py` — **STEP 2: field-wide tied-lens conv response map.**
  Sweep the shared lens over every (r,c); register at the peak. min-centroid-
  **distance** → identity **0.975** (row MAE 0.00, col MAE 0.01) = oracle 0.980;
  margin (gap) → only 0.620 (spurious off-target peaks). Retires stereo on this
  bench. Numpy proxy (~2 min); promotion uses `conv.forward_batch`.
- `volumetric_full.py` — **unified (z,r,c): conv register (x,y) + Fresnel focus
  (z).** identity **0.980 (= oracle)**, z MAE 0.010, r MAE 0.00, c MAE 0.01. The
  two solved channels compose without interference — the toy bench is solved on
  all four readouts (supersedes volumetric_localize's 0.675, pre–step-2 lateral).
- `taxonomy_receptor.py` / `taxonomy_manifold.py` — **first real(ish)-data test:
  the receptor on the 32-class hard taxonomy** (`data_hard`, K=32/D=12/M=3,
  Bayes 0.937, council 0.775). Finding (back-end-dependent, the honest one):
  against a WEAK back-end (numpy centroid/k-means) the per-feature phasor receptor
  BEATS raw (0.848 vs 0.818); against the REAL `ManifoldArchive` full-cov
  Mahalanobis it HURTS — **raw 0.901 vs phasor 0.876**. cos/sin of a Gaussian
  feature is non-Gaussian → breaks the density model's assumption. The receptor
  is a crutch for weak classifiers (and for images, where raw pixels suit no
  Gaussian); on near-Gaussian tabular data the substrate's QDA alone (0.901, near
  Bayes) doesn't need it. NOTE: literal per-SAMPLE `quantize` is image-specific
  (couples heterogeneous columns) — use a per-FEATURE phase frame for tabular.
- `taxonomy_lens1d.py` / `taxonomy_fold2d.py` — **the lens is a SPATIAL-LOCALITY
  prior** (Rocky's "1x2/1x3?" and "fold quanta to 2D to mimic image"). Faithful
  1xk lens: compression hurts monotonically (1x3 non-overlap 8-d → 0.227/0.591);
  per-feature (degenerate 1x1) best in the family. Fold-to-(feature×value)-image
  + 2D lens: underperforms per-feature (best 0.372/0.678 vs 0.471/0.876) —
  place→quantize→lens blurs the crisp value, dim blow-up breaks Mahalanobis,
  feature axis still non-local (corr-reorder only a tiny bump). Manufacturing 2D
  from non-local tabular adds no info; **raw + full-cov Mahalanobis 0.901 wins.**
  The lens earns its keep on IMAGES, not tabular. Positive control NOT YET run:
  same fold→lens→manifold on real MNIST (should win) — the clean boundary check.
- `taxonomy_fold34.py` — **corrected fold** (Rocky: 12 features → small 3x4 grid,
  not a 12x12 value-axis image). Right size (12-d) un-starves the covariance:
  Mahalanobis 0.670 → **0.762** (3x4/4x3 best; elongated 2x6/6x2 worse). Still
  loses to per-feature 0.876 / raw 0.901 — now a CLEAN locality result (2x2 avgs
  arbitrarily-ordered features; a 1D corr-chain can't be laid 2D-adjacent). The
  dim blow-up was half the earlier damage; locality mismatch is the rest.
- `taxonomy_perfeat_sweep.py` — **focus the filter on ONE quantum, read at many
  carriers** (Rocky: trade compute for capacity). Per-feature Fourier-feature
  expansion. WEAK back-end (centroid): **0.471 → 0.585** (+0.11) at w≤4, then
  aliases off — capacity bought with compute, confirmed. STRONG back-end
  (Mahalanobis): only hurts (0.876 → 0.799; covariance already has capacity, wide
  descriptor starves 128 samples). **META: front-end richness ≡ back-end richness
  are SUBSTITUTES.** The enrichment is the DEPLOYMENT lever — lift the cheap
  1.88 KB centroid with compute instead of the 92 KB full covariance.

## Key findings

- **The wavelength sweep is built but is not the accuracy lever** — the lens
  identifies near-perfectly once registered; richer carriers alias.
- **Perspective is a DEPTH phenomenon** (Rocky, confirmed): a curved wavefront
  does nothing on a flat object (only reparametrizes one plane); it recovers z
  near-losslessly once the data has depth. The math: both plane-wave emitters
  carry the same shape-phase bias `angle(S(w))`, so a parallel 2nd view can't
  triangulate an extended object — a *different-geometry* (curved) view can.
- **The plane wave is provably depth-blind** (z at chance); the Fresnel emitter
  is the z channel. Flat object → one focal plane (aggregate); relief/thick
  object → per-tile depth map.
- **The standing bench cap WAS lateral row registration — now SOLVED (f):** the
  field-wide conv response map (min-distance) recovers 0.975 = oracle. Register
  by **distance to a learned centroid**, not margin (margin peaks off-target).

## NEXT (priority — step 2 DONE)

1. **Promote the field-wide conv register** — the numpy proxy is a Python
   double-loop (~2 min/200 imgs). Build the substrate version: the shared lens as
   a tied conv cohort (one `lineage_root`, `conv.forward_batch`) swept over the
   field, batched. This is the promotion-shaped path (`ScatteringLens` + min-
   distance register). Generalize `spawn_conv_cohort` (kernel-from-patch) first.
2. **WHEN** channel — object moving over *time*; lock-in as the moving screen;
   ties to Axis 7 (temporal gene). Untested.
3. **Real-data test** — the front-end is toy-only (5×7 font); run the lens on
   real MNIST/chained-15.
4. **Word/symbol capacity** — continuous centroid descriptor vs the 1000-pocket
   aliasing ceiling. Untested.
5. **Promotion** to a package (`ScatteringLens` + stereo/Fresnel emitter +
   manifold path) — deferred until the exploration matures (Rocky's call).
   Generalize `spawn_conv_cohort` (kernel-from-patch) before any conv promotion.

## OPEN / unresolved

- Phasor front-end is toy-only; no real-data (chained-15/CIFAR) test yet.
- Depth toy uses a synthetic point-scatterer defocus model, not a rendered 3D
  volume; the z result is the mechanism, not a benchmark.
- s039/s041 carries (untouched): gradient-free conv kernel still open; plain-gabor
  discrepancy; chained-15 over-segmentation (cap=128); generative mean-template
  readout (−0.32); conv-depth on centered chained-15.
- All phasor numbers toy-only; conv numbers (s041) gradient-based n=2 seeds.

## State of the build

- Branch `progenitor-council`. Package code UNTOUCHED; gate battery green (s036).
  DO-NOT-COMMIT carries left alone. Output PNGs regenerable, uncommitted.
- Run cost: all s042 toys are seconds–low-tens-of-seconds each (CPU).

## Pointers

- **Design:** `docs/design/progenitor_council.md` **§9.1 (s042 — wavelength
  sweep, WHAT/WHERE/DEPTH decomposition, perspective=depth, volumetric)**; §9
  (s041 phasor optics); §8 (conjoined-twin conv weight-tie — the lever for
  step 2's field-wide conv).
- **s042 scripts** (all `experiments/progenitor/`): `spectral_lens.py`,
  `where_decompose.py`, `stereo_perspective.py`, `lens_depth_focus.py`,
  `volumetric_localize.py`. s041 base: `fingerprint_lens.py` (lens + centroid),
  `stereo_emitter_2d.py` (plane Vernier), `digit_bench_2d.py` (font, variant,
  place, descriptor, crop, where_2d — reused by the s042 scripts).
- **Conv weight-tie:** `trioron/phenotype/conv.py` (lines 1–24 docstring:
  parameter-sharing = the defining property; reduction guarantee = "no sharing →
  plain linear").
- **Receptor/lock-in:** `trioron/core/receptor.py` (quantize, θ=2πq/1000),
  `trioron/pcll/lockin.py`, `trioron/pcll/mixed.py` (canonical frame).
- **Back-end:** `trioron/learning/manifold.py` (ManifoldArchive = the per-class
  centroid the lens feeds).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`),
  `OMP_NUM_THREADS=8`. matplotlib 3.10.9, PIL 10.4.0 for renders.
- **Model note:** the phasor thread leans on optical metaphors
  (emitter/lens/wavelength/focus/perspective) — Rocky's framing, fine in
  conversation. In code/docs lead with the COMPUTATIONAL framing (phase code
  θ=2πq/1000; filterbank; complex correlation; Vernier dual-frequency unwrap;
  Fresnel quadratic-phase defocus; centroid fingerprint) — the optical metaphors
  can trip mid-session safety downgrades.
- **Regime distinction (DO NOT CONFLATE):** chained-15 headline 0.736/0.446 is
  the gradient-free PCLL organism. Conv depth (s041) is Adam. The phasor
  front-end is fixed-filter gradient-free but toy-only.
