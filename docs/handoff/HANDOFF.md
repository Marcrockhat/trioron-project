# Trioron Handoff

**Session date:** 2026-06-18
**Session number:** 043
**Session title:** **Deep scattering wins, SHIFTED quadruples it, and the
front-end becomes a LIVING trioron (learnable kernels + λ).** Continuing the
phasor→trioron wiring. (1) Built a gradient-free conv→pool→conv (Mallat
scattering S1⊕S2): centered MNIST 0.909→**0.931** centroid (depth lift +0.022),
SHIFTED-MNIST **0.765/0.971** with depth lift **quadrupling to +0.091 ±0.010**
(raw collapses to 0.399; Maha un-saturates). (2) Then, on Rocky's ask, wired two
of the triparametric node's three variables back in: **w** (conv kernels now
LEARNABLE via Adam through `conv.forward_batch`) + **λ** (the real
`epigenetic_lock` module). n=5 split-MNIST: λ eliminates **6.3pp** of catastrophic
forgetting (kernel drift 4.98→0.011) at a 2.8pp B-cost. Three files, committed.

---

## READ THIS FIRST

1. **Exploration only — nothing promoted into `trioron/`.** New file under
   `experiments/progenitor/`; the package is untouched.
2. **THE RESULT:** the 2nd scattering layer is real, and SHIFTED-MNIST is where it
   shines (centered MNIST saturates the back-end and hides it).
   - Front-end (both files): L1 fixed-Gabor conv → modulus (U1) → [S1 = pool(U1)]
     and [L2 fixed-Gabor conv on the UN-pooled U1 → modulus (U2) → S2 = pool(U2)].
     Descriptor = S1 ⊕ S2. **L2 is DEPTHWISE** (each L1 channel re-convolved
     independently by a fresh Gabor bank — the scattering transform; cross-channel
     fixed-Gabor mixing is unprincipled, so NOT done).
   - **CENTERED** (`mnist_scatter_deep.py`, n=3): raw 0.812 | S1 0.909/0.980
     (reproduces s042 EXACTLY → pipeline validated) | S2 0.873/0.979 |
     **S1⊕S2 0.931/0.985**. DEPTH lift +0.022 ±0.001 (centroid). Mahalanobis
     near ceiling → centroid is the cleaner readout here.
   - **SHIFTED** (`shifted_scatter_deep.py`, 28×28 digit at random offset [0,8]²
     on 36×36, same front-end/pooling, n=3): raw **0.399** | S1 0.674/0.915 |
     S2 0.741/0.962 | **S1⊕S2 0.765/0.971**. DEPTH lift **+0.091 ±0.010**
     (per-seed +0.096/+0.098/+0.079) — 4× the centered lift. Three reads:
     (a) conv-over-raw margin explodes (+0.275 vs +0.097 centered — the moving
     digit collapses the fixed-column raw centroid); (b) Mahalanobis no longer
     saturated (S1 0.915), depth fills +0.056; (c) **S2-only BEATS S1-only**
     (0.741>0.674) — the |·|*ψ cascade builds translation robustness, so 2nd-order
     coeffs survive the shift better than 1st-order.
3. **THE BRIDGE TO A LIVING TRIORON** (`scatter_learnable_lambda.py`): the
   scattering organ was fixed-Gabor / gradient-free (substrate as wiring fabric,
   (w,λ,u) dormant). This wires **w** and **λ** back in.
   - **w learnable:** Adam flows into the cohort-ROOT edges through
     `conv.forward_batch`'s differentiable gather; `lineage_root` ties the gradient
     across positions → ONE shared kernel/channel is learned. Learned kernels reach
     accA 0.970 (5-way). NOTE: eps inside the modulus `sqrt(c²+s²+1e-6)` is REQUIRED
     — `sqrt'(0)=inf` NaNs the kernels otherwise.
   - **λ wired** via the real `trioron/learning/epigenetic_lock.py`: `anchor` +
     `accumulate_saliency` (native |w·g|) → `refresh_lambda` (per-cell row-sum) →
     `strength·ewc_penalty` added to the next task's loss.
   - **n=5 split-MNIST** (A={0..4}→B={5..9}, shared learnable kernels, per-task
     heads → forgetting on A is PURE kernel drift), N_CH=2 (starved so kernels are
     load-bearing): λ OFF forgets 0.063 (drift 4.98); **λ ON forgets −0.000 (drift
     0.011)**, accB 0.928 vs 0.956 — the protect↔plasticity tradeoff. STRENGTH
     sweep {100..3000} all zero A-forgetting; 1000 = sweet spot.
   - **HONEST limit:** with the roomy N_CH=8 front-end there is NO forgetting to
     fix — general scattering features are drift-robust; forgetting lives in the
     READOUT, not a redundant perception organ (manual §7). λ only earns its keep
     when capacity is load-bearing. u (utility) still unused; no growth/replay/dream.
4. **Regime:** standard benches only (centered + shifted MNIST). Do NOT conflate
   with the chained-15 PCLL headline (0.736/0.446) or the s041 Adam conv-depth
   (shifted 0.842 vs raw 0.493). All conv runs through the real substrate
   `conv.forward_batch` (lineage_root weight-tie).
5. **DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM):**
   `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
   `trioron/viz/export.py`; `.claude/`, `runs/` untracked. Output PNGs regenerable,
   uncommitted. This session's three scripts ARE committed.

---

## What was built (committed, branch `progenitor-council`)

- **`mnist_scatter_deep.py`** (commit `fde74ab`) — gradient-free deep scattering
  (S1⊕S2) on CENTERED MNIST. Reuses s042's `spawn_fixed_cohort`, `conv_map`,
  `balanced`, `centroid_acc`; defines a K-parameterized `gabor(k,…)` (s042's was
  hardcoded to K=9 — L2 needs K2=5). Env `SEEDS=0,1,2`. L1 K9/s2 → 10×10, 8 ch
  (4 orient × 2 freq), pool 5; L2 K5/s1 → 6×6, 8 ch depthwise, pool 2. Dims
  S1=200, S2=256, S1⊕S2=456 (< PER_TRAIN=700 so full-cov Mahalanobis conditions).
  ~18s/seed.
- **`shifted_scatter_deep.py`** (commit `766e62e`) — the SAME front-end on
  SHIFTED-MNIST (the cleaner arena). Imports the pure scattering leaf-helpers
  (`build_bank`, `modulus_maps`, `pool_to`) from `mnist_scatter_deep`; local
  `shift_onto_canvas` (the `conv_depth_shifted_mnist` placement). Env `CANVAS
  OFFSET SEEDS S1_POOL S2_POOL`. ~63s/seed (bigger canvas: H1 14×14, H2 10×10).
- **`scatter_learnable_lambda.py`** (commit `418cc59`) — the LIVING-trioron bridge
  (learnable w + epigenetic lock λ; see READ-THIS-FIRST §3). Reuses
  `spawn_fixed_cohort`/`conv_map` (s042) + `gabor`/`pool_to` (s043). Env `SEEDS
  N_CH EPOCHS EPOCHS_B PER_CLASS STRENGTH LR`. Defaults = the λ-demo regime
  (N_CH=2 starved, EPOCHS_B=20, LR=0.02, STRENGTH=1e3). ~23s/seed.

## Key findings

- **The 2nd fixed-Gabor convolution lifts the single-layer wired-conv win,
  gradient-free, reproducibly** — centered +0.022 ±0.001, shifted **+0.091
  ±0.010** (every seed positive). The s041 conv→pool→conv depth claim now holds
  WITHOUT Adam: fixed kernels + modulus + a 2nd layer is enough.
- **Shifted-MNIST is the cleaner arena** (handoff hypothesis confirmed): the
  moving digit collapses raw to 0.399, un-saturates Mahalanobis (S1 0.915 vs
  centered 0.980), and 4×'s the depth lift. Centered hid all of this.
- **S2-only is strong and, under translation, BEATS S1-only** (shifted 0.741 >
  0.674) — the scattering cascade builds invariance; concatenating wins.
- **The front-end is now a living trioron** (learnable w + λ) — λ provably bites
  (kernel drift 4.98→0.011) and zeroes A's forgetting when capacity is starved.
  But λ ONLY matters when the perception organ is load-bearing; a roomy front-end
  doesn't forget (drift is harmless). Forgetting lives in the readout (manual §7).
- **ARCHITECTURE / DISK** (asked s043, shifted config): ~6,200 cells, ~294K edges
  across the two arenas, but only **32 distinct kernels = 1,696 floats = 6.6 KB**
  of real parameters (weight-tying; the 294K wires are deterministic structure,
  regenerable from `tile_patches`, need not ship). Naive full-arena ship = ~5.0 MB
  (edge connectivity). The genuine disk cost is the BACK-END: full-cov Mahalanobis
  = **8.1 MB** (10×456² covariances); diag-cov = **36 KB** for ~same accuracy.
  "Big network, tiny model."

## NEXT (priority — for the NEW session)

1. **Push the shifted-MNIST win further.** (a) Global / translation-INVARIANT
   pooling (S1_POOL=S2_POOL=1 via env) — the current pool 5/2 still leaks coarse
   position; full invariance should lift S1 above 0.674 and is the honest readout
   for a moving object. (b) Try a 3rd order or λ2>λ1 bank (S2-beats-S1 suggests
   the cascade has more to give under translation). Both via `shifted_scatter_deep.py`.
2. **Finish the living-trioron bridge.** (a) Wire **u (utility)** + credit-based
   locking so the organ self-throttles which kernels lock — the 3rd node variable.
   (b) Make the LEARNABLE kernels DEEP (the s043 S1⊕S2 is fixed-Gabor; learn both
   layers end-to-end with λ on both). (c) Multi-task (>2) sequence to show λ
   compounding, and compare soft-λ vs hard credit-freeze (manual §1: soft λ lets
   the base keep improving — does it here?).
3. **Promote** the now-validated front-end: `ScatteringLens` = receptor field →
   tied conv cohort (fixed OR learnable) → modulus → pool → (depthwise L2) →
   ManifoldArchive, in `trioron/`. Ship diag-cov back-end (36 KB) not full-cov
   (8 MB). Generalize `spawn_conv_cohort` first.
4. **Tie to PCLL / chained-15** — the original motive: a conv front-end for the
   chained-15 organism. The S1⊕S2 descriptor is exactly that; test on chained-15.
5. **Push shifted** (global pooling NEXT#1a above) and **sweep Gabor banks**
   (K1/σ/freq unswept). WHEN/DEPTH optics toys still untested (s042 carries).

## OPEN / unresolved

- All numbers standard MNIST (centered + shifted), n=3 sampling seeds, fixed
  kernels (no kernel randomness — σ is sampling noise only).
- Pooling 5/2 retains coarse position; NOT fully translation-invariant. The
  shifted win is therefore a LOWER bound on what global pooling would show
  (NEXT#1a). pool=2 on L2 keeps S2 dim (256) under PER_TRAIN so Mahalanobis
  conditions; a finer pool would raise dim past n.
- Depthwise-only L2 by design; cross-channel 2nd-layer conv (s041's form) is the
  untested alternative but is unprincipled with FIXED Gabor kernels.
- s039/s041/s042 carries (untouched): gradient-free conv on chained-15;
  over-segmentation (cap=128); generative mean-template readout (−0.32); WHAT/
  WHERE/DEPTH/Fresnel optics toys; tabular receptor (phase-code crutches weak
  back-ends only).

## State of the build

- Branch `progenitor-council`. Package code UNTOUCHED; gate battery green (s036).
  This session: `fde74ab` (centered scatter), `766e62e` (shifted), `418cc59`
  (learnable+λ), + handoffs. DO-NOT-COMMIT carries left alone. PNGs uncommitted.
- Run cost: centered scatter ~18s/seed, shifted ~63s/seed, learnable+λ ~23s/seed.

## Pointers

- **The win:** `experiments/progenitor/mnist_scatter_deep.py` (centered) +
  `shifted_scatter_deep.py` (shifted, the stronger demo). S1⊕S2 = Mallat
  scattering; `scatter()` returns (S1, S2); `modulus_maps()` = the oriented-energy
  (cos²+sin²)^½ per Gabor pair; depthwise L2 loops over U1 channels. The shifted
  file reuses the centered file's pure leaf-helpers + a local `shift_onto_canvas`.
- **The s042 single-layer win it builds on:** `mnist_conv_fixed.py` (fixed Gabor
  via `conv.forward_batch`, 0.907/0.980 vs raw 0.815).
- **The living-trioron bridge:** `scatter_learnable_lambda.py`. `build_kernels`
  (learnable cohorts, warm-start Gabor), `descriptor` (modulus+pool, eps in sqrt),
  `lock_in` (anchor + saliency→λ), `train_task` (CE + strength·ewc_penalty). The
  λ API lives in `trioron/learning/epigenetic_lock.py` (anchor / accumulate_saliency
  / refresh_lambda / ewc_penalty — read its docstring: λ is per-cell row-sum, native
  driver |w·g|).
- **The s041 Adam conv-depth reference (the form made gradient-free here):**
  `conv_depth_shifted_mnist.py` — shifted-MNIST, `tile_patches_mc` (cross-channel
  L2), `spawn_cohort`, `build_l1`/`build_l2`. Use its shifted-canvas data loader
  for shifted work.
- **Wiring helpers:** `conv_proposer.tile_patches` / `_bucket_for` /
  `spawn_conv_cohort`; `mnist_conv_fixed.spawn_fixed_cohort` / `conv_map`.
- **Conv primitive:** `trioron/phenotype/conv.py` (lines 1–24: parameter sharing =
  convolution via `lineage_root`; reduction guarantee = "no sharing → plain linear").
- **Back-end:** `trioron/learning/manifold.py` — `ManifoldArchive(arena,
  full_cov=True)`, `log_likelihood_full` (Mahalanobis). `taxonomy_manifold.
  classify_archive(Dtr, ytr, Dte, n_class, full_cov=True)` is the one-call wrapper.
- **Data:** `DatasetBundle(["mnist"]).task_view("mnist", range(10), range(10),
  split=…).all_examples()` → ([N,784] in [0,1], labels).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`),
  `OMP_NUM_THREADS=8`. Data `outputs/data/`.
- **Model note:** lead with the COMPUTATIONAL framing in code/docs (phase code;
  tied conv kernel via `lineage_root`; Gabor filterbank; modulus = oriented
  energy; scattering transform = |·|*ψ cascade; Mahalanobis fingerprint). The
  optical metaphors (lens/emitter/phasor/focus) are Rocky's conversational
  framing — keep them out of code to avoid mid-session safety downgrades.
