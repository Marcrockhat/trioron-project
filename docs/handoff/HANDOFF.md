# Trioron Handoff

**Session date:** 2026-06-14
**Session number:** 037
**Session title:** **PCLL keystone diagnostic dive — the s036 keystone is
REFRAMED. The features carry class (raw-pixel nearest-centroid 0.671 on
30-way); it is the phasor matched-filter READOUT/encoding that loses it
(0.249 on raw). A long ruling-out of data-side fixes (half-period,
black/grey/white sentinels, pedestal, un-masking, binarization) all
confirm the background MASK is load-bearing on images. The breakthrough
direction (Rocky's insight: light is frequency, compress spatial info
into angles) is the FREQUENCY / localized-Gabor encoding: global fft-PHASE
hits 0.493 and fills the bands 99.4% interior (2× the pixel-bag); a
complex quadrature GABOR front-end hits 0.597 (phase) and is shift-robust,
with the ENERGY channel BOTH background-swap-invariant (bit-exact under
inversion) AND shift-tolerant at 0.553. Converged perception design:
complex quadrature Gabor bank, low/mid scales, pooled → energy
(invariant) + local phase (accuracy) into the phasor matched filter. No
package code changed; all measurements via one diagnostic script.**

---

## READ THIS FIRST

1. **All numbers this session are ORACLE upper bounds, not full-organism
   results.** The diagnostic (`experiments/progenitor/diag_s037_control.py`)
   builds per-TRUE-class mean-phasor templates (perfect, un-fragmented
   clustering) and scores nearest-template on held-out test, 30-way,
   n=15000/split, chance 0.033. This isolates the ENCODING from the
   council/division machinery. It is NOT the streaming organism — do not
   quote these as bench-15 results. The full-pipeline number remains
   s034/s036's 0.552/0.172.
2. **The s036 keystone is REFRAMED (this is the headline).** s036 said
   "the encoding doesn't carry class identity." That is too strong about
   the *features*: a plain cosine nearest-centroid on raw pixels gets
   **0.671** on 30-way. The class signal is there. What loses it is the
   **phasor matched-filter readout**: 0.671 → 0.249. So the gap is the
   gradient-free phasor representation, not the input and not the
   per-sample frame (global frame = identical, 0.119/0.119 unmasked).
3. **Rocky's standing criterion (binding):** the right encoding must be
   invariant to BACKGROUND SWAP — "a human reads the digit even if you
   swap the background." This is why half-period was REJECTED despite
   scoring 0.464 (it is the LEAST invariant: inverted-test 0.008). Any
   future encoding is judged on inverted-test, not just fixed-polarity
   accuracy.
4. **DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM):**
   `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
   `trioron/viz/export.py`; `.claude/`, `runs/` untracked. This session's
   commit adds ONLY `experiments/progenitor/diag_s037_control.py` + this
   handoff. No package code touched.

---

## The result chain (read in order)

All accuracies are the oracle matched filter, 30-way, masked (q∈{0,1000}
dropped) unless noted. Baselines: chance 0.033, raw-pixel nearest-centroid
(cosine) **0.671** = the "features carry class" reference ceiling.

1. **Raw pixels through the phasor: 0.249** (masked; 0.119 unmasked — the
   q=0 background and q=1000 stroke COLLIDE at phasor +1 under the full
   2π period). Templates 89% identical. This reproduces the keystone.
2. **Discriminative features lift the SAME filter 2-4×:** gabor-relu
   0.489, cortex(MobileNetV3-S, frozen) 0.453. So the filter is sound on
   good features — keystone's direction confirmed.
3. **Per-sample vs global frame: identical (0.119).** The per-sample
   normalization is NOT the culprit.
4. **Half-period [0,π] → 0.464** (removes the wrap collision: silent +1,
   saturated −1). REJECTED by Rocky: inverted-test 0.008 = least invariant
   of all — it overfits fixed polarity.
5. **Sentinel/noise axis fully characterized (Rocky's black/grey/white
   "noise" idea, coded as gain-reference COLUMNS mirroring
   `run_m4_composer.sentinel`):**
   - **white** (just above ceiling, pins max): **+0.05 (0.249→0.298)** —
     the one win; stops images falsely saturating their own peak. KEEP.
   - **black** (just below floor, pins min): **−0.13 (→0.121)**. VERIFIED
     it fires exactly as designed (background val-0 → q=1, mask removed,
     evidence features 99→782/img) — and it hurts BECAUSE it works: it
     pulls the SHARED background into evidence, so every class template
     gets identical +1 there → templates alike. Un-masking shared
     background is wrong on images.
   - **grey** (mid-range): null (0.000) — frame-neutral, doesn't move
     lo/hi.
   - Pedestal, mask-q0-only: also hurt. The background mask is
     load-bearing; the background is the stable shared lower extreme.
6. **MNIST is NOT binary** (Rocky's test: binary ⇒ two peaks, 0 sd). It
   has **256 distinct values, sd 0.30, 82.7% at EXACTLY 0**, ~7% near 1,
   and a **flat ~10% continuous bridge** across the middle (anti-aliasing).
   So the continuous quantizer is the design-correct path; the level
   census (`K_DISCRETE=8`) correctly calls it continuous. Binarizing
   (threshold) IMPOSES binary structure → 250/750 labeled lines → 0.472
   (≈2× raw) but DISCARDS the grayscale bridge. Binary labeled lines are
   also polarity-sensitive (inverted-test 0.017).
7. **Canonical/stream frame stabilization (D10): confirmed real, inert
   here.** `controller.py` accumulates running E[lo],E[hi]; `mixed._canonical`
   re-expresses per-sample pockets in the running EXTREMES. But every
   chained-15 image spans [0,1] (lo_s=0 always, hi_s≈1 always, 0% < 0.99),
   so the canonical frame = identity and the matched filter is unchanged
   (0.249 → 0.262). Stabilization can't lift the background off q=0
   because 0 is the genuine shared floor of every image.
8. **THE DIRECTION — frequency / phase (Rocky's insight):**
   - **fft-PHASE = 0.493** (2× pixel-bag), fills the bands **99.4%
     interior** (vs 82% silent for pixels). fft-magnitude 0.376
     (energy, position-blind). amplitude-weighted complex 0.075 (DC
     swamps). ⇒ use UNIT phasors of the phase; phase > magnitude.
   - **Low+mid frequency is the signal AND compressible:** 79 of 420
     radial bands (<0.25) give **0.518 — BEATING all 420 (0.493)**. High
     freq is noise.
   - **Global fft-phase is shift-FRAGILE:** (0,0) 0.493 → (1,1) 0.232 →
     (2,2) **0.039**. A shift is a global linear phase ramp.
   - Discriminative ceiling on phase feats [cos,sin] with a trained
     softmax: test **0.572** (matched filter leaves ~0.08 on the table).
9. **Localized GABOR demonstration (closes the loop):** complex
   quadrature bank (even+odd, λ∈{4,8}, 4 orientations, reflect-pad),
   pooled 4×. Shift test vs global fft-phase:

   | shift | fft-phase | Gabor-phase | Gabor-energy |
   |---|---|---|---|
   | (0,0) | 0.493 | **0.597** | 0.553 |
   | (1,1) | 0.232 | 0.506 | 0.501 |
   | (2,2) | 0.039 | 0.254 | **0.334** |
   | inverted | 0.001 | 0.000 | **0.553 (bit-exact)** |

   Localization restores shift-tolerance (pooling = the CNN mechanism).
   **Gabor-energy gives BOTH invariances at once** (background-swap exact
   + shift-tolerant), Gabor-phase has the higher ceiling but is
   polarity-sensitive. Limit: degrades past ~3px (pool is 4px) → wants
   multi-scale.

---

## The converged perception design (what NEXT builds)

**A complex quadrature Gabor bank at low/mid scales, pooled, emitting two
channels into the phasor matched filter:**
- **energy** `|conv_even + i·conv_odd|` → invariant + shift-robust workhorse;
- **local phase** `angle(conv_even + i·conv_odd)` → high-accuracy, structure.

This is the windowed-Fourier / fixed-CNN-first-layer / space-frequency
middle ground. It is the convergence point of every probe: it fills the
bands (phase), is band-limited to the informative low/mid scales, is
shift-tolerant (pooling), and the energy channel is exactly background-swap
invariant. Locality MUST live in this perception front-end — the substrate
is a flat edge list with no spatial memory (`conv_by_emergence_null`,
`substrate_no_spatial_memory`).

Conceptual frame (the unifying axis): **space-frequency uncertainty.**
pixels = all space/no frequency (bag, 0.249); global FFT = all
frequency/no space (shift-fragile); **Gabor = the tunable middle.**

---

## NEXT (priority order)

1. **Build the Gabor front-end into the chained-15 runner**
   (`run_pcll_chained15.py`, as a `PCLL15_SENSE=gabor` arm) and measure
   the FULL STREAMING organism (not the oracle proxy) — task-aware/full
   vs the s034/s036 0.552/0.172. Use energy + phase channels. This is the
   first real test that the diagnostic's gains survive the council/division
   machinery.
2. **Multi-scale Gabor** (coarser pools at coarser scales) to extend
   shift-tolerance past 3px, the CNN-depth analogue.
3. **Combine phase + energy** in one representation (energy for
   invariance, phase for accuracy) and re-measure both axes.
4. **Re-examine the readout gap:** trained softmax on phase feats hit
   0.572 vs matched filter 0.493 — the generative mean-phasor template
   leaves ~0.08. Relates to the standing full-cov / manifold readout pump
   (`manifold_phase_c_anchored_readout`). Don't re-litigate; note it.
5. The keystone "discriminative perception" route (s036 NEXT#1) is now
   ANSWERED in principle: the right perception is localized oriented
   frequency (Gabor), not raw pixels and not a random conv. The remaining
   open s036 items (wire ongoing sensation growth into MixedStreamController;
   division self-arrest on dense surfaces) still stand but are downstream
   of getting perception right.

---

## State of the build

- Branch `progenitor-council`. This session's commit adds ONLY
  `experiments/progenitor/diag_s037_control.py` + this handoff. No package
  code changed; gate battery untouched (still green from s036; not re-run).
- The diagnostic is self-contained (reuses `core.receptor.quantize`,
  `legacy.donorkit.datasets`, `legacy.senses.cortex`). cortex needs
  torchvision MobileNetV3-S weights (downloaded/cached this session).
- developmental.py carries deliberately NOT staged (s034 carry, unchanged).

---

## Pointers

- **The diagnostic + all this session's experiments:**
  `experiments/progenitor/diag_s037_control.py` (transforms raw/gabor/
  cortex; helpers `balanced_slice`, `oracle_templates`, `evidence`,
  `centroid_acc`, `phasor`/`phasor_global`/`phasor_masked`). The Gabor
  quadrature bank, fft-phase, sentinel, and shift tests were run as inline
  `python3 -` snippets importing this module — re-author from this handoff
  if needed (not all saved as files).
- **Encoding internals:** `trioron/core/receptor.py` (per-sample quantize
  + the q=1000·(2j+1)/(2k) discrete labeled-line formula is in
  `scheduler.py:291`); `trioron/pcll/labels.py` (label = lock-in
  reference channel: deposit value-phasor into a per-label complex tap,
  demodulate = mean phasor; write-only); `trioron/pcll/mixed.py`
  `_evidence`/`_canonical`/`templates`; `trioron/pcll/controller.py:140`
  (frame registry D10).
- **Discriminative front-end already in repo:** `trioron/legacy/senses/
  cortex.py` (frozen MobileNetV3-S, 64-d projection).
- **Design:** `docs/design/progenitor_council.md`; spec §10.2 (receptor),
  §10.3 (mask rule), §10.5 (canonical frame).

---

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`),
  `OMP_NUM_THREADS=8`. Data at `outputs/data/`.
- Diagnostic runtimes: full table ~45s (cortex backbone dominates ~40s);
  fft/gabor/sentinel snippets ~3-10s each. The oracle harness is fast
  because it skips the council loop entirely.
- **Model note:** Fable 5 with biological-metaphor safety measures; lead
  with the computational/ML framing (frequency, convolution, matched
  filter) to avoid mid-session downgrades. bench-15 = 30 true global
  classes, 168,000 train samples; this session used a balanced
  500/class slice.
