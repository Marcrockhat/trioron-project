# Trioron Handoff

**Session date:** 2026-06-15
**Session number:** 039
**Session title:** **Perception is NOT the bottleneck — proven by exhausting
the front-end. The keystone is the division + readout.** A full diagnostic
sweep: (1) KNN proves the class info is FULLY in the quanta (0.83–0.88 ≈
raw pixels). (2) Centered matched-filter readout lifts the gabor organism
to 0.736/0.446 (from s038 0.690/0.304). (3) Rocky's receptor ideas, each
built and tested end-to-end: |center-surround| gives PERFECT inversion
invariance (0.782=0.782) but ≤ gabor; a neighbour-joining tree over the
dimensions RECOVERS the 2D image topology from data alone (1.58 px) but
tree-as-features is NULL (permutation-invariant readout can't use it);
t-SNE shows the quantum bands fold 2D position into ~1D; and a 2nd-order
wavelet SCATTERING cascade ("cascade of prisms") ties gabor in KNN, wins a
3-task smoke (+0.024), but **LOSES at full 15-task (0.592/0.368 vs gabor
0.736/0.446)** — its 1176 dims give division more rope to over-segment.
**Every path converges: the flat discriminative ceiling is ~0.87, the
organism realizes 0.37–0.74, and gabor is the front-end ceiling. The gap
is downstream: kill the cap=128 over-segmentation and replace the
generative mean-template readout.**

---

## READ THIS FIRST

1. **DIAGNOSTIC session. The conclusion is a PIVOT: stop tuning
   perception, fix the division/readout.** No package code changed; the
   runner got A/B readout + 4 new senses; 5 new probe scripts.
2. **The keystone, triangulated and final:** KNN on the raw pocket-integer
   vector = **0.826** ≈ raw pixels 0.834 (30-way, chance 0.033). Class info
   is fully in the quanta. On gabor the phasor is faithful (0.881→0.874)
   and the dominant single loss is the **mean-template oracle collapse**
   (KNN 0.874 → oracle 0.553, −0.32) → NEXT is a discriminative readout
   (ceiling ~0.87), and killing over-segmentation.
3. **Perception is exhausted as a lever.** Four front-end ideas tried; NONE
   beats gabor in the full organism:
   - **centered readout** (μ-subtract): +0.06 full on gabor (0.446). KEEP —
     committed, it is the current best.
   - **|center-surround| (|LoG|):** inversion-invariant (0.782=0.782),
     better than raw, but ≤ gabor (which is also invariant).
   - **NJ dimension-tree:** recovers 2D topology (1.58 px, Spearman +0.39)
     but tree-as-appended-features is NULL in BOTH KNN and PCLL (random
     groupings match/beat it — the readout is permutation-invariant).
   - **2nd-order scattering (prism cascade):** ties gabor in KNN, +0.024 on
     a 3-task smoke, but **−0.14 task / −0.08 full at full 15-task.** A
     richer (1176-d) front-end makes the organism WORSE: more spurious
     bimodal dims → division over-segments dirtier. Confirms the keystone.
4. **The mechanism truth that killed tree + scattering as features:** the
   matched filter / KNN sum over the feature axis → **permutation-invariant
   over features** → they CANNOT exploit feature grouping or spatial
   structure. The tree/cascade can only pay off by changing the DIVISION
   ALGORITHM (split on patch-units, or local connectivity), not by adding
   feature dimensions. The richer the bag, the worse division fragments.
5. **DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM):**
   `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
   `trioron/viz/export.py`; `.claude/`, `runs/` untracked. Two **stalled**
   raw runs (`..._s039_raw_centered.log`, `..._raw_nocomp_centered.log`)
   left untracked — CPU contention, never finished, NOT committed.

---

## What was built

**`run_pcll_chained15.py`** (one A/B + four senses, all gradient-free):
- `evidence_both()` / `_score()` — plain AND **centered** (μ-subtracted
  common-mode) matched filter in one pass; `run_seed` returns centered.
- `SENSE=cs` / `cstree` — `|center-surround|` leaves (`_build_cstree_sense`),
  optional NJ patch-dims; tree fit lazily on the genesis window.
- `SENSE=scatter` / `scatter_full` — 2nd-order wavelet scattering
  (`_build_scatter_sense`, `_energy_maps`), path-pruned (λ2>λ1) or full.
  Reuses the gabor quadrature bank; no S0 lowpass (keeps inversion inv.).

**Probes** (`experiments/progenitor/`):
- `diag_s039_knn.py` — KNN encoding probe; `S039_FE=raw|gabor|scatter`,
  `S039_INVERT_TEST`; reps raw/quanta/phasor/halfcirc/centered.
- `diag_s039_njtree.py` — NJ tree over dims (co-quantum dist) vs grid.
- `diag_s039_patchtree.py` — combined |LoG|+tree, KNN + random control.
- `diag_s039_tsne.py` — saves `outputs/diag_s039_tsne.png` (dims by
  co-quantum dist colored by grid; samples by class, raw vs cs).
- `diag_s039_bands.py` — band-offset encoding (FALSIFIED; superseded by NJ).

## The numbers

**KNN keystone (30-way, k=1, NORMAL / INVERTED):**

| rep | raw | gabor | scatter |
|---|---|---|---|
| quanta | 0.826 / 0.322 | 0.881 / 0.881 | 0.885 / 0.885 |
| phasor | 0.628 / 0.372 | 0.874 / 0.874 | 0.875 / 0.875 |

Info is in the quanta. Raw phasor loses 0.20 to the circular wrap (raw is
bimodal); gabor/scatter are phasor-faithful AND inversion-invariant.
scatter ≈ gabor (no KNN lift; the cascade's value is invisible to a
permutation-invariant readout).

**NJ topology recovery:** Spearman(tree,grid) +0.389 (shuffle +0.025);
tree-nearest leaf 1.58 px (random ~13.5). t-SNE: dims form a folded ~1D
manifold, not a 2D sheet → 2D position poorly represented in quantum bands.

**Full 15-task organism, centered (seed 0) — the headline:**

| sense | task-aware | full |
|---|---|---|
| **gabor (BEST)** | **0.736** | **0.446** |
| scatter | 0.592 | 0.368 |
| raw (s034) | 0.552 | 0.172 |

3-task smokes (FRAC=0.3) MISLED: gabor 0.981/0.880, scatter 0.984/0.904,
cs 0.749/0.661, cstree 0.753/0.662 — short streams hide cross-task
interference; do NOT trust 3-task smokes for the 15-task verdict.

## NEXT (priority — all downstream of perception now)

1. **Kill cap=128 over-segmentation (highest leverage).** Every sense tiles
   to 128 on task 1 at FRAC=1.0; 128 fragments for 30 classes is 4× over.
   Make `division.try_divide` self-arrest on dense surfaces, or split on
   patch-units, OR derive the cap. The FRAC knob is the cheap probe
   (fewer meetings → less tiling lifted the smoke).
2. **Discriminative readout (the −0.32).** Replace the generative
   mean-template matched filter with the manifold full-cov / H-routing
   readout (manual §5.5, `bench_chained_15_v2.py`). KNN proves ~0.87 is
   reachable; centering (this session) is a +0.06 down-payment.
3. **Tree as STRUCTURE, not features (Rocky's concern #2).** The ONLY
   untested place the validated NJ tree can help: make division split on
   tree-clades / wire clade-local connectivity. Tree-as-features is dead.
4. **(Lower) Modality-agnostic proof** of NJ on a 1D signal; **oriented
   center-surround** bank; revisit scattering only if div over-segmentation
   is fixed first (its loss was dimension-driven fragmentation).

## OPEN / unresolved

- **Plain gabor discrepancy:** s039 `evidence_both` PLAIN = 0.712/0.385 vs
  s038 committed PLAIN 0.690/0.304 (same seed/readout). Explain before
  fully trusting 0.736/0.446. Likely the reflect-pad fix (9fbf59b) post-
  dating the s038 headline, or eval ordering in `evidence_both`.
- **NJ scaling:** O(N³), fine at 784, dead at the 1.5 Mi vision target.
- All headline numbers are seed 0 only.

## State of the build

- Branch `progenitor-council`. Two commits this session: `24875be`
  (KNN/NJ/patchtree + centered readout) and this one (cs/cstree/scatter
  senses + t-SNE + scatter full-15 log + this handoff). No package code
  touched; gate battery untouched (green from s036).
- Carries NOT staged (developmental ×2, viz/export); stalled raw logs
  untracked.
- Run cost: scatter full-15 = 7958s (~2.2 h, slower than gabor's 3239s —
  1176 dims). KNN/NJ/t-SNE probes seconds–minutes.

## Pointers

- **Front-ends:** `run_pcll_chained15.py` `make_sense` (raw|gabor|scatter|
  scatter_full|cs|cstree|conv|lcn; dispatches on the `SENSE` env GLOBAL,
  not its arg); `_build_scatter_sense`, `_build_cstree_sense`,
  `_build_gabor_sense`, `_energy_maps`, `_gabor_quadrature_bank`.
- **Division to fix for NEXT#1/#3:** `trioron/pcll/division.py`
  (`try_divide`, per-dim bimodality, `NULL_SPLIT=0.72`).
- **Readout for NEXT#2:** `experiments/bench_chained_15_v2.py`
  (`--full-cov`, H-routing); `trioron/learning/manifold.py`.
- **Encoding internals:** `trioron/core/receptor.py` (`quantize`, per-sample
  global lo/hi = the spatial-blindness + bimodal-wrap root);
  `trioron/pcll/mixed.py` (`pockets_of` 1316, `_evidence` 445,
  `templates` 1297).
- **Design:** `docs/design/progenitor_council.md:66` (center-surround
  cells), `docs/design/pcll_substrate_integration.md:131`.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, scipy 1.15.3, sklearn 1.7.2, WSL2,
  `python3` (NOT `python`), `OMP_NUM_THREADS=8`. Data `outputs/data/`.
- bench-15 = 30 global classes, 15 tasks (MNIST/Fashion/EMNIST pairs),
  single gradient-free pass per task. **3-task smokes are NOT predictive.**
- **Model note:** lead with computational framing (frequency, convolution,
  matched filter, quadrature energy, neighbour-joining = phylogenetic
  clustering, scattering transform = cascaded wavelet+modulus) — biological
  metaphors ("eyes", "prism", "cells") can trip mid-session downgrades.
