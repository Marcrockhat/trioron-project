# Trioron Handoff

**Session date:** 2026-06-15
**Session number:** 039
**Session title:** **Receptor diagnostics — the keystone is the READOUT/
MECHANISM, proven three ways. (1) Centered matched-filter readout lifts the
gabor organism to 0.736/0.446 (from s038 0.690/0.304). (2) KNN proves the
class info is FULLY in the quanta (0.83–0.88 ≈ raw pixels) — so the gap is
downstream, not perception. (3) Rocky's two receptor fixes tested: |center-
surround| quanta give PERFECT inversion invariance (0.782=0.782), and a
neighbour-joining tree over the dimensions RECOVERS the 2D image topology
from data alone (tree-nearest leaf 1.58 px, Spearman +0.39) — BUT tree-as-
appended-features is NULL (random groupings match/beat it), because the
matched filter is permutation-invariant over features and literally cannot
see topology. The tree must enter the DIVISION/COUNCIL machinery, not the
feature vector.** Checkpoint commit; the architectural build (tree-into-
division) starts fresh next session.**

---

## READ THIS FIRST

1. **This was a DIAGNOSTIC session — no organism/package code changed**
   except the readout A/B in the runner. Four new probes under
   `experiments/progenitor/`. The findings redirect the work; act on them
   before building.
2. **The keystone is now triangulated and it is NOT perception.** KNN on
   the raw pocket-integer vector hits **0.826** (≈ raw pixels 0.834,
   30-way, chance 0.033). The class information is fully preserved in the
   quanta. Every loss is downstream: encoding-to-phasor, then the
   generative mean-template readout, then division/name-majority. On the
   headline GABOR sense the phasor encoding is faithful (0.881→0.874) and
   the dominant single loss is the **mean-template oracle collapse**
   (KNN 0.874 → oracle 0.553, **−0.32**) — i.e. NEXT is a discriminative/
   distributional readout, with a measured ceiling of ~0.87.
3. **Rocky's design (this session's core):** the receptor is spatially
   blind (flat bag-of-pixels; position survives only as an index, never as
   a patch) and polarity-bound (inverting the image drops raw KNN
   0.834→0.023, below chance). His two fixes:
   - **|center-surround| receptor** (a pixel vs its neighbours, |LoG|) →
     inversion invariance. **CONFIRMED:** cs-leaves 0.782 normal = 0.782
     inverted, bit-exact, and a better base than raw intensity (+0.15).
   - **neighbour-joining tree over the dimensions** → patches/hierarchy,
     modality-agnostic (1D/2D/3D — NJ needs only a distance matrix).
     **Topology recovery CONFIRMED** (1.58 px), but **tree-as-features
     NULL** (see #4). The tree is valid; the *consumption* was wrong.
4. **The decisive negative result:** appending tree-aggregated patch
   features to the matched-filter/KNN readout does NOT help — random
   groupings of the same sizes match or beat the tree (raw: random 0.767
   > tree 0.730; cs: tree 0.798 ≈ random 0.785). **Why:** cosine /
   matched-filter sums over features → permutation-invariant → it cannot
   see feature grouping at all, and tree-local pooling averages correlated
   (redundant) pixels while random pooling averages diverse ones. So the
   tree can only pay off where the mechanism processes features LOCALLY:
   **division splitting on patches instead of per-pixel** (the cause of
   the cap=128 over-segmentation), or substrate connectivity / weight-
   sharing. This is Rocky's concern #2 ("fix the council and architecture").
5. **DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM):**
   `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
   `trioron/viz/export.py`; `.claude/`, `runs/` untracked. NOT staged.
   Two **stalled** raw runs left untracked on disk
   (`outputs/pcll_chained15_s039_raw_centered.log`,
   `..._raw_nocomp_centered.log`) — they crawled (task 6 = 5449s, task 7 =
   20893s) from **CPU contention** (both launched concurrently), never
   finished, and are NOT committed. Re-run raw+centered ALONE if the A/B
   baseline is wanted.

---

## What was built (4 probes + 1 readout A/B)

- **`run_pcll_chained15.py`** — `evidence()` → `evidence_both()`: computes
  the **plain** matched filter AND a **centered** one (subtract the mean
  template μ = common-mode phasor) in one forward pass; `_score()` helper;
  `run_seed` returns the centered numbers. Readout-only A/B on the
  identical trained organism (no retrain). Result (gabor, seed 0):
  PLAIN 0.712/0.385, **CENTERED 0.736/0.446** (Δ full +0.061). Beats the
  s038 committed gabor 0.690/0.304. *(See OPEN: plain ≠ s038 plain.)*
- **`diag_s039_knn.py`** — KNN encoding probe. Representations: raw,
  quanta, phasor `exp(iθ)`, halfcirc `exp(iπq/N)`, centered; `S039_GABOR`,
  `S039_INVERT_TEST`. The keystone-triangulation tool.
- **`diag_s039_bands.py`** — band-offset (per-dimension offset circles)
  encoding probe. **FALSIFIED** (hand-coded spatial separation doesn't
  recover; the real superposition is the generative-readout common mode).
  Superseded by the NJ tree (the data-driven version).
- **`diag_s039_njtree.py`** — neighbour-joining a tree over the 784 pixel
  dimensions from **co-quantum agreement** distance; validates topology
  recovery vs the known 2D grid. `S039_METRIC=active` (background-
  conditioned, the working metric) | `raw`.
- **`diag_s039_patchtree.py`** — the COMBINED proposal end-to-end:
  |LoG| leaves + NJ tree clades as patch features, KNN normal vs inverted,
  with the random-grouping control that produced the null.

## The numbers

**KNN keystone (30-way, chance 0.033, k=1):**

| rep | raw normal | raw INVERTED | gabor normal | gabor INVERTED |
|---|---|---|---|---|
| raw cosine | 0.834 | 0.023 | 0.895 | 0.895 |
| quanta | 0.826 | 0.322 | 0.881 | 0.881 |
| phasor | 0.628 | 0.372 | 0.874 | 0.874 |
| halfcirc | 0.732 | 0.171 | 0.872 | 0.872 |

- Info is in the quanta (0.826≈0.834). Raw phasor loses 0.20 to the
  **circular wrap** (raw pixels bimodal → q=0/q=1000 antipodal); the
  half-circle `θ=πq/N` recovers +0.10, confirming the wrap is the cause.
- Gabor energy is phasor-COMPATIBLE (0.881→0.874, −0.007) AND inversion-
  invariant (bit-identical) — a *new* reason gabor wins, not in any prior
  handoff.

**NJ topology recovery (raw pixels, active-conditioned metric):**
Spearman(tree, grid) **+0.389** (shuffled control +0.025); tree-nearest
leaf **1.58 px** (random ~13.5). A tree built from data alone, given no
coordinates, reconstructs the image lattice — modality-agnostically.

**Combined patch-tree (phasor KNN, 30-way):**

| rep | normal | inverted |
|---|---|---|
| raw-leaves | 0.628 | 0.372 |
| raw + tree patch | 0.730 | 0.246 |
| raw + **random** patch | **0.767** | — |
| cs-leaves (\|LoG\|) | 0.782 | **0.782** |
| cs + tree patch | 0.798 | 0.798 |
| cs + **random** patch | 0.785 | — |

cs = perfect inversion invariance + better base. Tree patches ≤ random
patches → the lift is generic pooling, not topology (the null).

## NEXT (priority order)

1. **Tree-into-division (highest leverage, Rocky's concern #2).** Make
   `division.try_divide` split on tree-clades / patch-units instead of
   per-dimension, and measure whether it stops over-segmenting (cap=128
   from task 1) and lifts full toward the 0.55 oracle / 0.87 KNN ceiling.
   This is the ONLY place the validated tree can pay off (the readout is
   permutation-invariant). Build the NJ tree at genesis from the receptor
   stream, freeze it (or re-estimate per CL-task — open design question).
2. **Adopt the |center-surround| receptor.** Wire `|LoG|` (reflect-pad,
   it IS the s038 invariance fix) as a new `PCLL15_SENSE=cs` and run the
   full organism normal + inverted. Banks the confirmed inversion fix and
   the +0.15 base over raw intensity, independent of the tree work.
3. **Discriminative readout (the −0.32).** Wire the manifold full-cov /
   H-routing readout (manual §5.5, `bench_chained_15_v2.py`) — the
   measured gap from generative mean-template (0.553) to discriminative
   (KNN 0.874). Centering (this session) is a +0.06 down-payment on it.
4. **Modality-agnostic proof.** Run the NJ recovery on a 1D signal to
   demonstrate topology discovery off the 2D grid (Rocky's generality
   claim; cheap, strengthens the paper story).
5. **Oriented center-surround.** cs uses one isotropic |LoG| (0.782) <
   gabor's 8 oriented energy channels (0.874). An oriented |center-
   surround| bank would close that, still inversion-invariant.

## OPEN / unresolved

- **Plain gabor discrepancy:** s039 `evidence_both` PLAIN = 0.712/0.385,
  but s038 committed PLAIN = 0.690/0.304 (same seed, same readout). Must
  be explained before trusting the 0.736/0.446 headline. Candidates:
  reflect-pad fix (9fbf59b) changing gabor after the s038 headline was
  recorded; eval-set ordering in `evidence_both`; run nondeterminism.
- **NJ scaling:** O(N³), fine for 784, dead at the 1.5 Mi working-
  resolution target — a local/approximate NJ is needed before vision.

## State of the build

- Branch `progenitor-council`. This commit adds the 4 `diag_s039_*.py`
  probes + the `evidence_both` centered-readout A/B in
  `run_pcll_chained15.py` + `outputs/pcll_chained15_s039_gabor_centered.log`
  + this handoff. No package code touched; gate battery untouched (green
  from s036, not re-run).
- `developmental.py` ×2 + `viz/export.py` carries NOT staged (s034 carry).
- Two stalled raw logs left untracked (contention; not committed).
- Run cost: KNN/NJ probes seconds–minutes each; gabor-centered organism
  ~3239s seed 0 CPU OMP_NUM_THREADS=8.

## Pointers

- **Keystone tool:** `experiments/progenitor/diag_s039_knn.py` (raw/quanta/
  phasor/halfcirc/centered; `S039_GABOR`, `S039_INVERT_TEST`).
- **Topology tool:** `diag_s039_njtree.py` (co-quantum NJ vs grid).
- **Combined proposal + the null:** `diag_s039_patchtree.py`
  (`center_surround` |LoG| reflect-pad, `neighbour_join_clades`,
  random-grouping control).
- **Division to modify for NEXT#1:** `trioron/pcll/division.py`
  (`try_divide`, per-dimension bimodality, `NULL_SPLIT=0.72`).
- **Readout for NEXT#3:** `experiments/bench_chained_15_v2.py`
  (`--full-cov`, H-routing); `trioron/learning/manifold.py`.
- **Encoding internals:** `trioron/core/receptor.py` (`quantize`, per-sample
  global lo/hi — the spatial-blindness root); `trioron/pcll/mixed.py`
  (`pockets_of` 1316, `_evidence` 445, `templates` 1297).
- **The gabor base + senses:** `run_pcll_chained15.py` (`make_sense`
  dispatches on the `SENSE` env global — NOT its arg; `_build_gabor_sense`).
- **Design:** `docs/design/progenitor_council.md:66` (center-surround
  positional cells), `docs/design/pcll_substrate_integration.md:131`.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, scipy 1.15.3, WSL2, `python3` (NOT
  `python`), `OMP_NUM_THREADS=8`. Data at `outputs/data/`.
- bench-15 = 30 true global classes, 15 tasks (MNIST/Fashion/EMNIST-letters
  pairs), single gradient-free pass per task.
- **Model note:** lead with the computational/ML framing (frequency,
  convolution, matched filter, energy = magnitude of a quadrature pair,
  neighbour-joining = phylogenetic clustering on a distance matrix) — the
  biological metaphors can trip mid-session safety downgrades.
