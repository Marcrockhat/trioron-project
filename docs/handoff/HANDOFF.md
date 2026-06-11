# Trioron Handoff

**Session date:** 2026-06-11
**Session number:** 031
**Session title:** **REGIME DECISION (Rocky): mixed-stream — DIVISION IS HOW
CLASSES GET DISCOVERED. Division-as-discovery validated standalone (one blurred
world-class → 75 classes, one-shot 0.386→0.711, self-arresting growth);
composer-phenotype growth (NEXT 6) de-risked with four catalogued failure
modes; target set = the Bayes ceiling (0.993 on clean classes).**

---

## READ THIS FIRST

1. This session is ANALYSIS + STANDALONE DE-RISK ONLY — `trioron/` is
   untouched; everything lives in four probe runners under
   `experiments/progenitor/` (committed) and `runs/*.out` (untracked).
2. s030's I1–I5 integration (spec §10, `trioron/pcll/`) is still the current
   substrate state — see the s030 handoff in git history (`d3b5558`).
3. The s030 **class-sequential developmental flow is superseded by Rocky's
   decision** below; spec §10 has NOT yet been amended. Design doc first.

## The arc (probes, in order)

1. **`run_i5_perclass.py`** — per-class recall breakdown of the I5 one-shot
   eval (gate only printed aggregates). Reproduces gate means exactly
   (0.387/0.386/0.384). Spread is wide and seed-stable: sp007 0.67 … sp016
   0.08, sp026 0.10. The weak classes are the mode-blurred ones (one template
   averages 3 dispersed modes → points at empty space).
2. **`run_i5_diag.py`** — two findings:
   - **Bayes ceiling on the scored (27 clean) population is 0.993**, not the
     headline 0.937 (that includes disruptors). Rocky set the target: go after
     the ceiling. Two deficits: unimodal templates (~0.38→~0.62) AND the raw
     equal-weight readout (~0.62→0.99 needs coherence/σ weighting).
   - **FRUSTRATED fires only on the 5 disruptors** (14 periods, epoch 2), never
     on mode-blurred classes — frustration is period-level (500-sample
     aggregate), mode blur is sample-level. The 14 council decide() signals had
     no consumer. **Vote book false credit:** sensation paid to floor (4→1),
     discrimination 24→27, but ALL phenotype groups flat at 4.500 — settlement
     judged "stress gone next boundary" while the WORLD changed under it.
     Integration requirement: per-class settlement attribution.
   - Per-class signature coherence mean|T| correlates with one-shot acc at
     **r=0.79** — the division trigger metric already lives in the signature.
3. **ROCKY DECISION (supersedes s030 flow):** the mixed-stream regime.
   Trioron is a realtime continual-learning organism for device embedding;
   genesis births ONE blurred world-class and **division-by-frustration is the
   class-discovery mechanism**, not a refinement. Also: "1 period = 1000
   quanta" — period length in stream samples (W=1000), distinct from the
   N_QUANTA=1000 pocket resolution.
4. **`run_mixed_division.py`** — division-as-discovery VALIDATED. Shuffled
   16K-sample stream, no labels/boundaries/gradients, windows of 1000:
   - 62–76 classes from divisions, 29–32/32 truths covered, one-shot
     **0.711 mean vs 0.386** class-sequential (above the 0.62 mode estimate).
   - Trajectory (seed 0): exponential 1→2→4→8→16→31→52→68 over periods 2–8,
     refinement to 75 by p11, then **self-arrest — 0 divisions p12–16 from the
     gain criterion alone**. Data-rate floor: a mode needs MIN_CHILD=25
     recurrences; plateau ≈ p10 is near-optimal.
   - **Freeze-nodes-keep-weights: plateau ~0.75** (peak p40–48; +0.02 only
     from weight refinement; seed 2 froze under-divided at 62 → 0.669 stuck).
     Accuracy lives in the structure, not the weights. Slight late decay →
     post-quiescence templates should anneal.
   - Design lessons (v0 falsified): split-acceptance must be PER-DIM (a global
     12-dim mean gain can never fire, ~gain/12); judge per-class ROLLING
     BUFFERS, not window members (windows starve once classes multiply).
5. **`run_composer_growth.py`** — composer arm (NEXT 6) de-risk on a 6-class
   relational testbed (A: f1=f0, B: f1=1−f0 — identical marginals; C/D axis
   blobs; E/F concentric rings r=.40/.22; 2 noise dims). Division-only is
   structurally stuck: 0.595 mean, A/B/E/F confounded, blobs 0.99.
   **Mechanism loop validated:** frustration → trial (overproduce one candidate
   per phenotype × wiring over dim pairs) → spawn best → settle (division-uses-
   dim transfers a vote to the phenotype group, step4 semantics) → prune on
   PATIENCE. NOT yet a perf win (0.561 vs 0.595) — seed 0 wires noise pairs.

   **CORRECTION (s031, caught by Rocky):** the probe's candidate set
   {linear, quad, tanh, gcos, sinc, tan} is NOT the genome. The real
   expression genes are **LINEAR / ATTENTION / CONV / RECURRENT / DENDRITE /
   TANH** (`epigenome.py:28` — the council's 24 seats). Only linear and tanh
   match; the probe's "quad" is really the DENDRITE gene's branch
   nonlinearity σ(z)=z+z² (legitimate only in its `dendritic` wiring form);
   gcos/sinc/tan are probe inventions with **no consumer seat in the real
   vote book**. Therefore: (a) the mechanism loop (trial → permutation-null →
   split-half → spawn → settle → prune) is de-risked, but (b) the
   "book finally differentiates" claim does NOT transfer to the 28-vote book,
   and (c) integration must decide what an attention/conv/recurrent composer
   candidate even is — or restrict trials to the scalar-composable genes
   (linear, tanh, dendrite-quad). Also: `apply_dim` min-max normalizes over
   the BATCH — non-causal; a real spawned dim must emit a value the
   per-sample contrast quantizer (spec §10.2) can handle alone, which
   interacts with failure mode 2 below. Spawn log lacks the wiring field
   (sum/diff/dendritic indistinguishable in `runs/composer_growth.out`).

   **Four failure modes, in depth order:**
   1. NULL-SPLIT floor (fixed): halving any continuous dim yields 2/π=0.64
      coherence; division must beat the noise-slicing null (0.72) or the
      organism tiles instead of frustrating (s021's trigger lesson again).
   2. False coherence (fixed): normalize+compress makes ANY composition look
      coherent (lockin.py saturation lesson) → score by PERMUTATION-NULL-
      corrected gain (shuffle col j vs col i: marginals survive, relation dies).
   3. Selection bias (fixed): 84 candidates × sittings cherry-picks null noise
      → select on half the buffer, confirm on the held-out half.
   4. **Membership-induced confabulation (OPEN, the deep one):** buffer members
      are SELECTED by template match, so even noise dims acquire real,
      reproducible structure inside a buffer — split-half can't reject (both
      halves share the selection). Mima in pure form. Proposed fix uses two
      EXISTING pieces: wire candidates only from `perception_importance`
      (Germline computes it; noise has no lock-in margin in any class) and
      settle spawns on FUTURE deposits (D9 next-boundary semantics), never on
      the buffer that proposed them.

## Decisions made (Rocky, s031)

- Target = the Bayes ceiling (0.993 clean); per-class accuracy table is the
  yardstick (sp016/sp026/sp003/sp005/sp020 the bellwethers).
- Mixed-stream regime adopted; division = discovery; composer/dendrite/tanh
  phenotype growth must trigger from frustration — "it adapts, it grows, it
  prunes" is the point of the architecture.
- New class discovery currently makes TEMPLATE ROWS (astrocytes are
  forward_inclusion=False bookkeeping, controller.py:199) — no somas anywhere
  in the PCLL path; the composer arm is what grows real tissue. Growth ladder:
  frustrated → divide (cheap) → division can't clear → spawn winner phenotype
  dim → recruit; void → receptor attach.
- Honest caveat kept on record: data_hard is axis-separable — division +
  weighted readout reach its ceiling without composers; composers are for
  non-axis-separable (relational/ring) structure, hence the separate testbed.

## State of the build

- Branch `progenitor-council`. Committed this session: the four probe runners
  (`run_i5_perclass.py`, `run_i5_diag.py`, `run_mixed_division.py`,
  `run_composer_growth.py`) + this handoff. `trioron/` UNTOUCHED.
- **DO-NOT-COMMIT carries (unchanged):** `trioron/bases/developmental.py`,
  `trioron/lifecycle/developmental.py`, `trioron/viz/export.py`; `.claude/`,
  `runs/` untracked (probe outputs: `i5_perclass.out`, `i5_diag.out`,
  `mixed_division.out`, `composer_growth.out`).
- s030 gate runners + 70-passed/4-pre-existing-failures test state unchanged.

## Open questions / NEXT (priority order)

1. **Integration design doc** (`docs/design/`, new D-register entries): the
   mixed-stream regime — genesis world-class, division consumer for the
   council's discrimination decision (progenitor divides: buffer partition →
   sibling signatures + astrocytes), composer arm with importance-gated wiring
   + future-deposit settlement, **per-class settlement attribution** (fix the
   false-credit confound). Spec §10 amendment after sign-off.
2. Failure mode 4 in the probe: prove importance-gating + future-settlement
   kills the noise spawns and the composer arm beats division-only on the
   relational testbed (rings need the dendritic quad wiring c_i²+c_j²).
   **Redo with the real gene set first** (see CORRECTION above): candidates
   restricted to linear/tanh/dendrite-quad, vote book keyed to the six real
   genes, spawn log records wiring, composed dims per-sample-quantizable.
3. Coherence/σ-weighted readout (raw filter → likelihood-style) — the other
   half of the Bayes gap; keep the raw arm reported alongside (s029
   comparability).
4. Under-division (seed 2: 62 classes, 3 truths uncovered, plateau 0.669) —
   try second-worst dim on rejection; sibling merge for duplicates.
5. Post-quiescence template annealing (slow buffer turnover after divisions
   stop — the late decay in the freeze run).
6. Carried from s030: period segmentation (subsumed by the regime — windows
   ARE the segmentation for now), componential semantics, dense-field
   compression, 4 pre-existing test_v2 failures (bisect someday).

## Pointers

- Probes: `experiments/progenitor/run_{i5_perclass,i5_diag,mixed_division,composer_growth}.py`
- Outputs: `runs/{i5_perclass,i5_diag,mixed_division,composer_growth}.out`
- s030 integration handoff: git show `d3b5558`; spec §10; design docs
  `docs/design/pcll_substrate_integration.md` (D-register),
  `docs/design/receptor_period_frustration.md` (§10 mixture-semantics open
  question — now answered by the regime decision).
- Vote semantics: `trioron/pcll/stress.py` (the 28-vote book, transfer floors,
  habituation walk); `winner_phenotype()` is the unbuilt arm's consumer-to-be.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, WSL2, `python3`, `OMP_NUM_THREADS=8`.
  All probes seconds-fast on CPU.
- Session was analysis/de-risk heavy with long context; clean break point —
  nothing in-flight in the package.
