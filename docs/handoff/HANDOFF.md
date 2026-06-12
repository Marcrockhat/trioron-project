# Trioron Handoff

**Session date:** 2026-06-12
**Session number:** 033
**Session title:** **M3-M6 + D18 + D20 ALL DONE. The gap hunt's final
verdict: self-labeling. D20 triplet (margin membership + consolidation +
gate) ships: 0.818→0.854. D19 refuted at step 0. Sentinel quantum-derived;
M4 re-gated 0.729. OPEN RULING: member_margin default (+0.051 on M2).**

---

## READ THIS FIRST

1. **The governing design doc is `docs/design/mixed_stream_growth.md`** —
   D11–D17 all approved (D11–D16 s032; §4b incl. D17 family trials s033).
   **M1–M5 PASSED and committed; M6 (spec §10 amendment + manual update,
   spec-first rule D6) is the remaining phase.**
2. **All five rebuild features are now wired and demonstrably working**
   (Rocky's trust criterion): progenitor/genesis ✓, council/votes with
   real consumers + gene-targeted economy ✓, frustration gate ✓, PCLL
   with division + composer growth (census MOVES: computing +, edges +,
   depth 0→1) ✓, manifold adapter ✓.

## The arc (s033, one line each — details in design-doc gate rows M3–M5)

1. **M3 [D13] PASS** (`a2d5ee5`): per-class settlement; deferral kills
   false credit; divisions settle vs their own children (0.924).
2. **M4 pre-probe** (`92c26fd`…): genome redo; dendrite identity
   σ(−(c+½)) = c²−¼ at w=−1/(2π); §4b amendments measured + approved
   (ratio statistic, D17 family trials FAMILY_DEGREE=3, any-class
   importance, competition ladder).
3. **M4 [D14+D16+D17] PASS** (`51d732f`): composer spawns are REAL
   tissue (+1 computing cell, +2 edges, rank earned at compile);
   0.694 > division-only 0.624; depth 0→1 every seed; zero noise
   spawns; gene-targeted vote transfers live. TWO core bugs found+fixed:
   construct() empty dispatch table (grown cells silently skipped — the
   s020 inert-growth class) and worst-dim-only division never firing
   with pure-noise dims (`divide_tries`).
4. **data_hard per-class (Rocky's ask)** (`run_m4_hard.py`): ZERO spawns
   (genome regularizer holds on axis-separable frame-moving data); mean
   0.760→0.811 entirely from divide_tries=4 (sp026 +0.46, sp020 +0.30);
   94 vs 84 classes.
5. **M5 [D15] PASS** (`ace421f`): `trioron/pcll/manifold.py` —
   per-class μ/σ sketches over pocket space riding the class astrocyte
   rows; fed real window members each boundary. Consumers: annealing
   (sketch deposits displace the OLDEST buffer members at quiescent
   boundaries — templates re-anchor), σ-likelihood readout (σ floor
   2 pockets), replay + FULL ship/wake (mixed.state_dict rides
   lifecycle/ship's `_pcll` hook: classes, buffers, specs, sketches,
   codec, vote book; wake → byte-identical readout). `mixed.freeze`
   added (deployment mode: no divisions/trials; membership, deposits,
   sketches, annealing continue).
   Gate: σ-readout 0.777 > raw 0.765 (raw arm alongside); anneal end
   0.778 ≥ no-anneal 0.773; round-trip exact.

## M5 falsifications (IMPORTANT, on record in the runner + gate row)

1. **The s031 σ-weighting claim (~0.62→0.99) is dead.** Post-division
   fragments are tight; diagonal σ-readout gives +0.012, full-cov is
   within noise. The residual 0.78→0.99 gap is fragmentation/majority-
   mapping/crossing ambiguity — the ROUTING story (cf. chained-15
   h_space_routing finding), not readout weighting. Closing it needs
   fewer/purer classes (merge consumer, D17's unbuilt half) or a
   routing layer — NOT more readout math.
2. **The s031 freeze-decay does not reproduce** under the corrected
   controller (2 frozen epochs, no decay either arm). Annealing's
   measured value accrues during DISCOVERY. The machinery remains the
   lifetime-horizon guard the scenario was designed for.

## State of the build

- Branch `progenitor-council`, all pushed. s033 commits: `a2d5ee5` (M3),
  `92c26fd`/`594a01e`/`8a3b5c2`/`044884e` (probe+sweeps), `51d732f` (M4),
  `bdd4754` (M4 record + hard), `ace421f` (M5), + handoffs.
- Tests: **100 passed** + the SAME 4 pre-existing failures (credit ×2,
  growth trigger, dream lock — `bisect someday` carry). New this
  session: `test_composer.py` (11), `test_pcll_manifold.py` (6), 3 D13
  stress tests.
- Regressions all green at session end: M2 0.760 / M3 0.924 / M4 0.694
  byte-identical; I3, I5 (0.387/0.386/0.384) byte-identical.
- **`trioron/pcll/mixed.py` is 705 lines** — past the ~500 convention.
  Split candidate for M6: the composer-arm orchestration (~200 lines:
  `_trials`/`_spawn`/`_testimony`/`_prune_composers`/`_feed_fresh`)
  along the concept boundary into the composer module or a small
  `mixed_composer.py`. Behavior-preserving refactor; gates re-run.
- **DO-NOT-COMMIT carries (unchanged, still dirty):**
  `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
  `trioron/viz/export.py`; `.claude/`, `runs/` untracked.

## M6 DONE (s033) + the architecture review

- **M6 PASS:** spec amended (header bump 2026-06-12) — §10.6 per-class
  settlement correction, NEW §10.7-10.10 (mixed regime / composer arm /
  manifold adapter / census + dispatch default); §9.14/9.15 partition
  rows. Manual: PCLL growth section + 2 new drift-corrections
  (empty-dispatch trap; σ-readout falsification). mixed.py split
  deferred (own pass).
- **Full-stack data_hard** (`run_review.py`: composer+manifold+tries=4):
  raw 0.818 / σ-readout 0.825 mean (M2 baseline 0.760; Bayes 0.993).
  Anatomy: 12 receptors rank 0, ZERO edges/tissue (genome regularizer
  correctly quiet on the axis-separable world), 77-101 classes (lineage
  depth max 8), 77 sketches = 7.2 KiB. Testbed panel: 8 computing cells
  in 2 rank strata, 6 edges with the EXACT folded weights (1/2π =
  0.159; 2.5/2π = 0.398), 3 spawned cells (linear sum, linear diff,
  tanh diff on (f0,f1)).
- **Rocky is reviewing the architecture** — discussion points queued:
  no OUTPUT cells anywhere in the PCLL path (readout is
  template/sketch-side; the OUTPUT gene is unused); sensation side at
  its floor in both books (one-way drain — only EMPTY worlds repay it);
  gene groups only differentiate via composer settlements; the
  fragmentation/merge lever for the remaining 0.83→0.99 gap.

## The gap hunt (D18 → D19 → D20), the session's second arc

1. **D18 merge consumer BUILT** (two-sample duplicate test + balanced-
   union try_divide veto; survivor keeps both histories). Honest result:
   **fires ZERO times — data_hard has NO duplicate fragments** (min gap
   ~1.0σ over 1722 similar pairs). Three falsifications recorded incl.
   the inverted-try_divide form (−0.147) and the degenerate cat()[-BUF:]
   union (−0.265).
2. **D19 (mixture readout) REFUTED at step 0**: K-mixtures flat-to-worse;
   kNN over the organism's OWN members+labels ≡ raw filter to 3 decimals.
   My "model class ~0.14" attribution was a CONFOUND (ground-truth kNN
   measured label quality). kNN on our pockets with TRUE labels = 0.99 ≈
   Bayes — encoding lossless; **the gap is SELF-LABELING.**
3. **D20 BUILT + GATED as a TRIPLET** (`run_d20_membership.py`):
   (a) consolidate — EM round per quiescent boundary (the dream shape);
   (b) gate_k — per-sample margin floor ("belongs to nothing": refused
   samples deposit but don't buffer); (c) classify(x, k_reject) — the
   reject output (k=4: 100% clean coverage, 63.8% disruptor spray
   refused — Rocky's no-reject concern, confirmed at 47-58% of impurity
   and answered). **Factorial (8 arms): NO piece lifts alone; the
   triplet = 0.818→0.854** (+0.037, purity .715→.743, classes 91→79).
   Freeze-streaming holds the plateau. Flags default-off everywhere;
   M2/M4/M5 byte-identical.
4. **OPEN RULING (Rocky):** `member_margin` — ranking membership by E/σ
   (resolve.py's own semantics) was found by accident as a silent
   default change worth **+0.051 on the M2 gate alone** (0.760→0.811);
   reverted to opt-in for byte-identity. Decide: make it the default
   (re-record M2 baseline) or keep opt-in. Also still queued:
   divide_tries=4 default; mixed.py split (~900 lines now — the flag
   zoo wants profiles); dendrite settlement threshold.
5. Also this arc: supervised tanh-MLP baseline on data_hard (H=64:
   0.965; H=256: 0.983 vs Bayes 0.993 — Rocky's recollection confirmed;
   our organism ≈ a supervised H=16 net, label-free single-pass);
   sentinel band QUANTUM-DERIVED (F_MAX·(1+[1.0,1.2]/N_QUANTA), Rocky's
   catch — no hardcoded decimals; M4 re-gated stronger: 0.729 > 0.620,
   gain +0.637); prune-remap bug fixed (pending division records on a
   pruned composer's dim dissolve; others shift).

## Open questions / NEXT (priority order)

1. **Rocky: D19 decision** (exemplar/mixture readout — the measured
   ~0.14 lever). Then: mixed.py split (~770 lines now), divide_tries
   default, dendrite settlement threshold.
2. **Small decisions for Rocky:** divide_tries=4 as the M2/default
   (+0.051 on data_hard)? Merge CONSUMER for families (collapse
   fragments a composed dim explains — D17's unbuilt half; now also the
   identified lever for the remaining Bayes gap, see falsification 1)?
   Dendrite settlement threshold revisit (spawns occur, none survived
   the testbed gate)?
3. Carried: 4 pre-existing test_v2 failures (bisect someday);
   ATTENTION/CONV/RECURRENT composer semantics (deferred, doc §4);
   canonical injection for frame-moving composer sources; per-class
   σ-readout grew obsolete → routing/merge is the gap lever.

## Pointers

- Design: `docs/design/mixed_stream_growth.md` — gate rows M1–M5 carry
  the full records; §4b approved amendments.
- M5 code: `trioron/pcll/manifold.py` (the adapter),
  `trioron/pcll/mixed.py` (`_anneal`, `_rebuild_sketches`,
  `state_dict`/`load_state_dict`, `freeze`). Gate:
  `run_m5_manifold.py` (A=σ-readout, B=freeze-anneal, C=ship/wake).
- All gates green: `run_m2_mixed`, `run_m3_settlement`,
  `run_m4_composer`, `run_m4_hard`, `run_m5_manifold`,
  `run_i3_stress`, `run_i5_architecture`.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, WSL2, `python3`, `OMP_NUM_THREADS=8`.
  M5 gate ~6 min (part B streams 3 epochs ×2 arms); M4 ~2 min; tests ~11 s.
