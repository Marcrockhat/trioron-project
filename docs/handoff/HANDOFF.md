# Trioron Handoff

**Session date:** 2026-06-12
**Session number:** 032
**Session title:** **Composer-probe CORRECTION (genome, not arbitrary functions);
mixed-stream growth design APPROVED (D11–D16); M1 census PASS; M2
division-as-discovery THROUGH THE SUBSTRATE PASS (one-shot 0.760, every
council decision consumed, growth arena-countable). NEXT = M3 per-class
settlement attribution.**

---

## READ THIS FIRST

1. **The governing design doc is `docs/design/mixed_stream_growth.md`**
   (D11–D16, all APPROVED by Rocky s032; build phases M1–M6 with gates).
   Read it before touching code. M1+M2 are PASSED and committed; M3 is next.
2. Rocky's trust criterion (s032, restated and now enforced): the rebuild's
   aim is a network that **changes its own topology and depth** to adapt and
   learns continually with less forgetting. The proof is structural — cell /
   edge / rank counts must MOVE. Census instrumentation [D11] now reports
   this in every MeetingReport; gates assert structural deltas.
3. s031's composer probe (`run_composer_growth.py`) is CORRECTED on record:
   its phenotype set {gcos, sinc, tan, standalone-quad} is NOT the genome.
   The expression genes are LINEAR/ATTENTION/CONV/RECURRENT/DENDRITE/TANH
   (`epigenome.py:28`). **Overproduction is over the GENOME, not arbitrary
   functions** — a council seated by expression genes cannot vote for a
   function with no gene bit; the small fixed genome is a regularizer. The
   probe's vote-book claim does not transfer; its mechanism loop does.

## The arc (s032, in order)

1. **Corrections (Rocky caught both):** (a) the composer-probe genome
   violation above; (b) the M1↔D11 registers clarified — D## = decisions,
   M# = build phases (M1→D11, M2→D12, M3→D13, M4→D14+D16, M5→D15, M6=spec).
2. **Feature-status census (the indictment that drove the design):** after
   I5 learns 32 classes the integrated organism had 12 computing cells
   (fixed at tick 1), 0 edges, depth 0 — every growth event was a
   bookkeeping row. Council decide() signals had no consumer; settlement
   paid false credit; the manifold adapter for PCLL did not exist.
3. **Design doc written + approved** (`mixed_stream_growth.md`): D11
   structural contract + census; D12 division consumer; D13 per-class
   settlement; D14 genome-constrained composers (LINEAR/TANH/DENDRITE-quad
   only; ATTENTION/CONV/RECURRENT explicitly deferred; importance-gated
   wiring; future-deposit settlement; fixed static frames = causal
   quantization; spawn = real cell + 2 edges + rank>0); D15 manifold
   adapter (= `learning/manifold.py` sketches over pocket space: annealing,
   σ-readout, replay guard); D16 hard prune on PATIENCE.
4. **M1 PASS (`5fbf5b3`):** `trioron/core/census.py` (+`GENE_NAMES` in
   epigenome). Census partitions computing/germline/astrocyte from arena
   fields alone; depth over computing cells only; `delta()` for gates;
   carried in every MeetingReport. I5 accuracy byte-identical
   (0.387/0.386/0.384); baseline 12 computing / 0 edges / depth 0 asserted.
5. **M2 PASS (`a0031f6`):** `trioron/pcll/division.py` (split-vs-keep:
   worst dim, circular 2-means, BOTH children beat parent+GAIN_D and
   NULL_SPLIT=0.72, MIN_CHILD=25) + `trioron/pcll/mixed.py`
   (MixedStreamController: genesis world-class adopted from period 1,
   matched-filter membership into per-class ROLLING buffers, council
   judgment, stress settle/decide, progenitor executes divisions as
   sibling astrocytes WITH `arena.parent` lineage). Gate runner
   `run_m2_mixed.py`: one-shot (clean) **0.760 mean** (probe 0.711,
   class-sequential 0.386), 31–32/32 truths, every DISCRIMINATION decision
   consumed, votes conserved, astrocytes 1→149–167 rows (75–84 live).
   **Two findings (both were gate failures first):**
   - **Division requires the canonical frame.** Raw per-sample contrast
     pockets make every dim look multimodal (frame motion ≠ mode
     structure; s021's trigger lesson) → divide-to-cap, acc 0.523. And the
     canonical frame must be the EXTREMES frozen at the genesis boundary
     (world window ≈ population extremes, causal) — a MEAN frame clamps
     real values to pocket edges and the pileup divides forever. The D10
     per-sample affine in `mixed._canonical()` is exact (inverse of the
     quantizer's map).
   - **Self-arrest = RATE COLLAPSE, not zero windows.** data_hard is 32
     classes × 3 MODES = 96 true modes; division discovers MODES (the
     purity map bundles them). Verified cap-free over 2 epochs: ~75
     divisions/epoch → 3–6, classes 81–87 < 96, never tiling. The probe's
     0-division windows were a finite-stream artifact. NULL_SPLIT changed
     nothing here (late splits are real structure) but stays as the
     approved guard.

## Decisions made (Rocky, s032)

- D11–D16 all approved; build in M-phase order; M1 then M2 explicitly
  ordered ("start with M1", "proceed to M2").
- M1 ≡ D11 confirmed (registers are decisions vs build phases).
- The five-feature rebuild framing is on record: progenitor/genesis,
  council/votes, frustration gate, PCLL, manifold adapter — all five must
  be wired and demonstrably working or the architecture isn't trusted.
- Session ended cleanly at the M2/M3 boundary by Rocky's choice
  (context-pressure warning given).

## State of the build

- Branch `progenitor-council`, all s032 work committed AND this handoff
  pushed. Commits: `3acac44`+`c40f56b` (s031 handoff corrections),
  `7c15511`+`94b6349`+`418e69f` (design doc + status), `5fbf5b3` (M1),
  `a0031f6` (M2).
- Tests: 74 passed + the SAME 4 pre-existing failures (credit ×2, growth
  trigger, dream lock — `bisect someday` carry). I5 regression PASS.
- **DO-NOT-COMMIT carries (unchanged, still dirty in worktree):**
  `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
  `trioron/viz/export.py`; `.claude/`, `runs/` untracked.
- Census instrument is live: every MeetingReport carries `.census` and
  `.divisions`; `Census.delta(before)` prints structural deltas.

## Open questions / NEXT (priority order)

1. **M3 — per-class settlement attribution [D13]** (design doc §3): pending
   growth carries the class it answered; success = THAT class's stress
   cleared (buffer coherence recovered / margin cleared), not global period
   status. Touches `stress.py` (pending → (driver, class)) +
   `mixed.py`. Gate: the i5_diag false-credit scenario — phenotype groups
   must differentiate (no longer flat 4.500); Σ=28 conserved. Note: during
   M2's exponential discovery the global predicate pays DISCRIMINATION
   failures every frustrated window — M3 should also fix that drain.
2. **M4 — composer arm [D14+D16]**: genome-constrained candidates
   (LINEAR/TANH/DENDRITE-quad), importance-gated wiring, future-deposit
   settlement, fixed static frames, spawn = real cell + 2 edges + rank ≥ 1
   (the first depth the organism earns). Gate includes ZERO noise-pair
   spawns (failure-mode-4 kill) and census delta computing+k / edges+2k.
   Redo the relational testbed with the REAL gene set first.
3. **M5 — manifold adapter [D15]**: sketches over pocket space; annealing,
   σ-weighted readout (the 0.62→0.99 half of the Bayes-ceiling 0.993
   target), replay guard for grown tissue.
4. **M6 — spec §10 amendment + manual update** (spec-first rule D6).
5. Carried: 4 pre-existing test_v2 failures (bisect someday); composer
   semantics for ATTENTION/CONV/RECURRENT (deferred decision, doc §4);
   under-division/sibling-merge (s031 NEXT 4); post-quiescence annealing
   folds into M5.

## Pointers

- Design: `docs/design/mixed_stream_growth.md` (governing), D1–D10 in
  `docs/design/pcll_substrate_integration.md`, method in
  `docs/design/receptor_period_frustration.md`.
- New modules: `trioron/core/census.py`, `trioron/pcll/division.py`,
  `trioron/pcll/mixed.py`. Gate runners: `run_i5_architecture.py` (census
  baseline asserts), `run_m2_mixed.py` (M2 gate). Tests:
  `tests/test_v2/test_census.py`, `test_division.py`.
- Vote semantics: `trioron/pcll/stress.py` (28-vote book; settle/decide at
  lines ~105–143 — M3's target). s031 probes + per-class breakdowns:
  `experiments/progenitor/run_{i5_perclass,i5_diag,mixed_division,composer_growth}.py`.
- s031 corrections in git: `3acac44`, `c40f56b` (the genome principle).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, WSL2, `python3`, `OMP_NUM_THREADS=8`.
  M2 runner ~1 min CPU; tests ~9 s; I5 ~40 s.
