# Trioron Handoff

**Session date:** 2026-06-11
**Session number:** 030
**Session title:** **PCLL integrated into the substrate END-TO-END (I1–I5 all
gated PASS): RECEPTOR gene + lock-in arena state, germline (progenitor+council)
period-1 genesis with NATAL REPLAY, stress drivers on the conserved vote book,
dynamic-perception manifold (per-dim counts, shadows, frame translation), TANH
phenotype + council seats — and the ARCHITECTURE TEST matches the s029
standalone numbers with zero gradients anywhere.**

---

## READ THIS FIRST

1. `docs/design/pcll_substrate_integration.md` — the integration design,
   decision register **D1–D10** (all approved/built; D8/D9/D10 carry Rocky
   flags + rationale).
2. `paper/v3/spec.md` **§10** (the normative PCLL substrate contract, built
   this session: §10.1–10.6) + §9.15 (`trioron/pcll/` partition) + §2.2 gene
   table (RECEPTOR bit 11, TANH bit 12) + §3.10 (tanh phenotype).
3. `docs/design/receptor_period_frustration.md` — the PCLL method spec (s028/9).

## The arc (commits, in order, all pushed on `progenitor-council`)

1. **`39a0ff6`** — spec-first: §10 contract, RECEPTOR gene, §9.15 partition,
   integration design doc (D1–D7 approved by Rocky).
2. **`e291f70`** — **I1**: RECEPTOR injection in the scheduler (continuous
   quantizer + discrete labeled lines; per-sample gain frame over continuous
   receptor cols only), `lockin_re/im/n` + `receptor_levels` arena tensors
   (ship/wake round-trip), `attach_pcll` + `end_task` boundary meeting,
   `trioron/pcll/` (lockin/resolve/signature/controller/receptor),
   `PCLLResolution` frustration adapter. **Gate: exact parity with
   run_schedule_learning through the Substrate forward path** (5 seeds,
   births 1/9/17, alignment ≥0.999, zero structural forgetting). NB: pocket-
   valued test data needs an explicit gain-reference sentinel column (=1000).
3. **`c1c7dbb`** — **I2**: `progenitor.py` — Germline (progenitor + council,
   forward-invisible, never locks) + PerceptionGenesis (tick-1 spawn / tick-2
   receptor equip / distinct-value census K_DISCRETE=8 / first sitting:
   STARVE census-constant cols by WITHDRAWING the receptor gene — a constant
   is NOT at the q=0 floor under signed data; discrete verdicts + codec;
   handover). Habituation retirement (RETIRE_PATIENCE=3; empty periods carry
   no testimony; n≥K² testimony floor — below it coherence is unreachable).
   **Gate: kept-set == genesis gradient-probe baseline, zero gradients,
   3 seeds.**
4. **`1a70e7f`** — **natal replay** (Rocky: bug — period-1 class was lost):
   period-1 obs buffered, replayed through the FINAL receptor config at the
   first sitting, learned as a normal period (data_hard 32/32, was 31).
   **TANH** expression gene bit 12 (§3.10), bounded |y|≤1, council seats it
   automatically (6×4=24). 10-species note: chicken+duck and dog+goat MERGE
   under the unsupervised fit (the known overlap pairs) — every period
   assigned, splitting them = mixture-aware birth (NEXT).
5. **`77427d0`** — **I3**: `stress.py` — StressRouter over the germline's ONE
   book: 24 composer seats + 4 progenitor-held perception seats (D8, Σ=28;
   4 seats × floor ¼ = exactly 3 payable votes = the habituation walk).
   decide() on the exclusive status; settle() at the next boundary (D9).
   Controller: status mapping (birth = comprehension), resolver_templates
   known-world mode, recruit(), refresh_receptors() (cell-id remap).
   **Gate: all three stress.py scenarios on the substrate council** (hidden
   signal ≤2 attaches; true-void → accepted-empty after exactly 3, perception
   side paid to floor; ambiguity → recruit f3 → RESOLVED gap≈42).
6. **`70115e1`** — **I4** (D10, from Rocky's two discussion points: stored
   manifold stats must adjust when perception changes; quanta coordinates
   need a translation layer): per-dim counts m_f; SHADOW accumulation as
   default extension (recruit() promotes instantly; auto-promotion at
   PROMOTE_PATIENCE=3 READ matches); FRAME STAMPS + read-time closed-form
   translation (q'=a·q+b on the UNWRAPPED pocket coordinate — torch.angle's
   principal value scaled the wrong lift past q=500, found+fixed; R'=R^(a²);
   discrete lines frame-free); codec_levels (serializable); astrocyte cell
   per class; controller state_dict; ship() embeds the PCLL world, wake()
   exposes `_pcll_state`. **Gate (run_i4_dynamic): frame shift survived 9/9
   with 0 spurious births vs 3 in the no-translation control; col4 becomes
   the new gain reference (never deposits) and each class's OLD reference dim
   is RELEASED + auto-promotes; shadow recruit aligns ≥0.999 zero-relearning,
   gap 52.7→62.2; ship→wake round-trip.**
7. **`f6ed9e8`** — **I5 ARCHITECTURE TEST PASS** (run_i5_architecture, the
   full developmental stack on data_hard, 3 seeds): 32 classes purity 1.0,
   epoch-2 32/32 zero forgetting, **one-shot 0.384–0.387 (s029 raw arm
   0.38–0.41), dead-3 0.300–0.321 (s029 0.30–0.31)**, votes conserved, NO
   gradients anywhere. Eval mirrors s029 single_shot exactly (clean classes
   only — disruptors excluded by construction; raw-evidence argmax; fixed
   dead set) + per-sample frame translation.

## Decisions made (sign-off Rocky, s030)

- Developmental flow: period 1 = perception generation (council attached to
  the progenitor from tick 1 — "the council judges, the progenitor spawns");
  council votes AT period boundaries; genesis.py batch pre-pass REPLACED.
- Tick-2 "bias" = receptor adaptation; overproduce-then-prune confirmed
  (12-d hard taxonomy → 12 kept; 64-aperture → 7 kept, verified).
- Natal replay (no class lost to organ-building); TANH as 6th expression
  gene with council seats; D8/D9/D10 (see design doc register).
- Aperture SIZING is built+validated (step2/2b/genesis lineage, now
  gradient-free); what remains unbuilt is DENSE-FIELD compression to the
  1.5 Mi working resolution (every slot variance-bearing → local pooling,
  never dense-global) — staged post-I5, needs a wide-input testbed.

## State of the build

- Branch `progenitor-council`, everything committed AND pushed through
  `f6ed9e8` + this handoff. **DO-NOT-COMMIT carries (still excluded):**
  `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
  `trioron/viz/export.py`; `.claude/`, `runs/` untracked.
- New package: `trioron/pcll/` (receptor, lockin, resolve, signature,
  controller, stress, progenitor). Core touched: epigenome (bits 11/12),
  arena (lockin tensors), scheduler (injection + frame stash), construct
  (attach_pcll/end_task), receptor.py (new), ship/wake (pcll block),
  phenotype/tanh.py (new).
- Gate runners (all PASS, seconds-fast, CPU): `run_i1_substrate.py`,
  `run_i2_genesis.py` (-m), `run_i3_stress.py` (-m), `run_i4_dynamic.py`
  (-m), `run_i5_architecture.py` (-m). Standalone s029 runners untouched
  and still PASS.
- Tests: 25 PCLL/tanh tests in `tests/test_v2/` (test_pcll.py,
  test_phenotype_tanh.py). Suite: **70 passed + 4 PRE-EXISTING failures**
  (test_learning credit ×2, test_lifecycle growth-trigger + dream-lock) —
  verified identical at clean HEAD BEFORE this session's code; not ours.
  Worth a bisect someday.

## Open questions / NEXT (priority order)

1. **Mixture-aware birth** (handoff-NEXT item 5; now the top lever): split a
   class when its own deposits stop fitting — the chicken+duck / dog+goat
   splitter, and the 1-shot 0.38→0.62 (mode-level) lever. Slots cleanly into
   signature.py (per-dim m and shadows already in place).
2. **Period segmentation** (unknown boundaries — the real frontier).
3. **Componential semantics** (signed relational readout over signature
   space; design §10 of the method doc).
4. **Dense-field compression** to the 1.5 Mi working resolution (local
   pooling per the partitioning rule; census scaling debt: Python set per
   column).
5. The 4 pre-existing test_v2 failures (bisect).
6. Composer-phenotype spawns from FRUSTRATED stress (the gradient-path arm
   of I3's economy — winner_phenotype() is recorded but nothing spawns yet).

## Pointers

- Integration: `docs/design/pcll_substrate_integration.md` (D-register).
- Contract: spec §10; partition §9.15; cross-index rows at §9.14 end.
- Method: `docs/design/receptor_period_frustration.md` (+§11 corrections).
- Vote semantics: `trioron/pcll/stress.py` mirrors `step4_grow.py`.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12 (no nested same-quote f-strings), torch 2.11.0, WSL2,
  `python3`, `OMP_NUM_THREADS=8`. All runners seconds-fast on CPU.
- Session was very build-heavy (I1–I5 in one sitting); everything is
  committed, pushed, and green; safe break.
