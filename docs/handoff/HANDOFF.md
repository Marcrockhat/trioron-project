# Trioron Handoff

**Session date:** 2026-06-12
**Session number:** 033
**Session title:** **M3 per-class settlement attribution [D13] GATE PASS —
false credit dead (deferral: silence pays nobody), division drain fixed
(settle against your OWN children, success 0.92). NEXT = M4 composer arm.**

---

## READ THIS FIRST

1. **The governing design doc is `docs/design/mixed_stream_growth.md`**
   (D11–D16, all APPROVED by Rocky s032; build phases M1–M6 with gates).
   M1+M2+M3 are PASSED and committed; **M4 is next** (composer arm,
   genome-constrained). Read doc §4 before touching code.
2. Rocky's trust criterion (s032, standing): the rebuild's aim is a network
   that **changes its own topology and depth** to adapt and learns
   continually with less forgetting. Proof is structural — cell/edge/rank
   counts must MOVE (census [D11] reports them in every MeetingReport).
   M4 is the phase where the organism finally earns **computing cells,
   edges, and rank ≥ 1** — the first depth. The genome principle (s032
   correction) binds: candidates are LINEAR/TANH/DENDRITE-quad ONLY;
   ATTENTION/CONV/RECURRENT deferred explicitly.

## What M3 built (s033, commit `a2d5ee5`)

**[D13] A pending growth decision carries the class it answered and settles
only on that class's own testimony — never the period's global status.**
Vote transfers keep step4 semantics (equal share, floors, Σ=28); only the
success predicate moved.

- `trioron/pcll/stress.py`: `pending` is now a **list of (driver,
  subject)**; `settle(status, testify=...)` settles subject-bearing entries
  per-class and **DEFERS** entries with no testimony (the fix itself: the
  world changing is not evidence about a growth — silence pays nobody);
  `attach_subjects()` fans the boundary's decision out to one pending per
  executed division; `settlements` audit log (the book saturates at its
  floors; the log keeps the full signal). Subject-less entries (sensation)
  keep the s030 global predicate + habituation unchanged.
- `trioron/pcll/controller.py`: FRUSTRATED decisions carry
  `resolution.winner` as subject. Testimony rule is **strict**: the subject
  testifies only when it is THIS period's winner (resolved → success, still
  blurred → failure, anything else defers). Consequence (accepted, on
  record): an ambiguous pair attributed to the wrong tentative winner
  defers forever rather than collecting unearned credit — I3-C dog case
  shows disc=24 (no credit) vs chicken disc=25.
  `state_dict`/`load_state_dict` handle the new pending shape (+ pre-D13
  back-compat: bare string → [(str, None)]).
- `trioron/pcll/mixed.py`: `_execute` returns settlement subjects
  `(child_a, child_b, split_dim, acceptance_floor)`; at the NEXT boundary
  (after `_assign`) each division settles iff BOTH children still cohere on
  the split dim above the floor it was accepted at — the membership that
  arrives after the growth exists is the testimony (D9 spirit). Settle runs
  BEFORE this boundary's execute, so last boundary's children always exist.

**Gate (`run_m3_settlement.py`, 3 seeds, all asserts):**

- *False credit (i5_diag scenario):* the flat-4.500 signature is dead — 0
  settlements, **all 14 frustrated decisions defer** (their classes never
  testify again in 2-epoch class-sequential), book pristine sens=4/disc=24,
  Σ=28. (Old predicate: sens drained 4→1, every gene group flat 4.500.
  The 14 deferred match s031's "14 dropped signals" exactly.)
- *The drain (M2 mixed scenario):* 74–83 divisions/seed, ALL settled
  per-class, success 0.892/0.927/0.952 (mean 0.924, gate ≥ 0.90; the ~8%
  failures repay honestly); discrimination holds 26–27 votes (old global
  predicate paid FAILURE every frustrated discovery window).

**Regressions all green:** I3 stress PASS; I5 byte-identical
(0.387/0.386/0.384); M2 byte-identical (0.760 mean, same division
trajectories); tests 83 passed + the SAME 4 pre-existing failures
(credit ×2, growth trigger, dream lock — `bisect someday` carry).

## Known limitation (for M4, on record)

The side-level vote economy **saturates after 3 events** either direction
(sensation's 4 seats hold exactly 3 payable votes; in the mixed run true
credit drives the book to sens=1–2/disc=26–27 and the gene groups stay
mutually flat — side-level transfers mathematically cannot differentiate
genes). Gene-group differentiation arrives in **M4**: settlement success
transfers one vote to the WINNING GENE's council group
(`winner_phenotype()` finally gains its consumer, design §4). The
`settlements` audit log exists precisely because the book saturates.

## State of the build

- Branch `progenitor-council`. s033 commit: `a2d5ee5` (M3 + gate runner +
  design-doc PASS row + tests). s032 commits: `5fbf5b3` (M1), `a0031f6`
  (M2), `7c15511`/`94b6349`/`418e69f` (design doc).
- **DO-NOT-COMMIT carries (unchanged, still dirty in worktree):**
  `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
  `trioron/viz/export.py`; `.claude/`, `runs/` untracked.
- Census live in every MeetingReport; `router.settlements` +
  `router.pending` are the new settlement readouts.

## Open questions / NEXT (priority order)

1. **M4 — composer arm [D14+D16]** (design doc §4): genome-constrained
   candidates (LINEAR sum/diff, TANH bounded, DENDRITE-quad c_i²+c_j²);
   importance-gated wiring (pairs only from
   `Germline.perception_importance`); **future-deposit settlement** (the
   spawn is settled only by deposits arriving after it exists — now
   buildable directly on M3's subject/testify machinery: the spawn's
   subject = its own lock-in row evidence); fixed static frames (causal
   quantization — every candidate form is bounded by construction); spawn =
   real cell + 2 edges + rank = max(src)+1. Settlement success transfers
   one vote to the winning gene's group (the M3 limitation note above).
   PATIENCE failure hard-retires the cell, edges masked, bet repaid [D16].
   Gate: relational testbed through the substrate beats division-only
   (>0.595); census delta computing+k / edges+2k / max rank ≥ 1; **zero
   noise-pair spawns across seeds** (failure-mode-4 kill). Redo the
   relational testbed with the REAL gene set first (s031 probe used a
   non-genome phenotype set — its vote-book claim does not transfer).
2. **M5 — manifold adapter [D15]**: `ManifoldAstrocyte` sketches over
   pocket space; annealing (freeze-decay fix), σ-weighted readout (the
   0.62→0.99 half of the Bayes-ceiling 0.993 target), replay guard for
   grown tissue.
3. **M6 — spec §10 amendment + manual update** (spec-first rule D6).
4. Carried: 4 pre-existing test_v2 failures (bisect someday); composer
   semantics for ATTENTION/CONV/RECURRENT (deferred decision, doc §4);
   under-division/sibling-merge (s031 NEXT 4); post-quiescence annealing
   folds into M5.

## Pointers

- Design: `docs/design/mixed_stream_growth.md` (governing; M3 row has the
  full gate record), D1–D10 in `docs/design/pcll_substrate_integration.md`.
- M3 code: `trioron/pcll/stress.py` (settle/decide/attach_subjects),
  `trioron/pcll/controller.py` (~line 183 testimony), `trioron/pcll/mixed.py`
  (`_division_testimony`, `_execute` records). Gate:
  `experiments/progenitor/run_m3_settlement.py`. Tests:
  `tests/test_v2/test_pcll.py::TestStress` (3 new D13 tests).
- Other gates: `run_i5_architecture.py`, `run_m2_mixed.py`,
  `run_i3_stress.py` — all re-run green this session.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, WSL2, `python3`, `OMP_NUM_THREADS=8`.
  M3 gate runner ~4 min; M2 ~1 min; I5 ~40 s; tests ~10 s.
