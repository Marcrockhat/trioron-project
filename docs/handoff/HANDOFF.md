# Trioron Handoff

**Session date:** 2026-06-12
**Session number:** 033
**Session title:** **M3 [D13] PASS + M4 [D14+D16+D17] PASS — the composer arm
grows REAL tissue: computing +2/+3, edges +6, DEPTH 0→1 every gate seed;
0.694 > division-only 0.624. data_hard per-class: zero spawns (genome
regularizer holds), +0.051 from multi-try division. NEXT = M5 manifold
adapter.**

---

## READ THIS FIRST

1. **The governing design doc is `docs/design/mixed_stream_growth.md`**
   (D11–D16 approved s032; §4b amendments + D17 family trials approved by
   Rocky s033). M1–M4 are PASSED and committed; **M5 (manifold adapter) is
   next**, then M6 (spec §10 amendment + manual).
2. Rocky's trust criterion is now MET for the first time: the census moves.
   The M4 gate shows computing 5→7/8, edges 0→6, **depth 0→1** on every
   seed — cells with earned rank, real edges, expression genes
   (linear/tanh), books that differentiate. The remaining unbuilt feature
   of the five-feature rebuild is the **manifold adapter** (M5).

## The arc (s033, in order)

1. **M3 — per-class settlement [D13] PASS** (commit `a2d5ee5`): pendings
   carry the class answered; settle only on that class's own testimony,
   DEFER otherwise (silence pays nobody). False credit dead (i5_diag: 0
   settlements, 14/14 defer, book pristine); division drain fixed (settle
   vs your OWN children: success 0.924). Strict winner-testimony rule on
   the PCLL path (wrong tentative winner → defers forever, accepted).
2. **M4 pre-probe** (`run_composer_genome.py`, commits `92c26fd`+sweeps):
   the genome redo. KEY IDENTITY: the dendrite gene expresses the pure
   ring form EXACTLY — w = −1/(2π) over phase a = 2π(c+½) gives
   σ(−(c+½)) = c²−¼ (tied linear term cancels); soma bias ½ → c_i²+c_j².
   §4b amendments measured + approved (Rocky): residual-incoherence ratio
   (1−null)/(1−carve) > 2 as the trial statistic; **D17 family trials**
   (division fragments manifolds into raw-separable arcs first — trial
   unions of lineage siblings, FAMILY_DEGREE=3 by 8-seed sweep: d1 .676,
   d2 .698, d3 .736±.031 = plateau through d5, d6=unbounded .726 worse;
   tree depth measured 7); importance = any-class buffer coherence
   (genesis-margin absolute cut starves pairs; small buffers confabulate);
   division/composition COMPETE per sitting; PATIENCE judges available
   virgin data.
3. **M4 — composer arm through the substrate [D14+D16+D17] PASS**
   (commit `51d732f`): new `trioron/pcll/composer.py` (candidate set,
   ratio statistic, importance-gated overproduction + split-half confirm,
   spawn/prune, ComposerPending); mixed.py integration (per-buffer trials
   compete with division, ONE family trial/boundary, fresh stores follow
   divided lineages, buffer back-fill at spawn, `divide_tries` param);
   stress.py gene-targeted transfers (the winning gene's council group
   earns/repays — the book finally differentiates, winner_phenotype()'s
   economy live).
   **Gate** (`run_m4_composer.py`, 3 seeds, sentinel testbed): composer
   0.694 > division-only 0.624, relational gain (A+B+E) +0.51; census
   every seed computing +2/+3, edges +2·spawns, depth 0→1; ZERO
   noise-pair spawns; forward-path composed pockets ≡ buffer-side
   composition (≤ 6 quanta); Σ=28; future-deposit prunes fired.
4. **TWO REAL BUGS found by the first spawn:**
   - `construct()` defaulted to an EMPTY dispatch table — every computing
     cell the organism grows was silently skipped in forward (the s020
     "growth was inert" class). Fixed: default = the real phenotype
     table (`trioron/core/construct.py`). Receptor-only organisms
     unaffected (I5/M1/M2 byte-identical after fix — verified).
   - Worst-dim-only division NEVER fires on worlds with pure-noise dims
     (the worst dim is always noise, rejected at the NULL_SPLIT floor).
     `try_divide(Z, tries=)` + `MixedStreamController(divide_tries=)`;
     default 1 keeps M2 byte-identical.
5. **data_hard per-class check (Rocky's ask)** (`run_m4_hard.py`):
   composer arm on the 32-class capacity-hard set → **ZERO spawns, all
   seeds** — the genome regularizer + relation null correctly QUIET on
   axis-separable, frame-moving data (no frame-coupling false spawns).
   Mean 0.760 → 0.811 (+0.051) comes from multi-try division alone
   (divide_tries=4 discovers more of the 96 true modes: 94 vs 84
   classes). Biggest per-class moves: sp026 +0.46, sp020 +0.30,
   sp010 +0.25, sp014 +0.16; regressions sp029 −0.20, sp004 −0.12
   (finer fragmentation dilutes a few clean classes). Full table in the
   runner output.

## Decisions made (Rocky, s033)

- §4b approved in full (incl. D17 family trials).
- Family-trial relation scope BOUNDED: Rocky asked for 1–2 degrees;
  measurement said 3 (plateau 3–5, byte-identical 4/5; near-root pools
  degrade). FAMILY_DEGREE=3 shipped, frontier-local.
- M3's strict winner-testimony + the M4 competition ladder are on record
  in the design doc (§4b + M3/M4 gate rows).

## State of the build

- Branch `progenitor-council`, all pushed. s033 commits: `a2d5ee5` (M3),
  `f441b5a` (handoff), `92c26fd` (probe), `594a01e`+`8a3b5c2` (degree
  sweep), `044884e` (depth readout), `51d732f` (M4), + M4-record commit.
- Tests: 94 passed + the SAME 4 pre-existing failures (credit ×2, growth
  trigger, dream lock — `bisect someday` carry). New:
  `tests/test_v2/test_composer.py` (spawn identities incl. the dendrite
  σ-identity, structural contract, prune, gene vote transfer, multi-try
  division).
- Regressions all green this session: M2 byte-identical 0.760, M3 0.924,
  I3, I5 byte-identical (0.387/0.386/0.384).
- **DO-NOT-COMMIT carries (unchanged, still dirty):**
  `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
  `trioron/viz/export.py`; `.claude/`, `runs/` untracked.

## Open observations (not gate items, on record)

- **Dendrite spawns occur but none SURVIVED settlement** on the gate
  testbed (future-deposit verdicts on virgin stores fail near the ratio
  threshold; sentinel frame jitter ±0.1% may erode marginal verdicts).
  E still improved 0.41→0.51 via linear/tanh + multi-try division. Worth
  a focused look (is the virgin-store ratio for dendrites sitting just
  under 2?) but rings are also legitimately divisible into arcs here.
- **Frame caveat stands:** composer sources must be canonical. The gate
  testbed uses an I3-style sentinel gain-reference column (~1.0 ± 0.1%
  jitter — a constant would be starved at the first sitting). Composing
  over frame-moving receptors needs canonical injection (deferred). On
  data_hard (frames move) the composer stayed quiet, so no harm today.
- Pruned composer cells count as "germline" in the census (forward=False
  + genes ≠ 0) — a kind-classification quirk, cosmetic.
- Multi-try division (+0.051 on data_hard) is NOT yet the M2 default
  (byte-identity preserved). Whether divide_tries=4 becomes the default
  is a small open decision for Rocky.

## Open questions / NEXT (priority order)

1. **M5 — manifold adapter [D15]** (design doc §5): reuse
   `trioron/learning/manifold.py` (`ManifoldAstrocyte` μ/σ sketches)
   over POCKET space. Consumers in order: (a) post-quiescence annealing
   (the freeze-decay fix); (b) σ-weighted readout — the measured
   0.62→0.99 half of the Bayes-ceiling 0.993 target (raw arm reported
   alongside, s029 comparability); (c) replay guard for grown tissue
   (spec-§4.5 replay protecting PCLL classes when composer cells reshape
   the read space — the "less forgetful" half of the rebuild aim).
   Gate (doc §6): annealing recovers freeze-decay; σ-readout closes the
   measured gap on clean classes; ship/wake round-trips sketches.
2. **M6 — spec §10 amendment + manual update** (spec-first rule D6):
   composer arm, family trials [D17], per-class settlement [D13], the
   census instrument; fix the manual's PCLL section.
3. Small decisions for Rocky: divide_tries=4 as M2 default? dendrite
   settlement threshold revisit? merge CONSUMER for families (collapse
   fragments a composed dim explains — D17's unbuilt half)?
4. Carried: 4 pre-existing test_v2 failures (bisect someday); composer
   semantics for ATTENTION/CONV/RECURRENT (deferred, doc §4); canonical
   injection for frame-moving composer sources; post-quiescence annealing
   folds into M5.

## Pointers

- Design: `docs/design/mixed_stream_growth.md` (M1–M4 gate rows carry
  full records; §4b approved amendments incl. D17).
- M4 code: `trioron/pcll/composer.py` (all of it),
  `trioron/pcll/mixed.py` (`_trials`, `_spawn`, `_testimony`,
  `_prune_composers`, `_feed_fresh`), `trioron/pcll/stress.py`
  (`_report_gene`), `trioron/pcll/division.py` (tries),
  `trioron/core/construct.py` (dispatch default — read the comment).
- Gates: `run_m4_composer.py` (M4), `run_m4_hard.py` (data_hard
  per-class), `run_m3_settlement.py`, `run_m2_mixed.py`,
  `run_i5_architecture.py`, `run_i3_stress.py` — ALL green this session.
- Probes: `run_composer_genome.py` (the §4b evidence; FAMILY_DEGREE
  sweep + lineage-depth readout `run_seed.last_depths`).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, WSL2, `python3`, `OMP_NUM_THREADS=8`.
  M4 gate ~2 min; M4 hard ~4 min; M3 ~4 min; M2 ~1 min; tests ~11 s.
