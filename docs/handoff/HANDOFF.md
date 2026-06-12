# Trioron Handoff

**Session date:** 2026-06-12
**Session number:** 033
**Session title:** **THE REBUILD SHIPPED END TO END: M3+M4+M5+M6 all gates
PASS (real tissue, depth earned, manifold adapter, spec §10.6–10.10
amended). Then the gap-hunt arc: D18 merge (zero-fire honest), D19 refuted,
D20 membership-quality defaults shipped under the QUIESCENCE-DEFERRAL law;
M2 re-recorded 0.760→0.829. NEXT = D21 nursery (proposed) + cleanups.**

---

## READ THIS FIRST

1. **Governing docs:** `docs/design/mixed_stream_growth.md` (D11–D20
   built+gated, D21 proposed; the gate-table rows M1–M6 and the D-register
   carry the full measured records) and spec §10.6–10.10 (amended s033,
   header bump). The manual was updated (PCLL growth section + two new
   drift-corrections).
2. **The five-feature rebuild is complete and spec'd**: progenitor/genesis,
   council/votes (consumers + gene-targeted economy), frustration gate,
   PCLL with division + composer growth (census MOVES: computing +,
   edges +, depth 0→1 on the relational testbed), manifold adapter with
   full ship/wake.
3. **Defaults changed this session (Rocky's ruling, s033b)** — new
   organisms get the membership-quality stack: `divide_tries=4`,
   `member_margin`, `consolidate`, adaptive gate — ALL under **quiescence
   deferral** (below). `tries=1` + flags-off reproduces every pre-s033
   record exactly.

## Current gate battery (all green, final defaults)

| gate | number | note |
|---|---|---|
| M2 mixed-stream | **0.829** | RE-RECORDED (was 0.760); arrest tail ≤10%, ≤96-mode bound is the tiling guard |
| M3 settlement | 0.924 | pinned to its validated config (falsifies the PREDICATE, not membership) |
| M4 composer | 0.729 > 0.620 | quantum-derived sentinel; relational gain +0.637 |
| M5 manifold | 0.836 end, −0.013 decay | runner fix: truth map recomputed per epoch (stale cache collapsed a seed) |
| D20 membership | +0.006 acc, purity ↑, classes ↓, reject 65.6%@100% | re-pinned to final semantics |
| tests | 107 + the SAME 4 carries | credit ×2, growth trigger, dream lock (bisect someday) |

## The day's two laws (the things to NOT re-derive)

1. **The gap is self-labeling, and nothing downstream can fix it.**
   Measured chain on data_hard: encoding lossless (kNN on our canonical
   pockets = 0.99 ≈ raw Bayes); summarization lossless GIVEN labels
   (kNN-own-labels ≡ raw filter to 3 decimals — D19 refuted at step 0);
   the 0.83→0.99 remainder = label information (boundary placement,
   disruptor spray, one-pass path dependence). The lever for real worlds
   is reward/valence as the label source (grounded-curriculum doctrine).
2. **QUIESCENCE DEFERRAL — refusal starves discovery.** Margin/gate
   statistics presume the read space represents the classes; FALSE during
   relational discovery and for undiscovered truths. A refused member can
   never seed a class because division (the only birth path) feeds on
   buffered members — buffers are memory AND discovery substrate. Hence:
   credulous while growing, skeptical once settled (SETTLE_STREAK=2,
   per-class TRUST_R, adaptive floor GATE_FRAC·√(2W) — absolute margin
   thresholds are dimension-UNSAFE). Four falsifications led here
   (D20 register row has all of them).

## The session arc (compressed; full records live in the design-doc rows)

- **M3 [D13]** per-class settlement: deferral kills false credit; 0.924.
- **M4 [D14/D16/D17]** composer arm: genome candidates fold EXACTLY into
  real tissue (dendrite identity σ(−(c+½))=c²−¼ at w=−1/(2π)); family
  trials FAMILY_DEGREE=3 (8-seed sweep, plateau 3–5); found+fixed
  construct() EMPTY-dispatch bug (the s020 inert-growth class) and
  worst-dim-only division never firing with noise dims. data_hard: ZERO
  spawns (genome regularizer correct on axis-separable worlds).
- **M5 [D15]** manifold adapter: sketches on class astrocyte rows;
  annealing/σ-readout/replay/ship-wake exact. TWO falsifications: the
  s031 σ-gap claim (+0.012 only) and the freeze-decay (not reproduced).
- **M6** spec §10.6–10.10 + §9.14/9.15 + manual amended.
- **D18** merge consumer: two-sample duplicate test; fires ZERO times —
  data_hard has NO duplicates (min gap ~1.0σ / 1722 pairs).
- **D19** mixture readout REFUTED at step 0 (the "model class ~0.14"
  attribution was a label confound — ground-truth kNN measured labels).
- **D20** consolidation + margin gate + reject readout, shipped as
  defaults under quiescence deferral (law 2). `classify(x, k_reject)` —
  the organism can say "unknown" (Rocky's no-reject concern: 47-58% of
  impurity was disruptor spray; 65.6% now refused at 100% clean
  coverage).
- Also: sentinel band quantum-derived (F_MAX·(1+[1.0,1.2]/N_QUANTA),
  Rocky's catch — no hardcoded decimals; M4 re-gated stronger);
  prune-remap bug fixed (pending division records on a pruned composer's
  dim dissolve, others shift); supervised tanh-MLP reference (H=64:
  0.965; H=256: 0.983 vs Bayes 0.993 — our label-free single-pass
  organism ≈ a supervised H=16 net); architecture review delivered
  (run_review.py); novelty assessment given (candidate-novel: PCLL as
  the learning substrate, governed/falsifiable growth, the composer
  identity; needs a related-work pass + an external benchmark before
  any claim).

## State of the build

- Branch `progenitor-council`, all pushed through `7710f28`. ~20 commits
  this session (M3 `a2d5ee5` … defaults `bd70c5e`).
- **DO-NOT-COMMIT carries (unchanged):** `trioron/bases/developmental.py`,
  `trioron/lifecycle/developmental.py`, `trioron/viz/export.py`;
  `.claude/`, `runs/` untracked.
- `trioron/pcll/mixed.py` ≈ 1000 lines — the split is OVERDUE (composer
  orchestration + D20 machinery along concept boundaries; constructor
  profiles for the 8-flag zoo). Behavior-preserving pass, gates re-run.

## Open questions / NEXT (priority order)

1. **D21 — the NURSERY (proposed, needs Rocky):** refused members
   accumulate in an unlabeled DIVISIBLE buffer — "belongs to nothing I
   know YET" becomes a birthplace instead of a void. Resolves the
   refusal-vs-discovery tension structurally; would let the gate run
   during discovery (recovering the always-on gains on axis-separable
   worlds — 0.879 was measured — without starving relational/rare-truth
   discovery).
2. **mixed.py split + constructor profiles.**
3. Dendrite settlement threshold (spawns occur on the testbed, none
   survive virgin verdicts); ATTENTION/CONV/RECURRENT composer semantics
   (deferred, doc §4); canonical injection for frame-moving composer
   sources; PCLL-path compaction for retired astrocyte rows (lifetime
   memory growth — Rocky's standing concern); buffer drop-at-freeze
   (deployment footprint = sketches ~7 KiB vs ~2.5 MB working buffers).
4. External validation arc when Rocky calls it: related-work pass
   (streaming clustering / open-set / continual learning), one external
   streaming benchmark, then the chained-15 stack comparison.
5. Carried: 4 pre-existing test_v2 failures (bisect someday).

## Pointers

- Gates: `run_m2_mixed`, `run_m3_settlement`, `run_m4_composer`,
  `run_m5_manifold`, `run_d18_merge`, `run_d20_membership` (+ probes
  `run_d20_probe`, `run_composer_genome`; review `run_review`).
- D20 code: `trioron/pcll/mixed.py` (`_margins`, `_assign`,
  `_consolidate`, `_gate_floor`, `_skeptical`, `classify`),
  `division.py` (`try_merge`, the two-sample statistic),
  `manifold.py` (sketch merge).
- The day's laws live in: design-doc D20 register row; spec §10.7
  membership-defaults paragraph; this file's "two laws" section.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, WSL2, `python3`, `OMP_NUM_THREADS=8`.
  Full gate battery ~15 min; tests ~12 s.
