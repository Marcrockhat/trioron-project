# Trioron Handoff

**Session date:** 2026-06-13
**Session number:** 034
**Session title:** **COMPONENTIAL SEMANTICS (the s029 backlog item) BUILT +
LARGELY FALSIFIED: the modifier classes Rocky predicted (chicken-duck,
dog-goat) DO appear unsupervised and allowing them is worth +0.13/+0.16 —
but their NAMES are label information (three label-free readouts
falsified). Previous session (would-be s034) was killed before any work
landed; the M1–M6 check + bench-15 trials Rocky asked for there were NOT
done and left no trace.**

---

## READ THIS FIRST

1. **Governing docs:** `docs/design/mixed_stream_growth.md` (D11–D20
   built+gated, D21 nursery proposed) and spec §10.6–10.10. The
   componential verdict lives in `docs/design/receptor_period_frustration.md`
   §10 (the s029 bullet, amended s034) and in
   `experiments/progenitor/run_componential.py`'s docstring.
2. **Everything from s033 stands unchanged** (rebuild M3–M6 shipped, two
   laws, membership-quality defaults ON, M2 0.829). This session added
   ONE probe + a design-doc amendment; no package code touched.
3. **A session was killed between s033 and this one.** It was asked to
   (a) verify M1–M6 implemented, (b) run trials on bench-15 (chained-15).
   Neither happened — git was bit-identical to s033's end. Those tasks
   are still open if Rocky still wants them.

## The session's finding (componential semantics, s029 backlog)

Probe: `experiments/progenitor/run_componential.py` (committed
`8d363b2`). Three seeds on data_hard (M2 default stack) + 3-epoch
10-animal taxonomy. Four results:

1. **The semantic-modifier classes EXIST, unsupervised** — taxonomy
   discovers stable blend classes exactly where Rocky predicted:
   chicken:0.49/duck:0.48 (one class, stable across 3 epochs),
   goat:0.61/dog:0.39, cat:0.81/dog:0.19. data_hard: 14–16 true blends
   per seed among ~77 classes.
2. **Allowing them is worth +0.128 / +0.158** — scoring a prediction
   correct when the sample's species genuinely belongs (≥20%) to the
   predicted class's composition: data_hard 0.829→0.957, taxonomy
   0.785→0.943. Most blend-class "errors" are an artifact of forcing
   one species name per class.
3. **NO label-free readout recovers the names** — three falsifications:
   (a) signed neighbour-difference projection (the s029 design):
   precision 0.00 — prototypes self-contaminate (a blend that is its
   species' only class IS the prototype); (b) mixture decomposition
   over LOO pure prototypes (templates are phasor MEANS → mixtures are
   linear, the math is right): precision 0.00 — incoherent disruptor
   pollution averages to ZERO phasor (invisible in template space) and
   species prototypes are mode-smeared on multimodal worlds; (c) buffer
   crossmatch (re-score own members against all templates): precision
   0.02 — consolidation already moved every member the matched filter
   would place elsewhere; the residual pollution is exactly what the
   organism's own geometry cannot see. Every label-free relational
   lift ≤ the random-modifier control (mean +0.006 vs ctrl +0.010).
   **THE LAW (semantic form of s033 law 1): a buffer's blend identity
   is label information.** The lever is a label source (reward/valence,
   the grounded-curriculum doctrine) — same lever as law 1.
4. **The vocabulary gap is structural** — "chicken-duck" cannot be
   named without duck existing as a concept, and division never
   separates them (3 epochs) because the FEATURES genuinely don't:
   the single class is the correct perceptual taxon; two names is
   human nomenclature finer than the percept (consistent with Rocky's
   perceptual-taxonomy-only ruling).

## Current gate battery (unchanged from s033, all green)

| gate | number | note |
|---|---|---|
| M2 mixed-stream | 0.829 | defaults stack; arrest tail ≤10%, ≤96-mode bound |
| M3 settlement | 0.924 | pinned config |
| M4 composer | 0.729 > 0.620 | quantum-derived sentinel |
| M5 manifold | 0.836 end, −0.013 decay | truth map per epoch |
| D20 membership | +0.006 acc, purity ↑, reject 65.6%@100% | triplet |
| tests | 107 + same 4 carries | not re-run this session (no package code touched) |

## State of the build

- Branch `progenitor-council`, committed through `8d363b2`
  (componential probe + design-doc amendment), pushed.
- **DO-NOT-COMMIT carries (unchanged):** `trioron/bases/developmental.py`,
  `trioron/lifecycle/developmental.py`, `trioron/viz/export.py`;
  `.claude/`, `runs/` untracked.
- `trioron/pcll/mixed.py` split still OVERDUE (~1000 lines).

## Open questions / NEXT (priority order)

1. **The killed session's tasks, if still wanted:** M1–M6
   implementation check + bench-15 (chained-15) trials.
2. **D21 — the NURSERY (proposed, needs Rocky):** refused members
   accumulate in an unlabeled DIVISIBLE buffer. Now doubly motivated:
   it is also where blend/unknown matter could later be re-examined
   when a label source exists.
3. **Componential semantics follow-up (needs Rocky's call):** the
   +0.13/+0.16 is real but requires a label source. Options: (a) park
   it until reward/valence labels exist (grounded curriculum), (b)
   deployment answer = class identity + reject (already shipped, D20c)
   and let the consumer own naming, (c) explore composer/dendrite
   pressure to separate chicken-duck (probably wrong: the percept
   doesn't support it).
4. **mixed.py split + constructor profiles.**
5. Carried: dendrite settlement threshold; ATTENTION/CONV/RECURRENT
   composer semantics; PCLL-path compaction for retired astrocyte
   rows; buffer drop-at-freeze; external validation arc; 4 pre-existing
   test_v2 failures.

## Pointers

- This session: `run_componential` (probe + full verdict docstring);
  `receptor_period_frustration.md` §10 componential bullet (amended).
- Gates: `run_m2_mixed`, `run_m3_settlement`, `run_m4_composer`,
  `run_m5_manifold`, `run_d18_merge`, `run_d20_membership`.
- The laws: s033 two laws in the D20 register row + spec §10.7; the
  s034 semantic corollary in the componential bullet + probe docstring.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, WSL2, `python3`, `OMP_NUM_THREADS=8`.
  Componential probe ~6 min; full gate battery ~15 min; tests ~12 s.
