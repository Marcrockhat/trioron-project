# Trioron Handoff

**Session date:** 2026-06-13
**Session number:** 034
**Session title:** **THE LABEL ARC SHIPPED END TO END (componential
falsification → D22 tap bank → D22b counts → D22c label-supervised
consolidation, ALL DEFAULT): strict 0.829→0.957 at full label coverage
(Bayes 0.993). Then bench-15: legacy stack reproduced (0.958/0.551);
PCLL FIRST CONTACT on chained-15 (raw pixels, genesis, single pass):
0.552/0.172 — envelope-allocation under cap pressure is the measured
open problem. Rocky's standing correction: STICK TO THE DESIGN
PRINCIPLE (substrate adapts to input + resources; reasonable accuracy,
minimum forgetting) — no harness crutches. NEW SESSION RECOMMENDED.**

---

## READ THIS FIRST

1. **Rocky's design-principle correction (binding, this session):**
   trioron is a substrate that adapts to INPUT and RESOURCES while
   delivering reasonable accuracy and minimum forgetting. The session
   drifted toward harness crutches (hand-built projections, knob
   sweeps) more than once; the correction is to let the substrate
   adapt within its laws and report what it does. Apply this to every
   decision before reaching for a constant.
2. **Governing docs:** `docs/design/mixed_stream_growth.md` (D11–D22;
   the D22 row carries the tap-bank/counts/supervision measured
   record), spec §10.6–10.10 + §9.15 (`pcll/labels.py` row),
   `receptor_period_frustration.md` §10 (componential bullet, amended
   s034). Probe records live in the runner docstrings:
   `run_componential.py`, `run_label_taps.py`, `run_pcll_chained15.py`.
3. **A session was killed between s033 and s034** — its tasks (M1–M6
   check + bench-15 trials) were completed THIS session instead.

## Arc 1 — componential semantics: built + falsified (commit 8d363b2)

Blend classes EXIST unsupervised (taxonomy chicken .49/duck .48,
goat .61/dog .39; data_hard 14–16 true blends/seed) and allowing them
is worth +0.13/+0.16 — but three label-free naming readouts all
falsified (projection 0.00 precision; LOO mixture 0.00; buffer
crossmatch 0.02). **Law (semantic form of s033 law 1): a buffer's
blend identity is label information.**

## Arc 2 — D22 label channel, all three stages DEFAULT (Rocky-approved)

Rocky's design: labels enter "at a different frequency" — reference
carriers, multiply-at-deposit then integrate (`pcll/labels.py`,
`observe(x, labels=)`). Stages, each gated in `run_label_taps.py`:

- **D22 tap bank:** per-label demodulated prototypes. Gate A
  non-disturbance EXACT (templates bit-identical, labels-off ≡
  annotate-only). Taxonomy headroom recovered at 5% coverage
  (member-mix 0.948 = oracle 0.945); "chicken-duck"/"goat-dog" named
  at 100%. Mode-smearing limit on multimodal worlds recorded.
- **D22b (class × label) counts:** naming at the organism's own
  grain; count-majority 0.826 ≈ eval-side map 0.829 — **the
  self-label map is organism-internal**. Division children inherit
  counts by buffer-side fraction (tail-born-unnamed fix; the
  estimator smears — per-member tags are the exact repair).
- **D22c label-supervised consolidation (DEFAULT ON):** per-member
  TAGS ride buffers through division/merge/consolidation/annealing;
  a labeled row EVACUATES a class when its label holds <
  SUPERVISE_FRAC=0.4 of the class's tagged members (swept .2/.3/.4/.5
  → .895/.921/**.957**/.950), to its label's best tag-mature home
  (majority ≥0.6, n≥3). **data_hard strict 0.829 → 0.957 @100%
  coverage (Bayes 0.993), 0.873@20%, baseline ±0.007 @1–5%; purity
  .740→.813; classes 91 ≤ 96.** Three falsified rules recorded in the
  runner docstring (move-from-immature vacuums modes; refused-rescue
  force-feeds spray; LEDGER majorities are staleness-capped —
  routing majorities MUST read buffer tags). Genuine blends correctly
  left intact. Gate A redefined: annotate-only bit-identical;
  supervised must not regress.
- **Bayes context:** single-label 0.829→0.957 vs Bayes 0.993 (gap
  now 0.036 = fragmentation/crossing ambiguity — the H-space routing
  story). Taxonomy set-metric EXCEEDS single-label Bayes (0.948 >
  0.917): the discovered blend class sidesteps chicken/duck confusion
  — Rocky's "allow such new class" quantified.
- `_evidence()` chunking added (BIT-identical row-chunked matched
  filter; the image world's pooled broadcast was ~7 GB). M2
  re-verified 0.829 after.

## Arc 3 — bench-15 (the killed session's task)

- **M-battery re-verified on current tree:** M2 0.829 (bit-exact),
  M3 0.924, M4 0.729, M5 0.836, D20 pass. M1 = structural contract
  asserted inside M2; M6 = spec amendment (present).
- **Legacy bench reproduced** (`experiments/run_bench15_shim.py`
  aliases the moved modules; archive stays frozen): credit_manifold
  0.958±.003 task / 0.551±.007 full (record 0.962/0.568 was the
  cosine-head variant — within expected drift). Log:
  `outputs/axis6_credit_chained15_s034_run1.log`.
- **PCLL FIRST CONTACT on chained-15** (`run_pcll_chained15.py` —
  no historical record existed; the "0.386 class-sequential" number
  is a data_hard contrast line, NOT a bench-15 record): raw 784-d
  pixels → PerceptionGenesis (Rocky's correction — genesis faces the
  world; the hand-built LCN projection arms TILED to cap inside 1–2
  tasks and are kept only as a recorded comparison arm), single pass,
  no gradients, labels at 100% via D22 (parity: legacy arms are
  CE-supervised), class-sequential. **Seed 0: task-aware 0.552, full
  0.172, 128 classes.** Per-task: MNIST eroded to ~0.5, Fashion held
  .68–.98, letters 0.50/0.00/0.50/0.00/0.50 (two tasks own NO
  classes). Verdict through the design principle: genesis handled raw
  input; the organism self-managed at the envelope; but **the
  allocation law under cap pressure — who yields when the envelope is
  full — starves late arrivals and erodes the oldest tenants. That is
  the measured open problem.** Log:
  `outputs/pcll_chained15_s034_raw_smoke.log`.

## Current gate battery (all green)

| gate | number | note |
|---|---|---|
| M2 mixed-stream | 0.829 | re-verified bit-exact post-D22c + chunking |
| M3 settlement | 0.924 | re-verified |
| M4 composer | 0.729 > 0.620 | re-verified |
| M5 manifold | 0.836 end | re-verified |
| D20 membership | triplet | re-verified |
| D22 A/B/C/D/E | A exact; C 0.948=oracle@5%; D 0.909; E 0.957@100% | label channel |
| bench-15 legacy | 0.958/0.551 | reproduces record |
| bench-15 PCLL | 0.552/0.172 (n=1) | FIRST CONTACT record |
| tests | 107 + same 4 carries | tag-sync fix applied |

## State of the build

- Branch `progenitor-council`, committed through `9a85f8b`
  (componential `8d363b2`, D22 `6a0f200`, D22b `8f89414`, D22c
  `306bb8d`, bench15 `9a85f8b`), pushed after this rewrite.
- **DO-NOT-COMMIT carries (unchanged):** `trioron/bases/developmental.py`,
  `trioron/lifecycle/developmental.py`, `trioron/viz/export.py`;
  `.claude/`, `runs/` untracked.
- `trioron/pcll/mixed.py` now ~1200 lines — the split is URGENT
  (label machinery would move to a clean boundary with labels.py).

## Open questions / NEXT (priority order)

1. **Envelope-allocation law (from bench-15 first contact, needs
   design discussion with Rocky):** at CLASS_CAP, who yields? Current
   behavior: late tasks starve (no room to divide), old classes erode
   under consolidation churn. Candidate substrate-native directions
   (NOT harness knobs): engagement/recency-weighted husk retirement,
   D18 merge consumer ON (collapse duplicates to free rows), D21
   nursery, per-label envelope shares. This is the "adapts to
   resources" half of the design principle, now measurable on a real
   bench.
2. **PCLL bench-15 n=3 + manifold-off ablation** once (1) has a
   ruling — single-seed first contact needs σ.
3. **mixed.py split + constructor profiles** (now urgent).
4. **D21 nursery** (proposed, needs Rocky).
5. Per-member label tags as exact count repair (~0.05 set-metric);
   label-pressured division (recommend against — perceptual-taxonomy
   ruling). Carried: dendrite settlement threshold; composer
   semantics §4; PCLL-path compaction (lifetime memory — taps add
   vocab × dims); buffer drop-at-freeze; external validation arc;
   4 pre-existing test_v2 failures.

## Pointers

- Label channel: `trioron/pcll/labels.py` + the s034b/c blocks in
  `mixed.py` (`_supervise`, `_majorities`, `_sync_tags`, tags
  plumbing); gates `run_label_taps.py` (A/B/D/E + C descriptors).
- Bench-15: `run_pcll_chained15.py` (PCLL),
  `experiments/run_bench15_shim.py` (legacy via module aliases).
- Componential: `run_componential.py`.
- M gates: `run_m2_mixed`, `run_m3_settlement`, `run_m4_composer`,
  `run_m5_manifold`, `run_d18_merge`, `run_d20_membership`.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, WSL2, `python3`, `OMP_NUM_THREADS=8`.
  Label battery ~12 min; M battery ~15 min; legacy bench-15 ~12 min;
  PCLL bench-15 ~19 min/seed; tests ~10 s. Chained-15 data cached at
  `outputs/data/`.
