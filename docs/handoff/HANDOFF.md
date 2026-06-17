# Trioron Handoff

**Session date:** 2026-06-17
**Session number:** 040
**Session title:** **Conjoined-twin CONV: the spawn rule that makes the
CONV gene non-trivial — mechanism + proposer VALIDATED, but chained-15
correctly REJECTS it (single-layer, centered, perception-saturated).** A
design + experiment session, no package code touched. The table-vs-image
thread (Rocky): the substrate is a flat *table* learner — readout sums over
columns (permutation-invariant), division judges one column at a time, so
image geometry is invisible. Encoding position into a column *value* (ramp /
NJ-tree / chord / stereo) is inert — geometry is a *wiring + cross-column
op*, not a value. Three fixes put that wiring in the organism: (1) a
coordinate field `f→(row,col,orient,scale)` — a KNOWN exact lookup for our
senses; (2) a proposer (the §3.4 council trial-vote); (3) weight-tying via
**conjoined twins** (Rocky). Key mechanism finding: `conv.py` REDUCES TO
LINEAR at own-root, so a lone CONV cell is linear-in-disguise — convolution
needs ≥2 cells sharing a `lineage_root`. Conjoined twins = born sharing a
root = real convolution. All three legs validated on synthetic data; on the
REAL chained-15 sensor field the proposer correctly REJECTS CONV (single
conv layer doesn't beat raw-pixel logreg on centered data — NOT proof
"conv can't help": depth untested).

---

## READ THIS FIRST

1. **DESIGN/EXPERIMENT session. No package code changed.** New: design §8 in
   `docs/design/progenitor_council.md`; three experiment scripts under
   `experiments/progenitor/`. The conjoined-twin CONV spawn rule is the new
   primitive; it is VALIDATED as a mechanism but NOT promoted into `trioron/`.
2. **The mechanism truth that makes the rule necessary:**
   `trioron/phenotype/conv.py` (its own "Reduction guarantee," lines 15–20):
   a CONV cell that is its own `lineage_root` reads its own weights →
   `y=b+Σw·a`, **byte-identical to LINEAR**. Convolution does not exist until
   **≥2 cells share a root**. So a council CONV win, as specced, spawns a
   *linear cell in disguise* — the CONV gene is dead on arrival. The
   **conjoined-twin** rule (spawn a weight-tied pair/cohort) is what fixes it.
3. **DRIFT NOTE (Rocky caught it):** the first chained-15 run (C=12) concluded
   "proposer honestly rejects CONV / perception not the bottleneck" — dressed
   as confident, but the conv was UNDER-POWERED (one thin 12-ch layer). The
   corrected C=48 run AGREES (reject holds), so the answer was right, but the
   *confidence was unearned*. Discipline: **do not let an experiment that
   conveniently confirms the standing keystone off the hook for being
   under-powered.** This is confirmation drift, not architecture amnesia.
4. **DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM):**
   `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
   `trioron/viz/export.py`; `.claude/`, `runs/` untracked. Two stalled s039
   raw logs still untracked. This session's new files SHOULD be committed
   (design §8 + 3 scripts + this handoff).

---

## What was built

**`docs/design/progenitor_council.md` §8 — Conjoined-twin CONV spawn rule.**
The table-vs-image diagnosis; the three fixes; the `conv.py` reduction
justification; the rule (spawn a tied cohort, each twin reading the raw
input columns in its patch in matched tap-order, twins all CONV phenotype
sharing one `lineage_root`, outputs feed the quantizer); non-negotiable
conditions (different translation-related patches; both CONV); validation
results table.

**`experiments/progenitor/test_conjoined_conv.py`** — hand-wired mechanism
test against the REAL `conv.forward_batch`:
- CHECK 1 reduction PASS — lone CONV == linear (max|Δ|=4.8e-7).
- CHECK 2 PASS — translation equivariance exact + gradient tying (twin1's
  loss reaches twin0's kernel).
- CHECK 3 PASS — tied train 1.000/test 0.977 vs untied 0.571/0.273 on a
  held-out-position translation task. **Side-finding: the benefit only
  appears with a translation-invariant POOLING readout (global-max), never a
  per-position head (which anti-generalized to 0.26). "The consumer must pool
  too." Our PCLL matched filter already sums over columns → compatible.**

**`experiments/progenitor/conv_proposer.py`** — the §3.4 trial-vote +
coordinate field, on synthetic 2-D data. `tile_patches` (the `f→(r,c)`
field), `spawn_conv_cohort` (tied/untied), `propose` (CONV-cohort vs LINEAR
on held-out competence). DATA-TYPE ADAPTIVITY PASS: SPAWN on real grid
(adv +0.57), REJECT when columns shuffled (conv→chance 0.517). Vote needs a
COMPETENCE bar (conv > chance+margin), not just "beats linear" (linear can
go below chance and a chance-level conv would wrongly "win").

**`experiments/progenitor/conv_proposer_chained15.py`** — proposer on the
REAL 30-way chained-15 sensor field (env-tunable `CP_C/CP_K/CP_STRIDE/
CP_EPOCHS`). GRADIENT-BASED regime (Adam on kernels) — NOT the gradient-free
PCLL pipeline that produces 0.736/0.446.

## The numbers (real chained-15, 30-way, gradient-based, n=2 seeds)

| arm | C=12 REAL | C=48 REAL | C=48 SHUF | ~params |
|---|---|---|---|---|
| raw-logreg | 0.798 | 0.796 | 0.794 | 23.5k |
| conv-maxpool | 0.432 | 0.683 | 0.639 | 2.6k |
| conv-flatten | 0.763 | 0.754 | 0.762 | large |

**Vote REJECTS CONV at both channel counts.** Reading: (1) 4× channels did
NOT let conv cross logreg → reject is not a channel-count artifact;
(2) C=48 conv-maxpool real>shuf (+0.044) → locality IS used, just not enough
on CENTERED data; (3) conv-maxpool = 86% of logreg accuracy at ~11% params
→ an accuracy-per-param vote (the 50K envelope) MIGHT flip — UNSETTLED;
(4) **untested lever: DEPTH** (conv→pool→conv) — all of this is SINGLE-LAYER.
Honest claim: *single-layer conv doesn't beat logreg on centered chained-15*
— consistent with the s039 keystone (perception not the chained-15
bottleneck) but does NOT close depth.

## NEXT (priority)

1. **Decide CONV's value task.** chained-15 is the WRONG demo (centered,
   perception-saturated, single-layer conv loses). To show CONV *helping*,
   use a TRANSLATION-STRUCTURED task: CIFAR, shifted/translated-MNIST, or the
   embodied/Atari arc (objects move, not centered). The synthetic test
   already proves the mechanism; we need a real positive.
2. **Test DEPTH before any "conv can't help" conclusion** —
   conv→pool→conv on chained-15 (the §8 "later stage": layer-1 feature map
   must carry positions too). Single-layer is not the whole story.
3. **Then choose the package home & promote** (only after a real positive):
   the §3 council scaffold (unbuilt) OR the §7 branch-id sheet (the
   "image-localized lateral event" — a lateral growth that mints a conjoined
   CONV cohort when the spatial trial-vote wins). The three primitives are
   ready; promotion waits on validation, per §7 discipline.
4. **(Open question) gradient-free conv.** The proposer/cohort learn the
   kernel by Adam. The PCLL chained-15 organism is gradient-free. Whether a
   conv kernel can be learned gradient-free (lock-in / Hebbian) is unsolved
   and gates any drop-in to the matched-filter pipeline.

## OPEN / unresolved

- **Should the vote weigh accuracy-per-param?** Under the 50K envelope,
  conv-maxpool's 86%-accuracy-at-11%-params is attractive; the current vote
  (raw accuracy) ignores it. Design decision pending.
- **Depth untested** (see NEXT#2) — the honest gap in the "conv doesn't help
  chained-15" claim.
- s039 carry: plain-gabor discrepancy (0.712/0.385 vs s038 0.690/0.304) still
  unexplained; chained-15 over-segmentation (cap=128) and the generative
  mean-template readout (−0.32) remain the s039 keystone NEXTs, untouched
  this session.
- All conv numbers are gradient-based, n=2 seeds.

## State of the build

- Branch `progenitor-council`. Package code UNTOUCHED; gate battery green
  (from s036). New files staged for commit this session: design §8 + 3
  experiment scripts + this handoff. DO-NOT-COMMIT carries left alone.
- Run cost: synthetic tests seconds; chained-15 C=12 116s, C=48 756s (CPU,
  OMP_NUM_THREADS=8).

## Pointers

- **Design:** `docs/design/progenitor_council.md` §8 (conjoined-twin rule +
  validation); §3.3–3.4 (council / trial-vote); §3.6 (positional sensors =
  the coordinate field); §7 (branch-id sheet, supersedes council/divide).
- **Conv primitive:** `trioron/phenotype/conv.py` (weight-tying by
  `lineage_root`+tap; the LINEAR-reduction guarantee, lines 15–20).
- **Arena wiring:** `trioron/core/arena.py` (`add_edges`, `lineage_root`,
  `alloc`); `trioron/core/scheduler.py` (`Bucket`; RECEPTOR injection
  overwrites activations with PHASE at line 292 — conv must read RAW input,
  not receptors).
- **Experiments:** `experiments/progenitor/test_conjoined_conv.py`,
  `conv_proposer.py`, `conv_proposer_chained15.py`.
- **Real chained-15 data:** `trioron.legacy.donorkit.datasets`
  (`DatasetBundle`, `chained_15_specs`, `build_task_views`); 30 global
  classes, 784-d, [0,1], `outputs/data/`.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`),
  `OMP_NUM_THREADS=8`. Data `outputs/data/`.
- **Regime distinction (DO NOT CONFLATE):** the chained-15 headline
  0.736/0.446 is the GRADIENT-FREE PCLL organism (`run_pcll_chained15.py`).
  This session's conv work is GRADIENT-BASED (Adam). Different pipelines.
- **Model note:** lead with computational framing (convolution = weight-
  shared local filter; weight-tying = parameter sharing; trial-vote =
  competence-on-held-out; translation equivariance) — biological metaphors
  ("twins", "eyes") can trip mid-session downgrades.
