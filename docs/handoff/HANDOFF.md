# Trioron Handoff

**Session date:** 2026-06-11
**Session number:** 028
**Session title:** **Diagnosed the branch/GCU growth to a hard wall (capacity-limited
+ GCU detonation), built the asymmetric-fan fix / class-driven growth / conserved
council vote economy / chance-anchored frustration gate along the way — then
co-designed the replacement: a receptor → period-sweep → trig lock-in → one-cycle
frustration → empty-class-as-stress architecture. Restored dendrite to z+z².**

---

## READ THIS FIRST

The forward direction is **`docs/design/receptor_period_frustration.md`** — the full
s028 design, written in detail because it lives almost entirely in the s028
conversation. Read it before anything else. This handoff is the index + state; that
doc is the substance.

---

## The arc (what happened, in order)

This was a long design+diagnosis session. It started on the s027 branch architecture
and ended having decided to replace it.

1. **Asymmetric-fan fix.** Rocky asked why the branch net's accuracy was capped. Audit
   (`audit_branch.py`) showed it was **perfectly symmetric**: every branch read all 12
   perception + wrote all 32 outputs (all-to-all readout) → a rank-12 linear map →
   ceilinged at **0.409**. Fixed with **frustration-targeted sparse input**
   (`_frustrated_perc`: each branch reads the dims most correlated with residual error)
   + **wide axial fan** (deepening GCU reads its running feature + fresh perception).
   Lifted to only **0.484** — depth piled into one column.
2. **Class-driven growth** (`grow_class_driven`, `run_class.py`). Grow at the worst
   class (by per-class residual), widen→deepen. Distributed depth across 8 columns →
   **0.762** (≈ council 0.78). But it gates growth on `acc()` (test) and hit the budget
   still climbing.
3. **Clustering reality check** (`kmeans_probe.py`). The data is **96 modes, not 32
   classes** (32×3 Gaussians + 5 wide disruptors). k-means k=32 ARI 0.204; **hierarchical
   pure clusters cap at 27/32** (the 5 disruptors form ZERO pure clusters at any k).
   k-means k=96 purity 0.798, GMM-96 0.870 ≈ Bayes 0.938. **Per-class residual = mean CE
   ∈ [0,∞)** (chance = log(C)=3.466, NOT 0–1). So `mean+k·std` is baseless; a
   **chance-anchored** `f_c = CE_c/log(C)` is the right scale.
4. **Chance-anchored frustration gate** (`frustration_gate.py`). `frustrated = plateaued
   AND CE/log(C) > τ`, train/grow/stop trichotomy. τ=0.5→**0.582**, τ=0.25→**0.707**.
   Notably **auto-retired exactly the 5 disruptors** as irreducible — rediscovered the
   clustering finding from loss dynamics alone.
5. **Council vote economy** (`step4_grow.py`, Rocky's spec). Replaced the fixed-penalty
   exhaust with a **conserved per-cell vote pool**: 20 council cells, Σ votes = 20; on
   failure a phenotype loses 1 vote spread to the other 16 (−1/|P| per its cell), on
   success the mirror (+1 pulled from the others). Floor = 1 vote/group (`GROUP_VOTE_FLOOR`).
   Stop = plateau-patience (no Bayes — "we can't use Bayes as a cap, not all data has one").
   Works mechanically (Σ conserved); but **over-invests in the opening favourite**
   (attention) → 0.788, sticky ties.
6. **Capacity verdict** (`run_capacity_check.py`). Grow → freeze → long-train 6000 steps
   → asymptote **0.569**, and it **crashes** mid-train. **More epochs make it worse.**
   → CAPACITY-LIMITED.
7. **GCU-detonation diagnosis** (`run_gcu_diag.py`). Deepest column depth **10**.
   `σ(z)=z·cos z` is expansive for |z|>1; the stack compounds L3:216→L4:622→**119,400 by
   step 750** → acc 0.085 (chance) → **permanently dead**. **Weights stay ~5** → it's
   *forward activation amplification*, not weight blow-up → **grad clipping won't fix it**;
   RMS-norm the pre-activation does. **No rollback** keeps the regressive deep cells.
8. **Bias / zero-init / second-pass clarifications** (Rocky's questions). Linear cell's
   bias is aliased by the output bias (linear-on-linear degeneracy) — earns value only via
   weights. Zero-init: safe for outputs (label-driven), **fatal for hidden cells on both
   sides** (stillborn, zero gradient). We do train before measuring impact.
9. **The new architecture** (co-designed, captured in the design doc). Receptor → period
   sweep → trig lock-in → one-cycle frustration / √N noise floor / graded output → empty
   class as terminal-AND-stress → two growth drivers via the vote economy → habituation.
   **Restored dendrite to z+z²** (the trig oscillators take over, at the input only).
   **Built + verified the receptor** (`receptor.py`).

---

## State of the build

- **Branch `progenitor-council`.** This session's work is **committed (`e424c52`) and
  pushed** — see below.
- **DO-NOT-COMMIT carries (still excluded, verified):** `trioron/bases/developmental.py`,
  `trioron/lifecycle/developmental.py`, `trioron/viz/export.py`. `.claude/`, `runs/`
  untracked. Do not commit these.
- **Live new code (committed):** `receptor.py` (verified), `frustration_gate.py`,
  `kmeans_probe.py`, `audit_branch.py`, `run_capacity_check.py`, `run_class.py`,
  `run_gcu_diag.py`; modified `step4_grow.py` (vote economy), `step5_branch.py`
  (asymmetric fan + class-driven + gate), `dendrite.py` (z+z² restore).
- **The branch/GCU growth path is superseded** (capacity-limited + unstable). Keep for
  reference; do not invest more in deepening thin GCU columns.

## Decisions made (the *why*)

- **Replace gradient depth with a perceptual loop.** The branch net can't represent the
  target (0.569 asymptote) and detonates when deep. So the forward direction is the
  receptor/period/lock-in design, not more growth on the old substrate.
- **gcos/sinc/tan go right after perception, never stacked in depth** — that is the
  GCU-detonation fix (no deep oscillator chain to compound).
- **Dendrite back to z+z²** — the trig units take the oscillatory role at the input; z²
  in depth is bounded enough with the dendrite's per-branch use, and z·cos z stacked was
  the detonator.
- **Frustration = one cycle (period); resolution = margin over the √N noise floor** — a
  parameter-free basis (random-walk statistics), replacing τ / mean+std / Bayes-gap.
- **Empty class = the noise floor**, dual-purpose: valid terminal answer AND stress
  driver (sensory deprivation is aversive → drives growth). Bounded by **habituation**
  (decay on fruitless growth); skull = envelope, uncapped for the trial.
- **Receptor = per-sample, history-free, scale-invariant** (gain adaptation); negatives
  become the new MIN; partition = phase (2π/1000 per quantum).

## Open questions / next-up (priority order)

Build order is in the design doc §9. Concretely:

1. **Feed module** — class-as-range-schedule + the period sweep (design §3), reproduce the
   chicken/dog example as a unit test, on a 2-class/4-feature toy.
2. **Trig units + lock-in accumulator** (§4) — per-feature gcos/sinc/tan on the receptor
   phase; phase integrated over the stream (recurrent accumulator); per-class amplitude.
3. **Smallest de-risk test first:** coherent-vs-empty stream → does coherent clear the √N
   floor and empty not, **without labels**? If yes, the core mechanic is real.
4. **One-cycle resolution** (§5): margin-over-√N-floor, graded "more likely X than Y",
   empty = below floor.
5. **Empty-stress + two-driver growth + habituation** (§6–8) into the vote economy.
6. Open design Qs (design §10): lock-in carrier (one vs per-class); residual test for
   where-to-grow; RMS-norm if any oscillator ever sits in depth.

## Pointers

- **Design (read first):** `docs/design/receptor_period_frustration.md` (full s028 design).
- **Receptor:** `experiments/progenitor/receptor.py` (built, verified).
- **Diagnostics to re-run if needed:** `run_gcu_diag.py` (detonation), `run_capacity_check.py`
  (capacity wall), `kmeans_probe.py` (96-modes / 27-of-32 ceiling).
- **Superseded-but-referenced:** `step5_branch.py` (class-driven 0.762), `step4_grow.py`
  (vote economy), `frustration_gate.py` (chance-anchored gate).
- Spec §3.2–3.6 (phenotypes), §4.2 (native FrustrationDetector — note: progenitor loops do
  NOT use it; the new design replaces it with the period/√N test).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`. Python 3.10.12,
  torch 2.11.0, WSL2, 12 cores, **`python3`** (not `python`), `OMP_NUM_THREADS=8`.
- Long full-batch runs auto-background; rely on task-completion notifications. Growth
  runs are slow because of step count (~70 events × 120 steps + recompiles), NOT model
  size — the grown net is ~101 cells / ~1.3K used params (35K is preallocated capacity).
- **The grown organism collapses irreversibly once a deep GCU column detonates** — if you
  re-run the old branch path and see acc → ~0.08 frozen, that's the detonation, expected.
- Heavy, dense session (diagnosis + a full redesign). All work committed + pushed; safe to
  break here. Design doc is the source of truth for the new architecture.
