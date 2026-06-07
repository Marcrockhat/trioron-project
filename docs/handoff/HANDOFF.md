# Trioron Handoff

**Session date:** 2026-06-07
**Session number:** 022
**Session title:** **Numa-gated growth recovered + 6-class disruptor dataset + NEW progenitor–council architecture design (the big one).**

Resumed after a RAM-killed session that left `numa_growth.py` /
`calibrate_growth.py` uncommitted with **no saved logs**. Recovered the
Numa results by re-running, committed them, then extended the
chicken-goat probe (disruptor **dog**, separated **elephant**), then —
the bulk of the session — a long design conversation with Rocky that
produced a **new architecture direction**: the **progenitor–council**
model. Rocky's framing: the v2 growth mechanism is biologically wrong
(mature neurons divide; growth is undifferentiated), and the assistant
had been substituting the conventional ML decision for his. The full
design is captured in **`docs/design/progenitor_council.md`** — read it
first next session; this handoff summarises.

> Session 021 (`dc89713`) diagnosed the frustration trigger misfiring on
> LR noise and left "fix the trigger (a) vs validate-first (b)" open.
> This session answered it (the Numa controller does both) AND then
> reframed the whole growth question around stem-cell biology.

## Summary

1. **Recovered + committed Numa-gated growth** (`e26d0a1`). The killed
   session's `NumaGrowthController` separates **Numa (durable held-out
   learning, any means) from growth (a probe lever)**. Re-ran both tasks
   (n=5):
   - **chicken-goat 4-class** (capacity sufficient): grows 3–5 cells
     (NOT 40=cap), test **0.953** ≈ ceiling 0.952, self-terminates 5/5.
   - **rank-ladder** (capacity genuinely the wall): grows **8/8/8/8/8**
     (~C−1 rank), test **0.990** ≈ ceiling 1.0, STOP 5/5.
   - Symmetric fix to s021: stops over-growing where capacity suffices,
     still grows-and-helps where it's the bottleneck.

2. **Extended the dataset** (`chicken_goat.py`, uncommitted at the time
   of writing — commit with this handoff). Refactored `SPECIES` to a
   `Normal`/`Uniform` distribution registry; `bayes_accuracy` uses each
   class's true likelihood.
   - **dog = `Uniform(2, 85)`** — disruptor (chihuahua→great dane),
     overlaps cat+goat, **non-linearly-separable** Bayes partition.
   - **elephant = `Normal(5000, 800)`** — larger than cow, clean (recall 1.0).
   - 5-class ceiling 0.822; **6-class ceiling 0.852**.

3. **Architecture finding on the disruptor**: a linear readout caps at
   **0.784** (dog & goat each win two disjoint Bayes intervals → not
   linearly separable). The grown 11-cell *linear* net hits 0.792 — at
   the *linear* ceiling, not the Bayes 0.822. The gap is **nonlinearity,
   not cells, not under-training**. Minimal solution: a 1→2..4→5 net with
   a nonlinearity, ~19–33 params (ReLU h=4 → 0.814 ≈ ceiling); 6
   thresholds as a pure heuristic. This is what motivated the redesign.

4. **THE BIG OUTPUT — progenitor–council design** (`docs/design/progenitor_council.md`).
   Decisions locked this session (see that doc for the full model):
   - **Germline (progenitor + council, never frozen/pruned) vs soma**
     (differentiated branch cells, consolidatable).
   - **One wide progenitor** = retinal compressor; **working resolution
     locked at 1,572,864 (1536×1024, 1.5 Mi)** — above human optic nerve
     (~1.2M ≈ HD), below Full HD.
   - **Council = 5 phenotypes × 4 = 20**, even for ties, never pruned;
     phenotypes = existing `EXPRESSION_GENES`
     (LINEAR/DENDRITE/CONV/ATTENTION/RECURRENT). The vote **is** input
     classification.
   - **Differentiation = trial-vote**: progenitor spawns one daughter
     per phenotype, they compete on local learning signal, data topples
     the balance, winner commits to the branch, progenitor stays plastic.
   - **Mature neurons don't divide**: division plasticity-gated (λ lock).
     **P_divide(z,u) = [ln(1/z)/ln(1/z_inner)]·(1−u)** — ln niche over
     SHALLOW depth (tens of layers, not thousands), resource penalty at
     all layers, no base tax at u=0.
   - **Perception learns the transform** (log = magnitude-split hidden
     layer); sensor cells chunk input, take over the input layer; stop =
     council consensus ("make sense of the input").
   - **Capital reframe**: growth is working capital, not to be minimised
     during learning; minimum-param is the *consolidation* target. The
     s021 error was undifferentiated growth, not too-much growth.

## Headline numbers

| run (n=5) | grown | test | ceiling | STOP |
|---|---|---:|---:|---|
| numa chicken-goat 4cls (log) | 3–5 | 0.953 | 0.952 | 5/5 |
| numa rank-ladder | 8 | 0.990 | 1.000 | 5/5 |
| numa 5cls disruptor-dog (linear cells) | 3–5 | 0.792 | 0.822 | 5/5 |
| linear readout (10 param) on disruptor | — | 0.784 | 0.822 | — |
| ReLU 1→4→5 (33 param) on disruptor | — | 0.814 | 0.822 | — |

## State of the build

- Branch `v2.0-scaffold`. HEAD before this handoff = `e26d0a1` (numa files).
- **Committing with this handoff:** `experiments/growth_exercise/chicken_goat.py`
  (dog+elephant, distribution registry) + `docs/design/progenitor_council.md`
  (new) + this handoff.
- **DO NOT COMMIT / DO NOT TOUCH** (pre-existing, carried since s005):
  `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
  `trioron/viz/export.py`. NOTE: these hold the existing stem/morphogen
  prototype (`MorphogenField`, `spawn_stem`, `differentiate`) the design
  builds on — read them, don't commit the working-copy edits.
- `runs/`, `.claude/` not committed.
- **No core code written for the new design yet** — it is design-only.

## Decisions made (the why)

- **Recover-by-rerun, not reconstruct** — killed session left no logs;
  the code reproduces in ~2 min, so re-ran rather than guess.
- **dog as `Uniform`, not a middle Gaussian** — Rocky wants a disruptor
  spanning the whole breed range, not a tidy class; the hard upper edge
  at 85 kg is what creates the non-linear (double-interval) partition.
- **ln over tan for P_divide** — ln gives the niche shape (steep at
  input, graded tail) naturally; tan is either a pole-cliff or a
  near-linear ramp.
- **Working resolution 1.5 Mi** — above human optic nerve so we're not
  under-human, below Full HD which Rocky ruled out; 1536×1024 is
  compute-clean (1.5×2²⁰, 3:2).
- **Council 4/phenotype, one wide progenitor** — Rocky's explicit calls.
- **Capture before continuing** — the design lived only in chat; per the
  cross-PC handoff rule it had to be written down before going further.

## Open questions / next-up

1. **Build order for the design** (`progenitor_council.md` §4 lists the
   gaps). Natural first slice: **plasticity-gate `divide()`** (refuse
   mature-cell division) — small, and it's the clearest correctness fix.
   Then data-driven differentiation (council trial-vote) on the 6-class
   disruptor as the probe (should grow a DENDRITE for dog, stay linear
   elsewhere).
2. **Open knobs** (design §6): niche steepness γ; multiplicative vs
   subtractive resource tail; council vote integration window; retinal
   compression structure.
3. **Carried:** the 2 pre-existing `test_lifecycle` failures
   (`test_growth_trigger_logic`, `test_dream_locks_eligible_cells`).
4. **Carried from s020:** project-wide growth audit — but note the new
   framing: prior growth ran a *broken* (mature-cell-cloning) mechanism.

## Pointers

- **Design doc (READ FIRST): `docs/design/progenitor_council.md`.**
- Numa controller: `experiments/growth_exercise/numa_growth.py`
  (`NumaGrowthController`). Calibration harness: `calibrate_growth.py`
  (K/P knobs, not yet re-run this session).
- Dataset: `experiments/growth_exercise/chicken_goat.py` — `SPECIES`
  registry (`Normal`/`Uniform`), `make_animals`, `bayes_accuracy`.
  Run: `python3 -m experiments.growth_exercise.numa_growth --task
  chicken-goat --species chicken cat dog goat cow elephant --log --seeds 5`.
- Existing stem/morphogen prototype: `trioron/lifecycle/developmental.py`
  (`MorphogenField`, `differentiate`, `spawn_stem`),
  `trioron/core/epigenome.py:21` (`EXPRESSION_GENES`). `divide()` +
  (orphaned) `check_growth_trigger`: `trioron/lifecycle/grow.py`.
- Manual §4 (axes), §5 (CL machinery); spec §5.7 (developmental program).
- Memory: [[growth_sink_fix]] (s020), [[selective_quad_growth]],
  [[substrate_was_purely_linear]], [[triparametric_node_lambda]] (λ =
  plasticity), [[dysplastic_axis_criterion]].

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `v2.0-scaffold`. Python
  3.10.12, torch 2.11.0+cu130, WSL2, 12 cores. `python3`,
  `OMP_NUM_THREADS=8`.
- Exercise runs are fast (n=5 × 8000 steps ≈ 60–145s).
- **RAM/session note:** a prior session was RAM-killed mid-run; if it
  recurs, the growth scripts are cheap to re-run and leave no logs by
  default — redirect to `runs/` if you want them kept. Remote/mobile
  Claude Code sessions still occasionally drop (s021 diagnosed it as a
  >10-min network outage under WSL2 mirrored networking; Rocky keeps
  mirrored mode).
