# Trioron Handoff

**Session date:** 2026-06-06
**Session number:** 021
**Session title:** **Growth probe (chicken-goat) — the frustration→growth trigger MISFIRES on optimization noise, not capacity.**

A step-by-step exercise driven by Rocky's intuition that "the architecture is
not growing as expected." Built a minimal 1-feature animals-by-weight classifier
and ran n=10 growth probes on it. The clean finding: on these low-dimensional
problems growth **never helps**, and when it *does* fire (high LR) it grows to
the budget cap for **zero gain** — the trigger keys on loss magnitude/noise, not
on whether added capacity actually reduced loss. This is the same failure the
s015 `selective_quad_growth` author patched around locally; the root is still in
`check_growth_trigger`. **No core code changed this session** — it's a
diagnosis, not a fix. The decision of whether to fix the trigger (a) or first
confirm it works where capacity is genuinely the wall (b) is **left open for
next session.**

> Rewritten in full every session; prior handoffs in git history. Session 020
> (`9d34086`) fixed the `grow.py` sink bug + `graft.py` non-leaf bias and left a
> project-wide growth audit as NEXT. This session is a small slice of that audit
> from a fresh angle (does growth fire for the right reason?).

## Summary

1. **Built `experiments/growth_exercise/`** (NEW, committed `2584b39`):
   - `chicken_goat.py` — N-class animals-by-weight dataset. `make_animals(species)`
     over a `SPECIES` registry (chicken 2.5kg, cat 4.6, goat 51, cow 495). One
     standardized feature (weight). `bayes_accuracy()` = the weight-only
     information ceiling (true class-conditional Gaussians, equal priors).
     `log=True` applies log(kg) before standardizing.
   - `run_chicken_goat.py` — feeds it to a v2.0 substrate (`construct` + `seeded`
     + frustration-gated `divide`, relying on the s020 forward-projection fix, NO
     manual re-wiring), reports epochs-to-convergence (vs the Bayes ceiling),
     final acc, and cells grown. `--species`, `--log`, `--no-grow`, `--lr`, etc.

2. **Trivial binary (chicken vs goat).** n=10, 30 epochs: converges in 2.8±0.9
   epochs to ~0.99, **grows 0 cells**. Correct — capacity is sufficient, growth
   stays quiet.

3. **4-class (chicken/cat/goat/cow) — the feature trap.** Stalls at **0.728**
   (Bayes ceiling **0.952**), **grows 0 cells**. 0.728 ≈ separating cow+goat+one
   small animal and collapsing chicken+cat (3 of 4 groups = 0.75). Cause: global
   standardization's std (211 kg) is **cow-dominated**, crushing chicken (x=-0.643)
   and cat (x=-0.633) to a **0.0099** gap (vs ~2.0 for chicken-goat). The
   separating information survives in raw kg (hence Bayes 0.95) but is destroyed
   by preprocessing. **The bottleneck is the FEATURE, not capacity** — growth
   correctly does nothing because no amount of cells adds information the input
   lacks.

4. **log-transform fixes the feature (Rocky's call, confirmed).** `--log` →
   chicken-cat gap 0.0099→0.284, 4-class acc **0.728→0.903** (+18pts), still 0
   growth. Monotonic transform ⇒ Bayes ceiling unchanged; only *usability* changed.

5. **Pushed to the ceiling.** The residual 0.903→0.952 gap was **under-training**,
   not capacity: `epochs=100, lr=0.005` → **0.951 ≈ ceiling**, 10/10 converged
   (~epoch 37), **0 cells grown**.

6. **THE FINDING — growth misfires on optimization noise.** At `lr=0.02` the
   substrate **grows 29→40 cells (hits the budget cap)** and accuracy is **no
   better** (0.941 vs 0.948 with growth OFF — slightly worse). Faster convergence
   (9 vs 37 epochs) comes from the **higher LR, not the growth**. Higher LR ⇒
   spikier loss ⇒ `FrustrationDetector` reads spikes as "stuck" ⇒ repeated
   `divide()` on a problem one linear readout already solves. **The trigger
   conflates a noisy-but-descending loss with a genuinely-plateaued one.**

## Headline numbers (4-class, n=10, log feature)

| config | epochs-to-conv | test acc | cells (start→end) |
|---|---:|---:|---:|
| 30ep, lr 0.005 | — (0/10) | 0.903 | 7→7 |
| 100ep, lr 0.005 | 36.7±3.5 | **0.951** | 7→7 |
| 100ep, lr 0.02, grow ON | 9.2±1.8 | 0.941 | 7→**40** (cap) |
| 100ep, lr 0.02, grow OFF | 9.2±1.8 | **0.948** | 7→7 |

Bayes ceiling (weight-only) = 0.952. Growth adds cost, never accuracy, on this task.

## What was done (files)

- **NEW `experiments/growth_exercise/{__init__.py,chicken_goat.py,run_chicken_goat.py}`**
  — committed `2584b39`. No core changes.
- **`runs/` (LOCAL, NOT committed):** background sweep logs only; nothing to keep.

**Pre-existing, STILL DO NOT TOUCH / DO NOT COMMIT** (carried since s005, uncommitted):
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`, `trioron/viz/export.py`.

## Key findings

1. **`check_growth_trigger` gates on frustration magnitude, not on whether
   capacity helped.** High LR alone manufactures the loss spikes that trip it →
   runaway growth (to the budget cap) for zero gain. This is the s015
   "rebuilding self-sustains frustration" issue, unfixed at the root.
2. **Debugging order for an underperforming grown net:** check the FEATURE first
   (here +18pts from a log transform), then OPTIMIZATION (the next +5 from more
   epochs), and only THEN suspect capacity — the one thing growth addresses, and
   the one thing none of these problems needed.
3. **Growth stays correctly silent when capacity is sufficient** (all low-LR
   runs: 0 cells) — the s020 forward-projection fix did NOT make growth trigger-
   happy; the misfire is purely the LR-noise→trigger path.

## Decisions made

- **Probe with a deliberately trivial, near-separable task** (Rocky): isolates
  "does growth fire for the right reason?" from any capacity confound.
- **log-transform the feature** (Rocky predicted the lift; confirmed +18pts).
- **Convergence measured against the Bayes ceiling**, not 1.0 — honest target
  given the weight-only information limit (chicken/cat overlap).
- **Diagnosis only, no core fix this session** — the trigger fix vs. validate-
  first decision is Rocky's to make next.

## Open questions / next-up

1. **(a) Fix the trigger or (b) validate-first?** — OPEN, Rocky to decide.
   - **(a)** Make growth conditional on capacity actually being the bottleneck:
     only `divide()` if loss has plateaued AND a recent division failed to reduce
     it (promote the `selective_quad_growth` "adaptive escalation" logic into
     `check_growth_trigger` itself). Currently the trigger only sees the
     frustration multiplier + sustained-step count, never the
     did-the-last-divide-help signal.
   - **(b)** First point the same harness at a problem that genuinely needs
     capacity (relational SAME-DIFFERENT or n-bit XOR with `h_init` set too
     small) and confirm growth fires-and-HELPS there — so we know (a) is "make it
     pickier," not "rebuild it."
2. **Carried from s020:** the broader project-wide growth audit (did prior
   "grow under frustration" gains come from new cells or wiring side-effects?).
   This session adds a data point: on easy tasks growth fires on LR noise, not
   need — so audit whether prior growth-win configs ran at an LR that
   manufactured frustration.
3. **Carried:** the 2 pre-existing `test_lifecycle` failures
   (`test_growth_trigger_logic`, `test_dream_locks_eligible_cells`).

## Pointers

- Exercise: `python3 -m experiments.growth_exercise.run_chicken_goat --species chicken cat goat cow --seeds 10 --epochs 100 --lr 0.005 --log`
  (the ceiling run). Drop `--log` to see the feature trap; `--lr 0.02` to see the
  misfire; `--no-grow` to isolate growth's (non-)contribution.
- Growth trigger: `trioron/lifecycle/grow.py` — `check_growth_trigger` (keys on
  `frustration_multiplier >= threshold` + sustained steps; NO "did it help"
  term). `divide()` is the s020-fixed forward-projecting version.
- Frustration: `trioron/learning/frustration.py` (`FrustrationDetector`).
- The local-patch precedent: `experiments/selective_quad_growth.py` lines ~145–172
  ("adaptive escalation" + the "rebuilding self-sustains frustration" comment).
- Manual §4 (axes), §5 (CL machinery), §2.5 (phenotype chosen from problem).
- Memory: [[growth_sink_fix]] (s020), [[selective_quad_growth]],
  [[dysplastic_axis_criterion]] (structure-without-consulting-field-state rule —
  the LR-misfire is a cousin: growing without consulting whether it helped).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `v2.0-scaffold`. Python 3.10.12,
  torch 2.11.0+cu130, WSL2, 12 cores. `python3`, `OMP_NUM_THREADS=8`.
- Exercise runs are fast: n=10 × 100 epochs ≈ 60–75s; full sweep ≈ 4–5 min.
- **Non-trioron note (this session):** Rocky's Claude Code *remote/mobile*
  sessions keep closing. Diagnosed: NO routine on the machine does it (cron /
  systemd timers / sshd[off] / logind / TMOUT / Claude routines[none] / hooks[none]
  all clean; Windows never sleeps). The documented cause is a **>10-min network
  outage while awake** (per code.claude.com/docs/en/remote-control) → session
  process exits. Prime suspect: `C:\Users\Rockz\.wslconfig` uses experimental
  `networkingMode=mirrored` (flaky outbound connectivity). Rocky chose to KEEP
  mirrored mode for now. If drops persist, revert to default NAT + `wsl --shutdown`.
