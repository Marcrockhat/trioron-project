# Trioron Handoff

**Session date:** 2026-06-23
**Session number:** 046
**Session title:** **Design session — diagnosed the `arcsin_u_descriptor`
degeneracy (angle and magnitude are the same scalar → 1-DOF) and specced the fix:
a SEPARABLE magnitude carrying an independent, unsupervised, streaming intensity =
per-specimen surprise `|x−μ|/σ` against a PHASE-CONDITIONED baseline, with the
recurrence period `P` recovered from the stream and the EWMA forgetting factor
derived (not guessed) from a local-level Kalman SNR. Written up as spec §10.2.1.
No code yet — implementation + smoke test is the next session's job.**

---

## READ THIS FIRST

1. **Branch `progenitor-council`.** This session is **design only**. One file
   changed and committed: `paper/v3/spec.md` — new subsection **§10.2.1
   "Separable magnitude — independent intensity from a phase-conditioned surprise
   (DESIGN, s046, unbuilt)"**. No `trioron/` code touched. DO-NOT-COMMIT carries
   unchanged (see bottom).
2. Everything below is the reasoning chain behind §10.2.1, in case the spec note
   needs unpacking. The spec note is the source of truth; this is the narrative.

## WHAT WE DECIDED (the design, in order)

The conversation walked a chain of constraints to a single concrete design. Each
step pinned a parameter so **nothing in the final design is a free knob to guess**:

1. **The defect.** `arcsin_u_descriptor` ships `[u·cosθ, u·sinθ]` with
   `θ=arcsin(u)` — both magnitude and angle are functions of the *same* `u`. So
   the 2D output is a 1-DOF curve in a plane: no more information than `u` alone.
   This IS the s045 redundancy finding ("stereo win = redundant linear combo").
   **Fix: angle and magnitude must be independent** → `[r·cosθ, r·sinθ]`,
   `θ=arcsin(u_angle)`, `r` = a *separate* intensity. Then phasors can be summed
   to a resultant 2-vector → read 2+ logits (the `(μ,σ)`-resultant made literal).
2. **What is `r`?** Regime is **unsupervised** (Fisher/between-class is OUT) and
   **streaming/growing** (stats must be online, no stored data). Survivor =
   **per-specimen surprise `r = |x−μ|/σ`** = the diagonal per-coordinate
   Mahalanobis term. Diagonal only (O(d), stable); NOT full covariance. This is
   the unsupervised/incremental form of the codebase's best mechanism (Maha 0.901).
3. **The baseline `(μ,σ)` must be phase-conditioned** because Rocky confirmed the
   feed has **periodic recurrence** (types return every ~`P` samples). A plain
   EWMA blurs across types and inflates surprise; instead use the **seasonal /
   comb kernel** — `P` phase buckets per feature, only the current bucket updates,
   `r = |x−μ_φ|/√(v_φ+ε)`. Surprise is measured against "same phase last cycle."
4. **The forgetting factor is DERIVED, not guessed.** Local-level model
   (random-walk mean + observation noise) → steady-state Kalman gain gives the
   EWMA `α*` in closed form from the SNR `λ=q/r_obs`: `α*=s/(s+1)`,
   `s=(λ+√(λ²+4λ))/2`. `q` (drift var) and `r_obs` (noise var) are estimable
   online per feature. Exponential decay is the unique *memoryless* kernel and
   optimal here; the comb is its phase-indexed extension, justified ONLY because
   the feed recurs (we explicitly rejected a "wave/oscillating" kernel for the
   non-periodic case — it would add a frequency knob to guess).
5. **`P` is recovered from the stream** (Rocky: not known a priori). Two coupled
   loops: (i) a GLOBAL period detector on a scalar novelty trace
   `s_t=‖x_t−μ_global‖` via a small **resonator bank** (complex poles, the §10.2
   complex-pole machinery reused) or decayed-autocorrelation peak; (ii) the
   per-feature phase buckets consuming the locked `P`.

## THE FOUR IMPLEMENTATION TRAPS (must be handled — see §10.2.1)

1. **Phase-lock, not `t mod P`** — anchor bucket 0 to the novelty peak each cycle
   (a PLL), or buckets smear to the global mean when `P` is slightly off.
2. **Fundamental, not harmonic** — resonators lock onto `2P`/`P/2`; pick smallest
   lag whose multiples all carry energy (comb score `Σ_m R(mP)`).
3. **Non-integer `P`** — round + let `β` absorb slack (start here), or interpolate.
4. **Warm-up + drift fallback** — until the comb peak clears the noise floor, fall
   back to APERIODIC EWMA `(μ,σ)`; switch on phase-buckets only once `P` locks;
   hysteresis so locked `P` doesn't jitter; decayed resonators track drifting `P`.

## NEXT (priority — pick one to open the next session)

1. **Implement §10.2.1 in `trioron/core/receptor.py`** + a **smoke test on a
   synthetic periodic stream**. Falsification gate (from §10.2.1): the detector
   must lock the correct `P` and reject `2P`/`P/2`, AND the separable-magnitude
   descriptor must beat the same-source `[u·cosθ,u·sinθ]` on a **weak/linear**
   readout (the live niche). It is NOT expected to beat raw→dendrite on a strong
   readout — that is not the bar (s045 closed that door).
2. **BLOCKER for #1: get `P_min..P_max` from Rocky** — the candidate period range
   (in samples between type-recurrences). Sizes the detector / lag window
   (O(P_max) state) and rejects out-of-range harmonics. NOT yet supplied — ask
   first. Also confirm whether `P` itself drifts (→ adaptive `P` + hysteresis).
3. **(Carried from s045) dendrite soma n=5 in the CL stream** —
   `SEEDS=0,1,2,3,4 SOMA_GENE=dendrite STRENGTH=150 LOCK_RATE=0.1 REPLAY_BS=16
   ARMS=none,replay,all EPOCHS=6 python3 experiments/progenitor/mixed_stream_cl.py`.
   Confirm single-seed 0.540 vs 0.343.
4. **(Carried from s045) raw→dendrite taxonomy block swap** in `mixed_stream_cl.py`
   (joint showed raw→dendrite 0.863 > phasor-lens→dendrite 0.779; ~+0.08 if holds).
5. **(Carried from s045) frustration / council redesign (DESIGN, parked)** — local
   `u=|w·g|`-driven frustration + phenotype-by-audition; gate = can typed growth
   REDISCOVER the dendrite advantage? Do NOT dismantle the council before the gate.

## OPEN / unresolved

- §10.2.1 is **unbuilt** — design only. No measurement yet; the lift on a weak
  readout is hypothesized, not shown.
- `P_min..P_max` not supplied (blocks the smoke test sizing).
- s045 carries: replay storage 750KB input-space vs 30KB H-space headline (shrink
  before promotion); `forget` confounded when acq≈final≈0 (read acq alongside);
  s039–s044 scattering / learnable-kernels-λ / council-vote / chained-15 PCLL /
  optics toys / s044 mixed-stream lenses all untouched.

## State of the build / Pointers

- **Spec note (this session):** `paper/v3/spec.md` §10.2.1 (right before §10.3
  "Lock-in state"). Extends §10.2's `arc_phase`/`arcsin_u_descriptor` (s045).
- **Core encoders (s045):** `trioron/core/receptor.py` — `arc_phase`,
  `arcsin_u_descriptor` (both take `normalized=`). `quantize`/`phase`/
  `quanta_to_phase` unchanged. The descriptor at lines ~86–102 is what §10.2.1
  re-specs (separable `r` replacing the `u` magnitude).
- **Manifold (the resultant's existing home):** `trioron/learning/manifold.py`
  `ManifoldArchive` (per-class Gaussian; the SUPERVISED analog of the unsupervised
  streaming `(μ,σ)` §10.2.1 proposes).
- **The CL / joint experiments (s045, unchanged):**
  `experiments/progenitor/mixed_stream_cl.py`, `mixed_stream_joint.py`,
  `mixed_stream_lenses.py`; notebook `notebooks/01_emitter_and_quantization.ipynb`.
- s045's two headline results still stand: **manifold replay = best CL mechanism**
  (mean 0.456, +63% over s044 `both`) and **the bottleneck is acquisition not
  forgetting** (joint upper bound 0.706; taxonomy undertrained, CIFAR
  perception-bound, MNIST the real forgetting gap). DENDRITE soma wins at same
  params (joint 0.750 vs tanh 0.706).

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)

`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `trioron/legacy/outputs/`,
`notebooks/` checkpoints, output PNGs untracked (PNGs intentionally uncommitted).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`, Python
  3.10.12, torch 2.11.0, torchvision 0.26.0, WSL2, `python3` (NOT `python`),
  `OMP_NUM_THREADS=8`. Data `outputs/data/` (CIFAR-100 on disk).
- **Model note (keep):** lead with the COMPUTATIONAL framing in code/docs
  (per-feature Gaussian / Mahalanobis surprise; Kalman/EWMA forgetting; seasonal
  comb = phase-conditioned baseline; resonator bank = streaming periodogram).
  Optical metaphors (lens/emitter/phasor/atom/intensity) are Rocky's conversational
  framing — fine in prose, keep them out of code to avoid mid-session safety
  downgrades.
