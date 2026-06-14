# Trioron Handoff

**Session date:** 2026-06-14
**Session number:** 038
**Session title:** **Gabor front-end PROVEN through the full streaming
organism; RFF-on-energy NULL — both point at the division/readout machinery
as the keystone, not perception.** s037 predicted the converged perception
design (complex quadrature Gabor bank, energy channel) from oracle proxies
only. s038 WIRED it into the real chained-15 runner (`PCLL15_SENSE=gabor`)
and ran the full 15-task streaming organism. Result (seed 0, FRAC=1.0):
**task-aware 0.690 / full 0.304**, beating the s034 raw baseline **0.552 /
0.172** on BOTH metrics (+25% / **+77%**). The hypothesis HOLDS. Then Gemma
proposed holographic wave-vectors Ψ(r)=sin(2π·W·r+b) (Random Fourier
Features); wired as `PCLL15_SENSE=rff` ON the Gabor energy (to keep
invariance) → **NULL: the organism collapses to 1 class** (division never
fires), robust across bandwidth. Mechanism: `try_divide` needs a
per-dimension BIMODAL feature to discover a class; RFF's dense random
projection mixes every input into every dim → no dim is bimodal → no
class discovery. So Gabor OVER-segments (128) and RFF UNDER-segments (1) —
**the two failure modes of the same division machinery.** Rocky is **not
satisfied — accuracy is still too low** (organism full 0.304 ≪ its OWN
oracle ceiling 0.553). That ~0.25 gap is the keystone, and BOTH s038
results confirm it is the **division + readout**, NOT the features.

---

## READ THIS FIRST

1. **The numbers this session are FULL-ORGANISM, not oracle.** Unlike
   s037 (oracle templates, perfect clustering), s038's 0.690/0.304 are
   the real streaming organism end-to-end (`run_pcll_chained15.py`, seed 0,
   one gradient-free pass/task, D22 labels 100%, manifold on, composer on,
   cap=128). This is the honest bench-15 number and it SUPERSEDES s034's
   0.552/0.172 as the current best on this runner. Single seed (matches how
   the s034 baseline was reported — the comparison is apples-to-apples).
2. **Hypothesis (s037 NEXT#1) = SUPPORTED.** Gabor energy beats raw pixels
   through the full council/division machinery, biggest on full-softmax
   (the world's regime). The s037 oracle gain (0.249→0.553) was not an
   artifact of skipping the council loop.
3. **Rocky's standing dissatisfaction (the driver for NEXT): accuracy is
   still too low.** Full 0.304 vs the legacy gradient stack's 0.551 and vs
   the organism's own oracle ceiling 0.553. The gap is council/division +
   readout, NOT perception. See "The new keystone" below.
4. **What ships this session:** TWO arms added to
   `experiments/progenitor/run_pcll_chained15.py` (`PCLL15_SENSE=gabor`,
   `PCLL15_SENSE=rff`) + a reflect-pad correctness fix to the gabor base +
   run logs (`..._s038_gabor_full.log`, `..._s038_rff_smoke.log`) + this
   handoff. No package code touched; gate battery untouched (green from
   s036, not re-run).
   - **Reflect-pad fix:** the gabor base now `F.pad(..., mode="reflect")`
     before conv (was `F.conv2d` default ZERO pad). Zero-pad broke the
     background-swap invariance at the image border; reflect-pad makes it
     bit-exact (5.9e-08). Does NOT change the committed gabor 0.690/0.304
     — chained-15 borders are zero, so reflect-of-zero = zero-pad on the
     actual (non-inverted) data; the fix only restores the INVERTED-test
     invariance property (Rocky's binding criterion).
5. **DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM):**
   `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
   `trioron/viz/export.py`; `.claude/`, `runs/` untracked. NOT staged this
   session.

---

## What was built (the Gabor arm)

`make_sense` in `run_pcll_chained15.py` now dispatches a 4th sense,
`gabor` (alongside raw|conv|lcn). It is a FIXED, gradient-free sensory
transform (inside PCLL's gradient-free claim, like conv/lcn):

- **Complex quadrature Gabor bank** `_gabor_quadrature_bank()`: matched
  even (cos) / odd (sin) pairs, λ∈{4,8} (low/mid scales — high freq is
  noise, s037), 4 orientations, 11×11 kernel (fits σ=0.5·λ_max=4), both
  members zero-DC + L1-normalized. 8 complex filters.
- **Energy channel only:** `√(conv_even² + conv_odd²)`, avg-pool 4 →
  [N,8,7,7] → flatten **392-d**. Fed to the existing per-sample receptor
  quantizer exactly like raw/conv.
- **Why energy, not phase:** energy is a non-negative magnitude the
  contrast quantizer carries faithfully, AND it is **bit-exact under
  background swap** (Rocky's binding criterion) — the zero-DC kernels make
  `conv(1−x) = −conv(x)`, so `√(ce²+co²)` is invariant to a global sign
  flip. Phase has the higher oracle ceiling (0.597 vs energy 0.553) but is
  polarity-sensitive (inverted-test 0.000) AND is an angle the per-sample
  min/max quantizer would double-wrap. **Faithful phase needs direct
  phasor injection (`exp(iθ)`) — a pipeline change, deferred to NEXT.**

## The RFF null (Gemma's holographic wave-vectors) — and why it matters

`Ψ(r) = sin(2π·W·r + b)` = Random Fourier Features. Wired as
`PCLL15_SENSE=rff` ON the Gabor energy (NOT raw pixels — plain RFF on
pixels is shift/polarity fragile like a global FFT; energy keeps the
invariance, verified bit-exact through the projection). W = random
wave-vectors, b = random phase, r̂ = per-sample L2-normalized energy,
RFF_DIM=512, RFF_BANDWIDTH (σ_W) tunable.

**Result: NULL — organism collapses to 1 class, division never fires.**
Robust across bandwidth (smoke 0.5/1/2/4 and full-density 0.25/2.0, all
classes=1, ta≈0.26 full≈0.27 = degenerate single-class).

**Why (the instructive part):** `division.try_divide` (`division.py:81`)
accepts a class split only when some feature dim is BIMODAL across samples
— both children must cohere above `NULL_SPLIT=0.72` (a uniform/noise dim
splits to 2/π=0.64 and is rejected, the anti-noise-tiling guard). Class
discovery REQUIRES per-dimension structure. RFF's dense random projection
mixes every input into every output dim, so each dim is smooth-unimodal
(low bw, no split needed) or uniform-noise (high bw, split rejected) —
never cleanly bimodal. No bandwidth fixes this; it is structural. The
feature space is fine; it is incompatible with frustration-division.

**The takeaway:** Gabor energy has per-dim on/off structure → division
OVER-fires (128 fragments). RFF erases per-dim structure → division
NEVER fires (1 class). Two failure modes, ONE machinery. This is the
strongest evidence yet that the keystone is the division/readout, not
perception. Salvage ideas (unpursued, lower priority than the readout):
block-local W so RFF preserves some per-dim structure; or CONCAT
gabor-energy + RFF so division keeps the structured dims to split on.

---

## The result (seed 0, full 15-task stream)

| metric | raw (s034) | **gabor energy (s038)** | Δ |
|---|---|---|---|
| task-aware | 0.552 | **0.690** | +0.138 (+25%) |
| full-softmax | 0.172 | **0.304** | +0.132 (**+77%**) |
| classes | 128 (tiled) | 128 (tiled) | — |

**Per-task — a striking complementarity (the second finding):**

```
          MNIST(1-5)          Fashion(6-10)        EMNIST(11-15)
gabor:  1.00 .50 .73 .99 .83 | .50 .00 .50 .50 .50 | .93 .87 .93 .87 .69
raw:    .46 .49 .64 .48 .51  | .83 .68 .88 .82 .98 | .50 .00 .50 .00 .50
```

Gabor **dominates stroke-based classes** (digits, letters: oriented-edge
energy reads pen strokes) but **regresses on Fashion** (textures/
silhouettes — exactly inverted from raw). The two front-ends are
complementary; neither is universally right. A raw+energy concat or
per-channel frame is the obvious Fashion-recovery probe.

**Over-segmentation, NOT avoided:** gabor tiled to cap=128 on task 1 (raw
grew gradually 4→128 by task 4). So the win is NOT less fragmentation —
the division machinery over-segments gabor energy even harder. The win is
that the 128 fragments are PURER (better feature space → cleaner
name-majority mapping back to 30 classes). Oracle advantage survives
DESPITE over-segmentation, not because of avoiding it.

---

## The new keystone (what NEXT must close): organism ≪ its own oracle

s037 oracle ceilings on the SAME gabor features: energy **0.553**, phase
0.597, discriminative softmax 0.572. The s038 ORGANISM gets full **0.304**.
That ~0.25 gap is now the keystone, and it is downstream of perception:

- **Tiling-to-cap is the prime suspect.** cap=128 is hit on task 1; the
  division-by-frustration self-arrest (open s036 item) does NOT fire on
  the dense gabor-energy surface. 128 fragments for 30 true classes is a
  4× over-segmentation that the readout must then undo.
- **The readout is lossy.** name-majority (class→label count majority)
  collapses 128 fragments to 30 classes by a hard vote. The s037 probe
  already showed the generative mean-phasor matched filter leaves ~0.08
  vs a trained softmax (0.493→0.572) EVEN on perfect clusters — and the
  organism's fragmented clusters are worse than perfect.

This is the same gap the manifold machinery exists to close
(`manifold_phase_c_anchored_readout`, full-cov / H-routing — manual §5.5).

---

## NEXT (priority order — all aimed at the oracle↔organism gap)

1. **Kill the over-segmentation (highest leverage).** The organism tiles
   to 128 on task 1. Make division self-arrest on dense surfaces (the open
   s036 item) OR raise/derive the cap, and measure whether fewer, purer
   fragments lift full toward the 0.553 oracle. The FRAC knob is the cheap
   probe: s038 smoke at FRAC=0.1 (fewer meetings → less tiling) hit
   0.951/0.667 on 2 tasks — fragmentation is clearly the lever.
2. **Better readout than name-majority.** Wire the manifold full-cov /
   H-space routing readout (manual §5.5, `bench_chained_15_v2.py`) onto the
   gabor organism — the validated full-softmax pump (chained-15 0.55→0.76
   with full-cov). The matched filter leaves ~0.08+ on the table by s037.
3. **Add the PHASE channel via direct phasor injection.** Phase oracle
   0.597 > energy 0.553. Needs a pipeline path to inject `exp(iθ)` instead
   of quantizing a real feature — energy stays the invariant workhorse,
   phase adds accuracy. Re-measure both Rocky-criteria (inverted-test).
4. **Fashion regression.** raw+energy concat or a per-channel frame to
   recover the Fashion tasks (gabor 0.50/0.00/0.50/0.50/0.50 vs raw
   0.83/0.68/0.88/0.82/0.98) without losing the stroke gains.
5. **Multi-seed (n=3) the headline** once 1-3 land — seed 0 only so far.
   ~40 min/seed.

---

## State of the build

- Branch `progenitor-council`. This session's commits add the `gabor` +
  `rff` arms to `run_pcll_chained15.py` (+ reflect-pad fix to the gabor
  base) + `outputs/pcll_chained15_s038_{gabor_full,rff_smoke}.log` + this
  handoff. No package code changed; gate battery untouched.
- `developmental.py` ×2 + `viz/export.py` carries deliberately NOT staged
  (s034 carry, unchanged).
- Run cost: full 15-task gabor organism ~2343s (~39 min) seed 0, CPU,
  OMP_NUM_THREADS=8. Stays at cap=128 from task 1 (~150–170s/task).

---

## Pointers

- **The runner + the new arm:** `experiments/progenitor/run_pcll_chained15.py`
  — `make_sense` (raw|gabor|conv|lcn), `_gabor_quadrature_bank()`,
  `GABOR_LAMBDAS/THETAS/K/POOL`. Env: `PCLL15_SENSE=gabor`,
  `PCLL15_FRAC` (subsample → fewer meetings → less tiling, the NEXT#1
  probe), `PCLL15_CAP` (0=uncapped), `PCLL15_TASKS` (smoke subset),
  `PCLL15_SEEDS`.
- **The s037 oracle diagnostic (the ceilings NEXT chases):**
  `experiments/progenitor/diag_s037_control.py` — energy 0.553, phase
  0.597, softmax 0.572, raw 0.249.
- **Encoding internals:** `trioron/core/receptor.py` (per-sample quantize,
  N_QUANTA=1000); `trioron/pcll/mixed.py` (`_evidence`/`templates`/
  `pockets_of`); `trioron/pcll/controller.py` (frame registry D10);
  `trioron/pcll/labels.py` (D22 label taps).
- **Readout pump for NEXT#2:** `experiments/bench_chained_15_v2.py`
  (`--full-cov`, H-routing); `trioron/learning/manifold.py`.
- **Design:** `docs/design/progenitor_council.md`; spec §10.2 (receptor),
  §10.3 (mask), §10.5 (canonical frame), §10.6–10.10 (mixed stream /
  composer / manifold / census).

---

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`),
  `OMP_NUM_THREADS=8`. Data at `outputs/data/`.
- bench-15 = 30 true global classes, 15 tasks (MNIST/Fashion/EMNIST-letters
  pairs), single gradient-free pass per task. FRAC=1.0 = full stream.
- **Model note:** Fable 5 with biological-metaphor safety measures; lead
  with the computational/ML framing (frequency, convolution, matched
  filter, energy = magnitude of a quadrature pair) to avoid mid-session
  downgrades.
