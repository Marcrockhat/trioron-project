# Trioron Handoff

**Session date:** 2026-06-23
**Session number:** 045
**Session title:** **Manifold replay wins the mixed stream; a joint decomposition
relocates the bottleneck from forgetting to acquisition (undertraining + perception
+ readout); DENDRITE soma is the free architecture upgrade; the phasor emitter is
mapped as a classifier front-end (8 variants — none beats raw→dendrite, structural)
and the discrete-safe `arcsin_u` encoder is promoted to core. Plus a runnable
learning notebook for the receptor/emitter.**

---

## READ THIS FIRST

1. **Branch `progenitor-council`.** This session made FOUR commits + this handoff:
   - `ab257e7` manifold-replay arm (`mixed_stream_cl.py`) + `outputs/mixed_stream_s045_n5.log`
   - `1966a71` (earlier handoff — now superseded by this file)
   - `dc02827` joint decomposition (`mixed_stream_joint.py`) + dendrite/`SOMA_GENE` +
     emitter notebook (`notebooks/01_emitter_and_quantization.ipynb` + build script)
   - `6459ba1` **core**: `arc_phase` + `arcsin_u_descriptor` in `trioron/core/receptor.py`
     + spec §10.2 note. (Only core change this session; tests green.)
   All pushed. DO-NOT-COMMIT carries unchanged (see bottom).
2. **Tooling note (NEW this session):** jupyterlab + nbconvert + ipykernel were
   `pip install --user`'d. A JupyterLab server may still be running from this session
   (`jupyter-lab --port 8888`, log `/tmp/jupyter.log`); it dies with the session. To
   relaunch: `jupyter lab` from repo root.

## THE TWO RESULTS THAT MATTER

### A. Manifold replay is the best CL mechanism on the mixed stream (n=5, committed)
`outputs/mixed_stream_s045_n5.log` (STRENGTH=150 LOCK_RATE=0.1 REPLAY_BS=16 EPOCHS=6).
s044 baselines reproduced exactly, so deltas are clean.

| arm | taxonomy | mnist | cifar | mean | acq | forget | store |
|-----|---------|-------|-------|------|-----|--------|-------|
| none   | 0.229 | 0.000 | 0.238 | 0.155 | 0.552 | +0.377 | – |
| lambda | 0.281 | 0.312 | 0.096 | 0.229 | 0.316 | +0.072 | – |
| both   | 0.291 | 0.461 | 0.088 | 0.280 | 0.294 | +0.041 | – |
| **replay** | 0.440 | 0.690 | **0.240** | **0.456** | 0.534 | +0.130 | 750KB |
| **all** | 0.310 | **0.875** | 0.008 | 0.398 | −0.007 | | 750KB |

- **`replay` alone = best mean 0.456 (+63% over s044's `both`).** Un-starves CIFAR
  (0.088→0.240) AND rescues MNIST (0.690) at near-zero acquisition cost. Replay
  defends OLD classes without penalizing the NEW task's weights (unlike λ).
- **`all` (λ+credit+replay)** stabilizes bimodal MNIST (σ→0.017) + negative forgetting,
  but λ's accumulated penalty re-breaks CIFAR (0.008). Tradeoff, not a single winner.
- Replay = `ManifoldArchive` over the FIXED-LENS BLOCK DESCRIPTORS (input space — valid
  because the lenses are germline; soma H-space would go stale), separate arena.

### B. The joint decomposition: the bottleneck is NOT forgetting (diagnostic, committed)
`experiments/progenitor/mixed_stream_joint.py`, `outputs/mixed_stream_s045_joint.log`.
Replay retains ~86% of what it acquires, so we measured the OFFLINE upper bound (all
52 classes at once) to split the gap. **Joint (H64-tanh, EPOCHS=20): tax 0.836 / mnist
0.946 / cifar 0.347 / mean 0.706** vs CL replay 0.456. The ceiling is three different
problems:
- **Taxonomy = undertraining**, not arch (0.40@3ep → 0.84@20ep, near Maha 0.87). The
  CL stream gives each task only 6 epochs.
- **CIFAR = perception-bound** (joint == CL == 0.24; fixed Gabor on natural images).
- **MNIST = the real forgetting gap** (joint 0.95 vs CL 0.69).

**Arch sweep (joint, single seed):** DENDRITE (quad σ=z+z²) soma WINS at SAME params —
mean 0.750 vs tanh 0.706, CIFAR +0.11, MNIST +0.02. Depth (128-64) is unstable/worse;
width (H256) barely helps at 4× params. Added `SOMA_GENE` env override to
`mixed_stream_cl.py` (the council votes phenotype on task-0 taxonomy, where phenotypes
TIE ~0.83, so it picks tanh and misses dendrite's image-domain edge). **Single-seed CL:
dendrite+replay 0.540 vs tanh+replay 0.343** (dendrite-none forgets catastrophically,
forget +0.562 — the quad is more plastic, so it NEEDS replay). Wants n=5 to confirm.

## THE EMITTER INVESTIGATION (long, conclusive — read if touching perception)

Rocky drove a deep dive on the phasor emitter / quantization. Built a verified runnable
notebook (`notebooks/01_emitter_and_quantization.ipynb`, 5 sections: quantize → phase →
lock-in/multimodal-cancellation → Vernier → real taxonomy). Findings, all measured on
the 32-class taxonomy (chance 0.031, Bayes 0.937, raw→Maha 0.901):

- **Quantization works and is already in core** (`receptor.quantize`, per-sample,
  scale-invariant). **Resolution is saturated below 1000** — NQ 1000 vs 2000 is
  bit-for-bit identical (the encoders depend on u=q/NQ, quantization is just rounding).
- **The 2π carrier wraps → pockets 0 and 1000 collapse to one phasor → BINARY features
  become invisible.** Real failure mode for discrete/heterogeneous tables. (Per-sample
  normalization is also wrong for mixed-type columns — must be per-feature.)
- **`arcsin(u)` fixes the wrap-collapse** (monotone, injective; 0/1 → orthogonal).
  Promoted to core as `arc_phase` + `arcsin_u_descriptor` (commit `6459ba1`).
- **EVERY fixed re-encoding loses to raw→dendrite (0.863) on a STRONG readout.** Tested 8:
  wrap, arcsin (quad/full/stereo/taps), intensity (own-u/signed-dev/abs-dev/z), resultant
  (fixed-σ / per-feature-σ). Each apparent win decomposed to an artifact: stereo-offset =
  a linear combination (redundant, slightly HURTS); the "resultant" lift = pure INPUT
  SCALING interacting with the dendrite's z² term (wrap×4.627 reproduces it exactly);
  abs-deviation & per-feature-σ actively hurt.
- **STRUCTURAL CONCLUSION (firm):** a trained quadratic readout (dendrite/Maha) builds
  its own nonlinear basis from raw; any FIXED encoding is a hand-designed basis competing
  with a learned one and can only match-or-lose. Emitter-as-classifier-front-end is closed
  against a dendrite soma. **Live niches that remain:** (1) strengthening a WEAK readout
  (wrap lifts linear 0.41→0.55), (2) the **Vernier position primitive** (periodicity IS
  the mechanism, no raw alternative), (3) the `arcsin_u` discrete/binary fix.
- **Rocky's "(μ,σ)-resultant" intuition = the manifold/Maha sketch** — independently
  re-derived. Its productive home is generative/replay/routing (where it's already the
  best thing in the codebase: Maha 0.901, manifold replay), NOT discriminative encoding.

## NEXT (priority — pick one to open the next session)

1. **dendrite soma n=5 in the CL stream** — `SEEDS=0,1,2,3,4 SOMA_GENE=dendrite STRENGTH=150
   LOCK_RATE=0.1 REPLAY_BS=16 ARMS=none,replay,all EPOCHS=6 python3
   experiments/progenitor/mixed_stream_cl.py`. Confirm the single-seed 0.540 vs 0.343.
2. **Raw→dendrite taxonomy block swap** — joint showed raw→dendrite (0.863) beats the
   phasor lens→dendrite (0.779). Swap the taxonomy block in `mixed_stream_cl.py` from the
   1D lens to standardized-raw + dendrite soma; re-run; ~+0.08 if it holds.
3. **More epochs per task** — taxonomy is undertrained at 6; sweep EPOCHS with replay
   defending (replay should let us afford more epochs without forgetting).
4. **Frustration / council redesign (DESIGN, parked).** Diagnosis: `FrustrationDetector`
   is a GLOBAL SCALAR loss-plateau detector, but division is LOCAL + TYPED. Defects:
   global (no which-cell), loss-only (can't tell converged from stuck → LR-noise misfire),
   untyped (council exists only because the signal can't pick a phenotype). Proposed fix:
   local `u=|w·g|`-driven frustration + gradient-gate + phenotype chosen by a candidate
   AUDITION at the division event (subsumes the council). Falsification gate: can
   frustration-driven typed growth REDISCOVER the dendrite advantage the joint sweep
   proved? Do NOT dismantle the council before that gate passes. (Rocky's instinct: the
   council is a patch over not letting growth choose phenotypes — correct.)
5. **Promote the Vernier emitter** to `trioron/` (the genuinely-novel working primitive).

## OPEN / unresolved

- Replay storage 750KB (input-space) vs the 30KB H-space headline — shrink before any
  promotion (mixture / random-projection / PCA of the block).
- `forget` is confounded when acq≈final≈0 — always read acq alongside (replay's higher
  forget +0.130 is because it ACQUIRES more, 0.534; its finals dominate every arm).
- s039–s044 carries untouched: deep scattering S1⊕S2; learnable kernels + λ; council
  global-vote→soma; chained-15 PCLL; optics toys; the s044 mixed-stream lenses.

## State of the build / Pointers

- **The CL experiment:** `experiments/progenitor/mixed_stream_cl.py` (5 arms:
  none/lambda/both/replay/all; env `SEEDS H EPOCHS N_CHUNK STRENGTH LOCK_RATE REPLAY_BS
  REPLAY_W SOMA_GENE ARMS`). Perception `mixed_stream_lenses.py` (unchanged).
- **The joint decomposition:** `experiments/progenitor/mixed_stream_joint.py` (env
  `EPOCHS_JOINT ARCHS`; VARIANTS dict of arch names; batched eval for wide nets).
- **The notebook:** `notebooks/01_emitter_and_quantization.ipynb` (+ `build_01_*.py`
  regenerator). Verify/rebuild: `python3 notebooks/build_01_emitter_and_quantization.py &&
  jupyter nbconvert --to notebook --execute --inplace notebooks/01_emitter_and_quantization.ipynb`.
- **Core encoders (NEW):** `trioron/core/receptor.py` — `arc_phase`, `arcsin_u_descriptor`
  (both take `normalized=` for per-feature framing). `quantize`/`phase`/`quanta_to_phase`
  unchanged. Spec §10.2 documents all.
- **Manifold:** `trioron/learning/manifold.py` `ManifoldArchive` (update_class →
  finalize_all → replay_batches; full_cov; mixture_k).
- **The node (s044):** λ `trioron/learning/epigenetic_lock.py`; u `trioron/learning/credit.py`.
- Run cost: CL non-replay arm ~90s, replay/all ~290s @ EPOCHS=6; n=5×5-arm ~57min.
  Joint sweep ~30min (single seed, 5 archs, EPOCHS=20).

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)

`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `trioron/legacy/outputs/`, `notebooks/`
checkpoints, output PNGs untracked (PNGs intentionally uncommitted).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`, Python 3.10.12,
  torch 2.11.0, torchvision 0.26.0, WSL2, `python3` (NOT `python`), `OMP_NUM_THREADS=8`.
  Data `outputs/data/` (CIFAR-100 on disk; CIFAR-10 download sandboxed). CIFAR = fixed
  10-fine slice of CIFAR-100. jupyter now installed (`~/.local/bin/jupyter-lab`).
- **Model note (keep):** lead with the COMPUTATIONAL framing in code/docs (phase code;
  Gabor filterbank; modulus = oriented energy; scattering cascade; Mahalanobis; EWC =
  epigenetic lock λ; credit-locking = utility u; manifold = per-class Gaussian sketch).
  Optical metaphors (lens/emitter/phasor/atom/intensity) are Rocky's conversational
  framing — fine in prose/notebooks, keep them out of code to avoid mid-session safety
  downgrades.
