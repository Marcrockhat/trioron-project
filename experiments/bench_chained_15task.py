"""Chained-15 headline: MNIST → FashionMNIST → EMNIST-letters, hard param cap.

Tests the apoptosis claim: "trioron survives streams the baselines weren't
designed for, because dreaming reclaims substrate when capacity binds."

Curriculum:
    15 binary tasks total (5 per dataset). Global classes 0..29.
    Tasks 0-4   use MNIST classes 0..9               → global 0..9
    Tasks 5-9   use FashionMNIST classes 0..9        → global 10..19
    Tasks 10-14 use EMNIST-letters A..J (local 0..9) → global 20..29

KMNIST was planned for the third block but its torchvision mirror is dead;
EMNIST-letters fills the same role (different glyph distribution).

Architecture (grown_* arms):
    784 → L0_WIDTH=128 (FROZEN random projection) → H_init=32 (grown,
    GROWTH_TARGET) → head (grows 2 → 30)

    L0 is a frozen feature extractor: random Kaiming-init weights, no
    backward, excluded from the cap budget. The cap counts trainable
    substrate only — i.e. L1 + head — so the budget reflects what
    dreaming-driven apoptosis can actually reclaim.

Architecture (fixed_ewc baseline arm):
    784 → H_FIXED=64 (trainable) → 64 (trainable) → head — no growth,
    no dream, EWC-only. Note this baseline is intentionally NOT
    matched-params with the grown arms; it's the standard
    "fixed-MLP-with-EWC" comparator at a wider hidden than the grown
    arms start with, and is used to show that growth+dream beats a
    same-or-bigger frozen allocation under the chained stream.

Trigger choice (per session decision: "Option B"):
    Trigger-driven growth is OFF. Each task tries to deterministically
    grow N_GROW_PER_TASK hidden nodes into layer 1 BEFORE training. The
    growth happens iff projected params after grow are <= cap; otherwise
    it's denied. This isolates the apoptosis-reclaim claim from the
    trigger-calibration question.

Arms:
    1. fixed_ewc           — H=64, no growth, no dream, EWC. The
                             matched-fixed baseline.
    2. grown_capped_no_dream  — start H=32, deterministic grow N=4 per
                             task, hard cap. Once cap binds, can't grow
                             more. No dreaming. Control for "what does
                             pure growth-under-cap look like".
    3. grown_capped_dream  — same growth + cap as (2), with dreaming
                             (replay → starve+apoptosis → purge) called
                             on every task end + IMMEDIATELY when growth
                             is denied. The protagonist.
    4. grown_uncapped_dream — same growth + dream as (3), no cap.
                             Capacity-control: shows what's possible
                             when substrate is unlimited.

Headline metric:
    Final accuracy + accuracy on tasks 10-14 (the late-stream KMNIST
    block where the cap should be binding). Side panel: per-task
    purge_count + apoptosis_event_count for arm (3).

Run:
    python3 -m experiments.bench_chained_15task               # full budget
    python3 -m experiments.bench_chained_15task --smoke       # 1 epoch/task
    python3 -m experiments.bench_chained_15task --arms grown_capped_dream,fixed_ewc
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import random
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import TrioronNetwork
from trioron.packnet import PackNetController
from trioron.hat import HATController
from trioron.classification import (
    accuracy,
    extend_output_head,
    masked_cross_entropy,
    summarize,
)
from trioron.dreaming import (
    PurgeEvent,
    MergeEvent,
    apoptosis_decay,
    archive_block,
    compress,
    purge,
)

from experiments.datasets import (
    DEFAULT_DATA_ROOT,
    BrainstemBuffer,
    DatasetBundle,
    DifferentialReplayBuffer,
    EngramBuffer,
    HippocampalBuffer,
    ManifoldBuffer,
    MemoryBuffer,
    TaskDataView,
    build_task_views,
    chained_15_specs,
)


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

INPUT_DIM = 784
L0_WIDTH = 128                # frozen feature-extractor (random projection)
H_INIT_GROWN = 32
H_FIXED = 64
INIT_CLASSES = 2
GROWTH_TARGET_LAYER_IDX = 1   # second hidden (L1) — NOT the head, NOT L0
N_GROW_PER_TASK = 4           # deterministic per-task hidden growth

N_EPOCHS_PER_TASK = 8                 # full bench: ~180 batches × 8 = ~1440 steps
N_EPOCHS_PER_TASK_SMOKE = 4           # smoke: 4 epochs so Fix B (settle→grow→
                                       # post-grow) has room to operate
BATCH = 64
LR = 1e-3
SEED = 0

# Fix B (growth gating). Per Gemma's framing: don't let structural
# plasticity fire on epoch 1, the network has to settle on stable input
# shape first before any "we need more capacity" signal is meaningful.
# After K_SETTLE_EPOCHS of straight training, the deterministic growth
# block fires (with dream-rescue if cap binds). Then the remaining
# epochs train the post-growth network. K_SETTLE = ⌊N_EPOCHS / 2⌋ keeps
# settle and post-grow phases roughly balanced; for the smoke at 4
# epochs that's 2 / 2; for the full bench at 8 epochs that's 4 / 4.
K_SETTLE_EPOCHS = 2

LAMBDA_FLOOR = 1e-3           # epigenetic baseline only — close to zero, not zero.
                              # Was 0.1 (uniform); the chained-15 Fisher probe
                              # (2026-05-03) showed 100% of params at that floor
                              # → no Fisher selectivity. Combined with the
                              # update_lambda mean→sum patch, real Fisher row-
                              # sums (head ~0.01-0.5 active, L1 ~0.005-0.05
                              # active) now sit 5-500× above this floor while
                              # unused params keep a faint baseline pull.
EWC_INTERTASK = 30.0          # tuned for fan_in=128 + CE; bench_50task used 1000
EWC_DREAM_STRENGTH = 30.0     # match intertask strength inside dreaming

# Cap calibration. The cap counts TRAINABLE substrate only — L0 (the
# 784→128 random-projection feature extractor) is frozen and excluded
# from the budget. Trainable mass lives in L1 (128→H_init=32, growable)
# and the head L2 (32→2..30, growable in the head dimension).
#
# Init trainable params: (128+1)*32 + (32+1)*2 = 4128 + 66 = 4194.
# Per L1 grow cost: (L1.fan_in + 1) + head_size = 129 + 2..30 ≈ 131-159.
# Uncapped trajectory at K_grow=60, head=30: ≈ 14,600 trainable params.
#
# Setting cap at 8,000 trainable params (= 32,000 bytes at FP32):
#   - K_grow allowed before binding ≈ 24-25 (≈ task 7 of 15)
#   - tasks 7-15 see denials → dream-rescue must free room to fit them
#   - apoptosis on L1 reclaims ~131-159 params per purge (~2% of cap),
#     so a handful of purges materially advances the K_grow ceiling.
M_MAX_BYTES_CAPPED = 8_000 * 4      # 8k trainable params → 32 KB
M_MAX_BYTES_UNCAPPED = 2 * 1024 ** 3

# Dreaming-block configuration — substrate-preserving compression with
# apoptosis spike, plus aggressive purge so room actually frees.
#
# DREAM_REPLAY_FRACTION = fraction of past tasks sampled during the
# post-task replay_only mode (consolidation only; doesn't drive purge).
# Kept at 0.25 to bound per-task wall-clock — replay_only fires after
# every task and full coverage gets expensive late in the curriculum.
#
# DREAM_RECLAIM_REPLAY_FRACTION = fraction of past tasks sampled during
# dream-rescue (the cap-binding replay that drives purge victim
# selection). Set to 1.0 (full coverage) on 2026-05-03 after the
# n=12 saliency bench showed seed 6's catastrophic Fashion regression
# was caused by the 0.25 fraction sampling only 1-2 of 6 past tasks
# during a rescue → the saliency u was blind to non-replayed tasks
# → purge picked units that were critical for non-replayed tasks.
# Full coverage ensures all past tasks contribute to u before purge.
DREAM_REPLAY_FRACTION = 0.25
DREAM_RECLAIM_REPLAY_FRACTION = 1.0
DREAM_REPLAY_STEPS = 50       # smaller than bench_50task's 200 because the
                               # task data here is bigger and replay is
                               # called more often (post-task + on-deny)
DREAM_REPLAY_BATCH = BATCH
DREAM_AC_THRESHOLD = 0.85
DREAM_PROBE_BATCH_SIZE = 256
DREAM_COMPRESSION_ACTION = "starve"
DREAM_MAX_DOWNSCALES_PER_LAYER = 1
DREAM_STARVATION_ALPHA = 0.7
DREAM_STARVATION_FLOOR = 1e-3
DREAM_APOPTOSIS_ON = True
DREAM_APOPTOSIS_SPIKE_INIT = 0.8
DREAM_APOPTOSIS_DECAY_RATE = 0.7

# Purge needs a usable utility threshold. The contrastive benches kept
# this at 1e-3 because contributions there were small. For CE-on-MNIST
# the per-batch contributions are larger; raising the threshold means
# starved units (whose effective contribution decays toward 0) will
# actually be reclaimed.
DREAM_U_THRESHOLD = 0.01
DREAM_PURGE_SKIP_OUTPUT = True

# Per-event throttle on apoptosis. Without this, purge greedily reaps
# every unit below u_threshold in a single dream block — first smoke
# saw 23 of 56 L1 nodes (~41%) reclaimed in one event, which is
# closer to a stroke than a sleep cycle. Biological synaptic
# homeostasis runs at ~5-15% per cycle; apoptosis itself is sub-
# percent. Capping at N_GROW_PER_TASK gives ~7-10% per event at
# typical L1 widths and matches the deficit math (cap allows ~26
# grows; curriculum wants 60; difference of 34 spread over ~9
# denial-cycles ⇒ ~4 reclaims/event needed to fit everything).
# Maps onto the sRNA-cap analogy: per-cycle resource-limited pool,
# selects how many not which.
DREAM_MAX_PURGES_PER_EVENT = N_GROW_PER_TASK

# Infancy / L0 warmup. Lickliter (2002) on bobwhite quail: augmented
# prenatal sensory exposure DISRUPTS the perceptual-development cascade
# — biological infancy is brief, intense, and isolated from later
# learning. Mirror that: a small held-out warmup set (per dataset) is
# used ONCE before the continual stream begins, just to develop L0's
# feature extractor; L1 + head are reset to fresh random init after
# warmup so the curriculum starts with a developed perceptual layer
# but a naive learnable substrate.
WARMUP_ENABLED = False                # Off for Fix A baseline; flip True to
                                       # re-enable L0 warmup. Holdout is still
                                       # built so the option remains live.
N_INFANCY_PER_DATASET = 500          # 500 × 3 datasets = 1500 samples total
N_WARMUP_STEPS = 100                  # ~4× exposure/sample at BATCH=64 — brief
WARMUP_LR = 1e-3
WARMUP_TEMP_HIDDEN = 64               # temp L1 width during warmup; discarded
WARMUP_HEAD_WIDTH = 30                # all global classes (full 30-class CE)

# Curriculum revisit. Pass 1 = "developmental" (growth + dreaming +
# EWC consolidation). Pass 2 = "consolidation" (no new neurogenesis;
# just retraining + dream-rescue is moot since growth is off).
# EWC anchors carry forward (no reset between passes) — biologically
# the consolidated trace doesn't unwind on revisit.
# TEMPORARILY 1 to isolate the warmup effect — switch to 2 after we
# confirm warmup is at-least-neutral vs the no-warmup baseline.
N_CURRICULUM_PASSES = 1

# Path 2 — rehearsal during training. After each task finishes, a small
# random subset of its examples gets stored in MemoryBuffer. During
# every training step on subsequent tasks, a rehearsal batch is sampled
# from the union of stored tasks and contributes a CE loss masked to
# ALL classes seen so far (not just the current binary pair). This is
# the standard CL rehearsal recipe (iCaRL / ER family); the difference
# from dream-replay is that rehearsal happens at every training step,
# not just between tasks.
#
# The full-head active-class masking is what fights cross-class head
# drift directly — task-aware metric measured the head drift was
# benign within a known task (0.93+) but full-softmax (0.13-0.21)
# revealed argmax drift across the 30-class output. Rehearsal with
# all-classes-seen-so-far mask gives explicit gradient on cross-class
# discrimination.
REHEARSAL_ENABLED = False           # raw-input rehearsal — superseded
                                     # by HippocampalBuffer (L0-output
                                     # rehearsal, biologically modeled
                                     # on hippocampal place-cell coding).
                                     # Sanity-check at n=3 verified the
                                     # historical magnitudes reproduce.
REHEARSAL_SAMPLES_PER_TASK = 100
REHEARSAL_BATCH = 64
REHEARSAL_LOSS_WEIGHT = 1.0

# Parasitic-Dream (Learning without Forgetting, Li & Hoiem 2017).
# At task k+1, BEFORE training, we forward each batch through the
# network using W_anchor (the just-consolidated weights) to get Z_old —
# the "old network's view of new data." We then add a temperature-
# softened KL distillation penalty pulling the live network's logits
# toward Z_old on the OLD-class columns only. This preserves the
# consolidated decision boundaries without storing any old-task data.
# Storage cost: zero new (W_anchor already exists per layer).
# Biology: maps onto sleep-dependent cortical consolidation — the cortex
# preserves its pre-update response while integrating new experience.
LWF_ENABLED = False
LWF_TEMPERATURE = 2.0               # standard distillation temperature
LWF_LOSS_WEIGHT = 0.7               # modestly under 1.0 — proxy signal is
                                     # noisier than direct CE on the new task
                                     # because the anchor network includes
                                     # newly-grown L1 units at random init
                                     # (their W_anchor is set at grow time,
                                     # not at consolidate time). Empirical
                                     # tuning may push this back to ~1.0.

# Brainstem-Spark (latent Gaussian rehearsal). After consolidating a
# task, compute per-class (μ, σ) at the L1 output (the discriminative
# bottleneck). During training of subsequent tasks, sample synthetic
# latents from these per-class Gaussians and feed directly into the
# head, bypassing L0+L1 entirely. The head learns to maintain class
# boundaries even when fed synthetic activations sampled in-distribution.
# Storage cost: ~30 KB (30 classes × 2 stats × ~60 dims × 4 bytes).
# Biology: maps onto REM dreaming — brainstem-driven internal pattern
# generation propagating through cortex without external sensory drive.
# L1-bottleneck choice: L0 separability is too marginal (probe ratio
# 1.063 — see memory/l0_prototype_separation.md). L1 is the
# discriminative layer; per-class Gaussians at L1 output separate well.
# L1 grows during the curriculum; stored stats are zero-padded when L1
# widens (new units initialized fresh activate ~zero on old-class data).
BRAINSTEM_ENABLED = False
BRAINSTEM_BATCH = 64
BRAINSTEM_LOSS_WEIGHT = 0.2         # secondary regularizer; the per-class L1
                                     # Gaussians become STALE as L1 drifts
                                     # under EWC + LwF, so the synthetic
                                     # latents only approximate the current
                                     # network's L1 distribution. Keep weight
                                     # low so this acts as a head regularizer
                                     # rather than a primary signal.

# Engram Replay (triparametric A+B redesign). After each task's
# consolidation, run gradient ascent on the input through the *anchored*
# network to find one prototype x_c per class — the consolidated
# network's idealized class-c input ("engram"). During subsequent
# tasks, sample engrams and feed them through both live and anchored
# networks; KL-distill live → anchor on old-class output columns. This
# replaces both A (LwF on new-task data, OOD-weak) and B (Brainstem
# Gaussian on post-ReLU L1, stale-prone): the engram is a stable input
# upstream of L1; the anchored network's response to engrams is sharp
# by construction (we found the engram by maximizing it); the
# distillation target is therefore strong even on heterogeneous chained
# curricula. Triparametric: forward_with_anchors uses W_anchor +
# b_anchor + routing_scale_anchor, so all three legs of the consolidated
# trioron node contribute to the LwF target.
ENGRAM_ENABLED = False               # DROPPED per probe diagnostic
                                      # 2026-05-04. Gradient-ascent
                                      # engrams collapse to one
                                      # adversarial cluster (7.5x
                                      # narrower than real samples,
                                      # 9x further from the data
                                      # manifold) and don't scale with
                                      # K. Single real sample beats 100
                                      # engrams. See
                                      # experiments/probe_engram_diversity.py
                                      # for the full diagnostic.
                                      # Replaced by HippocampalBuffer
                                      # (real L0 codes, not generated).
                                      # Code paths preserved but dark.

# Hippocampal Replay (replaces engrams; replaces raw rehearsal too).
# Stores K real-sample L0 outputs per class at consolidation time;
# replays them via forward_from_layer(start=1), bypassing L0. Storage
# scales with L0_width (=128), not input_dim — buffer for ImageNet-
# scale inputs is the same size as for MNIST.
#
# Biological mapping: hippocampal place/concept cells encode sparse
# indices into cortical patterns (Quian Quiroga concept cells, O'Keefe
# place cells); CA3 recurrent attractor pattern-completes from those
# indices; sharp-wave ripples replay the compressed codes during sleep,
# driving cortical consolidation. trioron's frozen L0 plays the role
# of the cortical sensory hierarchy; this buffer plays CA3 + place
# cells.
#
# Real-sample probe (chained-5 MNIST, 2026-05-04):
#   real-K=1   full=0.559    real-K=10   full=0.656    real-K=100  0.736
# Single real-sample-per-class already beats every in-weights mechanism
# tested (LwF, brainstem, engram). Diversity probe confirmed real
# samples retain 7.5x wider pairwise spread than generated prototypes.
HIPPOCAMPAL_ENABLED = False
HIPPOCAMPAL_K_PER_CLASS = 50         # matched-K to raw rehearsal's
                                      # 100 samples / 2-class task.
                                      # Storage: 30 cls × 50 × 128 × 4
                                      # = 768 KB (vs raw rehearsal
                                      # 30 × 50 × 784 × 4 = 4.7 MB —
                                      # 6x compression at matched K).
                                      # K=1 ablation gave 0.19-0.36
                                      # full (chained-15 n=3); K=50
                                      # tests whether L0-output replay
                                      # matches raw-input replay at
                                      # matched K.
HIPPOCAMPAL_BATCH = 64               # per-step replay batch size
HIPPOCAMPAL_LOSS_WEIGHT = 1.0        # parity with new-task CE

# DeepInversion-style synthesis (Phase A, 2026-05-05). Replaces
# _store_hippocampal_codes' real-sample encoding with logit-driven
# inversion against the just-consolidated network. Storage shape is
# unchanged — (K, L0_WIDTH) per class — so the replay path is byte-
# identical. The benefit is "no real samples retained at consolidation"
# (the synthesizer needs only the network itself + class label). Phase B
# will skip the buffer entirely (synthesize-on-demand at replay).
HIPPOCAMPAL_SYNTHETIC = False        # True = invert logits to make codes;
                                      # False = legacy real-sample L0(x).
HIPPOCAMPAL_SYNTH_STEPS = 50         # inner Adam steps per class
HIPPOCAMPAL_SYNTH_LR = 0.05          # Adam lr on raw_z
HIPPOCAMPAL_SYNTH_L2 = 1e-3          # L2 reg on z (post-softplus)
HIPPOCAMPAL_SYNTH_INIT_SIGMA = 0.1   # init scale: raw_z ~ N(0, sigma^2)
HIPPOCAMPAL_SYNTH_DIV_WEIGHT = 1.0   # within-K diversity reg: penalize
                                      # squared cosine similarity between
                                      # the K codes of the same class to
                                      # break attractor-spike collapse.

# Manifold Replay — trioron-native pseudo-rehearsal (2026-05-05).
# Stores per-class diagonal Gaussian (μ_c, σ_c) at L0 output instead of
# K real-sample codes. Codes are SAMPLED on demand at each replay step
# from N(μ_c, σ_c²) — synthesize-on-demand, not synthesize-and-store.
# Storage = 30 classes × (mean+var) × L0_WIDTH × 4 = 30 KB total at
# chained-15 (vs hippo K=50 = 768 KB; vs raw rehearsal = 4.7 MB).
# Replay path identical to hippo: forward_from_layer(z, start=1) +
# masked_cross_entropy. Frozen-L0 only.
MANIFOLD_REPLAY_ENABLED = False
MANIFOLD_REPLAY_BATCH = 64           # per-step sample count
MANIFOLD_REPLAY_LOSS_WEIGHT = 1.0    # parity with new-task CE
MANIFOLD_NOISE_SCALE = 1.0           # multiplier on σ when sampling;
                                      # 0 = use mean only, 1 = full
                                      # diagonal Gaussian draw.
MANIFOLD_MAX_SAMPLES_PER_CLASS = 1000  # cap when computing stats; matches
                                        # brainstem's pattern.

# Dream-archive Phase 1 (per memory/dream_archive_stage.md). Marks
# stable rows as developmentally closed: snaps W/b to anchor, drops
# Fisher contribution, masks grads. Phase 2 quantization is gated by
# QUANTIZE_ARCHIVED_AT_END below.
#
# Triggers (forwarded to trioron.dreaming.archive_block):
#   streak_threshold     consecutive consolidations with high λ before
#                        the row is eligible.
#   lam_top_percentile   per-layer percentile boundary on λ (0.75 = top
#                        quartile counts as "high λ this consolidation").
#   grad_mag_floor       max sqrt-Fisher row-sum to call the row settled.
#                        Default 0.1 was conservative for fan_in=128;
#                        chained-15 typical row magnitudes need
#                        per-layer calibration on the first dry run.
#   pulse_max            apoptosis-pulse must be quiet (no recent death
#                        next door) before locking.
#   max_archives_per_layer  cap on archives per call.
ARCHIVE_ENABLED = False
ARCHIVE_STREAK_THRESHOLD = 3
ARCHIVE_LAM_TOP_PERCENTILE = 0.75
ARCHIVE_GRAD_MAG_FLOOR = 0.1
ARCHIVE_PULSE_MAX = 0.1
ARCHIVE_MAX_PER_LAYER = 8
ARCHIVE_SKIP_OUTPUT_LAYER = True

# Phase 2 — end-of-curriculum quantization simulation. After all passes
# finish, snap archived rows from FP32 to ternary or int8 (per-row
# symmetric scale), then re-evaluate to measure accuracy degradation
# and report the deployment-KB breakdown.
#
# QUANTIZE_MODE in {"ternary", "int8"}.
#   ternary  ~16× compression on archived weights (sign trit + per-row
#            scale), accuracy hit possible.
#   int8     ~4× compression on archived weights, lower accuracy hit.
QUANTIZE_ARCHIVED_AT_END = False
QUANTIZE_MODE = "ternary"

# Differential Replay (per Rocky 2026-05-04, post-hippocampal).
# The trioron network IS its memory; forward(x=0) reveals the
# accumulated memory state. Storing the DIFFERENTIAL between blank and
# task at each layer captures the task-specific signal independent of
# how the underlying memory drifts. The rehearsal "tuple" is
# (δL0_c, δL1_c, δlogit_c); injection happens in the middle (L1
# supervision via differential matching), with the full multi-level
# loss enforcing the network to maintain a stable per-class signature
# even as its biases evolve.
#
# Bias-drift invariance: hippocampal stores absolute L0(x_c). Live
# network's L1 may drift; stored L0(x_c) replayed through drifted L1
# produces a different L1 output than at consolidation. Differential
# replay says: regardless of how L1 drifts, the GAP between L1's blank
# response and L1's class-c response should equal the stored δL1_c.
# That's what's preserved across tasks.
#
# Storage scales with sum-of-layer-widths, not input_dim. Per class
# at chained-15: 128 (L0) + ~48 (L1) + 30 (head) = ~206 floats × 4 B
# = ~825 B. 30 classes ≈ 25 KB total.
DIFFERENTIAL_ENABLED = False         # ablation: differential alone
                                      # (HIPPO/ENGRAM/LWF/BRAINSTEM all
                                      # off). Tests bias-drift-invariant
                                      # multi-level rehearsal.
DIFFERENTIAL_BATCH = 64
DIFFERENTIAL_TEMPERATURE = 2.0       # KL temperature on logit-level
DIFFERENTIAL_WEIGHT_L1 = 0.5         # MSE on L1-output differential
DIFFERENTIAL_WEIGHT_LOGIT = 1.0      # KL on head-output differential
REANCHOR_AFTER_PURGE = False         # 2026-05-05: when a dream-rescue
                                      # purge mutates routing_scale on L1,
                                      # copy live routing_scale into
                                      # routing_scale_anchor so feature-
                                      # distillation losses (engram L1-MSE,
                                      # diff δL1) don't fight the purge
                                      # mutation. Targets gcd's task-aware
                                      # drop in combined-storage-free path.
DIFFERENTIAL_USE_ENGRAM = True       # 2026-05-05 chain-B: at storage
                                      # time, generate the per-class
                                      # source x_c via engram gradient
                                      # ascent (Gaussian-perturbed real-
                                      # seed init), then capture the
                                      # (δL0, δL1, δlogit) signature of
                                      # that GA-sharpened class anchor.
                                      # Hypothesis: GA-sharpened deltas
                                      # carry more class-specific signal
                                      # than the real class-mean a single
                                      # random sample provides.
                                      # False = legacy (one random real
                                      # x_c per class).
ENGRAM_LOSS_WEIGHT = 1.0             # engrams are the primary in-weights
                                      # rehearsal signal in this redesign;
                                      # weight at parity with new-task CE.
ENGRAM_BATCH = 64
ENGRAM_TEMPERATURE = 2.0             # KL distillation temperature
ENGRAM_GA_STEPS = 80                 # gradient-ascent steps per engram
ENGRAM_GA_LR = 0.05                  # gradient-ascent step size
ENGRAM_GA_L2 = 1e-3                  # L2 regularization weight on x —
                                      # prevents adversarial degeneracy by
                                      # bounding magnitude
ENGRAM_GA_INIT_NOISE_SCALE = 0.1     # init scale for x ~ noise * scale
ENGRAM_GA_CLIP_RANGE = (0.0, 1.0)    # input-space clipping (image domain
                                      # for chained-15 datasets)
ENGRAM_SEED_FROM_REAL = True         # 2026-05-05 construction repair:
                                      # initialize gradient ascent from a
                                      # random real x_c (Gaussian-perturbed)
                                      # instead of uniform noise. Anchors
                                      # the engram to the data manifold so
                                      # it isn't an adversarial off-manifold
                                      # collapse (probe_engram_diversity
                                      # 2026-05-04 found 7.5x narrower than
                                      # real samples, 9x off-manifold).
ENGRAM_SEED_NOISE_SIGMA = 0.05       # std of Gaussian noise added to the
                                      # real-sample init (post-clip).
ENGRAM_DISTILL_LEVEL = "l1"          # 2026-05-05 construction repair v2:
                                      # "l1"     — MSE on L1-output features
                                      #            between live and anchored
                                      #            networks (this version).
                                      # "logit"  — KL on output logits (the
                                      #            legacy mechanism that
                                      #            failed; preserved for
                                      #            ablation).
                                      # Hypothesis: logit-alignment at the
                                      # adversarial engram point trivially
                                      # holds (KL ≈ 0) and doesn't transfer
                                      # to real-data classification.
                                      # Feature-alignment is smoother and
                                      # constrains internal representation,
                                      # which transfers to nearby points on
                                      # the data manifold.

LOG_EVERY = 500


# ---------------------------------------------------------------------
# Network construction + forward helpers
# ---------------------------------------------------------------------


def make_classifier(
    input_dim: int,
    l0_width: int,
    hidden: int,
    init_classes: int,
    *,
    freeze_l0: bool,
) -> TrioronNetwork:
    """Build the chained-15 classifier.

    Architecture (when freeze_l0=True, used by all `grown_*` arms):
        L0: input_dim → l0_width  (frozen random-projection feature
            extractor — excluded from cap budget; doesn't grow)
        L1: l0_width → hidden    (growable, GROWTH_TARGET_LAYER_IDX=1)
        L2: hidden → init_classes (growable head)

    When freeze_l0=False the same shape is used but L0 is trainable
    (fixed_ewc baseline arm — uses a different width, so this branch
    builds a 2-layer net to keep the matched baseline interpretable).
    """
    if freeze_l0:
        net = TrioronNetwork(
            [
                (input_dim, l0_width, "relu"),
                (l0_width, hidden, "relu"),
                (hidden, init_classes, "linear"),
            ]
        )
        # Freeze L0 (the input adapter). After this, L0.W.grad stays
        # None, EWC penalty for L0 is identically zero (W stays at its
        # init = W_anchor), and Adam built with `requires_grad`-filtered
        # params won't allocate moments for L0.
        l0 = net.layers[0]
        l0.W.requires_grad_(False)
        l0.b.requires_grad_(False)
        return net
    # fixed_ewc baseline: trainable, no growth, 2-hidden MLP at H=hidden.
    return TrioronNetwork(
        [
            (input_dim, hidden, "relu"),
            (hidden, hidden, "relu"),
            (hidden, init_classes, "linear"),
        ]
    )


def trainable_params(net: TrioronNetwork) -> int:
    """Sum of `numel` over parameters with requires_grad=True. Used as
    the cap-accounting denominator so the frozen L0 doesn't eat the
    growable budget."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def trainable_param_iter(net: TrioronNetwork):
    return (p for p in net.parameters() if p.requires_grad)


def _build_optimizer(
    net: TrioronNetwork,
    hat_ctrl: Optional["HATController"] = None,
) -> optim.Optimizer:
    """Build Adam over net's trainable params, plus HAT's task
    embeddings if a HATController is present."""
    params = list(trainable_param_iter(net))
    if hat_ctrl is not None:
        params = params + list(hat_ctrl.parameters())
    return optim.Adam(params, lr=LR)


def warmup_l0(
    real_net: TrioronNetwork,
    infancy_view: TaskDataView,
    *,
    n_steps: int,
    batch: int,
    lr: float,
    temp_hidden: int,
    head_width: int,
    seed: int,
) -> Dict[str, float]:
    """Develop L0 by training a TEMPORARY classifier on the infancy view,
    then copy L0's learned weights into `real_net` (whose L0 is frozen)
    and discard the rest.

    The temp classifier shares L0's shape (input_dim → l0_width) but
    uses a wider scratch L1 and a head wide enough to cover ALL infancy
    classes — that's so warmup gradient flows back into L0 from a
    sufficient signal. After warmup, only L0's W and b are kept; the
    bench's real L1 + head stay at their fresh random init.

    Lickliter framing: brief, intense, disjoint from later experience.
    This is L0's prenatal cascade.
    """
    real_l0 = real_net.layers[0]
    input_dim = real_l0.fan_in
    l0_width = real_l0.n_nodes

    # Build the temp net at a separate seed so it doesn't co-vary with
    # the bench seed. Same input/L0 dims as real_net so we can copy.
    torch.manual_seed(seed)
    temp_net = TrioronNetwork(
        [
            (input_dim, l0_width, "relu"),
            (l0_width, temp_hidden, "relu"),
            (temp_hidden, head_width, "linear"),
        ]
    )
    # Critical: temp L0 starts from the same random init as real L0 so
    # warmup begins from the bench's perceptual prior, not a different
    # random projection.
    with torch.no_grad():
        temp_net.layers[0].W.copy_(real_l0.W.data)
        temp_net.layers[0].b.copy_(real_l0.b.data)

    # All warmup classes are active: standard CE over the full 30-output
    # head, no masking.
    active_all = list(range(head_width))
    opt = optim.Adam(temp_net.parameters(), lr=lr)
    last_loss = float("nan")
    for step in range(n_steps):
        x, y = infancy_view.sample(batch)
        logits = temp_net(x)
        loss = masked_cross_entropy(logits, y, active_classes=active_all)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        if step == 0 or (step + 1) % 100 == 0 or step == n_steps - 1:
            print(f"  [warmup] step {step:4d}  loss {last_loss:.4f}")

    # Copy trained L0 weights into the real net's L0. Update the EWC
    # anchor too — since L0 is frozen, W stays at this value forever
    # and the anchor must agree (else ewc_penalty would be non-zero
    # for L0 across the whole curriculum).
    with torch.no_grad():
        real_l0.W.data.copy_(temp_net.layers[0].W.data)
        real_l0.b.data.copy_(temp_net.layers[0].b.data)
        real_l0.W_anchor.copy_(temp_net.layers[0].W.data)
        real_l0.b_anchor.copy_(temp_net.layers[0].b.data)

    return {"warmup_final_loss": last_loss, "n_warmup_steps": n_steps}


# ---------------------------------------------------------------------
# Cap math (inline — bypasses CeilingsController whose arrest is sticky)
# ---------------------------------------------------------------------


def projected_trainable_after_grow(
    net: TrioronNetwork, target_layer_idx: int,
) -> int:
    """Predict trainable_params(net) after one grow_layer(target_layer_idx).

    Both the new row (W + b on target) and the new column (W on next
    layer) are trainable iff the affected layers are trainable. In the
    frozen-L0 design, target_layer_idx=1 (trainable) and the next layer
    is the head (trainable), so all delta params count.
    """
    target = net.layers[target_layer_idx]
    delta = 0
    if target.W.requires_grad:
        delta += target.fan_in       # +1 W row
        delta += 1                    # +1 b entry
    if target_layer_idx + 1 < len(net.layers):
        nxt = net.layers[target_layer_idx + 1]
        if nxt.W.requires_grad:
            delta += nxt.n_nodes      # +1 W col on next
    return trainable_params(net) + delta


def try_grow_one(
    net: TrioronNetwork,
    target_layer_idx: int,
    cap_bytes: int,
    task_idx: int,
    bytes_per_param: int = 4,
) -> Tuple[bool, str]:
    """Attempt one grow_layer call iff projected trainable params * 4 <= cap_bytes.

    The cap counts TRAINABLE substrate only — frozen layers (L0 in the
    grown_* arms) are excluded so the budget reflects what dreaming can
    actually reclaim. Bypasses CeilingsController whose arrest flag
    prevents resumed growth after dreaming-driven reclaim.
    """
    projected_bytes = projected_trainable_after_grow(
        net, target_layer_idx,
    ) * bytes_per_param
    if projected_bytes > cap_bytes:
        return False, f"cap_exceeded(projected={projected_bytes}B > cap={cap_bytes}B)"
    net.grow_layer(target_layer_idx, init_vec=None, task_idx=task_idx)
    return True, "ok"


# ---------------------------------------------------------------------
# Fisher / consolidation for classification
# ---------------------------------------------------------------------


def estimate_fisher_for_task(
    net: TrioronNetwork,
    train_view: TaskDataView,
    active_classes: Sequence[int],
    batch: int = BATCH,
    n_batches: int = 20,
) -> None:
    def batches():
        for _ in range(n_batches):
            x, y = train_view.sample(batch=batch)
            yield x, y

    active = list(active_classes)

    def loss_fn(pred_logits, y):
        return masked_cross_entropy(pred_logits, y, active_classes=active)

    net.estimate_fisher(batches(), loss_fn, n_batches=n_batches)


def consolidate_task(
    net: TrioronNetwork,
    train_view: TaskDataView,
    active_classes: Sequence[int],
    online_ewc_gamma: Optional[float] = None,
) -> None:
    """Compute Fisher, update lambda, anchor.

    If online_ewc_gamma is set, accumulate Fisher across tasks per
    Schwarz et al. 2018: F_t = γ·F_{t-1} + F_current_task. Otherwise
    use per-task Fisher (resets each task).
    """
    if online_ewc_gamma is not None:
        # Snapshot decayed prior-task Fisher.
        prior_W = []
        prior_b = []
        with torch.no_grad():
            for layer in net.layers:
                prior_W.append(layer.fisher_W.clone() * online_ewc_gamma)
                prior_b.append(layer.fisher_b.clone() * online_ewc_gamma)
        estimate_fisher_for_task(net, train_view, active_classes)
        # Add the decayed prior Fisher to the current-task Fisher.
        with torch.no_grad():
            for layer, pW, pb in zip(net.layers, prior_W, prior_b):
                layer.fisher_W.add_(pW)
                layer.fisher_b.add_(pb)
    else:
        estimate_fisher_for_task(net, train_view, active_classes)
    net.update_lambda_all()
    with torch.no_grad():
        for layer in net.layers:
            layer.lam.clamp_(min=LAMBDA_FLOOR)
    net.anchor_all()


def _store_differential_codes(
    net: TrioronNetwork,
    train_view: TaskDataView,
    active_classes: Sequence[int],
    diff_buf: DifferentialReplayBuffer,
    *,
    use_engram: bool = DIFFERENTIAL_USE_ENGRAM,
    engram_n_steps: int = ENGRAM_GA_STEPS,
    engram_lr: float = ENGRAM_GA_LR,
    engram_l2: float = ENGRAM_GA_L2,
    engram_clip: Tuple[float, float] = ENGRAM_GA_CLIP_RANGE,
    engram_seed_noise_sigma: float = ENGRAM_SEED_NOISE_SIGMA,
) -> None:
    """For each just-learned class, capture (δL0, δL1, δlogit) — the
    per-layer activation differential between processing one canonical
    class-c source and processing blank (zero) input. Stored as a
    tuple per class.

    Two modes for the per-class source:

    - `use_engram=False` (legacy): one random real x_c sample. The
      differential captures the network's interpretation of one
      arbitrary class-c example.

    - `use_engram=True` (2026-05-05 chain-B): generate the source via
      gradient ascent through the *anchored* network (same procedure as
      `_consolidate_engrams`, with Gaussian-perturbed real-seed init),
      then capture deltas from that GA-sharpened class anchor. The
      engram itself is NOT stored — only the per-layer differential
      signature it produces is. Hypothesis: GA-sharpened deltas carry
      more class-specific signal than a single random real sample's
      deltas.

    Call AFTER consolidate_task. For frozen-L0 arms, δL0 stays valid
    indefinitely; for trainable-L0 arms (skipped in this config), δL0
    would go stale.
    """
    net.eval()
    try:
        x_all, y_all = train_view.all_examples()
        device = net.layers[0].W.device

        # Pass 1: pick (or synthesize) the per-class source x_c.
        sources: Dict[int, torch.Tensor] = {}
        for c in active_classes:
            mask = (y_all == c)
            x_c = x_all[mask]
            if x_c.shape[0] == 0:
                continue
            idx = int(torch.randperm(x_c.shape[0])[:1].item())
            if use_engram:
                # Real-seeded gradient ascent through the anchored
                # network. Mirrors `_consolidate_engrams` so the deltas
                # captured here reflect the same class anchor that
                # engram-replay would distill toward, just stored as a
                # multi-level differential instead of an input-space
                # prototype.
                seed = x_c[idx].detach().clone().flatten()
                noise = torch.randn_like(seed) * engram_seed_noise_sigma
                x_init = (seed + noise).clamp_(*engram_clip)
                x_ga = x_init.detach().clone().requires_grad_(True)
                for _ in range(engram_n_steps):
                    logits = net.forward_with_anchors_grad(x_ga.unsqueeze(0))
                    if c >= logits.shape[1]:
                        break
                    loss = -logits[0, c] + engram_l2 * x_ga.pow(2).sum()
                    if x_ga.grad is not None:
                        x_ga.grad.zero_()
                    loss.backward()
                    with torch.no_grad():
                        x_ga.data = (
                            (x_ga.data - engram_lr * x_ga.grad)
                            .clamp_(*engram_clip)
                        )
                sources[int(c)] = x_ga.detach().unsqueeze(0).to(device)
            else:
                sources[int(c)] = x_c[idx:idx + 1].to(device)

        # Pass 2: compute deltas under no_grad for storage.
        with torch.no_grad():
            blank = torch.zeros(1, INPUT_DIM, device=device)
            a0_blank = net.layers[0](blank)            # (1, L0_width)
            a1_blank = net.layers[1](a0_blank)         # (1, L1_width)
            a_logit_blank = net.layers[-1](a1_blank)   # (1, n_classes)
            for c, x_one in sources.items():
                a0 = net.layers[0](x_one)
                a1 = net.layers[1](a0)
                alogit = net.layers[-1](a1)
                dL0 = (a0 - a0_blank).squeeze(0)
                dL1 = (a1 - a1_blank).squeeze(0)
                dlogit = (alogit - a_logit_blank).squeeze(0)
                diff_buf.add_class(int(c), dL0, dL1, dlogit)
    finally:
        net.train()


def _store_hippocampal_codes(
    net: TrioronNetwork,
    train_view: TaskDataView,
    active_classes: Sequence[int],
    hippo: HippocampalBuffer,
    *,
    K: int = HIPPOCAMPAL_K_PER_CLASS,
) -> None:
    """For each just-learned class, sample K real training examples,
    forward them through the (frozen) L0 layer, and store the L0
    outputs as compressed codes in the hippocampal buffer. Call AFTER
    consolidate_task — by then the network's anchored state reflects
    the just-finished task; for frozen L0 it doesn't actually matter
    when this fires (L0 is unchanged), but ordering keeps it parallel
    with brainstem / engram consolidation.

    Storage: K * L0_width float32 per class. For chained-15 with
    K=1, L0=128: 30 classes × 128 × 4 = 15 KB total. Compare raw
    rehearsal (REHEARSAL_SAMPLES_PER_TASK=100, INPUT_DIM=784): 30 ×
    100 × 784 × 4 = 9.4 MB. ~600× compression at K=1; ~6× at matched K.

    Frozen-L0 only: assumes L0 is not being updated. For a trainable-L0
    arm the codes would go stale and need re-encoding per task (or
    fallback to raw-input rehearsal).
    """
    net.eval()
    try:
        with torch.no_grad():
            x_all, y_all = train_view.all_examples()
            l0 = net.layers[0]
            for c in active_classes:
                mask = (y_all == c)
                x_c = x_all[mask]
                if x_c.shape[0] == 0:
                    continue
                k = min(K, x_c.shape[0])
                idx = torch.randperm(x_c.shape[0])[:k]
                x_sample = x_c[idx]
                code = l0(x_sample)  # frozen L0 forward — post-ReLU
                hippo.add_class(int(c), code)
    finally:
        net.train()


def _synthesize_hippocampal_codes(
    net: TrioronNetwork,
    active_classes: Sequence[int],
    seen_classes: Sequence[int],
    hippo: HippocampalBuffer,
    *,
    K: int = HIPPOCAMPAL_K_PER_CLASS,
    n_steps: int = HIPPOCAMPAL_SYNTH_STEPS,
    lr: float = HIPPOCAMPAL_SYNTH_LR,
    l2_weight: float = HIPPOCAMPAL_SYNTH_L2,
    init_sigma: float = HIPPOCAMPAL_SYNTH_INIT_SIGMA,
    div_weight: float = HIPPOCAMPAL_SYNTH_DIV_WEIGHT,
) -> None:
    """DeepInversion-style synthesis of K L0-output codes per class.

    For each class c in active_classes, optimize a (K, L0_WIDTH) tensor z
    via Adam to minimize masked_cross_entropy(forward_from_layer(z, 1), c)
    + L2(z), with z = softplus(raw_z) so post-ReLU non-negativity is
    enforced by reparameterization. Network params are frozen during
    inversion (requires_grad disabled, then restored).

    Active class set for the masked CE is `seen_classes` — all classes
    consolidated so far including the current task — so the synthesized
    code peaks at c relative to every other already-known class, not
    just the current task's pair.

    Storage shape is identical to _store_hippocampal_codes' real-sample
    output (K, l0_width); replay path is unchanged. Net effect: same
    buffer footprint, no peek at real training samples at consolidation.
    """
    device = net.layers[1].W.device
    l0_width = net.layers[0].n_nodes
    seen_set = sorted(set(int(c) for c in seen_classes))
    if not seen_set:
        return

    saved_grad = [(p, p.requires_grad) for p in net.parameters()]
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()
    try:
        for c in active_classes:
            raw_z = (init_sigma * torch.randn(
                K, l0_width, device=device,
            )).requires_grad_(True)
            opt = torch.optim.Adam([raw_z], lr=lr)
            target = torch.full(
                (K,), int(c), dtype=torch.long, device=device,
            )
            eye_K = torch.eye(K, dtype=torch.bool, device=device)
            for _ in range(n_steps):
                z = F.softplus(raw_z)
                logits = net.forward_from_layer(z, start_layer=1)
                ce = masked_cross_entropy(
                    logits, target, active_classes=seen_set,
                )
                l2 = (z * z).mean()
                # Within-K diversity: penalize squared cosine similarity
                # between every pair of codes for class c. Forces the K
                # codes to spread across the class-c attractor basin
                # instead of collapsing to a single spike.
                z_n = F.normalize(z, dim=1, eps=1e-8)
                sim = z_n @ z_n.T
                div = (sim[~eye_K] ** 2).mean()
                loss = ce + l2_weight * l2 + div_weight * div
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                z_final = F.softplus(raw_z).detach()
            hippo.add_class(int(c), z_final)
    finally:
        for p, was in saved_grad:
            p.requires_grad_(was)
        net.train()


def _consolidate_engrams(
    net: TrioronNetwork,
    active_classes: Sequence[int],
    engrams: EngramBuffer,
    *,
    train_view: Optional[TaskDataView] = None,
    input_dim: int = INPUT_DIM,
    n_steps: int = ENGRAM_GA_STEPS,
    lr: float = ENGRAM_GA_LR,
    l2_weight: float = ENGRAM_GA_L2,
    init_noise_scale: float = ENGRAM_GA_INIT_NOISE_SCALE,
    clip_range: Tuple[float, float] = ENGRAM_GA_CLIP_RANGE,
    seed_from_real: bool = ENGRAM_SEED_FROM_REAL,
    seed_noise_sigma: float = ENGRAM_SEED_NOISE_SIGMA,
) -> None:
    """Find one engram per class in `active_classes` by gradient ascent
    on the input through the anchored network. Stores each x_c in the
    `engrams` buffer. Call AFTER `consolidate_task` (so anchors reflect
    the just-finished task's consolidated state) and AFTER any other
    consolidation step (Brainstem, archive) that depends on the live
    state — engram consolidation only reads from the anchored forward.

    Procedure (per class c):
      1. Initialize x. Two modes:
         - seed_from_real=True (default, 2026-05-05 repair): sample one
           random real x_c from train_view, add Gaussian noise of std
           seed_noise_sigma, clip to clip_range. Anchors the engram to
           the data manifold so the GA result stays in the neighborhood
           of real class-c examples instead of collapsing to an
           adversarial off-manifold cluster.
         - seed_from_real=False (legacy): x ~ uniform noise scaled by
           init_noise_scale, clipped to clip_range. This is the original
           construction; probe_engram_diversity 2026-05-04 found it
           produces engrams 7.5x narrower than real samples and 9x off
           the data manifold, which is why engram-only bench delivers
           ~no-rehearsal-floor accuracy.
      2. Repeat n_steps times:
         - logits = net.forward_with_anchors(x.unsqueeze(0))
         - loss = -logits[0, c] + l2_weight * x.pow(2).sum()
         - x ← clip(x - lr * dloss/dx, *clip_range)
           (gradient ASCENT on logit_c → minus sign on the logit term;
            descent on L2 keeps magnitude bounded.)
      3. Store the resulting x as the engram for class c.

    Note: forward_with_anchors uses W_anchor + b_anchor +
    routing_scale_anchor (the full triparametric anchored state, after
    the routing-scale fix). Gradient flows through every layer
    including the frozen L0, so the engram is an end-to-end input-space
    prototype — what the consolidated network thinks the most class-c-
    confident input pattern is.
    """
    if seed_from_real and train_view is None:
        raise ValueError(
            "seed_from_real=True requires train_view (real-sample "
            "initialization needs access to the task's class examples)"
        )
    if seed_from_real:
        x_all, y_all = train_view.all_examples()
    net.eval()
    try:
        for c in active_classes:
            if seed_from_real:
                mask = (y_all == c)
                x_c = x_all[mask]
                if x_c.shape[0] == 0:
                    # No real samples — fall back to uniform noise.
                    x = torch.empty(input_dim).uniform_(
                        clip_range[0],
                        clip_range[0] + init_noise_scale
                        * (clip_range[1] - clip_range[0]),
                    )
                else:
                    idx = int(torch.randint(0, x_c.shape[0], (1,)).item())
                    seed = x_c[idx].detach().clone().flatten()
                    noise = torch.randn_like(seed) * seed_noise_sigma
                    x = (seed + noise).clamp_(*clip_range)
            else:
                x = torch.empty(input_dim).uniform_(
                    clip_range[0],
                    clip_range[0] + init_noise_scale
                    * (clip_range[1] - clip_range[0]),
                )
            x.requires_grad_(True)
            for _ in range(n_steps):
                logits = net.forward_with_anchors_grad(x.unsqueeze(0))
                if c >= logits.shape[1]:
                    # Head not yet wide enough to cover this class —
                    # shouldn't happen given engrams are consolidated
                    # AFTER head extension and CE training on the class,
                    # but guard anyway.
                    break
                loss = -logits[0, c] + l2_weight * x.pow(2).sum()
                if x.grad is not None:
                    x.grad.zero_()
                loss.backward()
                with torch.no_grad():
                    x.data = (
                        (x.data - lr * x.grad)
                        .clamp_(clip_range[0], clip_range[1])
                    )
            engrams.add_class(int(c), x.detach())
    finally:
        net.train()


def _store_manifold_stats(
    net: TrioronNetwork,
    train_view: TaskDataView,
    active_classes: Sequence[int],
    manifold: ManifoldBuffer,
    max_samples_per_class: int = MANIFOLD_MAX_SAMPLES_PER_CLASS,
) -> None:
    """Capture per-class (μ, σ) at L0 output. Forward each class's real
    samples through the (frozen) L0 layer once and fit a diagonal
    Gaussian. Frozen-L0 only — for trainable-L0 arms the stats would
    drift; the bench gates on arm_l0_frozen.

    Storage cost: 30 classes × (mean + var) × 128 × 4 = ~30 KB total.
    Independent of K — codes are sampled at replay time, not stored.
    """
    net.eval()
    try:
        with torch.no_grad():
            x_all, y_all = train_view.all_examples()
            l0 = net.layers[0]
            for c in active_classes:
                mask = (y_all == c)
                x_c = x_all[mask]
                if x_c.shape[0] == 0:
                    continue
                if x_c.shape[0] > max_samples_per_class:
                    idx = torch.randperm(
                        x_c.shape[0]
                    )[:max_samples_per_class]
                    x_c = x_c[idx]
                z_c = l0(x_c)  # (n, l0_width), post-ReLU
                mu = z_c.mean(dim=0).detach()
                sigma = z_c.std(dim=0).detach()
                manifold.add_class(int(c), mu, sigma)
    finally:
        net.train()


def _store_brainstem_stats(
    net: TrioronNetwork,
    train_view: TaskDataView,
    active_classes: Sequence[int],
    brainstem: BrainstemBuffer,
    max_samples_per_class: int = 1000,
) -> None:
    """Compute per-class (μ, σ) at the L1 output on this task's training
    data and store in the brainstem buffer. Caps at max_samples_per_class
    to keep wall-clock manageable on the larger MNIST tasks (~6000 per
    class). The cap is enough to estimate diagonal Gaussian stats with
    low variance.
    """
    net.eval()
    try:
        with torch.no_grad():
            x_all, y_all = train_view.all_examples()
            for c in active_classes:
                mask = (y_all == c)
                x_c = x_all[mask]
                if x_c.shape[0] == 0:
                    continue
                if x_c.shape[0] > max_samples_per_class:
                    idx = torch.randperm(x_c.shape[0])[:max_samples_per_class]
                    x_c = x_c[idx]
                # Forward through L0 + L1, stopping at the bottleneck
                # (L1 output, the head's input).
                h = x_c
                for layer in net.layers[: GROWTH_TARGET_LAYER_IDX + 1]:
                    h = layer(h)
                # h is now (n, l1_width); diagonal Gaussian.
                mu = h.mean(dim=0).detach()
                sigma = h.std(dim=0).detach()
                brainstem.add_class(int(c), mu, sigma)
    finally:
        net.train()


# ---------------------------------------------------------------------
# Utility-update during training (needed so purge has a real signal)
# ---------------------------------------------------------------------


def update_layer_utilities(net: TrioronNetwork) -> None:
    """Capture a per-node utility update via |y · ∂L/∂y| saliency.

    Switched from |W|·|grad_W| to true OBD saliency on 2026-05-03 after
    the chained-15 n=12 sum+floor result showed dream-vs-no_dream still
    +1.33σ no_dream-better with a variance balloon. The old |W|·|grad_W|
    summary is biased toward weight-magnitude rather than functional
    contribution: it can flag dead-relu nodes with large incoming
    weights as "important" (false positive) and active small-weight
    nodes as "unimportant" (false negative). Saliency directly answers
    "if I clamped this node's output to zero, by how much would loss
    change," which is exactly what purge victim selection needs.

    Call after .backward(), before optimizer.step(). Layers cache the
    forward y and capture upstream ∂L/∂y via a backward hook
    (trioron/node.py). On a no-grad/eval forward, no hook fires and
    the cached saliency from the previous training forward survives.
    """
    net.update_utilities_from_saliency()


# ---------------------------------------------------------------------
# Classification-shaped dreaming block
# ---------------------------------------------------------------------


def _classification_replay(
    net: TrioronNetwork,
    past_views: Sequence[TaskDataView],
    past_active_classes: Sequence[Sequence[int]],
    *,
    fraction: float,
    n_steps_per_task: int,
    batch: int,
    lr: float,
    ewc_strength: float,
    rng: random.Random,
    update_utilities: bool = False,
    mask_archived_grads: bool = False,
) -> Tuple[float, float, int, int]:
    """CE-shaped analog of dreaming.replay. Returns
    (avg_loss_before, avg_loss_after, n_tasks_sampled, total_steps).

    If update_utilities is True, the per-step backward updates the
    per-node utility u via OBD saliency. Tasks are visited round-robin
    (one batch per task per outer loop) so the u-EMA ends up reflecting
    a mix of past tasks rather than only the last task — fixes the
    seed-6-Fashion failure mode (n=12 saliency bench, 2026-05-03).
    """
    if not past_views:
        return (0.0, 0.0, 0, 0)
    n = len(past_views)
    k = max(1, int(round(fraction * n)))
    idxs = rng.sample(range(n), k=min(k, n))

    def _avg_loss() -> float:
        net.eval()
        total = 0.0
        with torch.no_grad():
            for i in idxs:
                v = past_views[i]
                active = list(past_active_classes[i])
                x, y = v.sample(batch)
                total += float(masked_cross_entropy(net(x), y, active).item())
        net.train()
        return total / len(idxs)

    loss_before = _avg_loss()
    opt = optim.Adam(trainable_param_iter(net), lr=lr)
    # Round-robin: total_steps = n_steps_per_task × tasks_sampled, but
    # each outer step samples ONE batch from ONE task and cycles
    # through tasks in order. Equivalent total work to the old
    # task-by-task loop but the EMA-weighted u at the end spans tasks.
    total_steps = n_steps_per_task * len(idxs)
    for step in range(total_steps):
        i = idxs[step % len(idxs)]
        v = past_views[i]
        active = list(past_active_classes[i])
        x, y = v.sample(batch)
        l_task = masked_cross_entropy(net(x), y, active)
        l = (l_task + ewc_strength * net.ewc_penalty()
             if ewc_strength > 0 else l_task)
        opt.zero_grad()
        l.backward()
        if mask_archived_grads:
            net.mask_archived_grads_all()
        if update_utilities:
            update_layer_utilities(net)
        opt.step()
    loss_after = _avg_loss()
    return (loss_before, loss_after, len(idxs), total_steps)


def _build_classification_probe(
    past_views: Sequence[TaskDataView],
    probe_batch_size: int,
    rng: random.Random,
) -> Optional[torch.Tensor]:
    if not past_views:
        return None
    per = max(1, probe_batch_size // len(past_views))
    chunks: List[torch.Tensor] = []
    for v in past_views:
        x, _ = v.sample(per, generator=None)
        chunks.append(x)
    out = torch.cat(chunks, dim=0)
    if out.shape[0] > probe_batch_size:
        out = out[:probe_batch_size]
    return out


def classification_dreaming_block(
    net: TrioronNetwork,
    past_views: Sequence[TaskDataView],
    past_active_classes: Sequence[Sequence[int]],
    *,
    rng: random.Random,
    mode: str,
) -> Dict[str, object]:
    """CE-shaped dreaming. Two modes:

    mode='replay_only' — used post-task to keep prior memories warm.
        Runs apoptosis_decay (so any spike from a prior block fades)
        then replay. NO compress, NO purge — substrate is unchanged.
        This is the "consolidation rest" mode.

    mode='reclaim' — used on growth-denial to free substrate.
        replay + compress(starve+apoptosis) + purge restricted to the
        growth-target layer. Purge u_threshold is high enough that
        starvation-decayed units actually vacate; layer 0 (the 784-fan
        adapter) is NEVER purged because dropping a layer-0 unit
        wipes 784 weights of feature-detector capacity for prior tasks.

    Returns a flat dict. Caller MUST rebuild the optimizer if
    `n_purges > 0` (purge replaces Parameter objects).
    """
    if mode not in ("replay_only", "reclaim"):
        raise ValueError(f"mode must be 'replay_only' or 'reclaim', got {mode!r}")

    n_before = net.n_parameters()
    arch_before = tuple(net.n_nodes_per_layer())

    if DREAM_APOPTOSIS_ON:
        apoptosis_decay(net, decay_rate=DREAM_APOPTOSIS_DECAY_RATE)

    # Reclaim mode: reset u and use full-coverage replay so the post-
    # replay u reflects EVERY past task's saliency, not a sampled
    # subset. Replay_only mode keeps the cheaper sampled-replay since
    # its u writes are inert (no purge follows).
    if mode == "reclaim":
        net.reset_utilities_all()
        replay_fraction = DREAM_RECLAIM_REPLAY_FRACTION
        replay_writes_u = True
    else:
        replay_fraction = DREAM_REPLAY_FRACTION
        replay_writes_u = False

    loss_before, loss_after, n_tasks, n_steps = _classification_replay(
        net, past_views, past_active_classes,
        fraction=replay_fraction,
        n_steps_per_task=DREAM_REPLAY_STEPS,
        batch=DREAM_REPLAY_BATCH,
        lr=LR,
        ewc_strength=EWC_DREAM_STRENGTH,
        rng=rng,
        update_utilities=replay_writes_u,
    )

    merges: List[MergeEvent] = []
    purges: List[PurgeEvent] = []

    if mode == "reclaim":
        probe = _build_classification_probe(
            past_views, DREAM_PROBE_BATCH_SIZE, rng,
        )
        if probe is not None:
            merges = compress(
                net,
                layer_idxs=[GROWTH_TARGET_LAYER_IDX],   # only the growth target
                redundancy_signal="activation",
                probe_batch=probe,
                ac_threshold=DREAM_AC_THRESHOLD,
                compression_action=DREAM_COMPRESSION_ACTION,
                max_downscales_per_layer=DREAM_MAX_DOWNSCALES_PER_LAYER,
                starvation_alpha=DREAM_STARVATION_ALPHA,
                starvation_floor=DREAM_STARVATION_FLOOR,
                apoptosis_on=DREAM_APOPTOSIS_ON,
                apoptosis_spike_init=DREAM_APOPTOSIS_SPIKE_INIT,
                skip_output_layer=True,
            )
        # Restrict purge to the growth-target layer ONLY. Layer 0
        # (the input adapter) and the head (output) stay untouched.
        # Throttle: at most DREAM_MAX_PURGES_PER_EVENT victims per
        # dream block (biology runs apoptosis slowly; the bench needs
        # multi-event reclaim across the curriculum, not single-event
        # collapse).
        purges = purge(
            net,
            layer_idxs=[GROWTH_TARGET_LAYER_IDX],
            u_threshold=DREAM_U_THRESHOLD,
            skip_output_layer=False,  # we already constrain via layer_idxs
            max_purges=DREAM_MAX_PURGES_PER_EVENT,
        )

    return {
        "n_params_before": n_before,
        "n_params_after": net.n_parameters(),
        "arch_before": arch_before,
        "arch_after": tuple(net.n_nodes_per_layer()),
        "replay_loss_before": loss_before,
        "replay_loss_after": loss_after,
        "n_replay_tasks": n_tasks,
        "n_replay_steps": n_steps,
        "n_merges": len(merges),
        "n_purges": len(purges),
        "n_latched": sum(1 for m in merges if m.victim_latched),
        "mode": mode,
    }


def consolidation_dream_pass(
    net: TrioronNetwork,
    past_views: Sequence[TaskDataView],
    past_active_classes: Sequence[Sequence[int]],
    *,
    rng: random.Random,
    replay_steps_per_task: int = DREAM_REPLAY_STEPS,
) -> Dict[str, object]:
    """Heavier "shipping consolidation" dream — full-coverage replay
    over ALL past tasks (replay_fraction = 1.0) with archive-aware
    gradient masking, then apoptosis_decay sweep, then one more
    archive_block call to catch any rows that just settled.

    Distinct from per-task replay_only (DREAM_REPLAY_FRACTION=0.25, no
    archive mask). This is the dream that fires before the network is
    "shipped" / extended onto a new curriculum: settled state, archived
    rows truly locked, no further drift on what's been consolidated.
    """
    if DREAM_APOPTOSIS_ON:
        apoptosis_decay(net, decay_rate=DREAM_APOPTOSIS_DECAY_RATE)

    loss_before, loss_after, n_tasks, n_steps = _classification_replay(
        net, past_views, past_active_classes,
        fraction=1.0,
        n_steps_per_task=replay_steps_per_task,
        batch=DREAM_REPLAY_BATCH,
        lr=LR,
        ewc_strength=EWC_DREAM_STRENGTH,
        rng=rng,
        update_utilities=False,
        mask_archived_grads=True,
    )

    archived_now: List[Tuple[int, int]] = []
    if ARCHIVE_ENABLED:
        archived_now = archive_block(
            net,
            streak_threshold=ARCHIVE_STREAK_THRESHOLD,
            lam_top_percentile=ARCHIVE_LAM_TOP_PERCENTILE,
            grad_mag_floor=ARCHIVE_GRAD_MAG_FLOOR,
            pulse_max=ARCHIVE_PULSE_MAX,
            skip_output_layer=ARCHIVE_SKIP_OUTPUT_LAYER,
            max_archives_per_layer=ARCHIVE_MAX_PER_LAYER,
        )

    return {
        "replay_loss_before": loss_before,
        "replay_loss_after": loss_after,
        "n_replay_tasks": n_tasks,
        "n_replay_steps": n_steps,
        "n_archived_now": len(archived_now),
        "n_archived_per_layer": net.n_archived_per_layer(),
    }


# ---------------------------------------------------------------------
# Per-task training loop
# ---------------------------------------------------------------------


def train_one_task(
    net: TrioronNetwork,
    task_idx: int,
    train_view: TaskDataView,
    active_classes: Sequence[int],
    n_epochs: int,
    opt: optim.Optimizer,
    *,
    ewc_baseline: float,
    label: str,
    n_total_tasks: int,
    epoch_offset: int = 0,
    total_epochs_outer: Optional[int] = None,
    epoch_label_suffix: str = "",
    memory: Optional[MemoryBuffer] = None,
    lwf_old_classes: Optional[Sequence[int]] = None,
    brainstem: Optional[BrainstemBuffer] = None,
    engrams: Optional[EngramBuffer] = None,
    engram_old_classes: Optional[Sequence[int]] = None,
    hippocampus: Optional[HippocampalBuffer] = None,
    differential: Optional[DifferentialReplayBuffer] = None,
    manifold: Optional[ManifoldBuffer] = None,
    packnet_ctrl: Optional[PackNetController] = None,
    hat_ctrl: Optional[HATController] = None,
) -> optim.Optimizer:
    """Train on one task for `n_epochs` proper minibatch epochs.

    Each epoch is a fresh random shuffle of the task data; every sample
    is touched exactly once per epoch. This is Gemma's settling-cycle
    framing — the model gets stable input shape to commit to a
    representation, instead of being firehosed by random-with-replacement
    batches that leave ~37% of samples unseen on a smoke-budget run.

    `epoch_offset` and `total_epochs_outer` let Fix B (growth gating)
    split a task's training into "settle" and "post-grow" phases while
    keeping the log labels coherent: epoch 1/4, epoch 2/4 [settle];
    epoch 3/4, epoch 4/4 [post-grow]. `epoch_label_suffix` is appended
    to the log line for the same purpose.
    """
    active = list(active_classes)
    total_steps = 0
    last_loss = float("nan")
    outer_total = total_epochs_outer if total_epochs_outer is not None else n_epochs
    # HAT temperature anneal — linear from s_min to s_max over the
    # task's full training. Estimate step budget from data size.
    if hat_ctrl is not None:
        n_per_epoch = (train_view.n_examples() + BATCH - 1) // BATCH
        hat_total_in_call = max(1, n_epochs * n_per_epoch)
    else:
        hat_total_in_call = 1
    for epoch in range(n_epochs):
        epoch_loss_sum = 0.0
        epoch_n_batches = 0
        for x, y_global in train_view.iter_epoch(BATCH):
            if hat_ctrl is not None:
                s = hat_ctrl.temperature_for_step(total_steps, hat_total_in_call)
                hat_ctrl.set_temperature(s)
            logits = net(x)
            l_task = masked_cross_entropy(logits, y_global, active_classes=active)

            # Rehearsal (Path 2 raw-sample): default off; preserved for
            # ablation. See REHEARSAL_ENABLED.
            l_rehearsal = None
            if memory is not None and memory.has_samples():
                head_size = net.layers[-1].n_nodes
                x_r, y_r = memory.sample(REHEARSAL_BATCH)
                if x_r is not None:
                    logits_r = net(x_r)
                    all_seen = list(range(head_size))
                    l_rehearsal = masked_cross_entropy(
                        logits_r, y_r, active_classes=all_seen,
                    )

            # A — Parasitic Dream / LwF distillation. Compute the
            # anchored network's logits on the SAME current batch (no
            # grad) and KL-distill the live network's response toward
            # the anchor's response on the OLD-class columns only. New-
            # class columns are excluded — their anchor is just init.
            l_lwf = None
            if (lwf_old_classes is not None
                    and len(lwf_old_classes) > 0
                    and LWF_LOSS_WEIGHT > 0):
                with torch.no_grad():
                    z_old = net.forward_with_anchors(x)
                T = LWF_TEMPERATURE
                old_idx = torch.as_tensor(
                    list(lwf_old_classes), dtype=torch.long,
                    device=logits.device,
                )
                # Slice to old-class columns; if head isn't yet that
                # wide (shouldn't happen in normal flow but guard
                # anyway), drop any out-of-range entries.
                in_range = old_idx[old_idx < logits.shape[1]]
                if in_range.numel() > 1:
                    z_old_old = z_old.index_select(1, in_range)
                    z_cur_old = logits.index_select(1, in_range)
                    p_old = F.softmax(z_old_old / T, dim=1)
                    log_p_cur = F.log_softmax(z_cur_old / T, dim=1)
                    l_lwf = F.kl_div(
                        log_p_cur, p_old, reduction="batchmean",
                    ) * (T * T)

            # B — Brainstem Spark / latent Gaussian rehearsal. Sample
            # synthetic L1-output latents from per-class Gaussians
            # stored at consolidation time; feed directly into the head
            # (skipping L0+L1) and CE-supervise to maintain the class
            # boundary on synthetic in-distribution latents.
            l_brainstem = None
            if (brainstem is not None
                    and brainstem.has_classes()
                    and BRAINSTEM_LOSS_WEIGHT > 0):
                head_in_dim = net.layers[GROWTH_TARGET_LAYER_IDX].n_nodes
                z_b, y_b = brainstem.sample(BRAINSTEM_BATCH, head_in_dim)
                if z_b is not None:
                    head_W = net.layers[-1].W
                    z_b = z_b.to(device=head_W.device, dtype=head_W.dtype)
                    y_b = y_b.to(device=head_W.device)
                    logits_b = net.forward_from_layer(
                        z_b, start_layer=GROWTH_TARGET_LAYER_IDX + 1,
                    )
                    head_size = net.layers[-1].n_nodes
                    all_seen_b = list(range(head_size))
                    l_brainstem = masked_cross_entropy(
                        logits_b, y_b, active_classes=all_seen_b,
                    )

            # Engram Replay (triparametric A+B redesign). Sample stored
            # input-space engrams; forward them through both live and
            # anchored networks; KL-distill live → anchor on old-class
            # output columns. This is the trioron-native LwF: the
            # rehearsal signal (engrams) traverses ALL three legs of
            # the consolidated triparametric trioron via
            # forward_with_anchors (W_anchor + b_anchor +
            # routing_scale_anchor). Old-class slice keeps current-task
            # CE the dominant signal on new classes.
            l_engram_lwf = None
            if (engrams is not None
                    and engrams.has_classes()
                    and engram_old_classes is not None
                    and len(engram_old_classes) > 0
                    and ENGRAM_LOSS_WEIGHT > 0):
                x_e, _y_e = engrams.sample(ENGRAM_BATCH)
                if x_e is not None:
                    x_e = x_e.to(device=net.layers[0].W.device)
                    if ENGRAM_DISTILL_LEVEL == "l1":
                        # L1-feature MSE distillation. Forward the engram
                        # through L0 + L1 with live params (gradient-tracked)
                        # and with anchored params (no grad), MSE the L1
                        # outputs. W_anchor / b_anchor / routing_scale_anchor
                        # extend with grow_node so dimensions match live.
                        h_live = x_e
                        for layer in net.layers[: GROWTH_TARGET_LAYER_IDX + 1]:
                            h_live = layer(h_live)
                        with torch.no_grad():
                            h_anchor = x_e
                            for layer in net.layers[: GROWTH_TARGET_LAYER_IDX + 1]:
                                if h_anchor.dtype != layer.W_anchor.dtype:
                                    h_anchor = h_anchor.to(layer.W_anchor.dtype)
                                scale = layer.routing_scale_anchor.unsqueeze(1).to(
                                    layer.W_anchor.dtype
                                )
                                W_eff = layer.W_anchor * scale
                                z = F.linear(h_anchor, W_eff, layer.b_anchor)
                                h_anchor = (
                                    F.relu(z)
                                    if layer.activation == "relu"
                                    else z
                                )
                        l_engram_lwf = F.mse_loss(h_live, h_anchor)
                    else:
                        # Legacy: logit-KL on old-class slice.
                        z_live = net(x_e)
                        with torch.no_grad():
                            z_anchor = net.forward_with_anchors(x_e)
                        T_eng = ENGRAM_TEMPERATURE
                        e_old_idx = torch.as_tensor(
                            list(engram_old_classes), dtype=torch.long,
                            device=z_live.device,
                        )
                        e_in_range = e_old_idx[e_old_idx < z_live.shape[1]]
                        if e_in_range.numel() > 1:
                            z_a_old = z_anchor.index_select(1, e_in_range)
                            z_l_old = z_live.index_select(1, e_in_range)
                            p_a = F.softmax(z_a_old / T_eng, dim=1)
                            log_p_l = F.log_softmax(z_l_old / T_eng, dim=1)
                            l_engram_lwf = F.kl_div(
                                log_p_l, p_a, reduction="batchmean",
                            ) * (T_eng * T_eng)

            # Differential Replay: bias-drift-invariant rehearsal.
            # Sample stored class-c differentials at L0/L1/head; force
            # the live network to maintain the same differential
            # signature regardless of memory drift. Injection is at L1
            # (the middle): we feed (live_blank_L0 + δL0_c) into live
            # L1 and check that the live differential matches the
            # stored δL1_c at the L1-output level (MSE) and the
            # δlogit_c at the head level (KL).
            l_diff = None
            if (differential is not None
                    and differential.has_classes()
                    and (DIFFERENTIAL_WEIGHT_L1 > 0
                         or DIFFERENTIAL_WEIGHT_LOGIT > 0)):
                l0_w = net.layers[0].n_nodes
                l1_w = net.layers[GROWTH_TARGET_LAYER_IDX].n_nodes
                head_w = net.layers[-1].n_nodes
                dL0_b, dL1_b, dlogit_b, _y_d = differential.sample(
                    DIFFERENTIAL_BATCH, l0_w, l1_w, head_w,
                )
                if dL0_b is not None:
                    device = net.layers[0].W.device
                    dL0_b = dL0_b.to(device)
                    dL1_b = dL1_b.to(device)
                    dlogit_b = dlogit_b.to(device)
                    blank = torch.zeros(1, INPUT_DIM, device=device)
                    with torch.no_grad():
                        a0_blank = net.layers[0](blank)              # (1, L0)
                        a1_blank_nograd = net.layers[1](a0_blank)    # (1, L1)
                        a_logit_blank_nograd = net.layers[-1](a1_blank_nograd)
                    # synth L0 input batch: (B, L0_width)
                    synth_L0 = a0_blank.expand(DIFFERENTIAL_BATCH, -1) + dL0_b
                    # live L1 forward — gradient-tracked
                    a1_live = net.layers[1](synth_L0)                # (B, L1)
                    a_logit_live = net.layers[-1](a1_live)           # (B, n_classes)
                    # live differentials (broadcast blank to batch)
                    live_dL1 = a1_live - a1_blank_nograd             # (B, L1)
                    live_dlogit = a_logit_live - a_logit_blank_nograd # (B, n_classes)
                    l_diff_terms: List[torch.Tensor] = []
                    if DIFFERENTIAL_WEIGHT_L1 > 0:
                        l_diff_terms.append(
                            DIFFERENTIAL_WEIGHT_L1
                            * F.mse_loss(live_dL1, dL1_b)
                        )
                    if DIFFERENTIAL_WEIGHT_LOGIT > 0:
                        T_d = DIFFERENTIAL_TEMPERATURE
                        # Match logit-differential distributions via KL
                        # (after softmax). The differential here can be
                        # negative on some classes (suppressed) and
                        # positive on the target — softmax normalizes.
                        p_target = F.softmax(dlogit_b / T_d, dim=1)
                        log_p_live = F.log_softmax(live_dlogit / T_d, dim=1)
                        l_diff_terms.append(
                            DIFFERENTIAL_WEIGHT_LOGIT
                            * F.kl_div(log_p_live, p_target,
                                       reduction="batchmean")
                            * (T_d * T_d)
                        )
                    if l_diff_terms:
                        l_diff = sum(l_diff_terms)

            # Hippocampal Replay: sample stored L0 codes and feed
            # them directly into L1 via forward_from_layer(start=1).
            # CE supervises against the all-classes-seen mask so the
            # head learns to discriminate every class via the
            # canonical L0 codes.
            l_hippo = None
            if (hippocampus is not None
                    and hippocampus.has_classes()
                    and HIPPOCAMPAL_LOSS_WEIGHT > 0):
                z_h, y_h = hippocampus.sample(HIPPOCAMPAL_BATCH)
                if z_h is not None:
                    z_h = z_h.to(device=net.layers[1].W.device)
                    y_h = y_h.to(device=net.layers[1].W.device)
                    logits_h = net.forward_from_layer(z_h, start_layer=1)
                    head_size = net.layers[-1].n_nodes
                    all_seen_h = list(range(head_size))
                    l_hippo = masked_cross_entropy(
                        logits_h, y_h, active_classes=all_seen_h,
                    )

            # Manifold Replay: sample fresh L0 codes from per-class
            # diagonal Gaussian and feed via the same path as hippo.
            # No stored codes — codes are drawn on demand each step.
            l_manifold = None
            if (manifold is not None
                    and manifold.has_classes()
                    and MANIFOLD_REPLAY_LOSS_WEIGHT > 0):
                z_m, y_m = manifold.sample(
                    MANIFOLD_REPLAY_BATCH,
                    noise_scale=MANIFOLD_NOISE_SCALE,
                )
                if z_m is not None:
                    z_m = z_m.to(device=net.layers[1].W.device)
                    y_m = y_m.to(device=net.layers[1].W.device)
                    logits_m = net.forward_from_layer(z_m, start_layer=1)
                    head_size = net.layers[-1].n_nodes
                    all_seen_m = list(range(head_size))
                    l_manifold = masked_cross_entropy(
                        logits_m, y_m, active_classes=all_seen_m,
                    )

            l_data = l_task
            if l_rehearsal is not None:
                l_data = l_data + REHEARSAL_LOSS_WEIGHT * l_rehearsal
            if l_lwf is not None:
                l_data = l_data + LWF_LOSS_WEIGHT * l_lwf
            if l_brainstem is not None:
                l_data = l_data + BRAINSTEM_LOSS_WEIGHT * l_brainstem
            if l_engram_lwf is not None:
                l_data = l_data + ENGRAM_LOSS_WEIGHT * l_engram_lwf
            if l_hippo is not None:
                l_data = l_data + HIPPOCAMPAL_LOSS_WEIGHT * l_hippo
            if l_manifold is not None:
                l_data = l_data + MANIFOLD_REPLAY_LOSS_WEIGHT * l_manifold
            if l_diff is not None:
                l_data = l_data + l_diff
            # HAT sparsity regularizer — pushes the current task to share
            # mask units with prior tasks (Serrà §3.3).
            if hat_ctrl is not None:
                l_data = l_data + (
                    hat_ctrl.sparsity_coef * hat_ctrl.sparsity_loss()
                )
            l = (l_data + ewc_baseline * net.ewc_penalty()
                 if ewc_baseline > 0 else l_data)
            opt.zero_grad()
            l.backward()
            # PackNet: zero gradients on weights belonging to past tasks
            # so the optimizer can't update them. Must fire after
            # backward() and before step().
            if packnet_ctrl is not None:
                packnet_ctrl.freeze_grads()
            # HAT: scale gradients on weights protected by prior-task
            # masks (input-side and output-side). Same lifecycle slot
            # as PackNet's freeze_grads.
            if hat_ctrl is not None:
                hat_ctrl.scale_grads()
            # Dream-archive Phase 1: zero grads on archived rows so
            # archived weights stay locked at consolidated values
            # regardless of the optimizer. Same lifecycle slot as
            # PackNet/HAT — between backward and step. No-op if no
            # rows archived.
            if ARCHIVE_ENABLED:
                net.mask_archived_grads_all()
            # Note: NOT updating per-node utilities during normal
            # training — u is now driven exclusively by dream-rescue
            # replay (set in classification_dreaming_block when
            # mode='reclaim'). Writing u during training would mix
            # current-task saliency into u, biasing purge victim
            # selection toward "what doesn't help the current task"
            # rather than "what doesn't help any past task" — the
            # exact failure mode that produced seed-6's catastrophic
            # Fashion regression in the n=12 saliency bench.
            opt.step()
            # HAT: clamp |e| to keep sigmoid in a learnable regime.
            if hat_ctrl is not None:
                hat_ctrl.clip_embeddings()
            total_steps += 1
            last_loss = float(l_task.item())
            epoch_loss_sum += last_loss
            epoch_n_batches += 1
        epoch_avg = epoch_loss_sum / max(1, epoch_n_batches)
        global_epoch = epoch_offset + epoch + 1
        print(f"  [{label}] task {task_idx+1}/{n_total_tasks} ({train_view.name}) "
              f"epoch {global_epoch}/{outer_total}{epoch_label_suffix}  "
              f"avg_loss {epoch_avg:.4f}  last_loss {last_loss:.4f}  "
              f"steps {total_steps}  arch {net.n_nodes_per_layer()}")
    return opt


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def evaluate_all_tasks(
    net: TrioronNetwork,
    eval_views: Sequence[TaskDataView],
    task_class_lists: Sequence[Sequence[int]],
    packnet_ctrl: Optional[PackNetController] = None,
    hat_ctrl: Optional[HATController] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """Evaluate every task with THREE metrics in one pass.

    Returns (full_softmax, domain_aware, task_aware) — all lists in
    eval_views order. Three concentric restrictions:

    full_softmax (30 classes):
        Argmax over the entire head. Canonical single-head
        class-incremental CL metric. Punishes argmax-bias toward
        whichever task was most recently trained.
    domain_aware (10 classes — MNIST | Fashion | EMNIST):
        Argmax restricted to the 10-class dataset group containing
        the test sample. Realistic for device-conscience deployment
        where context routing knows the modality (digit / clothing /
        letter) but not which specific binary task. The honest middle
        ground.
    task_aware (2 classes — the active binary pair):
        Argmax restricted to that task's active classes only. The
        easiest metric — caller must know exactly which binary task
        a sample belongs to. Useful for "what does the model
        fundamentally know?" diagnostic, less for deployment.

    All three measure the SAME forward pass, just with different
    argmax-restriction rules. Headline retains full_softmax (CL
    convention). domain_aware and task_aware are side panels.
    """
    full_accs: List[float] = []
    aware_accs: List[float] = []
    domain_accs: List[float] = []
    # PackNet / HAT inference protocols share the same dual approach:
    #   task-aware  → apply that task's mask (uses task-ID at inference)
    #   full/domain → apply the most-recent-tasks_done mask (PackNet
    #                  union mask) or the cumulative hooks (HAT)
    # Both produce different forwards, so we run two passes per view.
    use_packnet = (
        packnet_ctrl is not None and packnet_ctrl.tasks_done > 0
    )
    use_hat = (
        hat_ctrl is not None and hat_ctrl.tasks_done > 0
    )
    with torch.no_grad():
        for i, v in enumerate(eval_views):
            x, y = v.all_examples()

            if use_packnet:
                # task-aware: this view's own mask. Skip if this view's
                # task hasn't been trained yet (i+1 > tasks_done).
                eval_task_id = i + 1
                if eval_task_id <= packnet_ctrl.tasks_done:
                    snap = packnet_ctrl.apply_inference_mask(eval_task_id)
                    logits_aware = net(x)
                    packnet_ctrl.restore(snap)
                else:
                    logits_aware = None
                # full / domain: union of all tasks-done masks.
                snap = packnet_ctrl.apply_inference_mask(
                    packnet_ctrl.tasks_done,
                )
                logits = net(x)
                packnet_ctrl.restore(snap)
            elif use_hat:
                # task-aware: install hooks for this view's task.
                eval_task_id = i + 1
                if eval_task_id <= hat_ctrl.tasks_done:
                    snap = hat_ctrl.apply_inference_mask(eval_task_id)
                    logits_aware = net(x)
                    hat_ctrl.restore(snap)
                else:
                    logits_aware = None
                # full / domain: HAT has no native union-mask. Use the
                # most-recent task's mask as the "task-blind" eval —
                # standard treatment for HAT in class-incremental
                # benches (acknowledged weakness vs methods that can
                # operate without task ID).
                snap = hat_ctrl.apply_inference_mask(hat_ctrl.tasks_done)
                logits = net(x)
                hat_ctrl.restore(snap)
            else:
                logits = net(x)
                logits_aware = logits

            head_size = logits.shape[1]
            full_accs.append(accuracy(logits, y))

            # Task-aware: restrict to the binary pair.
            active = task_class_lists[i]
            if logits_aware is not None and max(active) < head_size:
                aware_accs.append(accuracy(
                    logits_aware, y, restrict_to=active,
                ))
            else:
                aware_accs.append(float("nan"))

            # Domain-aware: restrict to the 10-class dataset group.
            # Chained-15 layout: MNIST=0..9, Fashion=10..19, EMNIST=20..29.
            domain_idx = active[0] // 10
            domain_classes_full = list(
                range(domain_idx * 10, (domain_idx + 1) * 10)
            )
            # Filter to classes the head currently has — early in
            # the curriculum the head hasn't fully extended yet, so
            # restrict to only the classes that exist.
            domain_classes_avail = [c for c in domain_classes_full
                                    if c < head_size]
            if domain_classes_avail:
                domain_accs.append(accuracy(
                    logits, y, restrict_to=domain_classes_avail,
                ))
            else:
                domain_accs.append(float("nan"))
    return full_accs, aware_accs, domain_accs


# ---------------------------------------------------------------------
# Quantization simulation (Phase 2 of dream-archive stage)
# ---------------------------------------------------------------------


def _snapshot_layer_weights(net: TrioronNetwork) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Clone every layer's W and b for later restoration."""
    return [(layer.W.detach().clone(), layer.b.detach().clone())
            for layer in net.layers]


def _restore_layer_weights(
    net: TrioronNetwork,
    snap: List[Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    with torch.no_grad():
        for layer, (W, b) in zip(net.layers, snap):
            layer.W.data.copy_(W)
            layer.b.data.copy_(b)


def _quantize_archived_in_place(
    net: TrioronNetwork, *, mode: str = "ternary",
) -> None:
    """Snap archived rows' W (and b) to a quantized representation,
    de-quantized back to FP32 for the forward pass. Per-row symmetric
    scale. Non-archived rows are untouched.

    mode='ternary': W_q ∈ {-s, 0, +s} per row, threshold |W| ≥ s/2.
    mode='int8'   : W_q = round(W/s).clamp(-127, 127) * s, s = max|W|/127.
    """
    with torch.no_grad():
        for layer in net.layers:
            arch_mask = layer.archived
            if not arch_mask.any():
                continue
            idxs = arch_mask.nonzero(as_tuple=False).flatten().tolist()
            for i in idxs:
                row = layer.W.data[i]
                if mode == "ternary":
                    s = float(row.abs().max().item())
                    if s == 0.0:
                        continue
                    trit = torch.zeros_like(row)
                    trit[row >= 0.5 * s] = s
                    trit[row <= -0.5 * s] = -s
                    layer.W.data[i] = trit
                elif mode == "int8":
                    s = float(row.abs().max().item()) / 127.0
                    if s == 0.0:
                        continue
                    q = torch.clamp(torch.round(row / s), -127, 127)
                    layer.W.data[i] = q * s
                else:
                    raise ValueError(f"unknown quantize mode {mode!r}")


def _storage_breakdown(
    net: TrioronNetwork,
    *,
    mode: str = "ternary",
    baseline_dtype: str = "fp32",
) -> Dict[str, float]:
    """Per-layer byte accounting for the deployment image.

    baseline_dtype  active (non-archived) weights + biases:
        "fp32"      4 bytes/weight, 4 bytes/bias.
        "bf16"      2 bytes/weight, 2 bytes/bias. Realistic device target
                    (Orange Pi 5B / ESP32 deployment baseline).

    mode (archived rows):
        "ternary"   2 bits/weight + 4 B per-row scale.
        "int8"      1 byte/weight + 4 B per-row scale.
    """
    base_w_bytes = {"fp32": 4, "bf16": 2}[baseline_dtype]
    base_b_bytes = base_w_bytes
    arch_w_bytes = {"ternary": 0.25, "int8": 1.0}[mode]
    per_row_overhead = 4
    breakdown: Dict[str, float] = {}
    total = 0.0
    for L, layer in enumerate(net.layers):
        n_arch = int(layer.archived.sum().item())
        n_active = layer.n_nodes - n_arch
        fan_in = layer.fan_in
        bytes_W = (
            n_active * fan_in * base_w_bytes
            + n_arch * (fan_in * arch_w_bytes + per_row_overhead)
        )
        bytes_b = layer.n_nodes * base_b_bytes
        layer_bytes = bytes_W + bytes_b
        if L == 0:
            breakdown["L0_kb"] = layer_bytes / 1024.0
        elif L == len(net.layers) - 1:
            breakdown["head_kb"] = layer_bytes / 1024.0
        else:
            key = f"L{L}_kb" if L > 1 else "L1_kb"
            breakdown[key] = layer_bytes / 1024.0
        total += layer_bytes
    breakdown["total_kb"] = total / 1024.0
    breakdown.setdefault("L1_kb", 0.0)
    return breakdown


def _round_active_to_bf16_in_place(net: TrioronNetwork) -> None:
    """Round non-archived rows of W and ALL biases to BF16 precision
    then cast back to FP32 storage. Values lose BF16-mantissa bits but
    tensors stay FP32, so F.linear / forward / EWC continue to work on
    matched dtypes. Archived rows are untouched (already int8-snapped).

    Use only inside a snapshot/restore frame — this mutates W.data and
    b.data in place.
    """
    with torch.no_grad():
        for layer in net.layers:
            arch = layer.archived
            if (~arch).any():
                idx = (~arch).nonzero(as_tuple=False).flatten().tolist()
                rows = layer.W.data[idx]
                layer.W.data[idx] = rows.to(torch.bfloat16).to(torch.float32)
            layer.b.data = layer.b.data.to(torch.bfloat16).to(torch.float32)


# ---------------------------------------------------------------------
# Whole-curriculum runner
# ---------------------------------------------------------------------


def run_chained_curriculum(
    net: TrioronNetwork,
    label: str,
    *,
    do_growth: bool,
    do_dream: bool,
    cap_bytes: int,
    n_grow_per_task: int,
    train_views: Sequence[TaskDataView],
    eval_views: Sequence[TaskDataView],
    task_class_lists: Sequence[Sequence[int]],
    n_epochs_per_task: int,
    rng_seed: int,
    n_passes: int = 1,
    packnet_mode: Optional[str] = None,
    hat_mode: Optional[str] = None,
    online_ewc_gamma: Optional[float] = None,
    extension_train_views: Optional[Sequence[TaskDataView]] = None,
    extension_eval_views: Optional[Sequence[TaskDataView]] = None,
    extension_task_class_lists: Optional[Sequence[Sequence[int]]] = None,
    extension_cap_bytes: Optional[int] = None,
    extension_permanent_int8: bool = False,
    return_state: bool = False,
) -> Dict[str, object]:
    """Run the chained curriculum, optionally repeated for `n_passes`.

    On pass > 0 (revisit), `n_grow_per_task` is forced to 0: no new
    neurogenesis on revisit, only consolidation through retraining +
    dreaming. EWC anchors carry forward across passes.

    Per-task training is `n_epochs_per_task` proper minibatch epochs
    (each sample seen exactly once per epoch).

    Extension phase (chained-15 → chained-N): if `extension_train_views`
    is provided, after the main `train_views` loop completes the bench
    (a) fires `consolidation_dream_pass` over all past tasks (full-
    coverage replay + archive-aware grad masking + final archive_block),
    (b) optionally snaps archived rows permanently to int8 if
    `extension_permanent_int8` is set, (c) lifts the cap to
    `extension_cap_bytes`, and (d) iterates the extension tasks with the
    same train/consolidate/archive/dream cycle, sharing all internal
    state (manifold buffer, EWC anchors, archived rows, etc.). The
    boundary fires only on pass 0 (revisit passes don't re-trigger).
    """
    # Combine main + extension into a single sequence so the per-task
    # loop is unchanged. The extension boundary fires inline at
    # local_task_idx == K_main on pass 0.
    K_main = len(train_views)
    has_extension = extension_train_views is not None and len(extension_train_views) > 0
    if has_extension:
        train_views = list(train_views) + list(extension_train_views)
        eval_views = list(eval_views) + list(extension_eval_views or [])
        task_class_lists = (
            list(task_class_lists) + list(extension_task_class_lists or [])
        )
        if extension_cap_bytes is None:
            raise ValueError(
                "extension_train_views requires extension_cap_bytes"
            )
    K = len(train_views)
    boundary_idx = K_main if has_extension else None
    current_cap_bytes = cap_bytes
    n_total = K * n_passes
    initial_n_params = net.n_parameters()
    initial_trainable = trainable_params(net)
    initial_arch = tuple(net.n_nodes_per_layer())
    rng = random.Random(rng_seed)
    # Path 2 rehearsal buffer. Built once per arm; persists across passes.
    memory: Optional[MemoryBuffer] = None
    if REHEARSAL_ENABLED:
        memory = MemoryBuffer(samples_per_task=REHEARSAL_SAMPLES_PER_TASK)
    # B — Brainstem-Spark per-class latent stats at L1 output.
    brainstem: Optional[BrainstemBuffer] = None
    if BRAINSTEM_ENABLED:
        brainstem = BrainstemBuffer()
    # Engram Replay — input-space prototypes per class, found by
    # gradient ascent through the anchored network at consolidation
    # time. Replaces / augments brainstem when ENGRAM_ENABLED.
    engrams: Optional[EngramBuffer] = None
    if ENGRAM_ENABLED:
        engrams = EngramBuffer()
    # Hippocampal Replay — K real-sample L0 outputs per class.
    # Pre-conditions: this arm has a frozen L0 (so stored codes don't
    # go stale across tasks). For trainable-L0 arms, fall back to
    # MemoryBuffer (raw rehearsal) instead.
    hippocampus: Optional[HippocampalBuffer] = None
    arm_l0_frozen = not bool(net.layers[0].W.requires_grad)
    if HIPPOCAMPAL_ENABLED and arm_l0_frozen:
        hippocampus = HippocampalBuffer()
    # Differential Replay — multi-layer task differentials vs blank.
    # Same frozen-L0 precondition as hippocampal (δL0 stays valid).
    differential: Optional[DifferentialReplayBuffer] = None
    if DIFFERENTIAL_ENABLED and arm_l0_frozen:
        differential = DifferentialReplayBuffer()
    # Manifold Replay — per-class L0 diagonal Gaussian, sampled on demand.
    # Same frozen-L0 precondition; trainable-L0 arms would have stale stats.
    manifold: Optional[ManifoldBuffer] = None
    if MANIFOLD_REPLAY_ENABLED and arm_l0_frozen:
        manifold = ManifoldBuffer()

    # PackNet baseline — per-task disjoint subnets via magnitude pruning.
    # In 'matched' mode, L0 is treated as a shared frozen feature
    # extractor (skipped from PackNet's partition pool). In 'standard'
    # mode, the entire network (including L0) is partitioned across
    # tasks. PackNet uses task-ID at task-aware inference time.
    packnet_ctrl: Optional[PackNetController] = None
    if packnet_mode is not None:
        skip_ids: List[int] = [0] if packnet_mode == "matched" else []
        packnet_ctrl = PackNetController(
            net, n_total_tasks=K, frozen_layer_ids=skip_ids,
        )

    # HAT baseline — sigmoid-attention masks per task, gradient surgery
    # to protect prior-task weights. HAT's task embeddings are trainable
    # parameters that must be in the optimizer alongside net params.
    hat_ctrl: Optional[HATController] = None
    if hat_mode is not None:
        hat_ctrl = HATController(net, n_total_tasks=K)
        # Move HAT to the network's device so its embeddings live where
        # gradients flow.
        hat_ctrl = hat_ctrl.to(net.layers[0].W.device)
    print(f"\n[{label}] start — arch {initial_arch}  "
          f"params {initial_n_params} (trainable {initial_trainable})  "
          f"growth={do_growth} dream={do_dream}  "
          f"cap_bytes={cap_bytes}  K={K}  passes={n_passes}  "
          f"epochs/task={n_epochs_per_task}  "
          f"rehearsal={'on' if REHEARSAL_ENABLED else 'off'}  "
          f"lwf={'on' if LWF_ENABLED else 'off'}  "
          f"brainstem={'on' if BRAINSTEM_ENABLED else 'off'}  "
          f"engrams={'on' if ENGRAM_ENABLED else 'off'}  "
          f"hippocampus={'on' if hippocampus is not None else 'off'}  "
          f"differential={'on' if differential is not None else 'off'}  "
          f"manifold={'on' if manifold is not None else 'off'}")

    opt = _build_optimizer(net, hat_ctrl)
    # Accuracy matrix shape: (n_total, K). Row i = state after the i-th
    # task encounter; col j = accuracy on eval task j. Each pass adds
    # K rows. Final headline = last row.
    # Two matrices in parallel: full-softmax (canonical headline) and
    # task-aware (diagnostic — argmax over each task's active classes
    # only, ignoring head-column drift on inactive classes).
    accuracy_matrix: List[List[float]] = [[float("nan")] * K for _ in range(n_total)]
    accuracy_matrix_aware: List[List[float]] = [[float("nan")] * K for _ in range(n_total)]
    accuracy_matrix_domain: List[List[float]] = [[float("nan")] * K for _ in range(n_total)]
    per_task_log: List[Dict[str, object]] = []
    n_params_per_task: List[int] = []
    cumulative_grows = 0
    cumulative_grows_denied = 0
    cumulative_purges = 0
    cumulative_latched = 0
    cumulative_archives = 0
    ewc_baseline = 0.0
    pass_summary: List[Dict[str, float]] = []  # one entry per pass

    t0 = time.monotonic()
    for pass_idx in range(n_passes):
        # Pass 0 = developmental (growth on); pass >0 = consolidation
        # (no neurogenesis on revisit, just retraining + dreaming).
        pass_grows_allowed = (n_grow_per_task if pass_idx == 0 else 0)
        print(f"\n[{label}] >>> PASS {pass_idx+1}/{n_passes}  "
              f"grows_per_task={pass_grows_allowed}  "
              f"(developmental)" if pass_idx == 0
              else f"[{label}] >>> PASS {pass_idx+1}/{n_passes}  "
                   f"grows_per_task={pass_grows_allowed}  (consolidation)")

        for local_task_idx, train_view in enumerate(train_views):
            active = list(task_class_lists[local_task_idx])
            global_step_idx = pass_idx * K + local_task_idx

            # === Extension boundary: fires once on pass 0 right before
            # the first extension task. Runs a heavier "shipping
            # consolidation" dream over all main-curriculum tasks,
            # optionally snaps archived rows permanently to int8, then
            # lifts the cap to extension_cap_bytes for the rest of the
            # curriculum.
            if (boundary_idx is not None and pass_idx == 0
                    and local_task_idx == boundary_idx):
                print(f"\n[{label}] ============================================")
                print(f"[{label}] EXTENSION BOUNDARY at task "
                      f"{local_task_idx+1}/{K}: shipping consolidation")
                print(f"[{label}] ============================================")
                past_views = train_views[:local_task_idx]
                past_actives = task_class_lists[:local_task_idx]
                cd_rep = consolidation_dream_pass(
                    net, past_views, past_actives, rng=rng,
                )
                print(f"[{label}] consolidation dream: replay "
                      f"{cd_rep['replay_loss_before']:.4f}→"
                      f"{cd_rep['replay_loss_after']:.4f} on "
                      f"{cd_rep['n_replay_tasks']} tasks "
                      f"({cd_rep['n_replay_steps']} steps)  "
                      f"+{cd_rep['n_archived_now']} new archives  "
                      f"per-layer-archived {cd_rep['n_archived_per_layer']}")
                opt = _build_optimizer(net, hat_ctrl)
                if extension_permanent_int8 and ARCHIVE_ENABLED:
                    _quantize_archived_in_place(net, mode="int8")
                    print(f"[{label}] permanent int8 snap on archived rows")
                current_cap_bytes = extension_cap_bytes
                print(f"[{label}] cap lifted: {cap_bytes:_} B → "
                      f"{current_cap_bytes:_} B "
                      f"({current_cap_bytes//4:_} trainable params)")
                print(f"[{label}] ============================================\n")

            # Old classes (everything seen before THIS task in any pass)
            # — used for LwF distillation masking.
            old_classes_seen: List[int] = []
            for k in range(local_task_idx):
                old_classes_seen.extend(task_class_lists[k])
            if pass_idx > 0:
                # On revisit passes, every class has been seen; include
                # all of them (even the ones from later in this pass).
                for k in range(K):
                    old_classes_seen.extend(task_class_lists[k])
            old_classes_seen = sorted(set(old_classes_seen))
            lwf_old = old_classes_seen if LWF_ENABLED else None
            engram_old = old_classes_seen if ENGRAM_ENABLED else None

            # 1. Extend output head to fit this task's classes (idempotent
            #    on revisit — the head already covers earlier tasks).
            #    Head extension is mandatory before any training: the
            #    network can't compute logits for unseen classes
            #    otherwise. Hidden-layer growth (Fix B) IS gated.
            head_size = net.layers[-1].n_nodes
            max_active = max(active)
            if max_active >= head_size:
                n_new_head = max_active - head_size + 1
                extend_output_head(net, n_new_head)
                opt = _build_optimizer(net, hat_ctrl)

            # PackNet begin_task: re-init free weights for the new task
            # (frozen weights from prior tasks preserved). Adam state is
            # stale across the re-init, so rebuild the optimizer.
            if packnet_ctrl is not None and pass_idx == 0:
                packnet_ctrl.begin_task(local_task_idx + 1)
                opt = _build_optimizer(net, hat_ctrl)

            # HAT begin_task: zero active embeddings, install hooks,
            # set temperature to s_min (will anneal up to s_max during
            # training). Rebuild optimizer so Adam moments for the
            # newly-zeroed embeddings start clean.
            if hat_ctrl is not None and pass_idx == 0:
                hat_ctrl.begin_task(local_task_idx + 1)
                opt = _build_optimizer(net, hat_ctrl)

            # 2. Print task header (pre-settle). Fix B's growth comes
            #    after a settle phase; arch shown here is pre-growth.
            print(f"\n[{label}] === Pass {pass_idx+1}/{n_passes} "
                  f"Task {local_task_idx+1}/{K}: {train_view.name} "
                  f"(active {active})  pre-arch={net.n_nodes_per_layer()} "
                  f"params={net.n_parameters()} "
                  f"(trainable {trainable_params(net)}/{current_cap_bytes//4 if current_cap_bytes < M_MAX_BYTES_UNCAPPED else '∞'}) ===")

            # 3a. Settle phase — train K_SETTLE epochs BEFORE any
            #     hidden-layer growth fires. Per Gemma's framing: the
            #     network needs stable input shape before any structural
            #     plasticity decision is meaningful. On revisit passes
            #     (or when growth is disabled) we collapse settle and
            #     post-grow into one block — no need to split when no
            #     growth is going to happen mid-task.
            grows_this_task = (
                pass_grows_allowed if (do_growth and pass_grows_allowed > 0)
                else 0
            )
            split_training = grows_this_task > 0 and K_SETTLE_EPOCHS > 0
            if split_training:
                settle_epochs = min(K_SETTLE_EPOCHS, n_epochs_per_task)
                opt = train_one_task(
                    net, local_task_idx, train_view, active,
                    n_epochs=settle_epochs, opt=opt,
                    ewc_baseline=ewc_baseline,
                    label=label, n_total_tasks=K,
                    epoch_offset=0,
                    total_epochs_outer=n_epochs_per_task,
                    epoch_label_suffix=" [settle]",
                    memory=memory,
                    lwf_old_classes=lwf_old,
                    brainstem=brainstem,
                    engrams=engrams,
                    engram_old_classes=engram_old,
                    hippocampus=hippocampus,
                    differential=differential,
                    manifold=manifold,
                    packnet_ctrl=packnet_ctrl,
                    hat_ctrl=hat_ctrl,
                )

            # 3b. Deterministic hidden growth (with optional dream-rescue).
            #     On revisit passes, grows_this_task=0 so this whole
            #     block skips. After settle, the network has data-supported
            #     evidence of representation; growth that fires here can
            #     actually be informed by current activity.
            attempted = 0
            allowed = 0
            denied = 0
            if grows_this_task > 0:
                for _ in range(grows_this_task):
                    attempted += 1
                    ok, reason = try_grow_one(
                        net, GROWTH_TARGET_LAYER_IDX, current_cap_bytes,
                        local_task_idx,
                    )
                    if ok:
                        allowed += 1
                    else:
                        denied += 1
                        if do_dream:
                            # Dream-rescue: try to free room then retry once.
                            past_views = train_views[:local_task_idx]
                            past_actives = task_class_lists[:local_task_idx]
                            rescue = classification_dreaming_block(
                                net, past_views, past_actives,
                                rng=rng, mode="reclaim",
                            )
                            cumulative_purges += rescue["n_purges"]
                            cumulative_latched += rescue["n_latched"]
                            # Re-anchor routing_scale on every layer so
                            # feature-distillation losses (engram L1-MSE,
                            # diff δL1) compare live and anchor at
                            # consistent routing instead of fighting the
                            # purge mutation.
                            if (REANCHOR_AFTER_PURGE
                                    and (rescue["n_purges"] > 0
                                         or rescue["n_latched"] > 0)):
                                net.reanchor_routing_only()
                            opt = _build_optimizer(net, hat_ctrl)
                            ok2, reason2 = try_grow_one(
                                net, GROWTH_TARGET_LAYER_IDX,
                                current_cap_bytes, local_task_idx,
                            )
                            if ok2:
                                allowed += 1
                                denied -= 1   # not actually denied
                                print(f"  [{label}] dream-rescue freed room: "
                                      f"purges={rescue['n_purges']} → grow OK")
                            else:
                                print(f"  [{label}] dream-rescue insufficient: "
                                      f"purges={rescue['n_purges']}; growth still denied "
                                      f"({reason2})")
                                break  # no point trying more grows this task
                        else:
                            # No dreaming → cap binds, accept partial growth.
                            break
            cumulative_grows += allowed
            cumulative_grows_denied += denied

            if allowed > 0:
                opt = _build_optimizer(net, hat_ctrl)
                print(f"  [{label}] GROWTH after settle: "
                      f"{allowed}/{attempted} allowed, {denied} denied  "
                      f"new arch={net.n_nodes_per_layer()} "
                      f"params={net.n_parameters()} "
                      f"(trainable {trainable_params(net)})")

            # 3c. Post-growth training — remaining epochs at the (now
            #     possibly larger) architecture. If we didn't split,
            #     this is the entire training pass for the task.
            if split_training:
                remaining = n_epochs_per_task - K_SETTLE_EPOCHS
                if remaining > 0:
                    opt = train_one_task(
                        net, local_task_idx, train_view, active,
                        n_epochs=remaining, opt=opt,
                        ewc_baseline=ewc_baseline,
                        label=label, n_total_tasks=K,
                        epoch_offset=K_SETTLE_EPOCHS,
                        total_epochs_outer=n_epochs_per_task,
                        epoch_label_suffix=" [post-grow]",
                        memory=memory,
                        lwf_old_classes=lwf_old,
                        brainstem=brainstem,
                        engrams=engrams,
                        engram_old_classes=engram_old,
                        hippocampus=hippocampus,
                        differential=differential,
                        manifold=manifold,
                        packnet_ctrl=packnet_ctrl,
                        hat_ctrl=hat_ctrl,
                    )
            else:
                opt = train_one_task(
                    net, local_task_idx, train_view, active,
                    n_epochs=n_epochs_per_task, opt=opt,
                    ewc_baseline=ewc_baseline,
                    label=label, n_total_tasks=K,
                    memory=memory,
                    lwf_old_classes=lwf_old,
                    brainstem=brainstem,
                    engrams=engrams,
                    engram_old_classes=engram_old,
                    hippocampus=hippocampus,
                    differential=differential,
                    manifold=manifold,
                    packnet_ctrl=packnet_ctrl,
                    hat_ctrl=hat_ctrl,
                )

            # 4. Consolidate. Online EWC accumulates Fisher across
            #    tasks with decay γ; per-task EWC resets each call.
            consolidate_task(
                net, train_view, active,
                online_ewc_gamma=online_ewc_gamma,
            )
            # 4a. Archive (Phase 1). After consolidate (so λ + Fisher
            #     are fresh) mark stable rows as developmentally closed.
            #     Skipped on PackNet/HAT arms (their per-task masks
            #     already partition substrate; double-locking conflicts).
            archived_now: List[Tuple[int, int]] = []
            if (ARCHIVE_ENABLED and packnet_ctrl is None
                    and hat_ctrl is None):
                archived_now = archive_block(
                    net,
                    streak_threshold=ARCHIVE_STREAK_THRESHOLD,
                    lam_top_percentile=ARCHIVE_LAM_TOP_PERCENTILE,
                    grad_mag_floor=ARCHIVE_GRAD_MAG_FLOOR,
                    pulse_max=ARCHIVE_PULSE_MAX,
                    skip_output_layer=ARCHIVE_SKIP_OUTPUT_LAYER,
                    max_archives_per_layer=ARCHIVE_MAX_PER_LAYER,
                )
                cumulative_archives += len(archived_now)
                if archived_now:
                    by_layer: Dict[int, int] = {}
                    for L, _idx in archived_now:
                        by_layer[L] = by_layer.get(L, 0) + 1
                    layer_summary = " ".join(
                        f"L{L}:+{n}" for L, n in sorted(by_layer.items())
                    )
                    print(f"  [{label}] archive: {len(archived_now)} rows "
                          f"({layer_summary})  per-layer-archived "
                          f"{net.n_archived_per_layer()}")
            # PackNet / HAT replace EWC anchor-pull with their own
            # protection mechanisms. Disable ewc_baseline for the next
            # task so we don't double-supervise.
            if packnet_ctrl is None and hat_ctrl is None:
                ewc_baseline = EWC_INTERTASK
            # PackNet end_task: claim 1/(n-t+1) of currently-free
            # weights as task t's mask, zero the rest. Must fire AFTER
            # consolidate_task and BEFORE the eval at the bottom of
            # the task loop (eval applies per-task masks).
            if packnet_ctrl is not None:
                packnet_ctrl.end_task(local_task_idx + 1)
            # HAT end_task: snapshot current active embeddings as
            # task t's saved mask, update cumulative mask, remove
            # forward hooks. Must fire BEFORE eval (apply_inference_mask
            # reads task_embeddings).
            if hat_ctrl is not None:
                hat_ctrl.end_task(local_task_idx + 1)

            # 4b. Add a random subset of this task's training samples
            #     to the rehearsal memory buffer (Path 2). Skipped on
            #     revisit passes — re-adding the same task's samples
            #     would just bias the buffer. Pass-1 only.
            if memory is not None and pass_idx == 0:
                x_pool, y_pool = train_view.all_examples()
                memory.add_task(x_pool, y_pool)

            # 4c. Brainstem-Spark: compute per-class (μ, σ) at L1
            #     output on this task's training data and store. Done
            #     AFTER consolidate so the L1 reflects the consolidated
            #     state. Pass-1 only.
            if brainstem is not None and pass_idx == 0:
                _store_brainstem_stats(net, train_view, active, brainstem)

            # 4d. Engram Replay consolidation: for each just-learned
            #     class, run gradient ascent on the input through the
            #     anchored network to find a per-class prototype x_c.
            #     Triparametric: forward_with_anchors uses W_anchor +
            #     b_anchor + routing_scale_anchor, so the engram is the
            #     consolidated network's idealized input across all
            #     three legs of the trioron node. Done AFTER consolidate
            #     so anchors reflect the just-finished task. Pass-1 only.
            if engrams is not None and pass_idx == 0:
                _consolidate_engrams(
                    net, active, engrams, train_view=train_view,
                )

            # 4e. Hippocampal consolidation: sample K real examples per
            #     class, forward through (frozen) L0, store the
            #     compressed codes. Pass-1 only — buffer persists across
            #     passes. Storage = K * L0_width per class.
            #     If HIPPOCAMPAL_SYNTHETIC, replace real-sample encoding
            #     with logit-driven inversion against the just-
            #     consolidated network. Storage shape unchanged.
            if hippocampus is not None and pass_idx == 0:
                if HIPPOCAMPAL_SYNTHETIC:
                    # Refresh-all: re-synthesize codes for every class
                    # seen so far against the current (post-growth) head,
                    # not just the just-learned classes. Fixes the
                    # cross-task-discrimination collapse seen in the
                    # one-shot variant: codes inverted against an old
                    # narrow head don't separate from later-task classes.
                    # Cost grows O(C·T_inv) per boundary; storage unchanged.
                    seen_so_far = sorted(set(old_classes_seen + active))
                    _synthesize_hippocampal_codes(
                        net, seen_so_far, seen_so_far, hippocampus,
                        K=HIPPOCAMPAL_K_PER_CLASS,
                    )
                else:
                    _store_hippocampal_codes(
                        net, train_view, active, hippocampus,
                        K=HIPPOCAMPAL_K_PER_CLASS,
                    )

            # 4f. Differential Replay consolidation: capture per-layer
            #     activation differentials (blank vs canonical class
            #     example) at L0, L1, head. Triple stored per class.
            if differential is not None and pass_idx == 0:
                _store_differential_codes(
                    net, train_view, active, differential,
                )

            # 4g. Manifold Replay consolidation: per-class diagonal
            #     Gaussian fit at L0 output. Storage = O(C·d_L0); codes
            #     are sampled on demand at replay time (no per-sample
            #     storage). Trioron-native pseudo-rehearsal.
            if manifold is not None and pass_idx == 0:
                _store_manifold_stats(
                    net, train_view, active, manifold,
                )

            # 5. Post-task dreaming = REPLAY ONLY (keeps memories warm;
            #    does NOT touch substrate). Structural reclamation is
            #    reserved for the on-deny dream-rescue above. On revisit
            #    passes this still runs (it's the consolidation work).
            dream_rep = {"n_merges": 0, "n_purges": 0, "n_latched": 0,
                         "n_params_before": net.n_parameters(),
                         "n_params_after": net.n_parameters(),
                         "replay_loss_before": 0.0, "replay_loss_after": 0.0,
                         "n_replay_tasks": 0}
            if do_dream:
                # Past = all tasks ENCOUNTERED so far this pass plus the
                # entire prior pass(es).
                past_local_idx = local_task_idx
                past_views = train_views[: past_local_idx + 1]
                past_actives = task_class_lists[: past_local_idx + 1]
                dream_rep = classification_dreaming_block(
                    net, past_views, past_actives,
                    rng=rng, mode="replay_only",
                )
                opt = _build_optimizer(net, hat_ctrl)
                print(f"  [{label}] post-task DREAM: replay "
                      f"{dream_rep['replay_loss_before']:.4f}→"
                      f"{dream_rep['replay_loss_after']:.4f} on "
                      f"{dream_rep['n_replay_tasks']}p; "
                      f"merges={dream_rep['n_merges']} purges={dream_rep['n_purges']} "
                      f"latched={dream_rep['n_latched']} → "
                      f"arch {net.n_nodes_per_layer()} "
                      f"({dream_rep['n_params_before']}→{dream_rep['n_params_after']} params)")

            # 6. Eval ALL tasks (the bench measures full-stream
            #    accuracy on revisit passes, since by pass 2 every
            #    task has been "seen" in the prior pass).
            per_task_acc, per_task_acc_aware, per_task_acc_domain = (
                evaluate_all_tasks(
                    net, eval_views, task_class_lists,
                    packnet_ctrl=packnet_ctrl,
                    hat_ctrl=hat_ctrl,
                )
            )
            row = global_step_idx
            for j in range(K):
                if pass_idx > 0 or j <= local_task_idx:
                    accuracy_matrix[row][j] = per_task_acc[j]
                    accuracy_matrix_aware[row][j] = per_task_acc_aware[j]
                    accuracy_matrix_domain[row][j] = per_task_acc_domain[j]
                else:
                    accuracy_matrix[row][j] = float("nan")
                    accuracy_matrix_aware[row][j] = float("nan")
                    accuracy_matrix_domain[row][j] = float("nan")
            seen_count = (
                K if pass_idx > 0 else local_task_idx + 1
            )
            avg_so_far = sum(
                v for v in accuracy_matrix[row] if v == v
            ) / seen_count
            avg_so_far_aware = sum(
                v for v in accuracy_matrix_aware[row] if v == v
            ) / seen_count
            avg_so_far_domain = sum(
                v for v in accuracy_matrix_domain[row] if v == v
            ) / seen_count
            n_params_per_task.append(net.n_parameters())
            per_task_log.append({
                "pass_idx": pass_idx,
                "task_idx": local_task_idx,
                "task_name": train_view.name,
                "active_classes": active,
                "n_params_after": net.n_parameters(),
                "n_trainable_after": trainable_params(net),
                "arch_after": tuple(net.n_nodes_per_layer()),
                "grows_allowed": allowed,
                "grows_denied": denied,
                "dream_merges": dream_rep["n_merges"],
                "dream_purges": dream_rep["n_purges"],
                "dream_latched": dream_rep["n_latched"],
                "own_acc": per_task_acc[local_task_idx],
                "own_acc_aware": per_task_acc_aware[local_task_idx],
                "own_acc_domain": per_task_acc_domain[local_task_idx],
                "avg_to_date": avg_so_far,
                "avg_to_date_aware": avg_so_far_aware,
                "avg_to_date_domain": avg_so_far_domain,
            })
            print(f"[{label}] After pass {pass_idx+1} task {local_task_idx+1}: "
                  f"own={per_task_acc[local_task_idx]:.4f} "
                  f"(domain {per_task_acc_domain[local_task_idx]:.4f}, "
                  f"task {per_task_acc_aware[local_task_idx]:.4f})  "
                  f"avg={avg_so_far:.4f} "
                  f"(domain {avg_so_far_domain:.4f}, task {avg_so_far_aware:.4f})  "
                  f"arch={net.n_nodes_per_layer()} "
                  f"params={net.n_parameters()} (trainable {trainable_params(net)})  "
                  f"cum_grows={cumulative_grows} "
                  f"cum_denied={cumulative_grows_denied} cum_purges={cumulative_purges} "
                  f"cum_latched={cumulative_latched}")

        # End of pass — record per-pass headline + aware + domain acc
        # (mean over all K tasks in the final row of this pass).
        last_row_idx = (pass_idx + 1) * K - 1
        last_row = accuracy_matrix[last_row_idx]
        last_row_aware = accuracy_matrix_aware[last_row_idx]
        last_row_domain = accuracy_matrix_domain[last_row_idx]
        pass_final_acc = sum(v for v in last_row if v == v) / sum(
            1 for v in last_row if v == v
        )
        pass_final_acc_aware = sum(v for v in last_row_aware if v == v) / sum(
            1 for v in last_row_aware if v == v
        )
        pass_final_acc_domain = sum(v for v in last_row_domain if v == v) / sum(
            1 for v in last_row_domain if v == v
        )
        pass_summary.append({
            "pass_idx": pass_idx,
            "final_accuracy": pass_final_acc,
            "final_accuracy_aware": pass_final_acc_aware,
            "final_accuracy_domain": pass_final_acc_domain,
        })
        print(f"\n[{label}] <<< PASS {pass_idx+1}/{n_passes} done — "
              f"full={pass_final_acc:.4f}  "
              f"domain={pass_final_acc_domain:.4f}  "
              f"task={pass_final_acc_aware:.4f}")

    elapsed = time.monotonic() - t0
    # summarize expects a square K×K matrix; pass the last-pass rows so
    # final_accuracy and avg_forgetting reflect the end-of-curriculum
    # state. The full (K*n_passes)×K matrix stays in the return dict
    # for diagnostics + per-pass comparison.
    last_pass_matrix = accuracy_matrix[-K:]
    last_pass_matrix_aware = accuracy_matrix_aware[-K:]
    last_pass_matrix_domain = accuracy_matrix_domain[-K:]
    rep = summarize(last_pass_matrix, [v.name for v in eval_views])
    rep_aware = summarize(last_pass_matrix_aware, [v.name for v in eval_views])
    rep_domain = summarize(last_pass_matrix_domain, [v.name for v in eval_views])

    # Phase 2 — end-of-curriculum quantization simulation. Snap archived
    # rows from FP32 to ternary or int8 (per-row symmetric scale),
    # re-evaluate, then restore the originals.
    quantization_report: Optional[Dict[str, object]] = None
    if (QUANTIZE_ARCHIVED_AT_END and ARCHIVE_ENABLED
            and packnet_ctrl is None and hat_ctrl is None
            and any(net.n_archived_per_layer())):
        full_pre = sum(v for v in last_pass_matrix[-1] if v == v) / sum(
            1 for v in last_pass_matrix[-1] if v == v
        )
        domain_pre = sum(v for v in last_pass_matrix_domain[-1] if v == v) / sum(
            1 for v in last_pass_matrix_domain[-1] if v == v
        )
        task_pre = sum(v for v in last_pass_matrix_aware[-1] if v == v) / sum(
            1 for v in last_pass_matrix_aware[-1] if v == v
        )
        # Pass 1 — FP32 active + quantized archived (matches the
        # current paper figure's training-time baseline).
        snap = _snapshot_layer_weights(net)
        try:
            _quantize_archived_in_place(net, mode=QUANTIZE_MODE)
            per_task_full, per_task_aware, per_task_domain = (
                evaluate_all_tasks(
                    net, eval_views, task_class_lists,
                    packnet_ctrl=None, hat_ctrl=None,
                )
            )
            full_post = sum(per_task_full) / len(per_task_full)
            domain_post = sum(per_task_domain) / len(per_task_domain)
            task_post = sum(per_task_aware) / len(per_task_aware)
        finally:
            _restore_layer_weights(net, snap)
        storage_fp32 = _storage_breakdown(
            net, mode=QUANTIZE_MODE, baseline_dtype="fp32",
        )
        storage_bf16 = _storage_breakdown(
            net, mode=QUANTIZE_MODE, baseline_dtype="bf16",
        )

        # Pass 2 — BF16 active + quantized archived (realistic Orange
        # Pi 5B / ESP32 deployment image). Snap-to-BF16-round-back so
        # tensor dtype stays FP32 and F.linear keeps matched dtypes.
        snap2 = _snapshot_layer_weights(net)
        try:
            _quantize_archived_in_place(net, mode=QUANTIZE_MODE)
            _round_active_to_bf16_in_place(net)
            per_task_full2, per_task_aware2, per_task_domain2 = (
                evaluate_all_tasks(
                    net, eval_views, task_class_lists,
                    packnet_ctrl=None, hat_ctrl=None,
                )
            )
            full_post_bf16 = sum(per_task_full2) / len(per_task_full2)
            domain_post_bf16 = sum(per_task_domain2) / len(per_task_domain2)
            task_post_bf16 = sum(per_task_aware2) / len(per_task_aware2)
        finally:
            _restore_layer_weights(net, snap2)

        quantization_report = {
            "mode": QUANTIZE_MODE,
            "full_pre": full_pre, "full_post": full_post,
            "full_delta": full_post - full_pre,
            "domain_pre": domain_pre, "domain_post": domain_post,
            "domain_delta": domain_post - domain_pre,
            "task_pre": task_pre, "task_post": task_post,
            "task_delta": task_post - task_pre,
            "full_post_bf16_int8": full_post_bf16,
            "domain_post_bf16_int8": domain_post_bf16,
            "task_post_bf16_int8": task_post_bf16,
            "full_delta_bf16_int8": full_post_bf16 - full_pre,
            "domain_delta_bf16_int8": domain_post_bf16 - domain_pre,
            "task_delta_bf16_int8": task_post_bf16 - task_pre,
            "storage_breakdown": storage_fp32,
            "storage_breakdown_bf16": storage_bf16,
        }
        print(f"\n[{label}] quantization ({QUANTIZE_MODE}, FP32 active): "
              f"full {full_pre:.4f}→{full_post:.4f} "
              f"(Δ {full_post-full_pre:+.4f})  "
              f"domain {domain_pre:.4f}→{domain_post:.4f} "
              f"(Δ {domain_post-domain_pre:+.4f})  "
              f"task {task_pre:.4f}→{task_post:.4f} "
              f"(Δ {task_post-task_pre:+.4f})")
        print(f"[{label}] deployment ({QUANTIZE_MODE}, BF16 active): "
              f"full {full_pre:.4f}→{full_post_bf16:.4f} "
              f"(Δ {full_post_bf16-full_pre:+.4f})  "
              f"domain {domain_pre:.4f}→{domain_post_bf16:.4f} "
              f"(Δ {domain_post_bf16-domain_pre:+.4f})  "
              f"task {task_pre:.4f}→{task_post_bf16:.4f} "
              f"(Δ {task_post_bf16-task_pre:+.4f})")
        print(f"[{label}] storage  FP32-baseline: total "
              f"{storage_fp32['total_kb']:.1f} KB "
              f"(L0 {storage_fp32['L0_kb']:.1f}, "
              f"L1 {storage_fp32['L1_kb']:.1f}, "
              f"head {storage_fp32['head_kb']:.1f})")
        print(f"[{label}] storage  BF16-baseline: total "
              f"{storage_bf16['total_kb']:.1f} KB "
              f"(L0 {storage_bf16['L0_kb']:.1f}, "
              f"L1 {storage_bf16['L1_kb']:.1f}, "
              f"head {storage_bf16['head_kb']:.1f})")
    return {
        "label": label,
        "do_growth": do_growth,
        "do_dream": do_dream,
        "cap_bytes": cap_bytes,
        "extension_cap_bytes": (
            extension_cap_bytes if has_extension else None
        ),
        "K_main": K_main,
        "K_extension": K - K_main if has_extension else 0,
        "n_passes": n_passes,
        "initial_arch": initial_arch,
        "final_arch": tuple(net.n_nodes_per_layer()),
        "initial_n_params": initial_n_params,
        "initial_trainable": initial_trainable,
        "final_n_params": net.n_parameters(),
        "final_trainable": trainable_params(net),
        "accuracy_matrix": accuracy_matrix,
        "accuracy_matrix_aware": accuracy_matrix_aware,
        "accuracy_matrix_domain": accuracy_matrix_domain,
        "last_pass_matrix": last_pass_matrix,
        "last_pass_matrix_aware": last_pass_matrix_aware,
        "last_pass_matrix_domain": last_pass_matrix_domain,
        "final_accuracy": rep.final_accuracy,
        "final_accuracy_aware": rep_aware.final_accuracy,
        "final_accuracy_domain": rep_domain.final_accuracy,
        "avg_forgetting": rep.avg_forgetting,
        "avg_forgetting_aware": rep_aware.avg_forgetting,
        "avg_forgetting_domain": rep_domain.avg_forgetting,
        "pass_summary": pass_summary,
        "n_params_per_task": n_params_per_task,
        "cumulative_grows_allowed": cumulative_grows,
        "cumulative_grows_denied": cumulative_grows_denied,
        "cumulative_purges": cumulative_purges,
        "cumulative_latched": cumulative_latched,
        "cumulative_archives": cumulative_archives,
        "n_archived_per_layer_final": net.n_archived_per_layer(),
        "quantization_report": quantization_report,
        "per_task_log": per_task_log,
        "task_names": [v.name for v in eval_views],
        "wall_clock_seconds": elapsed,
        **({"net": net, "manifold": manifold} if return_state else {}),
    }


# ---------------------------------------------------------------------
# Arm dispatch
# ---------------------------------------------------------------------


ARM_DEFINITIONS = {
    "fixed_ewc": {
        "h_init": H_FIXED, "do_growth": False, "do_dream": False,
        "cap_bytes": M_MAX_BYTES_UNCAPPED, "freeze_l0": False,
    },
    # Matched-trainable baseline. H=56 + freeze_l0 yields
    # (128+1)*56 + (56+1)*30 = 7224 + 1710 = 8934 trainable params at
    # task 15 — identical to grown_capped_no_dream's final state. Same
    # frozen warmed-up L0 as the grown arms, so the comparison isolates
    # "growth helps" from "L0 trainable helps" or "L0 width helps."
    # Tests the matched-params claim that motivated the architecture.
    "fixed_ewc_small": {
        "h_init": 56, "do_growth": False, "do_dream": False,
        "cap_bytes": M_MAX_BYTES_UNCAPPED, "freeze_l0": True,
    },
    "grown_capped_no_dream": {
        "h_init": H_INIT_GROWN, "do_growth": True, "do_dream": False,
        "cap_bytes": M_MAX_BYTES_CAPPED, "freeze_l0": True,
    },
    "grown_capped_dream": {
        "h_init": H_INIT_GROWN, "do_growth": True, "do_dream": True,
        "cap_bytes": M_MAX_BYTES_CAPPED, "freeze_l0": True,
    },
    "grown_uncapped_dream": {
        "h_init": H_INIT_GROWN, "do_growth": True, "do_dream": True,
        "cap_bytes": M_MAX_BYTES_UNCAPPED, "freeze_l0": True,
    },
    # PackNet baselines (Mallya & Lazebnik 2018) — per-task disjoint
    # subnets carved by magnitude pruning. Reports task-aware inference
    # via apply_inference_mask(eval_task_id) for the task-aware metric;
    # full-softmax uses the union mask.
    #
    # packnet_matched: H=92 fixes total trainable to (128+1)*92 +
    # (92+1)*30 = 11868+2790 = 14658 — exactly grown_uncapped_dream's
    # final budget. L0 is frozen and shared (skipped from PackNet's
    # partition pool); PackNet allocates only L1+head, partitioned
    # 1/15 per task = ~977 trainable params per task.
    "packnet_matched": {
        "h_init": 92, "do_growth": False, "do_dream": False,
        "cap_bytes": M_MAX_BYTES_UNCAPPED, "freeze_l0": True,
        "packnet_mode": "matched",
    },
    # packnet_standard: full network (trainable L0), PackNet partitions
    # everything. Total trainable ≈ 109,414 (L0 dominates), partitioned
    # 1/15 per task ≈ 7,294 trainable params per task. Generous
    # capacity vs grown's 14,658 final — favors PackNet on raw budget,
    # disadvantages it on shared-feature reuse. Honest dual comparison.
    "packnet_standard": {
        "h_init": 56, "do_growth": False, "do_dream": False,
        "cap_bytes": M_MAX_BYTES_UNCAPPED, "freeze_l0": False,
        "packnet_mode": "standard",
    },
    # HAT baselines (Serrà et al. 2018) — task-conditional sigmoid
    # attention masks over hidden activations + gradient surgery to
    # protect prior-task weights. Like PackNet uses task-ID at task-
    # aware inference (apply_inference_mask).
    #
    # hat_matched: frozen L0 (matches grown_uncapped_dream's feature
    # condition), HAT masks gate L0+L1 outputs per task. H=92 →
    # 14,658 trainable params + HAT's task-embedding parameters
    # (small overhead).
    "hat_matched": {
        "h_init": 92, "do_growth": False, "do_dream": False,
        "cap_bytes": M_MAX_BYTES_UNCAPPED, "freeze_l0": True,
        "hat_mode": "matched",
    },
    # hat_standard: full network (trainable L0), standard HAT protocol.
    # Larger trainable budget — favors HAT in capacity, but loses the
    # feature-sharing benefit of matched.
    "hat_standard": {
        "h_init": 56, "do_growth": False, "do_dream": False,
        "cap_bytes": M_MAX_BYTES_UNCAPPED, "freeze_l0": False,
        "hat_mode": "standard",
    },
    # Online EWC (Schwarz et al. 2018) — accumulate Fisher across tasks
    # with decay γ instead of per-task Fisher reset. Cheaper memory
    # (one Fisher tensor instead of per-task), similar protection.
    # Same shape as fixed_ewc_small (matched-trainable, frozen L0).
    "online_ewc": {
        "h_init": 56, "do_growth": False, "do_dream": False,
        "cap_bytes": M_MAX_BYTES_UNCAPPED, "freeze_l0": True,
        "online_ewc_gamma": 0.95,
    },
}

DEFAULT_ARMS = list(ARM_DEFINITIONS.keys())


def run_arm(
    arm: str,
    *,
    seed: int,
    n_epochs_per_task: int,
    train_views,
    eval_views,
    task_class_lists,
    infancy_view: Optional[TaskDataView] = None,
    n_passes: int = 1,
    extension_train_views=None,
    extension_eval_views=None,
    extension_task_class_lists=None,
    extension_cap_bytes: Optional[int] = None,
    extension_permanent_int8: bool = False,
    return_state: bool = False,
) -> Dict[str, object]:
    cfg = ARM_DEFINITIONS[arm]
    torch.manual_seed(seed)
    # PackNet / HAT need a fixed network shape across the curriculum
    # (their per-layer masks are bound to layer shapes at __init__).
    # Pre-extend the head to the full class count instead of growing
    # it incrementally.
    init_classes = INIT_CLASSES
    if cfg.get("packnet_mode") is not None or cfg.get("hat_mode") is not None:
        init_classes = sum(2 for _ in range(15))  # 2 × 15 = 30
    net = make_classifier(
        INPUT_DIM, L0_WIDTH, cfg["h_init"], init_classes,
        freeze_l0=cfg["freeze_l0"],
    )

    # Frozen-L0 arms get a brief warmup before the curriculum begins.
    # The fixed_ewc baseline doesn't (its L0 is trainable; warming it
    # would just be a head-start that confounds the comparison).
    if cfg["freeze_l0"] and infancy_view is not None:
        print(f"\n[{arm}] L0 warmup ({N_WARMUP_STEPS} steps on "
              f"{infancy_view.n_examples()} infancy samples) ...")
        warmup_l0(
            net, infancy_view,
            n_steps=N_WARMUP_STEPS,
            batch=BATCH,
            lr=WARMUP_LR,
            temp_hidden=WARMUP_TEMP_HIDDEN,
            head_width=WARMUP_HEAD_WIDTH,
            seed=seed + 1009,
        )

    return run_chained_curriculum(
        net, label=arm,
        do_growth=cfg["do_growth"], do_dream=cfg["do_dream"],
        cap_bytes=cfg["cap_bytes"], n_grow_per_task=N_GROW_PER_TASK,
        train_views=train_views, eval_views=eval_views,
        task_class_lists=task_class_lists,
        n_epochs_per_task=n_epochs_per_task,
        rng_seed=seed + 7919,
        n_passes=n_passes,
        packnet_mode=cfg.get("packnet_mode"),
        hat_mode=cfg.get("hat_mode"),
        online_ewc_gamma=cfg.get("online_ewc_gamma"),
        extension_train_views=extension_train_views,
        extension_eval_views=extension_eval_views,
        extension_task_class_lists=extension_task_class_lists,
        extension_cap_bytes=extension_cap_bytes,
        extension_permanent_int8=extension_permanent_int8,
        return_state=return_state,
    )


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


def _phase_means(M: List[List[float]], task_names: Sequence[str]) -> Dict[str, float]:
    """Final-row accuracy averaged within each chained block.

    M is the LAST-PASS slice (K rows × K cols), not the full
    multi-pass matrix.
    """
    K = len(M)
    if K == 0:
        return {}
    final_row = M[K - 1]
    out: Dict[str, float] = {}
    for prefix, block_label in [
        ("mnist", "phase1_mnist"),
        ("fashion_mnist", "phase2_fashion"),
        ("emnist_letters", "phase3_emnist"),
    ]:
        block_idxs = [j for j, nm in enumerate(task_names) if nm.startswith(prefix)]
        if block_idxs:
            out[block_label] = sum(final_row[j] for j in block_idxs) / len(block_idxs)
    return out


def report(results: Sequence[Dict[str, object]]) -> None:
    print()
    print("=" * 78)
    print("bench_chained_15task — Final Report")
    print("=" * 78)
    for r in results:
        K = len(r["task_names"])
        print(f"\n[{r['label']}]")
        print(f"  arch:               {r['initial_arch']} → {r['final_arch']}")
        print(f"  params total:       {r['initial_n_params']} → {r['final_n_params']}")
        print(f"  params trainable:   {r['initial_trainable']} → {r['final_trainable']}  "
              f"(cap-relevant)")
        print(f"  cap_bytes:          {r['cap_bytes']:_}  "
              f"(= {r['cap_bytes']//4:_} trainable params)")
        print(f"  cum grows allowed:  {r['cumulative_grows_allowed']}")
        print(f"  cum grows denied:   {r['cumulative_grows_denied']}")
        print(f"  cum dream purges:   {r['cumulative_purges']}")
        print(f"  cum dream latched:  {r['cumulative_latched']}")
        if r.get("cumulative_archives", 0) > 0 or ARCHIVE_ENABLED:
            print(f"  cum archives:       {r.get('cumulative_archives', 0)}  "
                  f"(per-layer {r.get('n_archived_per_layer_final', [])})")
            if r.get("quantization_report") is not None:
                qr = r["quantization_report"]
                print(f"  quantization ({qr['mode']}, FP32 active):")
                print(f"     full pre→post:    {qr['full_pre']:.4f} → "
                      f"{qr['full_post']:.4f}  (Δ {qr['full_delta']:+.4f})")
                print(f"     domain pre→post:  {qr['domain_pre']:.4f} → "
                      f"{qr['domain_post']:.4f}  (Δ {qr['domain_delta']:+.4f})")
                print(f"     task pre→post:    {qr['task_pre']:.4f} → "
                      f"{qr['task_post']:.4f}  (Δ {qr['task_delta']:+.4f})")
                if "full_post_bf16_int8" in qr:
                    print(f"  deployment ({qr['mode']}, BF16 active):")
                    print(f"     full pre→post:    {qr['full_pre']:.4f} → "
                          f"{qr['full_post_bf16_int8']:.4f}  "
                          f"(Δ {qr['full_delta_bf16_int8']:+.4f})")
                    print(f"     domain pre→post:  {qr['domain_pre']:.4f} → "
                          f"{qr['domain_post_bf16_int8']:.4f}  "
                          f"(Δ {qr['domain_delta_bf16_int8']:+.4f})")
                    print(f"     task pre→post:    {qr['task_pre']:.4f} → "
                          f"{qr['task_post_bf16_int8']:.4f}  "
                          f"(Δ {qr['task_delta_bf16_int8']:+.4f})")
                sb = qr["storage_breakdown"]
                print(f"  storage FP32-baseline (KB): "
                      f"L0 {sb['L0_kb']:.1f}  L1 {sb['L1_kb']:.1f}  "
                      f"head {sb['head_kb']:.1f}  total {sb['total_kb']:.1f}")
                if "storage_breakdown_bf16" in qr:
                    sbb = qr["storage_breakdown_bf16"]
                    print(f"  storage BF16-baseline (KB): "
                          f"L0 {sbb['L0_kb']:.1f}  L1 {sbb['L1_kb']:.1f}  "
                          f"head {sbb['head_kb']:.1f}  total {sbb['total_kb']:.1f}")
        print(f"  final acc full:     {r['final_accuracy']:.4f}  "
              f"(30-class full-softmax — headline)")
        print(f"  final acc domain:   {r.get('final_accuracy_domain', float('nan')):.4f}  "
              f"(10-class restricted to dataset group — realistic deployment)")
        print(f"  final acc task:     {r.get('final_accuracy_aware', float('nan')):.4f}  "
              f"(2-class restricted to binary task — generous diagnostic)")
        print(f"  avg forgetting:     full {r['avg_forgetting']:.4f}  "
              f"domain {r.get('avg_forgetting_domain', float('nan')):.4f}  "
              f"task {r.get('avg_forgetting_aware', float('nan')):.4f}")
        print(f"  wall-clock:         {r['wall_clock_seconds']:.1f}s")
        if r.get("n_passes", 1) > 1:
            print(f"  per-pass headline acc (full / domain / task):")
            for ps in r.get("pass_summary", []):
                print(f"     pass {int(ps['pass_idx'])+1}: "
                      f"{ps['final_accuracy']:.4f} / "
                      f"{ps.get('final_accuracy_domain', float('nan')):.4f} / "
                      f"{ps.get('final_accuracy_aware', float('nan')):.4f}")
        phase_means = _phase_means(r["last_pass_matrix"], r["task_names"])
        phase_means_domain = _phase_means(
            r.get("last_pass_matrix_domain") or r["last_pass_matrix"],
            r["task_names"],
        )
        phase_means_aware = _phase_means(
            r.get("last_pass_matrix_aware") or r["last_pass_matrix"],
            r["task_names"],
        )
        print("  per-phase (full / domain / task):")
        for nm in phase_means:
            full_v = phase_means[nm]
            domain_v = phase_means_domain.get(nm, float("nan"))
            aware_v = phase_means_aware.get(nm, float("nan"))
            print(f"     {nm:<20s} {full_v:.4f}  /  "
                  f"{domain_v:.4f}  /  {aware_v:.4f}")

    print()
    print("Headline (full / domain / task across arms):")
    for r in results:
        print(f"  {r['label']:<28s}  "
              f"full {r['final_accuracy']:.4f}  "
              f"domain {r.get('final_accuracy_domain', float('nan')):.4f}  "
              f"task {r.get('final_accuracy_aware', float('nan')):.4f}  "
              f"(full-forget {r['avg_forgetting']:+.4f})")
    print()


# ---------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------


def _mean_std(xs: Sequence[float]) -> Tuple[float, float, int]:
    """Return (mean, sample-std, n) for a sequence of finite floats.
    NaNs are filtered out."""
    finite = [x for x in xs if isinstance(x, (int, float)) and x == x
              and not math.isinf(x)]
    n = len(finite)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    if n == 1:
        return (float(finite[0]), 0.0, 1)
    return (statistics.mean(finite), statistics.stdev(finite), n)


def _phase_means_for_metric(r: Dict[str, object], metric_key: str) -> Dict[str, float]:
    """Pull per-phase mean accuracies from one result dict for one metric.
    metric_key in {"last_pass_matrix", "last_pass_matrix_domain",
    "last_pass_matrix_aware"}."""
    M = r.get(metric_key) or r["accuracy_matrix"]
    return _phase_means(M, r["task_names"])


def _paired_sigma(
    by_arm: Dict[str, List[Dict[str, object]]],
    arm_a: str,
    arm_b: str,
    metric: str,
) -> Tuple[float, float, float, int]:
    """Paired-difference sigma for arm_a vs arm_b on a scalar metric.

    For each seed where both arms have a finite value, compute
    diff = a - b. Returns (mean_diff, std_diff, sigma, n).
    sigma = mean_diff / std_diff (positive ⇒ a > b on that metric).
    """
    if arm_a not in by_arm or arm_b not in by_arm:
        return (float("nan"), float("nan"), float("nan"), 0)
    # Index by seed so we can pair correctly.
    a_by_seed = {r["seed"]: r for r in by_arm[arm_a]}
    b_by_seed = {r["seed"]: r for r in by_arm[arm_b]}
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    diffs: List[float] = []
    for s in seeds:
        va = a_by_seed[s].get(metric)
        vb = b_by_seed[s].get(metric)
        if (isinstance(va, (int, float)) and isinstance(vb, (int, float))
                and va == va and vb == vb):
            diffs.append(float(va) - float(vb))
    if len(diffs) < 2:
        m = diffs[0] if diffs else float("nan")
        return (m, float("nan"), float("nan"), len(diffs))
    m = statistics.mean(diffs)
    s = statistics.stdev(diffs)
    sig = m / s if s > 0 else float("inf") if m != 0 else 0.0
    return (m, s, sig, len(diffs))


def report_multiseed(
    all_results: Sequence[Dict[str, object]],
    arms: Sequence[str],
) -> None:
    """Aggregate report across seeds. Prints mean ± std for the three
    headlines (full / domain / task) per arm, per-phase means, and
    paired σ-differences for the dream-vs-no-dream comparison.
    """
    by_arm: Dict[str, List[Dict[str, object]]] = {}
    for r in all_results:
        by_arm.setdefault(str(r["label"]), []).append(r)

    seeds_seen = sorted({int(r["seed"]) for r in all_results})
    n_seeds = len(seeds_seen)

    print()
    print("=" * 78)
    print(f"bench_chained_15task — Multi-seed Report (n={n_seeds} seeds)")
    print("=" * 78)
    print(f"Seeds: {seeds_seen}")
    print()

    for arm in arms:
        rs = by_arm.get(arm, [])
        if not rs:
            continue
        print(f"[{arm}]  ({len(rs)} seeds)")
        # Three headline scalars
        for metric_key, metric_label in [
            ("final_accuracy", "full   "),
            ("final_accuracy_domain", "domain "),
            ("final_accuracy_aware", "task   "),
        ]:
            xs = [float(r[metric_key]) for r in rs if metric_key in r]
            m, sd, n = _mean_std(xs)
            print(f"  final acc {metric_label}: {m:.4f} ± {sd:.4f}  (n={n})")
        for metric_key, metric_label in [
            ("avg_forgetting", "full   "),
            ("avg_forgetting_domain", "domain "),
            ("avg_forgetting_aware", "task   "),
        ]:
            xs = [float(r[metric_key]) for r in rs if metric_key in r]
            m, sd, n = _mean_std(xs)
            print(f"  forgetting {metric_label}: {m:+.4f} ± {sd:.4f}")
        # Per-phase per-metric
        # Collect phase means across seeds, per metric.
        for matrix_key, metric_label in [
            ("last_pass_matrix", "full"),
            ("last_pass_matrix_domain", "domain"),
            ("last_pass_matrix_aware", "task"),
        ]:
            phase_lists: Dict[str, List[float]] = {}
            for r in rs:
                phases = _phase_means_for_metric(r, matrix_key)
                for k, v in phases.items():
                    phase_lists.setdefault(k, []).append(v)
            if phase_lists:
                print(f"  per-phase {metric_label}:")
                for k in sorted(phase_lists):
                    m, sd, n = _mean_std(phase_lists[k])
                    print(f"     {k:<22s} {m:.4f} ± {sd:.4f}")
        # Substrate counters
        for ck, label in [
            ("cumulative_grows_allowed", "cum grows allowed "),
            ("cumulative_grows_denied",  "cum grows denied  "),
            ("cumulative_purges",        "cum dream purges  "),
        ]:
            xs = [float(r[ck]) for r in rs if ck in r]
            m, sd, _ = _mean_std(xs)
            print(f"  {label}: {m:.2f} ± {sd:.2f}")
        # Final trainable params (sanity check that arms held to budget)
        xs = [float(r["final_trainable"]) for r in rs]
        m, sd, _ = _mean_std(xs)
        print(f"  final trainable    : {m:.0f} ± {sd:.0f}")
        print()

    # Cross-arm summary
    print("Headline (mean ± std across seeds):")
    print(f"  {'arm':<28s}  {'full':<18s}  {'domain':<18s}  {'task':<18s}")
    for arm in arms:
        rs = by_arm.get(arm, [])
        if not rs:
            continue
        cells: List[str] = []
        for metric_key in ("final_accuracy", "final_accuracy_domain",
                           "final_accuracy_aware"):
            xs = [float(r[metric_key]) for r in rs if metric_key in r]
            m, sd, _ = _mean_std(xs)
            cells.append(f"{m:.4f}±{sd:.4f}")
        print(f"  {arm:<28s}  {cells[0]:<18s}  {cells[1]:<18s}  {cells[2]:<18s}")
    print()

    # Paired σ-differences. The protagonist comparison Rocky cares about
    # most is grown_capped_dream vs grown_capped_no_dream on task-aware.
    # Generate all pairs among requested arms.
    print("Paired σ-differences (arm_a − arm_b across seeds):")
    print(f"  {'comparison':<48s}  {'metric':<8s}  {'mean Δ':>10s}  "
          f"{'std Δ':>9s}  {'σ':>6s}  {'n':>3s}")
    metric_pairs = [
        ("final_accuracy", "full"),
        ("final_accuracy_domain", "domain"),
        ("final_accuracy_aware", "task"),
    ]
    arm_list = [a for a in arms if a in by_arm and by_arm[a]]
    for i, a in enumerate(arm_list):
        for b in arm_list[i + 1:]:
            for mkey, mlabel in metric_pairs:
                m, s, sig, n = _paired_sigma(by_arm, a, b, mkey)
                comp = f"{a} vs {b}"
                if n == 0:
                    continue
                sig_str = (f"{sig:+6.2f}" if sig == sig and not math.isinf(sig)
                           else "  inf" if math.isinf(sig) else "   nan")
                print(f"  {comp:<48s}  {mlabel:<8s}  {m:>+10.4f}  "
                      f"{s:>9.4f}  {sig_str:>6s}  {n:>3d}")
    print()


def write_csv_multiseed(
    all_results: Sequence[Dict[str, object]], csv_path: str,
) -> None:
    """Per-seed-per-arm scalar summary CSV.

    Wide format: one row per (seed, arm). Excludes the K×K accuracy
    matrix to keep things readable; matrices are in the .log.
    """
    fields = [
        "seed", "label", "do_growth", "do_dream", "cap_bytes",
        "initial_arch", "final_arch",
        "initial_n_params", "final_n_params",
        "initial_trainable", "final_trainable",
        "wall_clock_seconds",
        "final_accuracy", "final_accuracy_domain", "final_accuracy_aware",
        "avg_forgetting", "avg_forgetting_domain", "avg_forgetting_aware",
        "cumulative_grows_allowed", "cumulative_grows_denied",
        "cumulative_purges", "cumulative_latched",
    ]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in all_results:
            row = []
            for f in fields:
                v = r.get(f, "")
                if isinstance(v, float):
                    row.append(f"{v:.6f}")
                elif isinstance(v, tuple):
                    row.append(str(v))
                else:
                    row.append(v)
            w.writerow(row)
    print(f"  log: {csv_path}")


def write_csv(results: Sequence[Dict[str, object]], csv_path: str) -> None:
    K = len(results[0]["task_names"]) if results else 0
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = [
            "label", "do_growth", "do_dream", "cap_bytes",
            "initial_arch", "final_arch",
            "initial_n_params", "final_n_params",
            "wall_clock_seconds", "final_accuracy", "avg_forgetting",
            "cum_grows_allowed", "cum_grows_denied",
            "cum_purges", "cum_latched",
        ]
        for i in range(K):
            for j in range(K):
                header.append(f"A[{i+1}][{j+1}]")
        w.writerow(header)
        for r in results:
            row = [
                r["label"], r["do_growth"], r["do_dream"], r["cap_bytes"],
                str(r["initial_arch"]), str(r["final_arch"]),
                r["initial_n_params"], r["final_n_params"],
                f"{r['wall_clock_seconds']:.2f}",
                f"{r['final_accuracy']:.6f}", f"{r['avg_forgetting']:.6f}",
                r["cumulative_grows_allowed"], r["cumulative_grows_denied"],
                r["cumulative_purges"], r["cumulative_latched"],
            ]
            # CSV stores the LAST-PASS K×K slice for back-compat and
            # readability. Multi-pass diagnostics are in the .log.
            last_pass_matrix = r.get("last_pass_matrix") or r["accuracy_matrix"]
            for i in range(K):
                for j in range(K):
                    v = last_pass_matrix[i][j]
                    row.append("" if v != v else f"{v:.6f}")
            w.writerow(row)
    print(f"  log: {csv_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny budget for fast smoke test (1 epoch/task).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--seeds", default="",
        help="Comma-separated list of seeds (e.g. 0,1,2,...,11) for "
             "multi-seed run. Overrides --seed when provided.",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--arms", default=",".join(DEFAULT_ARMS),
        help=f"Comma-separated subset of {DEFAULT_ARMS}",
    )
    parser.add_argument(
        "--csv", default="bench_chained_15task_log.csv",
        help="Output CSV filename (under outputs/).",
    )
    args = parser.parse_args(argv)

    n_epochs = N_EPOCHS_PER_TASK_SMOKE if args.smoke else N_EPOCHS_PER_TASK
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_DEFINITIONS:
            raise SystemExit(
                f"Unknown arm {a!r}. Available: {list(ARM_DEFINITIONS)}"
            )

    if args.seeds.strip():
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]

    print("=" * 78)
    print("Trioron — bench_chained_15task: MNIST → FashionMNIST → EMNIST-letters")
    print("=" * 78)
    print(f"Epochs/task:        {n_epochs}{' [SMOKE]' if args.smoke else ''}")
    print(f"K_settle epochs:    {K_SETTLE_EPOCHS}  (Fix B — growth deferred "
          f"until after settle)")
    print(f"L0 width (frozen):  {L0_WIDTH}")
    print(f"H_init grown (L1):  {H_INIT_GROWN}")
    print(f"H fixed:            {H_FIXED}")
    print(f"N_grow_per_task:    {N_GROW_PER_TASK}")
    print(f"Cap (trainable):    {M_MAX_BYTES_CAPPED:_} B "
          f"= {M_MAX_BYTES_CAPPED // 4:_} params")
    print(f"EWC intertask:      {EWC_INTERTASK}")
    print(f"Curriculum passes:  {N_CURRICULUM_PASSES}")
    print(f"Warmup enabled:     {WARMUP_ENABLED}")
    if WARMUP_ENABLED:
        print(f"Infancy:            {N_INFANCY_PER_DATASET}/dataset, "
              f"{N_WARMUP_STEPS} warmup steps")
    print(f"Arms:               {arms}")
    if len(seeds) == 1:
        print(f"Seed:               {seeds[0]}")
    else:
        print(f"Seeds (n={len(seeds)}):     {seeds}")
    print()

    # Build the bundle with the holdout reserved (so we can flip
    # WARMUP_ENABLED back on without restructuring). When warmup is
    # disabled, infancy_view is None and run_arm skips the warmup.
    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=args.data_root,
        n_holdout_per_dataset=N_INFANCY_PER_DATASET,
    )
    specs = chained_15_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [s.global_classes for s in specs]
    if WARMUP_ENABLED:
        infancy_view = bundle.infancy_view(specs)
        print(f"[infancy] view built: {infancy_view.n_examples()} samples "
              f"covering {len(set(infancy_view.labels_global.tolist()))} global classes")
    else:
        infancy_view = None

    all_results: List[Dict[str, object]] = []
    for seed_idx, seed in enumerate(seeds):
        if len(seeds) > 1:
            print()
            print("#" * 78)
            print(f"#   SEED {seed}  ({seed_idx+1}/{len(seeds)})")
            print("#" * 78)
        seed_results: List[Dict[str, object]] = []
        for arm in arms:
            r = run_arm(
                arm,
                seed=seed + (hash(arm) % 7919),
                n_epochs_per_task=n_epochs,
                train_views=train_views, eval_views=eval_views,
                task_class_lists=task_class_lists,
                infancy_view=infancy_view,
                n_passes=N_CURRICULUM_PASSES,
            )
            r["seed"] = seed
            seed_results.append(r)
            all_results.append(r)

        # Per-seed report so each block is readable while the run is in
        # progress. Single-seed mode looks identical to before.
        report(seed_results)

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, args.csv)

    if len(seeds) > 1:
        report_multiseed(all_results, arms)
        # Multi-seed CSV: per-(seed, arm) scalar summary. Single-seed
        # path keeps the legacy K×K-matrix CSV for back-compat.
        ms_csv_path = csv_path.replace(".csv", "_multiseed.csv")
        write_csv_multiseed(all_results, ms_csv_path)
    else:
        write_csv(all_results, csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
