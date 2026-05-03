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
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import TrioronNetwork
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
    compress,
    purge,
)

from experiments.datasets import (
    DEFAULT_DATA_ROOT,
    DatasetBundle,
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
) -> None:
    estimate_fisher_for_task(net, train_view, active_classes)
    net.update_lambda_all()
    with torch.no_grad():
        for layer in net.layers:
            layer.lam.clamp_(min=LAMBDA_FLOOR)
    net.anchor_all()


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
    for epoch in range(n_epochs):
        epoch_loss_sum = 0.0
        epoch_n_batches = 0
        for x, y_global in train_view.iter_epoch(BATCH):
            logits = net(x)
            l_task = masked_cross_entropy(logits, y_global, active_classes=active)
            l = (l_task + ewc_baseline * net.ewc_penalty()
                 if ewc_baseline > 0 else l_task)
            opt.zero_grad()
            l.backward()
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
    with torch.no_grad():
        for i, v in enumerate(eval_views):
            x, y = v.all_examples()
            logits = net(x)
            head_size = logits.shape[1]
            full_accs.append(accuracy(logits, y))

            # Task-aware: restrict to the binary pair.
            active = task_class_lists[i]
            if max(active) < head_size:
                aware_accs.append(accuracy(
                    logits, y, restrict_to=active,
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
) -> Dict[str, object]:
    """Run the chained curriculum, optionally repeated for `n_passes`.

    On pass > 0 (revisit), `n_grow_per_task` is forced to 0: no new
    neurogenesis on revisit, only consolidation through retraining +
    dreaming. EWC anchors carry forward across passes.

    Per-task training is `n_epochs_per_task` proper minibatch epochs
    (each sample seen exactly once per epoch).
    """
    K = len(train_views)
    n_total = K * n_passes
    initial_n_params = net.n_parameters()
    initial_trainable = trainable_params(net)
    initial_arch = tuple(net.n_nodes_per_layer())
    rng = random.Random(rng_seed)
    print(f"\n[{label}] start — arch {initial_arch}  "
          f"params {initial_n_params} (trainable {initial_trainable})  "
          f"growth={do_growth} dream={do_dream}  "
          f"cap_bytes={cap_bytes}  K={K}  passes={n_passes}  "
          f"epochs/task={n_epochs_per_task}")

    opt = optim.Adam(trainable_param_iter(net), lr=LR)
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
                opt = optim.Adam(trainable_param_iter(net), lr=LR)

            # 2. Print task header (pre-settle). Fix B's growth comes
            #    after a settle phase; arch shown here is pre-growth.
            print(f"\n[{label}] === Pass {pass_idx+1}/{n_passes} "
                  f"Task {local_task_idx+1}/{K}: {train_view.name} "
                  f"(active {active})  pre-arch={net.n_nodes_per_layer()} "
                  f"params={net.n_parameters()} "
                  f"(trainable {trainable_params(net)}/{cap_bytes//4 if cap_bytes < M_MAX_BYTES_UNCAPPED else '∞'}) ===")

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
                        net, GROWTH_TARGET_LAYER_IDX, cap_bytes, local_task_idx,
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
                            opt = optim.Adam(trainable_param_iter(net), lr=LR)
                            ok2, reason2 = try_grow_one(
                                net, GROWTH_TARGET_LAYER_IDX, cap_bytes,
                                local_task_idx,
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
                opt = optim.Adam(trainable_param_iter(net), lr=LR)
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
                    )
            else:
                opt = train_one_task(
                    net, local_task_idx, train_view, active,
                    n_epochs=n_epochs_per_task, opt=opt,
                    ewc_baseline=ewc_baseline,
                    label=label, n_total_tasks=K,
                )

            # 4. Consolidate.
            consolidate_task(net, train_view, active)
            ewc_baseline = EWC_INTERTASK

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
                opt = optim.Adam(trainable_param_iter(net), lr=LR)
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
                evaluate_all_tasks(net, eval_views, task_class_lists)
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
    return {
        "label": label,
        "do_growth": do_growth,
        "do_dream": do_dream,
        "cap_bytes": cap_bytes,
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
        "per_task_log": per_task_log,
        "task_names": [v.name for v in eval_views],
        "wall_clock_seconds": elapsed,
    }


# ---------------------------------------------------------------------
# Arm dispatch
# ---------------------------------------------------------------------


ARM_DEFINITIONS = {
    "fixed_ewc": {
        "h_init": H_FIXED, "do_growth": False, "do_dream": False,
        "cap_bytes": M_MAX_BYTES_UNCAPPED, "freeze_l0": False,
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
) -> Dict[str, object]:
    cfg = ARM_DEFINITIONS[arm]
    torch.manual_seed(seed)
    net = make_classifier(
        INPUT_DIM, L0_WIDTH, cfg["h_init"], INIT_CLASSES,
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
