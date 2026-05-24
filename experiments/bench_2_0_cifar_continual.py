"""v2.0 vs v1.0 on split-CIFAR-100 (10x10 continual classification).

Four arms isolate three contributions:
  (a) v1.0 random Gaussian L0      — apples-to-apples 1.0 baseline
  (b) v1.0 + LCN L0                 — isolates LCN's contribution
  (c) v2.0 static oracle            — LCN + baked axes 5,6 at t=0
  (d) v2.0 self-arrange             — LCN + axes 5,6 grow on internal stress

Reading: (a)->(b) = LCN. (b)->(c) = baked axes. (c)->(d) = self-arrangement.

CL machinery is the 567b2c1 stack across all arms (credit-based row
freezing + manifold replay + cosine head), so only substrate primitives
differ across arms.

Axes deliberately not enabled in self-arrange first pass:
  - Axis 3 insert_layer regressed in isolation on chained-15
    ([[all_axes_chained15_regression]]). Follow-up bench can add it.
  - Long-range edges (Axis 1) and archive_input (Axis 2) need a
    curriculum signal that this bench doesn't generate naturally.

Env vars:
  CIFAR_SMOKE=1               -- single seed, 4 epochs, prints per phase
  CIFAR_N_SEEDS=5             -- default 1
  CIFAR_EPOCHS=8              -- per-phase epochs, default 4 in smoke
  CIFAR_BATCH=256
  CIFAR_LR=0.01
  CIFAR_ARMS=a,b,c,d          -- subset of arms to run
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trioron.network import TrioronNetwork
from trioron import masked_cross_entropy
from trioron.spatial import (
    build_lcn_mask, grid_positions_2d, pixel_positions_2d,
)
from trioron.manifold import (
    ManifoldStore, cosine_logits, settle_head_with_retry,
)
from trioron.api import axis6_spawn
from experiments.cifar.datasets import load_cifar100

# Conv-by-emergence machinery — reused from axis6_credit_chained15.
# These let L1 cells acquire positions and self-organize into a
# locally-connected layer with shared weights (Hebbian + diffusion).
# Per [[conv_by_emergence_null_result]] these were null on chained-15
# glyphs; CIFAR's real spatial structure is the natural test bed.
from experiments.axis6_credit_chained15 import (
    extend_l1_lcn_mask,
    reapply_l1_lcn_mask,
    update_activation_correlation,
    extend_act_corr_on_spawn,
    install_learnable_positions,
    extend_learnable_positions,
    sync_positions_to_buffer,
    alignment_aux_loss,
    relax_after_spawn,
)


# ----------------------------------------------------------------------
# Architecture constants (CIFAR-100 scale)
# ----------------------------------------------------------------------
IMAGE_HW = 32
INPUT_DIM = 3 * IMAGE_HW * IMAGE_HW   # 3072
L0_GRID_X = 16
L0_GRID_Y = 16
L0_WIDTH = L0_GRID_X * L0_GRID_Y      # 256 cells, retinotopic
LCN_SIGMA = 0.10                      # image-coord units
H_INIT = 64                           # L1 starting cells (self-arrange)
ORACLE_EXTRA_CELLS = 24               # oracle baked at H_INIT + 24 = 88; sized
                                      # to approximately match self-arrange's
                                      # likely final cell count (spawn_cap=8/
                                      # phase x 10 phases x ~30% trigger rate)
N_CLASSES = 100
COSINE_TEMP = 16.0

# CIFAR-100 normalization (standard torchvision constants)
CIFAR_MEAN = torch.tensor([0.5071, 0.4866, 0.4409]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------
# RGB LCN mask: spatial locality, all-channels at each location
# ----------------------------------------------------------------------
def build_rgb_lcn_mask() -> torch.Tensor:
    """Return (L0_WIDTH, INPUT_DIM=3072) locality mask.

    Each L0 cell reads a Gaussian patch in (x,y) image coords across
    all 3 channels: mask is the single-channel (L0_WIDTH, 1024) mask
    tiled 3 times along the input dim. Matches CHW flatten order:
    indices [0..1023] = R, [1024..2047] = G, [2048..3071] = B.
    """
    cell_pos = grid_positions_2d(L0_GRID_X, L0_GRID_Y)        # (256, 2)
    pix_pos = pixel_positions_2d(IMAGE_HW, IMAGE_HW)          # (1024, 2)
    mask_1c = build_lcn_mask(
        cell_pos, pix_pos, mode="soft", sigma=LCN_SIGMA,
    )                                                          # (256, 1024)
    return mask_1c.repeat(1, 3).contiguous()                  # (256, 3072)


def pixel_positions_rgb() -> torch.Tensor:
    """Return (3072, 2) per-input-index position tensor for the
    CHW-flattened image. Tile pixel_positions_2d(32,32) three times
    so all three channels at pixel (r,c) share position (x,y)."""
    pp = pixel_positions_2d(IMAGE_HW, IMAGE_HW)               # (1024, 2)
    return pp.repeat(3, 1).contiguous()                       # (3072, 2)


def recompute_l0_lcn_mask(net: TrioronNetwork) -> None:
    """Recompute L0's LCN mask from current L0 cell_position. Used
    when L0 positions are learnable and have moved. apply_to_weights
    stays False so L0.W (the frozen random Gaussian) is untouched;
    forward applies the new mask multiplicatively."""
    L0 = net.layers[0]
    pix_pos = getattr(L0, "lcn_in_positions", None)
    if pix_pos is None:
        return
    L0.enable_lcn(
        pix_pos, mode="soft", sigma=LCN_SIGMA,
        apply_to_weights=False,
    )




# ----------------------------------------------------------------------
# Augmentation: pad-4 random crop + horizontal flip (per-batch, GPU-cheap)
# ----------------------------------------------------------------------
def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """x: (B, 3, 32, 32). Returns augmented + normalized + flattened
    (B, 3072) tensor. Normalization uses CIFAR-100 mean/std."""
    B = x.shape[0]
    # Pad-4 + random crop back to 32x32
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    offs_x = torch.randint(0, 9, (B,))
    offs_y = torch.randint(0, 9, (B,))
    cropped = torch.empty_like(x)
    for i in range(B):
        cropped[i] = padded[i, :, offs_y[i]:offs_y[i]+32, offs_x[i]:offs_x[i]+32]
    # Random horizontal flip per sample (50%)
    flip_mask = torch.rand(B) < 0.5
    if flip_mask.any():
        cropped[flip_mask] = cropped[flip_mask].flip(dims=[-1])
    # Normalize + flatten
    normalized = (cropped - CIFAR_MEAN) / CIFAR_STD
    return normalized.flatten(start_dim=1)


def normalize_only(x: torch.Tensor) -> torch.Tensor:
    """Eval/test: normalize + flatten, no augmentation."""
    return ((x - CIFAR_MEAN) / CIFAR_STD).flatten(start_dim=1)


# ----------------------------------------------------------------------
# Per-task view
# ----------------------------------------------------------------------
@dataclass
class TaskView:
    name: str
    classes: List[int]                     # global class IDs in this task
    x_train: torch.Tensor                  # (N_tr, 3, 32, 32)
    y_train: torch.Tensor                  # (N_tr,)
    x_test: torch.Tensor                   # (N_te, 3, 32, 32)
    y_test: torch.Tensor                   # (N_te,)


def build_cifar_tasks(n_tasks: int = 10, per_task: int = 10) -> List[TaskView]:
    if n_tasks * per_task > 100:
        raise ValueError(f"n_tasks*per_task={n_tasks*per_task} > 100")
    xtr, ytr = load_cifar100(train=True, download=True)
    xte, yte = load_cifar100(train=False, download=True)
    tasks = []
    for i in range(n_tasks):
        cls = list(range(i * per_task, (i + 1) * per_task))
        mtr = torch.zeros_like(ytr, dtype=torch.bool)
        mte = torch.zeros_like(yte, dtype=torch.bool)
        for c in cls:
            mtr |= ytr == c
            mte |= yte == c
        tasks.append(TaskView(
            name=f"cifar_{cls[0]:03d}_{cls[-1]:03d}",
            classes=cls,
            x_train=xtr[mtr], y_train=ytr[mtr],
            x_test=xte[mte], y_test=yte[mte],
        ))
    return tasks


# ----------------------------------------------------------------------
# Arm builders
# ----------------------------------------------------------------------
def _freeze_l0(net: TrioronNetwork) -> None:
    L0 = net.layers[0]
    L0.W.requires_grad_(False)
    L0.b.requires_grad_(False)
    if hasattr(L0, "branch_weight"):
        L0.branch_weight.requires_grad_(False)


def _setup_head(net: TrioronNetwork) -> None:
    """Cosine head: freeze bias at zero."""
    head = net.layers[2]
    with torch.no_grad():
        head.b.data.zero_()
    head.b.requires_grad_(False)


def _seed_l0_positions(net: TrioronNetwork) -> None:
    """Image-plane positions for L0 cells (used by axis6_spawn for the
    grown arm; harmless for floor arms)."""
    L0 = net.layers[0]
    cell_xy = grid_positions_2d(L0_GRID_X, L0_GRID_Y)  # (256, 2)
    with torch.no_grad():
        L0.cell_position[:L0_WIDTH, 0] = cell_xy[:, 0]
        L0.cell_position[:L0_WIDTH, 1] = cell_xy[:, 1]
        L0.cell_position[:L0_WIDTH, 2] = 0.0


def _seed_l1_grid_positions(net: TrioronNetwork, n_l1: int) -> None:
    side = max(1, int(math.ceil(math.sqrt(n_l1))))
    cx = (torch.arange(side).float() + 0.5) / side
    cy = (torch.arange(side).float() + 0.5) / side
    gx, gy = torch.meshgrid(cx, cy, indexing="ij")
    pos = torch.stack([gx, gy], dim=-1).reshape(-1, 2)[:n_l1]
    L1 = net.layers[1]
    with torch.no_grad():
        L1.cell_position[:n_l1, 0] = pos[:, 0]
        L1.cell_position[:n_l1, 1] = pos[:, 1]
        L1.cell_position[:n_l1, 2] = 0.0


def build_arm_a_random_gaussian(seed: int) -> TrioronNetwork:
    """v1.0 floor: dense L0 (3072->256) random Gaussian, frozen.
    L1 starts at H_INIT, no axes."""
    torch.manual_seed(seed)
    net = TrioronNetwork([
        (INPUT_DIM, L0_WIDTH, "relu"),
        (L0_WIDTH, H_INIT, "relu"),
        (H_INIT, N_CLASSES, "linear"),
    ])
    _freeze_l0(net)
    _setup_head(net)
    _seed_l0_positions(net)
    _seed_l1_grid_positions(net, H_INIT)
    return net


def build_arm_b_lcn(seed: int) -> TrioronNetwork:
    """v1.0 + LCN: L0 (3072->256) random Gaussian masked to retinotopic
    locality, frozen. L1 starts at H_INIT, no axes."""
    torch.manual_seed(seed)
    net = TrioronNetwork([
        (INPUT_DIM, L0_WIDTH, "relu"),
        (L0_WIDTH, H_INIT, "relu"),
        (H_INIT, N_CLASSES, "linear"),
    ])
    L0 = net.layers[0]
    mask = build_rgb_lcn_mask().to(device=L0.W.device, dtype=L0.W.dtype)
    L0.register_buffer("W_lcn_mask", mask)
    with torch.no_grad():
        L0.W.data.mul_(mask)
        if hasattr(L0, "W_anchor"):
            L0.W_anchor.data.mul_(mask.to(L0.W_anchor.dtype))
    _freeze_l0(net)
    _setup_head(net)
    _seed_l0_positions(net)
    _seed_l1_grid_positions(net, H_INIT)
    return net


def build_arm_c_oracle(seed: int) -> TrioronNetwork:
    """v2.0 oracle: LCN + L1 baked at H_INIT + ORACLE_EXTRA_CELLS, no
    growth during training. (Axis 5 K=2 dendrites kept off in first
    pass — Phase 6 K=1->K=2 transition is a known discontinuity per
    [[trioron-2-0-axis5-handoff]] risk #9; bake K=2 separately when
    we add it.)"""
    torch.manual_seed(seed)
    n_l1 = H_INIT + ORACLE_EXTRA_CELLS
    net = TrioronNetwork([
        (INPUT_DIM, L0_WIDTH, "relu"),
        (L0_WIDTH, n_l1, "relu"),
        (n_l1, N_CLASSES, "linear"),
    ])
    L0 = net.layers[0]
    mask = build_rgb_lcn_mask().to(device=L0.W.device, dtype=L0.W.dtype)
    L0.register_buffer("W_lcn_mask", mask)
    with torch.no_grad():
        L0.W.data.mul_(mask)
        if hasattr(L0, "W_anchor"):
            L0.W_anchor.data.mul_(mask.to(L0.W_anchor.dtype))
    _freeze_l0(net)
    _setup_head(net)
    _seed_l0_positions(net)
    _seed_l1_grid_positions(net, n_l1)
    return net


def build_arm_d_self_arrange(seed: int) -> TrioronNetwork:
    """v2.0 self-arrange: LCN + L1 starts at H_INIT, Axis 6 spawn
    enabled (caller drives spawning in the train loop on internal
    stress)."""
    return build_arm_b_lcn(seed)  # same shape; arm differs by training loop


# ----------------------------------------------------------------------
# Training loop (one task) — shared across arms; flags toggle features
# ----------------------------------------------------------------------
@dataclass
class TaskResult:
    name: str
    engagement_frac: Optional[torch.Tensor]
    train_loss_final: float
    train_acc_final: float
    n_spawns: int
    spawn_events: List[Tuple[int, str]]


def _forward(
    net: TrioronNetwork,
    x_flat: torch.Tensor,
    use_cosine: bool = True,
) -> torch.Tensor:
    """L0 -> L1 -> head with cosine logits."""
    h0 = net.layers[0](x_flat)
    h1 = net.layers[1](h0)
    head_W = net.layers[2].W
    if use_cosine:
        return cosine_logits(h1, head_W, temperature=COSINE_TEMP)
    return h1 @ head_W.t() + net.layers[2].b


def _apply_credit_mask(L1, cell_credit: Optional[torch.Tensor]) -> None:
    """Zero out gradients for credited (frozen) L1 cells, in-place."""
    if cell_credit is None or not cell_credit.any():
        return
    n = L1.n_nodes
    if cell_credit.numel() < n:
        cell_credit = torch.cat([cell_credit, torch.zeros(
            n - cell_credit.numel(), dtype=torch.bool,
        )])
    mask = (~cell_credit).to(dtype=L1.W.dtype).unsqueeze(1)
    if L1.W.grad is not None:
        L1.W.grad.mul_(mask)
    if L1.b.grad is not None:
        L1.b.grad.mul_(mask.squeeze(1))


def train_one_task(
    net: TrioronNetwork,
    task: TaskView,
    seen_classes: List[int],
    n_epochs: int,
    batch_size: int,
    lr: float,
    cell_credit: Optional[torch.Tensor],
    manifold: Optional[ManifoldStore],
    enable_axis6: bool,
    enable_conv_emergence: bool = False,
    enable_l0_plastic: bool = False,
    spawn_cooldown_steps: int = 50,
    spawn_cap_per_phase: int = 16,
    spawn_trigger: str = "frustrated",   # "frustrated" | "always"
    manifold_batch: int = 256,
    manifold_loss_weight: float = 1.0,
    conv_lambda_diff: float = float(os.environ.get("CIFAR_CONV_LAMBDA_DIFF", "0.0")),
    conv_align_weight: float = float(os.environ.get("CIFAR_CONV_ALIGN_W", "0.01")),
    conv_hebbian_decay: float = 0.9,
    verbose: bool = False,
) -> TaskResult:
    L0 = net.layers[0]
    L1 = net.layers[1]
    head = net.layers[2]

    # Engagement: per-cell mean activation rate across the phase (used
    # by post-phase credit-update logic in the caller).
    engagement_sum = torch.zeros(L1.n_nodes)
    engagement_count = 0

    # Active class list for masked CE (gradient only on seen classes).
    active = list(seen_classes)

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    n_tr = task.x_train.shape[0]
    n_batches = max(1, n_tr // batch_size)
    last_loss = float("nan")
    last_acc = float("nan")
    spawn_events: List[Tuple[int, str]] = []
    n_spawns = 0
    last_spawn_step = -spawn_cooldown_steps
    step = 0

    for epoch in range(n_epochs):
        perm = torch.randperm(n_tr)
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            x = task.x_train[idx]
            y = task.y_train[idx]
            x_in = augment_batch(x)

            if enable_conv_emergence:
                # Conv-emergence diffusion was removed (null results).
                # Fall through to standard forward.
                h0 = net.layers[0](x_in)
                h1 = L1(h0)
                # EMA updates for alignment. L1 always when
                # conv_emergence; L0 only when l0_plastic so we don't
                # pay the cost for the existing arms.
                update_activation_correlation(L1, h1, decay=conv_hebbian_decay)
                if enable_l0_plastic:
                    update_activation_correlation(L0, h0, decay=conv_hebbian_decay)
                logits = cosine_logits(h1, head.W, temperature=COSINE_TEMP)
            else:
                logits = _forward(net, x_in)
            loss = masked_cross_entropy(logits, y, active_classes=active)

            # Manifold replay (head-only gradient on synthetic samples
            # from previously-stored classes).
            if manifold is not None and manifold.has_classes():
                synth = manifold.sample_synthetic(manifold_batch)
                if synth is not None:
                    s_feats, s_labels = synth
                    with torch.no_grad():
                        s_h1 = net.layers[1](s_feats)
                    s_logits = cosine_logits(
                        s_h1, head.W, temperature=COSINE_TEMP,
                    )
                    syn_active = list(manifold.mu_per_class.keys())
                    syn_loss = masked_cross_entropy(
                        s_logits, s_labels, active_classes=syn_active,
                    )
                    loss = loss + manifold_loss_weight * syn_loss

            # Boquila alignment: pull activation-correlated cells
            # toward each other in position space. No-op until corr EMA
            # is seeded; only fires when enable_conv_emergence + the
            # learnable_positions param exists on that layer.
            if enable_conv_emergence and conv_align_weight > 0:
                l_align_l1 = alignment_aux_loss(L1, dims=2)
                if l_align_l1.requires_grad:
                    loss = loss + conv_align_weight * l_align_l1
                if enable_l0_plastic:
                    l_align_l0 = alignment_aux_loss(L0, dims=2)
                    if l_align_l0.requires_grad:
                        loss = loss + conv_align_weight * l_align_l0

            opt.zero_grad()
            loss.backward()
            _apply_credit_mask(L1, cell_credit)
            opt.step()

            if enable_conv_emergence:
                # Off-window L1.W entries can't accumulate under LCN
                # mask; also sync learnable positions back to buffer so
                # Axis 6, Hebbian, LCN all see migrated positions.
                reapply_l1_lcn_mask(L1)
                sync_positions_to_buffer(L1)
            if enable_l0_plastic:
                # L0 positions moved this step (alignment-loss gradient
                # via L1's correlation EMA — we don't track L0's own EMA
                # in this first cut). Sync to buffer and rebuild the L0
                # LCN mask so forward picks up the new cells' positions.
                sync_positions_to_buffer(L0)
                recompute_l0_lcn_mask(net)

            with torch.no_grad():
                last_loss = float(loss.item())
                preds = logits.argmax(dim=-1)
                last_acc = float((preds == y).float().mean().item())
                # Engagement: fraction of inputs each L1 cell fires on.
                if L1._last_y is not None:
                    fire = (L1._last_y > 0).float().mean(dim=0)
                    # Resize if L1 grew mid-task.
                    if fire.numel() > engagement_sum.numel():
                        engagement_sum = torch.cat([
                            engagement_sum,
                            torch.zeros(fire.numel() - engagement_sum.numel()),
                        ])
                    engagement_sum[:fire.numel()] += fire
                    engagement_count += 1

            # Axis 6 spawn trigger.
            #   "frustrated": spawn when train acc < 0.5 with cooldown
            #                 (the default; original behavior).
            #   "always":     spawn every cooldown_steps when under the
            #                 cap, irrespective of accuracy. Used to test
            #                 whether the governor (not the gate) is the
            #                 binding constraint on cell-first growth.
            trigger_fires = (
                last_acc < 0.5 if spawn_trigger == "frustrated" else True
            )
            if (enable_axis6
                and step - last_spawn_step >= spawn_cooldown_steps
                and n_spawns < spawn_cap_per_phase
                and trigger_fires
                and epoch >= 1):
                try:
                    parent_idx = int(L1._last_y.mean(dim=0).argmax().item())
                    new_idx = axis6_spawn(
                        net, layer_idx=1,
                        candidate_idx=parent_idx,
                        position_jitter=0.15,
                    )
                    spawn_events.append(
                        (step, f"AXIS6 spawn: parent={parent_idx} -> child={new_idx}")
                    )
                    n_spawns += 1
                    last_spawn_step = step
                    # Extend conv-emergence buffers for the new cell
                    # (LCN mask row, act-corr EMA row/col, learnable
                    # positions) and run a few relaxation steps to push
                    # the new cell away from neighbors.
                    if enable_conv_emergence:
                        extend_l1_lcn_mask(net)
                        extend_act_corr_on_spawn(L1, parent_idx, new_idx)
                        extend_learnable_positions(L1)
                        relax_after_spawn(
                            L1, n_steps=8, min_sep=0.08,
                            strength=0.4, mode="unit_box", dims=2,
                        )
                    # Rebuild optimizer — L1.W (and possibly
                    # cell_position_learnable) are new Parameter objects.
                    params = [p for p in net.parameters() if p.requires_grad]
                    opt = torch.optim.Adam(params, lr=lr)
                except Exception as e:
                    spawn_events.append((step, f"AXIS6 spawn FAILED: {e}"))
            step += 1
        if verbose:
            print(f"    epoch {epoch+1}/{n_epochs}: loss={last_loss:.3f} "
                  f"acc={last_acc:.3f} L1={L1.n_nodes} spawns={n_spawns}")

    engagement_frac = (
        engagement_sum / max(1, engagement_count)
        if engagement_count > 0 else None
    )
    return TaskResult(
        name=task.name,
        engagement_frac=engagement_frac,
        train_loss_final=last_loss,
        train_acc_final=last_acc,
        n_spawns=n_spawns,
        spawn_events=spawn_events,
    )


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_full(net: TrioronNetwork, task: TaskView) -> float:
    """Full-softmax accuracy across all 100 classes."""
    n = task.x_test.shape[0]
    correct = 0
    for start in range(0, n, 512):
        end = min(start + 512, n)
        x_in = normalize_only(task.x_test[start:end])
        logits = _forward(net, x_in)
        preds = logits.argmax(dim=-1)
        correct += int((preds == task.y_test[start:end]).sum().item())
    return correct / max(1, n)


@torch.no_grad()
def eval_task_aware(net: TrioronNetwork, task: TaskView) -> float:
    """Task-aware accuracy: argmax restricted to task's classes."""
    n = task.x_test.shape[0]
    cls_idx = torch.tensor(task.classes, dtype=torch.long)
    correct = 0
    for start in range(0, n, 512):
        end = min(start + 512, n)
        x_in = normalize_only(task.x_test[start:end])
        logits = _forward(net, x_in)
        logits_task = logits[:, cls_idx]
        preds_local = logits_task.argmax(dim=-1)
        preds_global = cls_idx[preds_local]
        correct += int((preds_global == task.y_test[start:end]).sum().item())
    return correct / max(1, n)


# ----------------------------------------------------------------------
# Arm runner
# ----------------------------------------------------------------------
ARM_CONFIGS = {
    "a": dict(
        label="v1.0 random Gaussian",
        builder=build_arm_a_random_gaussian,
        manifold=True, credit=True, axis6=False, conv_emergence=False,
    ),
    "b": dict(
        label="v1.0 + LCN",
        builder=build_arm_b_lcn,
        manifold=True, credit=True, axis6=False, conv_emergence=False,
    ),
    "c": dict(
        label="v2.0 static oracle",
        builder=build_arm_c_oracle,
        manifold=True, credit=True, axis6=False, conv_emergence=False,
    ),
    "d": dict(
        label="v2.0 self-arrange",
        builder=build_arm_d_self_arrange,
        manifold=True, credit=True, axis6=True, conv_emergence=False,
    ),
}


def run_arm(
    arm_key: str,
    seed: int,
    tasks: List[TaskView],
    n_epochs: int,
    batch_size: int,
    lr: float,
    credit_thr: float,
    manifold_batch: int = 256,
    manifold_loss_weight: float = 1.0,
    spawn_cap_per_phase: int = 16,
    spawn_cooldown_steps: int = 50,
    spawn_trigger: str = "frustrated",
    verbose: bool = False,
) -> dict:
    cfg = ARM_CONFIGS[arm_key]
    label = cfg["label"]
    if verbose:
        print(f"\n=== arm {arm_key}: {label}  seed={seed} ===")
    net = cfg["builder"](seed)
    cell_credit = (
        torch.zeros(net.layers[1].n_nodes, dtype=torch.bool)
        if cfg["credit"] else None
    )
    manifold = (
        ManifoldStore(n_l0=net.layers[0].n_nodes)
        if cfg["manifold"] else None
    )

    K = len(tasks)
    retention_full: List[List[float]] = []
    retention_task: List[List[float]] = []
    seen_classes: List[int] = []
    all_events: List[Tuple[str, int, str]] = []
    t0 = time.time()

    for ti, task in enumerate(tasks):
        for c in task.classes:
            if c not in seen_classes:
                seen_classes.append(c)
        if verbose:
            print(f"  [phase {ti+1}/{K}: {task.name}] "
                  f"L1.H={net.layers[1].n_nodes} "
                  f"frozen={int(cell_credit.sum().item()) if cell_credit is not None else 0}")
        res = train_one_task(
            net=net, task=task, seen_classes=seen_classes,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            cell_credit=cell_credit, manifold=manifold,
            enable_axis6=cfg["axis6"],
            enable_conv_emergence=cfg.get("conv_emergence", False),
            enable_l0_plastic=cfg.get("l0_plastic", False),
            spawn_cap_per_phase=spawn_cap_per_phase,
            spawn_cooldown_steps=spawn_cooldown_steps,
            spawn_trigger=spawn_trigger,
            manifold_batch=manifold_batch,
            manifold_loss_weight=manifold_loss_weight,
            verbose=verbose,
        )
        for step, msg in res.spawn_events:
            all_events.append((task.name, step, msg))

        # Eval on every task seen so far + future tasks (full=zero on future).
        row_full = [eval_full(net, tasks[k]) for k in range(K)]
        row_task = [eval_task_aware(net, tasks[k]) for k in range(K)]
        retention_full.append(row_full)
        retention_task.append(row_task)
        if verbose:
            print(f"    full(this)={row_full[ti]:.3f} "
                  f"task(this)={row_task[ti]:.3f} "
                  f"spawns={res.n_spawns}")

        # Credit update: cells with engagement > thr get frozen.
        if cell_credit is not None and res.engagement_frac is not None:
            L1 = net.layers[1]
            ef = res.engagement_frac
            if cell_credit.numel() < L1.n_nodes:
                cell_credit = torch.cat([
                    cell_credit,
                    torch.zeros(L1.n_nodes - cell_credit.numel(), dtype=torch.bool),
                ])
            if ef.numel() < L1.n_nodes:
                ef = torch.cat([ef, torch.zeros(L1.n_nodes - ef.numel())])
            earned = ef > credit_thr
            newly = earned & (~cell_credit)
            cell_credit = cell_credit | earned
            if verbose:
                print(f"    [credit] +{int(newly.sum().item())} new; "
                      f"total frozen = {int(cell_credit.sum().item())}/{cell_credit.numel()}")

        # Snapshot manifold for the task's classes (after credit).
        if manifold is not None:
            xall, yall = task.x_train, task.y_train
            # Apply normalize-only (no augmentation) for stable per-class stats.
            x_in = normalize_only(xall)
            manifold.store_codes(net, x_in, yall, task.classes, batch_size=512)

            # Head re-settle: re-calibrate head's per-class biases against
            # the accumulated manifold. Per [[absorption_sentinel_n12_collapse]]
            # this is the documented manifold-prior mechanism — "head-settle
            # then does 100% of the recovery." Without this the head's
            # per-class biases drift toward the latest task and prior
            # classes go to zero on full-softmax.
            best_trial, best_score = settle_head_with_retry(
                net, manifold, seen_classes=seen_classes,
                n_attempts=5,
                head_logits_fn=lambda h, w: cosine_logits(
                    h, w, temperature=COSINE_TEMP,
                ),
                settle_kwargs=dict(
                    n_steps=200, batch_size=64, lr=0.01,
                ),
            )
            if verbose:
                print(f"    [head-settle] best_trial={best_trial} "
                      f"score={best_score:.3f}")

            # Re-eval after head-settle to capture the recovered retention.
            row_full = [eval_full(net, tasks[k]) for k in range(K)]
            row_task = [eval_task_aware(net, tasks[k]) for k in range(K)]
            retention_full[-1] = row_full
            retention_task[-1] = row_task
            if verbose:
                print(f"    [post-settle] full(this)={row_full[ti]:.3f} "
                      f"task(this)={row_task[ti]:.3f}")

    elapsed = time.time() - t0
    final_full = retention_full[-1] if retention_full else []
    final_task = retention_task[-1] if retention_task else []
    return {
        "arm": arm_key,
        "label": label,
        "seed": seed,
        "elapsed_s": elapsed,
        "n_l1_final": net.layers[1].n_nodes,
        "n_spawns": sum(
            1 for _, _, m in all_events if m.startswith("AXIS6 spawn:")
        ),
        "retention_full": retention_full,
        "retention_task": retention_task,
        "final_full": final_full,
        "final_task": final_task,
        "mean_final_full": sum(final_full) / max(1, len(final_full)),
        "mean_final_task": sum(final_task) / max(1, len(final_task)),
        "events": all_events,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def mean_std(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / max(1, n - 1)
    return m, v ** 0.5


def main() -> int:
    smoke = os.environ.get("CIFAR_SMOKE", "0") == "1"
    n_seeds = int(os.environ.get("CIFAR_N_SEEDS", "1" if smoke else "5"))
    n_epochs = int(os.environ.get("CIFAR_EPOCHS", "4" if smoke else "8"))
    batch_size = int(os.environ.get("CIFAR_BATCH", "256"))
    lr = float(os.environ.get("CIFAR_LR", "0.01"))
    credit_thr = float(os.environ.get("CIFAR_CREDIT_THR", "0.3"))
    manifold_batch = int(os.environ.get("CIFAR_MANIFOLD_BATCH", "256"))
    manifold_loss_weight = float(os.environ.get("CIFAR_MANIFOLD_W", "2.0"))
    arms_str = os.environ.get("CIFAR_ARMS", "a,b,c,d")
    arms = [a.strip() for a in arms_str.split(",") if a.strip()]
    n_tasks = int(os.environ.get("CIFAR_N_TASKS", "10"))
    per_task = int(os.environ.get("CIFAR_PER_TASK", "10"))
    spawn_cap = int(os.environ.get("CIFAR_SPAWN_CAP", "16"))
    spawn_cool = int(os.environ.get("CIFAR_SPAWN_COOLDOWN", "50"))
    spawn_trigger = os.environ.get("CIFAR_SPAWN_TRIGGER", "frustrated")
    if spawn_trigger not in ("frustrated", "always"):
        raise ValueError(
            f"CIFAR_SPAWN_TRIGGER must be 'frustrated' or 'always', "
            f"got {spawn_trigger!r}"
        )

    print(f"bench_2_0_cifar_continual: arms={arms} seeds={n_seeds} "
          f"epochs={n_epochs} tasks={n_tasks}x{per_task} batch={batch_size} "
          f"lr={lr} smoke={smoke} "
          f"spawn(cap={spawn_cap},cool={spawn_cool},trig={spawn_trigger})")

    print("loading CIFAR-100 + building tasks (one-time)...")
    tasks = build_cifar_tasks(n_tasks=n_tasks, per_task=per_task)
    print(f"  built {len(tasks)} tasks; first task class range "
          f"{tasks[0].classes[0]}..{tasks[0].classes[-1]}, "
          f"train={tasks[0].x_train.shape[0]}")

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "bench_2_0_cifar_continual.csv"

    results: List[dict] = []
    for seed in range(n_seeds):
        for arm in arms:
            r = run_arm(
                arm_key=arm, seed=seed, tasks=tasks,
                n_epochs=n_epochs, batch_size=batch_size, lr=lr,
                credit_thr=credit_thr,
                manifold_batch=manifold_batch,
                manifold_loss_weight=manifold_loss_weight,
                spawn_cap_per_phase=spawn_cap,
                spawn_cooldown_steps=spawn_cool,
                spawn_trigger=spawn_trigger,
                verbose=smoke,
            )
            results.append(r)
            print(f"[seed {seed} arm {arm} {ARM_CONFIGS[arm]['label']}] "
                  f"full={r['mean_final_full']:.4f} "
                  f"task={r['mean_final_task']:.4f} "
                  f"L1={r['n_l1_final']} spawns={r['n_spawns']} "
                  f"t={r['elapsed_s']:.0f}s")

    # Per-arm aggregate
    print("\n=== aggregate (mean ± std across seeds) ===")
    by_arm: Dict[str, List[dict]] = {}
    for r in results:
        by_arm.setdefault(r["arm"], []).append(r)
    for arm in arms:
        rs = by_arm.get(arm, [])
        if not rs:
            continue
        full_m, full_s = mean_std([r["mean_final_full"] for r in rs])
        task_m, task_s = mean_std([r["mean_final_task"] for r in rs])
        l1_m, _ = mean_std([float(r["n_l1_final"]) for r in rs])
        spawn_m, _ = mean_std([float(r["n_spawns"]) for r in rs])
        print(f"  arm {arm} {ARM_CONFIGS[arm]['label']:>30s}  "
              f"full={full_m:.4f} ± {full_s:.4f}  "
              f"task={task_m:.4f} ± {task_s:.4f}  "
              f"L1={l1_m:.1f}  spawns={spawn_m:.1f}")

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "arm", "label", "seed", "elapsed_s", "n_l1_final", "n_spawns",
            "mean_final_full", "mean_final_task",
        ] + [f"final_full_{i}" for i in range(n_tasks)]
          + [f"final_task_{i}" for i in range(n_tasks)])
        for r in results:
            w.writerow([
                r["arm"], r["label"], r["seed"], f"{r['elapsed_s']:.1f}",
                r["n_l1_final"], r["n_spawns"],
                f"{r['mean_final_full']:.4f}",
                f"{r['mean_final_task']:.4f}",
            ] + [f"{x:.4f}" for x in r["final_full"]]
              + [f"{x:.4f}" for x in r["final_task"]])
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
