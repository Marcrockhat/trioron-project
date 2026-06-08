"""Saccadic input adapter — substrate-level smoke probe on 10-class CIFAR-100.

Tests whether overlapping position-augmented patches + LCN-locked L1 +
Axis 6 spawn produces a "convolutional layer" via substrate growth.

Adapter:
    overlapping_patches(images, patch=8, stride=4) →
        (N, P, 3·patch² + 2) where each fixation is
        [flat_patch_pixels || x_center || y_center] and
        P = ((H - patch)//stride + 1)².

Network (real TrioronNetwork, three layers):
    L0  (fan_in = 3·patch² + 2, n_nodes = l0_dim,  ReLU,  no LCN)
    L1  (fan_in = l0_dim,        n_nodes = h1_init, ReLU,
         LCN hard top-K + Axis 6 enabled — starts UNDERPOPULATED so
         spawn events fill it in under signal pressure)
    HEAD (fan_in = h1_init,      n_nodes = n_classes, linear)

When Axis 6 fires on L1, network.axis6_spawn extends HEAD's fan_in by
one — that's the "trioron generates a convolutional layer" mechanism:
new L1 cells are positioned near a hot parent, inherit the LCN locality
convention via extend_lcn_mask, and the HEAD picks up a new input column.
After enough spawns L1 is a populated locally-connected substrate.

Three arms (single seed, ~10 min target on CPU):

    baseline_classical             classical sense (33-d), flat dense
                                   TrioronNetwork, no LCN, no Axis 6.
    patches_no_position            overlapping patches without (x,y),
                                   no LCN, no Axis 6.
    patches_position_lcn_axis6     overlapping patches + (x,y) + LCN
                                   hard-topK on L1 + Axis 6 spawn loop.

Decision signals reported per arm:

    final_acc                 10-class test accuracy.
    n_L1_spawns               Axis 6 spawn events on L1 (0 in non-axis6 arms).
    locality_metric           in-window |W| mass / total |W| mass on L1
                              (n/a for arms without LCN).
    spatial_specialization    correlation between each L1 cell's
                              cell_position and its activation-weighted
                              fixation centroid on the test set.
                              > 0 = cells learned position-locality.

If the axis6 arm shows n_L1_spawns > 0, locality_metric > ~0.5, and
spatial_specialization > 0 — the substrate mechanism works at smoke scale.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from trioron.network import TrioronNetwork
from trioron.senses import apply_sense, Standardizer
from trioron.spatial import grid_positions_2d, locality_metric
from experiments.cifar.datasets import DEFAULT_DATA_ROOT, load_cifar100


# ----------------------------------------------------------------------
# Saccadic adapter — strided overlapping patches with (x, y) tail
# ----------------------------------------------------------------------


def overlapping_patches(
    images: torch.Tensor,
    patch: int,
    stride: int,
) -> torch.Tensor:
    """(N, C, H, W) → (N, P, C·patch²) — strided unfold, overlap when stride<patch."""
    if patch < 1 or stride < 1:
        raise ValueError(f"patch and stride must be >= 1, got ({patch}, {stride})")
    N, C, H, W = images.shape
    x = images.unfold(2, patch, stride).unfold(3, patch, stride)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    gh, gw = x.shape[1], x.shape[2]
    return x.view(N, gh * gw, C * patch * patch)


def fixation_positions(H: int, W: int, patch: int, stride: int) -> torch.Tensor:
    """(P, 2) patch-center positions on [0, 1]², row-major matching overlapping_patches."""
    gh = (H - patch) // stride + 1
    gw = (W - patch) // stride + 1
    cy = (torch.arange(gh).float() * stride + patch / 2.0) / H
    cx = (torch.arange(gw).float() * stride + patch / 2.0) / W
    rows, cols = torch.meshgrid(cy, cx, indexing="ij")
    return torch.stack([cols, rows], dim=-1).reshape(-1, 2)


def saccadic_decompose(
    images: torch.Tensor,
    patch: int,
    stride: int,
    append_pos: bool,
) -> torch.Tensor:
    """(N, C, H, W) → (N, P, C·patch² [+ 2])."""
    patches = overlapping_patches(images, patch, stride)
    if not append_pos:
        return patches
    _, _, H, W = images.shape
    pos = fixation_positions(H, W, patch, stride)
    N, P, _ = patches.shape
    return torch.cat([patches, pos.unsqueeze(0).expand(N, -1, -1)], dim=-1)


# ----------------------------------------------------------------------
# Substrate organism — real TrioronNetwork
# ----------------------------------------------------------------------


class SubstrateOrganism(nn.Module):
    """L0 → L1 (LCN + Axis 6) → HEAD. Forward folds (B,P,D)→(B·P,D)
    and mean-pools logits over P. For non-fixation arms pass P=1."""

    def __init__(
        self,
        *,
        input_dim: int,
        l0_dim: int,
        h1_init: int,
        n_classes: int,
        seed: int,
        lcn_on: bool,
        axis6_on: bool,
        lcn_k: int = 8,
        field_sigma: float = 0.15,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.lcn_on = lcn_on
        self.axis6_on = axis6_on
        self.lcn_k = int(lcn_k)
        self.field_sigma = float(field_sigma)

        self.network = TrioronNetwork([
            (input_dim, l0_dim,  "relu"),
            (l0_dim,    h1_init, "relu"),
            (h1_init,   n_classes, "linear"),
        ])

        self._seed_grid_positions(self.network.layers[0], l0_dim)
        self._seed_grid_positions(self.network.layers[1], h1_init)

        if lcn_on:
            in_pos = self.network.layers[0].cell_position[:, :2].detach().clone()
            self.network.layers[1].enable_lcn(
                in_pos, mode="hard", k=self.lcn_k, apply_to_weights=True,
            )
        if axis6_on:
            self.network.layers[1].enable_axis6_field(field_sigma=self.field_sigma)

        self._spawn_count = 0

    @staticmethod
    def _seed_grid_positions(layer, n_nodes: int) -> None:
        side = max(2, int(n_nodes ** 0.5 + 0.999))
        pos2d = grid_positions_2d(side, side)[:n_nodes]
        with torch.no_grad():
            layer.cell_position[:, :2] = pos2d
            if layer.cell_position.shape[1] >= 3:
                layer.cell_position[:, 2] = 0.0
        layer._field_kernel = None

    def forward(self, x_fix: torch.Tensor) -> torch.Tensor:
        B, P, D = x_fix.shape
        z = self.network(x_fix.view(B * P, D))
        return z.view(B, P, -1).mean(dim=1)

    def encode_l1(self, x_fix: torch.Tensor) -> torch.Tensor:
        B, P, D = x_fix.shape
        flat = x_fix.view(B * P, D)
        z0 = self.network.layers[0](flat)
        z1 = self.network.layers[1](z0)
        return z1.view(B, P, -1)


# ----------------------------------------------------------------------
# Axis 6 spawn step
# ----------------------------------------------------------------------


def maybe_spawn(
    model: SubstrateOrganism,
    *,
    spawn_cap: int,
    stress_floor: float,
    b_threshold: float,
) -> bool:
    """Field-conditional spawn on L1. Returns True on spawn event.
    Caller MUST rebuild optimizer after a True return."""
    if not model.axis6_on:
        return False
    if model._spawn_count >= spawn_cap:
        return False
    L1 = model.network.layers[1]
    L1.update_internal_stress()
    L1.update_epi_field(dt=0.1, stress_tolerance=0.0)
    cand = L1.field_conditional_growth_candidate(
        mode="absolute", stress_floor=stress_floor, b_threshold=b_threshold,
    )
    if cand is None:
        return False
    model.network.axis6_spawn(layer_idx=1, candidate_idx=cand, position_jitter=0.1)
    model._spawn_count += 1
    return True


# ----------------------------------------------------------------------
# Per-arm training loop
# ----------------------------------------------------------------------


def train_arm(
    label: str,
    x_train: torch.Tensor, y_train: torch.Tensor,
    x_test: torch.Tensor, y_test: torch.Tensor,
    *,
    input_dim: int, l0_dim: int, h1_init: int, n_classes: int,
    n_steps: int, lr: float, batch: int, seed: int,
    lcn_on: bool, axis6_on: bool,
    is_fixation_input: bool,
    fix_positions: torch.Tensor | None = None,
    spawn_cap: int = 24,
    stress_floor: float = 1e-3,
    cooldown_steps: int = 5,
    device: str = "cpu",
):
    torch.manual_seed(seed)

    if not is_fixation_input:
        x_train = x_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)

    model = SubstrateOrganism(
        input_dim=input_dim, l0_dim=l0_dim, h1_init=h1_init,
        n_classes=n_classes, seed=seed,
        lcn_on=lcn_on, axis6_on=axis6_on,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    N = x_train.shape[0]
    last_spawn_step = -cooldown_steps
    for step in range(n_steps):
        idx = torch.randint(0, N, (batch,))
        xb = x_train[idx].to(device)
        yb = y_train[idx].to(device)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if lcn_on:
            model.network.layers[1].reapply_lcn_mask()

        if axis6_on and (step - last_spawn_step) >= cooldown_steps:
            if maybe_spawn(
                model,
                spawn_cap=spawn_cap,
                stress_floor=stress_floor,
                b_threshold=1.0,
            ):
                last_spawn_step = step
                opt = torch.optim.Adam(model.parameters(), lr=lr)

        if (step + 1) % max(n_steps // 4, 1) == 0:
            spawns = model._spawn_count if axis6_on else 0
            print(
                f"  [{label}] step {step+1}/{n_steps}  "
                f"ce={loss.item():.4f}  spawns={spawns}"
            )

    elapsed = time.time() - t0

    # --- diagnostics --------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits = model(x_test.to(device))
        pred = logits.argmax(dim=-1).cpu()
        acc = (pred == y_test).float().mean().item()

        L1 = model.network.layers[1]
        if lcn_on and hasattr(L1, "W_lcn_mask"):
            loc = locality_metric(L1.W, L1.W_lcn_mask)
        else:
            loc = float("nan")

        if is_fixation_input and fix_positions is not None:
            h_test = model.encode_l1(x_test.to(device))         # (N, P, H1)
            act = h_test.sum(dim=0)                              # (P, H1)
            weight = act.sum(dim=0).clamp_min(1e-6)              # (H1,)
            patch_pos = fix_positions.to(device)
            cell_centroid = (act.t() @ patch_pos) / weight.unsqueeze(-1)
            pos = L1.cell_position[:, :2]

            def _corr(a, b):
                a = a - a.mean(); b = b - b.mean()
                return (a * b).sum() / (a.norm() * b.norm() + 1e-6)

            corr_x = _corr(pos[:, 0], cell_centroid[:, 0]).item()
            corr_y = _corr(pos[:, 1], cell_centroid[:, 1]).item()
            spatial_specialization = 0.5 * (corr_x + corr_y)
        else:
            spatial_specialization = float("nan")

    return {
        "label": label,
        "acc": acc,
        "n_L1_spawns": model._spawn_count,
        "h1_final": L1.n_nodes,
        "locality": loc,
        "spatial_specialization": spatial_specialization,
        "elapsed_s": elapsed,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--n-train-per-class", type=int, default=400)
    parser.add_argument("--n-test-per-class",  type=int, default=80)
    parser.add_argument("--patch", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--h1-init", type=int, default=16)
    parser.add_argument("--l0-dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spawn-cap", type=int, default=24)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("probe_saccadic_smoke — substrate-level adapter + LCN + Axis 6")
    print("=" * 78)
    print(f"n_classes={args.n_classes}  patch={args.patch}  stride={args.stride}")
    print(f"l0={args.l0_dim}  h1_init={args.h1_init}  spawn_cap={args.spawn_cap}")
    print(f"steps={args.steps}  batch={args.batch}  lr={args.lr}")
    print()

    torch.manual_seed(args.seed)

    train_imgs, train_labs = load_cifar100(args.data_root, train=True)
    test_imgs,  test_labs  = load_cifar100(args.data_root, train=False)
    keep = list(range(args.n_classes))
    relabel = {c: i for i, c in enumerate(keep)}

    def _subset(imgs, labs, per_class):
        rows = []
        for c in keep:
            mask = (labs == c).nonzero(as_tuple=True)[0]
            rows.append(mask[:per_class])
        idx = torch.cat(rows)
        idx = idx[torch.randperm(
            idx.numel(), generator=torch.Generator().manual_seed(args.seed),
        )]
        return imgs[idx], torch.tensor(
            [relabel[c.item()] for c in labs[idx]], dtype=torch.long,
        )

    Xtr_img, ytr = _subset(train_imgs, train_labs, args.n_train_per_class)
    Xte_img, yte = _subset(test_imgs,  test_labs,  args.n_test_per_class)
    print(f"train: {Xtr_img.shape[0]}  test: {Xte_img.shape[0]}")
    _, _, H, W = Xtr_img.shape
    fix_pos = fixation_positions(H, W, args.patch, args.stride)
    print(f"fixations per image: {fix_pos.shape[0]}  (patch={args.patch}, stride={args.stride})")
    print()

    # Arm 1: classical sense (33-d), flat dense substrate.
    Xtr_cls = apply_sense("classical", Xtr_img)
    Xte_cls = apply_sense("classical", Xte_img)
    std = Standardizer.fit(Xtr_cls)
    Xtr_cls = std.transform(Xtr_cls).contiguous()
    Xte_cls = std.transform(Xte_cls).contiguous()
    cls_dim = Xtr_cls.shape[-1]

    # Arm 2: overlapping patches, no (x, y).
    Xtr_np = saccadic_decompose(Xtr_img, args.patch, args.stride, append_pos=False)
    Xte_np = saccadic_decompose(Xte_img, args.patch, args.stride, append_pos=False)
    mu_np = Xtr_np.mean(dim=(0, 1), keepdim=True)
    sd_np = Xtr_np.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
    Xtr_np = (Xtr_np - mu_np) / sd_np
    Xte_np = (Xte_np - mu_np) / sd_np
    np_dim = Xtr_np.shape[-1]

    # Arm 3: overlapping patches + (x, y) + LCN + Axis 6.
    Xtr_wp = saccadic_decompose(Xtr_img, args.patch, args.stride, append_pos=True)
    Xte_wp = saccadic_decompose(Xte_img, args.patch, args.stride, append_pos=True)
    pix_dim = np_dim
    mu_wp = Xtr_wp[..., :pix_dim].mean(dim=(0, 1), keepdim=True)
    sd_wp = Xtr_wp[..., :pix_dim].std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
    Xtr_wp = torch.cat([
        (Xtr_wp[..., :pix_dim] - mu_wp) / sd_wp,
        Xtr_wp[..., pix_dim:],
    ], dim=-1)
    Xte_wp = torch.cat([
        (Xte_wp[..., :pix_dim] - mu_wp) / sd_wp,
        Xte_wp[..., pix_dim:],
    ], dim=-1)
    wp_dim = Xtr_wp.shape[-1]

    arms = []

    print("\n>>> arm 1: baseline_classical (33-d, dense substrate)")
    arms.append(train_arm(
        "baseline_classical",
        Xtr_cls, ytr, Xte_cls, yte,
        input_dim=cls_dim, l0_dim=args.l0_dim, h1_init=args.h1_init,
        n_classes=args.n_classes, n_steps=args.steps, lr=args.lr,
        batch=args.batch, seed=args.seed,
        lcn_on=False, axis6_on=False, is_fixation_input=False,
    ))

    print(f"\n>>> arm 2: patches_no_position (patch={args.patch}, stride={args.stride})")
    arms.append(train_arm(
        "patches_no_position",
        Xtr_np, ytr, Xte_np, yte,
        input_dim=np_dim, l0_dim=args.l0_dim, h1_init=args.h1_init,
        n_classes=args.n_classes, n_steps=args.steps, lr=args.lr,
        batch=args.batch, seed=args.seed,
        lcn_on=False, axis6_on=False, is_fixation_input=True,
        fix_positions=fix_pos,
    ))

    print(f"\n>>> arm 3: patches_position_lcn_axis6 (LCN k=8, axis6 cap={args.spawn_cap})")
    arms.append(train_arm(
        "patches_position_lcn_axis6",
        Xtr_wp, ytr, Xte_wp, yte,
        input_dim=wp_dim, l0_dim=args.l0_dim, h1_init=args.h1_init,
        n_classes=args.n_classes, n_steps=args.steps, lr=args.lr,
        batch=args.batch, seed=args.seed,
        lcn_on=True, axis6_on=True, is_fixation_input=True,
        fix_positions=fix_pos, spawn_cap=args.spawn_cap,
    ))

    print("\n" + "=" * 78)
    print("smoke probe — summary")
    print("=" * 78)
    header = (
        f"  {'arm':<30} {'acc':>6} {'spawns':>7} {'h1':>5} "
        f"{'locality':>9} {'spatial':>9} {'elapsed_s':>10}"
    )
    print(header)
    for r in arms:
        loc = (f"{r['locality']:>9.3f}"
               if r["locality"] == r["locality"] else f"{'n/a':>9}")
        spec = (f"{r['spatial_specialization']:>9.3f}"
                if r["spatial_specialization"] == r["spatial_specialization"]
                else f"{'n/a':>9}")
        print(
            f"  {r['label']:<30} {r['acc']:>6.3f} "
            f"{r['n_L1_spawns']:>7d} {r['h1_final']:>5d} "
            f"{loc} {spec} {r['elapsed_s']:>10.1f}"
        )
    print()
    print("Read: axis6 arm should show n_L1_spawns > 0 (substrate grew),")
    print("      locality > ~0.5 (LCN held under training), and")
    print("      spatial_specialization > 0 (cells learned position-locality).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
