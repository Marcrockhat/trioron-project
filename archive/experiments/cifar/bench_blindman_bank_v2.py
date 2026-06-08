"""bench_blindman_bank_v2.py — v1 bank + CIFAR augmentation + extensible
decisive trioron on top.

Two-phase pipeline:

  Phase A  Train positional bank (same architecture as v1) with CIFAR-style
           augmentation applied to images BEFORE patch split: random crop
           (pad 4), horizontal flip, light brightness/contrast jitter.

  Phase B  Freeze bank. Train an extensible decisive trioron whose input
           is the flattened per-branch class logits + per-branch saliency
           predictions, (16·100 + 16) = 1616-d → 100-way classification.
           This replaces v1's hand-coded gating (uniform / archive / saliency
           / combined) with a learned combiner.

Eval reports:
  - Phase A (bank uniform, with augmentation)         — sanity vs v1's 0.2402
  - Phase B (decisive trioron on bank features)       — the v2 headline

Decisive trioron is extensible by construction: TrioronNetwork supports
class-extension via head growth (add new CIFAR-100 classes later by
training only the new rows + absorbing into the head). v2 doesn't
exercise extension; v3 will.

Reuses v1's BlindmanBank, patch_split, sobel_saliency, and standardizers.

Run:
  python3 -m experiments.cifar.bench_blindman_bank_v2 \\
      > outputs/bench_blindman_bank_v2_run1.log 2>&1
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.cifar.datasets import (
    load_cifar100, cifar100_fine_to_coarse, DEFAULT_DATA_ROOT,
)
from experiments.cifar.bench_blindman_bank import (
    patch_split, sobel_saliency,
    PerPatchStandardizer, ScalarStandardizer,
    BlindmanBank,
)
from trioron.network import TrioronNetwork


N_CLASSES = 100
IMG_HW = 32


# ---------------------------------------------------------------------
# Augmentation (tensor-only, batched, no torchvision.transforms)
# ---------------------------------------------------------------------


def cifar_augment(images: torch.Tensor,
                  *, pad: int = 4,
                  flip_prob: float = 0.5,
                  brightness: float = 0.2,
                  contrast: float = 0.2) -> torch.Tensor:
    """Per-batch CIFAR augmentation. images: (N, 3, 32, 32) → (N, 3, 32, 32).

    Random crop with reflect padding, random horizontal flip, multiplicative
    brightness, mean-preserving contrast jitter. All batched.
    """
    N, C, H, W = images.shape
    device = images.device

    # Random crop (reflect-pad then crop a random 32x32 window per image).
    padded = F.pad(images, (pad, pad, pad, pad), mode="reflect")
    # Different offset per image. Build a (N, 2) index then gather.
    h_off = torch.randint(0, 2 * pad + 1, (N,), device=device)
    w_off = torch.randint(0, 2 * pad + 1, (N,), device=device)
    # Vectorized crop via advanced indexing.
    h_idx = h_off.view(N, 1, 1, 1) + torch.arange(H, device=device).view(1, 1, H, 1)
    w_idx = w_off.view(N, 1, 1, 1) + torch.arange(W, device=device).view(1, 1, 1, W)
    h_idx = h_idx.expand(N, C, H, W)
    w_idx = w_idx.expand(N, C, H, W)
    n_idx = torch.arange(N, device=device).view(N, 1, 1, 1).expand(N, C, H, W)
    c_idx = torch.arange(C, device=device).view(1, C, 1, 1).expand(N, C, H, W)
    cropped = padded[n_idx, c_idx, h_idx, w_idx]

    # Random horizontal flip.
    flip_mask = torch.rand(N, device=device) < flip_prob
    if flip_mask.any():
        flipped = cropped.flip(-1)
        cropped = torch.where(flip_mask.view(N, 1, 1, 1), flipped, cropped)

    # Brightness jitter — multiply by ~Unif(1-b, 1+b) per image.
    if brightness > 0:
        b_factor = (1.0 + (torch.rand(N, device=device) * 2 - 1) * brightness)
        cropped = cropped * b_factor.view(N, 1, 1, 1)

    # Contrast jitter — scale around per-image mean.
    if contrast > 0:
        c_factor = (1.0 + (torch.rand(N, device=device) * 2 - 1) * contrast)
        mu = cropped.mean(dim=(1, 2, 3), keepdim=True)
        cropped = mu + c_factor.view(N, 1, 1, 1) * (cropped - mu)

    return cropped.clamp(0.0, 1.0)


# ---------------------------------------------------------------------
# Phase A — bank training with augmentation
# ---------------------------------------------------------------------


def train_bank_with_aug(
    bank: BlindmanBank,
    Xtr_img: torch.Tensor,           # (N, 3, 32, 32) raw images on CPU
    ytr: torch.Tensor,
    pstd: PerPatchStandardizer,
    sstd: ScalarStandardizer,
    *,
    grid: int,
    epochs: int, batch_size: int, lr: float,
    saliency_weight: float, device: str,
) -> None:
    """Bank training with per-batch CIFAR augmentation. Patches and Sobel
    targets are computed live each batch on the augmented images."""
    bank.to(device)
    pstd_mu = pstd.mu.to(device)
    pstd_sigma = pstd.sigma.to(device)
    sstd_mu = sstd.mu.to(device)
    sstd_sigma = sstd.sigma.to(device)

    opt = torch.optim.Adam(bank.parameters(), lr=lr)
    N = Xtr_img.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(N)
        total = 0.0
        n_batches = 0
        t0 = time.time()
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            imgs = Xtr_img[idx].to(device)
            y = ytr[idx].to(device)

            imgs = cifar_augment(imgs)
            patches = patch_split(imgs, grid)               # (B, P, patch_dim)
            sal = sobel_saliency(imgs, grid)                # (B, P)
            patches_std = (patches - pstd_mu) / pstd_sigma
            sal_std = (sal - sstd_mu) / sstd_sigma

            z0 = bank.encode(patches_std)
            logits, sal_pred = bank.forward_per_branch(z0)
            B, P, C = logits.shape
            ce = F.cross_entropy(logits.view(B * P, C),
                                 y.repeat_interleave(P))
            mse = F.mse_loss(sal_pred, sal_std)
            loss = ce + saliency_weight * mse

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n_batches += 1
        print(f"  [bank] epoch {epoch+1:3d}/{epochs}  "
              f"loss {total/n_batches:.4f}  ({time.time()-t0:.1f}s)",
              flush=True)


# ---------------------------------------------------------------------
# Phase B — extensible decisive trioron on bank features
# ---------------------------------------------------------------------


def build_decisive_trioron(in_dim: int, h: int, n_classes: int) -> TrioronNetwork:
    """Small extensible-by-construction trioron stack: in_dim → h → n_classes.
    No CL machinery exercised in v2 — TrioronNetwork is here so v3 can
    test class-extension via head row growth."""
    return TrioronNetwork([
        (in_dim, h, "relu"),
        (h, n_classes, "linear"),
    ])


def bank_features(
    bank: BlindmanBank,
    imgs: torch.Tensor,                 # (N, 3, 32, 32) on device
    pstd_mu: torch.Tensor, pstd_sigma: torch.Tensor,
    sstd_mu: torch.Tensor, sstd_sigma: torch.Tensor,
    grid: int,
    features_mode: str = "logits",
) -> torch.Tensor:
    """Forward through frozen bank → flat decisive-head input.

    features_mode:
      "logits"  → (N, P*C + P)              [v2/v2b default]
      "hidden"  → (N, P*h1 + P)              L1 hidden + saliency only
      "both"    → (N, P*C + P + P*h1)        logits + saliency + hidden
    """
    patches = patch_split(imgs, grid)
    sal = sobel_saliency(imgs, grid)
    patches_std = (patches - pstd_mu) / pstd_sigma
    sal_std = (sal - sstd_mu) / sstd_sigma
    z0 = bank.encode(patches_std)                          # (N, P, l0_dim)

    if features_mode == "logits":
        logits, sal_pred = bank.forward_per_branch(z0)
        N = imgs.shape[0]
        return torch.cat([logits.view(N, -1), sal_pred], dim=-1)

    # "hidden" or "both": pull L1 activations alongside the heads.
    logits_list, sals_list, hidden_list = [], [], []
    for i, br in enumerate(bank.branches):
        h_i = F.relu(br.L1(z0[:, i]))                      # (N, h1)
        logits_i = br.class_head(h_i)                      # (N, n_classes)
        sal_i = br.saliency_head(h_i).squeeze(-1)          # (N,)
        logits_list.append(logits_i)
        sals_list.append(sal_i)
        hidden_list.append(h_i)
    logits = torch.stack(logits_list, dim=1)               # (N, P, C)
    sals = torch.stack(sals_list, dim=1)                   # (N, P)
    hidden = torch.stack(hidden_list, dim=1)               # (N, P, h1)
    N = imgs.shape[0]
    if features_mode == "hidden":
        return torch.cat([hidden.view(N, -1), sals], dim=-1)
    if features_mode == "both":
        return torch.cat([logits.view(N, -1), sals, hidden.view(N, -1)], dim=-1)
    raise ValueError(f"unknown features_mode {features_mode!r}")


def train_decisive(
    decisive: TrioronNetwork,
    bank: BlindmanBank,
    Xtr_img: torch.Tensor, ytr: torch.Tensor,
    pstd: PerPatchStandardizer, sstd: ScalarStandardizer,
    *,
    grid: int, epochs: int, batch_size: int, lr: float,
    use_augmentation: bool, device: str,
    features_mode: str = "logits",
) -> None:
    """Phase B: bank is frozen, only decisive head trains. Augmentation is
    optional — turning it on means the decisive head sees a wider feature
    distribution, which usually helps."""
    decisive.to(device)
    bank.to(device).eval()
    for p in bank.parameters():
        p.requires_grad_(False)

    pstd_mu = pstd.mu.to(device)
    pstd_sigma = pstd.sigma.to(device)
    sstd_mu = sstd.mu.to(device)
    sstd_sigma = sstd.sigma.to(device)

    opt = torch.optim.Adam(decisive.parameters(), lr=lr)
    N = Xtr_img.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(N)
        total = 0.0
        n_batches = 0
        t0 = time.time()
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            imgs = Xtr_img[idx].to(device)
            y = ytr[idx].to(device)
            if use_augmentation:
                imgs = cifar_augment(imgs)
            with torch.no_grad():
                feats = bank_features(bank, imgs,
                                      pstd_mu, pstd_sigma,
                                      sstd_mu, sstd_sigma, grid,
                                      features_mode=features_mode)
            logits = decisive(feats)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n_batches += 1
        print(f"  [decisive] epoch {epoch+1:3d}/{epochs}  "
              f"loss {total/n_batches:.4f}  ({time.time()-t0:.1f}s)",
              flush=True)


# ---------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------


def _restricted_argmax(
    logits: torch.Tensor,            # (N, C) on device
    true_fine: torch.Tensor,         # (N,) on device
    fine_to_coarse: torch.Tensor,    # (C,) on device
) -> torch.Tensor:
    """Per-row argmax restricted to the fine classes that share the row's
    true coarse label. Used for CIFAR-100 superclass-restricted task-aware
    accuracy (5-class restriction, chance = 0.20)."""
    N, C = logits.shape
    coarse_per_image = fine_to_coarse[true_fine]                   # (N,)
    mask = fine_to_coarse.unsqueeze(0) == coarse_per_image.unsqueeze(1)  # (N, C)
    masked = logits.masked_fill(~mask, float("-inf"))
    return masked.argmax(dim=-1)


def eval_bank_uniform(
    bank: BlindmanBank,
    Xte_img: torch.Tensor, yte: torch.Tensor,
    pstd: PerPatchStandardizer, sstd: ScalarStandardizer,
    *,
    grid: int, device: str, batch: int = 512,
    fine_to_coarse: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """Uniform 1/P gating, no decisive head. Returns (full_acc, task_acc).
    task_acc requires fine_to_coarse mapping; returns NaN if not provided."""
    bank.to(device).eval()
    pstd_mu = pstd.mu.to(device)
    pstd_sigma = pstd.sigma.to(device)
    sstd_mu = sstd.mu.to(device)
    sstd_sigma = sstd.sigma.to(device)
    f2c_dev = fine_to_coarse.to(device) if fine_to_coarse is not None else None
    correct_full = 0
    correct_task = 0
    N = Xte_img.shape[0]
    with torch.no_grad():
        for i in range(0, N, batch):
            imgs = Xte_img[i:i + batch].to(device)
            y = yte[i:i + batch].to(device)
            patches = patch_split(imgs, grid)
            patches_std = (patches - pstd_mu) / pstd_sigma
            z0 = bank.encode(patches_std)
            logits, _ = bank.forward_per_branch(z0)        # (B, P, C)
            agg = logits.mean(dim=1)                        # (B, C)
            correct_full += int((agg.argmax(-1) == y).sum().item())
            if f2c_dev is not None:
                pred_task = _restricted_argmax(agg, y, f2c_dev)
                correct_task += int((pred_task == y).sum().item())
    full_acc = correct_full / N
    task_acc = correct_task / N if f2c_dev is not None else float("nan")
    return full_acc, task_acc


def eval_decisive(
    decisive: TrioronNetwork, bank: BlindmanBank,
    Xte_img: torch.Tensor, yte: torch.Tensor,
    pstd: PerPatchStandardizer, sstd: ScalarStandardizer,
    *,
    grid: int, device: str, batch: int = 512,
    features_mode: str = "logits",
    fine_to_coarse: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """Returns (full_acc, task_acc). task_acc is NaN if fine_to_coarse is None."""
    decisive.to(device).eval()
    bank.to(device).eval()
    pstd_mu = pstd.mu.to(device)
    pstd_sigma = pstd.sigma.to(device)
    sstd_mu = sstd.mu.to(device)
    sstd_sigma = sstd.sigma.to(device)
    f2c_dev = fine_to_coarse.to(device) if fine_to_coarse is not None else None
    correct_full = 0
    correct_task = 0
    N = Xte_img.shape[0]
    with torch.no_grad():
        for i in range(0, N, batch):
            imgs = Xte_img[i:i + batch].to(device)
            y = yte[i:i + batch].to(device)
            feats = bank_features(bank, imgs,
                                  pstd_mu, pstd_sigma,
                                  sstd_mu, sstd_sigma, grid,
                                  features_mode=features_mode)
            logits = decisive(feats)
            correct_full += int((logits.argmax(-1) == y).sum().item())
            if f2c_dev is not None:
                pred_task = _restricted_argmax(logits, y, f2c_dev)
                correct_task += int((pred_task == y).sum().item())
    full_acc = correct_full / N
    task_acc = correct_task / N if f2c_dev is not None else float("nan")
    return full_acc, task_acc


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=4)
    p.add_argument("--l0-dim", type=int, default=128)
    p.add_argument("--h1", type=int, default=32)
    p.add_argument("--decisive-h", type=int, default=128)
    p.add_argument("--bank-epochs", type=int, default=30)
    p.add_argument("--decisive-epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--saliency-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--decisive-aug", action="store_true",
                   help="apply augmentation during Phase B too (default off)")
    p.add_argument("--features", choices=["logits", "hidden", "both"],
                   default="logits",
                   help="bank features fed to decisive head (default logits, "
                        "matches v2/v2b runs; 'both' adds L1 hidden activations)")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    print("Blindman positional bank v2 — bank+aug + decisive trioron")
    print(f"  grid={args.grid}  patches={args.grid**2}  "
          f"patch_dim={3*(IMG_HW//args.grid)**2}")
    print(f"  l0_dim={args.l0_dim}  h1={args.h1}  decisive_h={args.decisive_h}")
    print(f"  bank_epochs={args.bank_epochs}  decisive_epochs={args.decisive_epochs}  "
          f"batch={args.batch}  lr={args.lr}")
    print(f"  device={args.device}  seed={args.seed}  "
          f"decisive_aug={args.decisive_aug}")

    # ---- data ----
    print("[data] loading CIFAR-100 ...", flush=True)
    Xtr_img, ytr = load_cifar100(args.data_root, train=True)
    Xte_img, yte = load_cifar100(args.data_root, train=False)
    print(f"  train: {tuple(Xtr_img.shape)}  test: {tuple(Xte_img.shape)}")

    # Standardizer fit on un-augmented patches (v1 convention).
    Xtr_patches_clean = patch_split(Xtr_img, args.grid)
    sal_tr_clean = sobel_saliency(Xtr_img, args.grid)
    pstd = PerPatchStandardizer.fit(Xtr_patches_clean)
    sstd = ScalarStandardizer.fit(sal_tr_clean)
    P = Xtr_patches_clean.shape[1]
    patch_dim = Xtr_patches_clean.shape[2]
    del Xtr_patches_clean, sal_tr_clean

    # ---- bank ----
    bank = BlindmanBank(
        n_branches=P, patch_dim=patch_dim,
        l0_dim=args.l0_dim, h1=args.h1, n_classes=N_CLASSES,
        l0_seed=args.seed,
    )
    n_bank = sum(p.numel() for p in bank.parameters() if p.requires_grad)
    print(f"  bank trainable params: {n_bank}  (per branch: {n_bank // P})")

    # ---- Phase A: train bank with augmentation ----
    print("[Phase A] training bank with CIFAR augmentation ...", flush=True)
    train_bank_with_aug(
        bank, Xtr_img, ytr, pstd, sstd,
        grid=args.grid,
        epochs=args.bank_epochs, batch_size=args.batch, lr=args.lr,
        saliency_weight=args.saliency_weight, device=args.device,
    )

    # CIFAR-100 superclass-restricted task-aware metric (5-class restriction,
    # chance = 0.20). Allows apples-to-apples vs cortex pipeline's task=0.788.
    fine_to_coarse = cifar100_fine_to_coarse(args.data_root)

    # Sanity eval — uniform-gated bank with augmentation training.
    bank_full, bank_task = eval_bank_uniform(
        bank, Xte_img, yte, pstd, sstd,
        grid=args.grid, device=args.device,
        fine_to_coarse=fine_to_coarse,
    )
    print()
    print(f"[eval] bank uniform (post-aug training): "
          f"full={bank_full:.4f}  task={bank_task:.4f}")
    print(f"       v1 reference (no aug):            full=0.2402  task=?")

    # ---- Phase B: train decisive trioron on bank features ----
    if args.features == "logits":
        feat_dim = P * N_CLASSES + P
    elif args.features == "hidden":
        feat_dim = P * args.h1 + P
    elif args.features == "both":
        feat_dim = P * N_CLASSES + P + P * args.h1
    else:
        raise ValueError(f"unknown --features {args.features!r}")
    decisive = build_decisive_trioron(feat_dim, args.decisive_h, N_CLASSES)
    n_dec = sum(p.numel() for p in decisive.parameters() if p.requires_grad)
    print()
    print(f"[Phase B] decisive trioron: features={args.features}  "
          f"in_dim={feat_dim}  h={args.decisive_h}  params={n_dec}",
          flush=True)
    train_decisive(
        decisive, bank, Xtr_img, ytr, pstd, sstd,
        grid=args.grid,
        epochs=args.decisive_epochs, batch_size=args.batch, lr=args.lr,
        use_augmentation=args.decisive_aug, device=args.device,
        features_mode=args.features,
    )

    dec_full, dec_task = eval_decisive(
        decisive, bank, Xte_img, yte, pstd, sstd,
        grid=args.grid, device=args.device,
        features_mode=args.features,
        fine_to_coarse=fine_to_coarse,
    )

    # ---- summary ----
    print()
    print("=" * 78)
    print("V2 RESULT — CIFAR-100 test accuracy "
          "(full = 100-way, task = 5-way superclass-restricted)")
    print("=" * 78)
    print(f"{'mode':40s}  {'full':>8s}  {'task':>8s}")
    print("-" * 78)
    print(f"{'v1 baseline (no aug, uniform)':40s}  {0.2402:>8.4f}  {'?':>8s}")
    print(f"{'v2 bank uniform (with aug)':40s}  {bank_full:>8.4f}  {bank_task:>8.4f}")
    print(f"{'v2 decisive trioron on bank features':40s}  {dec_full:>8.4f}  {dec_task:>8.4f}")
    print()
    print(f"Total params: bank {n_bank} + decisive {n_dec} = {n_bank + n_dec}")
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
