"""bench_blindman_bank_v3.py — hierarchical decisive composition over blindman
bank features with discovered taxonomy and absorbed expert donors.

Pipeline per seed:
  1. Train v2b bank (192→128→32 per branch, grid 4, with augmentation).
  2. Compute 100 class centroids on bank features (1616-d).
  3. k-means cluster centroids → K=20 macro assignments (taxonomy discovered
     from blindman's perceptual world, not borrowed from CIFAR-100 official
     superclasses or the prior taxonomic-contrastive arc).
  4. Train macro classifier (1616 → 128 → K) on (bank_features, macro_label).
  5. For each macro c ∈ [0..K-1]:
       Train one expert donor (shared frozen Gaussian L0 1616 → 128, then
       L1 128 → 64, head 64 → n_classes_in_c). Manifold archive computed
       over L0 codes per fine class. Save as donor checkpoint.
  6. Load all K experts as Branches; absorb into MultiBranchOrganism with
     the shared canonical L0.
  7. Eval: hierarchical composition
       P(fine_i | x) = P(macro_of_i | x) · softmax(expert_macro_of_i)[local_i]
     Argmax over all 100 fine classes.

Reports:
  - alignment of discovered taxonomy with CIFAR-100's official superclasses
    (probe; not used downstream)
  - macro classifier accuracy (20-way)
  - per-expert task-aware accuracy (within-macro fine classification)
  - hierarchical composition: full + task (5-way superclass-restricted)

Run:
  python3 -m experiments.cifar.bench_blindman_bank_v3 --seed 42 \\
      > outputs/bench_blindman_bank_v3_seed_42.log 2>&1
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

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
from experiments.cifar.bench_blindman_bank_v2 import (
    cifar_augment, train_bank_with_aug, bank_features, eval_bank_uniform,
    _restricted_argmax,
)
from trioron.network import TrioronNetwork
from trioron.multibranch import Branch, MultiBranchOrganism


N_CLASSES = 100
IMG_HW = 32


# ---------------------------------------------------------------------
# k-means clustering of class centroids
# ---------------------------------------------------------------------


def kmeans(
    X: torch.Tensor,           # (N, D)
    k: int,
    *,
    n_iter: int = 100,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plain k-means. Returns (assignments (N,), centers (k, D))."""
    N, D = X.shape
    g = torch.Generator().manual_seed(seed)
    init_idx = torch.randperm(N, generator=g)[:k]
    centers = X[init_idx].clone()
    for _ in range(n_iter):
        # Assign: each point to nearest center.
        d = torch.cdist(X, centers)              # (N, k)
        assign = d.argmin(dim=1)                  # (N,)
        # Update: each center to mean of assigned points.
        new_centers = centers.clone()
        for c in range(k):
            mask = assign == c
            if mask.any():
                new_centers[c] = X[mask].mean(dim=0)
        if torch.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers
    return assign, centers


def cluster_alignment(
    discovered: torch.Tensor,         # (100,) int64 — discovered macro per fine
    reference: torch.Tensor,          # (100,) int64 — official superclass per fine
) -> float:
    """Adjusted-rand-style cluster alignment in [0, 1]: average over all
    pairs of fine classes whether they agree on co-clustering."""
    N = discovered.shape[0]
    same_disc = discovered.unsqueeze(0) == discovered.unsqueeze(1)   # (N, N)
    same_ref = reference.unsqueeze(0) == reference.unsqueeze(1)
    # Diagonal is trivially True; mask it out.
    mask = ~torch.eye(N, dtype=torch.bool)
    return float((same_disc[mask] == same_ref[mask]).float().mean())


# ---------------------------------------------------------------------
# Per-class centroids on bank features
# ---------------------------------------------------------------------


@torch.no_grad()
def compute_class_centroids(
    bank: BlindmanBank,
    Xtr_img: torch.Tensor, ytr: torch.Tensor,
    pstd: PerPatchStandardizer, sstd: ScalarStandardizer,
    *,
    grid: int, device: str, n_classes: int = N_CLASSES,
    batch: int = 512,
) -> torch.Tensor:
    """For each fine class, average bank features (1616-d) over training images.
    Returns (n_classes, feat_dim)."""
    bank.to(device).eval()
    pstd_mu = pstd.mu.to(device)
    pstd_sigma = pstd.sigma.to(device)
    sstd_mu = sstd.mu.to(device)
    sstd_sigma = sstd.sigma.to(device)
    feat_dim = bank.n_branches * N_CLASSES + bank.n_branches
    sums = torch.zeros(n_classes, feat_dim, device=device)
    counts = torch.zeros(n_classes, device=device)
    N = Xtr_img.shape[0]
    for i in range(0, N, batch):
        imgs = Xtr_img[i:i + batch].to(device)
        y = ytr[i:i + batch].to(device)
        feats = bank_features(bank, imgs, pstd_mu, pstd_sigma,
                              sstd_mu, sstd_sigma, grid,
                              features_mode="logits")
        for c in range(n_classes):
            mask = y == c
            if mask.any():
                sums[c] += feats[mask].sum(dim=0)
                counts[c] += mask.sum()
    return (sums / counts.clamp_min(1).unsqueeze(-1)).cpu()


# ---------------------------------------------------------------------
# Macro classifier
# ---------------------------------------------------------------------


def train_macro_classifier(
    bank: BlindmanBank,
    Xtr_img: torch.Tensor, ytr: torch.Tensor,
    fine_to_macro: torch.Tensor,        # (100,) int64
    pstd: PerPatchStandardizer, sstd: ScalarStandardizer,
    *,
    grid: int, K: int, h: int,
    epochs: int, batch_size: int, lr: float,
    use_augmentation: bool, device: str,
) -> TrioronNetwork:
    """Train macro classifier on bank features → K-way."""
    feat_dim = bank.n_branches * N_CLASSES + bank.n_branches
    macro = TrioronNetwork([
        (feat_dim, h, "relu"),
        (h, K, "linear"),
    ]).to(device)
    bank.to(device).eval()
    for p in bank.parameters():
        p.requires_grad_(False)
    pstd_mu = pstd.mu.to(device)
    pstd_sigma = pstd.sigma.to(device)
    sstd_mu = sstd.mu.to(device)
    sstd_sigma = sstd.sigma.to(device)
    f2m = fine_to_macro.to(device)

    opt = torch.optim.Adam(macro.parameters(), lr=lr)
    N = Xtr_img.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(N)
        total = 0.0
        nb = 0
        t0 = time.time()
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            imgs = Xtr_img[idx].to(device)
            y_fine = ytr[idx].to(device)
            y_macro = f2m[y_fine]
            if use_augmentation:
                imgs = cifar_augment(imgs)
            with torch.no_grad():
                feats = bank_features(bank, imgs, pstd_mu, pstd_sigma,
                                      sstd_mu, sstd_sigma, grid,
                                      features_mode="logits")
            logits = macro(feats)
            loss = F.cross_entropy(logits, y_macro)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        print(f"  [macro] epoch {epoch+1:3d}/{epochs}  loss {total/nb:.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
    return macro


# ---------------------------------------------------------------------
# Expert donor — TrioronNetwork (1616 → L0:128 → L1:64 → head:n_in_macro)
# with shared frozen Gaussian L0 across all experts (canonical absorb)
# ---------------------------------------------------------------------


def make_expert(
    feat_dim: int, l0_dim: int, h1: int, n_classes: int,
    *, l0_seed: int,
) -> TrioronNetwork:
    """Build an expert with frozen Gaussian L0 of shape (l0_dim, feat_dim)
    seeded by l0_seed. Same seed across all experts → canonical L0 → lossless
    absorption per multibranch_absorption_result.md.
    """
    net = TrioronNetwork([
        (feat_dim, l0_dim, "relu"),
        (l0_dim, h1, "relu"),
        (h1, n_classes, "linear"),
    ])
    # Re-seed L0 weights to the canonical Gaussian; freeze.
    g = torch.Generator().manual_seed(l0_seed)
    std = math.sqrt(2.0 / feat_dim)
    with torch.no_grad():
        net.layers[0].W.copy_(torch.randn(l0_dim, feat_dim, generator=g) * std)
        net.layers[0].b.zero_()
    net.layers[0].W.requires_grad_(False)
    net.layers[0].b.requires_grad_(False)
    return net


def train_expert(
    expert: TrioronNetwork,
    feats_tr: torch.Tensor,            # (N_macro, feat_dim) on cpu
    y_local_tr: torch.Tensor,          # (N_macro,) local fine indices in [0..n_in_macro)
    *,
    epochs: int, batch_size: int, lr: float, device: str,
) -> None:
    expert.to(device)
    feats_tr = feats_tr.to(device)
    y_local_tr = y_local_tr.to(device)
    opt = torch.optim.Adam(
        [p for p in expert.parameters() if p.requires_grad], lr=lr,
    )
    N = feats_tr.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        total = 0.0
        nb = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            x = feats_tr[idx]
            y = y_local_tr[idx]
            logits = expert(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1


@torch.no_grad()
def compute_expert_archive(
    expert: TrioronNetwork,
    feats_tr: torch.Tensor,
    y_local_tr: torch.Tensor,
    n_in_macro: int,
    device: str,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Per-class diagonal-Gaussian archive over L0 outputs (canonical
    z-space). Keys are LOCAL fine indices in [0..n_in_macro). The caller
    remaps local→global before saving the donor checkpoint.
    """
    expert.to(device).eval()
    feats_tr = feats_tr.to(device)
    y_local_tr = y_local_tr.to(device)
    L0 = expert.layers[0]
    z_all = F.relu(F.linear(feats_tr, L0.W, L0.b))      # (N, l0_dim)
    archive: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for local_c in range(n_in_macro):
        mask = y_local_tr == local_c
        if mask.sum() < 2:
            d = z_all.shape[1]
            archive[local_c] = (torch.zeros(d), torch.ones(d))
            continue
        zc = z_all[mask]
        archive[local_c] = (zc.mean(dim=0).cpu(),
                            zc.std(dim=0).clamp_min(1e-3).cpu())
    return archive


def save_expert_donor(
    expert: TrioronNetwork,
    archive_local: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    classes_covered_global: List[int],
    *,
    path: str, l0_seed: int, label: str,
) -> None:
    """Save in poc_donor_*.pt format so Branch.from_checkpoint loads cleanly."""
    n_nodes = [layer.W.shape[0] for layer in expert.layers]
    input_dim = expert.layers[0].W.shape[1]
    # Remap local archive keys to global class IDs.
    archive_global = {
        int(classes_covered_global[local_c]): (mu.cpu(), sg.cpu())
        for local_c, (mu, sg) in archive_local.items()
    }
    payload = {
        "state_dict": expert.state_dict(),
        "n_nodes_per_layer": n_nodes,
        "input_dim": int(input_dim),
        "classes_covered": [int(c) for c in classes_covered_global],
        "l0_seed": int(l0_seed),
        "manifold_stats": archive_global,
        "arm": "blindman_v3_expert",
        "label": label,
    }
    torch.save(payload, path)


# ---------------------------------------------------------------------
# Hierarchical eval — macro classifier × experts
# ---------------------------------------------------------------------


@torch.no_grad()
def eval_hierarchical(
    bank: BlindmanBank,
    macro: TrioronNetwork,
    experts: List[TrioronNetwork],
    macro_classes_covered: List[List[int]],
    Xte_img: torch.Tensor, yte: torch.Tensor,
    pstd: PerPatchStandardizer, sstd: ScalarStandardizer,
    *,
    grid: int, device: str, batch: int = 512,
) -> Tuple[float, float, float]:
    """Returns (full_acc, perceptual_task_acc, macro_acc).

    `perceptual_task_acc` restricts argmax to the fine classes belonging to
    the true *discovered* macro (per `feedback_perceptual_taxonomy_only.md`)
    — NOT to CIFAR-100's official superclasses.

    Composition:
      P(fine_i | x) = softmax(macro(x))[macro_of_i]
                    · softmax(expert_{macro_of_i}(x))[local_idx_of_i]
    """
    bank.to(device).eval()
    macro.to(device).eval()
    for e in experts:
        e.to(device).eval()
    pstd_mu = pstd.mu.to(device)
    pstd_sigma = pstd.sigma.to(device)
    sstd_mu = sstd.mu.to(device)
    sstd_sigma = sstd.sigma.to(device)

    # Build fine→discovered_macro mapping; this also serves as the task-aware
    # restriction (perceptual task-aware).
    fine_to_macro = torch.full((N_CLASSES,), -1, dtype=torch.long)
    fine_to_local = torch.full((N_CLASSES,), -1, dtype=torch.long)
    for m_idx, classes in enumerate(macro_classes_covered):
        for local_c, global_c in enumerate(classes):
            fine_to_macro[global_c] = m_idx
            fine_to_local[global_c] = local_c
    if (fine_to_macro < 0).any() or (fine_to_local < 0).any():
        raise RuntimeError("Some fine classes not covered by any macro.")
    fine_to_macro_dev = fine_to_macro.to(device)
    fine_to_local = fine_to_local.to(device)

    correct_full = 0
    correct_task = 0
    correct_macro = 0
    N = Xte_img.shape[0]
    for i in range(0, N, batch):
        imgs = Xte_img[i:i + batch].to(device)
        y = yte[i:i + batch].to(device)
        B = imgs.shape[0]
        feats = bank_features(bank, imgs, pstd_mu, pstd_sigma,
                              sstd_mu, sstd_sigma, grid,
                              features_mode="logits")
        macro_logp = F.log_softmax(macro(feats), dim=-1)         # (B, K)
        fine_logp = torch.empty(B, N_CLASSES, device=device)
        for k, classes in enumerate(macro_classes_covered):
            expert_logp = F.log_softmax(experts[k](feats), dim=-1)  # (B, n_k)
            for local_idx, global_c in enumerate(classes):
                fine_logp[:, global_c] = (macro_logp[:, k]
                                          + expert_logp[:, local_idx])
        pred_full = fine_logp.argmax(dim=-1)
        correct_full += int((pred_full == y).sum().item())
        # Perceptual task-aware: restrict to true discovered macro's siblings.
        pred_task = _restricted_argmax(fine_logp, y, fine_to_macro_dev)
        correct_task += int((pred_task == y).sum().item())
        # Macro classifier alone: argmax over K macros vs true discovered macro.
        true_macro = fine_to_macro_dev[y]
        pred_macro = macro_logp.argmax(dim=-1)
        correct_macro += int((pred_macro == true_macro).sum().item())
    return (correct_full / N, correct_task / N, correct_macro / N)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=4)
    p.add_argument("--l0-dim", type=int, default=128)
    p.add_argument("--h1", type=int, default=32)
    p.add_argument("--K", type=int, default=20, help="number of discovered macros")
    p.add_argument("--macro-h", type=int, default=128)
    p.add_argument("--expert-l0-dim", type=int, default=128)
    p.add_argument("--expert-h1", type=int, default=64)
    p.add_argument("--bank-epochs", type=int, default=30)
    p.add_argument("--macro-epochs", type=int, default=20)
    p.add_argument("--expert-epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--saliency-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint-dir",
                   default="outputs/blindman_v3_donors")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print("Blindman positional bank v3 — hierarchical with discovered taxonomy")
    print(f"  grid={args.grid}  K={args.K}  seed={args.seed}  device={args.device}")

    # ---- data ----
    print("[data] loading CIFAR-100 ...", flush=True)
    Xtr_img, ytr = load_cifar100(args.data_root, train=True)
    Xte_img, yte = load_cifar100(args.data_root, train=False)
    Xtr_patches_clean = patch_split(Xtr_img, args.grid)
    sal_tr_clean = sobel_saliency(Xtr_img, args.grid)
    pstd = PerPatchStandardizer.fit(Xtr_patches_clean)
    sstd = ScalarStandardizer.fit(sal_tr_clean)
    P = Xtr_patches_clean.shape[1]
    patch_dim = Xtr_patches_clean.shape[2]
    del Xtr_patches_clean, sal_tr_clean
    fine_to_coarse = cifar100_fine_to_coarse(args.data_root)

    # ---- Phase A: train bank (same as v2b) ----
    bank = BlindmanBank(
        n_branches=P, patch_dim=patch_dim,
        l0_dim=args.l0_dim, h1=args.h1, n_classes=N_CLASSES,
        l0_seed=args.seed,
    )
    n_bank = sum(p.numel() for p in bank.parameters() if p.requires_grad)
    print(f"  bank trainable params: {n_bank}")
    print("[Phase A] training bank with augmentation ...", flush=True)
    train_bank_with_aug(
        bank, Xtr_img, ytr, pstd, sstd,
        grid=args.grid,
        epochs=args.bank_epochs, batch_size=args.batch, lr=args.lr,
        saliency_weight=args.saliency_weight, device=args.device,
    )

    # ---- Phase B: cluster centroids → discovered taxonomy ----
    print("[Phase B] computing 100 class centroids ...", flush=True)
    centroids = compute_class_centroids(
        bank, Xtr_img, ytr, pstd, sstd,
        grid=args.grid, device=args.device,
    )
    print(f"  centroids shape: {tuple(centroids.shape)}")
    print(f"[Phase B] k-means K={args.K} on centroids ...", flush=True)
    fine_to_macro, _ = kmeans(centroids, args.K, n_iter=200, seed=args.seed)
    macro_classes_covered = [
        sorted(int(c) for c in (fine_to_macro == k).nonzero().squeeze(-1).tolist())
        for k in range(args.K)
    ]
    sizes = [len(m) for m in macro_classes_covered]
    print(f"  macro sizes: min={min(sizes)} max={max(sizes)} "
          f"mean={sum(sizes)/len(sizes):.1f}")
    align = cluster_alignment(fine_to_macro, fine_to_coarse)
    print(f"  alignment with CIFAR-100 official superclasses: {align:.4f} "
          f"(1.0 = identical, 0.5 = random)")

    # ---- Phase C: macro classifier ----
    print("[Phase C] training macro classifier (K-way) ...", flush=True)
    macro = train_macro_classifier(
        bank, Xtr_img, ytr, fine_to_macro, pstd, sstd,
        grid=args.grid, K=args.K, h=args.macro_h,
        epochs=args.macro_epochs, batch_size=args.batch, lr=args.lr,
        use_augmentation=True, device=args.device,
    )

    # ---- Phase D: experts (precompute clean bank features for fast training) ----
    print("[Phase D] precomputing clean bank features for expert training ...",
          flush=True)
    pstd_mu_dev = pstd.mu.to(args.device)
    pstd_sigma_dev = pstd.sigma.to(args.device)
    sstd_mu_dev = sstd.mu.to(args.device)
    sstd_sigma_dev = sstd.sigma.to(args.device)
    feat_dim = P * N_CLASSES + P
    feats_tr_chunks = []
    bank.to(args.device).eval()
    with torch.no_grad():
        for i in range(0, Xtr_img.shape[0], 512):
            imgs = Xtr_img[i:i + 512].to(args.device)
            f = bank_features(bank, imgs, pstd_mu_dev, pstd_sigma_dev,
                              sstd_mu_dev, sstd_sigma_dev, args.grid,
                              features_mode="logits")
            feats_tr_chunks.append(f.cpu())
    feats_tr = torch.cat(feats_tr_chunks, dim=0)
    del feats_tr_chunks
    print(f"  feats_tr shape: {tuple(feats_tr.shape)}")

    # Per-expert L0 seed: macro k uses base_seed * 1000 + 1 (one shared seed
    # across all experts → canonical L0 → lossless absorb).
    canonical_l0_seed = int(args.seed) * 1000 + 1
    print(f"  canonical expert L0 seed: {canonical_l0_seed}")
    print(f"[Phase D] training {args.K} expert donors ...", flush=True)
    expert_paths = []
    experts: List[TrioronNetwork] = []
    for k, classes in enumerate(macro_classes_covered):
        n_in = len(classes)
        if n_in == 0:
            print(f"  expert {k}: EMPTY macro — skipping (this should be rare)")
            continue
        # Build expert with shared L0 seed.
        expert = make_expert(
            feat_dim=feat_dim,
            l0_dim=args.expert_l0_dim,
            h1=args.expert_h1,
            n_classes=n_in,
            l0_seed=canonical_l0_seed,
        )
        # Filter training data and remap fine labels to local indices.
        global_to_local = {int(g): local for local, g in enumerate(classes)}
        mask = torch.zeros(ytr.shape[0], dtype=torch.bool)
        for g in classes:
            mask |= (ytr == g)
        feats_k = feats_tr[mask]
        y_local = torch.tensor(
            [global_to_local[int(c)] for c in ytr[mask].tolist()],
            dtype=torch.long,
        )
        t0 = time.time()
        train_expert(
            expert, feats_k, y_local,
            epochs=args.expert_epochs,
            batch_size=args.batch, lr=args.lr, device=args.device,
        )
        # Compute archive (per local fine class).
        archive = compute_expert_archive(
            expert, feats_k, y_local, n_in, args.device,
        )
        path = os.path.join(
            args.checkpoint_dir,
            f"v3_seed{args.seed}_macro{k:02d}.pt",
        )
        save_expert_donor(
            expert, archive, classes,
            path=path, l0_seed=canonical_l0_seed,
            label=f"v3_macro{k:02d}",
        )
        expert_paths.append(path)
        experts.append(expert)
        print(f"  expert {k:02d}  classes={n_in}  examples={int(mask.sum())}  "
              f"({time.time()-t0:.1f}s)  → {path}", flush=True)

    # ---- Phase E: absorb (load donors as Branches, build organism) ----
    print("[Phase E] absorbing experts via MultiBranchOrganism ...", flush=True)
    branches = [Branch.from_checkpoint(pth, label=f"macro{k:02d}")
                for k, pth in enumerate(expert_paths)]
    try:
        organism = MultiBranchOrganism.from_branches(branches)
        print(f"  organism: {len(branches)} branches, "
              f"L0 seed={organism.l0_seed}, "
              f"covers {len(organism._union_classes)} unique fine classes")
    except Exception as e:
        print(f"  WARNING: organism assembly failed: {e}")
        print(f"  proceeding with hierarchical eval using direct expert calls")

    # ---- Phase F: eval (perceptual task-aware throughout) ----
    print("[Phase F] hierarchical eval on test set "
          "(task-aware uses DISCOVERED macros) ...", flush=True)
    full_acc, task_acc, macro_acc = eval_hierarchical(
        bank, macro, experts, macro_classes_covered,
        Xte_img, yte, pstd, sstd,
        grid=args.grid, device=args.device,
    )

    # Sanity: bank uniform alone for reference, also perceptual task-aware
    # (eval_bank_uniform restricts to siblings of whatever mapping is passed).
    bank_full, bank_task = eval_bank_uniform(
        bank, Xte_img, yte, pstd, sstd,
        grid=args.grid, device=args.device,
        fine_to_coarse=fine_to_macro,    # discovered macros, NOT human superclasses
    )

    # ---- summary ----
    n_macro = sum(p.numel() for p in macro.parameters() if p.requires_grad)
    n_expert_total = sum(
        sum(pp.numel() for pp in e.parameters() if pp.requires_grad)
        for e in experts
    )
    print()
    print("=" * 78)
    print("V3 RESULT — CIFAR-100 hierarchical decisive over blindman bank")
    print("=" * 78)
    print(f"  taxonomy alignment with CIFAR-100 official superclasses (probe only, NOT used downstream): {align:.4f}")
    print(f"  macro classifier K-way acc on DISCOVERED macros: {macro_acc:.4f}")
    print()
    print(f"{'mode (task-aware = DISCOVERED-macro restriction)':50s}  {'full':>8s}  {'task':>8s}")
    print("-" * 78)
    print(f"{'bank uniform (with aug)':50s}  {bank_full:>8.4f}  {bank_task:>8.4f}")
    print(f"{'v2b reference (decisive flat, n=3 / human task)':50s}  "
          f"{0.3089:>8.4f}  {0.6062:>8.4f}*")
    print(f"{'v3 hierarchical (macro × absorbed experts)':50s}  "
          f"{full_acc:>8.4f}  {task_acc:>8.4f}")
    print(f"  * v2b task-aware was vs CIFAR-100 official superclasses — not directly comparable to v3's perceptual task-aware")
    print()
    print(f"Total params (trainable): bank {n_bank} + macro {n_macro} "
          f"+ {len(experts)} experts {n_expert_total} = "
          f"{n_bank + n_macro + n_expert_total}")
    print(f"Donor checkpoints: {args.checkpoint_dir}/v3_seed{args.seed}_macro*.pt")
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
