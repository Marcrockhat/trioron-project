"""bench_blindman_bank.py — positional trioron bank with edge-saliency
gating on CIFAR-100. V1 smoke test.

Each "blindman" branch sees ONE 8×8 patch of the 32×32 image and never
the rest. 16 blindmen on a 4×4 grid. Each branch is trained to:

  (a) classify the full image label given only its patch        (CE)
  (b) regress the Sobel edge magnitude of its patch (saliency)  (MSE)

At eval, each branch contributes a per-class log-pdf over its archive
(diagonal Gaussian on the L0(patch) embedding) and a saliency scalar.
Branches are gated softly:

    gate_i = softmax(log_pdf_i + α · saliency_i)        across i ∈ [16]

The bleed pattern across branches is the system's emergent "where the
salient signal is" heatmap; the sum of branch class logits weighted by
gate is the bank's class prediction.

V1 = bank only. No hierarchical contrastive, no extensible decisive
trioron. Per `feedback_cl_machinery_scope.md`, the bank is frozen-
after-curriculum and carries no continual-learning machinery (no
growth, no dream, no manifold replay). The decisive trioron stack
goes on top in v2 if the bank beats the prior CIFAR-100 multimodal /
hierarchical baselines (0.12 full / 0.59 task per
`hierarchical_trioron_result.md`).

Run:
  python3 -m experiments.cifar.bench_blindman_bank \\
      > outputs/bench_blindman_bank_run1.log 2>&1
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.cifar.datasets import load_cifar100, DEFAULT_DATA_ROOT


N_CLASSES = 100
IMG_HW = 32


# ---------------------------------------------------------------------
# Patch utilities
# ---------------------------------------------------------------------


def patch_split(images: torch.Tensor, grid: int) -> torch.Tensor:
    """(N, 3, 32, 32) → (N, grid*grid, 3*ph*pw) where ph = pw = 32 / grid."""
    N, C, H, W = images.shape
    ph = H // grid
    pw = W // grid
    if H % grid or W % grid:
        raise ValueError(f"grid {grid} doesn't evenly divide {H}×{W}")
    # (N, C, grid, ph, grid, pw) → (N, grid, grid, C, ph, pw)
    x = images.unfold(2, ph, ph).unfold(3, pw, pw)
    x = x.contiguous().view(N, C, grid, grid, ph, pw)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (N, grid, grid, C, ph, pw)
    x = x.view(N, grid * grid, C * ph * pw)
    return x


def sobel_saliency(images: torch.Tensor, grid: int) -> torch.Tensor:
    """Per-patch mean Sobel gradient magnitude. (N, 3, 32, 32) → (N, grid²)."""
    N, C, H, W = images.shape
    gray = images.mean(dim=1, keepdim=True)  # (N, 1, H, W) — luminance proxy
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]]).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3).contiguous()
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = (gx * gx + gy * gy).sqrt()  # (N, 1, H, W)
    # Mean over each patch.
    ph = H // grid
    pw = W // grid
    mag = mag.view(N, 1, grid, ph, grid, pw).mean(dim=(3, 5))  # (N, 1, grid, grid)
    return mag.view(N, grid * grid)


# ---------------------------------------------------------------------
# Standardization (per-patch-position, fit on train, apply to all)
# ---------------------------------------------------------------------


@dataclass
class PerPatchStandardizer:
    mu: torch.Tensor   # (P, D)
    sigma: torch.Tensor  # (P, D)

    @staticmethod
    def fit(patches: torch.Tensor) -> "PerPatchStandardizer":
        # patches: (N, P, D)
        mu = patches.mean(dim=0)
        sigma = patches.std(dim=0).clamp_min(1e-4)
        return PerPatchStandardizer(mu=mu, sigma=sigma)

    def transform(self, patches: torch.Tensor) -> torch.Tensor:
        return (patches - self.mu) / self.sigma


@dataclass
class ScalarStandardizer:
    mu: torch.Tensor   # (P,)
    sigma: torch.Tensor  # (P,)

    @staticmethod
    def fit(values: torch.Tensor) -> "ScalarStandardizer":
        mu = values.mean(dim=0)
        sigma = values.std(dim=0).clamp_min(1e-4)
        return ScalarStandardizer(mu=mu, sigma=sigma)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mu) / self.sigma


# ---------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------


class BlindmanBranch(nn.Module):
    """One positional branch. L0 is shared (passed in at forward), so
    this module only owns L1, class head, and saliency head."""

    def __init__(self, l0_dim: int, h1: int, n_classes: int):
        super().__init__()
        self.L1 = nn.Linear(l0_dim, h1)
        self.class_head = nn.Linear(h1, n_classes)
        self.saliency_head = nn.Linear(h1, 1)

    def forward(self, z0: torch.Tensor):
        h = F.relu(self.L1(z0))
        return self.class_head(h), self.saliency_head(h).squeeze(-1)


class BlindmanBank(nn.Module):
    """Shared frozen Gaussian L0 (patch_dim → l0_dim) + N positional branches."""

    def __init__(self, n_branches: int, patch_dim: int,
                 l0_dim: int, h1: int, n_classes: int, l0_seed: int = 42):
        super().__init__()
        gen = torch.Generator().manual_seed(l0_seed)
        std = math.sqrt(2.0 / patch_dim)
        W0 = torch.randn(l0_dim, patch_dim, generator=gen) * std
        # Frozen — register as buffer (not parameter) so it never updates.
        self.register_buffer("L0_W", W0)
        self.register_buffer("L0_b", torch.zeros(l0_dim))
        self.branches = nn.ModuleList(
            [BlindmanBranch(l0_dim, h1, n_classes) for _ in range(n_branches)]
        )
        self.n_branches = n_branches
        self.l0_dim = l0_dim

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (N, P, patch_dim) → (N, P, l0_dim) post-ReLU."""
        return F.relu(F.linear(patches, self.L0_W, self.L0_b))

    def forward_per_branch(self, z0_per_branch: torch.Tensor):
        """z0_per_branch: (N, P, l0_dim). Returns (logits (N, P, C),
        saliency (N, P))."""
        logits = []
        sals = []
        for i, br in enumerate(self.branches):
            lg, sl = br(z0_per_branch[:, i])
            logits.append(lg)
            sals.append(sl)
        return torch.stack(logits, dim=1), torch.stack(sals, dim=1)


# ---------------------------------------------------------------------
# Archive (diagonal Gaussian per (branch, class) on L0 outputs)
# ---------------------------------------------------------------------


def build_archive(
    bank: BlindmanBank,
    z0: torch.Tensor,        # (N, P, l0_dim)  L0 of training patches
    labels: torch.Tensor,    # (N,)
    n_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-(branch, class) μ, σ on L0 outputs. Returns
    mu (P, C, l0_dim), sigma (P, C, l0_dim)."""
    P = z0.shape[1]
    D = z0.shape[2]
    mu = torch.zeros(P, n_classes, D)
    sigma = torch.ones(P, n_classes, D)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() < 2:
            continue
        zc = z0[mask]                                 # (Nc, P, D)
        mu[:, c] = zc.mean(dim=0)
        sigma[:, c] = zc.std(dim=0).clamp_min(1e-3)
    return mu, sigma


def archive_logpdf(
    z0: torch.Tensor,        # (N, P, D)
    mu: torch.Tensor,         # (P, C, D)
    sigma: torch.Tensor,      # (P, C, D)
) -> torch.Tensor:
    """Per-(image, branch, class) log-pdf under diagonal Gaussian.
    Returns (N, P, C)."""
    # diff: (N, P, 1, D) - (1, P, C, D) = (N, P, C, D)
    diff = z0.unsqueeze(2) - mu.unsqueeze(0)
    inv_var = 1.0 / (sigma * sigma)                     # (P, C, D)
    log_norm = -0.5 * (torch.log(2 * torch.pi * sigma * sigma)).sum(dim=-1)  # (P, C)
    quad = -0.5 * (diff * diff * inv_var.unsqueeze(0)).sum(dim=-1)  # (N, P, C)
    return quad + log_norm.unsqueeze(0)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train_bank(
    bank: BlindmanBank,
    Xtr_patches_std: torch.Tensor,    # (N, P, patch_dim) standardized
    sal_targets_std: torch.Tensor,    # (N, P) standardized per patch
    ytr: torch.Tensor,                # (N,)
    *,
    epochs: int, batch_size: int, lr: float,
    saliency_weight: float, device: str,
) -> List[float]:
    """Joint training loop — all branches trained in parallel on each
    minibatch. Returns per-epoch mean loss."""
    bank.to(device)
    Xtr_patches_std = Xtr_patches_std.to(device)
    sal_targets_std = sal_targets_std.to(device)
    ytr = ytr.to(device)

    opt = torch.optim.Adam(bank.parameters(), lr=lr)
    history = []
    N = Xtr_patches_std.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        total = 0.0
        n_batches = 0
        t0 = time.time()
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            x = Xtr_patches_std[idx]               # (B, P, patch_dim)
            y = ytr[idx]                            # (B,)
            sal = sal_targets_std[idx]              # (B, P)
            z0 = bank.encode(x)                     # (B, P, l0_dim)
            logits, sal_pred = bank.forward_per_branch(z0)
            B, P, C = logits.shape
            ce = F.cross_entropy(logits.view(B * P, C), y.repeat_interleave(P))
            mse = F.mse_loss(sal_pred, sal)
            loss = ce + saliency_weight * mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n_batches += 1
        avg = total / n_batches
        history.append(avg)
        dt = time.time() - t0
        print(f"  epoch {epoch+1:3d}/{epochs}  loss {avg:.4f}  ({dt:.1f}s)", flush=True)
    return history


# ---------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------


def evaluate(
    bank: BlindmanBank,
    archive_mu: torch.Tensor,        # (P, C, D)
    archive_sigma: torch.Tensor,     # (P, C, D)
    Xte_patches_std: torch.Tensor,   # (N, P, patch_dim)
    sal_targets_std: torch.Tensor,   # (N, P) — only used by saliency-aware gates
    yte: torch.Tensor,
    *,
    alpha: float,
    mode: str,                       # "uniform" | "archive" | "saliency" | "combined"
    device: str,
    batch_size: int = 512,
) -> Tuple[float, torch.Tensor]:
    """Returns (accuracy, mean per-branch gate weight (P,))."""
    bank.to(device).eval()
    archive_mu = archive_mu.to(device)
    archive_sigma = archive_sigma.to(device)
    Xte_patches_std = Xte_patches_std.to(device)
    sal_targets_std = sal_targets_std.to(device)
    yte = yte.to(device)

    N = Xte_patches_std.shape[0]
    correct = 0
    gate_sum = torch.zeros(bank.n_branches, device=device)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            x = Xte_patches_std[i:i + batch_size]   # (B, P, patch_dim)
            y = yte[i:i + batch_size]
            B = x.shape[0]
            z0 = bank.encode(x)                     # (B, P, D)
            logits, sal_pred = bank.forward_per_branch(z0)  # (B, P, C), (B, P)

            if mode == "uniform":
                gates = torch.full((B, bank.n_branches), 1.0 / bank.n_branches,
                                   device=device)
            else:
                # branch confidence = logsumexp_c log_pdf(z | μ_c, σ_c)
                logpdf = archive_logpdf(z0, archive_mu, archive_sigma)  # (B, P, C)
                branch_conf = torch.logsumexp(logpdf, dim=-1)           # (B, P)
                if mode == "archive":
                    score = branch_conf
                elif mode == "saliency":
                    score = sal_pred
                elif mode == "combined":
                    score = branch_conf + alpha * sal_pred
                else:
                    raise ValueError(f"unknown mode {mode}")
                gates = F.softmax(score, dim=-1)                        # (B, P)

            # Bleed-weighted aggregated logits.
            agg = (gates.unsqueeze(-1) * logits).sum(dim=1)             # (B, C)
            pred = agg.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            gate_sum += gates.sum(dim=0)
    acc = correct / N
    mean_gate = gate_sum / N
    return acc, mean_gate.cpu()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=4,
                   help="patch grid resolution (4 → 16 patches of 8×8)")
    p.add_argument("--l0-dim", type=int, default=128)
    p.add_argument("--h1", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--saliency-weight", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="saliency weight in gating: gate ∝ log_pdf + α·saliency")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    print("Blindman positional bank — CIFAR-100 v1 smoke test")
    print(f"  grid={args.grid}  patches={args.grid**2}  patch_dim={3*(IMG_HW//args.grid)**2}")
    print(f"  l0_dim={args.l0_dim}  h1={args.h1}  epochs={args.epochs}  "
          f"batch={args.batch}  lr={args.lr}  α={args.alpha}  "
          f"saliency_weight={args.saliency_weight}")
    print(f"  device={args.device}")

    # ---- data ----
    print("[data] loading CIFAR-100 ...", flush=True)
    Xtr_img, ytr = load_cifar100(args.data_root, train=True)
    Xte_img, yte = load_cifar100(args.data_root, train=False)
    print(f"  train: {tuple(Xtr_img.shape)}  test: {tuple(Xte_img.shape)}")

    Xtr_patches = patch_split(Xtr_img, args.grid)   # (N, P, patch_dim)
    Xte_patches = patch_split(Xte_img, args.grid)
    sal_tr = sobel_saliency(Xtr_img, args.grid)     # (N, P)
    sal_te = sobel_saliency(Xte_img, args.grid)
    P = Xtr_patches.shape[1]
    patch_dim = Xtr_patches.shape[2]
    print(f"  patches: train {tuple(Xtr_patches.shape)}  "
          f"test {tuple(Xte_patches.shape)}")

    # standardize patches (per patch position) and saliency targets
    pstd = PerPatchStandardizer.fit(Xtr_patches)
    Xtr_patches_s = pstd.transform(Xtr_patches)
    Xte_patches_s = pstd.transform(Xte_patches)
    sstd = ScalarStandardizer.fit(sal_tr)
    sal_tr_s = sstd.transform(sal_tr)
    sal_te_s = sstd.transform(sal_te)

    # ---- bank ----
    bank = BlindmanBank(
        n_branches=P, patch_dim=patch_dim,
        l0_dim=args.l0_dim, h1=args.h1, n_classes=N_CLASSES,
        l0_seed=args.seed,
    )
    n_params = sum(p.numel() for p in bank.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params}  "
          f"(per branch: {n_params // P})")

    # ---- train ----
    print("[train] joint bank training (CE + λ·MSE saliency) ...", flush=True)
    train_bank(
        bank, Xtr_patches_s, sal_tr_s, ytr,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        saliency_weight=args.saliency_weight, device=args.device,
    )

    # ---- archive ----
    print("[archive] building per-(branch, class) diagonal-Gaussian archive ...",
          flush=True)
    bank.eval()
    with torch.no_grad():
        # encode on CPU in chunks to avoid OOM
        z0_chunks = []
        for i in range(0, Xtr_patches_s.shape[0], 1024):
            z0_chunks.append(
                bank.encode(Xtr_patches_s[i:i + 1024].to(args.device)).cpu()
            )
        z0_train = torch.cat(z0_chunks, dim=0)
    archive_mu, archive_sigma = build_archive(
        bank, z0_train, ytr, n_classes=N_CLASSES,
    )

    # ---- eval ----
    print()
    print("=" * 78)
    print("EVAL — full-softmax accuracy on CIFAR-100 test set")
    print("=" * 78)
    print(f"{'mode':12s}  {'α':>5s}  {'accuracy':>10s}")
    print("-" * 78)
    rows = []
    for mode, alpha in [
        ("uniform",  0.0),
        ("archive",  0.0),
        ("saliency", 0.0),
        ("combined", args.alpha),
    ]:
        acc, mean_gate = evaluate(
            bank, archive_mu, archive_sigma,
            Xte_patches_s, sal_te_s, yte,
            alpha=alpha, mode=mode, device=args.device,
        )
        print(f"{mode:12s}  {alpha:5.2f}  {acc:10.4f}")
        rows.append((mode, alpha, acc, mean_gate))

    # ---- diagnostics ----
    print()
    print("Per-branch mean gate (combined mode):")
    grid = args.grid
    mean_gate = rows[-1][3].view(grid, grid)
    for r in range(grid):
        line = "  "
        for c in range(grid):
            line += f"{mean_gate[r, c].item():6.3f} "
        print(line)

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
