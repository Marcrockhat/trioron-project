"""Probe: L0 class-prototype separation for prototype-based pseudo-rehearsal.

Question: For the chained-15 architecture, the network's task-aware
accuracy hits 0.93+ across all arms. Rocky's observation (2026-05-04)
is that this high accuracy means the network's L1+head is doing real
discriminative work — so could we replace the raw-sample rehearsal
buffer (~4.7 MB) with a class-prototype pseudo-rehearsal scheme
(~15 KB) that just stores ONE feature-space mean per class and
generates synthetic past-task samples by Gaussian noise around each
prototype?

This works ONLY if class prototypes are well-separated in L0 feature
space relative to within-class spread. L0 is a frozen 784→128 random
Kaiming projection (no learning); the question is whether the random
projection preserves class structure well enough for prototype +
noise to be discriminative.

Statistics computed (all in L0 post-activation feature space, 128-dim):

    Per class c:
      μ_c = mean(L0(x))                      # the prototype
      σ_c = mean over feats of std(L0(x))    # within-class spread

    Aggregate:
      mean_inter = mean over (i,j) of ||μ_i − μ_j||₂
      mean_intra = mean over c of mean over feats of σ_c
                                              # turned into a distance
                                              # via sqrt(d) × σ̄
      discriminability = mean_inter / mean_intra

    Per pair:
      closest pair (most confusable)
      furthest pair (most distinguishable)

Decision rule:
    discriminability > 2  → prototypes well-separated; prototype +
                              Gaussian noise rehearsal will work cleanly
    1 < dr ≤ 2            → marginal; prototype rehearsal might work but
                              expect noisy class boundaries; widen σ for
                              the noise carefully
    discriminability ≤ 1  → prototypes overlap with within-class spread;
                              prototype rehearsal would produce ambiguous
                              samples; need richer representation
                              (cluster prototypes, or pseudo-rehearsal
                              via stored logits instead).

L0 is RANDOM (Kaiming-init Gaussian projection, no training). Random
projections theoretically preserve distances by Johnson-Lindenstrauss
when output_dim ≥ O(log(N)/ε²). For 30 classes with reasonable margin,
log(30)/ε² ≈ 3.4/0.25 = 13 dims should suffice. We have 128 dims, so
the JL bound says we have plenty. The question is whether the actual
class structure in 784-dim pixel space survives projection — JL
preserves *distances*, not necessarily *class boundaries*.

Run:
    python3 -m experiments.probe_l0_class_prototypes \
        > outputs/probe_l0_class_prototypes.log 2>&1
"""
from __future__ import annotations
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import TrioronNetwork
from experiments.datasets import (
    DEFAULT_DATA_ROOT,
    DatasetBundle,
    build_task_views,
    chained_15_specs,
)


SEED = 0
INPUT_DIM = 784
L0_WIDTH = 128


def main():
    print("=" * 78)
    print("L0 class-prototype separation probe")
    print("=" * 78)
    print(f"seed: {SEED}   L0: {INPUT_DIM} → {L0_WIDTH} (frozen Kaiming)")
    print()

    # Build the same frozen L0 the bench uses.
    torch.manual_seed(SEED)
    # Just the L0 layer — we don't need L1/head for this probe.
    net = TrioronNetwork([(INPUT_DIM, L0_WIDTH, "relu")])
    l0 = net.layers[0]
    l0.W.requires_grad_(False)
    l0.b.requires_grad_(False)

    # Load all 15 tasks and their training data.
    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=DEFAULT_DATA_ROOT,
        n_holdout_per_dataset=0,   # use full pool for prototype computation
    )
    specs = chained_15_specs()
    train_views = build_task_views(bundle, specs, split="train")

    # For each global class, collect L0 activations.
    # Chained-15: 30 classes total (MNIST 0..9, Fashion 10..19, EMNIST 20..29).
    n_classes = 30
    prototypes = torch.zeros(n_classes, L0_WIDTH)
    intra_class_std = torch.zeros(n_classes, L0_WIDTH)
    n_per_class = [0] * n_classes
    domains = ["MNIST", "Fashion", "EMNIST"]

    print(f"{'class':>5s}  {'domain':<8s}  {'n':>5s}  "
          f"{'||μ||':>8s}  {'mean σ':>8s}  {'feat-min σ':>10s}  {'feat-max σ':>10s}")

    with torch.no_grad():
        # Across all task views, find samples for each global class.
        for v in train_views:
            x_all, y_all = v.all_examples()
            for c in v.global_classes:
                mask = (y_all == c)
                x_c = x_all[mask]
                if x_c.shape[0] == 0:
                    continue
                # Forward through L0 only.
                z = F.linear(x_c, l0.W, l0.b)
                a = F.relu(z)            # (n, 128)
                prototypes[c] = a.mean(dim=0)
                intra_class_std[c] = a.std(dim=0)
                n_per_class[c] = int(x_c.shape[0])
                proto_norm = float(prototypes[c].norm())
                feat_std = intra_class_std[c]
                print(
                    f"  {c:>3d}  {domains[c // 10]:<8s}  "
                    f"{n_per_class[c]:>5d}  "
                    f"{proto_norm:>8.3f}  "
                    f"{float(feat_std.mean()):>8.4f}  "
                    f"{float(feat_std.min()):>10.4f}  "
                    f"{float(feat_std.max()):>10.4f}"
                )

    # Pairwise prototype distances.
    print()
    diffs = prototypes.unsqueeze(0) - prototypes.unsqueeze(1)
    pair_dist = diffs.norm(dim=-1)              # (30, 30)
    # Mask diagonal.
    eye_mask = torch.eye(n_classes, dtype=torch.bool)
    pair_dist_off = pair_dist.masked_fill(eye_mask, float("inf"))
    mean_inter = float(pair_dist.masked_fill(eye_mask, 0).sum() / (n_classes * (n_classes - 1)))
    min_pair_dist, min_idx = pair_dist_off.view(-1).min(dim=0)
    closest_a, closest_b = int(min_idx) // n_classes, int(min_idx) % n_classes
    pair_dist_off2 = pair_dist.masked_fill(eye_mask, -1.0)
    max_pair_dist, max_idx = pair_dist_off2.view(-1).max(dim=0)
    furthest_a, furthest_b = int(max_idx) // n_classes, int(max_idx) % n_classes

    # Aggregate within-class spread, converted to a comparable distance
    # metric. If features are independent, the L2 norm of a Gaussian
    # cloud with per-feature std σ̄ is roughly sqrt(d) × σ̄.
    mean_intra_per_feat = float(intra_class_std.mean())
    mean_intra_dist = mean_intra_per_feat * (L0_WIDTH ** 0.5)

    discriminability = mean_inter / mean_intra_dist if mean_intra_dist > 0 else float("inf")

    print("Aggregate statistics:")
    print(f"  Mean inter-class distance       = {mean_inter:.4f}")
    print(f"  Mean intra-class spread (σ̄)     = {mean_intra_per_feat:.4f}")
    print(f"  Mean intra-class dist (√d × σ̄) = {mean_intra_dist:.4f}")
    print(f"  Discriminability ratio          = {discriminability:.3f}")
    print()
    print(f"  Closest pair  : class {closest_a} ({domains[closest_a // 10]}) ↔ "
          f"class {closest_b} ({domains[closest_b // 10]})  d = {float(min_pair_dist):.4f}")
    print(f"  Furthest pair : class {furthest_a} ({domains[furthest_a // 10]}) ↔ "
          f"class {furthest_b} ({domains[furthest_b // 10]})  d = {float(max_pair_dist):.4f}")
    print()

    # Decision summary
    if discriminability > 2.0:
        verdict = ("PROTOTYPE REHEARSAL VIABLE — prototypes well-separated "
                   "relative to within-class spread.")
    elif discriminability > 1.0:
        verdict = ("PROTOTYPE REHEARSAL MARGINAL — prototypes separable but "
                   "noise envelope must be tuned carefully.")
    else:
        verdict = ("PROTOTYPE REHEARSAL UNSAFE — within-class spread "
                   "exceeds inter-class distance; prototype + Gaussian "
                   "noise would produce ambiguous samples. Need richer "
                   "representation (per-class cluster set, or stored-logit "
                   "pseudo-rehearsal).")
    print(verdict)
    print()

    # Domain-aware breakdown — within-domain vs cross-domain distances.
    print("Domain breakdown:")
    for d_idx, d_name in enumerate(domains):
        domain_classes = list(range(d_idx * 10, (d_idx + 1) * 10))
        within = pair_dist[domain_classes][:, domain_classes]
        within_off = within.masked_fill(
            torch.eye(10, dtype=torch.bool), float("nan"),
        )
        within_mean = float(within_off.nanmean())
        across = []
        for d2 in range(3):
            if d2 == d_idx:
                continue
            other_classes = list(range(d2 * 10, (d2 + 1) * 10))
            across.append(pair_dist[domain_classes][:, other_classes].mean())
        cross_mean = float(torch.stack(across).mean())
        print(f"  {d_name:<8s}  within-domain μ = {within_mean:.4f}  "
              f"cross-domain μ = {cross_mean:.4f}  "
              f"ratio = {cross_mean/within_mean:.3f}")
    print()

    return discriminability


if __name__ == "__main__":
    sys.exit(0 if main() > 0 else 1)
