"""L1 — per-FINE-class L0 statistics. Tackles the disjoint problem.

Step 2-corrected. The previous probe aggregated L0 activations per
binary label (Living vs Non-living), which averages over 8 disjoint
visual sub-distributions per side. The δ between those mixture
centroids is structurally meaningless.

This probe instead computes per-FINE-class μ, σ in L0 space (one
centroid per CIFAR-100 fine class). It then reports:

  * The 16×16 pairwise cosine-similarity matrix between fine-class
    centroids — which pairs are *actually* confusable in trioron's
    L0 representation, regardless of taxonomic label.
  * Within-binary-group vs across-binary-group cos-sim distributions.
    If contrast at the binary level is real, across-group cos-sim
    should be lower than within-group on average.
  * The per-fine-class δ structure for a few hand-picked pairs:
      - wolf vs motorcycle (across Living/Non-living)
      - wolf vs dolphin (within Living, but visually disjoint)
      - rose vs oak_tree (within Living, both plants — should be
        more confusable)
      - chair vs bottle (within Non-living, both objects)

  Per-pair |δ|, separability, and cos-sim are reported directly —
  honest comparison of how the L0 representation handles real
  fine-class contrasts vs the spurious binary-label contrast we
  measured in the previous probe.
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from trioron.senses import apply_sense, Standardizer
from experiments.cifar.datasets import load_cifar100, DEFAULT_DATA_ROOT
from experiments.cifar.bench_taxonomy_l1 import (
    LIVING_NAMES, NON_LIVING_NAMES, _resolve_names_to_ids, _binary_subset,
)
from experiments.cifar.probe_taxonomy_l1_delta import (
    _load_donor, _l0_activations,
)


PAIRS_OF_INTEREST = [
    ("wolf", "motorcycle", "across (Living vs Non-living)"),
    ("wolf", "dolphin", "within Living (carnivore vs aquatic mammal)"),
    ("rose", "oak_tree", "within Living (flower vs tree)"),
    ("butterfly", "spider", "within Living (insect vs arachnid)"),
    ("chair", "bottle", "within Non-living (furniture vs container)"),
    ("motorcycle", "pickup_truck", "within Non-living (both vehicles)"),
    ("man", "oak_tree", "within Living (human vs plant)"),
    ("mushroom", "mountain", "across (fungus vs landscape)"),
]


def _per_fine_class_stats(
    h0: torch.Tensor, fine_labels: torch.Tensor, fine_ids: List[int],
) -> Dict[int, Dict[str, torch.Tensor]]:
    out = {}
    for c in fine_ids:
        mask = fine_labels == int(c)
        if mask.sum().item() == 0:
            continue
        h_c = h0[mask]
        out[int(c)] = {
            "mu": h_c.mean(dim=0),
            "sigma": h_c.std(dim=0).clamp_min(1e-6),
            "n": mask.sum().item(),
        }
    return out


def _cos(u: torch.Tensor, v: torch.Tensor) -> float:
    return (u @ v / (u.norm() * v.norm() + 1e-9)).item()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--donor-path",
        default="outputs/cifar_taxonomy/donor_l1_living_vs_nonliving.pt",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)

    net, payload = _load_donor(args.donor_path)
    sense_name = payload["sense"]
    std = Standardizer.from_dict(payload["standardizer"])
    print(f"[L1-fine] donor: {args.donor_path}")
    print(f"[L1-fine]   sense={sense_name}  arch={payload['n_nodes_per_layer']}")

    name_to_id = _resolve_names_to_ids(args.data_root)
    living_ids = [name_to_id[n] for n in LIVING_NAMES]
    nonliving_ids = [name_to_id[n] for n in NON_LIVING_NAMES]
    id_to_name = {v: k for k, v in name_to_id.items()}

    train_imgs, train_labs = load_cifar100(args.data_root, train=True)
    Xtr_raw, ytr_bin, ytr_fine = _binary_subset(
        train_imgs, train_labs, living_ids, nonliving_ids,
    )
    Xtr_sensed = std.transform(apply_sense(sense_name, Xtr_raw)).contiguous()
    h0 = _l0_activations(net, Xtr_sensed)        # (N, 128)
    print(f"[L1-fine] L0 activations: {tuple(h0.shape)}")

    all_ids = living_ids + nonliving_ids
    stats = _per_fine_class_stats(h0, ytr_fine, all_ids)
    n_l0 = h0.shape[1]
    print(f"[L1-fine] per-fine-class μ computed for "
          f"{len(stats)} classes (n=500/class typically)")

    # Order by name for readable matrix.
    ordered_ids = living_ids + nonliving_ids
    ordered_names = LIVING_NAMES + NON_LIVING_NAMES

    # Cosine-similarity matrix.
    print(f"\n[L1-fine] === pairwise cos-sim of μ in L0 (16×16) ===")
    print(f"           " + " ".join(f"{n[:8]:>8s}" for n in ordered_names))
    cos_matrix = torch.zeros(16, 16)
    for i, ci in enumerate(ordered_ids):
        cos_matrix[i, i] = 1.0
        for j, cj in enumerate(ordered_ids):
            if j <= i:
                continue
            c = _cos(stats[ci]["mu"], stats[cj]["mu"])
            cos_matrix[i, j] = c
            cos_matrix[j, i] = c
    for i, ni in enumerate(ordered_names):
        row = "  ".join(f"{cos_matrix[i, j].item():+.2f}" for j in range(16))
        print(f"  {ni[:9]:<9s}  {row}")

    # Within-binary-group vs across-binary-group statistics.
    # Living block = first 8x8 (excluding diagonal), non-living block =
    # last 8x8, across = first 8 rows × last 8 cols.
    L = 8
    within_living = []
    within_nonliving = []
    across = []
    for i in range(L):
        for j in range(L):
            if j > i:
                within_living.append(cos_matrix[i, j].item())
            if j > i:
                within_nonliving.append(cos_matrix[L + i, L + j].item())
            across.append(cos_matrix[i, L + j].item())
    wl = torch.tensor(within_living)
    wn = torch.tensor(within_nonliving)
    ac = torch.tensor(across)
    print(f"\n[L1-fine] === binary-group cos-sim distributions ===")
    print(f"  within Living      (n={len(within_living):>3d})  "
          f"mean={wl.mean().item():+.3f}  median={wl.median().item():+.3f}  "
          f"min={wl.min().item():+.3f}  max={wl.max().item():+.3f}")
    print(f"  within Non-living  (n={len(within_nonliving):>3d})  "
          f"mean={wn.mean().item():+.3f}  median={wn.median().item():+.3f}  "
          f"min={wn.min().item():+.3f}  max={wn.max().item():+.3f}")
    print(f"  across groups      (n={len(across):>3d})  "
          f"mean={ac.mean().item():+.3f}  median={ac.median().item():+.3f}  "
          f"min={ac.min().item():+.3f}  max={ac.max().item():+.3f}")
    print(f"  contrast spread (within − across, by mean): "
          f"{((wl.mean() + wn.mean()) / 2 - ac.mean()).item():+.3f}")
    print(f"    positive means within-group avg cos-sim is higher than "
          f"across-group → genuine binary structure in L0")
    print(f"    near-zero means binary boundary doesn't show up in L0")

    # Pairs of interest.
    print(f"\n[L1-fine] === per-pair δ statistics (hand-picked pairs) ===")
    print(f"  {'pair':<28s}  {'cos':>6s}  {'‖δ‖':>6s}  "
          f"{'maxsep':>7s}  {'group':<24s}")
    for a, b, group in PAIRS_OF_INTEREST:
        ca, cb = name_to_id[a], name_to_id[b]
        mu_a, mu_b = stats[ca]["mu"], stats[cb]["mu"]
        sig_a, sig_b = stats[ca]["sigma"], stats[cb]["sigma"]
        delta = mu_a - mu_b
        cos_ab = _cos(mu_a, mu_b)
        sep = (delta.abs() / (sig_a + sig_b + 1e-9))
        max_sep = sep.max().item()
        print(f"  {a:>10s} ↔ {b:<14s}  {cos_ab:>+6.3f}  "
              f"{delta.norm().item():>6.3f}  {max_sep:>7.3f}  {group}")

    print(f"\n[L1-fine] interpretation:")
    print(f"  (compare to previous binary-aggregate probe: "
          f"cos(μ_l, μ_n)=+0.92, ‖δ‖=2.47, max sep=0.35)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
