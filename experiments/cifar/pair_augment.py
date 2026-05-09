"""Augment-not-replace probe.

Stage C's replace pattern (pair-mode picks among top-K) gives +0.2pp
full at -3pp task-aware. Test the hypothesis that pair-distance
contains a signal complementary to fused logits when ADDED rather
than substituted:

    final_logit[c] = fused_logit[c] - λ · normalize(distance[c])

Sweep λ. If +λ helps full and doesn't hurt task too much, distances
carry independent information; if not, the distance signal is
redundant with the fused logit.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from trioron.senses.organism import SensoryOrganism
from trioron.senses.calibrator import (
    BRANCH_FEATURE_NAMES, stack_branch_features,
    CalibratedRouter, AmbiguityHead,
    fuse_with_router, fused_confidence_aux,
)
from trioron.senses.pair import PrototypeBank, per_branch_distances
from experiments.cifar.datasets import (
    load_cifar100, SLICES, DEFAULT_DATA_ROOT,
)
from experiments.cifar.conductor_eval import _eval_logits


GREEDY_7 = ["eye", "color_smell", "frequency_print", "heat_diffusion",
            "skeleton", "taste", "pulse"]


def _normalize_per_branch(d: torch.Tensor) -> torch.Tensor:
    """Per-branch z-score over the class axis. Makes the magnitudes
    comparable across senses with different L0 scales."""
    mu = d.mean(dim=-1, keepdim=True)
    sd = d.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (d - mu) / sd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", choices=sorted(SLICES), default="full")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--donor-dir", default=None)
    parser.add_argument("--senses", nargs="+", default=GREEDY_7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--lambdas", type=float, nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    )
    args = parser.parse_args(argv)

    if args.donor_dir is None:
        sub = "cifar_donors" if args.slice == "first" else "cifar_donors_full"
        args.donor_dir = os.path.join(os.path.dirname(args.data_root), sub)
    donor_paths = [
        os.path.join(args.donor_dir, f"sense_donor_{s}.pt")
        for s in args.senses
    ]
    org = SensoryOrganism.from_sense_donors(donor_paths).eval()
    n_branches = len(org.branches)
    union_classes = org.union_classes
    class_groups = SLICES[args.slice]

    proto_path = os.path.join(args.donor_dir, "class_prototypes.pt")
    bank = PrototypeBank.load(proto_path)

    test_imgs, test_labs = load_cifar100(args.data_root, train=False)
    N = test_imgs.shape[0]

    print(f"slice={args.slice}  N={N}  λ sweep={args.lambdas}")

    # Compute fused single-look + per-branch distances.
    print("computing fused logits + per-branch distances...")
    t0 = time.time()
    padded_full = torch.empty(N, n_branches, len(union_classes), dtype=torch.float16)
    dist_full = torch.empty(N, n_branches, len(bank.class_order))
    with torch.no_grad():
        for i in range(0, N, args.batch_size):
            j = min(i + args.batch_size, N)
            fd = org.branch_features(test_imgs[i:j])
            padded_full[i:j] = fd["branch_logits_padded"].to(torch.float16)
            dist_full[i:j] = per_branch_distances(org, test_imgs[i:j], bank)
    uniform_g = padded_full.new_full((N, n_branches), 1.0 / n_branches)
    fused = fuse_with_router(padded_full.float(), uniform_g)               # (N, n_union)

    # Aggregate distances across branches (uniform mean), z-score per branch first.
    dist_norm = _normalize_per_branch(dist_full)                            # (N, N_b, C)
    dist_agg = dist_norm.mean(dim=1)                                        # (N, C)
    # Reorder to match union_classes — class_order may differ from union_classes.
    if bank.class_order != union_classes:
        col_remap = torch.tensor(
            [union_classes.index(c) for c in bank.class_order],
            dtype=torch.long,
        )
        # We want dist_agg in union_classes order. Currently it's in
        # bank.class_order. Build a permutation.
        permuted = torch.empty_like(dist_agg)
        for i, c in enumerate(bank.class_order):
            ui = union_classes.index(c)
            permuted[:, ui] = dist_agg[:, i]
        dist_agg = permuted
    print(f"  done in {time.time()-t0:.1f}s")

    base = _eval_logits(fused, test_labs, union_classes, class_groups)
    print(f"\nbaseline single-look: full={base['full']:.4f}  "
          f"task={base['task']:.4f}")

    print("\n=== fixed-λ augment: final = fused - λ · z(distance) ===")
    print(f"{'λ':>8s} {'full':>8s} {'task':>8s}  {'Δfull':>8s} {'Δtask':>8s}")
    for lam in args.lambdas:
        final = fused - lam * dist_agg
        m = _eval_logits(final, test_labs, union_classes, class_groups)
        print(f"{lam:>8.3f} {m['full']:>8.4f} {m['task']:>8.4f}  "
              f"{m['full']-base['full']:>+8.4f} {m['task']-base['task']:>+8.4f}")

    # Dynamic λ — gated by ambiguity head: λ_i = λ_max · (1 - ambig_p_i).
    # Low ambig (uncertain) → bigger λ → heavier reliance on pair distance.
    # High ambig (confident) → smaller λ → trust fused logits more.
    print("\n=== dynamic-λ augment: λ_i = λ_max · (1 - ambig_p_i) ===")
    calib_path = os.path.join(args.donor_dir, "sensory_calibrator.pt")
    if os.path.exists(calib_path):
        payload = torch.load(calib_path, map_location="cpu", weights_only=False)
        head = AmbiguityHead(n_branches=n_branches)
        head.load_state_dict(payload["ambig_state_dict"])
        head.eval()
        # Compute per-image features for the ambig head.
        feats_full = torch.empty(N, n_branches, len(BRANCH_FEATURE_NAMES))
        with torch.no_grad():
            for i in range(0, N, args.batch_size):
                j = min(i + args.batch_size, N)
                fd = org.branch_features(test_imgs[i:j])
                feats_full[i:j] = stack_branch_features(fd)
            fused_aux = fused_confidence_aux(fused)
            ambig_p = head(feats_full, fused_aux)
        print(f"  ambig_p range: [{ambig_p.min():.4f}, {ambig_p.max():.4f}]  "
              f"mean {ambig_p.mean():.4f}")
        print(f"{'λ_max':>8s} {'full':>8s} {'task':>8s}  {'Δfull':>8s} {'Δtask':>8s}")
        for lam_max in args.lambdas:
            lam_i = lam_max * (1.0 - ambig_p).unsqueeze(-1)               # (N, 1)
            final = fused - lam_i * dist_agg
            m = _eval_logits(final, test_labs, union_classes, class_groups)
            print(f"{lam_max:>8.3f} {m['full']:>8.4f} {m['task']:>8.4f}  "
                  f"{m['full']-base['full']:>+8.4f} {m['task']-base['task']:>+8.4f}")
    else:
        print(f"  (skipped — no calibrator at {calib_path})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
