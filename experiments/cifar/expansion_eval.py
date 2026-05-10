"""Stage D.6 bench — compare uniform fusion vs Coordinator nn.Module
vs trioron expansion donor on CIFAR-100 test.

All three layer on top of the same 12 frozen sense branches, but
differ in the coordination head:

  uniform:      none — each branch gets gate 1/N
  Coordinator:  per-(branch, class) static + per-image dynamic linear
                (Stage D.3, ~1900 params)
  expansion:    real trioron grown via api.build_donor
                (random L0 → grown L1 → head, ~50K trainable)
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from trioron.network import TrioronNetwork
from trioron.senses.organism import SensoryOrganism
from trioron.senses.coordinator import Coordinator
from trioron.senses.calibrator import (
    BRANCH_FEATURE_NAMES, stack_branch_features, fuse_with_router,
)
from experiments.cifar.datasets import (
    SLICES, DEFAULT_DATA_ROOT,
)
from experiments.cifar.conductor_eval import _eval_logits


ALL_12 = sorted([
    "cortex", "color_smell", "frequency_print", "taste", "random_walk",
])


def _load_expansion_donor(path: str) -> TrioronNetwork:
    """Reconstruct the trioron expansion donor from its checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    n_nodes = list(payload["n_nodes_per_layer"])
    input_dim = int(payload["input_dim"])
    layer_specs = []
    prev = input_dim
    for i, n in enumerate(n_nodes):
        act = "linear" if i == len(n_nodes) - 1 else "relu"
        layer_specs.append((prev, int(n), act))
        prev = int(n)
    net = TrioronNetwork(layer_specs)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, payload


def _bench_uniform(
    test_padded: torch.Tensor,
    test_labels: torch.Tensor,
    union_classes: List[int],
    class_groups: List[List[int]],
):
    n_branches = test_padded.shape[1]
    g = test_padded.new_full((test_padded.shape[0], n_branches), 1.0 / n_branches)
    fused = fuse_with_router(test_padded.float(), g)
    return _eval_logits(fused, test_labels, union_classes, class_groups)


def _bench_coordinator(
    coord_path: str,
    test_padded: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    union_classes: List[int],
    class_groups: List[List[int]],
):
    if not os.path.exists(coord_path):
        return None
    payload = torch.load(coord_path, map_location="cpu", weights_only=False)
    coord = Coordinator(
        n_branches=int(payload["n_branches"]),
        n_classes=int(payload["n_classes"]),
        n_features=int(payload["n_features"]),
        dynamic=bool(payload.get("dynamic", True)),
    )
    coord.load_state_dict(payload["state_dict"])
    coord.eval()
    with torch.no_grad():
        logits = coord(test_padded.float(), test_feats)
    return _eval_logits(logits, test_labels, union_classes, class_groups)


def _bench_expansion(
    net: TrioronNetwork,
    Xte: torch.Tensor,
    test_labels: torch.Tensor,
    union_classes: List[int],
    class_groups: List[List[int]],
    *,
    batch_size: int,
):
    # net's head_size may be smaller than 100 if the donor's curriculum
    # only covered some classes. Pad to union space if needed by reading
    # classes_covered from the payload (caller passes union_classes).
    N = Xte.shape[0]
    chunks = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            chunks.append(net(Xte[i:i+batch_size].float()))
    head = torch.cat(chunks, dim=0)
    # The expansion donor's head columns ARE the union_classes order
    # because we trained it on the full curriculum [0..99].
    return _eval_logits(head, test_labels, union_classes, class_groups)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", choices=sorted(SLICES), default="full")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--donor-dir", default=None)
    parser.add_argument("--senses", nargs="+", default=ALL_12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--coord-path", default=None)
    parser.add_argument("--expansion-path", default=None)
    args = parser.parse_args(argv)

    if args.donor_dir is None:
        sub = "cifar_donors" if args.slice == "first" else "cifar_donors_full"
        args.donor_dir = os.path.join(os.path.dirname(args.data_root), sub)
    if args.cache_path is None:
        args.cache_path = os.path.join(args.donor_dir, "sense_logits_cache.pt")
    if args.coord_path is None:
        args.coord_path = os.path.join(args.donor_dir, "coordinator.pt")
    if args.expansion_path is None:
        args.expansion_path = os.path.join(args.donor_dir, "expansion_donor.pt")

    if not os.path.exists(args.cache_path):
        print(f"missing sense-logit cache: {args.cache_path}", file=sys.stderr)
        print("  run experiments.cifar.expansion_train first.", file=sys.stderr)
        return 2

    print(f"loading sense-logit cache from {args.cache_path}")
    d = torch.load(args.cache_path, map_location="cpu", weights_only=False)
    Xtr = d["Xtr"].float()
    Xte = d["Xte"].float()
    yte = d["yte"].long()
    n_branches = int(d["n_branches"])
    n_union = int(d["n_union"])
    print(f"  Xte={tuple(Xte.shape)}  branches={n_branches}  classes={n_union}")

    test_padded = Xte.view(Xte.shape[0], n_branches, n_union)
    union_classes = list(range(n_union))
    class_groups = SLICES[args.slice]

    # For the Coordinator, also need per-branch features (not in the cache).
    # If we're benching the linear coordinator and cache doesn't hold them,
    # warn — the linear Coordinator path will be skipped.
    test_feats = None
    if os.path.exists(args.coord_path):
        donor_paths = [
            os.path.join(args.donor_dir, f"sense_donor_{s}.pt")
            for s in args.senses
        ]
        org = SensoryOrganism.from_sense_donors(donor_paths).eval()
        # We need test branch_features. Cache them on the fly.
        from experiments.cifar.datasets import load_cifar100
        test_imgs, _ = load_cifar100(args.data_root, train=False)
        N = test_imgs.shape[0]
        test_feats = torch.empty(N, n_branches, len(BRANCH_FEATURE_NAMES))
        with torch.no_grad():
            for i in range(0, N, args.batch_size):
                j = min(i + args.batch_size, N)
                fd = org.branch_features(test_imgs[i:j])
                test_feats[i:j] = stack_branch_features(fd)

    print("\n=== bench ===")
    print(f"{'mode':<22s} {'full':>8s} {'task':>8s}  Δfull   Δtask  (Δ vs uniform)")

    base = _bench_uniform(test_padded, yte, union_classes, class_groups)
    print(f"{'uniform fusion':<22s} {base['full']:>8.4f} {base['task']:>8.4f}  "
          f"  +0.0000  +0.0000")

    if test_feats is not None:
        m = _bench_coordinator(
            args.coord_path, test_padded, test_feats, yte,
            union_classes, class_groups,
        )
        if m is not None:
            print(f"{'Coordinator (linear)':<22s} {m['full']:>8.4f} {m['task']:>8.4f}  "
                  f"  {m['full']-base['full']:+.4f}  {m['task']-base['task']:+.4f}")
    else:
        print("(skipped Coordinator — no checkpoint at "
              f"{args.coord_path})")

    if os.path.exists(args.expansion_path):
        net, payload = _load_expansion_donor(args.expansion_path)
        print(f"\nexpansion donor: arch={payload['n_nodes_per_layer']}  "
              f"params={sum(p.numel() for p in net.parameters())}  "
              f"l0_width={payload['n_nodes_per_layer'][0]}")
        m = _bench_expansion(
            net, Xte, yte, union_classes, class_groups,
            batch_size=args.batch_size,
        )
        print(f"\n{'trioron expansion':<22s} {m['full']:>8.4f} {m['task']:>8.4f}  "
              f"  {m['full']-base['full']:+.4f}  {m['task']-base['task']:+.4f}")
    else:
        print(f"(skipped expansion — no checkpoint at {args.expansion_path})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
