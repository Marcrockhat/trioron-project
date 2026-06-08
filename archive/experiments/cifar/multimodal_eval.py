"""Stage E.3 bench — multimodal trioron vs 12-sense uniform fusion.

Loads the trained multimodal donor, runs it on the full CIFAR-100
test set, computes full + task-aware accuracy, prints the
storage-vs-accuracy comparison.
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from trioron.network import TrioronNetwork
from experiments.cifar.datasets import (
    SLICES, DEFAULT_DATA_ROOT,
)
from experiments.cifar.conductor_eval import _eval_logits


def _load_donor(path: str):
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", choices=sorted(SLICES), default="full")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--donor-dir", default=None)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--donor-path", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)

    if args.donor_dir is None:
        sub = "cifar_donors" if args.slice == "first" else "cifar_donors_full"
        args.donor_dir = os.path.join(os.path.dirname(args.data_root), sub)
    if args.cache_path is None:
        args.cache_path = os.path.join(args.donor_dir, "multimodal_concat_cache.pt")
    if args.donor_path is None:
        args.donor_path = os.path.join(args.donor_dir, "multimodal_donor.pt")

    if not os.path.exists(args.cache_path):
        print(f"missing multimodal cache: {args.cache_path}", file=sys.stderr)
        return 2
    if not os.path.exists(args.donor_path):
        print(f"missing donor: {args.donor_path}", file=sys.stderr)
        return 2

    print(f"loading multimodal cache from {args.cache_path}")
    d = torch.load(args.cache_path, map_location="cpu", weights_only=False)
    Xte = d["Xte"].float()
    yte = d["yte"].long()
    print(f"  Xte={tuple(Xte.shape)}")

    net, payload = _load_donor(args.donor_path)
    print(f"\nmultimodal donor:")
    print(f"  arch = {payload['n_nodes_per_layer']}")
    print(f"  input_dim = {payload['input_dim']}")
    print(f"  total params = {sum(p.numel() for p in net.parameters())}")
    print(f"  donor file size = {os.path.getsize(args.donor_path)/1024:.2f} KB")

    union_classes = list(range(100))
    class_groups = SLICES[args.slice]

    chunks = []
    with torch.no_grad():
        for i in range(0, Xte.shape[0], args.batch_size):
            chunks.append(net(Xte[i:i+args.batch_size]))
    logits = torch.cat(chunks, dim=0)
    m = _eval_logits(logits, yte, union_classes, class_groups)
    print(f"\n=== full test set (10000 images, 100 classes) ===")
    print(f"  full accuracy: {m['full']:.4f}")
    print(f"  task-aware:    {m['task']:.4f}")

    # Comparison summary
    print("\n=== comparison ===")
    print(f"{'config':<32s} {'full':>8s} {'task':>8s} "
          f"{'storage_BF16':>12s}")
    print(f"{'12-sense uniform fusion':<32s} {0.1498:>8.4f} {0.6254:>8.4f} "
          f"{'~880 KB':>12s}  (12 × ~73 KB)")
    print(f"{'multimodal trioron':<32s} {m['full']:>8.4f} {m['task']:>8.4f} "
          f"{'~71 KB':>12s}  (this run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
