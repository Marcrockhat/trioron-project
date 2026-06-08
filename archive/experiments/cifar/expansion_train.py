"""Stage-D expansion: grow a real trioron on top of the absorbed
sense organism via api.build_donor.

Architecture:

    image ──► SensoryOrganism (12 frozen sense branches) ──►
              (B, 12 × 100 = 1200) flattened branch logits
              ──►  NEW trioron ──► (B, 100) refined logits
                   • L0 random projection (frozen, configurable width)
                   • L1 grows during 20-task curriculum
                   • frustration → growth, dream-phase consolidation,
                     manifold archive — same primitives as any donor

The grown trioron IS the expansion layer. Not a head, not a vanilla
nn.Module — a real trioron whose input modality is the absorbed
sense organism's per-image branch logits.

Caching: sense-logit features get cached to disk on first run
(50K train + 10K test images × 1200 floats fp16 ≈ 144 MB) so
subsequent training/bench iterations don't re-run the organism.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from trioron.api import TaskData, TrioronConfig, AdvancedConfig, build_donor
from trioron.senses.organism import SensoryOrganism
from experiments.cifar.datasets import (
    load_cifar100, SLICES, DEFAULT_DATA_ROOT,
)


ALL_12 = sorted([
    "cortex", "color_smell", "frequency_print", "taste", "random_walk",
])


def _flat_logits_for(
    org: SensoryOrganism,
    images: torch.Tensor,
    *,
    batch_size: int,
    tag: str,
) -> torch.Tensor:
    """Run the organism, return (N, n_branches × n_union) flat logits."""
    N = images.shape[0]
    n_branches = len(org.branches)
    n_union = len(org.union_classes)
    out = torch.empty(N, n_branches * n_union, dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            fd = org.branch_features(images[i:j])
            padded = fd["branch_logits_padded"]                 # (B, N, n_union)
            out[i:j] = padded.reshape(j - i, -1)
            if (i // batch_size) % 20 == 0:
                print(f"  [{tag}] {j}/{N} ({j/N:.0%}) "
                      f"{time.time()-t0:.1f}s", flush=True)
    print(f"  [{tag}] done in {time.time()-t0:.1f}s "
          f"shape={tuple(out.shape)}", flush=True)
    return out


def _build_or_load_cache(
    org: SensoryOrganism,
    *,
    cache_path: str,
    data_root: str,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache flattened sense-logit features for CIFAR-100 train + test.

    Saves as {Xtr, ytr, Xte, yte} dict in fp16 on disk for compactness.
    The bench reads them back as fp32 contiguous tensors.
    """
    if os.path.exists(cache_path):
        print(f"loading sense-logit cache from {cache_path}")
        d = torch.load(cache_path, map_location="cpu", weights_only=False)
        return (d["Xtr"].float(), d["ytr"].long(),
                d["Xte"].float(), d["yte"].long())
    print(f"building sense-logit cache (will save to {cache_path}) ...")
    train_imgs, train_labs = load_cifar100(data_root, train=True)
    test_imgs, test_labs = load_cifar100(data_root, train=False)
    Xtr = _flat_logits_for(org, train_imgs, batch_size=batch_size, tag="train")
    Xte = _flat_logits_for(org, test_imgs,  batch_size=batch_size, tag="test")
    ytr = train_labs.long()
    yte = test_labs.long()
    torch.save({
        "kind": "sense_logit_cache",
        "n_branches": len(org.branches),
        "n_union": len(org.union_classes),
        "Xtr": Xtr.to(torch.float16),
        "ytr": ytr,
        "Xte": Xte.to(torch.float16),
        "yte": yte,
    }, cache_path)
    print(f"  cache saved ({os.path.getsize(cache_path)/1024**2:.1f} MB)")
    return Xtr, ytr, Xte, yte


def _build_tasks(
    Xtr: torch.Tensor, ytr: torch.Tensor,
    Xte: torch.Tensor, yte: torch.Tensor,
    class_groups: List[List[int]],
) -> List[TaskData]:
    tasks: List[TaskData] = []
    for i, group in enumerate(class_groups):
        group = [int(c) for c in group]
        m_tr = torch.zeros(ytr.shape[0], dtype=torch.bool)
        m_te = torch.zeros(yte.shape[0], dtype=torch.bool)
        for c in group:
            m_tr |= ytr == c
            m_te |= yte == c
        tasks.append(TaskData(
            name=f"cifar100_expansion_task{i}",
            X_train=Xtr[m_tr],
            y_train=ytr[m_tr],
            X_test=Xte[m_te],
            y_test=yte[m_te],
            classes=group,
        ))
    return tasks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", choices=sorted(SLICES), default="full")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--donor-dir", default=None)
    parser.add_argument("--senses", nargs="+", default=ALL_12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap-bytes", type=int, default=200_000)
    parser.add_argument(
        "--l0-width", type=int, default=256,
        help="L0 random-projection width. Default 256 (1200 → 256, "
             "~5x compression). Default for image inputs is 128 but "
             "1200-d inputs benefit from a wider random projection.",
    )
    parser.add_argument(
        "--cache-path", default=None,
        help="Where to cache sense-logit features. Default: "
             "outputs/cifar_donors_full/sense_logits_cache.pt",
    )
    parser.add_argument(
        "--out-path", default=None,
        help="Where to save the expansion donor. Default: "
             "outputs/cifar_donors_full/expansion_donor.pt",
    )
    args = parser.parse_args(argv)

    if args.donor_dir is None:
        sub = "cifar_donors" if args.slice == "first" else "cifar_donors_full"
        args.donor_dir = os.path.join(os.path.dirname(args.data_root), sub)
    if args.cache_path is None:
        args.cache_path = os.path.join(args.donor_dir, "sense_logits_cache.pt")
    if args.out_path is None:
        args.out_path = os.path.join(args.donor_dir, "expansion_donor.pt")

    donor_paths = [
        os.path.join(args.donor_dir, f"sense_donor_{s}.pt")
        for s in args.senses
    ]
    missing = [p for p in donor_paths if not os.path.exists(p)]
    if missing:
        print("missing donor checkpoints:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 2

    print(f"slice={args.slice}  donors ({len(args.senses)}): {args.senses}")
    org = SensoryOrganism.from_sense_donors(donor_paths).eval()
    n_branches = len(org.branches)
    n_union = len(org.union_classes)
    input_dim = n_branches * n_union
    print(f"input_dim = {n_branches} senses × {n_union} classes = {input_dim}")

    Xtr, ytr, Xte, yte = _build_or_load_cache(
        org,
        cache_path=args.cache_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
    )
    print(f"\ncache shapes: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    class_groups = SLICES[args.slice]
    tasks = _build_tasks(Xtr, ytr, Xte, yte, class_groups)
    print(f"curriculum: {len(tasks)} tasks of {len(class_groups[0])} classes "
          f"each")

    cfg = TrioronConfig(
        cap_bytes=args.cap_bytes,
        advanced=AdvancedConfig(l0_width=args.l0_width),
    )
    print(f"\ntrioron config: cap_bytes={args.cap_bytes}  "
          f"l0_width={args.l0_width}  epochs/task={args.epochs}  "
          f"seed={args.seed}")
    print(f"\n=== building trioron expansion donor ===")
    t0 = time.time()
    out = build_donor(
        label="cifar100_expansion",
        tasks=tasks,
        seed=args.seed,
        epochs_per_task=args.epochs,
        config=cfg,
        out_path=args.out_path,
    )
    print(f"\n[SAVE] {out}  ({os.path.getsize(out)/1024:.2f} KB)")
    print(f"total training time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
