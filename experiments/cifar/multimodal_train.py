"""Stage E — multimodal trioron.

A single trioron whose input is the *concatenation* of all 12
standardized sense readings (159-d total). Same primitives as any
trioron donor — small L0 random projection (128-d, unchanged),
grown L1, dream-phase consolidation, manifold archive — applied to
multi-modal sensory input.

This replaces the parallel-branches-+-coordinator pattern with a
single substrate. Senses do the perception lift (3072-d image → 159-d
features); trioron handles processing. L0=128 against 159-d input
is 1.24× compression — well within JL-comfort and far less aggressive
than the 1200-d expansion donor's 4.7× squeeze.

Storage: per-sense standardizers (already in donor checkpoints) +
~50 KB trioron substrate. NO change to L0_WIDTH; NO change to the
build_donor API; NO breakage of the "small L0" framing.
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
from trioron.senses import apply_sense, sense_dim, Standardizer
from experiments.cifar.datasets import (
    load_cifar100, SLICES, DEFAULT_DATA_ROOT,
)


ALL_12 = sorted([
    "cortex", "color_smell", "frequency_print", "taste", "random_walk",
])


def _load_donor_standardizer(path: str) -> Standardizer:
    """Pull a sense donor checkpoint's saved standardizer."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return Standardizer.from_dict(payload["standardizer"])


def _multimodal_features(
    images: torch.Tensor,
    senses: List[str],
    standardizers: List[Standardizer],
    *,
    batch_size: int,
    tag: str,
) -> torch.Tensor:
    """Concatenate per-sense standardized readings: (N, sum_d) float32."""
    N = images.shape[0]
    total_d = sum(sense_dim(s) for s in senses)
    out = torch.empty(N, total_d, dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            chunks = []
            for s, std in zip(senses, standardizers):
                raw = apply_sense(s, images[i:j])
                chunks.append((raw - std.mean) / std.std)
            out[i:j] = torch.cat(chunks, dim=-1)
            if (i // batch_size) % 20 == 0:
                print(f"  [{tag}] {j}/{N} ({j/N:.0%}) "
                      f"{time.time()-t0:.1f}s", flush=True)
    print(f"  [{tag}] done in {time.time()-t0:.1f}s "
          f"shape={tuple(out.shape)}", flush=True)
    return out


def _build_or_load_cache(
    senses: List[str],
    standardizers: List[Standardizer],
    *,
    cache_path: str,
    data_root: str,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if os.path.exists(cache_path):
        print(f"loading multimodal cache from {cache_path}")
        d = torch.load(cache_path, map_location="cpu", weights_only=False)
        return (d["Xtr"].float(), d["ytr"].long(),
                d["Xte"].float(), d["yte"].long())
    print(f"building multimodal cache (will save to {cache_path}) ...")
    train_imgs, train_labs = load_cifar100(data_root, train=True)
    test_imgs, test_labs = load_cifar100(data_root, train=False)
    Xtr = _multimodal_features(
        train_imgs, senses, standardizers,
        batch_size=batch_size, tag="train",
    )
    Xte = _multimodal_features(
        test_imgs, senses, standardizers,
        batch_size=batch_size, tag="test",
    )
    ytr = train_labs.long()
    yte = test_labs.long()
    torch.save({
        "kind": "multimodal_concat_cache",
        "senses": list(senses),
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
            name=f"cifar100_multimodal_task{i}",
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
    parser.add_argument("--n-passes", type=int, default=1,
                        help="Curriculum sweeps. Default 1; 3-5 for multi-pass.")
    parser.add_argument(
        "--l0-width", type=int, default=128,
        help="Default 128 — preserves the standard trioron L0 size. "
             "Don't change without weighing API/claim impact.",
    )
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--out-path", default=None)
    args = parser.parse_args(argv)

    if args.donor_dir is None:
        sub = "cifar_donors" if args.slice == "first" else "cifar_donors_full"
        args.donor_dir = os.path.join(os.path.dirname(args.data_root), sub)
    if args.cache_path is None:
        args.cache_path = os.path.join(args.donor_dir, "multimodal_concat_cache.pt")
    if args.out_path is None:
        args.out_path = os.path.join(args.donor_dir, "multimodal_donor.pt")

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

    print(f"slice={args.slice}  senses ({len(args.senses)}): {args.senses}")
    standardizers = [_load_donor_standardizer(p) for p in donor_paths]
    total_d = sum(sense_dim(s) for s in args.senses)
    print(f"input dim (concat of senses) = {total_d}  L0 = {args.l0_width} "
          f"(compression {total_d/args.l0_width:.2f}x)")

    Xtr, ytr, Xte, yte = _build_or_load_cache(
        args.senses, standardizers,
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
          f"seed={args.seed}  n_passes={args.n_passes}")
    print(f"\n=== building multimodal trioron donor ===")
    t0 = time.time()
    out = build_donor(
        label="cifar100_multimodal",
        tasks=tasks,
        seed=args.seed,
        epochs_per_task=args.epochs,
        config=cfg,
        out_path=args.out_path,
        n_passes=args.n_passes,
    )
    print(f"\n[SAVE] {out}  ({os.path.getsize(out)/1024:.2f} KB)")
    print(f"total training time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
