"""One-shot pretrain script: build a 5-digit (0-4) trioron donor on
MNIST, save to disk, ready to be loaded + extended at runtime.

Usage:
    python3 -m examples.hf_space_demo.drawing.pretrain
        [--n-per-class 200] [--out donor_5digit.pt]
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from trioron.api import TaskData, TrioronConfig, build_donor  # noqa: E402

from examples.hf_space_demo.drawing.data import load_mnist_subset  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=200,
                   help="MNIST samples per digit at pretrain time")
    p.add_argument("--out", default=str(_HERE / "donor_5digit.pt"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cap-bytes", type=int, default=24_000)
    p.add_argument("--epochs-per-task", type=int, default=4)
    args = p.parse_args()

    classes = [0, 1, 2, 3, 4]
    print(f"[pretrain] loading MNIST subset for digits {classes} "
          f"({args.n_per_class} samples/class)...")
    t0 = time.time()
    subset = load_mnist_subset(
        classes=classes, n_per_class=args.n_per_class, seed=args.seed,
    )
    print(f"[pretrain] loaded in {time.time() - t0:.1f}s")

    tasks = [
        TaskData(
            name=f"digit_{c}",
            X_train=subset[c]["X_train"], y_train=subset[c]["y_train"],
            X_test=subset[c]["X_test"],   y_test=subset[c]["y_test"],
            classes=[c],
        )
        for c in classes
    ]

    cfg = TrioronConfig(cap_bytes=args.cap_bytes)
    out_path = Path(args.out)
    print(f"[pretrain] training donor (cap={args.cap_bytes} B, "
          f"{args.epochs_per_task} epochs/task, seed={args.seed})...")
    t0 = time.time()
    build_donor(
        tasks=tasks, label="drawing_5digit", out_path=out_path,
        seed=args.seed, epochs_per_task=args.epochs_per_task, config=cfg,
    )
    elapsed = time.time() - t0
    size_kb = out_path.stat().st_size / 1024
    print(f"[pretrain] done in {elapsed:.1f}s → {out_path}  ({size_kb:.1f} KB)")

    # Quick eval against the held-out test set so we know the donor isn't
    # garbage before we ship it.
    from trioron.api import load_organism
    org = load_organism(out_path)
    branch = org.branches[0]
    correct = total = 0
    for c in classes:
        Xt = subset[c]["X_test"]
        yt = subset[c]["y_test"]
        with torch.no_grad():
            z = org.project_l0(Xt)
            log_lik = branch.per_class_log_likelihood(z)  # (N, n_classes)
            pred_local = log_lik.argmax(dim=-1)
            # Map local → global via archive_classes
            pred_global = torch.tensor(
                [int(branch.archive_classes[int(i)]) for i in pred_local]
            )
        correct += int((pred_global == yt).sum())
        total += len(yt)
    acc = correct / max(1, total)
    print(f"[pretrain] held-out accuracy on digits 0-4: {acc:.3f} "
          f"({correct}/{total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
