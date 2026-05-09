"""Stage-D trainer: fit Coordinator over a frozen SensoryOrganism's
12 (or 7) branch logits + per-branch features.

Same one-pass cache pattern as Stage B's calibrator_train. The
coordinator is bigger (per-(branch, class) static weights + per-
branch dynamic correction) so it can express "branch X is good at
class Y" patterns the 42-param router could not.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from trioron.senses.organism import SensoryOrganism
from trioron.senses.calibrator import (
    BRANCH_FEATURE_NAMES, stack_branch_features,
)
from trioron.senses.coordinator import Coordinator
from experiments.cifar.datasets import (
    load_cifar100, SLICES, DEFAULT_DATA_ROOT,
)
from experiments.cifar.conductor_eval import _eval_logits


GREEDY_7 = ["eye", "color_smell", "frequency_print", "heat_diffusion",
            "skeleton", "taste", "pulse"]
ALL_12 = sorted([
    "eye", "color_smell", "frequency_print", "heat_diffusion",
    "skeleton", "taste", "pulse",
    "mass_moment", "proprioception", "sonification",
    "random_walk", "echolocation",
])


def _build_cache(
    org: SensoryOrganism,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    tag: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = images.shape[0]
    n_branches = len(org.branches)
    n_features = len(BRANCH_FEATURE_NAMES)
    n_union = len(org.union_classes)
    feats = torch.empty(N, n_branches, n_features, dtype=torch.float32)
    padded = torch.empty(N, n_branches, n_union, dtype=torch.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            fd = org.branch_features(images[i:j])
            feats[i:j] = stack_branch_features(fd)
            padded[i:j] = fd["branch_logits_padded"].to(torch.float16)
            if (i // batch_size) % 20 == 0:
                print(f"  [{tag}] {j}/{N} ({j/N:.0%})  "
                      f"{time.time()-t0:.1f}s", flush=True)
    print(f"  [{tag}] cache built in {time.time()-t0:.1f}s "
          f"feats={tuple(feats.shape)}  padded={tuple(padded.shape)} fp16",
          flush=True)
    return feats, padded, labels.long()


def _train_coord(
    coord: Coordinator,
    train_feats: torch.Tensor,
    train_padded: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> List[Dict[str, float]]:
    opt = torch.optim.Adam(coord.parameters(), lr=lr, weight_decay=weight_decay)
    N = train_feats.shape[0]
    log: List[Dict[str, float]] = []
    for ep in range(epochs):
        perm = torch.randperm(N)
        loss_sum = 0.0
        correct = 0
        n_seen = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            f = train_feats[idx]
            p = train_padded[idx].float()
            y = train_labels[idx]
            logits = coord(p, f)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.shape[0]
            correct += (logits.argmax(-1) == y).sum().item()
            n_seen += y.shape[0]
        log.append({"epoch": ep + 1,
                    "train_ce": loss_sum / n_seen,
                    "train_acc": correct / n_seen})
        print(f"  [coord ep {ep+1:>2d}] CE={loss_sum/n_seen:.4f}  "
              f"acc={correct/n_seen:.4f}", flush=True)
    return log


def _eval_test(
    coord: Coordinator,
    test_feats: torch.Tensor,
    test_padded: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    union_classes: List[int],
    class_groups: List[List[int]],
    batch_size: int,
    tag: str,
) -> Dict[str, float]:
    N = test_feats.shape[0]
    chunks = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            chunks.append(coord(test_padded[i:j].float(), test_feats[i:j]))
    logits = torch.cat(chunks, dim=0)
    m = _eval_logits(logits, test_labels, union_classes, class_groups)
    print(f"  [eval {tag}] full={m['full']:.4f}  task={m['task']:.4f}",
          flush=True)
    return m


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", choices=sorted(SLICES), default="full")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--donor-dir", default=None)
    parser.add_argument(
        "--senses", nargs="+", default=ALL_12,
        help=f"Default = ALL_12 ({len(ALL_12)} senses).",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--weight-decay", type=float, default=0.0,
        help="Default 0 — weight decay pulls static_W away from the "
             "1/N uniform-parity init toward zero (which destroys the "
             "architecture). Don't enable unless you've reinitialized "
             "static_W to zero.",
    )
    parser.add_argument("--no-dynamic", action="store_true",
                        help="Disable per-image dynamic correction (static-only).")
    parser.add_argument("--out-path", default=None)
    args = parser.parse_args(argv)

    if args.donor_dir is None:
        sub = "cifar_donors" if args.slice == "first" else "cifar_donors_full"
        args.donor_dir = os.path.join(os.path.dirname(args.data_root), sub)

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
    union_classes = org.union_classes
    class_groups = SLICES[args.slice]
    n_classes = len(union_classes)

    train_imgs, train_labs = load_cifar100(args.data_root, train=True)
    test_imgs, test_labs = load_cifar100(args.data_root, train=False)
    print(f"train: {train_imgs.shape}  test: {test_imgs.shape}")

    print("\n=== caching features ===")
    train_feats, train_padded, train_y = _build_cache(
        org, train_imgs, train_labs,
        batch_size=args.batch_size, tag="train",
    )
    test_feats, test_padded, test_y = _build_cache(
        org, test_imgs, test_labs,
        batch_size=args.batch_size, tag="test",
    )

    # Uniform parity baseline (sanity).
    uniform_W = torch.full((n_branches, n_classes), 1.0 / n_branches)
    uniform_logits = (test_padded.float() * uniform_W.unsqueeze(0)).sum(dim=1)
    base = _eval_logits(uniform_logits, test_y, union_classes, class_groups)
    print(f"\nbaseline uniform fusion: full={base['full']:.4f}  "
          f"task={base['task']:.4f}")

    coord = Coordinator(
        n_branches=n_branches,
        n_classes=n_classes,
        n_features=len(BRANCH_FEATURE_NAMES),
        dynamic=not args.no_dynamic,
    )
    print(f"\ncoordinator: {coord.num_params()} params  "
          f"(static={coord.static_W.numel()}+{coord.bias.numel()}, "
          f"dyn={'on' if coord.dynamic else 'off'})")

    # Verify init is at exact uniform parity.
    with torch.no_grad():
        init_logits = coord(test_padded[:512].float(), test_feats[:512])
        ref_logits = uniform_logits[:512]
        max_dev = (init_logits - ref_logits).abs().max().item()
    print(f"init parity check: max|coord - uniform| = {max_dev:.2e}")

    print("\n=== training coordinator ===")
    _train_coord(
        coord, train_feats, train_padded, train_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("\n=== test eval ===")
    m_dyn = _eval_test(
        coord, test_feats, test_padded, test_y,
        union_classes=union_classes, class_groups=class_groups,
        batch_size=args.batch_size, tag="full coord",
    )
    print(f"  Δfull={m_dyn['full']-base['full']:+.4f}  "
          f"Δtask={m_dyn['task']-base['task']:+.4f}  vs uniform")

    if coord.dynamic:
        # Static-only ablation: zero out the dyn pathway by feeding zeros.
        zero_feats = torch.zeros_like(test_feats)
        m_static = _eval_test(
            coord, zero_feats, test_padded, test_y,
            union_classes=union_classes, class_groups=class_groups,
            batch_size=args.batch_size, tag="static only (dyn=0)",
        )
        print(f"  Δfull={m_static['full']-base['full']:+.4f}  "
              f"Δtask={m_static['task']-base['task']:+.4f}  vs uniform")

    # Save.
    if args.out_path is None:
        args.out_path = os.path.join(args.donor_dir, "coordinator.pt")
    payload = {
        "kind": "sensory_coordinator",
        "n_branches": n_branches,
        "n_classes": n_classes,
        "n_features": len(BRANCH_FEATURE_NAMES),
        "dynamic": coord.dynamic,
        "senses": list(args.senses),
        "union_classes": union_classes,
        "state_dict": coord.state_dict(),
    }
    torch.save(payload, args.out_path)
    print(f"\n[SAVE] {args.out_path}  "
          f"({os.path.getsize(args.out_path)/1024:.2f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
