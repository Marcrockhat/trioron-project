"""Stage-B trainer: fit CalibratedRouter + AmbiguityHead on top of a
frozen SensoryOrganism.

One-pass cache of per-branch features + padded logits over the
CIFAR-100 train set, then a fast in-memory training loop on those
cached tensors.

Pipeline:
  1. Load 7-sense organism (greedy-7).
  2. Cache (features (N, 7, 5), padded (N, 7, 100), labels (N,)) for
     train + test sets. Padded stored fp16 to keep memory ~70 MB.
  3. Train CalibratedRouter with CE loss on `fused = sum gates·padded`.
  4. Freeze router, compute fused_correct on train, train AmbiguityHead
     with BCE.
  5. Save heads + report test-set bench.
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
    CalibratedRouter, AmbiguityHead,
    fuse_with_router, fused_confidence_aux,
)
from experiments.cifar.datasets import (
    load_cifar100, SLICES, DEFAULT_DATA_ROOT,
)
from experiments.cifar.conductor_eval import _eval_logits


GREEDY_7 = ["eye", "color_smell", "frequency_print", "heat_diffusion",
            "skeleton", "taste", "pulse"]


def _build_cache(
    org: SensoryOrganism,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    tag: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache per-branch features and padded logits over the dataset.

    Returns:
      feats:  (N, n_branches, n_features)  fp32
      padded: (N, n_branches, n_union)     fp16 (memory-saving)
      labels: (N,)                          int64
    """
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
            feat_dict = org.branch_features(images[i:j])
            feats[i:j] = stack_branch_features(feat_dict)
            padded[i:j] = feat_dict["branch_logits_padded"].to(torch.float16)
            if (i // batch_size) % 20 == 0:
                done = j / N
                print(f"  [{tag}] {j}/{N} ({done:.0%})  "
                      f"{time.time()-t0:.1f}s", flush=True)
    print(f"  [{tag}] cache built in {time.time()-t0:.1f}s "
          f"feats={tuple(feats.shape)}  padded={tuple(padded.shape)} fp16",
          flush=True)
    return feats, padded, labels.long()


def _train_router(
    router: CalibratedRouter,
    train_feats: torch.Tensor,
    train_padded: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> List[Dict[str, float]]:
    """Train the router by CE on fused logits vs labels."""
    opt = torch.optim.Adam(router.parameters(), lr=lr)
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
            gates = router(f)
            fused = fuse_with_router(p, gates)
            loss = F.cross_entropy(fused, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.shape[0]
            correct += (fused.argmax(-1) == y).sum().item()
            n_seen += y.shape[0]
        log.append({"epoch": ep + 1,
                    "train_ce": loss_sum / n_seen,
                    "train_acc": correct / n_seen})
        print(f"  [router ep {ep+1:>2d}] CE={loss_sum/n_seen:.4f}  "
              f"acc={correct/n_seen:.4f}", flush=True)
    return log


def _train_ambig(
    head: AmbiguityHead,
    feats: torch.Tensor,
    padded: torch.Tensor,
    labels: torch.Tensor,
    *,
    router: CalibratedRouter,
    epochs: int,
    batch_size: int,
    lr: float,
) -> List[Dict[str, float]]:
    """Train ambiguity head by BCE against (fused-prediction == label)."""
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    N = feats.shape[0]
    log: List[Dict[str, float]] = []
    # Pre-compute fused_correct + fused_aux once with the frozen router.
    with torch.no_grad():
        fused_correct = torch.empty(N)
        fused_aux = torch.empty(N, head.n_fused_aux)
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            gates = router(feats[i:j])
            fused = fuse_with_router(padded[i:j].float(), gates)
            fused_correct[i:j] = (fused.argmax(-1) == labels[i:j]).float()
            fused_aux[i:j] = fused_confidence_aux(fused)
    pos_rate = fused_correct.mean().item()
    print(f"  [ambig] target positive rate (fused correct on train): "
          f"{pos_rate:.4f}", flush=True)

    for ep in range(epochs):
        perm = torch.randperm(N)
        loss_sum = 0.0
        n_seen = 0
        n_pos_pred = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            f = feats[idx]
            aux = fused_aux[idx]
            y = fused_correct[idx]
            p = head(f, aux)
            loss = F.binary_cross_entropy(p, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.shape[0]
            n_pos_pred += (p > 0.5).sum().item()
            n_seen += y.shape[0]
        log.append({"epoch": ep + 1, "train_bce": loss_sum / n_seen,
                    "frac_pred_pos": n_pos_pred / n_seen})
        print(f"  [ambig  ep {ep+1:>2d}] BCE={loss_sum/n_seen:.4f}  "
              f"frac>0.5={n_pos_pred/n_seen:.4f}", flush=True)

    # AUC on train (cheap monitor; held-out AUC reported in main bench).
    with torch.no_grad():
        p_all = torch.empty(N)
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            p_all[i:j] = head(feats[i:j], fused_aux[i:j])
    auc = _binary_auc(p_all, fused_correct)
    print(f"  [ambig] train AUC = {auc:.4f}", flush=True)
    return log


def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """ROC AUC via the rank-equivalent formula. Labels are 0/1 floats."""
    s = scores.detach().cpu().numpy()
    y = labels.detach().cpu().numpy().astype(bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    import numpy as np
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    sum_pos_ranks = ranks[y].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------------------------------------------------------------
# Bench helpers
# ---------------------------------------------------------------------


def _bench_modes(
    test_feats: torch.Tensor,
    test_padded: torch.Tensor,
    test_labels: torch.Tensor,
    union_classes: List[int],
    class_groups: List[List[int]],
    router: CalibratedRouter,
) -> None:
    """Compare uniform / soft / learned routing on the cached test set."""
    def metrics(gates: torch.Tensor) -> Dict[str, float]:
        fused = fuse_with_router(test_padded.float(), gates)
        return _eval_logits(fused, test_labels, union_classes, class_groups)

    # Uniform.
    n = test_feats.shape[1]
    uniform_g = test_feats.new_full((test_feats.shape[0], n), 1.0 / n)

    # Naive softmax-over-archive-loglik (the regressing mode from Stage A).
    naive_scores = test_feats[..., BRANCH_FEATURE_NAMES.index("archive_loglik")]
    naive_g = F.softmax(naive_scores, dim=-1)

    # Learned router.
    with torch.no_grad():
        learned_g = router(test_feats)

    print("\n=== test-set bench ===")
    print(f"{'mode':<28s} {'full':>8s} {'task':>8s}")
    for tag, g in [("uniform (parity)", uniform_g),
                   ("naive archive softmax", naive_g),
                   ("learned router", learned_g)]:
        m = metrics(g)
        print(f"{tag:<28s} {m['full']:>8.4f} {m['task']:>8.4f}")

    print("\n=== learned router gate stats (test mean) ===")
    mean_g = learned_g.mean(0).tolist()
    print(f"  mean per-branch gate: {[f'{x:.4f}' for x in mean_g]}")


def _bench_ambig(
    head: AmbiguityHead,
    feats: torch.Tensor,
    padded: torch.Tensor,
    labels: torch.Tensor,
    router: CalibratedRouter,
    *,
    batch_size: int,
) -> None:
    """Report ambig AUC on test, plus a coverage curve at thresholds."""
    N = feats.shape[0]
    with torch.no_grad():
        fused_correct = torch.empty(N)
        fused_aux = torch.empty(N, head.n_fused_aux)
        ambig_p = torch.empty(N)
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            gates = router(feats[i:j])
            fused = fuse_with_router(padded[i:j].float(), gates)
            fused_correct[i:j] = (fused.argmax(-1) == labels[i:j]).float()
            fused_aux[i:j] = fused_confidence_aux(fused)
            ambig_p[i:j] = head(feats[i:j], fused_aux[i:j])
    auc = _binary_auc(ambig_p, fused_correct)
    print(f"\n=== ambiguity head — test AUC = {auc:.4f} ===")
    print(f"{'thresh':>7s} {'kept':>7s} {'kept-acc':>9s} {'rejected':>9s} "
          f"{'rej-acc':>8s}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        keep = ambig_p >= t
        n_keep = int(keep.sum().item())
        n_rej = N - n_keep
        keep_acc = (
            fused_correct[keep].mean().item() if n_keep > 0 else float("nan")
        )
        rej_acc = (
            fused_correct[~keep].mean().item() if n_rej > 0 else float("nan")
        )
        print(f"{t:>7.2f} {n_keep:>7d} {keep_acc:>9.4f} {n_rej:>9d} "
              f"{rej_acc:>8.4f}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", choices=sorted(SLICES), default="full")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--donor-dir", default=None)
    parser.add_argument("--senses", nargs="+", default=GREEDY_7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--router-epochs", type=int, default=8)
    parser.add_argument("--router-lr", type=float, default=1e-2)
    parser.add_argument("--ambig-epochs", type=int, default=8)
    parser.add_argument("--ambig-lr", type=float, default=1e-2)
    parser.add_argument(
        "--out-path", default=None,
        help="Where to save trained heads (default: outputs/cifar_donors_full"
             "/sensory_calibrator.pt).",
    )
    args = parser.parse_args(argv)

    if args.donor_dir is None:
        sub = "cifar_donors" if args.slice == "first" else "cifar_donors_full"
        args.donor_dir = os.path.join(os.path.dirname(args.data_root), sub)

    donor_paths = [
        os.path.join(args.donor_dir, f"sense_donor_{s}.pt")
        for s in args.senses
    ]
    for p in donor_paths:
        if not os.path.exists(p):
            print(f"missing: {p}", file=sys.stderr)
            return 2

    print(f"slice={args.slice}  donors={args.senses}")
    org = SensoryOrganism.from_sense_donors(donor_paths).eval()
    n_branches = len(org.branches)
    union_classes = org.union_classes

    train_imgs, train_labs = load_cifar100(args.data_root, train=True)
    test_imgs, test_labs = load_cifar100(args.data_root, train=False)
    print(f"train: {train_imgs.shape}  test: {test_imgs.shape}")

    print("\n=== caching features (one full pass each) ===")
    train_feats, train_padded, train_y = _build_cache(
        org, train_imgs, train_labs,
        batch_size=args.batch_size, tag="train",
    )
    test_feats, test_padded, test_y = _build_cache(
        org, test_imgs, test_labs,
        batch_size=args.batch_size, tag="test",
    )

    router = CalibratedRouter(n_branches=n_branches)
    head = AmbiguityHead(n_branches=n_branches)
    print(f"\nrouter params: {router.num_params()}  "
          f"ambig params: {head.num_params()}")

    print("\n=== training router ===")
    _train_router(
        router, train_feats, train_padded, train_y,
        epochs=args.router_epochs,
        batch_size=args.batch_size,
        lr=args.router_lr,
    )

    print("\n=== training ambiguity head (router frozen) ===")
    _train_ambig(
        head, train_feats, train_padded, train_y,
        router=router,
        epochs=args.ambig_epochs,
        batch_size=args.batch_size,
        lr=args.ambig_lr,
    )

    # Bench on test.
    class_groups = SLICES[args.slice]
    _bench_modes(test_feats, test_padded, test_y,
                 union_classes, class_groups, router)
    _bench_ambig(head, test_feats, test_padded, test_y, router,
                 batch_size=args.batch_size)

    # Save.
    if args.out_path is None:
        args.out_path = os.path.join(args.donor_dir, "sensory_calibrator.pt")
    payload = {
        "kind": "sensory_calibrator",
        "n_branches": n_branches,
        "n_features": len(BRANCH_FEATURE_NAMES),
        "feature_names": BRANCH_FEATURE_NAMES,
        "senses": list(args.senses),
        "union_classes": union_classes,
        "router_state_dict": router.state_dict(),
        "ambig_state_dict": head.state_dict(),
    }
    torch.save(payload, args.out_path)
    size_kb = os.path.getsize(args.out_path) / 1024.0
    print(f"\n[SAVE] {args.out_path}  ({size_kb:.2f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
