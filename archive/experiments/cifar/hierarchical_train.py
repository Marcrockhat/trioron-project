"""Stage F — hierarchical trioron stack.

Architecture (Rocky's design):

  image → 12 senses (159-d concat) →
      FEATURE trioron (chained 20 binary tasks: "is image in superclass i?") →
          20-d superclass feature vector →
              OBJECT trioron (curriculum on 100 fine classes) →
                  100-class output

Each layer is a real trioron with the standard primitives (small frozen
L0, growable L1, dream-phase consolidation, manifold archive). The
feature trioron lives in trioron's strongest validated regime —
chained binary tasks with sequential growth, exactly the chained-15
setup. The object trioron sits on a 20-d input that's small, dense,
and semantically meaningful.

Each binary feature task introduces 2 NEW classes:
  task i has classes [i (positive: superclass i), 20+i (negative)]
giving the feature trioron a 40-column head.

For each image, the 20-d feature vector is:
  feature[i] = softmax([logit_i, logit_{20+i}])[0]
            = exp(logit_i) / (exp(logit_i) + exp(logit_{20+i}))
i.e., the binary probability of "in superclass i" under task i's
2-class head pair.
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

from trioron.api import TaskData, TrioronConfig, AdvancedConfig, build_donor
from trioron.network import TrioronNetwork
from experiments.cifar.datasets import (
    SLICES, DEFAULT_DATA_ROOT,
)
from experiments.cifar.conductor_eval import _eval_logits


# CIFAR-100 fine-class index → superclass index. Verified by reading
# the raw cifar-100-python train file (each (fine, coarse) pair is
# consistent across all images, so the mapping is well-defined).
FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]


def _coarse_labels(fine_labels: torch.Tensor) -> torch.Tensor:
    fine_to_coarse = torch.tensor(FINE_TO_COARSE, dtype=torch.long)
    return fine_to_coarse[fine_labels.long()]


# ---------------------------------------------------------------------
# Feature trioron — chained 20 binary tasks
# ---------------------------------------------------------------------


def _build_feature_tasks(
    Xtr: torch.Tensor, coarse_tr: torch.Tensor,
    Xte: torch.Tensor, coarse_te: torch.Tensor,
    *,
    n_super: int = 20,
    seed: int = 42,
) -> List[TaskData]:
    """Each task i is a binary detection: "is image in superclass i?"
    Class space: pos = i, neg = 20+i. New head columns per task = 2.

    Negatives are subsampled to match positives per task (avoids
    1:19 class imbalance). The per-task subsample is reproducible via
    the seed argument.
    """
    rng = torch.Generator().manual_seed(int(seed))
    tasks: List[TaskData] = []
    for i in range(n_super):
        # Train
        pos_mask_tr = coarse_tr == i
        neg_pool_tr = (~pos_mask_tr).nonzero(as_tuple=True)[0]
        n_pos_tr = int(pos_mask_tr.sum().item())
        # Sample n_pos_tr negatives with the seeded rng
        perm = torch.randperm(neg_pool_tr.numel(), generator=rng)[:n_pos_tr]
        neg_idx_tr = neg_pool_tr[perm]
        pos_idx_tr = pos_mask_tr.nonzero(as_tuple=True)[0]
        idx_tr = torch.cat([pos_idx_tr, neg_idx_tr])
        X_tr = Xtr[idx_tr]
        y_tr = torch.empty(idx_tr.numel(), dtype=torch.long)
        y_tr[:n_pos_tr] = i
        y_tr[n_pos_tr:] = n_super + i

        # Test (use the same balanced setup so the per-task test metric
        # the bench reports is meaningful)
        pos_mask_te = coarse_te == i
        neg_pool_te = (~pos_mask_te).nonzero(as_tuple=True)[0]
        n_pos_te = int(pos_mask_te.sum().item())
        perm = torch.randperm(neg_pool_te.numel(), generator=rng)[:n_pos_te]
        neg_idx_te = neg_pool_te[perm]
        pos_idx_te = pos_mask_te.nonzero(as_tuple=True)[0]
        idx_te = torch.cat([pos_idx_te, neg_idx_te])
        X_te = Xte[idx_te]
        y_te = torch.empty(idx_te.numel(), dtype=torch.long)
        y_te[:n_pos_te] = i
        y_te[n_pos_te:] = n_super + i

        tasks.append(TaskData(
            name=f"superclass_{i:02d}",
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            y_test=y_te,
            classes=[i, n_super + i],   # 2 NEW classes per task
        ))
        if i < 3 or i == n_super - 1:
            print(f"  task {i:02d}: pos={n_pos_tr}/{n_pos_te}  "
                  f"neg subsampled to {n_pos_tr}/{n_pos_te}")
    return tasks


def _load_feature_donor(path: str) -> Tuple[TrioronNetwork, Dict]:
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


def _extract_superclass_features(
    net: TrioronNetwork,
    X: torch.Tensor,
    *,
    n_super: int = 20,
    batch_size: int = 512,
) -> torch.Tensor:
    """For each row in X, run the feature trioron, then reduce the
    40-col head to a 20-d superclass feature via softmax over each
    (i, n_super+i) pair: feature[i] = P(class i | task i pair)."""
    N = X.shape[0]
    feats = torch.empty(N, n_super, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            logits = net(X[i:j])                         # (B, 40)
            # Stack into (B, n_super, 2) — pair (logit[i], logit[n_super+i])
            stacked = torch.stack(
                [logits[:, :n_super], logits[:, n_super:2*n_super]],
                dim=-1,
            )                                            # (B, n_super, 2)
            probs = F.softmax(stacked, dim=-1)           # (B, n_super, 2)
            feats[i:j] = probs[:, :, 0]                  # P(in superclass i)
    return feats


# ---------------------------------------------------------------------
# Object trioron — curriculum on fine classes with 20-d input
# ---------------------------------------------------------------------


def _build_object_tasks(
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
            name=f"object_task{i}",
            X_train=Xtr[m_tr],
            y_train=ytr[m_tr],
            X_test=Xte[m_te],
            y_test=yte[m_te],
            classes=group,
        ))
    return tasks


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", choices=sorted(SLICES), default="full")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--donor-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-epochs", type=int, default=8)
    parser.add_argument("--feature-cap-bytes", type=int, default=200_000)
    parser.add_argument("--feature-l0-width", type=int, default=128)
    parser.add_argument("--object-epochs", type=int, default=8)
    parser.add_argument("--object-cap-bytes", type=int, default=200_000)
    parser.add_argument("--object-l0-width", type=int, default=128)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--feature-donor-path", default=None)
    parser.add_argument("--feature-cache-path", default=None)
    parser.add_argument("--object-donor-path", default=None)
    parser.add_argument(
        "--reuse-feature", action="store_true",
        help="Skip F.1; load feature donor from --feature-donor-path.",
    )
    parser.add_argument(
        "--object-input-mode", choices=["super-only", "concat"],
        default="super-only",
        help="'super-only' = object trioron sees only 20-d superclass "
             "features (limited; can't discriminate within superclass). "
             "'concat' = object trioron sees [159-d senses | 20-d super] "
             "= 179-d, with both perception and context cue.",
    )
    args = parser.parse_args(argv)

    if args.donor_dir is None:
        sub = "cifar_donors" if args.slice == "first" else "cifar_donors_full"
        args.donor_dir = os.path.join(os.path.dirname(args.data_root), sub)
    if args.cache_path is None:
        args.cache_path = os.path.join(args.donor_dir, "multimodal_concat_cache.pt")
    if args.feature_donor_path is None:
        args.feature_donor_path = os.path.join(args.donor_dir, "feature_donor.pt")
    if args.feature_cache_path is None:
        args.feature_cache_path = os.path.join(args.donor_dir, "superclass_feature_cache.pt")
    if args.object_donor_path is None:
        args.object_donor_path = os.path.join(args.donor_dir, "object_donor.pt")

    if not os.path.exists(args.cache_path):
        print(f"missing multimodal cache: {args.cache_path}", file=sys.stderr)
        print("  run experiments.cifar.multimodal_train first.", file=sys.stderr)
        return 2

    print(f"loading multimodal cache from {args.cache_path}")
    d = torch.load(args.cache_path, map_location="cpu", weights_only=False)
    Xtr = d["Xtr"].float()
    Xte = d["Xte"].float()
    ytr_fine = d["ytr"].long()
    yte_fine = d["yte"].long()
    print(f"  Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    coarse_tr = _coarse_labels(ytr_fine)
    coarse_te = _coarse_labels(yte_fine)
    n_super = 20
    n_fine = 100

    # ------------------------------------------------------------------
    # Stage F.1 — train feature trioron
    # ------------------------------------------------------------------

    if args.reuse_feature and os.path.exists(args.feature_donor_path):
        print(f"\n=== F.1: reusing existing feature donor ===\n  "
              f"{args.feature_donor_path}")
    else:
        print(f"\n=== F.1: building feature trioron ({n_super} binary tasks) ===")
        feat_tasks = _build_feature_tasks(
            Xtr, coarse_tr, Xte, coarse_te,
            n_super=n_super, seed=args.seed,
        )
        cfg = TrioronConfig(
            cap_bytes=args.feature_cap_bytes,
            advanced=AdvancedConfig(l0_width=args.feature_l0_width),
        )
        t0 = time.time()
        build_donor(
            label="cifar100_feature_superclass",
            tasks=feat_tasks,
            seed=args.seed,
            epochs_per_task=args.feature_epochs,
            config=cfg,
            out_path=args.feature_donor_path,
        )
        print(f"  feature donor saved → {args.feature_donor_path} "
              f"({os.path.getsize(args.feature_donor_path)/1024:.1f} KB)")
        print(f"  total time: {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Stage F.2 — extract 20-d superclass features
    # ------------------------------------------------------------------

    print(f"\n=== F.2: extracting {n_super}-d superclass features ===")
    feat_net, _ = _load_feature_donor(args.feature_donor_path)
    feat_arch = feat_net.n_nodes_per_layer()
    print(f"  feature net arch: {list(feat_arch)}")
    if feat_arch[-1] < 2 * n_super:
        print(f"  WARNING: feature head has {feat_arch[-1]} cols but "
              f"need at least {2*n_super}. Trioron may have skipped some "
              f"binary tasks.", file=sys.stderr)
    t0 = time.time()
    Ftr = _extract_superclass_features(feat_net, Xtr, n_super=n_super,
                                       batch_size=args.batch_size)
    Fte = _extract_superclass_features(feat_net, Xte, n_super=n_super,
                                       batch_size=args.batch_size)
    print(f"  Ftr={tuple(Ftr.shape)}  Fte={tuple(Fte.shape)}  "
          f"({time.time()-t0:.1f}s)")
    print(f"  feature stats — mean per dim: {Ftr.mean(0).tolist()[:5]}...")
    print(f"  feature stats — std per dim:  {Ftr.std(0).tolist()[:5]}...")

    # Save the feature cache.
    torch.save({
        "kind": "superclass_feature_cache",
        "n_super": n_super,
        "Ftr": Ftr.to(torch.float16),
        "ytr": ytr_fine,
        "Fte": Fte.to(torch.float16),
        "yte": yte_fine,
    }, args.feature_cache_path)
    print(f"  feature cache saved → {args.feature_cache_path} "
          f"({os.path.getsize(args.feature_cache_path)/1024:.1f} KB)")

    # ------------------------------------------------------------------
    # Stage F.3 — train object trioron on 20-d features
    # ------------------------------------------------------------------

    print(f"\n=== F.3: building object trioron ===")
    if args.object_input_mode == "concat":
        Ftr_in = torch.cat([Xtr, Ftr], dim=-1)
        Fte_in = torch.cat([Xte, Fte], dim=-1)
        print(f"  input mode = concat: senses({Xtr.shape[1]}) + "
              f"super({Ftr.shape[1]}) = {Ftr_in.shape[1]}")
    else:
        Ftr_in = Ftr
        Fte_in = Fte
        print(f"  input mode = super-only: {Ftr.shape[1]}-d")
    class_groups = SLICES[args.slice]
    obj_tasks = _build_object_tasks(Ftr_in, ytr_fine, Fte_in, yte_fine, class_groups)
    cfg = TrioronConfig(
        cap_bytes=args.object_cap_bytes,
        advanced=AdvancedConfig(l0_width=args.object_l0_width),
    )
    print(f"  curriculum: {len(obj_tasks)} tasks of "
          f"{len(class_groups[0])} fine classes each")
    print(f"  trioron config: cap_bytes={args.object_cap_bytes}  "
          f"l0_width={args.object_l0_width}  epochs/task={args.object_epochs}")
    t0 = time.time()
    build_donor(
        label="cifar100_object_finehead",
        tasks=obj_tasks,
        seed=args.seed,
        epochs_per_task=args.object_epochs,
        config=cfg,
        out_path=args.object_donor_path,
    )
    print(f"  object donor saved → {args.object_donor_path} "
          f"({os.path.getsize(args.object_donor_path)/1024:.1f} KB)")
    print(f"  total time: {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Stage F.4 — bench end-to-end
    # ------------------------------------------------------------------

    print(f"\n=== F.4: bench hierarchical stack on full test ===")
    obj_net, _ = _load_feature_donor(args.object_donor_path)
    chunks = []
    with torch.no_grad():
        for i in range(0, Fte_in.shape[0], args.batch_size):
            chunks.append(obj_net(Fte_in[i:i+args.batch_size]))
    logits = torch.cat(chunks, dim=0)
    union_classes = list(range(n_fine))
    m = _eval_logits(logits, yte_fine, union_classes, class_groups)
    print(f"  full = {m['full']:.4f}  task = {m['task']:.4f}")
    print(f"\n=== comparison ===")
    print(f"  12-sense uniform fusion : full 0.1498  task 0.6254")
    print(f"  multimodal trioron      : full 0.1210  task 0.5913")
    print(f"  hierarchical (this run) : full {m['full']:.4f}  task {m['task']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
