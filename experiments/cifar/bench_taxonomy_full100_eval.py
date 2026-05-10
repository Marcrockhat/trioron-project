"""Full-100 hierarchical 100-way fine-class evaluation.

L1 (k-way macro) → routes to L2 expert per macrocluster.
Singletons (clusters of size 1) require no L2 expert: their macro
prediction maps directly to that singleton's fine class.
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from trioron.network import TrioronNetwork
from trioron.senses import apply_sense, Standardizer
from experiments.cifar.datasets import DEFAULT_DATA_ROOT, load_cifar100
from experiments.cifar.bench_taxonomy_l1 import _resolve_names_to_ids


def _load(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    n_nodes = list(payload["n_nodes_per_layer"])
    specs = []
    prev = int(payload["input_dim"])
    for i, n in enumerate(n_nodes):
        act = "linear" if i == len(n_nodes) - 1 else "relu"
        specs.append((prev, int(n), act))
        prev = int(n)
    net = TrioronNetwork(specs)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-file",
                        default="outputs/cifar_taxonomy/cluster_assignment_full100_k20.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--l1-donor", default=None)
    parser.add_argument("--expert-suffix", default="",
                        help="Suffix on expert file names (e.g. '_contrastive').")
    args = parser.parse_args(argv)

    ca = torch.load(args.cluster_file, map_location="cpu", weights_only=False)
    K = ca["k"]
    fine_to_cluster = ca["fine_to_cluster"]
    if args.l1_donor is None:
        args.l1_donor = (f"outputs/cifar_taxonomy/donor_full100_l1_k{K}_"
                         f"seed{args.seed}.pt")

    l1_net, l1_p = _load(args.l1_donor)
    sense_name = l1_p["sense"]
    l1_std = Standardizer.from_dict(l1_p["standardizer"])
    print(f"[full100-eval] L1: {args.l1_donor}  arch={l1_p['n_nodes_per_layer']}")
    print(f"[full100-eval]   k={K}  expert_suffix={args.expert_suffix!r}")

    # Load L2 experts (one per non-singleton cluster).
    name_to_id = _resolve_names_to_ids(args.data_root)
    l2_experts: Dict[int, dict] = {}
    singletons: Dict[int, int] = {}      # cluster_id → fine_id
    for cid, names in enumerate(ca["clusters"]):
        if len(names) == 1:
            singletons[cid] = name_to_id[names[0]]
            continue
        path = (f"outputs/cifar_taxonomy/donor_full100_l2_c{cid:02d}_"
                f"{len(names)}way_seed{args.seed}{args.expert_suffix}.pt")
        if not os.path.exists(path):
            print(f"  WARNING: missing expert for c{cid:02d} at {path}")
            continue
        net, payload = _load(path)
        std = Standardizer.from_dict(payload["standardizer"])
        l2_experts[cid] = {
            "net": net, "std": std,
            "fine_names": payload["fine_class_names"],
            "fine_ids": payload["fine_class_ids"],
            "K": int(payload["n_nodes_per_layer"][-1]),
        }
    print(f"[full100-eval] loaded {len(l2_experts)} experts; "
          f"{len(singletons)} singletons (handled directly)")

    # Test set.
    test_imgs, test_labs = load_cifar100(args.data_root, train=False)
    n_test = test_imgs.shape[0]
    print(f"[full100-eval] test: {n_test} imgs across "
          f"{len(set(test_labs.tolist()))} fine classes")
    raw_test = apply_sense(sense_name, test_imgs)
    Xte_l1 = l1_std.transform(raw_test).contiguous()

    # L1 forward.
    with torch.no_grad():
        l1_logits = l1_net(Xte_l1)
        l1_pred = l1_logits.argmax(dim=1)
    l1_macro_acc = (l1_pred ==
                    torch.tensor(fine_to_cluster, dtype=torch.long)[
                        test_labs.long()]).float().mean().item()
    print(f"[full100-eval] L1 macro acc: {l1_macro_acc:.4f}")

    # L2 forwards (per expert, full batch).
    l2_pred_local: Dict[int, torch.Tensor] = {}
    for cid, info in l2_experts.items():
        Xte_l2 = info["std"].transform(raw_test).contiguous()
        with torch.no_grad():
            l2_pred_local[cid] = info["net"](Xte_l2).argmax(dim=1)

    # Hierarchical fine prediction.
    pred_fine_id = torch.zeros(n_test, dtype=test_labs.dtype)
    for i in range(n_test):
        cid = int(l1_pred[i].item())
        if cid in l2_experts:
            local = int(l2_pred_local[cid][i].item())
            pred_fine_id[i] = l2_experts[cid]["fine_ids"][local]
        elif cid in singletons:
            pred_fine_id[i] = singletons[cid]
        else:
            pred_fine_id[i] = -1   # missing expert; will register as wrong

    overall_acc = (pred_fine_id == test_labs).float().mean().item()
    print(f"\n[full100-eval] === results ===")
    print(f"  100-way fine acc:   {overall_acc:.4f}")
    print(f"  chance (1/100):     0.0100")
    print(f"  margin over chance: {overall_acc - 0.01:+.4f}  "
          f"({overall_acc / 0.01:.1f}× chance)")

    # Error decomposition.
    n_correct = 0
    n_l1_miss = 0
    n_l2_miss = 0
    n_singleton_miss = 0
    for i in range(n_test):
        true_fine = int(test_labs[i].item())
        true_cluster = fine_to_cluster[true_fine]
        pred_cluster = int(l1_pred[i].item())
        if pred_cluster != true_cluster:
            n_l1_miss += 1
        elif int(pred_fine_id[i].item()) == true_fine:
            n_correct += 1
        elif true_cluster in singletons:
            # Should never happen — singleton is uniquely identified by macro
            n_singleton_miss += 1
        else:
            n_l2_miss += 1
    print(f"\n[full100-eval] === error decomposition ===")
    print(f"  correct (L1 + L2 right):       {n_correct:>5d}/{n_test} = {n_correct/n_test:.4f}")
    print(f"  L1 routing miss:               {n_l1_miss:>5d}/{n_test} = {n_l1_miss/n_test:.4f}")
    print(f"  L1 right but L2 fine miss:     {n_l2_miss:>5d}/{n_test} = {n_l2_miss/n_test:.4f}")

    # Per-cluster fine accuracy.
    print(f"\n[full100-eval] === per-cluster fine accuracy ===")
    print(f"  {'cid':>4s}  {'size':>4s}  {'L1 recall':>10s}  {'fine acc':>9s}  members[:3]")
    fine_to_cluster_t = torch.tensor(fine_to_cluster, dtype=torch.long)
    for cid, names in enumerate(ca["clusters"]):
        cluster_ids = [name_to_id[n] for n in names]
        m = torch.zeros(n_test, dtype=torch.bool)
        for c in cluster_ids:
            m |= test_labs == c
        n_c = int(m.sum().item())
        if n_c == 0:
            continue
        l1_recall = (l1_pred[m] == cid).float().mean().item()
        fine_acc = (pred_fine_id[m] == test_labs[m]).float().mean().item()
        print(f"  c{cid:02d}  {len(names):>4d}  {l1_recall:>10.4f}  "
              f"{fine_acc:>9.4f}  {names[:3]}{'…' if len(names) > 3 else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
