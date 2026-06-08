"""Committee eval over n=3 seeds of the full-100 hierarchical pipeline.

Each seed has its own (L1 20-way macroclassifier, 17 L2 experts, 3
singletons). For each test image we compute, per seed, a 100-dim fine-
class probability vector via soft routing:

    P(fine_i | image, seed s) =
        Σ_c  P(macro_c | image, seed s) * P(fine_i | image, macro_c, seed s)

  * P(macro_c | image, seed s) = softmax(L1_s(image))[c]
  * P(fine_i | image, macro_c, seed s) =
        - 1 if c is a singleton and fine_i is its only member
        - softmax(L2_s_c(image))[local_idx_of_fine_i_in_c] if fine_i ∈ c
        - 0 otherwise

This 100-d distribution is well-defined per seed. Three aggregations:

    hard-vote:       per-seed argmax (hard-routing pipeline), majority
                     vote across seeds (ties → seed=lowest mean-prob).
    mean-softmax:    average the 100-d soft-routing distributions across
                     seeds, argmax.
    mean-logit:      sum the 100-d soft-routing log-probs across seeds,
                     argmax. (Equivalent to geometric mean of probs.)

Reports per-aggregation accuracy and Δ vs the n=3 single-seed mean.
"""
from __future__ import annotations
import argparse
import os
import sys
from collections import Counter
from typing import Dict, List

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

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


def _load_seed_pipeline(seed: int, K: int, ca: dict, suffix: str = "",
                        name_to_id: dict = None):
    """Load one seed's full pipeline: L1 + 17 L2 experts + 3 singletons."""
    out = {"seed": seed}
    l1_path = (f"outputs/cifar_taxonomy/donor_full100_l1_k{K}_seed{seed}.pt")
    l1_net, l1_p = _load(l1_path)
    out["l1_net"] = l1_net
    out["l1_std"] = Standardizer.from_dict(l1_p["standardizer"])
    out["sense"] = l1_p["sense"]
    out["experts"] = {}      # cluster_id → {net, std, fine_ids, K, local_idx}
    out["singletons"] = {}   # cluster_id → fine_id
    for cid, names in enumerate(ca["clusters"]):
        if len(names) == 1:
            out["singletons"][cid] = name_to_id[names[0]]
            continue
        path = (f"outputs/cifar_taxonomy/donor_full100_l2_c{cid:02d}_"
                f"{len(names)}way_seed{seed}{suffix}.pt")
        if not os.path.exists(path):
            continue
        net, payload = _load(path)
        std = Standardizer.from_dict(payload["standardizer"])
        out["experts"][cid] = {
            "net": net, "std": std,
            "fine_names": payload["fine_class_names"],
            "fine_ids": payload["fine_class_ids"],
            "K": int(payload["n_nodes_per_layer"][-1]),
        }
    return out


def _seed_softrouting_probs(seed_pipe: dict, raw_test: torch.Tensor,
                             K_macro: int, n_test: int) -> torch.Tensor:
    """Returns (n_test, 100) soft-routing fine-class probability matrix."""
    Xte_l1 = seed_pipe["l1_std"].transform(raw_test).contiguous()
    with torch.no_grad():
        l1_logits = seed_pipe["l1_net"](Xte_l1)            # (N, K_macro)
        l1_probs = F.softmax(l1_logits, dim=1)
    P = torch.zeros(n_test, 100, dtype=torch.float32)
    # Singletons: P(fine_i) += P(macro_c) for the singleton's only fine.
    for cid, fine_id in seed_pipe["singletons"].items():
        P[:, fine_id] += l1_probs[:, cid]
    # Multi-class clusters: distribute P(macro_c) by L2's softmax.
    for cid, info in seed_pipe["experts"].items():
        Xte_l2 = info["std"].transform(raw_test).contiguous()
        with torch.no_grad():
            l2_logits = info["net"](Xte_l2)               # (N, K_local)
            l2_probs = F.softmax(l2_logits, dim=1)        # (N, K_local)
        for local, fine_id in enumerate(info["fine_ids"]):
            P[:, fine_id] += l1_probs[:, cid] * l2_probs[:, local]
    return P


def _seed_hardrouting_pred(seed_pipe: dict, raw_test: torch.Tensor,
                            n_test: int) -> torch.Tensor:
    """Per-seed hard pipeline pred (one fine_id per image)."""
    Xte_l1 = seed_pipe["l1_std"].transform(raw_test).contiguous()
    with torch.no_grad():
        l1_pred = seed_pipe["l1_net"](Xte_l1).argmax(dim=1)
    pred = torch.full((n_test,), -1, dtype=torch.long)
    # First handle every cluster.
    for cid, info in seed_pipe["experts"].items():
        m = (l1_pred == cid)
        if m.sum() == 0:
            continue
        Xte_l2 = info["std"].transform(raw_test[m]).contiguous()
        with torch.no_grad():
            local = info["net"](Xte_l2).argmax(dim=1)
        fine_id_t = torch.tensor(info["fine_ids"], dtype=torch.long)
        pred[m] = fine_id_t[local]
    for cid, fine_id in seed_pipe["singletons"].items():
        m = (l1_pred == cid)
        pred[m] = fine_id
    return pred


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-file",
                        default="outputs/cifar_taxonomy/cluster_assignment_full100_k20.pt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--expert-suffix", default="_contrastive",
                        help="Use contrastive-refined experts by default; "
                             "set to '' for vanilla experts.")
    args = parser.parse_args(argv)

    ca = torch.load(args.cluster_file, map_location="cpu", weights_only=False)
    K = ca["k"]
    name_to_id = _resolve_names_to_ids(args.data_root)
    print(f"[committee] cluster file: {args.cluster_file}  k={K}  "
          f"seeds={args.seeds}  expert_suffix={args.expert_suffix!r}")

    # Load test set.
    test_imgs, test_labs = load_cifar100(args.data_root, train=False)
    n_test = test_imgs.shape[0]
    print(f"[committee] test: {n_test} imgs")

    # Apply sense once.
    pipes = [_load_seed_pipeline(s, K, ca, args.expert_suffix, name_to_id)
             for s in args.seeds]
    sense_name = pipes[0]["sense"]
    raw_test = apply_sense(sense_name, test_imgs)

    # Per-seed soft-routing distributions and hard-routing preds.
    P_per_seed = []      # list of (n_test, 100)
    hard_per_seed = []   # list of (n_test,) fine_id
    per_seed_acc = []
    print(f"\n[committee] === per-seed accuracy ===")
    for s, pipe in zip(args.seeds, pipes):
        P = _seed_softrouting_probs(pipe, raw_test, K, n_test)
        hard = _seed_hardrouting_pred(pipe, raw_test, n_test)
        soft_pred = P.argmax(dim=1)
        soft_acc = (soft_pred == test_labs).float().mean().item()
        hard_acc = (hard == test_labs).float().mean().item()
        per_seed_acc.append(hard_acc)
        print(f"  seed {s}: hard-routing acc = {hard_acc:.4f}  "
              f"soft-routing acc = {soft_acc:.4f}")
        P_per_seed.append(P)
        hard_per_seed.append(hard)

    n_seeds = len(args.seeds)
    P_stack = torch.stack(P_per_seed, dim=0)             # (S, N, 100)
    hard_stack = torch.stack(hard_per_seed, dim=0)       # (S, N)

    # mean-softmax: average soft-routing distributions, argmax.
    P_mean = P_stack.mean(dim=0)
    pred_mean_softmax = P_mean.argmax(dim=1)
    acc_mean_softmax = (pred_mean_softmax == test_labs).float().mean().item()

    # mean-logit (geometric-mean of probs): sum logs, argmax.
    log_P = torch.log(P_stack.clamp_min(1e-9))
    log_P_mean = log_P.mean(dim=0)
    pred_mean_logit = log_P_mean.argmax(dim=1)
    acc_mean_logit = (pred_mean_logit == test_labs).float().mean().item()

    # hard-vote: majority on per-seed hard preds; ties broken by mean-softmax.
    pred_hard_vote = torch.zeros(n_test, dtype=torch.long)
    for i in range(n_test):
        votes = hard_stack[:, i].tolist()
        c = Counter(votes)
        top, top_n = c.most_common(1)[0]
        if top_n > n_seeds // 2:
            pred_hard_vote[i] = top
        else:
            pred_hard_vote[i] = pred_mean_softmax[i]
    acc_hard_vote = (pred_hard_vote == test_labs).float().mean().item()

    single_mean = sum(per_seed_acc) / len(per_seed_acc)
    print(f"\n[committee] === aggregation ===")
    print(f"  single-seed hard mean (n={n_seeds}):  {single_mean:.4f}")
    print(f"  hard-vote committee:               {acc_hard_vote:.4f}  "
          f"(Δ = {acc_hard_vote - single_mean:+.4f})")
    print(f"  mean-softmax committee:            {acc_mean_softmax:.4f}  "
          f"(Δ = {acc_mean_softmax - single_mean:+.4f})")
    print(f"  mean-logit committee:              {acc_mean_logit:.4f}  "
          f"(Δ = {acc_mean_logit - single_mean:+.4f})")
    print(f"  chance (1/100):                    0.0100")
    return 0


if __name__ == "__main__":
    sys.exit(main())
