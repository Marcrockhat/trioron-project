"""Dual-trioron full 16-way hierarchical evaluation.

The complete multi-branch organism over the 16-class subset:

    L1 (4-way macro)   → predicts macro 0..3
        ↓
    L2 expert per macro:
        compact-object (3 fines)   →  chair, bottle, cup
        central-object (9 fines)   →  wolf, man, clock, motorcycle,
                                       pickup_truck, rose, spider,
                                       butterfly, mushroom
        horizontal-landscape (2)   →  dolphin, mountain
        vertical-landscape (2)     →  oak_tree, castle

For each test image:
  1. L1 → macro_pred ∈ {0..3}
  2. L2[macro_pred] → fine_local ∈ {0..K_macro−1}
  3. fine_local → global CIFAR-100 fine ID via the macro's class list
  4. Compare to ground truth fine ID (one of the 16 in the subset)

Reports overall 16-way accuracy, per-cluster per-fine-class accuracy,
and the breakdown of error sources (L1 routing miss vs. L2 fine miss).
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from trioron.network import TrioronNetwork
from trioron.senses import apply_sense, Standardizer
from experiments.cifar.datasets import DEFAULT_DATA_ROOT, load_cifar100
from experiments.cifar.bench_taxonomy_l1 import (
    LIVING_NAMES, NON_LIVING_NAMES, _resolve_names_to_ids, _binary_subset,
)
from experiments.cifar.bench_taxonomy_l1_perceptual import PERCEPTUAL_GROUPS


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


def _slug(label: str) -> str:
    return label.replace("-", "_")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--l1-donor",
        default="outputs/cifar_taxonomy/donor_l1_perceptual_4way.pt",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed-tagged donor paths if --l2-* not specified.",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--l2-compact",
        default=None,
    )
    parser.add_argument(
        "--l2-central",
        default=None,
    )
    parser.add_argument(
        "--l2-horizontal",
        default=None,
    )
    parser.add_argument(
        "--l2-vertical",
        default=None,
    )
    args = parser.parse_args(argv)

    # Default L2 paths from PERCEPTUAL_GROUPS K and seed.
    DEFAULTS = {
        "compact-object": (args.l2_compact,
                           "outputs/cifar_taxonomy/donor_l2_compact_object_3way_seed{s}.pt"),
        "central-object": (args.l2_central,
                           "outputs/cifar_taxonomy/donor_l2_central_object_9way_seed{s}.pt"),
        "horizontal-landscape": (args.l2_horizontal,
                                 "outputs/cifar_taxonomy/donor_l2_horizontal_landscape_2way_seed{s}.pt"),
        "vertical-landscape": (args.l2_vertical,
                               "outputs/cifar_taxonomy/donor_l2_vertical_landscape_2way_seed{s}.pt"),
    }

    l1_net, l1_p = _load(args.l1_donor)
    l1_std = Standardizer.from_dict(l1_p["standardizer"])
    sense_name = l1_p["sense"]
    print(f"[16-way] L1: {args.l1_donor}  arch={l1_p['n_nodes_per_layer']}  "
          f"l0_seed={l1_p['l0_seed']}")

    # Build name → macro (perc id) and load L2 experts.
    name_to_id = _resolve_names_to_ids(args.data_root)
    name_to_macro: Dict[str, int] = {}
    macro_id_to_label: Dict[int, str] = {}
    macro_id_to_names: Dict[int, List[str]] = {}
    for pid, info in PERCEPTUAL_GROUPS.items():
        macro_id_to_label[pid] = info["label"]
        macro_id_to_names[pid] = list(info["names"])
        for n in info["names"]:
            name_to_macro[n] = pid

    l2_experts = {}
    for label, (override, default_template) in DEFAULTS.items():
        path = override or default_template.format(s=args.seed)
        if not os.path.exists(path):
            print(f"[16-way] WARNING: missing L2 expert for {label!r} at {path}")
            continue
        net, payload = _load(path)
        std = Standardizer.from_dict(payload["standardizer"])
        l2_experts[label] = {
            "net": net, "std": std,
            "fine_names": payload["fine_class_names"],
            "fine_ids": payload["fine_class_ids"],
            "K": int(payload["n_nodes_per_layer"][-1]),
        }
        print(f"[16-way] L2[{label}]: {path}  K={l2_experts[label]['K']}  "
              f"fines={l2_experts[label]['fine_names']}")

    # 16-class test set.
    living_ids = [name_to_id[n] for n in LIVING_NAMES]
    nonliving_ids = [name_to_id[n] for n in NON_LIVING_NAMES]
    test_imgs, test_labs = load_cifar100(args.data_root, train=False)
    Xte, _, yte_fine = _binary_subset(test_imgs, test_labs,
                                      living_ids, nonliving_ids)
    n_test = Xte.shape[0]
    print(f"[16-way] test: {n_test} imgs across "
          f"{len(set(yte_fine.tolist()))} fine classes")

    # Apply sense once; standardize per-donor.
    raw_test = apply_sense(sense_name, Xte)
    Xte_l1 = l1_std.transform(raw_test).contiguous()

    # L1 forward.
    with torch.no_grad():
        l1_pred_macro = l1_net(Xte_l1).argmax(dim=1)         # (N,)

    # L2 forward (per expert, full batch — small enough).
    l2_pred_local: Dict[str, torch.Tensor] = {}
    for label, info in l2_experts.items():
        Xte_l2 = info["std"].transform(raw_test).contiguous()
        with torch.no_grad():
            l2_pred_local[label] = info["net"](Xte_l2).argmax(dim=1)  # (N,)

    # Map: macro_id → cluster_label string.
    macro_to_label = {pid: info["label"] for pid, info in PERCEPTUAL_GROUPS.items()}

    # Hierarchical decision per image.
    pred_fine_id = torch.zeros(n_test, dtype=yte_fine.dtype)
    for i in range(n_test):
        macro = int(l1_pred_macro[i].item())
        label = macro_to_label[macro]
        if label not in l2_experts:
            # No expert — fall back to first fine in macro (deterministic).
            pred_fine_id[i] = name_to_id[macro_id_to_names[macro][0]]
            continue
        local = int(l2_pred_local[label][i].item())
        fine_global = l2_experts[label]["fine_ids"][local]
        pred_fine_id[i] = fine_global

    overall_acc = (pred_fine_id == yte_fine).float().mean().item()

    # Per-cluster, per-fine breakdown.
    print(f"\n[16-way] === results ===")
    print(f"  overall 16-way fine acc: {overall_acc:.4f}  "
          f"(chance 1/16 = {1/16:.4f})")
    print(f"  margin over chance:      {overall_acc - 1/16:+.4f}")

    # Per-cluster.
    print(f"\n[16-way] === per-cluster (true cluster ID known from ground truth) ===")
    for pid, info in PERCEPTUAL_GROUPS.items():
        names = info["names"]
        cluster_ids = [name_to_id[n] for n in names]
        m = torch.zeros(n_test, dtype=torch.bool)
        for c in cluster_ids:
            m |= yte_fine == c
        n_c = int(m.sum().item())
        if n_c == 0:
            continue
        # L1 macro recall on this cluster.
        l1_recall = (l1_pred_macro[m] == pid).float().mean().item()
        # End-to-end fine acc on this cluster.
        cluster_acc = (pred_fine_id[m] == yte_fine[m]).float().mean().item()
        print(f"  {info['label']:<22s}  n={n_c}  "
              f"L1 recall={l1_recall:.4f}  "
              f"fine acc={cluster_acc:.4f}")

    # Per-fine-class.
    print(f"\n[16-way] === per-fine-class accuracy ===")
    all_names = LIVING_NAMES + NON_LIVING_NAMES
    for n in all_names:
        cid = name_to_id[n]
        m = yte_fine == cid
        n_i = int(m.sum().item())
        if n_i == 0:
            continue
        macro = name_to_macro[n]
        acc_i = (pred_fine_id[m] == cid).float().mean().item()
        l1_corr = (l1_pred_macro[m] == macro).float().mean().item()
        print(f"  {n:<14s}  (macro={macro_to_label[macro]:<22s})  "
              f"n={n_i}  acc={acc_i:.4f}  L1 routes={l1_corr:.4f}")

    # Source of error: L1 routing miss vs L2 fine miss.
    print(f"\n[16-way] === error decomposition ===")
    n_l1_miss = 0
    n_l2_miss = 0
    n_correct = 0
    for i in range(n_test):
        true_fine = int(yte_fine[i].item())
        true_name = next(n for n, fid in name_to_id.items() if fid == true_fine
                          and n in name_to_macro)
        true_macro = name_to_macro[true_name]
        pred_macro = int(l1_pred_macro[i].item())
        if pred_macro != true_macro:
            n_l1_miss += 1
        elif int(pred_fine_id[i].item()) == true_fine:
            n_correct += 1
        else:
            n_l2_miss += 1
    print(f"  correct (L1 routes + L2 fine):  {n_correct:>4d}/{n_test}  "
          f"= {n_correct/n_test:.4f}")
    print(f"  L1 routing miss:                {n_l1_miss:>4d}/{n_test}  "
          f"= {n_l1_miss/n_test:.4f}")
    print(f"  L1 right but L2 fine miss:      {n_l2_miss:>4d}/{n_test}  "
          f"= {n_l2_miss/n_test:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
