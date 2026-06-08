"""Dual-trioron evaluation: L1 macro + L2 within-central-object as
two specialty branches sharing L0, no expansion.

Sidesteps the negative-transfer problem of api.extend by NOT forcing
one substrate to encode both `wolf ≈ motorcycle` (L1) and `wolf ≠
motorcycle` (L2). Each trioron stays in its lane:

  Trioron #1 (L1):  4-way macro classifier on classical senses.
                    Trained on 16-class subset, mapped to 4 perceptual
                    macros. classes_covered=[0,1,2,3].
  Trioron #2 (L2):  9-way fine classifier within central-object only.
                    Trained on 9 central-object fines (wolf, man,
                    clock, motorcycle, pickup_truck, rose, spider,
                    butterfly, mushroom). classes_covered=[0..8].

Both share l0_seed=42 → identical L0 random projection (= shared
substrate at the perception level). Independent L1 + head per branch.

Hierarchical inference:

    L1(image) → macro_pred
    if macro_pred == 1 (central-object):
        fine_pred = L2(image).argmax (mapped back to global fine id)
    else:
        we have no L2 expert for the predicted macro — image's true
        class can still be central-object's, in which case this is a
        miss. For the 9-way central-object test, "wrong macro" → wrong
        fine.

Comparison anchors (n=1, seed=42):

  * single donor api.extend(L1, L2) on 9-way central:   0.3844
  * single donor api.extend(consolidated L1, L2):       0.3811
  * fresh L2 (no L1 prior):                              0.4389  ← upper bound
  * dual-trioron hierarchical (this run):                ?
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from trioron.network import TrioronNetwork
from trioron.senses import apply_sense, Standardizer
from experiments.cifar.datasets import DEFAULT_DATA_ROOT
from experiments.cifar.bench_taxonomy_l2_central_object import (
    CENTRAL_OBJECT_NAMES, _build_subset,
)


def _load_donor(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    n_nodes = list(payload["n_nodes_per_layer"])
    layer_specs = []
    prev = int(payload["input_dim"])
    for i, n in enumerate(n_nodes):
        act = "linear" if i == len(n_nodes) - 1 else "relu"
        layer_specs.append((prev, int(n), act))
        prev = int(n)
    net = TrioronNetwork(layer_specs)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--l1-donor",
        default="outputs/cifar_taxonomy/donor_l1_perceptual_4way.pt",
    )
    parser.add_argument(
        "--l2-donor",
        default="outputs/cifar_taxonomy/donor_l2_central_object_9way.pt",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)

    l1_net, l1_p = _load_donor(args.l1_donor)
    l2_net, l2_p = _load_donor(args.l2_donor)
    print(f"[dual] L1: {args.l1_donor}")
    print(f"  arch={l1_p['n_nodes_per_layer']}  l0_seed={l1_p['l0_seed']}")
    print(f"[dual] L2: {args.l2_donor}")
    print(f"  arch={l2_p['n_nodes_per_layer']}  l0_seed={l2_p['l0_seed']}")
    if l1_p["l0_seed"] != l2_p["l0_seed"]:
        print(f"[dual] WARNING: L0 seeds differ — branches don't share substrate")

    if l1_p["sense"] != l2_p["sense"]:
        raise ValueError(f"Sense mismatch: L1={l1_p['sense']} L2={l2_p['sense']}")
    sense_name = l1_p["sense"]
    # L1 and L2 used different Standardizers (each fit on its training set).
    # For the dual-trioron we use each donor's standardizer for its own
    # forward pass — same input image, different normalization per
    # donor. This matches how the donors were trained.
    std_l1 = Standardizer.from_dict(l1_p["standardizer"])
    std_l2 = Standardizer.from_dict(l2_p["standardizer"])

    # Test set: the 9-way central-object slice.
    Xtr, ytr_local, ytr_fine, Xte, yte_local, yte_fine, fine_ids = (
        _build_subset(args.data_root, CENTRAL_OBJECT_NAMES)
    )
    print(f"[dual] central-object test: {Xte.shape[0]} imgs, "
          f"{len(CENTRAL_OBJECT_NAMES)} fines")

    # Apply the sense once.
    raw_test = apply_sense(sense_name, Xte)
    Xte_l1 = std_l1.transform(raw_test).contiguous()
    Xte_l2 = std_l2.transform(raw_test).contiguous()

    with torch.no_grad():
        l1_logits = l1_net(Xte_l1)        # (N, 4) — macro logits
        l2_logits = l2_net(Xte_l2)        # (N, 9) — fine logits
    l1_pred_macro = l1_logits.argmax(dim=1)        # (N,)
    l2_pred_fine = l2_logits.argmax(dim=1)         # (N,)

    # All test images here are central-object (macro=1 ground truth).
    n_total = Xte.shape[0]
    l1_central_recall = (l1_pred_macro == 1).float().mean().item()
    l2_fine_acc_solo = (l2_pred_fine == yte_local).float().mean().item()
    print(f"\n[dual] === component-level ===")
    print(f"  L1 macro recall on central-object: {l1_central_recall:.4f}  "
          f"(L1 says 'central' for {int(l1_central_recall*n_total)}/{n_total} "
          f"central-object images)")
    print(f"  L2 fine accuracy solo:             {l2_fine_acc_solo:.4f}  "
          f"(matches the fresh L2 baseline 0.4389 in memory)")

    # Hierarchical: only L2's fine prediction counts when L1 said "central".
    routed_correct = ((l1_pred_macro == 1) & (l2_pred_fine == yte_local))
    hier_acc = routed_correct.float().mean().item()
    print(f"\n[dual] === hierarchical (L1-routed → L2) ===")
    print(f"  end-to-end fine acc:  {hier_acc:.4f}")
    print(f"  decomposition: {l1_central_recall:.4f} (L1 routes correctly) × "
          f"{l2_fine_acc_solo:.4f} (L2 correct given correct routing) "
          f"≈ {l1_central_recall * l2_fine_acc_solo:.4f} (independence assumed)")

    # Per-class hierarchical accuracy.
    print(f"\n[dual] === per-class hierarchical accuracy ===")
    K = len(CENTRAL_OBJECT_NAMES)
    print(f"  {'class':<14s}  {'l2-solo':>8s}  {'dual':>8s}  "
          f"{'l1-correct':>11s}")
    for i, n in enumerate(CENTRAL_OBJECT_NAMES):
        mask = yte_local == i
        n_i = int(mask.sum().item())
        if n_i == 0:
            continue
        solo_i = (l2_pred_fine[mask] == i).float().mean().item()
        dual_i = ((l1_pred_macro[mask] == 1) & (l2_pred_fine[mask] == i)
                  ).float().mean().item()
        l1_recall_i = (l1_pred_macro[mask] == 1).float().mean().item()
        print(f"  {n:<14s}  {solo_i:>8.4f}  {dual_i:>8.4f}  "
              f"{l1_recall_i:>11.4f}")

    print(f"\n[dual] === comparison vs single-trioron approaches ===")
    print(f"  fresh L2, no L1 prior (upper bound):      0.4389")
    print(f"  api.extend(L1, L2):                       0.3844")
    print(f"  api.extend(consolidated L1, L2):          0.3811")
    print(f"  dual-trioron hierarchical (this run):     {hier_acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
