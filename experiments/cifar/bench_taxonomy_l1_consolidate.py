"""L1 — extra consolidation passes before L2 extension.

Test: does additional dream-style consolidation on L1 mitigate the
negative-transfer result api.extend produced?

api.extend's internal "shipping-consolidation dream" is one boundary
replay pass over base_tasks. Here we add N extra passes BEFORE calling
api.extend, on the trained L1 donor:

  * Frozen L0 (random projection — never updated post-training).
  * Trainable L1 + head with gentle LR (1e-4, 10× lower than the
    initial 1e-3 training LR) so the consolidation refines rather
    than retrains.
  * CE on real L1 4-way task data, no growth, no compression, no
    archive changes — pure substrate refinement.

After consolidation, save the donor with the same metadata layout so
bench_taxonomy_l2_expansion.py can pick it up via --l1-donor-path.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from trioron.network import TrioronNetwork
from trioron.senses import apply_sense, Standardizer
from experiments.cifar.bench_taxonomy_l1_perceptual import _build_subset


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
    return net, payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--l1-donor-path",
        default="outputs/cifar_taxonomy/donor_l1_perceptual_4way.pt",
    )
    parser.add_argument("--epochs", type=int, default=16,
                        help="Extra consolidation epochs.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Gentle LR — 10× lower than initial training.")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    from experiments.cifar.datasets import DEFAULT_DATA_ROOT
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out-path",
        default="outputs/cifar_taxonomy/donor_l1_perceptual_4way_consolidated.pt",
    )
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)

    net, payload = _load_donor(args.l1_donor_path)
    sense_name = payload["sense"]
    std = Standardizer.from_dict(payload["standardizer"])
    print(f"[L1-consol] donor: {args.l1_donor_path}")
    print(f"[L1-consol]   sense={sense_name}  arch={payload['n_nodes_per_layer']}")
    print(f"[L1-consol]   classes_covered={payload['classes_covered']}")

    # Reconstruct L1 4-way data.
    Xtr, ytr_perc, _, Xte, yte_perc, _, _, _ = _build_subset(args.data_root)
    Xtr = std.transform(apply_sense(sense_name, Xtr)).contiguous()
    Xte = std.transform(apply_sense(sense_name, Xte)).contiguous()
    print(f"[L1-consol] train {Xtr.shape}, test {Xte.shape}")

    # Baseline accuracy on test before consolidation.
    net.eval()
    with torch.no_grad():
        acc_before = (net(Xte).argmax(dim=1) == yte_perc).float().mean().item()
    print(f"[L1-consol] L1 acc BEFORE: {acc_before:.4f}")

    # Freeze L0; consolidate L1 + head with gentle LR.
    for p in net.layers[0].parameters():
        p.requires_grad_(False)
    trainable = [p for layer in net.layers[1:] for p in layer.parameters()
                 if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr)
    n_train = Xtr.shape[0]

    print(f"[L1-consol] consolidating: {args.epochs} epochs, "
          f"batch={args.batch}, lr={args.lr}")
    t0 = time.time()
    net.train()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, args.batch):
            idx = perm[i:i + args.batch]
            x = Xtr[idx]
            y = ytr_perc[idx]
            logits = net(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        net.eval()
        with torch.no_grad():
            acc_now = (net(Xte).argmax(dim=1) == yte_perc).float().mean().item()
        net.train()
        print(f"  epoch {epoch:>2d}/{args.epochs}  "
              f"loss={epoch_loss/n_batches:.4f}  acc={acc_now:.4f}")
    print(f"[L1-consol] consolidation done ({time.time()-t0:.1f}s)")

    net.eval()
    with torch.no_grad():
        acc_after = (net(Xte).argmax(dim=1) == yte_perc).float().mean().item()
    print(f"\n[L1-consol] === results ===")
    print(f"  acc BEFORE consolidation: {acc_before:.4f}")
    print(f"  acc AFTER  consolidation: {acc_after:.4f}")
    print(f"  Δ:                        {acc_after - acc_before:+.4f}")

    # Save with same metadata layout as the original L1 donor so
    # bench_taxonomy_l2_expansion.py can pick it up.
    payload["state_dict"] = net.state_dict()
    payload["consolidation_pass"] = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch,
        "acc_before": acc_before,
        "acc_after": acc_after,
    }
    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(payload, out_path)
    print(f"[L1-consol] [SAVE] {out_path}  "
          f"({os.path.getsize(out_path)/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
