"""L1 — extract and inspect δ_AB in L0 space.

Step 2 of the taxonomic-contrastive scaffold. No training, no
architectural changes. Pure analysis: load the L1 donor, push training
data through L0 only, compute per-class μ and σ, then the contrast
vector δ = μ_living − μ_nonliving.

Reports:
  * ||δ||₂ (magnitude in L0 space; calibrate vs. ||μ_living||₂)
  * top-k L0 dimensions by |δ_i| (where the contrast lives)
  * cosine similarity between μ_living and μ_nonliving
    (1 = identical centroids; the contrastive task is harder when this
    is high, since the two classes' L0 centroids overlap heavily)
  * per-dimension σ overlap on the top-k δ dims (how separable each
    contrast dim is)
  * Bhattacharyya-style separability score per dim:
        sep_i = |δ_i| / (σ_living[i] + σ_nonliving[i] + ε)
    high values → that dim genuinely separates the two classes.
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Tuple

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from trioron.network import TrioronNetwork
from trioron.senses import apply_sense, Standardizer
from experiments.cifar.datasets import load_cifar100, DEFAULT_DATA_ROOT
from experiments.cifar.bench_taxonomy_l1 import (
    LIVING_NAMES, NON_LIVING_NAMES, _resolve_names_to_ids, _binary_subset,
)


def _load_donor(path: str) -> Tuple[TrioronNetwork, dict]:
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


def _l0_activations(
    net: TrioronNetwork, X: torch.Tensor, batch_size: int = 512,
) -> torch.Tensor:
    """Run forward through L0 layer only (i.e., layer 0). Returns
    post-activation output of layer 0 — what trioron's manifold buffer
    treats as 'L0 code space.'"""
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            outs.append(net.layers[0](X[i:i + batch_size]))
    return torch.cat(outs, dim=0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--donor-path",
        default="outputs/cifar_taxonomy/donor_l1_living_vs_nonliving.pt",
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)

    net, payload = _load_donor(args.donor_path)
    sense_name = payload["sense"]
    std = Standardizer.from_dict(payload["standardizer"])
    print(f"[L1-δ] donor: {args.donor_path}")
    print(f"[L1-δ]   sense={sense_name}  arch={payload['n_nodes_per_layer']}")
    print(f"[L1-δ]   L0 dim = {payload['n_nodes_per_layer'][0]} "
          f"(layer 0 output)")

    name_to_id = _resolve_names_to_ids(args.data_root)
    living_ids = [name_to_id[n] for n in LIVING_NAMES]
    nonliving_ids = [name_to_id[n] for n in NON_LIVING_NAMES]
    train_imgs, train_labs = load_cifar100(args.data_root, train=True)
    Xtr_raw, ytr_bin, ytr_fine = _binary_subset(
        train_imgs, train_labs, living_ids, nonliving_ids,
    )
    Xtr_sensed = std.transform(apply_sense(sense_name, Xtr_raw)).contiguous()
    print(f"[L1-δ] sensed train: {tuple(Xtr_sensed.shape)} "
          f"({int((ytr_bin == 0).sum())} Living, "
          f"{int((ytr_bin == 1).sum())} Non-living)")

    h0 = _l0_activations(net, Xtr_sensed)        # (N, 128)
    print(f"[L1-δ] L0 activations: {tuple(h0.shape)}  "
          f"mean={h0.mean().item():+.4f}  std={h0.std().item():.4f}")

    # Per-class statistics in L0 space.
    mu_l = h0[ytr_bin == 0].mean(dim=0)             # (128,)
    mu_n = h0[ytr_bin == 1].mean(dim=0)
    sig_l = h0[ytr_bin == 0].std(dim=0).clamp_min(1e-6)
    sig_n = h0[ytr_bin == 1].std(dim=0).clamp_min(1e-6)
    delta = mu_l - mu_n                             # (128,)

    norm_mu_l = mu_l.norm().item()
    norm_mu_n = mu_n.norm().item()
    norm_delta = delta.norm().item()
    cos_mu = (mu_l @ mu_n / (norm_mu_l * norm_mu_n + 1e-9)).item()

    print(f"\n[L1-δ] === aggregate δ statistics ===")
    print(f"  ||μ_living||₂       = {norm_mu_l:.4f}")
    print(f"  ||μ_nonliving||₂    = {norm_mu_n:.4f}")
    print(f"  ||δ||₂              = {norm_delta:.4f}  "
          f"(= {norm_delta / max(norm_mu_l, norm_mu_n) * 100:.1f}% of larger μ)")
    print(f"  cos(μ_l, μ_n)       = {cos_mu:+.4f}  "
          f"(1 = identical centroids → contrast is hard)")

    # Top-k dimensions by |δ| with separability score.
    abs_delta = delta.abs()
    top_vals, top_idx = abs_delta.topk(args.top_k)
    sep = abs_delta / (sig_l + sig_n + 1e-9)        # (128,) Bhattacharyya-ish
    print(f"\n[L1-δ] === top-{args.top_k} L0 dims by |δ_i| ===")
    print(f"  {'dim':>4s}  {'δ_i':>8s}  {'μ_l[i]':>8s}  {'μ_n[i]':>8s}  "
          f"{'σ_l[i]':>7s}  {'σ_n[i]':>7s}  {'sep_i':>7s}")
    for d in top_idx.tolist():
        print(f"  {d:>4d}  {delta[d].item():>+8.4f}  "
              f"{mu_l[d].item():>+8.4f}  {mu_n[d].item():>+8.4f}  "
              f"{sig_l[d].item():>7.4f}  {sig_n[d].item():>7.4f}  "
              f"{sep[d].item():>7.3f}")

    print(f"\n[L1-δ] === separability summary ===")
    print(f"  mean sep over all 128 dims:  {sep.mean().item():.4f}")
    print(f"  max sep:                      {sep.max().item():.4f}  "
          f"(at dim {sep.argmax().item()})")
    print(f"  median sep:                   {sep.median().item():.4f}")
    print(f"  fraction of dims with sep > 0.5: "
          f"{(sep > 0.5).float().mean().item() * 100:.1f}%")
    print(f"  fraction of dims with sep > 1.0: "
          f"{(sep > 1.0).float().mean().item() * 100:.1f}%")

    # Number of L0 dims that genuinely contribute to the contrast (sep > 0.5
    # is a rough rule-of-thumb threshold for "the centroids are visibly
    # offset relative to within-class spread on this dim").
    contributing = (sep > 0.5).sum().item()
    print(f"\n[L1-δ] interpretation:")
    print(f"  ~{contributing}/{sep.shape[0]} L0 dimensions carry the "
          f"Living vs Non-living contrast at sep > 0.5.")
    if contributing < 10:
        print(f"  → contrast is sparse: a small subset of L0 dims does the "
              f"work. Storing δ explicitly would be cheap.")
    elif contributing < 60:
        print(f"  → contrast is moderately distributed. Storing δ as a "
              f"dense 128-d vector is reasonable.")
    else:
        print(f"  → contrast is dense: most L0 dims carry signal. Less "
              f"clear that a single δ vector helps over per-class μ.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
