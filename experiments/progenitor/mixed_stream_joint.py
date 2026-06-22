"""Joint (offline) upper-bound decomposition for the MIXED cross-domain stream
(s045 diagnostic). Trains the SAME-family soma+head net on ALL 52 classes at
ONCE — no CL, no interleaving — to split the low CL accuracy into:

    (architecture + perception ceiling)   vs   (CL / forgetting penalty).

If the joint number is high and CL is far below it, the loss is forgetting.
If the joint number is itself low, the loss is upstream — perception (lenses)
or the readout (capacity / depth / nonlinearity). The s045 CL `replay` arm
retains ~86% of what it acquires (acq 0.534 -> final 0.456), so the smart money
is on the readout/perception ceiling — this script tests that and ALSO sweeps
architecture upgrades (depth / width / phenotype) to see which lifts the ceiling.

Run:  python3 experiments/progenitor/mixed_stream_joint.py
Env:  SEEDS EPOCHS_JOINT ARCHS  (ARCHS = comma list of names from VARIANTS)
"""
from __future__ import annotations

import os
import time
import torch
import torch.nn.functional as F

from trioron.core.arena import Arena
from trioron.core.envelope import Envelope
from trioron.core.epigenome import CREDIT_ELIGIBLE, OUTPUT, LINEAR, TANH, DENDRITE
from trioron.phenotype import conv
from experiments.progenitor.conv_proposer import _bucket_for
from experiments.progenitor import mixed_stream_lenses as L
from experiments.progenitor.mixed_stream_cl import NONLIN, GNAME, DOM_OF, block_matrices

EPOCHS_JOINT = int(os.environ.get("EPOCHS_JOINT", "30"))
BATCH, LR = 128, 0.01

# Architecture variants: name -> list of (width, gene) hidden layers. Head is
# always a LINEAR N_CLASS readout. "H64-tanh" is the current CL architecture.
VARIANTS = {
    "H64-lin":   [(64, LINEAR)],
    "H64-tanh":  [(64, TANH)],
    "H64-dend":  [(64, DENDRITE)],
    "H128-tanh": [(128, TANH)],
    "H256-tanh": [(256, TANH)],
    "64-64-tanh":   [(64, TANH), (64, TANH)],
    "128-64-tanh":  [(128, TANH), (64, TANH)],
    "128-64-dend":  [(128, DENDRITE), (64, TANH)],
}


# ── flexible builder: input(block) -> hidden layers -> LINEAR head ─────────
def build_net(block_dim, layers, seed):
    widths = [w for w, _ in layers]
    real_cells = block_dim + sum(widths) + L.N_CLASS
    # The soma is FULLY connected to the block (fan-in = block_dim, not the
    # arena's assumed sparse ~64), so size capacity from the real edge count:
    # ecap = capacity * 64 must cover every dense edge.
    dims = [block_dim] + widths + [L.N_CLASS]
    edges = sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))
    cap = max(real_cells + 16, -(-edges // 64) + 16)
    a = Arena(Envelope(), capacity=cap)
    a.alloc(block_dim)
    g = torch.Generator().manual_seed(seed)

    prev_ids = list(range(block_dim))
    layer_ids, buckets, genes = [], [], []
    for (w, gene) in layers:
        ids = a.alloc(w).tolist()
        for c in ids:
            a.epigenome[c] = (1 << gene) | (1 << CREDIT_ELIGIBLE)
            a.phenotype_cache[c] = gene
            src = torch.tensor(prev_ids, dtype=torch.int32)
            dst = torch.full((len(prev_ids),), c, dtype=torch.int32)
            a.add_edges(src, dst, torch.randn(len(prev_ids), generator=g) * (1.0 / len(prev_ids) ** 0.5))
        layer_ids.append(ids); buckets.append(_bucket_for(a, ids)); genes.append(gene)
        prev_ids = ids

    head_ids = a.alloc(L.N_CLASS).tolist()
    for c in head_ids:
        a.epigenome[c] = (1 << LINEAR) | (1 << CREDIT_ELIGIBLE) | (1 << OUTPUT)
        a.phenotype_cache[c] = LINEAR
        src = torch.tensor(prev_ids, dtype=torch.int32)
        dst = torch.full((len(prev_ids),), c, dtype=torch.int32)
        a.add_edges(src, dst, torch.randn(len(prev_ids), generator=g) * (1.0 / len(prev_ids) ** 0.5))
    head_bucket = _bucket_for(a, head_ids)

    a.edge_weight.requires_grad_(True); a.bias.requires_grad_(True)
    return a, layer_ids, buckets, genes, head_ids, head_bucket


def forward(a, block_dim, layer_ids, buckets, genes, head_bucket, Xblock):
    A = torch.zeros(len(Xblock), a.capacity)
    A[:, :block_dim] = Xblock
    for ids, bk, gene in zip(layer_ids, buckets, genes):
        act = NONLIN[gene](conv.forward_batch(A, bk, a))     # [N, w]
        A = torch.zeros(len(Xblock), a.capacity)
        A[:, ids] = act
    return conv.forward_batch(A, head_bucket, a)             # [N, N_CLASS]


def run_joint(per, block_dim, layers, seed):
    Xtr, ytr = block_matrices(per, "train")
    Xte, yte = block_matrices(per, "test")
    a, lids, bks, genes, head_ids, hbk = build_net(block_dim, layers, seed)
    opt = torch.optim.Adam([a.edge_weight, a.bias], lr=LR)
    gg = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS_JOINT):
        perm = torch.randperm(len(Xtr), generator=gg)
        for i in range(0, len(Xtr), BATCH):
            bi = perm[i:i + BATCH]
            opt.zero_grad()
            F.cross_entropy(forward(a, block_dim, lids, bks, genes, hbk, Xtr[bi]), ytr[bi]).backward()
            opt.step()

    @torch.no_grad()
    def acc(classes):
        m = torch.isin(yte, torch.tensor(classes))
        Xc, yc = Xte[m], yte[m]
        correct = 0
        for i in range(0, len(Xc), BATCH):                # batch eval (wide nets OOM otherwise)
            pred = forward(a, block_dim, lids, bks, genes, hbk, Xc[i:i + BATCH]).argmax(1)
            correct += int((pred == yc[i:i + BATCH]).sum())
        return correct / len(Xc)

    dom = {name: acc(list(range(off, off + k))) for name, (off, k) in DOM_OF.items()}
    n_par = int(a.edge_cursor) + sum(len(i) for i in lids) + L.N_CLASS
    return dict(dom=dom, mean=sum(dom.values()) / 3, params=n_par)


def main():
    t0 = time.time()
    seeds = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]
    archs = os.environ.get("ARCHS", ",".join(VARIANTS)).split(",")

    print(f"JOINT upper bound (offline, all {L.N_CLASS} classes at once) — "
          f"EPOCHS={EPOCHS_JOINT} | n={len(seeds)} seeds")
    print("  s045 CL `replay` for reference: tax 0.440  mnist 0.690  cifar 0.240  mean 0.456\n")
    print(f"  {'arch':<14} {'taxonomy':>11} {'mnist':>11} {'cifar':>11} {'mean':>11} {'params':>8}")

    def ms(vals):
        t = torch.tensor(vals)
        return f"{t.mean():.3f}" + (f"±{t.std(0):.3f}" if len(vals) > 1 else "")

    descr = {s: L.build_descriptors(s) for s in seeds}
    for name in archs:
        layers = VARIANTS[name]
        R = [run_joint(*descr[s][:2], layers, s) for s in seeds]
        td = ms([r["dom"]["taxonomy"] for r in R]); md = ms([r["dom"]["mnist"] for r in R])
        cd = ms([r["dom"]["cifar"] for r in R]); mn = ms([r["mean"] for r in R])
        print(f"  {name:<14} {td:>11} {md:>11} {cd:>11} {mn:>11} {R[0]['params']:>8}")
    print(f"\n[{time.time() - t0:.0f}s]  (gap = joint − CL replay; large gap => forgetting, "
          f"small gap => ceiling is perception/readout)")


if __name__ == "__main__":
    main()
