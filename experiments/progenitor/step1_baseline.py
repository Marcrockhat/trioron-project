"""Step 1 — the floor: the simplest substrate, hand-wired, on real data.

Before any progenitor or council, establish the baseline the design must beat:
ONE perception cell (the weight feature) wired directly to 6 LINEAR output cells.
Every node and edge is allocated by hand and printed, so the structure is fully
visible. Train on the 6-animal weight data and read per-class accuracy against
the Bayes ceiling.

Expected (design doc §5): overall ~0.79, with dog the casualty — a linear readout
cannot carve dog's disjoint band. That gap is the thing the council exists to fix.

Run: python3 -m experiments.progenitor.step1_baseline
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from trioron.core import Envelope, construct
from trioron.core.epigenome import (
    PERCEPTION, OUTPUT, DENDRITE, has_gene, set_gene, primary_phenotype,
)
from trioron.phenotype import default_dispatch_table

from .data import make_data, bayes_accuracy, bayes_per_class

_PHENO_NAME = {0: "linear", 1: "attention", 2: "conv", 3: "recurrent", 4: "dendrite"}


def build_baseline(n_in: int = 1, n_out: int = 6):
    """Hand-wire n_in perception cells → n_out linear output cells on the Arena."""
    def base(sub):
        a = sub.arena
        perc = a.alloc(n_in)
        outs = a.alloc(n_out)
        for p in perc.tolist():
            a.epigenome[p] = set_gene(int(a.epigenome[p].item()), PERCEPTION)
            a.rank[p] = 0
        for o in outs.tolist():
            a.epigenome[o] = set_gene(int(a.epigenome[o].item()), OUTPUT)
            a.rank[o] = 1
        a.refresh_all_phenotypes()
        # fully connect every perception cell → every output cell
        src = perc.repeat(n_out)
        dst = outs.repeat_interleave(n_in)
        a.add_edges(src, dst)
    return construct(base=base, envelope=Envelope(),
                     dispatch_table=default_dispatch_table(), capacity=64)


def print_structure(sub) -> None:
    a = sub.arena
    ids = a.alive_ids().tolist()
    print("  nodes:")
    for cid in ids:
        epi = int(a.epigenome[cid].item())
        role = ("perception" if has_gene(epi, PERCEPTION)
                else "output" if has_gene(epi, OUTPUT) else "interior")
        pheno = _PHENO_NAME.get(primary_phenotype(epi), "?")
        print(f"    cell {cid:>2}  rank {int(a.rank[cid].item())}  {role:<10} {pheno}")
    print("  edges (src → dst : weight):")
    for i in range(a.edge_cursor):
        s, d = int(a.edge_src[i].item()), int(a.edge_dst[i].item())
        print(f"    {s:>2} → {d:>2} : {float(a.edge_weight[i].item()):+.3f}")


def train(sub, data, steps=400, lr=0.05, seed=0):
    torch.manual_seed(seed)
    sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(sub(data.x), data.y).backward()
        opt.step()


def evaluate(sub, data):
    with torch.no_grad():
        pred = sub(data.x).argmax(1)
    overall = float((pred == data.y).float().mean())
    per = [float((pred[data.y == c] == c).float().mean()) for c in range(len(data.names))]
    return overall, per


def main() -> None:
    train_d = make_data(seed=0, n_per_class=256)
    test_d = make_data(seed=1, n_per_class=512)

    sub = build_baseline()
    print("Step 1 — baseline substrate (1 perception → 6 linear outputs)\n")
    print_structure(sub)

    print("\n  ... training (400 steps, Adam) ...")
    train(sub, train_d)
    overall, per = evaluate(sub, test_d)
    bpc = bayes_per_class(test_d)

    print(f"\n  {'species':>8}  {'baseline':>9}  {'bayes':>6}")
    for ci, name in enumerate(test_d.names):
        flag = "  <- disjoint band (linear can't)" if name == "dog" else ""
        print(f"  {name:>8}  {per[ci]:>9.3f}  {bpc[ci]:>6.3f}{flag}")
    print(f"  {'OVERALL':>8}  {overall:>9.3f}  {bayes_accuracy(test_d):>6.3f}")


if __name__ == "__main__":
    main()
