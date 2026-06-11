"""Edge audit for the NEW branch-id architecture (step5_branch), read-only.

Rocky's question: is the fan ASYMMETRIC? A growing/self-organizing net should
differentiate — different cells reading different input subsets, writing different
output subsets. This dumps per-cell fan_in/fan_out and the readout shape so we can
SEE whether the wiring is symmetric (every branch identical) or asymmetric.

Run: python3 -m experiments.progenitor.audit_branch
"""
from __future__ import annotations

from collections import Counter

import torch

from .data_hard import make_split, bayes_accuracy
from .step5_branch import build_base, grow_branches


def main() -> None:
    tr, te = make_split()
    torch.manual_seed(0)
    sub = build_base(n_in=tr.x.shape[1], n_out=len(te.names))
    grow_branches(sub, tr, te, te.names, bayes_accuracy(te))

    a = sub.arena
    ec = a.edge_cursor
    src = a.edge_src[:ec].tolist()
    dst = a.edge_dst[:ec].tolist()
    rank = {i: int(a.rank[i].item()) for i in range(a.cursor)}

    fan_in: dict[int, int] = Counter()
    fan_out: dict[int, int] = Counter()
    for s, d in zip(src, dst):
        fan_out[s] += 1
        fan_in[d] += 1

    print("\n=== rank histogram ===")
    hist = Counter(rank[i] for i in range(a.cursor))
    for r in sorted(hist):
        print(f"  rank {r:>2}: {hist[r]:>3} cells")

    print("\n=== edge transitions (src_rank -> dst_rank) ===")
    trans = Counter((rank[s], rank[d]) for s, d in zip(src, dst))
    for (sr, dr), n in sorted(trans.items()):
        flag = "  <-- not feed-forward" if dr <= sr else ""
        print(f"  rank {sr} -> rank {dr}: {n}{flag}")

    n_perc = len(sub.perc)
    n_out = len(sub.outs)
    print(f"\n  perception={n_perc}  outputs={n_out}  branches={len(sub.branch_top)}")

    print("\n=== per-branch wiring (is every branch identical = symmetric?) ===")
    for bid in sorted(sub.branch_top):
        cells = sub.branch_cells[bid]
        top = sub.branch_top[bid]
        l1 = cells[0]
        print(f"  branch {bid:>2}: depth {len(cells)}  cells {cells}")
        print(f"      L1 cell {l1}: fan_in {fan_in[l1]} (reads perception), "
              f"fan_out {fan_out[l1]}")
        print(f"      TOP cell {top}: fan_in {fan_in[top]}, "
              f"fan_out {fan_out[top]} (-> outputs)")

    print("\n=== readout shape (output cells) ===")
    for o in sub.outs[:4]:
        print(f"  output {o}: fan_in {fan_in[o]} (reads this many branch tops)")
    print(f"  ... ({n_out} outputs total)")
    tops = set(sub.branch_top.values())
    out_edges = sum(1 for s, d in zip(src, dst) if d in set(sub.outs))
    print(f"  total top->output edges: {out_edges}  "
          f"(= {len(tops)} tops x {n_out} outputs all-to-all = {len(tops)*n_out}? "
          f"{'YES — SYMMETRIC' if out_edges == len(tops)*n_out else 'no'})")

    print("\n=== fan asymmetry summary ===")
    hidden = [c for c in range(a.cursor) if rank[c] == 1 or (rank[c] >= 1 and c not in set(sub.outs) and c not in set(sub.perc))]
    fis = [fan_in[c] for c in range(a.cursor) if c not in set(sub.perc) and c not in set(sub.outs) and (fan_in[c] or fan_out[c])]
    print(f"  hidden-cell fan_in distribution: {Counter(fis)}")
    print(f"  → {'UNIFORM (symmetric)' if len(set(fis)) <= 2 else 'varied (asymmetric)'}")


if __name__ == "__main__":
    main()
