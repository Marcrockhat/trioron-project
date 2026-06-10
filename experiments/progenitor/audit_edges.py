"""Edge audit (s027, read-only): is the wiring what we think, and how big is it?

Rebuilds the deterministic frustration organism and reports:
  • edge count broken down by source-rank → dest-rank (is it a flat rank-0→1→2
    feed-forward, or are there stray same-rank / back edges?);
  • duplicate-edge check;
  • for one cohort cell and one grown cell of each phenotype, the EXACT incoming
    edges (src id, src rank, branch) and outgoing edges — to explain fan_in=14.

Run: python3 -m experiments.progenitor.audit_edges
"""
from __future__ import annotations

import torch

from .data_hard import make_split, bayes_accuracy
from .step3_council import build_council
from .step4_grow import grow_council_frustration


def main() -> None:
    tr, te = make_split()
    torch.manual_seed(0)
    sub = build_council(n_in=tr.x.shape[1], n_out=len(te.names))
    print(f"  n_in (perception cells) = {tr.x.shape[1]}   n_out = {len(te.names)}\n")
    grow_council_frustration(sub, tr, te, te.names, bayes_accuracy(te))

    a = sub.arena
    ec = a.edge_cursor
    src = a.edge_src[:ec]
    dst = a.edge_dst[:ec]
    rank = a.rank

    print(f"\n=== edge totals ===")
    print(f"  cells {a.cursor}   edges {ec}")

    # breakdown by rank transition
    from collections import Counter
    trans = Counter()
    for s, d in zip(src.tolist(), dst.tolist()):
        trans[(int(rank[s].item()), int(rank[d].item()))] += 1
    print("  edges by (src_rank → dst_rank):")
    for (sr, dr), n in sorted(trans.items()):
        flag = "  <-- not feed-forward!" if dr <= sr else ""
        print(f"    rank {sr} → rank {dr}: {n}{flag}")

    # duplicate check
    pairs = list(zip(src.tolist(), dst.tolist()))
    dups = len(pairs) - len(set(pairs))
    print(f"  duplicate (src,dst) edges: {dups}")

    # readout dilution
    n_hidden = int((rank[:a.cursor] == 1).sum())
    n_out = int((rank[:a.cursor] == 2).sum())
    print(f"\n=== readout shape ===")
    print(f"  rank-1 hidden cells: {n_hidden}   rank-2 output cells: {n_out}")
    print(f"  hidden→output edges: {trans[(1, 2)]}  "
          f"(= {n_hidden}×{n_out} all-to-all = {n_hidden * n_out}? "
          f"{'YES' if trans[(1,2)] == n_hidden*n_out else 'no'})")
    print(f"  → each output reads {n_hidden} hidden weights; "
          f"each hidden cell drives all {n_out} outputs")

    # trace specific cells
    def trace(cid: int, label: str) -> None:
        m_in = (dst == cid)
        ins = [(int(src[i].item()), int(rank[src[i]].item()),
                int(a.edge_branch[:ec][i].item())) for i in m_in.nonzero().flatten().tolist()]
        m_out = (src == cid)
        outs_ranks = Counter(int(rank[dst[i]].item()) for i in m_out.nonzero().flatten().tolist())
        src_ranks = Counter(r for _, r, _ in ins)
        branches = Counter(b for _, _, b in ins)
        print(f"  {label} cell {cid} (rank {int(rank[cid].item())}, K={int(a.n_branches[cid].item())}):")
        print(f"     fan_in {len(ins)}: by src-rank {dict(src_ranks)}, by branch {dict(branches)}")
        print(f"     fan_out {sum(outs_ranks.values())}: to ranks {dict(outs_ranks)}")

    print(f"\n=== per-cell edge trace ===")
    trace(64, "cohort linear")
    trace(68, "cohort dendrite")
    trace(89, "grown  linear")
    trace(77, "grown  dendrite")
    # a council germline cell for comparison
    lin_council = sub.council[0] if isinstance(sub.council, list) else None
    for ph, cells in sub.council.items():
        trace(cells[0], f"germline ph={ph}")
        break


if __name__ == "__main__":
    main()
