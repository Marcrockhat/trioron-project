"""Trace the depth tangle (s027, read-only): is the rank really clean composition,
or every-sibling-wired-into-every-earlier-sibling?

Hypothesis: all somas divide from ONE shared parent (council[ph][0]) and
project_to_consumers wires each child to that parent's consumers; once an early soma
becomes a consumer, every LATER sibling gets an edge INTO it, so early cells climb in
rank via a sibling tangle rather than a perception→A→B→out chain.

Test signatures:
  • spawned cell X with incoming edges from sources with id > X  (a LATER sibling
    wiring backward into an earlier one — impossible at division time, only added by
    a later sibling's forward projection);
  • the longest path to a deep cell is mostly SIBLINGS, not perception.

Run: python3 -m experiments.progenitor.trace_tangle
"""
from __future__ import annotations

import torch

from .data_hard import make_split, bayes_accuracy
from .step3_council import build_council
from .step4_grow import grow_council_frustration

N_PERC = 12          # ids 0..11
N_COUNCIL = 20       # ids 12..31
N_OUT = 32           # ids 32..63
SPAWN0 = N_PERC + N_COUNCIL + N_OUT   # first spawned id = 64


def _cat(cid: int) -> str:
    if cid < N_PERC:
        return "perc"
    if cid < N_PERC + N_COUNCIL:
        return "council"
    if cid < SPAWN0:
        return "OUT"
    return "soma"


def main() -> None:
    tr, te = make_split()
    torch.manual_seed(0)
    sub = build_council(n_in=tr.x.shape[1], n_out=len(te.names))
    grow_council_frustration(sub, tr, te, te.names, bayes_accuracy(te))

    a = sub.arena
    ec = a.edge_cursor
    src = a.edge_src[:ec].tolist()
    dst = a.edge_dst[:ec].tolist()
    rank = {i: int(a.rank[i].item()) for i in range(a.cursor)}

    # incoming-source map
    incoming: dict[int, list[int]] = {}
    for s, d in zip(src, dst):
        incoming.setdefault(d, []).append(s)

    # rank histogram
    from collections import Counter
    hist = Counter(rank[i] for i in range(a.cursor))
    print("\n=== rank histogram (all cells) ===")
    for r in sorted(hist):
        print(f"  rank {r:>2}: {hist[r]:>3} cells")

    # tangle metric: spawned cells fed by LATER siblings
    print("\n=== tangle metric: incoming edges from LATER-spawned siblings ===")
    print("  (an edge src_id > dst_id into a soma can ONLY come from a later sibling's")
    print("   forward projection — clean depth would have ZERO of these)\n")
    total_back = 0
    for cid in range(SPAWN0, a.cursor):
        srcs = incoming.get(cid, [])
        later = [s for s in srcs if s > cid]
        total_back += len(later)
        if later:
            print(f"  soma {cid:>3} (rank {rank[cid]:>2}): "
                  f"{len(later):>2}/{len(srcs):>2} inputs are LATER siblings "
                  f"{sorted(later)[:8]}{'...' if len(later) > 8 else ''}")
    print(f"\n  TOTAL later-sibling back-wires into earlier somata: {total_back}")

    # longest-path reconstruction to the deepest cells
    def longest_path(cid: int) -> list[int]:
        path = [cid]
        cur = cid
        seen = {cid}
        while rank[cur] > 0:
            ins = [s for s in incoming.get(cur, []) if s not in seen]
            if not ins:
                break
            nxt = max(ins, key=lambda s: rank[s])
            path.append(nxt); seen.add(nxt); cur = nxt
        return list(reversed(path))

    deepest = sorted(range(SPAWN0, a.cursor), key=lambda c: -rank[c])[:3]
    print("\n=== longest path to the deepest somata (is it siblings or perception?) ===")
    for cid in deepest:
        path = longest_path(cid)
        comp = Counter(_cat(c) for c in path)
        chain = " → ".join(f"{c}[{_cat(c)[:4]},r{rank[c]}]" for c in path)
        print(f"\n  soma {cid} (rank {rank[cid]}): path length {len(path)}, composition {dict(comp)}")
        print(f"    {chain}")


if __name__ == "__main__":
    main()
