"""Lineage routing against the exhaustive membership (s070; Rocky: "why it need to compare
each class ... and not taking the important"; "Yes, try the lineage routing on a copy").

READ-ONLY: a saved body is loaded and measured in memory; nothing is saved, no organism
or copy changes. One organism, one seed.

Today a member is placed with the class whose template matches it best, found by comparing
it with EVERY class. Every class also knows the class it was split from. Two ways of
comparing with kin only, each against the exhaustive answer on the same members:

  descent   start at the root of her lineage; at each division go to the side whose
            pooled template matches best; arrive at a class; then compare with every
            class under the ancestor `up` divisions above it. (For a new moment.)
  home      start at the class the member is in now; compare with every class under the
            ancestor `up` divisions above it. (For the consolidation round.)

A retired parent has no template of its own any more, so a division's side is read as the
pooled mean of the members of every living class under it.

    BODY=outputs/incubator/copies/weiss60_plain/body.pt python3 experiments/incubator/lineage_route_probe.py
Env: N=40000 members sampled per organ (all, if fewer); SEED=0; ORGANS=ear,soma,kin.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

BODY = os.environ.get("BODY", "outputs/incubator/copies/weiss60_plain/body.pt")
N, SEED = int(os.environ.get("N", 40000)), int(os.environ.get("SEED", 0))
UPS = (0, 1, 2, 3, 4, 6, 8)


def tree(m):
    """Her lineage with dead branches cut and single-child chains closed up.
    Returns kids {node: [nodes]}, leaf {node: class index}, root, parent {node: node}."""
    live = {m._node[c.name]: k for k, c in enumerate(m.classes) if c.name in m._node}
    kids: dict = {}
    for nid in live:                                    # every ancestor of a living class
        a = nid
        while a is not None:
            p = m._parent_node.get(a)
            sibs = kids.setdefault(p, [])
            if a in sibs:
                break
            sibs.append(a)
            a = p
    def close(n):                                       # skip nodes with one child
        while n not in live and len(kids.get(n, [])) == 1:
            n = kids[n][0]
        return n
    out, parent, stack = {}, {}, ["ROOT"]
    top = [close(n) for n in kids.get(None, [])]
    out["ROOT"] = top
    for t in top:
        parent[t] = "ROOT"
    stack = list(top)
    while stack:
        n = stack.pop()
        if n in live:
            continue
        out[n] = [close(c) for c in kids.get(n, [])]
        for c in out[n]:
            parent[c] = n
        stack.extend(out[n])
    orphans = [k for k, c in enumerate(m.classes) if c.name not in m._node]
    return out, live, "ROOT", parent, orphans


def measure(name, m, g):
    kids, live, root, parent, orphans = tree(m)
    C = len(m.bufs)
    sizes = torch.tensor([len(b) for b in m.bufs])
    home_all = torch.repeat_interleave(torch.arange(C), sizes)
    M_all = torch.cat(m.bufs)
    pick = torch.randperm(len(M_all), generator=g)[:N].sort().values
    Z, home = M_all[pick], home_all[pick]
    T = torch.stack([b.mean(0) for b in m.bufs])
    best, _ = m._margins(Z)                              # the exhaustive answer (the code as it is)
    sig = ((T.abs() ** 2).sum(-1) / 2).sqrt().clamp_min(1e-9)
    ranked = m.member_margin and m._skeptical            # a settled organ ranks by margin, not raw evidence
    E_best = (Z * T[best].conj()).real.sum(-1)

    # pooled sums under every node, leaves under every node, depth
    S = {n: m.bufs[k].sum(0) for n, k in live.items()}
    cnt = {n: len(m.bufs[k]) for n, k in live.items()}
    under = {n: [k] for n, k in live.items()}
    order, stack = [], [root]
    while stack:
        n = stack.pop()
        order.append(n)
        stack.extend(kids.get(n, []))
    for n in reversed(order):
        if n in live:
            continue
        ch = kids[n]
        S[n] = torch.stack([S[c] for c in ch]).sum(0)
        cnt[n] = sum(cnt[c] for c in ch)
        under[n] = [k for c in ch for k in under[c]]
    node_of = {k: n for n, k in live.items()}
    depth = {root: 0}
    for n in order[1:]:
        depth[n] = depth[parent[n]] + 1
    dl = torch.tensor([depth[n] for n in live], dtype=torch.float)

    # descent: members travel down together
    landed = torch.full((len(Z),), -1, dtype=torch.long)
    cmp_desc = torch.zeros(len(Z))
    work = [(root, torch.arange(len(Z)))]
    while work:
        n, idx = work.pop()
        if n in live:
            landed[idx] = live[n]
            continue
        ch = kids[n]
        Tc = torch.stack([S[c] / max(1, cnt[c]) for c in ch])
        side = (Z[idx].unsqueeze(1) * Tc.conj().unsqueeze(0)).real.sum(-1).argmax(1)
        cmp_desc[idx] += len(ch)
        for j, c in enumerate(ch):
            sel = idx[side == j]
            if len(sel):
                work.append((c, sel))

    def kin_best(start, up):
        """Best class under the ancestor `up` divisions above each member's start class."""
        anc = []
        for k in range(C):
            n = node_of.get(k)
            for _ in range(up):
                if n is None or n == root:
                    break
                n = parent[n]
            anc.append(n)
        out = start.clone()
        ncmp = torch.zeros(len(Z))
        groups: dict = {}
        for i, k in enumerate(start.tolist()):
            groups.setdefault(anc[k], []).append(i)
        for n, rows in groups.items():
            if n is None:
                continue
            rows = torch.tensor(rows)
            cand = torch.tensor(under[n])
            for i in range(0, len(rows), 2000):          # never a whole table
                r = rows[i:i + 2000]
                E = m._evidence(Z[r], T[cand])
                out[r] = cand[(E / sig[cand] if ranked else E).argmax(1)]   # her own ranking
            ncmp[rows] = len(cand)
        return out, ncmp

    print(f"\n== {name}: {C} classes, {len(M_all)} members held, {len(Z)} measured; her lineage is "
          f"{dl.mean():.1f} divisions deep on average (deepest {int(dl.max())}); {len(orphans)} classes outside it")
    print(f"   exhaustive: {C} comparisons a member; {float((best == home).float().mean()):.3f} of members "
          f"are already in their best class; she ranks by {'margin (settled)' if ranked else 'raw evidence'}")
    print("   way      up   same class as exhaustive   of the rest, evidence kept   comparisons a member (share of exhaustive)")
    for way, start, base in (("descent", landed, cmp_desc), ("home", home, torch.zeros(len(Z)))):
        for up in UPS:
            got, ncmp = kin_best(start, up) if up else (start, torch.zeros(len(Z)))
            same = got == best
            E_got = (Z * T[got].conj()).real.sum(-1)
            kept = float((E_got[~same] / E_best[~same].clamp_min(1e-9)).mean()) if bool((~same).any()) else 1.0
            n_c = float((base + ncmp).mean())
            print(f"   {way:8s} {up:2d}   {float(same.float().mean()):.3f}                     {kept:.3f}"
                  f"                        {n_c:7.1f}  ({n_c / C:.3f})")


def main():
    body = torch.load(BODY, weights_only=False)
    print(f"# lineage routing against the exhaustive membership; body {BODY}; "
          f"moments lived {body.moments}; one organism, read only")
    g = torch.Generator().manual_seed(SEED)
    only = os.environ.get("ORGANS")
    for name, leaf in body.organs().items():
        if only is None or name in only.split(","):
            measure(name, leaf.mixed, g)


if __name__ == "__main__":
    main()
