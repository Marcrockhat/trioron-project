"""Step 5 — branch-id thin-chain growth (s027 design, Rocky's locality model).

Supersedes the council/divide growth (which tangled: every soma drew from a shared
parent and projected to all consumers → rank-12 sibling pile-up, all-to-all readout).
The new architecture is a 2-D sheet of SHALLOW thin columns, addressed by a hard
``branch_id`` (the ``[x,y]``) and a ``layer`` (depth):

  • LATERAL growth  → mint a FRESH branch_id: a new linear cell at layer 1 reading the
    whole perception (a parallel feature / new column). Width = number of branches.
  • AXIAL growth    → INHERIT the branch_id: a GCU dendrite at layer+1 drawing only from
    its own column's current top (a nonlinear warp of the column's running feature).
    Depth is bounded and LOCAL — composition never crosses branches, so no tangle.
  • READOUT reads TOPS only: each branch's top cell feeds the outputs. When a column
    deepens, its output edges are RE-POINTED (edge_src rewritten in place) from the old
    top to the new one — sparse readout, no all-to-all dilution.

Frustration is split and measured PER LOCALITY:
  • Phase 1 (F_lateral): keep minting branches until a new one no longer helps — width
    SATURATES on its own (an unread input direction stops existing).
  • Phase 2 (F_depth): the residual that survives width is non-linear in the features, so
    deepen the MOST-FRUSTRATED branch (highest top-cell |w·g|). Because the depth trigger
    is per-branch, it is NOT starved by global width — this is the fix for "depth never
    fires" (the recurring failure across past experiments).

Run: python3 -m experiments.progenitor.run_branch
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from trioron.core import Envelope, construct
from trioron.core.epigenome import (
    PERCEPTION, OUTPUT, DENDRITE, LINEAR, set_gene, clear_gene,
)
from trioron.phenotype import default_dispatch_table

from .positions import sensor_positions
from .step3c_council_decides import measure, LR

SEED = 0xBEEF
TRAIN_PER = 120       # training steps after each growth event
WARMUP = 120          # initial settle with the first branch
EPS_W = 0.004         # width saturates when a new branch raises overall by less than this
EPS_D = 0.004         # depth stops when deepening raises overall by less than this
MAX_BRANCHES = 24     # width cap (safety)
MAX_DEPTH = 6         # per-column depth cap (bounded, cortex-like)


# ── base organism: perception + outputs, NO hidden (growth builds every path) ──
def build_base(n_in: int, n_out: int, capacity: int = 512):
    def base(sub):
        a = sub.arena
        perc = a.alloc(n_in)
        ppos = sensor_positions(n_in, seed=SEED, dim=1)
        for k, p in enumerate(perc.tolist()):
            a.epigenome[p] = set_gene(int(a.epigenome[p].item()), PERCEPTION)
            a.rank[p] = 0
            a.position[p] = ppos[k]
        outs = a.alloc(n_out)
        for o in outs.tolist():
            a.epigenome[o] = set_gene(int(a.epigenome[o].item()), OUTPUT)
            a.rank[o] = 2
        a.refresh_all_phenotypes()
        base.perc = perc.tolist()
        base.outs = outs.tolist()
    sub = construct(base=base, envelope=Envelope(),
                    dispatch_table=default_dispatch_table(), capacity=capacity)
    sub.perc, sub.outs = base.perc, base.outs
    # bookkeeping for the locality index
    sub.branch_top, sub.branch_out, sub.branch_cells = {}, {}, {}
    sub.cell_branch, sub.cell_layer = {}, {}
    return sub


def add_branch(sub, bid: int) -> int:
    """LATERAL: a fresh branch_id — one linear cell at layer 1 reading all perception,
    its top wired to every output (the readout reads this branch's top)."""
    a = sub.arena
    cid = int(a.alloc(1)[0].item())
    a.rank[cid] = 1                                        # layer 1
    a.position[cid] = torch.tensor([bid * 0.02, 0.0, 0.1], device=a.device)
    a.refresh_phenotype(cid)                              # linear (default gene)
    perc = torch.tensor(sub.perc, dtype=torch.int32)
    a.add_edges(perc, torch.full((len(sub.perc),), cid, dtype=torch.int32))
    e0 = a.edge_cursor
    outs = torch.tensor(sub.outs, dtype=torch.int32)
    a.add_edges(torch.full((len(sub.outs),), cid, dtype=torch.int32), outs)
    sub.branch_top[bid] = cid
    sub.branch_out[bid] = (e0, a.edge_cursor)            # edge-index range of top→outputs
    sub.branch_cells[bid] = [cid]
    sub.cell_branch[cid], sub.cell_layer[cid] = bid, 1
    return cid


def deepen_branch(sub, bid: int) -> int:
    """AXIAL: GCU dendrite at layer+1 drawing only from this column's current top, then
    RE-POINT the branch's output edges to the new top (readout reads tops only)."""
    a = sub.arena
    old = sub.branch_top[bid]
    L = sub.cell_layer[old] + 1
    cid = int(a.alloc(1)[0].item())
    a.epigenome[cid] = set_gene(clear_gene(int(a.epigenome[cid].item()), LINEAR), DENDRITE)
    a.rank[cid] = L
    a.position[cid] = torch.tensor([bid * 0.02, 0.0, 0.1 * L], device=a.device)
    # two edges old→cid on dendritic branches 0 and 1 → engage the K=2 GCU σ(z)=z·cos z
    e0 = a.edge_cursor
    a.add_edges(torch.tensor([old, old], dtype=torch.int32),
                torch.tensor([cid, cid], dtype=torch.int32))
    a.edge_branch[e0] = 0
    a.edge_branch[e0 + 1] = 1
    a.n_branches[cid] = 2
    with torch.no_grad():
        a.branch_alpha[cid, 1] = 1.0
    a.refresh_phenotype(cid)
    # re-point the readout: the branch's output edges now emanate from the new top
    os, oe = sub.branch_out[bid]
    a.edge_src[os:oe] = cid
    a.rank_dirty = True
    sub.branch_top[bid] = cid
    sub.branch_cells[bid].append(cid)
    sub.cell_branch[cid], sub.cell_layer[cid] = bid, L
    return cid


def _branch_utility(sub, train_d) -> dict[int, float]:
    """Per-branch top-cell saliency Σ|w·g| — the local frustration that picks which
    column to deepen. One backward pass on the live net."""
    a = sub.arena
    a.edge_weight.grad = None
    F.cross_entropy(sub(train_d.x), train_d.y).backward()
    g, w, ec = a.edge_weight.grad, a.edge_weight.detach(), a.edge_cursor
    src, dst = a.edge_src[:ec], a.edge_dst[:ec]
    out = {}
    for bid, top in sub.branch_top.items():
        m = (src == top) | (dst == top)
        out[bid] = float((w[:ec][m] * g[:ec][m]).abs().sum())
    return out


def grow_branches(sub, train_d, test_d, names, bayes_overall):
    sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=LR)

    def step():
        opt.zero_grad()
        F.cross_entropy(sub(train_d.x), train_d.y).backward()
        opt.step()

    def train(n):
        for _ in range(n):
            step()

    def acc():
        return measure(sub, test_d, names)[2]

    # ── Phase 1: lateral saturation (grow width until a new branch stops helping) ──
    print("  Phase 1 — lateral (width): mint branches until F_lateral saturates\n")
    add_branch(sub, 0); sub.compile(); train(WARMUP)
    best = acc()
    print(f"    branch 0 (seed) → overall {best:.3f}")
    bid = 1
    while bid < MAX_BRANCHES:
        add_branch(sub, bid); sub.compile(); train(TRAIN_PER)
        m = acc(); gain = m - best
        print(f"    +branch {bid} (width {bid + 1}) → overall {m:.3f}  Δ{gain:+.3f}")
        if gain < EPS_W:
            print(f"    → F_lateral saturated (Δ<{EPS_W}); width = {bid + 1} branches\n")
            best = m
            break
        best = m; bid += 1
    width_acc = best
    n_width = len(sub.branch_top)

    # ── Phase 2: depth on the most-frustrated branch (per-locality, can't be starved) ──
    print("  Phase 2 — axial (depth): deepen the most-frustrated branch (GCU)\n")
    while True:
        u = _branch_utility(sub, train_d)
        # only branches under the depth cap are eligible
        elig = {b: s for b, s in u.items() if len(sub.branch_cells[b]) < MAX_DEPTH}
        if not elig:
            print("    → all columns at depth cap\n"); break
        target = max(elig, key=elig.get)
        deepen_branch(sub, target); sub.compile(); train(TRAIN_PER)
        m = acc(); gain = m - best
        depth = len(sub.branch_cells[target])
        print(f"    deepen branch {target} → layer {depth} (GCU) → overall {m:.3f}  "
              f"Δ{gain:+.3f}  [most-frustrated u={u[target]:.3f}]")
        if gain < EPS_D:
            print(f"    → F_depth saturated (Δ<{EPS_D})\n"); break
        best = m

    final = acc()
    depths = {b: len(c) for b, c in sub.branch_cells.items()}
    deepened = {b: d for b, d in depths.items() if d > 1}
    print("  ── final organism ──")
    print(f"    branches (width): {len(sub.branch_top)}   "
          f"max depth: {max(depths.values())}   deepened columns: {deepened}")
    print(f"    overall  base→width {width_acc:.3f}  (n={n_width})  →  +depth {final:.3f}")
    print(f"    width contributed {width_acc:.3f}; depth added {final - width_acc:+.3f}  "
          f"(Bayes {bayes_overall:.3f})")
    return sub
