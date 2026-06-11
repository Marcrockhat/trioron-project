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
from .frustration_gate import FrustrationGate

SEED = 0xBEEF
TRAIN_PER = 120       # training steps after each growth event
WARMUP = 120          # initial settle with the first branch
EPS_W = 0.004         # width saturates when a new branch raises overall by less than this
EPS_D = 0.004         # depth stops when deepening raises overall by less than this
MAX_BRANCHES = 40     # width cap (safety)
MAX_DEPTH = 10        # per-column depth cap (bounded, cortex-like)
SUBSET_FRAC = 0.5     # lateral branch reads this fraction of perception (frustration-picked)
AXIAL_PERC_K = 4      # fresh perception dims a deepening GCU cell pulls in (wide axial fan)
FRUST_TAU = 0.25      # a class is "stuck at a bad level" when CE/log(C) > this (1.0 = chance)
FRUST_EPS = 0.01      # per-class CE drop over a train block below which it has plateaued
NOGAIN_PATIENCE = 2   # retire a class after this many consecutive no-gain growth events (hopeless)


def _frustrated_perc(sub, train_d, k: int):
    """Frustration-targeted input selection: the k perception dims whose feature most
    correlates with the live per-sample error. As earlier branches absorb a direction
    its residual-correlation falls, so later branches pick DIFFERENT dims → the fan
    differentiates instead of every branch reading all of perception (the symmetric
    full-fan that capped width at the linear ceiling). Returns perception CELL ids, or
    ``None`` when the error has no variance yet (no net → degenerate, caller falls back).
    """
    with torch.no_grad():
        logits = sub(train_d.x)
        resid = torch.softmax(logits, dim=1) - F.one_hot(train_d.y, logits.shape[1]).float()
        err = resid.abs().sum(dim=1)                          # [N] per-sample error
    if float(err.std()) < 1e-6:
        return None
    X = train_d.x
    Xc = X - X.mean(0, keepdim=True)
    ec = err - err.mean()
    corr = (Xc * ec.unsqueeze(1)).mean(0) / (Xc.std(0) * ec.std() + 1e-8)   # [n_in]
    top = corr.abs().topk(min(k, X.shape[1])).indices.tolist()
    return [sub.perc[j] for j in top]


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


def add_branch(sub, bid: int, perc_ids=None) -> int:
    """LATERAL: a fresh branch_id — one linear cell at layer 1, its top wired to every
    output (the readout reads this branch's top). ``perc_ids`` is the (frustration-picked)
    perception subset this branch reads; ``None`` = read all (the generalist seed branch).
    Sparse, branch-specific fan is the asymmetry: columns specialize to different inputs."""
    a = sub.arena
    cid = int(a.alloc(1)[0].item())
    a.rank[cid] = 1                                        # layer 1
    a.position[cid] = torch.tensor([bid * 0.02, 0.0, 0.1], device=a.device)
    a.refresh_phenotype(cid)                              # linear (default gene)
    reads = sub.perc if perc_ids is None else perc_ids
    perc = torch.tensor(reads, dtype=torch.int32)
    a.add_edges(perc, torch.full((len(reads),), cid, dtype=torch.int32))
    sub.branch_reads = getattr(sub, "branch_reads", {})
    sub.branch_reads[bid] = list(reads)
    e0 = a.edge_cursor
    outs = torch.tensor(sub.outs, dtype=torch.int32)
    a.add_edges(torch.full((len(sub.outs),), cid, dtype=torch.int32), outs)
    sub.branch_top[bid] = cid
    sub.branch_out[bid] = (e0, a.edge_cursor)            # edge-index range of top→outputs
    sub.branch_cells[bid] = [cid]
    sub.cell_branch[cid], sub.cell_layer[cid] = bid, 1
    return cid


def deepen_branch(sub, bid: int, perc_ids=None) -> int:
    """AXIAL: GCU dendrite at layer+1. WIDE FAN — it mixes the column's running feature
    (its current top) with fresh frustration-targeted perception ``perc_ids``, split
    across the two dendritic branches so each branch's z = w·(running) + Σ w·(raw) and the
    GCU σ(z)=z·cos z couples them (a real cross-layer nonlinear feature, not a pointwise
    warp of one scalar). Perception is rank-0 and the running top is this column's own, so
    composition stays branch-local — no sibling tangle. Then RE-POINT the branch's output
    edges to the new top (readout reads tops only)."""
    a = sub.arena
    old = sub.branch_top[bid]
    L = sub.cell_layer[old] + 1
    cid = int(a.alloc(1)[0].item())
    a.epigenome[cid] = set_gene(clear_gene(int(a.epigenome[cid].item()), LINEAR), DENDRITE)
    a.rank[cid] = L
    a.position[cid] = torch.tensor([bid * 0.02, 0.0, 0.1 * L], device=a.device)
    # branch 0 and 1 each carry the running feature (old) + a half of the fresh perception
    perc = list(perc_ids or [])
    srcs = [old, old]                                       # running feature into both branches
    brs = [0, 1]
    for i, p in enumerate(perc):
        srcs.append(int(p))
        brs.append(i % 2)                                  # alternate raw evidence across branches
    e0 = a.edge_cursor
    a.add_edges(torch.tensor(srcs, dtype=torch.int32),
                torch.full((len(srcs),), cid, dtype=torch.int32))
    for j, b in enumerate(brs):
        a.edge_branch[e0 + j] = b
    a.n_branches[cid] = 2                                   # K=2 → engage GCU σ(z)=z·cos z
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

    n_in = len(sub.perc)
    sub_k = max(3, round(n_in * SUBSET_FRAC))

    # ── Phase 1: lateral saturation (grow width until a new branch stops helping) ──
    print("  Phase 1 — lateral (width): mint branches until F_lateral saturates\n")
    add_branch(sub, 0); sub.compile(); train(WARMUP)        # seed = full-fan generalist
    best = acc()
    print(f"    branch 0 (seed, full fan {n_in}) → overall {best:.3f}")
    bid = 1
    while bid < MAX_BRANCHES:
        picks = _frustrated_perc(sub, train_d, sub_k)       # asymmetric: pick still-wrong dims
        add_branch(sub, bid, perc_ids=picks); sub.compile(); train(TRAIN_PER)
        m = acc(); gain = m - best
        nf = n_in if picks is None else len(picks)
        print(f"    +branch {bid} (width {bid + 1}, fan {nf}) → overall {m:.3f}  Δ{gain:+.3f}")
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
        picks = _frustrated_perc(sub, train_d, AXIAL_PERC_K)   # wide axial fan: fresh evidence
        deepen_branch(sub, target, perc_ids=picks); sub.compile(); train(TRAIN_PER)
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


# ── class-driven growth: grow capacity at the WORST class (widen, then deepen) ──

def _class_perc(sub, train_d, cls: int, k: int):
    """Class-discriminative input selection: the k perception dims whose feature most
    separates class ``cls`` from the rest. The growing cell reads the inputs that *see*
    the struggling class — Rocky's rule: a class the net is bad at recruits cells wired
    to that class's evidence. Always well-defined while the class has samples."""
    X = train_d.x
    t = (train_d.y == cls).float()
    Xc = X - X.mean(0, keepdim=True)
    tc = t - t.mean()
    if float(tc.std()) < 1e-6:
        return None
    corr = (Xc * tc.unsqueeze(1)).mean(0) / (Xc.std(0) * tc.std() + 1e-8)
    top = corr.abs().topk(min(k, X.shape[1])).indices.tolist()
    return [sub.perc[j] for j in top]


def _branch_utility_class(sub, train_d, cls: int) -> dict[int, float]:
    """Per-branch top-cell saliency Σ|w·g| measured on class-``cls`` samples ONLY — which
    column is most responsible for (most able to relieve) this class. Picks the deepen
    locus so depth lands where the struggling class actually flows, not in one global sink."""
    a = sub.arena
    mask = train_d.y == cls
    a.edge_weight.grad = None
    F.cross_entropy(sub(train_d.x[mask]), train_d.y[mask]).backward()
    g, w, ec = a.edge_weight.grad, a.edge_weight.detach(), a.edge_cursor
    src, dst = a.edge_src[:ec], a.edge_dst[:ec]
    out = {}
    for bid, top in sub.branch_top.items():
        m = (src == top) | (dst == top)
        out[bid] = float((w[:ec][m] * g[:ec][m]).abs().sum())
    return out


def grow_class_driven(sub, train_d, test_d, names, bayes_overall, max_events: int = 40):
    """Unified growth driven by the chance-anchored frustration gate (s028).

    Each round measures per-class CE and consults the gate (``frustration_gate``):
      • no class high (CE/log C ≤ τ for all)        → converged, stop;
      • some class high but still improving          → TRAIN more, don't grow yet;
      • a class high AND plateaued (frustrated)      → grow capacity reading its evidence —
        WIDEN (new linear column) while linear width helps it, then DEEPEN (GCU on the column
        most responsible for it).
    A class that survives ``NOGAIN_PATIENCE`` consecutive no-gain growths is retired as
    likely-irreducible (e.g. a wide disruptor), so capacity is not burnt on it. No Bayes
    ceiling and no soma cap drive the stop — frustration clearing does."""
    sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=LR)

    def train(n):
        for _ in range(n):
            opt.zero_grad()
            F.cross_entropy(sub(train_d.x), train_d.y).backward()
            opt.step()

    def acc():
        return measure(sub, test_d, names)[2]

    n_in = len(sub.perc)
    C = len(sub.outs)
    sub_k = max(3, round(n_in * SUBSET_FRAC))
    gate = FrustrationGate(C, tau=FRUST_TAU, eps_improve=FRUST_EPS)

    print(f"  Class-driven growth — gate: plateau AND CE/log(C)>{FRUST_TAU} "
          f"(chance CE=log{C}={gate.logC:.2f})\n")
    add_branch(sub, 0); sub.compile(); train(WARMUP)        # seed generalist
    best = acc()
    print(f"    branch 0 (seed, full fan {n_in}) → overall {best:.3f}\n")

    bid = 1
    mode = {c: "WIDEN" for c in range(C)}                   # per-class growth phase
    nogain = {c: 0 for c in range(C)}                       # consecutive no-gain growths
    retired: set[int] = set()
    why = "soma cap"
    for ev in range(max_events):
        res = gate.step(measure(sub, train_d, names)[1])
        if not res.any_high:
            why = f"converged — every class below τ (max f={res.max_f:.2f})"; break
        cand = [c for c in range(C) if res.frustrated[c] and c not in retired]
        if not cand:
            still_open = [c for c in range(C) if res.f[c] > FRUST_TAU and c not in retired]
            if not still_open:                             # all high classes are retired
                why = "converged — remaining high classes retired (irreducible)"; break
            train(TRAIN_PER)                               # high but still improving → train
            print(f"    [{ev:>2}] training (high but improving; max f={res.max_f:.2f})")
            continue
        cstar = max(cand, key=lambda c: res.f[c])

        if mode[cstar] == "WIDEN":
            if bid >= MAX_BRANCHES:
                mode[cstar] = "DEEPEN"; continue
            picks = _class_perc(sub, train_d, cstar, sub_k)
            add_branch(sub, bid, perc_ids=picks); sub.compile(); train(TRAIN_PER)
            m = acc(); gain = m - best
            print(f"    [{ev:>2}] class {cstar:>2} ({names[cstar][:10]:<10}) f={res.f[cstar]:.2f} "
                  f"WIDEN  +branch {bid} (fan {len(picks)}) → {m:.3f}  Δ{gain:+.3f}")
            bid += 1
            if gain < EPS_W:
                mode[cstar] = "DEEPEN"                      # linear width spent for this class
        else:  # DEEPEN
            u = _branch_utility_class(sub, train_d, cstar)
            elig = {b: s for b, s in u.items() if len(sub.branch_cells[b]) < MAX_DEPTH}
            if not elig:
                mode[cstar] = "WIDEN"; continue            # no eligible column → widen instead
            target = max(elig, key=elig.get)
            picks = _class_perc(sub, train_d, cstar, AXIAL_PERC_K)
            deepen_branch(sub, target, perc_ids=picks); sub.compile(); train(TRAIN_PER)
            m = acc(); gain = m - best
            depth = len(sub.branch_cells[target])
            print(f"    [{ev:>2}] class {cstar:>2} ({names[cstar][:10]:<10}) f={res.f[cstar]:.2f} "
                  f"DEEPEN branch {target} → L{depth} → {m:.3f}  Δ{gain:+.3f}")

        if gain < EPS_D:
            nogain[cstar] += 1
            if nogain[cstar] >= NOGAIN_PATIENCE:
                retired.add(cstar)                         # repeatedly unhelped → likely irreducible
        else:
            nogain[cstar] = 0
            best = m

    final = acc()
    depths = {b: len(c) for b, c in sub.branch_cells.items()}
    deepened = {b: d for b, d in depths.items() if d > 1}
    print(f"\n  ── final organism ── (stopped: {why})")
    print(f"    branches (width): {len(sub.branch_top)}   "
          f"max depth: {max(depths.values())}   deepened columns: {deepened}")
    print(f"    overall {final:.3f}   (Bayes {bayes_overall:.3f})   "
          f"events {ev + 1}   retired classes {len(retired)}/{C}")
    return sub
