"""Architecture review (Rocky, s033) — the FULL trained stack on
data_hard: accuracy (raw + σ readouts) and the standing anatomy
printed cell by cell.

Configuration = everything the rebuild has: composer arm ON (zero
spawns expected — data_hard is axis-separable and frame-moving; the
genome regularizer staying quiet IS the result), manifold adapter ON,
divide_tries=4 (the measured +0.051). A second panel grows the M4
relational-testbed organism so the review can see what spawned TISSUE
looks like in the same format.

Run: python3 -m experiments.progenitor.run_review
"""
from __future__ import annotations

import math
from collections import Counter

import torch

from trioron.core import census, construct
from trioron.core.epigenome import EXPRESSION_GENES, GENE_NAMES, has_gene
from trioron.core.state import CellState
from trioron.pcll import (
    MixedStreamController,
    PerceptionGenesis,
    germline_base,
)

from .data_hard import make_spec, sample
from .run_m2_mixed import N_TEST, S_PERIOD, SEEDS, WINDOW
from .run_m4_composer import data as testbed_data
from .run_m5_manifold import mapping_of, read_raw, read_sigma

STATE_NAMES = {int(CellState.ACTIVE): "ACTIVE",
               int(CellState.DORMANT): "DORMANT"}


def describe(sub, mixed, router) -> None:
    """The standing anatomy, cell by cell, from arena fields alone."""
    a = sub.arena
    n = a.cursor
    print(f"  {census(a)}")

    # ── computing cells (the forward path) ────────────────────────
    fi = a.forward_inclusion[:n]
    comp_ids = fi.nonzero(as_tuple=False).squeeze(-1).tolist()
    print(f"\n  COMPUTING CELLS ({len(comp_ids)}) — the forward path:")
    print(f"    {'id':>4s} {'rank':>4s} {'genes':<22s} {'state':<8s} "
          f"{'fan-in':>6s}  edges (src→w)")
    ec = a.edge_cursor
    for cid in comp_ids:
        genes = "+".join(GENE_NAMES[g] for g in range(16)
                         if has_gene(a.epigenome[cid:cid + 1], g).item())
        mask = a.edge_dst[:ec] == cid
        edges = " ".join(
            f"{int(s)}→{float(w):+.3f}"
            for s, w in zip(a.edge_src[:ec][mask], a.edge_weight[:ec][mask]))
        extra = ""
        if int(a.n_branches[cid]) > 1:
            extra = f"  K={int(a.n_branches[cid])} branches"
        print(f"    {cid:>4d} {int(a.rank[cid]):>4d} {genes:<22s} "
              f"{STATE_NAMES.get(int(a.state[cid]), '?'):<8s} "
              f"{int(mask.sum()):>6d}  {edges}{extra}")

    # ── layers = connectivity strata (emergent, spec §1) ──────────
    ranks = Counter(int(a.rank[c]) for c in comp_ids)
    print(f"\n  LAYERS (emergent rank strata): " + ", ".join(
        f"rank {r}: {k} cells" for r, k in sorted(ranks.items())))
    print(f"  EDGES: {ec} total"
          + (f" ({int((a.edge_weight[:ec] == 0).sum())} masked)" if ec else ""))

    # ── germline (bookkeeping with genes) ─────────────────────────
    g = router.germline
    book = "  ".join(
        f"{GENE_NAMES[p]}={sum(g.votes[c] for c in ids):.2f}"
        for p, ids in g.council_ids.items())
    sens = sum(g.votes[s] for s in g.perception_seats)
    print(f"\n  GERMLINE: progenitor cell {g.progenitor_id} + council "
          f"{len(EXPRESSION_GENES)} genes × 4 seats")
    print(f"    vote book (Σ={router.total_votes():.0f}): sensation={sens:.2f}  {book}")

    # ── classes (bookkeeping without genes) ───────────────────────
    live = len(mixed.classes)
    depths = sorted(mixed._depth[mixed._node[c.name]]
                    for c in mixed.classes if c.name in mixed._node)
    hist = Counter(depths)
    print(f"\n  CLASSES: {live} live astrocyte rows "
          f"(+{mixed._births - live} retired by division)")
    print(f"    lineage depth: max {max(depths)}, median "
          f"{depths[len(depths) // 2]}, hist "
          f"{dict(sorted(hist.items()))}")
    sizes = sorted(len(b) for b in mixed.bufs)
    print(f"    buffers: {sum(sizes)} members total, sizes "
          f"min/median/max = {sizes[0]}/{sizes[len(sizes) // 2]}/{sizes[-1]}")

    # ── composer tissue + sketches ────────────────────────────────
    if mixed.specs:
        kept = ", ".join(f"{s.gene}({s.i},{s.j})" for s in mixed.specs)
        print(f"  COMPOSER TISSUE: {len(mixed.specs)} cells [{kept}]")
    else:
        print("  COMPOSER TISSUE: none (zero spawns)")
    if mixed.manifold is not None:
        n_sk = len(mixed.manifold.sketches)
        width = (next(iter(mixed.manifold.sketches.values())).code_dim
                 if n_sk else 0)
        print(f"  SKETCHES: {n_sk} (μ/σ over {width}-dim pocket space, "
              f"{n_sk * width * 2 * 4 / 1024:.1f} KiB)")


def grow_hard(seed: int):
    torch.manual_seed(seed)
    spec = make_spec()
    tr, norm = sample(spec, n_per_class=S_PERIOD, seed=seed)
    te, _ = sample(spec, n_per_class=N_TEST, seed=100 + seed, norm=norm)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(tr.y), generator=g)
    Xs, ys = tr.x[order], tr.y[order]

    sub = construct(germline_base, capacity=512)
    pg = PerceptionGenesis(sub)
    pg.feed(Xs[:WINDOW])
    sub.end_task()
    mixed = MixedStreamController(sub, stress=pg.router, adopt=pg.controller,
                                  composer=True, manifold=True,
                                  divide_tries=4)
    mixed.observe(Xs[:WINDOW])
    sub.end_task()
    for w0 in range(WINDOW, len(Xs), WINDOW):
        mixed.observe(Xs[w0:w0 + WINDOW])
        sub.end_task()

    clean = [c for c in range(len(tr.names)) if float(spec.std[c]) <= 0.3]
    keep = torch.isin(te.y, torch.tensor(clean))
    return sub, mixed, pg.router, (Xs, ys), (te.x[keep], te.y[keep])


def grow_testbed(seed: int):
    torch.manual_seed(seed)
    Xs, ys, Xt, yt = testbed_data(seed)
    sub = construct(germline_base, capacity=128)
    pg = PerceptionGenesis(sub)
    pg.feed(Xs[:WINDOW * 0 + 600])
    sub.end_task()
    mixed = MixedStreamController(sub, stress=pg.router, adopt=pg.controller,
                                  composer=True, manifold=True,
                                  divide_tries=4)
    mixed.observe(Xs[:600])
    sub.end_task()
    for w0 in range(600, len(Xs), 600):
        mixed.observe(Xs[w0:w0 + 600])
        sub.end_task()
    return sub, mixed, pg.router, (Xs, ys), (Xt, yt)


def main() -> None:
    print("══ data_hard (32 classes × 3 modes, 12 dims) — FULL STACK ══")
    raws, sigs = [], []
    keep_first = None
    for seed in range(SEEDS):
        sub, mixed, router, (Xs, ys), (Xt, yt) = grow_hard(seed)
        truth_of = mapping_of(mixed, Xs, ys)
        raw = read_raw(mixed, truth_of, Xt, yt)
        sig = read_sigma(mixed, truth_of, Xt, yt)
        raws.append(raw)
        sigs.append(sig)
        print(f"  seed={seed}: {len(mixed.classes)} classes; "
              f"raw {raw:.3f}  σ-readout {sig:.3f}")
        if seed == 0:
            keep_first = (sub, mixed, router)
    print(f"  mean (clean classes): raw {sum(raws)/SEEDS:.3f}  "
          f"σ-readout {sum(sigs)/SEEDS:.3f}  (Bayes ceiling 0.993)")

    print("\n══ trained architecture — data_hard seed 0 ══")
    describe(*keep_first)

    print("\n══ comparison panel: the M4 relational-testbed organism "
          "(seed 0) — what spawned TISSUE looks like ══")
    sub, mixed, router, (Xs, ys), (Xt, yt) = grow_testbed(0)
    truth_of = mapping_of(mixed, Xs, ys)
    raw = read_raw(mixed, truth_of, Xt, yt)
    sig = read_sigma(mixed, truth_of, Xt, yt)
    print(f"  accuracy: raw {raw:.3f}  σ-readout {sig:.3f}")
    describe(sub, mixed, router)


if __name__ == "__main__":
    main()
