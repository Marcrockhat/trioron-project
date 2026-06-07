"""Spec §3.6 dendrite phenotype — branch partition, per-branch quad, α pooling.

Covers the three things the build conflict turned on:
  1. K=1 dendrite is byte-identical to linear (back-compat guarantee).
  2. K=2 computes the spec formula y = b + Σ_k α_k·σ(z_k), σ(z)=z+z², exactly.
  3. grow_branch + the quad term let a substrate learn a quadratic boundary
     (concentric rings) that a linear (K=1) substrate provably cannot.
"""
from __future__ import annotations

import torch

from trioron.core import Envelope, construct
from trioron.core.arena import Arena
from trioron.core.epigenome import (
    LINEAR, DENDRITE, PERCEPTION, OUTPUT, has_gene,
)
from trioron.core.scheduler import Scheduler
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table


def _to_dendrite(arena, cid):
    epi = int(arena.epigenome[cid].item())
    arena.epigenome[cid] = (epi & ~(1 << LINEAR)) | (1 << DENDRITE)
    arena.refresh_phenotype(cid)


def test_k1_dendrite_byte_identical_to_linear():
    """A K=1 dendrite cell (no grown branch) forwards exactly as linear."""
    x = torch.randn(16, 4)

    torch.manual_seed(0)
    sub_lin = construct(base=seeded(4, 3, interior_cells=6),
                        envelope=Envelope(max_parameter_bytes=200_000),
                        dispatch_table=default_dispatch_table())
    sub_lin.compile()
    y_lin = sub_lin(x)

    torch.manual_seed(0)
    sub_den = construct(base=seeded(4, 3, interior_cells=6),
                        envelope=Envelope(max_parameter_bytes=200_000),
                        dispatch_table=default_dispatch_table())
    # Flip every interior cell to the dendrite gene but DO NOT grow a branch.
    a = sub_den.arena
    for cid in a.alive_ids().tolist():
        epi = int(a.epigenome[cid].item())
        if not has_gene(epi, PERCEPTION) and not has_gene(epi, OUTPUT):
            _to_dendrite(a, cid)
    sub_den.compile()
    y_den = sub_den(x)

    assert torch.allclose(y_lin, y_den, atol=1e-6), (y_lin - y_den).abs().max()


def test_k2_forward_matches_spec_formula():
    """Hand-built K=2 dendrite: out = b + α0·σ(a·x0) + α1·σ(b·x1), σ=z+z²."""
    arena = Arena(Envelope(max_parameter_bytes=10_000), capacity=8, branch_cap=4)
    ids = arena.alloc(3)  # 0,1 perception ; 2 dendrite+output
    arena.epigenome[0] = (1 << LINEAR) | (1 << PERCEPTION)
    arena.epigenome[1] = (1 << LINEAR) | (1 << PERCEPTION)
    arena.epigenome[2] = (1 << DENDRITE) | (1 << OUTPUT)
    arena.bias[2] = 0.5
    wa, wb = 0.7, -1.3
    arena.add_edges(torch.tensor([0, 1]), torch.tensor([2, 2]),
                    torch.tensor([wa, wb]))
    arena.grow_branch(2, [1])           # edge from src 1 → branch 1; src 0 stays branch 0
    assert int(arena.n_branches[2].item()) == 2
    arena.rank[0] = 0; arena.rank[1] = 0; arena.rank[2] = 1
    arena.refresh_all_phenotypes()

    sched = Scheduler(arena, default_dispatch_table())
    sched.compile()

    x = torch.tensor([[2.0, 3.0], [-1.0, 0.5]])
    out = sched.forward(x)

    z0 = wa * x[:, 0]
    z1 = wb * x[:, 1]
    expect = 0.5 + (z0 + z0 * z0) + (z1 + z1 * z1)   # α0=α1=1
    assert torch.allclose(out[:, 0], expect, atol=1e-6), (out[:, 0], expect)


def _rings(n, gen):
    x = (torch.rand(n, 2, generator=gen) * 2 - 1) * 2.0
    r2 = (x ** 2).sum(1)
    y = (r2 > 1.15).long()      # quadratic boundary — linearly inseparable
    return x, y


def _train(sub, steps=400, lr=0.05, seed=0):
    sub.prepare_training()
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    for _ in range(steps):
        x, y = _rings(128, gen)
        loss = torch.nn.functional.cross_entropy(sub(x), y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
        opt.step()
    with torch.no_grad():
        x, y = _rings(2000, gen)
        return (sub(x).argmax(1) == y).float().mean().item()


def test_grown_branch_solves_quadratic_boundary():
    """K=2 (grown branch + quad) clears concentric rings; K=1 linear can't."""
    torch.manual_seed(1)
    lin = construct(base=seeded(2, 2, interior_cells=8),
                    envelope=Envelope(max_parameter_bytes=200_000),
                    dispatch_table=default_dispatch_table())
    lin.compile()
    acc_lin = _train(lin)

    torch.manual_seed(1)
    den = construct(base=seeded(2, 2, interior_cells=8),
                    envelope=Envelope(max_parameter_bytes=200_000),
                    dispatch_table=default_dispatch_table())
    a = den.arena
    perc = [c for c in a.alive_ids().tolist()
            if has_gene(int(a.epigenome[c].item()), PERCEPTION)]
    for cid in a.alive_ids().tolist():
        epi = int(a.epigenome[cid].item())
        if not has_gene(epi, PERCEPTION) and not has_gene(epi, OUTPUT):
            # branch 0 keeps perc[0]; branch 1 gets perc[1] → per-axis quad
            a.grow_branch(cid, [perc[1]])
    den.compile()
    acc_den = _train(den)

    assert acc_lin < 0.80, f"linear unexpectedly solved rings: {acc_lin}"
    assert acc_den > 0.90, f"dendrite failed rings: {acc_den}"
    assert acc_den - acc_lin > 0.15


def test_grown_dendrite_survives_ship_wake(tmp_path):
    """Branch structure + α + per-edge branch survive a ship/wake round-trip."""
    from trioron.lifecycle.ship import ship
    from trioron.lifecycle.wake import wake

    torch.manual_seed(2)
    sub = construct(base=seeded(3, 2, interior_cells=4),
                    envelope=Envelope(max_parameter_bytes=200_000),
                    dispatch_table=default_dispatch_table())
    a = sub.arena
    perc = [c for c in a.alive_ids().tolist()
            if has_gene(int(a.epigenome[c].item()), PERCEPTION)]
    den_cell = next(c for c in a.alive_ids().tolist()
                    if not has_gene(int(a.epigenome[c].item()), PERCEPTION)
                    and not has_gene(int(a.epigenome[c].item()), OUTPUT))
    a.grow_branch(den_cell, [perc[0]])
    with torch.no_grad():
        a.branch_alpha[den_cell, 1] = 0.37
    sub.compile()
    x = torch.randn(8, 3)
    y_before = sub(x)

    p = tmp_path / "den.pt"
    ship(sub, p)
    woken = wake(p, dispatch_table=default_dispatch_table())
    wa = woken.arena

    assert int(wa.n_branches[den_cell].item()) == 2
    assert abs(float(wa.branch_alpha[den_cell, 1].item()) - 0.37) < 1e-6
    assert int(has_gene(int(wa.epigenome[den_cell].item()), DENDRITE)) == 1
    assert torch.allclose(woken(x), y_before, atol=1e-6)
