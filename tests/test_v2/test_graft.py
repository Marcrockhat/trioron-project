"""Spec §5.3 grafting — head-merged absorption + dendrite-state carry.

  1. merge_output=True, wiring="none": merged(x) == recipient(x) + donor(x)
     (Q-heads sum), output width unchanged, donor OUTPUT cells not copied.
  2. A quad-dendrite donor (seeded nonlinear=True, K=2) keeps K=2 /
     branch_alpha / edge_branch after the graft — the transplanted cells
     compute the same values (test 1 would fail on a linearised donor
     with trained, non-trivial weights).
  3. Protocol B (freeze=False) recipient still trains after the graft
     (trainable tensors stay leaves; a step runs).
  4. Output-width mismatch raises.
"""
from __future__ import annotations

import pytest
import torch

from trioron.core import Envelope, construct
from trioron.core.epigenome import OUTPUT, has_gene
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from trioron.lifecycle.graft import graft


def _build(n_in=9, n_out=4, interior=8, nonlinear=True, seed=0):
    torch.manual_seed(seed)
    sub = construct(base=seeded(n_in, n_out, interior_cells=interior,
                                nonlinear=nonlinear),
                    envelope=Envelope(max_parameter_bytes=200_000),
                    dispatch_table=default_dispatch_table(),
                    capacity=256, sparsity_k=0)
    sub.compile()
    sub.prepare_training()
    # non-trivial weights so a linearised transplant would be detectable
    with torch.no_grad():
        for t in sub.trainable_tensors():
            t.add_(torch.randn_like(t) * 0.3)
    return sub


def test_merge_output_sums_heads():
    rec, don = _build(seed=0), _build(seed=1)
    x = torch.randn(32, 9)
    with torch.no_grad():
        ref = rec(x) + don(x)
        n_out_before = int(has_gene(rec.arena.epigenome[rec.arena.alive],
                                    OUTPUT).sum())
        r = graft(rec, don, freeze=False, wiring="none", merge_output=True)
        y = rec(x)
    n_out_after = int(has_gene(rec.arena.epigenome[rec.arena.alive],
                               OUTPUT).sum())
    assert n_out_after == n_out_before == 4
    assert y.shape == (32, 4)
    assert len(r.recipient_ids) == 8          # interior only
    assert torch.allclose(y, ref, atol=1e-5), (y - ref).abs().max()


def test_dendrite_state_carried():
    rec, don = _build(seed=2), _build(seed=3)
    d_alive = don.arena.alive
    assert int(don.arena.n_branches[d_alive].max()) == 2
    r = graft(rec, don, freeze=True, wiring="none", merge_output=True)
    ids = torch.tensor(r.recipient_ids)
    assert (rec.arena.n_branches[ids] == 2).all()
    src = torch.tensor(r.donor_ids)
    assert torch.allclose(rec.arena.branch_alpha[ids], don.arena.branch_alpha[src])
    ec = rec.arena.edge_cursor
    dst_new = rec.arena.edge_dst[:ec]
    into_new = torch.isin(dst_new, ids.to(dst_new.dtype))
    assert int(rec.arena.edge_branch[:ec][into_new].max()) == 1


def test_protocol_b_trains_after_graft():
    rec, don = _build(seed=4), _build(seed=5)
    graft(rec, don, freeze=False, wiring="none", merge_output=True)
    rec.prepare_training()
    opt = torch.optim.Adam(rec.trainable_tensors(), lr=1e-3)
    x = torch.randn(16, 9)
    before = rec(x).detach().clone()
    loss = rec(x).pow(2).mean()
    opt.zero_grad(); loss.backward(); rec.zero_dormant_grads(); opt.step()
    assert not torch.allclose(rec(x), before)


def test_output_width_mismatch_raises():
    rec, don = _build(n_out=4, seed=6), _build(n_out=3, seed=7)
    with pytest.raises(ValueError):
        graft(rec, don, freeze=False, wiring="none", merge_output=True)
