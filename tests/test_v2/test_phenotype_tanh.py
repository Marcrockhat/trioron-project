"""Spec §3.10 tanh phenotype — bounded-saturation affine (added s030).

  1. The forward computes exactly tanh(linear) on the same wiring.
  2. Output is bounded |y| ≤ 1 regardless of fan-in magnitude (the
     anti-detonation property the palette lacked).
  3. The gene is a registered expression gene with a dispatch entry and
     a council seat group (EXPRESSION_GENES membership).
"""
from __future__ import annotations

import torch

from trioron.core import Envelope, construct
from trioron.core.epigenome import (
    EXPRESSION_GENES, LINEAR, OUTPUT, PERCEPTION, TANH, has_gene,
    primary_phenotype,
)
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table


def _build(to_tanh: bool, seed: int = 0):
    torch.manual_seed(seed)
    sub = construct(base=seeded(4, 3, interior_cells=6),
                    envelope=Envelope(max_parameter_bytes=200_000),
                    dispatch_table=default_dispatch_table())
    if to_tanh:
        a = sub.arena
        for cid in a.alive_ids().tolist():
            epi = int(a.epigenome[cid].item())
            if not has_gene(epi, PERCEPTION) and not has_gene(epi, OUTPUT):
                a.epigenome[cid] = (epi & ~(1 << LINEAR)) | (1 << TANH)
                a.refresh_phenotype(cid)
    sub.compile()
    return sub


def test_tanh_is_tanh_of_linear():
    x = torch.randn(16, 4)
    sub_lin = _build(False)
    sub_tan = _build(True)
    a = sub_lin.arena
    # interior activations must satisfy act_tanh == tanh(act_linear) is NOT
    # generally true through depth; assert on a single interior rank instead:
    # outputs read interiors, so compare with one interior layer only when
    # ranks are flat. Here: check the registered fn directly on a live bucket.
    sub_tan(x)
    from trioron.phenotype import linear as lin_mod, tanh as tanh_mod
    plan = sub_tan.scheduler._plan
    bucket = next(b for b in plan.buckets if b.phenotype == TANH)
    act = torch.randn(8, sub_tan.arena.capacity)
    got = tanh_mod.forward_batch(act, bucket, sub_tan.arena)
    want = torch.tanh(lin_mod.forward_batch(act, bucket, sub_tan.arena))
    assert torch.allclose(got, want)


def test_tanh_bounded_under_huge_fanin():
    sub = _build(True)
    a = sub.arena
    with torch.no_grad():
        a.edge_weight[: a.edge_cursor] = 50.0   # detonation-grade weights
    y_interior = []
    sub(torch.randn(8, 4) * 100)
    act = sub.last_activations
    for cid in a.alive_ids().tolist():
        epi = int(a.epigenome[cid].item())
        if has_gene(epi, TANH):
            y_interior.append(act[:, cid])
    assert y_interior, "no tanh cells dispatched"
    assert torch.cat(y_interior).abs().max() <= 1.0 + 1e-6


def test_tanh_is_expression_gene_with_council_seat():
    assert TANH in EXPRESSION_GENES
    assert primary_phenotype(1 << TANH) == TANH
    assert TANH in default_dispatch_table()
    # the germline seats one 4-cell group per expression gene
    from trioron.pcll import Germline, germline_base
    sub = construct(germline_base, capacity=32)
    g = Germline(sub)
    assert TANH in g.council_ids and len(g.council_ids[TANH]) == 4
