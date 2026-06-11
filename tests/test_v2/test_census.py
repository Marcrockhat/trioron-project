"""Structural census [D11] — fast CI guards.

Pins the kind partition (computing / germline / astrocyte), depth over
computing cells only, the delta readout, and that every PCLL meeting
report carries a census.
"""
from __future__ import annotations

import torch

from trioron.core import CellState, census, construct
from trioron.core.epigenome import PERCEPTION, RECEPTOR, set_gene


def _arena(cap: int = 32):
    return construct(lambda sub: None, capacity=cap).arena


def test_kind_partition_and_genes():
    a = _arena()
    ids = a.alloc(4)                       # computing linear cells
    book = a.alloc(3)                      # bookkeeping: 2 germline + 1 astrocyte
    a.forward_inclusion[book] = False
    a.epigenome[book[2]] = 0               # astrocyte: no genes
    c = census(a)
    assert (c.cells, c.computing, c.germline, c.astrocytes) == (7, 4, 2, 1)
    assert dict(c.genes)["linear"] == 4
    assert c.dormant == 0
    a.state[ids[0]] = CellState.DORMANT
    assert census(a).dormant == 1


def test_depth_over_computing_only():
    a = _arena()
    ids = a.alloc(3)
    a.add_edges(ids[:1], ids[1:2], torch.zeros(1))
    a.add_edges(ids[1:2], ids[2:3], torch.zeros(1))
    a.rank[ids[1]] = 1
    a.rank[ids[2]] = 2
    book = a.alloc(1)                      # bookkeeping at rank 0
    a.forward_inclusion[book] = False
    c = census(a)
    assert c.edges == 2
    assert c.depth == 2 and c.ranks == (0, 1, 2)
    a.forward_inclusion[ids[2]] = False    # retire the deep cell
    assert census(a).depth == 1            # bookkeeping rank never counts


def test_delta_readout():
    a = _arena()
    before = census(a)
    ids = a.alloc(2)
    a.add_edges(ids[:1], ids[1:2], torch.zeros(1))
    a.rank[ids[1]] = 1
    d = census(a).delta(before)
    assert "computing +2" in d and "edges +1" in d and "depth +1" in d
    assert census(a).delta(census(a)) == "Δ[no structural change]"


def test_meeting_report_carries_census():
    def base(sub):
        a = sub.arena
        ids = a.alloc(2)
        for cid in ids.tolist():
            epi = int(a.epigenome[cid].item())
            a.epigenome[cid] = set_gene(set_gene(epi, PERCEPTION), RECEPTOR)

    sub = construct(base, capacity=16)
    from trioron.pcll import PCLLController

    ctrl = PCLLController(sub)
    ctrl.observe(torch.rand(50, 2))
    r = sub.end_task()
    assert r.census is not None
    assert r.census.computing == 2 and r.census.edges == 0
    assert str(r.census).startswith("census[")
