"""Recurrent phenotype: reduction-to-linear guarantee + genuine unroll math.

The council stands recurrent cells up on the 1-D probe; they must be (a) exactly
linear when no back-edge exists (so they don't perturb the linear floor) and
(b) genuinely recurrent when a self-edge is present. See spec §3.5.
"""
from __future__ import annotations

import torch

from trioron.core import Envelope, construct
from trioron.core.arena import Arena
from trioron.core.epigenome import (
    LINEAR, RECURRENT, PERCEPTION, OUTPUT, set_gene, clear_gene,
)
from trioron.core.scheduler import Bucket
from trioron.phenotype import default_dispatch_table
from trioron.phenotype import recurrent as rec


def _two_cell_chain(hidden_gene: int):
    """perception(1) → hidden(1, hidden_gene) → output(1), fixed weights."""
    def base(sub):
        a = sub.arena
        p = a.alloc(1); h = a.alloc(1); o = a.alloc(1)
        pid, hid, oid = int(p[0]), int(h[0]), int(o[0])
        a.epigenome[pid] = set_gene(int(a.epigenome[pid].item()), PERCEPTION)
        a.rank[pid] = 0
        epi = clear_gene(int(a.epigenome[hid].item()), LINEAR)
        a.epigenome[hid] = set_gene(epi, hidden_gene)
        a.rank[hid] = 1
        a.epigenome[oid] = set_gene(int(a.epigenome[oid].item()), OUTPUT)
        a.rank[oid] = 2
        a.refresh_all_phenotypes()
        a.add_edges(p, h, torch.tensor([0.7]))
        a.add_edges(h, o, torch.tensor([1.3]))
        with torch.no_grad():
            a.bias[hid] = 0.2
            a.bias[oid] = -0.1
        base.ids = (pid, hid, oid)
    sub = construct(base=base, envelope=Envelope(),
                    dispatch_table=default_dispatch_table(), capacity=16)
    sub.ids = base.ids
    return sub


def test_recurrent_reduces_to_linear_without_back_edge():
    """A RECURRENT cell with only feed-forward fan-in == a LINEAR cell."""
    x = torch.linspace(-2, 2, 13).unsqueeze(1)
    lin = _two_cell_chain(LINEAR)
    rnn = _two_cell_chain(RECURRENT)
    with torch.no_grad():
        yl = lin(x)
        yr = rnn(x)
    assert torch.allclose(yl, yr, atol=1e-6), (yl - yr).abs().max().item()


def test_recurrent_self_edge_unrolls_geometrically():
    """A self-edge w_self unrolled K times gives base·(1+w_self)^K (spec §3.5)."""
    env = Envelope()
    a = Arena(env, capacity=8)
    h = int(a.alloc(1)[0])                       # the recurrent cell (its own bucket)
    src = int(a.alloc(1)[0])                     # a feed-forward source
    with torch.no_grad():
        a.bias[h] = 0.5
    w_ff, w_self, K = 0.9, 0.3, 3
    a.add_edges(torch.tensor([src]), torch.tensor([h]), torch.tensor([w_ff]))    # ff
    a.add_edges(torch.tensor([h]),   torch.tensor([h]), torch.tensor([w_self]))  # self
    a.k_unroll[h] = K

    batch = 4
    act = torch.zeros(batch, a.capacity)
    act[:, src] = torch.tensor([1.0, -1.0, 2.0, 0.0])

    # Bucket containing only the recurrent cell h; both its incoming edges.
    eidx = torch.tensor([0, 1], dtype=torch.long)
    bucket = Bucket(
        rank=1, phenotype=RECURRENT,
        cell_ids=torch.tensor([h], dtype=torch.int32),
        edge_indices=eidx,
        edge_src=a.edge_src[eidx],
        edge_dst_local=torch.tensor([0, 0], dtype=torch.long),
        bias_ids=torch.tensor([h], dtype=torch.int32),
    )
    out = rec.forward_batch(act, bucket, a).squeeze(1)
    base = a.bias[h] + w_ff * act[:, src]
    expected = base * (1.0 + w_self) ** K
    assert torch.allclose(out, expected, atol=1e-6), (out, expected)
