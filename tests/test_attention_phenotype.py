"""Attention phenotype: reduction-to-linear at 1 token + genuine softmax at ≥2.

See spec §3.3. On the 1-D probe an attention cell has a single fan-in token and
must be exactly linear; with multiple tokens it must compute true segment-softmax
attention over its fan-in.
"""
from __future__ import annotations

import torch

from trioron.core import Envelope, construct
from trioron.core.arena import Arena
from trioron.core.epigenome import (
    LINEAR, ATTENTION, PERCEPTION, OUTPUT, set_gene, clear_gene,
)
from trioron.core.scheduler import Bucket
from trioron.phenotype import default_dispatch_table
from trioron.phenotype import attention as attn


def _chain(hidden_gene: int):
    """perception(1) → hidden(1, hidden_gene) → output(1), fixed weights."""
    def base(sub):
        a = sub.arena
        p = a.alloc(1); h = a.alloc(1); o = a.alloc(1)
        pid, hid, oid = int(p[0]), int(h[0]), int(o[0])
        a.epigenome[pid] = set_gene(int(a.epigenome[pid].item()), PERCEPTION)
        a.rank[pid] = 0
        a.epigenome[hid] = set_gene(clear_gene(int(a.epigenome[hid].item()), LINEAR), hidden_gene)
        a.rank[hid] = 1
        a.epigenome[oid] = set_gene(int(a.epigenome[oid].item()), OUTPUT)
        a.rank[oid] = 2
        a.refresh_all_phenotypes()
        a.add_edges(p, h, torch.tensor([0.7]))
        a.add_edges(h, o, torch.tensor([1.3]))
        with torch.no_grad():
            a.bias[hid] = 0.2; a.bias[oid] = -0.1
    return construct(base=base, envelope=Envelope(),
                     dispatch_table=default_dispatch_table(), capacity=16)


def test_attention_reduces_to_linear_single_token():
    x = torch.linspace(-2, 2, 13).unsqueeze(1)
    with torch.no_grad():
        yl = _chain(LINEAR)(x)
        ya = _chain(ATTENTION)(x)
    assert torch.allclose(yl, ya, atol=1e-6), (yl - ya).abs().max().item()


def test_attention_multi_token_is_softmax_weighted():
    """Two sources into one attention cell → softmax(value)·value, exactly."""
    env = Envelope()
    a = Arena(env, capacity=8)
    h = int(a.alloc(1)[0])
    s0 = int(a.alloc(1)[0]); s1 = int(a.alloc(1)[0])
    with torch.no_grad():
        a.bias[h] = 0.0
    w0, w1 = 0.5, -1.2
    a.add_edges(torch.tensor([s0, s1]), torch.tensor([h, h]), torch.tensor([w0, w1]))

    act = torch.zeros(3, a.capacity)
    act[:, s0] = torch.tensor([1.0, 2.0, -1.0])
    act[:, s1] = torch.tensor([0.5, -1.0, 3.0])

    eidx = torch.tensor([0, 1], dtype=torch.long)
    bucket = Bucket(
        rank=1, phenotype=ATTENTION,
        cell_ids=torch.tensor([h], dtype=torch.int32),
        edge_indices=eidx, edge_src=a.edge_src[eidx],
        edge_dst_local=torch.tensor([0, 0], dtype=torch.long),
        bias_ids=torch.tensor([h], dtype=torch.int32),
    )
    out = attn.forward_batch(act, bucket, a).squeeze(1)

    v0 = w0 * act[:, s0]; v1 = w1 * act[:, s1]
    vals = torch.stack([v0, v1], dim=1)                  # [3, 2]
    alpha = torch.softmax(vals, dim=1)
    expected = (alpha * vals).sum(1)
    assert torch.allclose(out, expected, atol=1e-6), (out, expected)
