"""Conv phenotype: reduction-to-linear at own-root + genuine kernel sharing.

See spec §3.4. On the 1-D probe a conv cell is its own root and must be exactly
linear; when receptors share a lineage_root they must share one kernel weight
(forward tied, and gradients tied).
"""
from __future__ import annotations

import torch

from trioron.core import Envelope, construct
from trioron.core.arena import Arena
from trioron.core.epigenome import (
    LINEAR, CONV, PERCEPTION, OUTPUT, set_gene, clear_gene,
)
from trioron.core.scheduler import Bucket
from trioron.phenotype import default_dispatch_table
from trioron.phenotype import conv


def _chain(hidden_gene: int):
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


def test_conv_reduces_to_linear_at_own_root():
    """A conv cell with no shared lineage (own root) == a linear cell."""
    x = torch.linspace(-2, 2, 13).unsqueeze(1)
    with torch.no_grad():
        yl = _chain(LINEAR)(x)
        yc = _chain(CONV)(x)
    assert torch.allclose(yl, yc, atol=1e-6), (yl - yc).abs().max().item()


def _conv_bucket(a, recv_cells):
    cids = torch.tensor(recv_cells, dtype=torch.int32)
    dst = a.edge_dst[: a.edge_cursor]
    eidx = torch.isin(dst, cids).nonzero(as_tuple=False).squeeze(-1)
    local_of = {c: i for i, c in enumerate(recv_cells)}
    dst_local = torch.tensor([local_of[int(dst[i])] for i in eidx], dtype=torch.long)
    return Bucket(rank=1, phenotype=CONV, cell_ids=cids, edge_indices=eidx,
                  edge_src=a.edge_src[eidx], edge_dst_local=dst_local, bias_ids=cids)


def test_conv_shares_kernel_across_receptors():
    """Two receptors sharing a root apply the ROOT's weight, not their own;
    and the shared weight receives both receptors' gradients."""
    a = Arena(Envelope(), capacity=8)
    root = int(a.alloc(1)[0]); recv = int(a.alloc(1)[0])
    s0 = int(a.alloc(1)[0]); s1 = int(a.alloc(1)[0])
    a.add_edges(torch.tensor([s0]), torch.tensor([root]), torch.tensor([0.5]))   # kernel = 0.5
    a.add_edges(torch.tensor([s1]), torch.tensor([recv]), torch.tensor([9.9]))   # own weight ignored
    a.lineage_root[root] = root
    a.lineage_root[recv] = root        # recv shares root's kernel

    act = torch.zeros(2, a.capacity)
    act[:, s0] = torch.tensor([1.0, 2.0])
    act[:, s1] = torch.tensor([3.0, 4.0])

    a.edge_weight.requires_grad_(True)
    bucket = _conv_bucket(a, [root, recv])
    out = conv.forward_batch(act, bucket, a)        # [2, 2] columns: [root, recv]

    # recv must use the ROOT's kernel (0.5), NOT its own 9.9
    expected_recv = 0.5 * act[:, s1]
    assert torch.allclose(out[:, 1], expected_recv, atol=1e-6), (out[:, 1], expected_recv)

    # gradient on the shared kernel edge gets contributions from BOTH receptors
    out.sum().backward()
    g = a.edge_weight.grad
    kernel_edge_grad = float(g[0])                   # edge 0 = root's kernel weight
    own_recv_grad = float(g[1])                      # edge 1 = recv's (unused) weight
    assert kernel_edge_grad != 0.0
    assert own_recv_grad == 0.0                      # recv's own weight is bypassed
