"""Composer CONV arm (s036) — fast CI guards.

Pins conv-by-spatial-reuse: the reuse statistic, the lineage spawn's
forward exactness, the weight tie through lineage_root, and the
touching-pair geometry. The full path (trial -> promote -> lineage)
is exercised by the bench runner; these pin the parts.
"""
from __future__ import annotations

import math

import torch

from trioron.core import construct
from trioron.core.epigenome import CONV, PERCEPTION, RECEPTOR, set_gene
from trioron.core.receptor import N_QUANTA
from trioron.pcll import composer as comp


def _receptor_base(n: int):
    def base(sub):
        a = sub.arena
        ids = a.alloc(n)
        for cid in ids.tolist():
            epi = int(a.epigenome[cid].item())
            a.epigenome[cid] = set_gene(set_gene(epi, PERCEPTION), RECEPTOR)
    return base


class TestConvSpec:
    def test_conv_value_is_linear_kernel(self):
        s_conv = comp.ComposerSpec("conv", 0, 1, (1.0, -1.0))
        s_lin = comp.ComposerSpec("linear", 0, 1, (1.0, -1.0))
        c = torch.linspace(-0.5, 0.5, 11)
        assert torch.allclose(s_conv.value(c, -c), s_lin.value(c, -c))
        assert s_conv.frame() == (-1.0, 1.0)

    def test_gene_of_maps_conv(self):
        assert comp.GENE_OF["conv"] == CONV


class TestConvReuse:
    def test_same_relation_at_two_positions(self):
        # cols (0,1) and (2,3) carry the SAME jittered diff relation
        # (no mod-wrap: base+offset stays interior); (4,5) is noise
        g = torch.Generator().manual_seed(0)
        n = 400
        cluster = torch.randint(0, 2, (n,), generator=g).float()
        base = torch.randint(200, 500, (n,), generator=g).float()
        jit = torch.randint(-30, 31, (n,), generator=g).float()
        q = torch.empty(n, 6)
        q[:, 0] = base
        q[:, 1] = base + 150 + 300 * cluster + jit
        base2 = torch.randint(200, 500, (n,), generator=g).float()
        jit2 = torch.randint(-30, 31, (n,), generator=g).float()
        q[:, 2] = base2
        q[:, 3] = base2 + 150 + 300 * cluster + jit2
        q[:, 4] = torch.randint(1, 999, (n,), generator=g).float()
        q[:, 5] = torch.randint(1, 999, (n,), generator=g).float()
        spec = comp.ComposerSpec("linear", 0, 1, (1.0, -1.0))
        hits = comp.conv_reuse(q, spec, [(2, 3), (4, 5)], seed=7)
        assert (2, 3) in hits
        assert (4, 5) not in hits


class TestConvLineageSpawn:
    def test_forward_exactness_and_tie(self):
        sub = construct(_receptor_base(4), capacity=16)
        a = sub.arena
        ids = a.alive.nonzero().squeeze(-1).tolist()
        spec_r = comp.ComposerSpec("conv", 0, 1, (1.0, -1.0))
        spec_s = comp.ComposerSpec("conv", 2, 3, (1.0, -1.0))
        root = comp.spawn_conv(sub, spec_r, (ids[0], ids[1]))
        sib = comp.spawn_conv(sub, spec_s, (ids[2], ids[3]), root=root)
        assert int(a.lineage_root[sib]) == root
        assert int(a.lineage_root[root]) == -1

        x = torch.rand(8, 4)
        sub.forward(x)
        act = sub.scheduler._last_activations
        q = sub.scheduler._last_receptor_q
        c = q / N_QUANTA - 0.5
        # both cells compute the kernel form EXACTLY over pocket values
        assert torch.allclose(act[:, root], spec_r.value(c[:, 0], c[:, 1]),
                              atol=1e-5)
        assert torch.allclose(act[:, sib], spec_s.value(c[:, 2], c[:, 3]),
                              atol=1e-5)

        # the tie: rewrite the ROOT's kernel; the sibling must follow
        ec = a.edge_cursor
        root_edges = (a.edge_dst[:ec] == root).nonzero().squeeze(-1)
        with torch.no_grad():
            a.edge_weight[root_edges] *= 2.0
        sub.forward(x)
        act2 = sub.scheduler._last_activations
        expect = 2.0 * (spec_s.value(c[:, 2], c[:, 3])
                        - (-(1.0 - 1.0) / 2)) + (-(1.0 - 1.0) / 2)
        assert torch.allclose(act2[:, sib], expect, atol=1e-5)

    def test_pruned_root_falls_back_to_own_copy(self):
        sub = construct(_receptor_base(4), capacity=16)
        a = sub.arena
        ids = a.alive.nonzero().squeeze(-1).tolist()
        spec_r = comp.ComposerSpec("conv", 0, 1, (1.0, 1.0))
        spec_s = comp.ComposerSpec("conv", 2, 3, (1.0, 1.0))
        root = comp.spawn_conv(sub, spec_r, (ids[0], ids[1]))
        sib = comp.spawn_conv(sub, spec_s, (ids[2], ids[3]), root=root)
        x = torch.rand(8, 4)
        sub.forward(x)
        before = sub.scheduler._last_activations[:, sib].clone()
        comp.prune(sub, root)
        sub.forward(x)
        after = sub.scheduler._last_activations[:, sib]
        assert torch.allclose(before, after, atol=1e-5)
