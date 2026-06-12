"""Composer arm unit tests [D14+D16+§4b] — the spawn identities, the
hard prune, the gene-targeted vote economy, and multi-try division."""
from __future__ import annotations

import math

import pytest
import torch

from trioron.core import census, construct
from trioron.core.epigenome import DENDRITE, LINEAR, TANH, has_gene
from trioron.core.receptor import N_QUANTA
from trioron.pcll import Germline, StressRouter, germline_base
from trioron.pcll.composer import (
    ComposerSpec,
    best_candidate,
    candidates,
    prune,
    spawn,
    trial_ratio,
)
from trioron.pcll.division import try_divide


def _organism(n_cols=4):
    """Germline + receptor-equipped perception layer, compiled."""
    sub = construct(germline_base, capacity=64)
    germ = Germline(sub)
    ids = germ.spawn_perception(n_cols)
    germ.equip_receptors(ids)
    sub.compile()
    return sub, germ, ids


def _phase(q):
    return 2 * math.pi * q / N_QUANTA


class TestSpawnIdentities:
    """The phenotype forward over phase activations must reproduce
    spec.value() EXACTLY (affine folding; the σ(−(c+½)) = c²−¼
    identity for the dendrite)."""

    @pytest.mark.parametrize("spec", list(candidates(0, 1)))
    def test_forward_matches_closed_form(self, spec):
        sub, germ, ids = _organism()
        cid = spawn(sub, spec, (int(ids[0]), int(ids[1])),
                    parent=germ.progenitor_id)
        q = torch.randint(1, N_QUANTA, (64, 4)).float()
        # drive the receptor phases directly: x columns are consumed by
        # the per-sample quantizer, so inject via a controlled frame —
        # x = q/N_QUANTA with a fixed 0/1 span (col of 0 and col of 1
        # would distort; use raw phases through forward activations)
        sub.forward(q / N_QUANTA)
        a = sub.scheduler._last_activations
        # recover what the substrate actually saw (its own quantizer)
        q_seen = sub.scheduler._last_receptor_q
        c = q_seen / N_QUANTA - 0.5
        want = spec.value(c[:, 0], c[:, 1])
        got = a[:, cid]
        assert torch.allclose(got, want, atol=1e-5), spec

    def test_spawn_structural_contract(self):
        sub, germ, ids = _organism()
        c0 = census(sub.arena)
        spec = ComposerSpec("dendrite", 0, 1, None)
        cid = spawn(sub, spec, (int(ids[0]), int(ids[1])),
                    parent=germ.progenitor_id)
        c1 = census(sub.arena)
        assert c1.computing - c0.computing == 1      # +1 computing cell
        assert c1.edges - c0.edges == 2              # +2 edges
        assert c1.depth >= 1                         # depth is earned
        assert int(sub.arena.rank[cid]) >= 1
        assert has_gene(sub.arena.epigenome[cid], DENDRITE)
        assert not has_gene(sub.arena.epigenome[cid], LINEAR)
        assert int(sub.arena.parent[cid]) == germ.progenitor_id

    def test_tanh_gene_set(self):
        sub, germ, ids = _organism()
        cid = spawn(sub, ComposerSpec("tanh", 0, 1, (1.0, 1.0)),
                    (int(ids[0]), int(ids[1])))
        assert has_gene(sub.arena.epigenome[cid], TANH)

    def test_prune_hard_retires(self):
        sub, germ, ids = _organism()
        spec = ComposerSpec("linear", 0, 1, (1.0, -1.0))
        cid = spawn(sub, spec, (int(ids[0]), int(ids[1])))
        c1 = census(sub.arena)
        prune(sub, cid)
        c2 = census(sub.arena)
        assert c2.computing == c1.computing - 1      # out of the census
        ec = sub.arena.edge_cursor
        mask = sub.arena.edge_dst[:ec] == cid
        assert float(sub.arena.edge_weight[:ec][mask].abs().sum()) == 0.0
        # out of the forward path: activation stays zero
        sub.forward(torch.rand(8, 4))
        assert float(sub.scheduler._last_activations[:, cid].abs().sum()) == 0


class TestGeneVoteTransfer:
    def test_gene_settlement_targets_council_group(self):
        from trioron.pcll import FRUSTRATED, RESOLVED

        class Subj:
            gene = TANH

        sub, germ, _ = _organism()
        router = StressRouter(germ)
        total = sum(germ.votes.values())
        router.decide(FRUSTRATED, subject=Subj())
        router.settle(RESOLVED, testify=lambda s: True)
        votes_tanh = sum(germ.votes[c] for c in germ.council_ids[TANH])
        assert votes_tanh == pytest.approx(5.0)      # earned from the others
        assert sum(germ.votes[s] for s in germ.perception_seats) == \
            pytest.approx(4.0)                       # sensation untouched
        assert sum(germ.votes.values()) == pytest.approx(total)
        # failure bleeds the gene's group back
        router.decide(FRUSTRATED, subject=Subj())
        router.settle(RESOLVED, testify=lambda s: False)
        votes_tanh = sum(germ.votes[c] for c in germ.council_ids[TANH])
        assert votes_tanh == pytest.approx(4.0)
        assert sum(germ.votes.values()) == pytest.approx(total)


class TestTrialStatistic:
    def test_ratio_finds_relation_rejects_noise(self):
        g = torch.Generator().manual_seed(0)
        n = 240
        # two rings (radii .40/.22, the E/F pair) — a constant composed
        # dim cannot seat two 2-means children; the relation needs the
        # bimodal radius structure plus sensor noise, as in the world
        phi = 2 * math.pi * torch.rand(n, generator=g)
        r = torch.where(torch.rand(n, generator=g) > 0.5, 0.40, 0.22)
        eps = 0.02 * torch.randn(n, 2, generator=g)
        q = torch.stack([
            (0.5 + r * torch.cos(phi) + eps[:, 0]),
            (0.5 + r * torch.sin(phi) + eps[:, 1]),
            torch.rand(n, generator=g), torch.rand(n, generator=g),
        ], dim=1).clamp(0.001, 0.999) * N_QUANTA
        ring = ComposerSpec("dendrite", 0, 1, None)
        noise = ComposerSpec("dendrite", 2, 3, None)
        gg = torch.Generator().manual_seed(1)
        _, r_ring = trial_ratio(q, ring, gg)
        _, r_noise = trial_ratio(q, noise, gg)
        assert r_ring > 2.0 and r_noise <= 2.0
        # best_candidate over wired dims picks the ring form
        spec, _, _ = best_candidate(q, [0, 1, 2, 3], set(), seed=2)
        assert spec is not None and spec.gene == "dendrite"
        assert (spec.i, spec.j) == (0, 1)


class TestDivisionTries:
    def test_default_is_worst_dim_semantics(self):
        g = torch.Generator().manual_seed(3)
        n = 200
        # dim 0: clean bimodal (divisible); dim 1: uniform (worst R,
        # not divisible) — worst-dim semantics must return None
        mode = (torch.rand(n, generator=g) > 0.5).float()
        q0 = (0.25 + 0.5 * mode + 0.01 * torch.randn(n, generator=g))
        q1 = torch.rand(n, generator=g)
        Z = torch.exp(1j * 2 * math.pi * torch.stack([q0, q1], 1))
        assert try_divide(Z) is None
        got = try_divide(Z, tries=2)
        assert got is not None and got[1] == 0       # found on the 2nd try
