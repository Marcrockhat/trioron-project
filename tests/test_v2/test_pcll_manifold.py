"""Manifold adapter unit tests [D15] — sketches over pocket space."""
from __future__ import annotations

import pytest
import torch

from trioron.core import construct
from trioron.pcll import germline_base
from trioron.pcll.manifold import (
    ANNEAL_N,
    MIN_SKETCH,
    PCLLManifold,
    SIGMA_FLOOR,
)


def _adapter():
    sub = construct(germline_base, capacity=32)
    return PCLLManifold(sub.arena)


class TestSketches:
    def test_update_and_moments(self):
        m = _adapter()
        torch.manual_seed(0)
        q = 500 + 30 * torch.randn(200, 6)
        m.adopt("u1", -1, 6)
        m.update("u1", q)
        astro = m.sketches["u1"]
        assert torch.allclose(astro.mu, q.mean(0), atol=1e-3)
        assert torch.allclose(astro.sigma, q.std(0), atol=1e-2)

    def test_log_likelihood_prefers_own_class(self):
        m = _adapter()
        torch.manual_seed(1)
        qa = 300 + 10 * torch.randn(120, 4)
        qb = 700 + 10 * torch.randn(120, 4)
        m.adopt("a", -1, 4)
        m.update("a", qa)
        m.adopt("b", -1, 4)
        m.update("b", qb)
        ll = m.log_likelihood(torch.cat([qa[:20], qb[:20]]), ["a", "b"])
        pred = ll.argmax(1)
        assert (pred[:20] == 0).all() and (pred[20:] == 1).all()
        # σ floor: a near-constant dim must not explode the likelihood
        m.adopt("c", -1, 4)
        m.update("c", torch.full((MIN_SKETCH + 5, 4), 400.0))
        sigma = m.sketches["c"].sigma.clamp(min=SIGMA_FLOOR)
        assert (sigma >= SIGMA_FLOOR).all()

    def test_below_floor_is_unconsumable(self):
        m = _adapter()
        m.adopt("u1", -1, 4)
        m.update("u1", torch.rand(MIN_SKETCH - 10, 4) * 1000)
        ll = m.log_likelihood(torch.rand(5, 4) * 1000, ["u1"])
        assert torch.isinf(ll).all()
        assert m.anneal_phasors("u1") is None
        assert m.replay() == []

    def test_anneal_phasors_shape_and_range(self):
        m = _adapter()
        torch.manual_seed(2)
        m.adopt("u1", -1, 5)
        m.update("u1", 500 + 20 * torch.randn(MIN_SKETCH + 40, 5))
        z = m.anneal_phasors("u1")
        assert z is not None and z.shape == (ANNEAL_N, 5)
        assert torch.allclose(z.abs(), torch.ones_like(z.abs()), atol=1e-5)

    def test_state_dict_round_trip(self):
        m = _adapter()
        torch.manual_seed(3)
        q = 400 + 25 * torch.randn(150, 3)
        m.adopt("u7", 11, 3)
        m.update("u7", q)
        state = m.state_dict()
        m2 = _adapter()
        m2.load_state_dict(state)
        a, b = m.sketches["u7"], m2.sketches["u7"]
        assert a._n == b._n and b.cell_id == 11
        assert torch.equal(a.mu, b.mu) and torch.equal(a.sigma, b.sigma)
        x = torch.rand(8, 3) * 1000
        assert torch.equal(m.log_likelihood(x, ["u7"]),
                           m2.log_likelihood(x, ["u7"]))

    def test_division_rebuild_replaces_parent(self):
        m = _adapter()
        torch.manual_seed(4)
        m.adopt("u1", -1, 4)
        m.update("u1", torch.rand(100, 4) * 1000)
        kid_q = 200 + 5 * torch.randn(80, 4)
        m.retire("u1")
        m.rebuild("u2", -1, kid_q)
        assert "u1" not in m.sketches and "u2" in m.sketches
        assert torch.allclose(m.sketches["u2"].mu, kid_q.mean(0), atol=1e-3)
