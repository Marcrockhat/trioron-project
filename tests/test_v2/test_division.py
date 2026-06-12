"""Division mechanics [D12] — fast CI guards.

Pins the split-vs-keep judgment: bimodal dims divide, unimodal and
uniform-noise dims do not, small modes wait for recurrence.
"""
from __future__ import annotations

import math

import torch

from trioron.pcll import circ_2means, try_divide


def _phasors(q: torch.Tensor) -> torch.Tensor:
    return torch.exp(1j * 2 * math.pi * q / 1000)


def _buf(n: int, centers, spread: float = 20.0, seed: int = 0):
    """[n, 2] phasor buffer: dim 0 mixes the given pocket centers,
    dim 1 is tight unimodal."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randint(0, len(centers), (n,), generator=g)
    c = torch.tensor([float(centers[i]) for i in k.tolist()])
    q0 = c + torch.randn(n, generator=g) * spread
    q1 = 500 + torch.randn(n, generator=g) * spread
    return _phasors(torch.stack([q0, q1], 1))


def test_bimodal_divides_on_the_blurred_dim():
    Z = _buf(200, centers=(200, 700))
    verdict = try_divide(Z)
    assert verdict is not None
    side, d = verdict
    assert d == 0
    assert min(int((~side).sum()), int(side.sum())) >= 25


def test_unimodal_keeps():
    assert try_divide(_buf(200, centers=(400,))) is None


def test_uniform_noise_keeps():
    g = torch.Generator().manual_seed(1)
    q = torch.rand(300, 2, generator=g) * 998 + 1
    assert try_divide(_phasors(q)) is None


def test_starved_buffer_waits():
    assert try_divide(_buf(40, centers=(200, 700))) is None


def test_rare_mode_waits_for_recurrence():
    # 190 vs 10 members: the small mode is below MIN_CHILD
    g = torch.Generator().manual_seed(2)
    q0 = torch.cat([200 + torch.randn(190, generator=g) * 15,
                    700 + torch.randn(10, generator=g) * 15])
    q1 = 500 + torch.randn(200, generator=g) * 15
    assert try_divide(_phasors(torch.stack([q0, q1], 1))) is None


def test_circ_2means_handles_wraparound():
    # two modes straddling the 0/1000 seam — circular stats must not split
    # the seam itself
    g = torch.Generator().manual_seed(3)
    q = torch.cat([torch.randn(100, generator=g) * 15 % 1000,
                   500 + torch.randn(100, generator=g) * 15])
    side = circ_2means(2 * math.pi * q / 1000)
    agree = (side[:100].float().mean() - side[100:].float().mean()).abs()
    assert agree > 0.9


# ── the merge consumer [D18] ──────────────────────────────────────

def test_merge_collapses_duplicates():
    from trioron.pcll.division import try_merge
    # two fragments of the SAME mode, split by membership accident
    g = torch.Generator().manual_seed(5)
    qa = torch.stack([300 + torch.randn(120, generator=g) * 12,
                      600 + torch.randn(120, generator=g) * 12], 1)
    qb = torch.stack([302 + torch.randn(110, generator=g) * 12,
                      598 + torch.randn(110, generator=g) * 12], 1)
    assert try_merge(_phasors2(qa), _phasors2(qb), tries=4)


def test_merge_refuses_distinct_modes():
    from trioron.pcll.division import try_merge
    g = torch.Generator().manual_seed(6)
    qa = torch.stack([250 + torch.randn(120, generator=g) * 12,
                      500 + torch.randn(120, generator=g) * 12], 1)
    qb = torch.stack([700 + torch.randn(120, generator=g) * 12,
                      500 + torch.randn(120, generator=g) * 12], 1)
    assert not try_merge(_phasors2(qa), _phasors2(qb), tries=4)


def test_merge_refuses_fresh_split_children():
    # the self-consistency guard: children of an accepted division have
    # a bimodal union (the parent) — try_merge must reject by its own law
    from trioron.pcll.division import try_merge
    g = torch.Generator().manual_seed(7)
    q = torch.stack([torch.cat([250 + torch.randn(100, generator=g) * 12,
                                700 + torch.randn(100, generator=g) * 12]),
                     500 + torch.randn(200, generator=g) * 12], 1)
    Z = _phasors2(q)
    verdict = try_divide(Z, tries=2)
    assert verdict is not None
    side, _ = verdict
    assert not try_merge(Z[~side], Z[side], tries=4)


def test_sketch_merge_is_exact():
    from trioron.core import construct
    from trioron.pcll import germline_base
    from trioron.pcll.manifold import PCLLManifold
    sub = construct(germline_base, capacity=16)
    m = PCLLManifold(sub.arena)
    g = torch.Generator().manual_seed(8)
    qa = 400 + 20 * torch.randn(130, 3, generator=g)
    qb = 420 + 25 * torch.randn(90, 3, generator=g)
    m.adopt("a", -1, 3)
    m.update("a", qa)
    m.adopt("b", -1, 3)
    m.update("b", qb)
    m.merge("a", "b")
    pooled = torch.cat([qa, qb])
    a = m.sketches["a"]
    assert "b" not in m.sketches and a._n == 220
    assert torch.allclose(a.mu, pooled.mean(0), atol=1e-3)
    assert torch.allclose(a.sigma, pooled.std(0), atol=1e-2)


def _phasors2(q):
    return torch.exp(1j * 2 * math.pi * q / 1000)
