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
