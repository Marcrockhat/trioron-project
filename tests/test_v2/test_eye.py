"""Spec §10.11 — the eye (retina-pooled, foveated receptor body).

  1. Layout: M-stream regions tile the 32×32 sheet exactly once; P-stream
     tiles the 16×16 fovea+parafovea window once; ring counts as designed
     (fovea 64 px, parafovea 48 2×2, periphery 48 4×4; M 16/12/48).
  2. DoG: a flat image gives ~zero features (centre == surround); a
     bright dot at the fovea centre lights the ON receptor of the pixel
     region containing it and the OFF receptor of its neighbours.
  3. Off-centre fixations clip to the visual field (fewer M regions), and
     ``positions`` has one row per feature.
  4. Determinism without jitter; jitter changes the offset only within
     ±j and keeps the feature width.
  5. sense() width == n_features(P) + n_features(M).
"""
from __future__ import annotations

from collections import Counter

import torch

from trioron.pcll.eye import Eye, fixations, retina_layout


def test_layout_tiles_once():
    e = Eye((16, 16))
    for stream, want in (("M", {"fovea": 16, "parafovea": 12, "periphery": 48}),
                         ("P", {"fovea": 64, "parafovea": 48})):
        regs = e.regions(stream)
        assert Counter(r.ring for r in regs) == want
        cov = torch.zeros(32 * 32)
        for r in regs:
            cov[r.members] += 1
        assert cov.max() == 1
        assert int((cov > 0).sum()) == (1024 if stream == "M" else 256)
    assert e.n_features("P") == 2 * 3 * 112
    assert e.n_features("M") == 2 * 1 * 76


def test_dog_flat_zero_and_dot_on_off():
    e = Eye((16, 16))
    flat = torch.full((1, 3, 32, 32), 0.5)
    assert e.sense(flat).abs().max() < 1e-2
    img = torch.zeros(1, 3, 32, 32)
    img[0, :, 16, 16] = 1.0                       # white dot (Y only)
    fP = e.sense_P(img)[0]
    regs = e.regions("P")
    n = len(regs)
    idx = next(i for i, r in enumerate(regs) if r.members == [16 * 32 + 16])
    on_Y = fP[:n]                                  # ON, channel Y, region-major
    off_Y = fP[3 * n: 4 * n]
    assert on_Y[idx] > 0 and on_Y[idx] == on_Y.max()
    nb = next(i for i, r in enumerate(regs) if r.members == [16 * 32 + 17])
    assert off_Y[nb] > 0 and on_Y[nb] == 0


def test_offcentre_clips_and_positions_match():
    e = Eye((10, 10))
    assert e.n_features("M") < Eye((16, 16)).n_features("M")
    for s in ("P", "M"):
        assert e.positions(s).shape == (e.n_features(s), 3)
    assert len(fixations(5)) == 5 and len(fixations(9)) == 9


def test_deterministic_and_jitter():
    x = torch.rand(3, 3, 32, 32)
    e = Eye((16, 16))
    assert torch.equal(e.sense(x), e.sense(x))
    g = torch.Generator().manual_seed(0)
    ej = Eye((16, 16), jitter=1, generator=g)
    outs = [ej.sense(x) for _ in range(6)]
    assert all(o.shape == outs[0].shape for o in outs)
    assert any(not torch.equal(outs[0], o) for o in outs[1:])
    assert ej.sense(x).shape[1] == ej.n_features("P") + ej.n_features("M")
