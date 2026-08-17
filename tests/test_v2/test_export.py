"""Spec §5.4 dense export — a shipped substrate folds to a fixed forward.

  1. Linear seeded substrate: export == arena forward (fp32 ~1e-7).
  2. Nonlinear seeded (quad dendrite, K=2 via seeded(nonlinear=True)):
     export == arena forward, and the quad term is live (not linear).
  3. TANH interior: export == arena forward.
  4. Trained substrate (a few Adam steps) still exports exactly, and
     torch.jit.trace of the export matches too.
  5. Multi-layer seeded (interior_layers=2) exports exactly (chunked
     upstream weights, output gather path).
  6. Receptor substrates refuse (NotImplementedError) rather than
     silently mis-export.
"""
from __future__ import annotations

import pytest
import torch

from trioron.core import Envelope, construct
from trioron.core.epigenome import LINEAR, OUTPUT, PERCEPTION, TANH, has_gene
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from trioron.lifecycle import export_dense, verify_export


def _build(n_in=7, n_out=3, interior=6, layers=1, nonlinear=False, seed=0):
    torch.manual_seed(seed)
    sub = construct(base=seeded(n_in, n_out, interior_cells=interior,
                                interior_layers=layers, nonlinear=nonlinear),
                    envelope=Envelope(max_parameter_bytes=200_000),
                    dispatch_table=default_dispatch_table())
    sub.compile()
    return sub


def _randomize(sub, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        n = sub.arena.edge_cursor
        sub.arena.edge_weight[:n] = torch.randn(n, generator=g) * 0.5
        sub.arena.bias[:] = torch.randn(sub.arena.capacity, generator=g) * 0.1


def test_linear_export_exact():
    sub = _build()
    _randomize(sub)
    x = torch.randn(32, 7)
    assert verify_export(sub, export_dense(sub), x) < 1e-5


def test_nonlinear_export_exact_and_quad_live():
    sub = _build(nonlinear=True)
    _randomize(sub)
    x = torch.randn(32, 7)
    m = export_dense(sub)
    assert verify_export(sub, m, x) < 1e-5
    # quad is live: doubling the input must NOT double the interior response
    a = sub.arena
    assert bool((a.n_branches[a.alive_ids()] >= 2).any())
    y1, y2 = sub(x), sub(2 * x)
    assert not torch.allclose(y2 - a.bias[sub.scheduler._plan.output_ids.long()],
                              2 * (y1 - a.bias[sub.scheduler._plan.output_ids.long()]),
                              atol=1e-4)


def test_tanh_export_exact():
    sub = _build()
    a = sub.arena
    for cid in a.alive_ids().tolist():
        epi = int(a.epigenome[cid].item())
        if not has_gene(epi, PERCEPTION) and not has_gene(epi, OUTPUT):
            a.epigenome[cid] = (epi & ~(1 << LINEAR)) | (1 << TANH)
            a.refresh_phenotype(cid)
    sub.compile()
    _randomize(sub, seed=3)
    x = torch.randn(32, 7) * 3
    assert verify_export(sub, export_dense(sub), x) < 1e-5


def test_trained_export_exact_and_traceable():
    sub = _build(nonlinear=True, seed=2)
    sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=1e-2)
    x = torch.randn(64, 7)
    y = torch.randint(0, 3, (64,))
    for _ in range(10):
        loss = torch.nn.functional.cross_entropy(sub(x), y)
        opt.zero_grad(); loss.backward(); sub.zero_dormant_grads(); opt.step()
    m = export_dense(sub)
    with torch.no_grad():
        assert verify_export(sub, m, x) < 1e-5
        jm = torch.jit.freeze(torch.jit.trace(m, x[:1]))
        assert (jm(x) - sub(x)).abs().max() < 1e-5
        assert all(not b.requires_grad for b in m.buffers())


def test_two_layer_export_exact():
    sub = _build(interior=5, layers=2, nonlinear=True, seed=4)
    _randomize(sub, seed=5)
    x = torch.randn(16, 7)
    # quad through two layers reaches |y|~1e2; compare relative to scale
    scale = float(sub(x).abs().max())
    assert verify_export(sub, export_dense(sub), x) / scale < 1e-5


def test_receptor_substrate_refuses():
    from trioron.core.epigenome import RECEPTOR
    sub = _build()
    a = sub.arena
    cid = int(a.alive_ids()[0])
    a.epigenome[cid] = int(a.epigenome[cid].item()) | (1 << RECEPTOR)
    a.refresh_phenotype(cid)
    sub.compile()
    if sub.scheduler._plan.receptor_ids.numel() == 0:
        pytest.skip("receptor gene did not produce receptor cells in this build")
    with pytest.raises(NotImplementedError):
        export_dense(sub)
