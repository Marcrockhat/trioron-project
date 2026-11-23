"""trioron.organisms stable API (s062): Logophosis + Kinopsis on synthetic data."""
import torch
import pytest

from trioron.organisms import (Kinopsis, Logophosis, Organism, QDAVoter, Router, Std,
                               chunked, leaf, train_leaf, vote)


def _not_world(n, depth, seed, L=6, cap=8):
    """truth symbol (1|2) + m NOT tokens (3); label = truth XOR (m % 2)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(n, L, dtype=torch.long); y = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        t = int(torch.randint(1, 3, (1,), generator=g)); m = int(torch.randint(0, depth + 1, (1,), generator=g))
        ids[i, 0] = t; ids[i, 1:1 + m] = 3; y[i] = (t - 1) ^ (m % 2)
    return torch.zeros(n, 4), ids, y


@pytest.mark.parametrize("mode", ["seq", "tied"])
def test_logophosis_oracle_fit_and_roundtrip(tmp_path, mode):
    S, I, y = _not_world(600, 2, 0)
    St, It, yt = _not_world(400, 2, 1)
    org = Logophosis(4, 8, seed=0, mode=mode, link0="oracle", max_links=2).wake(I)
    d = org.fit(S, I, y, St, It, yt, epochs=3, patience=2, target=0.99, log=lambda s: None)
    assert 1 <= d <= 2
    p = org.comprehend(St, It)
    assert p.shape == (400, 2) and torch.allclose(p.sum(1), torch.ones(400), atol=1e-5)
    assert org.n_params() > 0 and isinstance(org, Organism)
    path = tmp_path / "lp.pt"; org.save(str(path))
    org2 = Logophosis.load(str(path))
    assert torch.equal(org2.comprehend(St, It).argmax(1), p.argmax(1))
    with pytest.raises(AssertionError):
        org.comprehend(St, It[:, :3])          # token length is fixed after the first encode


def test_logophosis_phasecyte_link0_evidence_shape():
    S, I, y = _not_world(600, 1, 0)
    org = Logophosis(4, 8, seed=0, mode="seq", link0="phasecyte",
                     link0_kw=dict(wake=400, window=200, verbose=False)).wake(I)
    ev = org.link0.evidence(I[:5])
    assert ev.shape == (5, 6, 8) and (ev[I[:5] == 0] == 0).all()
    assert org.encode(S[:5], I[:5]).shape == (5, 4 + 6 * 8)


def _fake_organ(seed=0):
    class Organ: pass
    o = Organ(); g = torch.Generator().manual_seed(seed)
    dims = {"shape": 12, "whole": 10, "fill": 9}; n_out = {"shape": 5, "whole": 5, "fill": 4}
    o.leaves, o.std = {}, {}
    for k, d in dims.items():
        Z = torch.randn(300, d, generator=g); yk = torch.randint(0, n_out[k], (300,), generator=g)
        o.std[k] = (Z.mean(0), Z.std(0) + 1e-6)
        o.leaves[k] = train_leaf(leaf(d, n_out[k], seed), (Z - o.std[k][0]) / o.std[k][1], yk, seed, 1)
    return o, g


def test_kinopsis_schedule_and_roundtrip(tmp_path):
    o, g = _fake_organ()
    org = Kinopsis.from_organ(o, torch.zeros(12), seed=0)
    n = 200
    S = {k: torch.randn(n, d, generator=g) for k, d in {"shape": 12, "whole": 10, "fill": 9, "mshape": 12}.items()}
    y = torch.randint(0, 5, (n,), generator=g)
    E = torch.randn(n, 11, generator=g); E[:, 9] = 1.0
    Ef = E.clone(); Ef[:, 9] = 0.0
    Y = {"y_shape": y, "y_fill": torch.randint(0, 4, (n,), generator=g)}
    old = {k: S[k] for k in ("shape", "whole", "fill")}
    assert org.shape(old, None, "uniform").shape == (n, 5)       # stage 1: msil silent
    org.settle_heads(S, Y, old, Y, n=100, steps=5).acquire_msil(S, y, epochs=1)
    org.fit_router(S, E, Ef, y, steps=5)
    Zm = torch.randn(n, 20, generator=g); org.acquire_vel(Zm, torch.randint(0, 17, (n,), generator=g), epochs=1)
    lp = org.shape(S, E); assert lp.shape == (n, 5) and torch.isfinite(lp).all()
    assert org.velocity(Zm).shape == (n, 17)
    w = org._w(org._feat(org.voters(old), Ef))
    assert (w[:, 2] == 0).all()                                  # hard gate: empty mask -> msil silent
    assert org.n_params() > 0 and len(org.substrates()) == 5 and org.torch_modules() == [org.router]
    path = tmp_path / "k.pt"; org.save(str(path)); org2 = Kinopsis.load(str(path))
    assert torch.equal(org2.shape(S, E).argmax(1), lp.argmax(1))
    with pytest.raises(TypeError):
        Logophosis.load(str(path))


def test_helpers():
    g = torch.Generator().manual_seed(0)
    H = torch.randn(60, 6, generator=g); y = torch.randint(0, 3, (60,), generator=g)
    assert QDAVoter(H, y)(H).shape == (60, 3)
    r = Router(4, 2, gate_col=3, motion_idx=1); e = torch.randn(5, 4); e[:, 3] = -1
    assert (r(e)[:, 1] == 0).all()
    assert vote([torch.randn(5, 3), torch.randn(5, 3)], torch.ones(5, 2) / 2).shape == (5, 3)
    s = Std(H); assert s(H).mean().abs() < 1e-5
    sub = leaf(6, 3, 0); assert chunked(sub, H, n=7).shape == (60, 3)


def test_vee_doubt_roundtrip(tmp_path):
    """s064: hidden-state sense + doubt leaf learn a planted signal, save/load exact."""
    from trioron.organisms import Vee
    from trioron.senses import HiddenStateSense
    torch.manual_seed(0)
    basis = torch.randn(6, 96)                      # states live near a 6-d subspace
    code = torch.randn(600, 6)
    H = code @ basis + 0.05 * torch.randn(600, 96)
    wrong = (code[:, 0] * code[:, 1] > 0).long()    # a product: needs the quad leaf
    sense = HiddenStateSense(H[:450], 8)
    assert sense(H).shape == (600, 16) and sense.out_dim == 16
    o = Vee(0, k=8).fit(H[:450], wrong[:450], epochs=150, lr=1e-2)
    d = o.doubt(H[450:])
    assert d.shape == (150,) and float(d.min()) >= 0 and float(d.max()) <= 1
    acc = ((d > 0.5).long() == wrong[450:]).float().mean().item()
    assert acc > 0.8, acc
    p = tmp_path / "syn.pt"
    assert o.save(str(p)) > 0
    o2 = Vee.load(str(p))
    assert torch.equal(o2.doubt(H[450:]), d)
    assert o.n_params() == sum(int(t.numel()) for t in o.doubt_leaf.trainable_tensors())
