"""Tests for trioron.growth_direction.from_activation_residuals and
TrioronNetwork.insert_layer's noise_scale kwarg.

Both were added in response to pneuma's 2026-05-20 field report:
identity-init insert_layer landed at identity and stayed there under
cosine-decaying LR + fresh Adam state, leaving a 1M-param layer doing
no useful work. noise_scale and the label-free residual-SVD helper
give callers two clean ways to escape that failure mode.
"""

from __future__ import annotations

import pytest
import torch

from trioron.growth_direction import from_activation_residuals
from trioron.network import TrioronNetwork


# ---------- from_activation_residuals ----------

def test_residual_svd_returns_correct_shape():
    torch.manual_seed(2)
    acts = torch.randn(64, 32)
    vecs = from_activation_residuals(acts, k=8)
    assert vecs.shape == (8, 32)


def test_residual_svd_rows_are_unit_norm():
    torch.manual_seed(3)
    acts = torch.randn(50, 24)
    vecs = from_activation_residuals(acts, k=5)
    norms = vecs.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_residual_svd_rows_are_orthogonal():
    """Right singular vectors of any matrix are pairwise orthogonal."""
    torch.manual_seed(5)
    acts = torch.randn(80, 16)
    vecs = from_activation_residuals(acts, k=4)
    gram = vecs @ vecs.T
    expected = torch.eye(4)
    assert torch.allclose(gram, expected, atol=1e-5)


def test_residual_svd_singular_values_descending():
    """Top-K right singular vectors should correspond to descending
    singular values. We verify by projecting the centered data onto
    each vector and checking the resulting variances are non-increasing."""
    torch.manual_seed(7)
    acts = torch.randn(120, 20) * torch.tensor([float(i + 1) for i in range(20)])
    vecs = from_activation_residuals(acts, k=10)
    centered = (acts - acts.mean(dim=0)).to(torch.float32)
    projections = centered @ vecs.T   # (batch, k)
    variances = projections.var(dim=0, unbiased=False)
    for i in range(len(variances) - 1):
        assert variances[i].item() >= variances[i + 1].item() - 1e-5, (
            f"variance not descending at index {i}: "
            f"{variances[i].item()} < {variances[i + 1].item()}"
        )


def test_residual_svd_is_deterministic_under_seed():
    """SVD is deterministic — same input must give bitwise-identical
    output. Critical for the live-growth callsite where reproducibility
    matters."""
    torch.manual_seed(11)
    acts = torch.randn(40, 16)
    v1 = from_activation_residuals(acts, k=4)
    v2 = from_activation_residuals(acts, k=4)
    assert torch.equal(v1, v2)


def test_residual_svd_rejects_non_2d_input():
    with pytest.raises(ValueError, match="must be 2D"):
        from_activation_residuals(torch.randn(10, 8, 4), k=2)


def test_residual_svd_rejects_invalid_k():
    with pytest.raises(ValueError, match="k must be >= 1"):
        from_activation_residuals(torch.randn(10, 8), k=0)


def test_residual_svd_rejects_k_above_rank_ceiling():
    """min(batch, feat_dim) is the rank ceiling — Vh has only that many
    rows. Requesting more must error cleanly."""
    acts = torch.randn(5, 32)   # rank ≤ 5 (batch < feat_dim)
    with pytest.raises(ValueError, match="yields only"):
        from_activation_residuals(acts, k=10)


def test_residual_svd_handles_constant_input():
    """A constant activation tensor has zero residual everywhere. SVD
    on a zero matrix is well-defined (all singular values 0); the
    returned vectors should still be unit-norm (they're an arbitrary
    orthonormal basis)."""
    acts = torch.ones(20, 8) * 3.5
    vecs = from_activation_residuals(acts, k=4)
    assert vecs.shape == (4, 8)
    norms = vecs.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_residual_svd_feeds_insert_layer_growth_direction():
    """End-to-end: gather activations at an insert point, compute
    label-free direction, plug into insert_layer. The post-insert
    forward should be a valid computation (just verifying the shape
    contract holds and the call doesn't raise)."""
    torch.manual_seed(13)
    net = TrioronNetwork([(4, 6, "relu"), (6, 3, "linear")])
    x = torch.randn(32, 4)
    with torch.no_grad():
        acts_at_layer_1 = net.layers[0](x)
    assert acts_at_layer_1.shape == (32, 6)
    vecs = from_activation_residuals(acts_at_layer_1, k=6)
    assert vecs.shape == (6, 6)
    net.insert_layer(
        between=(0, 1),
        n_nodes=6,
        activation="linear",
        init_mode="growth_direction",
        init_vecs=vecs,
    )
    assert len(net.layers) == 3


# ---------- insert_layer noise_scale ----------

def test_noise_scale_zero_is_byte_identical_to_pre_change():
    """Default noise_scale=0 must preserve the byte-exact identity-init
    contract that existed before this kwarg landed."""
    torch.manual_seed(17)
    net = TrioronNetwork([(4, 5, "relu"), (5, 3, "linear")])
    new_idx = net.insert_layer(between=(0, 1))   # default noise_scale=0
    eye = torch.eye(5, 5)
    assert torch.equal(net.layers[new_idx].W.data, eye)
    assert (net.layers[new_idx].b.data == 0).all()


def test_noise_scale_positive_perturbs_W_off_identity():
    torch.manual_seed(19)
    net = TrioronNetwork([(4, 5, "relu"), (5, 3, "linear")])
    new_idx = net.insert_layer(between=(0, 1), noise_scale=0.1)
    eye = torch.eye(5, 5)
    diff = (net.layers[new_idx].W.data - eye).abs()
    # noise_scale=0.1 with 5×5 randn → typical element ~0.1, mean ~0.08.
    assert diff.max().item() > 0.01, "noise_scale failed to perturb W"
    assert diff.mean().item() > 0.05


def test_noise_scale_anchor_matches_perturbed_W():
    """W_anchor must mirror the post-noise W (so EWC pulls toward the
    actual init, not the pre-noise identity)."""
    torch.manual_seed(23)
    net = TrioronNetwork([(4, 5, "relu"), (5, 3, "linear")])
    new_idx = net.insert_layer(between=(0, 1), noise_scale=0.05)
    layer = net.layers[new_idx]
    assert torch.allclose(layer.W.data, layer.W_anchor.float())


def test_noise_scale_rejects_negative():
    net = TrioronNetwork([(4, 5, "relu"), (5, 3, "linear")])
    with pytest.raises(ValueError, match="noise_scale must be >= 0"):
        net.insert_layer(between=(0, 1), noise_scale=-0.01)


def test_noise_scale_small_preserves_forward_approximately():
    """Small noise_scale (~1e-3) keeps post-insert forward within a
    small tolerance of pre-insert — the 'approximate function
    preservation' that motivates this kwarg over a clean random init."""
    torch.manual_seed(29)
    net = TrioronNetwork([(4, 5, "relu"), (5, 3, "linear")])
    x = torch.randn(8, 4)
    before = net(x).detach()
    net.insert_layer(
        between=(0, 1), activation="linear", noise_scale=1e-3,
    )
    after = net(x).detach()
    # 1e-3 noise → forward perturbation should be small but nonzero.
    diff = (before - after).abs().max().item()
    assert diff > 0, "noise didn't perturb forward at all (suspicious)"
    assert diff < 0.5, (
        f"noise_scale=1e-3 produced unexpectedly large forward "
        f"perturbation: max abs diff = {diff}"
    )


def test_noise_scale_ignored_with_growth_direction_init():
    """noise_scale is documented as 'ignored when init_mode='growth_direction''.
    Verify by passing both — the W should match init_vecs exactly,
    untouched by noise."""
    torch.manual_seed(31)
    net = TrioronNetwork([(4, 5, "relu"), (5, 3, "linear")])
    init_vecs = torch.randn(5, 5) * 0.1
    net.insert_layer(
        between=(0, 1),
        init_mode="growth_direction",
        init_vecs=init_vecs,
        noise_scale=0.5,           # would be huge if applied
    )
    assert torch.allclose(net.layers[1].W.data, init_vecs.float())


# ---------- isolation hygiene ----------

@pytest.fixture(autouse=True)
def _reset_profile():
    """Some tests in other files mutate the active profile; restore
    OPEN before each test here so noise_scale tests aren't subject to
    profile-induced surprises."""
    from trioron.profile import TrioronProfile, OPEN
    TrioronProfile.set_active(OPEN)
    yield
    TrioronProfile.set_active(OPEN)
