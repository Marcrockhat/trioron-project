"""Trioron 2.0 Phase 5 tests — R·S handshake migration primitives.

Covers the two localized primitives that prepare the absorption
pipeline for long-range edges and post-Phase-1.5 dendritic state:

  - TrioronLayer.reset_dendritic_state() — drops all Axis 5 state to
    K=1 / point-neuron form per spec §5.2 ("donor's dendritic state
    reset to single-branch point-neuron form on absorption").
  - TrioronLayer.standardized_column_mask() — bool mask for columns
    at sequential-default provenance, used by the R·S handshake to
    restrict cross-donor factorization to standardized columns.
  - L0Translator(column_filter=...) — accepts the bool mask above and
    restricts the M / c factorization to filtered columns.

Call-site integration (multibranch.py / api.absorb wiring) is left as
a forward-looking follow-up — no current donor has long-range edges,
so the 1.0 → 2.0 migration is non-breaking under sequential default.
"""

from __future__ import annotations

import torch

from trioron.node import TrioronLayer
from trioron.composition import L0Translator


# ---------- reset_dendritic_state ----------

def test_reset_dendritic_state_reverts_K2_to_K1():
    layer = TrioronLayer(fan_in=6, n_nodes=2, B_max=4)
    # Drift cell 0 to K=2 with custom branch state.
    layer.grow_branch(node_idx=0, source_cols=[2, 3, 4])
    layer.prune_branch(node_idx=0, branch_idx=1)
    # Drift utility / stress so we can confirm they reset.
    with torch.no_grad():
        layer.internal_stress.copy_(torch.tensor([0.5, 0.3]))
        layer.branch_utility[0, 0] = 0.8
        layer.fisher_branch_weight[0, 0] = 0.4

    layer.reset_dendritic_state()

    # Buffers back to construction defaults.
    assert (layer.branch_id == 0).all()
    expected_bw = torch.zeros(2, 4)
    expected_bw[:, 0] = 1.0
    assert torch.allclose(layer.branch_weight.data, expected_bw)
    assert torch.allclose(layer.branch_weight_anchor, expected_bw)
    assert (layer.fisher_branch_weight == 0).all()
    assert (layer.B_per_node == 1).all()
    assert (layer.internal_stress == 0).all()
    assert (layer.branch_utility == 0).all()
    assert not layer.dendrite_orphan.any()


def test_reset_dendritic_state_is_idempotent_on_K1_layer():
    """Calling on an already-K=1 layer is a no-op modulo EMA clearing."""
    layer = TrioronLayer(fan_in=4, n_nodes=2)
    # Drift only the EMA buffers (no structural change).
    with torch.no_grad():
        layer.internal_stress[0] = 0.2
        layer.branch_utility[0, 0] = 0.1
    layer.reset_dendritic_state()
    assert (layer.B_per_node == 1).all()
    assert (layer.internal_stress == 0).all()
    assert (layer.branch_utility == 0).all()
    expected_bw = torch.zeros(2, 8)
    expected_bw[:, 0] = 1.0
    assert torch.allclose(layer.branch_weight.data, expected_bw)


def test_reset_preserves_W_and_b():
    """Spec §5.2: 'R·S factorizes W's column space and does not depend
    on branch_id.' Dendrite reset must NOT touch W or b."""
    torch.manual_seed(127)
    layer = TrioronLayer(fan_in=4, n_nodes=2)
    W_before = layer.W.data.clone()
    b_before = layer.b.data.clone()
    layer.grow_branch(node_idx=0, source_cols=[1])
    layer.reset_dendritic_state()
    assert torch.equal(layer.W.data, W_before)
    assert torch.equal(layer.b.data, b_before)


def test_reset_forward_byte_identical_to_fresh_K1_layer():
    """After reset, forward must match a freshly-constructed K=1 layer
    with the same W / b (the K=1 fast path is the only one running)."""
    torch.manual_seed(131)
    layer = TrioronLayer(fan_in=4, n_nodes=2, activation="relu")
    layer.grow_branch(node_idx=0, source_cols=[1])
    W_snapshot = layer.W.data.clone()
    b_snapshot = layer.b.data.clone()

    layer.reset_dendritic_state()

    fresh = TrioronLayer(fan_in=4, n_nodes=2, activation="relu")
    with torch.no_grad():
        fresh.W.data.copy_(W_snapshot)
        fresh.b.data.copy_(b_snapshot)

    x = torch.randn(3, 4)
    assert torch.equal(layer(x), fresh(x))


# ---------- standardized_column_mask ----------

def test_default_columns_are_all_standardized():
    layer = TrioronLayer(fan_in=5, n_nodes=3)
    mask = layer.standardized_column_mask()
    assert mask.shape == (5,)
    assert mask.dtype == torch.bool
    assert mask.all()


def test_long_range_column_is_not_standardized():
    layer = TrioronLayer(fan_in=4, n_nodes=2)
    # Column 2 reads from layer 0 node 3 (long-range).
    with torch.no_grad():
        layer.input_sources[2] = torch.tensor([0, 3])
    mask = layer.standardized_column_mask()
    assert mask.tolist() == [True, True, False, True]


def test_grow_input_with_explicit_source_creates_unstandardized_column():
    layer = TrioronLayer(fan_in=3, n_nodes=2)
    layer.grow_input(source=(0, 1))   # explicit long-range source
    mask = layer.standardized_column_mask()
    assert mask.tolist() == [True, True, True, False]


# ---------- L0Translator column_filter ----------

def test_translator_default_unfiltered_path_unchanged():
    """No column_filter → behavior identical to pre-Phase-5."""
    torch.manual_seed(137)
    W_a = torch.randn(8, 12)
    W_b = torch.randn(8, 12)
    b_a = torch.randn(8)
    b_b = torch.randn(8)
    t_old = L0Translator(W_a, b_a, W_b, b_b)
    t_new = L0Translator(W_a, b_a, W_b, b_b, column_filter=None)
    assert torch.allclose(t_old.M, t_new.M)
    assert torch.allclose(t_old.c, t_new.c)
    assert t_new._column_filter is None


def test_translator_column_filter_restricts_factorization():
    torch.manual_seed(139)
    W_a = torch.randn(6, 10)
    W_b = torch.randn(6, 10)
    b_a = torch.randn(6)
    b_b = torch.randn(6)
    cf = torch.tensor(
        [True, True, True, True, True, False, False, False, True, True],
        dtype=torch.bool,
    )

    # Build the reference: pinv over filtered columns only.
    W_a_filt = W_a[:, cf]
    W_b_filt = W_b[:, cf]
    expected_M = W_b_filt.to(torch.float32) @ torch.linalg.pinv(
        W_a_filt.to(torch.float32)
    )
    expected_c = b_b.to(torch.float32) - expected_M @ b_a.to(torch.float32)

    t = L0Translator(W_a, b_a, W_b, b_b, column_filter=cf)
    assert torch.allclose(t.M, expected_M, atol=1e-5)
    assert torch.allclose(t.c, expected_c, atol=1e-5)
    assert torch.equal(t._column_filter, cf)


def test_translator_column_filter_rejects_wrong_shape():
    import pytest
    W = torch.randn(4, 6)
    b = torch.zeros(4)
    bad_cf = torch.tensor([True, False])  # wrong length
    with pytest.raises(ValueError, match="column_filter shape"):
        L0Translator(W, b, W, b, column_filter=bad_cf)


def test_translator_column_filter_rejects_empty_selection():
    import pytest
    W = torch.randn(4, 6)
    b = torch.zeros(4)
    cf = torch.zeros(6, dtype=torch.bool)
    with pytest.raises(ValueError, match="zero columns"):
        L0Translator(W, b, W, b, column_filter=cf)


def test_translator_filtered_handshake_recovers_identity_on_self():
    """Self-translation (W_a == W_b) restricted to standardized
    columns must still reduce to identity in the filtered subspace."""
    torch.manual_seed(149)
    W = torch.randn(8, 12)
    b = torch.randn(8)
    cf = torch.ones(12, dtype=torch.bool)
    cf[5:8] = False  # treat these as long-range
    t = L0Translator(W, b, W, b, column_filter=cf)
    # M = W @ W.pinv on the filtered columns. Self-translate any
    # pre-activation through M + c — should reproduce the input
    # (within fp32 floor) in the column subspace.
    pre = torch.randn(3, 8)
    out = t.translate(pre)
    assert torch.allclose(out, pre, atol=1e-4)


# ---------- end-to-end: layer mask feeds translator ----------

def test_layer_mask_feeds_translator_directly():
    """Common Phase 5 usage: take the AND of two donors' standardized
    column masks and pass it to L0Translator as the handshake filter."""
    torch.manual_seed(151)
    layer_a = TrioronLayer(fan_in=10, n_nodes=6)
    layer_b = TrioronLayer(fan_in=10, n_nodes=6)
    # Donor A has cols 7, 8 as long-range. Donor B has col 9.
    with torch.no_grad():
        layer_a.input_sources[7] = torch.tensor([0, 1])
        layer_a.input_sources[8] = torch.tensor([0, 2])
        layer_b.input_sources[9] = torch.tensor([0, 3])

    mask_a = layer_a.standardized_column_mask()
    mask_b = layer_b.standardized_column_mask()
    shared = mask_a & mask_b
    # Cols 7, 8, 9 are excluded; cols 0..6 standardized in both.
    assert shared.tolist() == [True] * 7 + [False, False, False]

    t = L0Translator(
        layer_a.W.data, layer_a.b.data,
        layer_b.W.data, layer_b.b.data,
        column_filter=shared,
    )
    assert t._column_filter is not None
    assert int(t._column_filter.sum().item()) == 7
