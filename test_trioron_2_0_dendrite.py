"""Trioron 2.0 Phase 1.5 tests — Axis 5 dendritic compartmentalization.

Covers the per-cell internal-tree extension added to TrioronLayer:

  - new buffers (branch_id, branch_weight, branch_weight_anchor,
    fisher_branch_weight, B_per_node, internal_stress, branch_utility)
  - K=1 byte-identical forward (the entire installed base)
  - K>1 dendritic forward (scatter_add per-branch sum → σ_branch → pool)
  - per-cell K=1 shortcut inside mixed-K populations
  - EWC penalty extension (branch_weight participates via fisher_branch_weight)
  - update_fisher accumulating fisher_branch_weight
  - anchor_weights snapshotting branch_weight_anchor
  - structural plasticity (grow_node, prune_node, grow_input, prune_input)
    keeping all dendrite buffers consistent
  - v1 state-dict back-compat (auto-flip branch_activation to 'identity'
    and inject Axis 5 buffer defaults when keys absent)

See trioron_2_0.md §3.5 and §5.1 for the design spec.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from trioron.node import TrioronLayer


# ---------- default state at construction ----------

def test_default_branch_id_all_zero():
    layer = TrioronLayer(fan_in=5, n_nodes=3)
    assert layer.branch_id.shape == (3, 5)
    assert layer.branch_id.dtype == torch.long
    assert (layer.branch_id == 0).all()


def test_default_branch_weight_is_one_zero_pattern():
    layer = TrioronLayer(fan_in=4, n_nodes=3, B_max=6)
    assert layer.branch_weight.shape == (3, 6)
    # branch 0 = 1.0, branches 1..B_max-1 = 0.0 per cell.
    expected = torch.zeros(3, 6)
    expected[:, 0] = 1.0
    assert torch.allclose(layer.branch_weight.data, expected)


def test_default_branch_weight_anchor_mirrors_branch_weight():
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    assert torch.allclose(layer.branch_weight_anchor, layer.branch_weight.data)


def test_default_b_per_node_is_one():
    layer = TrioronLayer(fan_in=4, n_nodes=5)
    assert layer.B_per_node.shape == (5,)
    assert (layer.B_per_node == 1).all()


def test_default_internal_stress_and_utility_zero():
    layer = TrioronLayer(fan_in=4, n_nodes=3, B_max=4)
    assert layer.internal_stress.shape == (3,)
    assert (layer.internal_stress == 0).all()
    assert layer.branch_utility.shape == (3, 4)
    assert (layer.branch_utility == 0).all()


def test_default_branch_activation_is_quad():
    layer = TrioronLayer(fan_in=4, n_nodes=2)
    assert layer.branch_activation == "quad"


def test_default_B_max_is_eight():
    layer = TrioronLayer(fan_in=4, n_nodes=2)
    assert layer.B_max == 8


def test_unknown_branch_activation_raises():
    raised = False
    try:
        TrioronLayer(fan_in=4, n_nodes=2, branch_activation="cubic")
    except ValueError:
        raised = True
    assert raised


def test_invalid_B_max_raises():
    raised = False
    try:
        TrioronLayer(fan_in=4, n_nodes=2, B_max=0)
    except ValueError:
        raised = True
    assert raised


# ---------- K=1 byte-identical forward ----------

def _reference_forward(layer: TrioronLayer, x: torch.Tensor) -> torch.Tensor:
    """Hand-rolled F.linear path matching what a point-neuron TrioronLayer
    should compute. Mirrors the K=1 fast path verbatim so we can assert
    bit-identical output across branch_activation settings.
    """
    if x.dtype != layer.W.dtype:
        x = x.to(layer.W.dtype)
    W_eff = layer.W * layer.routing_scale.unsqueeze(1).to(layer.W.dtype)
    z = F.linear(x, W_eff, layer.b)
    return {
        "relu": F.relu,
        "tanh": torch.tanh,
        "linear": lambda t: t,
    }[layer.activation](z)


def test_k1_forward_bit_identical_relu():
    torch.manual_seed(7)
    layer = TrioronLayer(fan_in=8, n_nodes=5, activation="relu")
    x = torch.randn(4, 8)
    y = layer(x)
    ref = _reference_forward(layer, x)
    assert torch.equal(y, ref), "K=1 forward must be bit-identical to F.linear path"


def test_k1_forward_bit_identical_tanh():
    torch.manual_seed(11)
    layer = TrioronLayer(fan_in=6, n_nodes=4, activation="tanh")
    x = torch.randn(3, 6)
    y = layer(x)
    ref = _reference_forward(layer, x)
    assert torch.equal(y, ref)


def test_k1_forward_bit_identical_linear():
    torch.manual_seed(13)
    layer = TrioronLayer(fan_in=6, n_nodes=4, activation="linear")
    x = torch.randn(3, 6)
    y = layer(x)
    ref = _reference_forward(layer, x)
    assert torch.equal(y, ref)


def test_k1_forward_bit_identical_across_branch_activations():
    """σ_branch must be fully bypassed at K=1 regardless of config.
    Two layers built with the same seed but different branch_activation
    settings must produce bit-identical forward output."""
    torch.manual_seed(17)
    layer_q = TrioronLayer(fan_in=6, n_nodes=4, activation="relu",
                           branch_activation="quad")
    torch.manual_seed(17)
    layer_id = TrioronLayer(fan_in=6, n_nodes=4, activation="relu",
                            branch_activation="identity")
    torch.manual_seed(17)
    layer_sig = TrioronLayer(fan_in=6, n_nodes=4, activation="relu",
                             branch_activation="sigmoid")
    x = torch.randn(5, 6)
    y_q = layer_q(x)
    y_id = layer_id(x)
    y_sig = layer_sig(x)
    assert torch.equal(y_q, y_id)
    assert torch.equal(y_q, y_sig)


def test_k1_fast_path_when_routing_scale_modulated():
    """routing_scale at non-default values must still take the fast path
    and match the F.linear reference exactly."""
    torch.manual_seed(19)
    layer = TrioronLayer(fan_in=5, n_nodes=4, activation="relu")
    layer.routing_scale.copy_(torch.tensor([1.0, 0.3, 0.0, 0.7]))
    x = torch.randn(3, 5)
    y = layer(x)
    ref = _reference_forward(layer, x)
    assert torch.equal(y, ref)


# ---------- K>1 dendritic forward ----------

def _set_k2_partition(
    layer: TrioronLayer,
    cell_idx: int,
    branch_0_cols: list[int],
    branch_1_cols: list[int],
    w0: float = 1.0,
    w1: float = 1.0,
) -> None:
    """Manually configure cell `cell_idx` for K=2 testing: assign columns
    to branches, set branch_weight, bump B_per_node. Used by tests to
    exercise the K>1 path without grow_branch (which lands in Phase 2.5).
    """
    with torch.no_grad():
        layer.branch_id[cell_idx].zero_()
        for j in branch_1_cols:
            layer.branch_id[cell_idx, j] = 1
        layer.B_per_node[cell_idx] = 2
        layer.branch_weight.data[cell_idx, 0] = w0
        layer.branch_weight.data[cell_idx, 1] = w1


def test_k2_dendritic_path_matches_handrolled_identity():
    """With branch_activation='identity' and a K=2 partition, the K>1 path
    is reducible to a flat weighted sum and must match a hand-rolled
    reference."""
    torch.manual_seed(23)
    layer = TrioronLayer(
        fan_in=6, n_nodes=3, activation="linear",
        branch_activation="identity",
    )
    # Cell 1 gets K=2; cells 0 and 2 stay at K=1.
    _set_k2_partition(
        layer, cell_idx=1,
        branch_0_cols=[0, 1, 2], branch_1_cols=[3, 4, 5],
        w0=0.5, w1=2.0,
    )
    x = torch.randn(4, 6)

    # Hand-rolled reference for cell 1 under identity σ_branch:
    #   z_0 = Σ_{j ∈ {0,1,2}} W[1, j] · x[batch, j]
    #   z_1 = Σ_{j ∈ {3,4,5}} W[1, j] · x[batch, j]
    #   soma_input = 0.5·z_0 + 2.0·z_1 + b[1]
    #   y[1] = σ_soma(soma_input)  (linear → identity)
    W = layer.W.data
    z0 = (W[1, :3] * x[:, :3]).sum(dim=1)
    z1 = (W[1, 3:] * x[:, 3:]).sum(dim=1)
    expected_cell1 = 0.5 * z0 + 2.0 * z1 + layer.b.data[1]

    # Cells 0 and 2 stay K=1 → F.linear-equivalent.
    expected_cell0 = (W[0] * x).sum(dim=1) + layer.b.data[0]
    expected_cell2 = (W[2] * x).sum(dim=1) + layer.b.data[2]

    y = layer(x)
    assert torch.allclose(y[:, 0], expected_cell0, atol=1e-6)
    assert torch.allclose(y[:, 1], expected_cell1, atol=1e-6)
    assert torch.allclose(y[:, 2], expected_cell2, atol=1e-6)


def test_k2_dendritic_path_matches_handrolled_quad():
    """K=2 under branch_activation='quad' (NMDA-style supralinear) must
    apply z² inside the soma pool."""
    torch.manual_seed(29)
    layer = TrioronLayer(
        fan_in=6, n_nodes=2, activation="linear",
        branch_activation="quad",
    )
    _set_k2_partition(
        layer, cell_idx=0,
        branch_0_cols=[0, 1, 2], branch_1_cols=[3, 4, 5],
        w0=1.0, w1=1.0,
    )
    x = torch.randn(3, 6)

    W = layer.W.data
    z0 = (W[0, :3] * x[:, :3]).sum(dim=1)
    z1 = (W[0, 3:] * x[:, 3:]).sum(dim=1)
    expected_cell0 = 1.0 * (z0 ** 2) + 1.0 * (z1 ** 2) + layer.b.data[0]

    y = layer(x)
    assert torch.allclose(y[:, 0], expected_cell0, atol=1e-6)


def test_per_cell_k1_shortcut_inside_mixed_population():
    """When at least one cell has K>1, the dendritic path runs for ALL
    cells. K=1 cells inside that pass must still produce the F.linear-
    equivalent answer (σ_branch bypassed via the per-cell shortcut)."""
    torch.manual_seed(31)
    layer = TrioronLayer(
        fan_in=4, n_nodes=4, activation="relu",
        branch_activation="quad",   # would square a K=1 cell's z if mis-applied
    )
    # Cell 0 alone gets K=2; cells 1, 2, 3 must stay point-neuron-equivalent.
    _set_k2_partition(
        layer, cell_idx=0,
        branch_0_cols=[0, 1], branch_1_cols=[2, 3],
        w0=1.0, w1=1.0,
    )
    x = torch.randn(2, 4)
    y_mixed = layer(x)

    # Reference for K=1 cells: F.linear path (no σ_branch).
    W = layer.W.data
    for i in (1, 2, 3):
        ref_i = F.relu((W[i] * x).sum(dim=1) + layer.b.data[i])
        assert torch.allclose(y_mixed[:, i], ref_i, atol=1e-6), (
            f"K=1 cell {i} drifted under mixed-K forward (σ_branch=quad "
            f"leaked into K=1 path)"
        )


def test_k2_forward_gradients_reach_branch_weight():
    """Under the K>1 path, branch_weight must be on the gradient graph.
    K=1 cells stay off the gradient (their branch_weight entries don't
    enter the pool since branch_weight[i, 1..] = 0 and branch_id[i, :] = 0
    means z_branches[i, 1..] = 0 too)."""
    torch.manual_seed(37)
    layer = TrioronLayer(
        fan_in=4, n_nodes=2, activation="linear",
        branch_activation="identity",
    )
    _set_k2_partition(
        layer, cell_idx=0,
        branch_0_cols=[0, 1], branch_1_cols=[2, 3],
        w0=0.5, w1=1.5,
    )
    x = torch.randn(3, 4, requires_grad=False)
    y = layer(x)
    y.sum().backward()
    assert layer.branch_weight.grad is not None
    # Active K=2 cell: both branches receive gradient (z_branches[0, 0]
    # and z_branches[0, 1] are both nonzero in expectation).
    assert layer.branch_weight.grad[0, 0].abs().item() > 0
    assert layer.branch_weight.grad[0, 1].abs().item() > 0


# ---------- update_fisher accumulates fisher_branch_weight ----------

def test_update_fisher_accumulates_branch_weight_fisher():
    torch.manual_seed(41)
    layer = TrioronLayer(
        fan_in=4, n_nodes=2, activation="linear",
        branch_activation="identity",
    )
    _set_k2_partition(
        layer, cell_idx=0,
        branch_0_cols=[0, 1], branch_1_cols=[2, 3],
        w0=0.5, w1=1.5,
    )
    assert (layer.fisher_branch_weight == 0).all()
    x = torch.randn(5, 4)
    y = layer(x)
    y.sum().backward()
    layer.update_fisher()
    # Cell 0 (K=2): both active branches carry Fisher mass.
    assert layer.fisher_branch_weight[0, 0].item() > 0
    assert layer.fisher_branch_weight[0, 1].item() > 0
    # Cell 1 (K=1, but routed through the dendritic path because the
    # layer's max-K is 2): branch 0 is the only entry that enters the
    # soma pool with a nonzero coefficient (bw[1, 0] = 1, bw[1, 1..] = 0),
    # so gradient flows into branch 0 and Fisher accumulates there.
    # Branches 1..B_max-1 stay zero because their pool entries are zero
    # and their per-branch sums z_branches[1, b>0] are zero (nothing
    # scattered into them).
    assert layer.fisher_branch_weight[1, 0].item() > 0
    assert layer.fisher_branch_weight[1, 1:].abs().sum().item() == 0.0


# ---------- ewc_penalty includes branch_weight term ----------

def test_ewc_penalty_includes_branch_weight_term():
    torch.manual_seed(43)
    layer = TrioronLayer(fan_in=4, n_nodes=2, activation="linear")
    # Baseline: λ=0, no drift → penalty zero across all terms.
    assert layer.ewc_penalty().item() == 0.0

    # Set up the branch_weight term: nonzero λ, nonzero fisher_branch_weight,
    # branch_weight drifted away from anchor.
    layer.lam.fill_(1.0)
    layer.fisher_branch_weight.fill_(2.0)
    with torch.no_grad():
        layer.branch_weight.data[0, 0] += 0.5
    # pen_W and pen_b stay 0 (W and b are unchanged from anchor).
    # pen_bw = stiffness · fisher_bw · (Δ)² summed:
    #       = 1.0 · 2.0 · (0.5)² = 0.5 for cell 0 branch 0, plus zeros.
    pen = layer.ewc_penalty()
    assert abs(pen.item() - 0.5) < 1e-6


def test_ewc_penalty_branch_weight_term_scales_with_fisher():
    """Changing fisher_branch_weight changes the penalty proportionally —
    confirms fisher_branch_weight enters the term directly (not via λ)."""
    torch.manual_seed(47)
    layer = TrioronLayer(fan_in=4, n_nodes=2, activation="linear")
    layer.lam.fill_(1.0)
    with torch.no_grad():
        layer.branch_weight.data[0, 0] += 1.0
    layer.fisher_branch_weight.fill_(1.0)
    pen_small = layer.ewc_penalty().item()
    layer.fisher_branch_weight.fill_(10.0)
    pen_large = layer.ewc_penalty().item()
    assert abs(pen_large - 10.0 * pen_small) < 1e-6


# ---------- anchor_weights snapshots branch_weight ----------

def test_anchor_weights_snapshots_branch_weight():
    torch.manual_seed(53)
    layer = TrioronLayer(fan_in=4, n_nodes=2)
    with torch.no_grad():
        layer.branch_weight.data[0, 0] = 0.42
        layer.branch_weight.data[1, 0] = 0.99
    layer.anchor_weights()
    assert torch.equal(layer.branch_weight_anchor, layer.branch_weight.data)


# ---------- grow_node / prune_node / grow_input / prune_input ----------

def test_grow_node_extends_dendrite_buffers():
    layer = TrioronLayer(fan_in=4, n_nodes=2, B_max=5)
    new_idx = layer.grow_node()
    assert new_idx == 2
    assert layer.branch_id.shape == (3, 4)
    assert (layer.branch_id[2] == 0).all()
    assert layer.branch_weight.shape == (3, 5)
    expected_new_row = torch.zeros(5)
    expected_new_row[0] = 1.0
    assert torch.allclose(layer.branch_weight.data[2], expected_new_row)
    assert layer.branch_weight_anchor.shape == (3, 5)
    assert torch.allclose(layer.branch_weight_anchor[2], expected_new_row)
    assert layer.fisher_branch_weight.shape == (3, 5)
    assert (layer.fisher_branch_weight[2] == 0).all()
    assert layer.B_per_node.shape == (3,)
    assert layer.B_per_node[2].item() == 1
    assert layer.internal_stress.shape == (3,)
    assert layer.internal_stress[2].item() == 0.0
    assert layer.branch_utility.shape == (3, 5)
    assert (layer.branch_utility[2] == 0).all()


def test_grow_node_preserves_forward_byte_identity_at_k1():
    torch.manual_seed(59)
    layer = TrioronLayer(fan_in=4, n_nodes=2, activation="relu")
    x = torch.randn(3, 4)
    before = layer(x).detach().clone()
    layer.grow_node()
    # Existing cells' output rows must be unchanged (new row added at the
    # tail). The new cell's row is at index 2; we ignore it.
    after = layer(x).detach()
    assert after.shape == (3, 3)
    assert torch.equal(after[:, :2], before)


def test_prune_node_drops_dendrite_rows():
    layer = TrioronLayer(fan_in=4, n_nodes=3, B_max=4)
    # Drift dendrite state on cell 1 so we can detect it's actually gone.
    with torch.no_grad():
        layer.branch_id[1, 0] = 1
        layer.branch_weight.data[1, 1] = 0.7
        layer.B_per_node[1] = 2
        layer.internal_stress[1] = 0.3
        layer.branch_utility[1, 1] = 0.5
    layer.prune_node(1)
    assert layer.branch_id.shape == (2, 4)
    assert layer.branch_weight.shape == (2, 4)
    assert layer.branch_weight_anchor.shape == (2, 4)
    assert layer.fisher_branch_weight.shape == (2, 4)
    assert layer.B_per_node.shape == (2,)
    assert layer.internal_stress.shape == (2,)
    assert layer.branch_utility.shape == (2, 4)
    # Surviving cells are 0 and the original 2. The drifted cell 1 is gone.
    assert (layer.branch_id == 0).all()
    assert layer.B_per_node[0].item() == 1
    assert layer.B_per_node[1].item() == 1


def test_grow_input_adds_branch_id_column():
    layer = TrioronLayer(fan_in=3, n_nodes=2)
    layer.grow_input()
    assert layer.branch_id.shape == (2, 4)
    # New column defaults to branch 0.
    assert (layer.branch_id[:, -1] == 0).all()


def test_prune_input_drops_branch_id_column():
    layer = TrioronLayer(fan_in=4, n_nodes=2)
    # Mark column 2 as branch 1 so we can confirm the right column gets dropped.
    with torch.no_grad():
        layer.branch_id[:, 2] = 1
    layer.prune_input(2)
    assert layer.branch_id.shape == (2, 3)
    # The remaining branch_id should be all-zero (we removed the only
    # nonzero column).
    assert (layer.branch_id == 0).all()


# ---------- v1 state-dict back-compat ----------

def test_v1_state_dict_load_flips_branch_activation_to_identity():
    """A state_dict missing the branch_id key (i.e., pre-2.0 / pre-Axis-5)
    must auto-flip the destination layer's branch_activation to 'identity'
    so the absorbed substrate stays point-neuron-equivalent under future
    grow_branch calls."""
    src = TrioronLayer(fan_in=4, n_nodes=3, activation="relu")
    sd = src.state_dict()
    # Strip every Axis 5 key to simulate a v1-era checkpoint.
    for k in (
        "branch_id", "branch_weight", "branch_weight_anchor",
        "fisher_branch_weight", "B_per_node", "internal_stress",
        "branch_utility",
    ):
        sd.pop(k, None)
    # Also strip Phase 1 (Axis 1/2/4) keys so the v1 contract is honest.
    for k in (
        "input_sources", "input_archived", "axonal_gain", "axonal_gain_anchor",
    ):
        sd.pop(k, None)

    dst = TrioronLayer(fan_in=4, n_nodes=3, activation="relu",
                       branch_activation="quad")
    assert dst.branch_activation == "quad"
    dst.load_state_dict(sd)
    assert dst.branch_activation == "identity"


def test_v1_state_dict_load_injects_axis5_defaults():
    src = TrioronLayer(fan_in=4, n_nodes=3, activation="relu")
    sd = src.state_dict()
    for k in (
        "branch_id", "branch_weight", "branch_weight_anchor",
        "fisher_branch_weight", "B_per_node", "internal_stress",
        "branch_utility",
        "input_sources", "input_archived", "axonal_gain", "axonal_gain_anchor",
    ):
        sd.pop(k, None)

    dst = TrioronLayer(fan_in=4, n_nodes=3, activation="relu", B_max=8)
    dst.load_state_dict(sd)
    # Axis 5 buffers / param should hold their construction defaults.
    assert (dst.branch_id == 0).all()
    assert (dst.B_per_node == 1).all()
    expected_bw = torch.zeros(3, 8)
    expected_bw[:, 0] = 1.0
    assert torch.allclose(dst.branch_weight.data, expected_bw)
    assert torch.allclose(dst.branch_weight_anchor, expected_bw)
    assert (dst.fisher_branch_weight == 0).all()
    assert (dst.internal_stress == 0).all()
    assert (dst.branch_utility == 0).all()


def test_v1_state_dict_load_forward_is_bit_identical():
    """After a v1 load, the destination layer must reproduce the source
    layer's forward exactly (K=1 fast path on both sides)."""
    torch.manual_seed(61)
    src = TrioronLayer(fan_in=5, n_nodes=4, activation="relu")
    sd = src.state_dict()
    for k in (
        "branch_id", "branch_weight", "branch_weight_anchor",
        "fisher_branch_weight", "B_per_node", "internal_stress",
        "branch_utility",
        "input_sources", "input_archived", "axonal_gain", "axonal_gain_anchor",
    ):
        sd.pop(k, None)

    dst = TrioronLayer(fan_in=5, n_nodes=4, activation="relu")
    dst.load_state_dict(sd)
    x = torch.randn(3, 5)
    assert torch.equal(src(x), dst(x))


def test_round_trip_2_0_state_dict_preserves_axis5():
    """A current-version state_dict must round-trip without invoking the
    v1 fallback. branch_activation stays as constructed."""
    src = TrioronLayer(fan_in=4, n_nodes=3, branch_activation="sigmoid")
    # Drift some Axis 5 state so the round-trip is meaningful.
    with torch.no_grad():
        src.branch_id[1, 2] = 1
        src.branch_weight.data[1, 1] = 0.4
        src.B_per_node[1] = 2
        src.internal_stress[0] = 0.25
        src.branch_utility[1, 1] = 0.6
    sd = src.state_dict()

    dst = TrioronLayer(fan_in=4, n_nodes=3, branch_activation="sigmoid")
    dst.load_state_dict(sd)
    assert dst.branch_activation == "sigmoid"
    assert torch.equal(dst.branch_id, src.branch_id)
    assert torch.equal(dst.branch_weight.data, src.branch_weight.data)
    assert torch.equal(dst.B_per_node, src.B_per_node)
    assert torch.equal(dst.internal_stress, src.internal_stress)
    assert torch.equal(dst.branch_utility, src.branch_utility)
