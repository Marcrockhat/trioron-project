"""Trioron 2.0 Phase 2.5 tests — dendritic growth & pruning events.

Covers the structural-plasticity layer built on top of Phase 1.5's
dendritic buffers:

  - grow_branch (split a cell's fan-in into a new dendritic branch)
  - prune_branch (retract a branch + orphan its columns + compact slots)
  - inherit_dendrite (sister-specialist seeding from a parent cell)
  - parent_idx kwarg on grow_node
  - dendrite_orphan masking in the forward path
  - update_internal_stress + update_branch_utility EMAs
  - select_parent helper
  - internal_frustration_candidates trigger helper

See trioron_2_0.md §3.5 for the design spec.
"""

from __future__ import annotations

import torch

from trioron.node import TrioronLayer


# ---------- dendrite_orphan buffer ----------

def test_default_dendrite_orphan_is_all_false():
    layer = TrioronLayer(fan_in=5, n_nodes=3)
    assert layer.dendrite_orphan.shape == (3, 5)
    assert layer.dendrite_orphan.dtype == torch.bool
    assert not layer.dendrite_orphan.any()


def test_grow_node_appends_unmarked_orphan_row():
    layer = TrioronLayer(fan_in=4, n_nodes=2)
    layer.grow_node()
    assert layer.dendrite_orphan.shape == (3, 4)
    assert not layer.dendrite_orphan[2].any()


def test_grow_input_appends_unmarked_orphan_col():
    layer = TrioronLayer(fan_in=3, n_nodes=2)
    layer.grow_input()
    assert layer.dendrite_orphan.shape == (2, 4)
    assert not layer.dendrite_orphan[:, -1].any()


# ---------- grow_branch ----------

def test_grow_branch_assigns_columns_and_increments_K():
    layer = TrioronLayer(fan_in=6, n_nodes=2, B_max=4)
    new_b = layer.grow_branch(node_idx=0, source_cols=[2, 3, 4])
    assert new_b == 1
    assert layer.B_per_node[0].item() == 2
    # Reassigned columns now sit on branch 1.
    assert (layer.branch_id[0, [2, 3, 4]] == 1).all()
    # Other columns stay on branch 0.
    assert (layer.branch_id[0, [0, 1, 5]] == 0).all()
    # Other cells untouched.
    assert (layer.branch_id[1] == 0).all()
    assert layer.B_per_node[1].item() == 1


def test_grow_branch_initializes_new_weight_as_dampened_mean():
    layer = TrioronLayer(fan_in=4, n_nodes=2, B_max=4)
    # Drift branch 0 so the mean isn't a degenerate 1.0.
    with torch.no_grad():
        layer.branch_weight.data[0, 0] = 0.8
    layer.grow_branch(node_idx=0, source_cols=[1, 2])
    # New branch weight = 0.1 · mean(active branches before grow) = 0.08.
    assert abs(layer.branch_weight.data[0, 1].item() - 0.08) < 1e-6
    # Anchor matches the new weight so EWC doesn't immediately tug.
    assert (
        layer.branch_weight_anchor[0, 1].item()
        == layer.branch_weight.data[0, 1].item()
    )
    # Fisher and utility start fresh.
    assert layer.fisher_branch_weight[0, 1].item() == 0.0
    assert layer.branch_utility[0, 1].item() == 0.0


def test_grow_branch_refuses_at_B_max():
    layer = TrioronLayer(fan_in=4, n_nodes=1, B_max=2)
    layer.grow_branch(node_idx=0, source_cols=[1])
    assert layer.B_per_node[0].item() == 2
    raised = False
    try:
        layer.grow_branch(node_idx=0, source_cols=[2])
    except ValueError:
        raised = True
    assert raised, "grow_branch must refuse when B_per_node == B_max"


def test_grow_branch_rejects_empty_or_duplicate_or_out_of_range():
    layer = TrioronLayer(fan_in=4, n_nodes=1, B_max=4)
    for bad_cols, exc in (
        ([], ValueError),
        ([1, 1], ValueError),
        ([5], IndexError),
        ([-1], IndexError),
    ):
        raised = None
        try:
            layer.grow_branch(node_idx=0, source_cols=bad_cols)
        except (ValueError, IndexError) as e:
            raised = type(e)
        assert raised is exc, f"source_cols={bad_cols} → expected {exc.__name__}, got {raised}"


def test_grow_branch_clears_orphan_flag_for_reassigned_cols():
    """If a column was previously orphaned (by an earlier prune_branch),
    reassigning it to a fresh branch un-orphans it for that cell."""
    layer = TrioronLayer(fan_in=4, n_nodes=1, B_max=4)
    with torch.no_grad():
        layer.dendrite_orphan[0, 2] = True
    layer.grow_branch(node_idx=0, source_cols=[2])
    assert not layer.dendrite_orphan[0, 2]


# ---------- prune_branch ----------

def test_prune_branch_refuses_at_K1():
    layer = TrioronLayer(fan_in=4, n_nodes=2)
    raised = False
    try:
        layer.prune_branch(node_idx=0, branch_idx=0)
    except ValueError:
        raised = True
    assert raised


def test_prune_branch_orphans_columns_for_this_cell_only():
    layer = TrioronLayer(fan_in=6, n_nodes=2, B_max=4)
    layer.grow_branch(node_idx=0, source_cols=[3, 4, 5])
    layer.prune_branch(node_idx=0, branch_idx=1)
    # Cell 0: cols 3, 4, 5 are now orphaned.
    assert layer.dendrite_orphan[0, [3, 4, 5]].all()
    # Cell 0: cols 0, 1, 2 still active.
    assert not layer.dendrite_orphan[0, [0, 1, 2]].any()
    # Cell 1: untouched.
    assert not layer.dendrite_orphan[1].any()
    # Branch count back to 1.
    assert layer.B_per_node[0].item() == 1


def test_prune_branch_compacts_higher_branches():
    """Pruning branch 1 of a K=3 cell must renumber branch 2 down to
    branch 1, and shift branch_weight column 2 → column 1."""
    layer = TrioronLayer(fan_in=8, n_nodes=1, B_max=4)
    layer.grow_branch(node_idx=0, source_cols=[2, 3])  # → branch 1
    layer.grow_branch(node_idx=0, source_cols=[4, 5])  # → branch 2
    assert layer.B_per_node[0].item() == 3
    # Mark branch 2's weight so we can detect it shifted into slot 1.
    with torch.no_grad():
        layer.branch_weight.data[0, 2] = 0.42

    layer.prune_branch(node_idx=0, branch_idx=1)

    assert layer.B_per_node[0].item() == 2
    # Branch 2's columns are now on branch 1 (renumbered down).
    assert (layer.branch_id[0, [4, 5]] == 1).all()
    # Branch 1's old columns are orphaned (their branch_id reset to 0,
    # but they're masked out via dendrite_orphan).
    assert layer.dendrite_orphan[0, [2, 3]].all()
    # Branch 2's weight migrated to slot 1.
    assert abs(layer.branch_weight.data[0, 1].item() - 0.42) < 1e-6
    # Vacated tail slot reset to defaults.
    assert layer.branch_weight.data[0, 2].item() == 0.0
    assert layer.branch_weight_anchor[0, 2].item() == 0.0
    assert layer.fisher_branch_weight[0, 2].item() == 0.0
    assert layer.branch_utility[0, 2].item() == 0.0


def test_prune_branch_rejects_out_of_range_branch_idx():
    layer = TrioronLayer(fan_in=4, n_nodes=1, B_max=4)
    layer.grow_branch(node_idx=0, source_cols=[1])  # K=2
    raised = False
    try:
        layer.prune_branch(node_idx=0, branch_idx=5)
    except IndexError:
        raised = True
    assert raised


# ---------- forward respects dendrite_orphan ----------

def test_forward_masks_orphaned_contribs():
    """A cell with an orphaned column must compute its branch sum
    excluding that column. Other cells reading the same column are
    unaffected."""
    torch.manual_seed(67)
    layer = TrioronLayer(
        fan_in=4, n_nodes=2, activation="linear",
        branch_activation="identity",
    )
    # Push cell 0 into K=2 with branch 1 carrying cols [2, 3].
    layer.grow_branch(node_idx=0, source_cols=[2, 3])
    # Prune branch 1 → cols [2, 3] orphaned for cell 0 only.
    layer.prune_branch(node_idx=0, branch_idx=1)
    assert layer.dendrite_orphan[0, [2, 3]].all()
    assert not layer.dendrite_orphan[1].any()

    # Force the dendritic path by lifting another cell to K=2 (so the
    # global K=1 fast path doesn't short-circuit).
    layer.grow_branch(node_idx=1, source_cols=[3])

    x = torch.randn(3, 4)
    y = layer(x)

    # Cell 0: branch 0 sums cols [0, 1] (cols [2, 3] orphaned).
    W = layer.W.data
    expected_cell0 = (
        W[0, 0] * x[:, 0] + W[0, 1] * x[:, 1]
    ) * layer.branch_weight.data[0, 0] + layer.b.data[0]
    assert torch.allclose(y[:, 0], expected_cell0, atol=1e-5)

    # Cell 1: no orphans, normal K=2 dendritic forward across all 4 cols.
    z0 = (W[1, [0, 1, 2]] * x[:, [0, 1, 2]]).sum(dim=1)
    z1 = (W[1, 3] * x[:, 3])
    expected_cell1 = (
        layer.branch_weight.data[1, 0] * z0
        + layer.branch_weight.data[1, 1] * z1
        + layer.b.data[1]
    )
    assert torch.allclose(y[:, 1], expected_cell1, atol=1e-5)


# ---------- inherit_dendrite ----------

def test_inherit_dendrite_copies_structure_at_K1_no_perturb():
    layer = TrioronLayer(fan_in=6, n_nodes=2, B_max=4)
    # K=1 parent → child is a faithful clone (perturbation no-ops at K=1).
    layer.inherit_dendrite(parent_idx=0, child_idx=1, perturb_frac=0.5)
    assert torch.equal(layer.branch_id[1], layer.branch_id[0])
    assert torch.equal(
        layer.branch_weight.data[1], layer.branch_weight.data[0]
    )
    assert layer.B_per_node[1].item() == layer.B_per_node[0].item()


def test_inherit_dendrite_copies_K2_then_perturbs():
    torch.manual_seed(73)
    layer = TrioronLayer(fan_in=20, n_nodes=2, B_max=4)
    layer.grow_branch(node_idx=0, source_cols=list(range(10, 20)))
    # Now cell 0 is K=2: cols 0..9 on branch 0, cols 10..19 on branch 1.
    assert layer.B_per_node[0].item() == 2

    layer.inherit_dendrite(parent_idx=0, child_idx=1, perturb_frac=0.10)
    # Child inherits K and the bulk of the partition.
    assert layer.B_per_node[1].item() == 2
    # At perturb_frac=0.10 over fan_in=20, ~2 columns get flipped to
    # the other branch — strictly: 1 ≤ flipped ≤ 20.
    diff = (layer.branch_id[1] != layer.branch_id[0]).sum().item()
    assert 1 <= diff <= layer.fan_in, (
        f"expected 1..{layer.fan_in} columns perturbed, got {diff}"
    )
    # Fisher and orphan reset on child even though the parent had them.
    assert (layer.fisher_branch_weight[1] == 0).all()
    assert not layer.dendrite_orphan[1].any()


def test_inherit_dendrite_rejects_self_and_out_of_range():
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    raised = False
    try:
        layer.inherit_dendrite(parent_idx=0, child_idx=0)
    except ValueError:
        raised = True
    assert raised
    raised = False
    try:
        layer.inherit_dendrite(parent_idx=5, child_idx=1)
    except IndexError:
        raised = True
    assert raised


# ---------- grow_node parent_idx ----------

def test_grow_node_with_parent_idx_seeds_child_from_parent():
    torch.manual_seed(79)
    layer = TrioronLayer(fan_in=10, n_nodes=2, B_max=4)
    layer.grow_branch(node_idx=0, source_cols=[5, 6, 7, 8, 9])
    # n_nodes = 2; new child will land at index 2.
    new_idx = layer.grow_node(parent_idx=0)
    assert new_idx == 2
    assert layer.B_per_node[2].item() == 2
    # The child inherited the partition (modulo ε perturbation).
    overlap = (layer.branch_id[2] == layer.branch_id[0]).sum().item()
    assert overlap >= layer.fan_in - 2, (
        f"child diverged too far from parent: overlap={overlap}/{layer.fan_in}"
    )


def test_grow_node_without_parent_idx_is_blank_slate_K1():
    layer = TrioronLayer(fan_in=8, n_nodes=2, B_max=4)
    layer.grow_branch(node_idx=0, source_cols=[4, 5, 6, 7])
    layer.grow_node()  # no parent_idx
    # New cell (idx 2) starts at K=1.
    assert layer.B_per_node[2].item() == 1
    assert (layer.branch_id[2] == 0).all()


# ---------- update_internal_stress ----------

def test_update_internal_stress_accumulates_relu_gated():
    torch.manual_seed(83)
    layer = TrioronLayer(fan_in=4, n_nodes=3, activation="relu")
    assert (layer.internal_stress == 0).all()
    x = torch.randn(8, 4, requires_grad=True)
    y = layer(x)
    # Asymmetric loss so each cell sees different upstream grad magnitude.
    (y * torch.tensor([1.0, 2.0, 0.0])).sum().backward()
    layer.update_internal_stress()
    # Cell 0 + cell 1 see nonzero upstream grad through their engaged
    # entries. Cell 2's upstream grad is 0 → no stress.
    assert layer.internal_stress[2].item() == 0.0
    # At least one of cells 0, 1 should have nonzero stress (depends on
    # whether any sample's y > 0; with random init this is very likely).
    assert (layer.internal_stress[:2] > 0).any()


def test_update_internal_stress_uses_eps_gate_for_nonrelu():
    torch.manual_seed(89)
    layer = TrioronLayer(fan_in=4, n_nodes=2, activation="tanh")
    x = torch.randn(8, 4) * 2.0  # large input → |tanh(y)| ≫ 0.05 likely
    x.requires_grad_()
    y = layer(x)
    y.sum().backward()
    layer.update_internal_stress()
    # tanh activations should mostly clear the ε_engage = 0.05 gate,
    # producing nonzero stress across most cells.
    assert (layer.internal_stress > 0).any()


def test_update_internal_stress_noop_before_backward():
    layer = TrioronLayer(fan_in=4, n_nodes=2)
    # No forward yet — should not crash.
    layer.update_internal_stress()
    assert (layer.internal_stress == 0).all()


# ---------- update_branch_utility ----------

def test_update_branch_utility_populates_K2_cells():
    torch.manual_seed(97)
    layer = TrioronLayer(
        fan_in=6, n_nodes=2, activation="linear",
        branch_activation="identity",
    )
    layer.grow_branch(node_idx=0, source_cols=[3, 4, 5])
    x = torch.randn(5, 6, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    layer.update_branch_utility()
    # Cell 0 K=2: both active branches have positive utility.
    assert layer.branch_utility[0, 0].item() > 0
    assert layer.branch_utility[0, 1].item() > 0
    # Cell 0 inactive branches (slots 2..) stay zero.
    assert layer.branch_utility[0, 2:].abs().sum().item() == 0.0


def test_update_branch_utility_noop_when_no_K2_forward():
    """All-K=1 networks take the fast path and never compute y_branches.
    update_branch_utility() must be a silent no-op in that case."""
    torch.manual_seed(101)
    layer = TrioronLayer(fan_in=4, n_nodes=2, activation="linear")
    x = torch.randn(3, 4, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert layer._last_y_branches is None
    layer.update_branch_utility()
    assert (layer.branch_utility == 0).all()


# ---------- select_parent ----------

def test_select_parent_picks_highest_mean_activation():
    torch.manual_seed(103)
    layer = TrioronLayer(fan_in=4, n_nodes=3, activation="relu")
    x = torch.randn(8, 4, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    expected = int(y.detach().mean(dim=0).argmax().item())
    assert layer.select_parent() == expected


def test_select_parent_raises_before_forward():
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    raised = False
    try:
        layer.select_parent()
    except RuntimeError:
        raised = True
    assert raised


# ---------- internal_frustration_candidates ----------

def test_internal_frustration_candidates_filters_by_threshold():
    layer = TrioronLayer(fan_in=4, n_nodes=3)
    # Hand-set internal_stress: cells 0, 2 above threshold, cell 1 below.
    with torch.no_grad():
        layer.internal_stress.copy_(torch.tensor([0.10, 0.01, 0.20]))
    # No saliency cache → saliency_utility() returns zeros → all candidates
    # pass the overall-stress ceiling.
    cands = layer.internal_frustration_candidates(
        threshold=0.05, overall_saliency_ceiling=1.0,
    )
    # Sorted by internal_stress descending.
    assert cands == [2, 0]


def test_internal_frustration_candidates_filters_by_saliency_ceiling():
    """A cell with high internal stress but loud overall contribution
    is NOT a within-niche candidate (population-level frustration owns
    those)."""
    torch.manual_seed(107)
    layer = TrioronLayer(fan_in=4, n_nodes=2, activation="relu")
    # Run a forward+backward so saliency_utility has data.
    x = torch.randn(8, 4, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    sal = layer.saliency_utility()

    with torch.no_grad():
        layer.internal_stress.copy_(torch.tensor([0.5, 0.5]))
    # Ceiling tight enough to exclude both → empty.
    tight = layer.internal_frustration_candidates(
        threshold=0.05,
        overall_saliency_ceiling=float(sal.min().item()) - 1e-6,
    )
    assert tight == []
    # Ceiling loose enough to include both → both in descending stress.
    loose = layer.internal_frustration_candidates(
        threshold=0.05,
        overall_saliency_ceiling=float(sal.max().item()) + 1.0,
    )
    assert sorted(loose) == [0, 1]


# ---------- structural integrity end-to-end ----------

def test_grow_then_prune_then_grow_round_trip():
    """A grow_branch / prune_branch / grow_branch cycle must leave the
    layer in a self-consistent state."""
    layer = TrioronLayer(fan_in=6, n_nodes=1, B_max=4)
    layer.grow_branch(node_idx=0, source_cols=[3, 4, 5])
    layer.prune_branch(node_idx=0, branch_idx=1)
    assert layer.B_per_node[0].item() == 1
    # Cols 3, 4, 5 are orphaned. Re-growing onto col 3 must un-orphan it.
    layer.grow_branch(node_idx=0, source_cols=[3])
    assert not layer.dendrite_orphan[0, 3]
    # Cols 4 and 5 remain orphaned (not reassigned).
    assert layer.dendrite_orphan[0, [4, 5]].all()
    assert layer.B_per_node[0].item() == 2


def test_state_dict_round_trip_preserves_phase_2_5_state():
    """After grow_branch / prune_branch / inherit_dendrite, a state_dict
    save/load must reproduce the full dendritic state including
    dendrite_orphan."""
    torch.manual_seed(109)
    src = TrioronLayer(fan_in=6, n_nodes=3, B_max=4)
    src.grow_branch(node_idx=0, source_cols=[2, 3])
    src.grow_branch(node_idx=0, source_cols=[4, 5])
    src.prune_branch(node_idx=0, branch_idx=1)
    src.inherit_dendrite(parent_idx=0, child_idx=2, perturb_frac=0.0)

    sd = src.state_dict()
    dst = TrioronLayer(fan_in=6, n_nodes=3, B_max=4)
    dst.load_state_dict(sd)

    assert torch.equal(dst.branch_id, src.branch_id)
    assert torch.equal(dst.branch_weight.data, src.branch_weight.data)
    assert torch.equal(dst.B_per_node, src.B_per_node)
    assert torch.equal(dst.dendrite_orphan, src.dendrite_orphan)
