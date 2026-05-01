"""Self-contained tests for trioron.dreaming.

Run with:    python3 test_dreaming.py

Verifies the dreaming-block contract for Phase 4.5:
  - find_redundant_pairs returns pairs above threshold (W_anchor cosine).
  - merge_nodes preserves the linear pre-activation when w_i == w_j.
  - merge averages incoming/anchor/lambda/fisher; sums outgoing on next layer.
  - merge takes max(u_peer, u_victim).
  - compress reduces n_nodes when planted duplicates exist; no-op otherwise.
  - compress skips the output layer by default.
  - replay decreases loss on revisited pairs.
  - replay tolerates empty past_pair_names.
  - purge drops only nodes below u_threshold; never the last node in a layer.
  - dreaming_block end-to-end: replay, merge, purge interact cleanly,
    optimizer rebuild leaves the net trainable.
"""
from __future__ import annotations
import math
import random
import sys
import traceback

import torch
import torch.nn.functional as F
import torch.optim as optim

from trioron.network import TrioronNetwork
from trioron.dreaming import (
    DreamingReport,
    MergeEvent,
    PurgeEvent,
    _pick_starvation_victim,
    apoptosis_decay,
    apoptosis_redistribute,
    apoptosis_spike,
    compress,
    dreaming_block,
    find_activation_redundant_pairs,
    find_redundant_pairs,
    max_off_diag_activation_cosine,
    max_off_diag_cosine,
    merge_nodes,
    purge,
    replay,
    routing_regrow,
    routing_starve,
    synaptic_downscale,
)


_RESULTS: list[tuple[str, bool, str]] = []


def _run(name: str, fn) -> None:
    try:
        fn()
        _RESULTS.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as e:
        _RESULTS.append((name, False, str(e)))
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc(limit=3)


def _make_net(hidden=4, fan_in=6, latent=2):
    """Three-layer net: fan_in → hidden → hidden → latent."""
    return TrioronNetwork([
        (fan_in, hidden, "relu"),
        (hidden, hidden, "relu"),
        (hidden, latent, "tanh"),
    ])


def _plant_duplicate(net, layer_idx, src=0, dst=1):
    """Copy node `src`'s state into node `dst` on layer_idx so a redundant
    pair is guaranteed. Anchors duplicated as well, since cosine is taken
    over W_anchor."""
    layer = net.layers[layer_idx]
    with torch.no_grad():
        layer.W.data[dst] = layer.W.data[src].clone()
        layer.b.data[dst] = layer.b.data[src].clone()
        layer.W_anchor[dst] = layer.W_anchor[src].clone()
        layer.b_anchor[dst] = layer.b_anchor[src].clone()
        layer.lam[dst] = layer.lam[src].clone()
        layer.fisher_W[dst] = layer.fisher_W[src].clone()
        layer.fisher_b[dst] = layer.fisher_b[src].clone()
        layer.u[dst] = layer.u[src].clone()


def _contrastive_loss(h_a, h_b, margin=1.0):
    d = (h_a - h_b).pow(2).sum(dim=1).clamp_min(1e-12).sqrt()
    return F.relu(margin - d).pow(2).mean()


def _make_pair_fn(state_dim, rng_seed=0):
    g = torch.Generator().manual_seed(rng_seed)

    def fn(name: str, batch: int):
        a = torch.rand((batch, state_dim), generator=g) * 2 - 1
        # Make pair semantics depend on `name` so different pairs differ
        # — encode name to a sign per dim.
        sign = 1.0 if (hash(name) % 2 == 0) else -1.0
        b = sign * a + 0.05 * (torch.rand((batch, state_dim), generator=g) * 2 - 1)
        return a, b

    return fn


# --------------------------------------------------------------------------- #
# find_redundant_pairs                                                        #
# --------------------------------------------------------------------------- #


def test_find_redundant_pairs_empty_when_no_duplicates():
    torch.manual_seed(0)
    net = _make_net(hidden=6)
    pairs = find_redundant_pairs(net, layer_idx=0, cos_threshold=0.99)
    # Random init: extremely unlikely to have cos > 0.99 between any pair.
    assert pairs == [], f"unexpected redundant pairs at random init: {pairs}"


def test_find_redundant_pairs_finds_planted_duplicate():
    torch.manual_seed(0)
    net = _make_net(hidden=6)
    _plant_duplicate(net, layer_idx=0, src=0, dst=3)
    pairs = find_redundant_pairs(net, layer_idx=0, cos_threshold=0.95)
    assert len(pairs) >= 1, "planted duplicate not detected"
    i, j, s = pairs[0]
    assert {i, j} == {0, 3}, f"expected pair (0,3), got ({i},{j})"
    assert s > 0.999, f"expected cos~1, got {s}"


def test_find_redundant_pairs_threshold_excludes_low_sim():
    torch.manual_seed(0)
    net = _make_net(hidden=6)
    _plant_duplicate(net, layer_idx=0, src=0, dst=3)
    # Threshold above the planted pair's similarity excludes it.
    pairs_high = find_redundant_pairs(net, layer_idx=0, cos_threshold=1.001)
    assert pairs_high == [], "threshold > 1 should yield no pairs"


# --------------------------------------------------------------------------- #
# merge_nodes                                                                 #
# --------------------------------------------------------------------------- #


def test_merge_preserves_forward_when_duplicate():
    """When peer and victim are exact duplicates, the merged forward
    output equals the original up to floating-point noise."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=5, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=1)

    x = torch.randn(8, 5)
    y_before = net(x).detach().clone()

    merge_nodes(net, layer_idx=0, peer_idx=0, victim_idx=1)
    y_after = net(x).detach()

    err = (y_after - y_before).abs().max().item()
    assert err < 1e-5, f"forward drift after merging duplicates: {err}"


def test_merge_averages_per_node_state():
    torch.manual_seed(0)
    net = _make_net(hidden=4)
    layer = net.layers[0]
    with torch.no_grad():
        layer.lam.zero_(); layer.lam[0] = 2.0; layer.lam[1] = 4.0
        layer.fisher_b.zero_(); layer.fisher_b[0] = 1.0; layer.fisher_b[1] = 5.0
        layer.u.zero_(); layer.u[0] = 0.3; layer.u[1] = 0.9
        b0 = layer.b.data[0].clone()
        b1 = layer.b.data[1].clone()
    merge_nodes(net, layer_idx=0, peer_idx=0, victim_idx=1)
    layer = net.layers[0]
    assert abs(layer.lam[0].item() - 3.0) < 1e-6, layer.lam[0]
    assert abs(layer.fisher_b[0].item() - 3.0) < 1e-6, layer.fisher_b[0]
    assert abs(layer.u[0].item() - 0.9) < 1e-6, "u should be max"
    assert abs(layer.b.data[0].item() - 0.5 * (b0 + b1).item()) < 1e-6


def test_merge_sums_next_layer_outgoing():
    torch.manual_seed(0)
    net = _make_net(hidden=4)
    nxt = net.layers[1]
    with torch.no_grad():
        col0 = nxt.W.data[:, 0].clone()
        col1 = nxt.W.data[:, 1].clone()
    merge_nodes(net, layer_idx=0, peer_idx=0, victim_idx=1)
    nxt = net.layers[1]
    expected = col0 + col1
    err = (nxt.W.data[:, 0] - expected).abs().max().item()
    assert err < 1e-6, f"next-layer outgoing-sum mismatch: {err}"


def test_merge_drops_victim_and_input_column():
    torch.manual_seed(0)
    net = _make_net(hidden=5)
    n0_before = net.layers[0].n_nodes
    fan1_before = net.layers[1].fan_in
    merge_nodes(net, layer_idx=0, peer_idx=0, victim_idx=2)
    assert net.layers[0].n_nodes == n0_before - 1
    assert net.layers[1].fan_in == fan1_before - 1


def test_merge_rejects_self_or_oob():
    torch.manual_seed(0)
    net = _make_net(hidden=4)
    try:
        merge_nodes(net, layer_idx=0, peer_idx=0, victim_idx=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for peer == victim")
    try:
        merge_nodes(net, layer_idx=0, peer_idx=0, victim_idx=99)
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError for OOB victim")


# --------------------------------------------------------------------------- #
# compress                                                                    #
# --------------------------------------------------------------------------- #


def test_compress_reduces_n_nodes_with_planted_duplicate():
    torch.manual_seed(0)
    net = _make_net(hidden=6)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)
    n0_before = net.layers[0].n_nodes
    events = compress(net, cos_threshold=0.95)
    assert len(events) >= 1, "compress did nothing despite planted duplicate"
    assert net.layers[0].n_nodes == n0_before - len(events)
    for ev in events:
        assert isinstance(ev, MergeEvent)
        assert ev.cos_sim >= 0.95


def test_compress_no_op_when_no_redundancy():
    torch.manual_seed(0)
    net = _make_net(hidden=4)
    arch_before = tuple(net.n_nodes_per_layer())
    events = compress(net, cos_threshold=0.999)
    assert events == []
    assert tuple(net.n_nodes_per_layer()) == arch_before


def test_compress_skips_output_layer_by_default():
    torch.manual_seed(0)
    net = _make_net(hidden=4, latent=4)
    # Plant a duplicate on the output layer (layer 2).
    _plant_duplicate(net, layer_idx=2, src=0, dst=1)
    events = compress(net, cos_threshold=0.95)
    # No event on layer 2 — and since layers 0/1 had no duplicates planted,
    # likely no events at all on a 4-wide random layer.
    for ev in events:
        assert ev.layer_idx != 2, f"output layer {ev.layer_idx} should be skipped"


def test_compress_includes_output_when_explicitly_listed():
    torch.manual_seed(0)
    net = _make_net(hidden=4, latent=4)
    _plant_duplicate(net, layer_idx=2, src=0, dst=1)
    events = compress(
        net, cos_threshold=0.95, layer_idxs=[2], skip_output_layer=False,
    )
    assert any(ev.layer_idx == 2 for ev in events)


# --------------------------------------------------------------------------- #
# replay                                                                      #
# --------------------------------------------------------------------------- #


def test_replay_handles_empty_past_pairs():
    torch.manual_seed(0)
    net = _make_net()
    pair_fn = _make_pair_fn(state_dim=6)
    before, after, n_pairs, steps = replay(
        net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
        past_pair_names=[], n_steps_per_pair=10, batch=8,
    )
    assert n_pairs == 0 and steps == 0
    assert before == 0.0 and after == 0.0


def test_replay_decreases_loss_on_familiar_pairs():
    """After 200 steps of plain training, replay with EWC=0 (so it's just
    fine-tuning) should not INCREASE the loss on the same pairs."""
    torch.manual_seed(0)
    net = _make_net(hidden=8)
    pair_fn = _make_pair_fn(state_dim=6, rng_seed=1)
    past = ["alpha", "beta", "gamma", "delta"]

    rng = random.Random(0)
    before, after, n_pairs, steps = replay(
        net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
        past_pair_names=past, fraction=1.0, n_steps_per_pair=80,
        batch=16, ewc_strength=0.0, lr=3e-3, rng=rng,
    )
    assert n_pairs == 4
    assert steps == 4 * 80
    # Replay is fine-tuning — should reduce or at worst hold the loss.
    assert after <= before + 1e-3, (
        f"replay increased loss: before={before:.4f} after={after:.4f}"
    )


# --------------------------------------------------------------------------- #
# purge                                                                       #
# --------------------------------------------------------------------------- #


def test_purge_drops_low_utility_nodes():
    torch.manual_seed(0)
    net = _make_net(hidden=5)
    layer = net.layers[0]
    with torch.no_grad():
        # Two below threshold, three above.
        layer.u.copy_(torch.tensor([0.5, 0.0, 0.5, 0.0001, 0.5]))
    n_before = layer.n_nodes
    events = purge(net, u_threshold=1e-3, layer_idxs=[0])
    # Indices 1 and 3 are below threshold.
    assert len(events) == 2, f"expected 2 purges, got {len(events)}"
    assert net.layers[0].n_nodes == n_before - 2
    for ev in events:
        assert ev.u_at_purge < 1e-3
        assert isinstance(ev, PurgeEvent)


def test_purge_no_op_when_all_above_threshold():
    torch.manual_seed(0)
    net = _make_net(hidden=4)
    with torch.no_grad():
        net.layers[0].u.fill_(1.0)
        net.layers[1].u.fill_(1.0)
    arch_before = tuple(net.n_nodes_per_layer())
    events = purge(net, u_threshold=1e-3)
    assert events == []
    assert tuple(net.n_nodes_per_layer()) == arch_before


def test_purge_skips_output_layer_by_default():
    torch.manual_seed(0)
    net = _make_net(hidden=4, latent=3)
    with torch.no_grad():
        net.layers[2].u.zero_()  # Output layer all below threshold.
    events = purge(net, u_threshold=1e-3)
    for ev in events:
        assert ev.layer_idx != 2


def test_purge_never_drops_last_node():
    torch.manual_seed(0)
    net = _make_net(hidden=2)
    with torch.no_grad():
        net.layers[0].u.zero_()  # All below threshold.
    events = purge(net, u_threshold=1e-3, layer_idxs=[0])
    # Only one purge event — can't drop the last remaining node.
    assert net.layers[0].n_nodes == 1
    assert len(events) == 1


# --------------------------------------------------------------------------- #
# max_off_diag_cosine probe                                                   #
# --------------------------------------------------------------------------- #


def test_max_off_diag_cosine_random_init_is_low():
    torch.manual_seed(0)
    net = _make_net(hidden=8)
    m = max_off_diag_cosine(net, layer_idx=0)
    # Random Kaiming 6-d vectors: pairwise cosine should be well below 0.9.
    assert m < 0.9, f"random-init max cosine unexpectedly high: {m}"


def test_max_off_diag_cosine_planted_duplicate_is_one():
    torch.manual_seed(0)
    net = _make_net(hidden=6)
    _plant_duplicate(net, layer_idx=0, src=0, dst=2)
    m = max_off_diag_cosine(net, layer_idx=0)
    assert m > 0.999, f"planted duplicate should give cos~1, got {m}"


def test_max_off_diag_cosine_handles_singleton_layer():
    torch.manual_seed(0)
    # latent=1 layer can be probed but has no off-diagonal pair.
    net = _make_net(hidden=4, latent=1)
    m = max_off_diag_cosine(net, layer_idx=2)
    assert m == float("-inf")


def test_dreaming_report_pre_compress_max_cosines_populated():
    """The probe field reports per-layer max cosines for the layers
    compress considered (respects skip_output_layer)."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)

    pair_fn = _make_pair_fn(state_dim=6, rng_seed=2)
    rep = dreaming_block(
        net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
        past_pair_names=["alpha"], replay_fraction=1.0,
        replay_steps_per_pair=2, replay_batch=4, ewc_strength=0.0,
        cos_threshold=0.95, u_threshold=1e9, rng=random.Random(0),
        skip_output_layer=True,
    )
    cosines = dict(rep.pre_compress_max_cosines)
    assert 0 in cosines and 1 in cosines, (
        f"hidden layers missing from probe: {rep.pre_compress_max_cosines}"
    )
    assert 2 not in cosines, (
        f"output layer should be skipped under default: {rep.pre_compress_max_cosines}"
    )


def test_dreaming_report_probe_includes_output_when_unskipped():
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=3)
    pair_fn = _make_pair_fn(state_dim=6, rng_seed=2)
    rep = dreaming_block(
        net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
        past_pair_names=["alpha"], replay_fraction=1.0,
        replay_steps_per_pair=2, replay_batch=4, ewc_strength=0.0,
        cos_threshold=2.0, u_threshold=1e9, rng=random.Random(0),
        skip_output_layer=False,
    )
    cosines = dict(rep.pre_compress_max_cosines)
    assert 2 in cosines, (
        f"output layer missing from probe under skip=False: "
        f"{rep.pre_compress_max_cosines}"
    )


# --------------------------------------------------------------------------- #
# activation-correlation detector                                             #
# --------------------------------------------------------------------------- #


def test_find_activation_pairs_empty_when_no_correlation():
    """Random Kaiming init + random probe batch: no pair should clear 0.95
    activation cosine."""
    torch.manual_seed(0)
    net = _make_net(hidden=8, fan_in=6)
    probe = torch.randn(128, 6)
    pairs = find_activation_redundant_pairs(
        net, layer_idx=0, probe_batch=probe, ac_threshold=0.95,
    )
    assert pairs == [], (
        f"random init produced spurious activation-redundant pairs: {pairs}"
    )


def test_find_activation_pairs_finds_planted_duplicate():
    """Planting a W/b duplicate forces identical activations across any
    probe batch — activation cosine should hit ~1.0 on that pair."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6)
    _plant_duplicate(net, layer_idx=0, src=0, dst=3)
    probe = torch.randn(128, 6)
    pairs = find_activation_redundant_pairs(
        net, layer_idx=0, probe_batch=probe, ac_threshold=0.95,
    )
    assert len(pairs) >= 1, "planted duplicate not detected by activation probe"
    i, j, s = pairs[0]
    assert {i, j} == {0, 3}, f"expected pair (0,3), got ({i},{j})"
    assert s > 0.999, f"expected cos~1, got {s}"


def test_find_activation_pairs_orthogonal_W_correlated_activations():
    """The case the W-cosine detector misses: two nodes with orthogonal
    incoming weights whose activation patterns nevertheless correlate
    because the data distribution lies on a manifold where w1·x ≈ w2·x.

    Construction: hidden layer with 2 nodes, fan_in=2. w1=(1,0), w2=(0,1)
    are orthogonal (W cosine = 0). Probe batch sampled from the diagonal
    line x[0] ≈ x[1] → preactivations are nearly identical → identical
    relu outputs → activation cosine ≈ 1. The grow_layer mechanism
    produces orthogonal-by-construction dims that may still be
    functionally correlated when the data sees them on similar slices.
    """
    torch.manual_seed(0)
    net = TrioronNetwork([(2, 2, "relu"), (2, 1, "tanh")])
    layer0 = net.layers[0]
    with torch.no_grad():
        layer0.W.data[0] = torch.tensor([1.0, 0.0])
        layer0.W.data[1] = torch.tensor([0.0, 1.0])
        layer0.b.data.zero_()
        layer0.W_anchor[0] = torch.tensor([1.0, 0.0])
        layer0.W_anchor[1] = torch.tensor([0.0, 1.0])
        layer0.b_anchor.zero_()

    # Sanity: W cosine is exactly 0 for these rows.
    w_cos = max_off_diag_cosine(net, 0)
    assert abs(w_cos) < 1e-6, f"W cosine should be 0 for orthogonal rows, got {w_cos}"

    # Probe on the diagonal x[0] ≈ x[1] (positive side so relu doesn't zero
    # everything out — both pre-activations stay > 0).
    t = torch.rand(128, 1) * 2.0 + 0.1  # values in [0.1, 2.1]
    probe = torch.cat([t, t + 0.01 * torch.randn(128, 1)], dim=1)

    pairs = find_activation_redundant_pairs(
        net, layer_idx=0, probe_batch=probe, ac_threshold=0.95,
    )
    assert len(pairs) == 1, (
        f"expected exactly 1 redundant pair on diagonal probe, got {len(pairs)}"
    )
    i, j, s = pairs[0]
    assert {i, j} == {0, 1}
    assert s > 0.95, f"orthogonal-W diagonal-probe cos should be high, got {s}"


def test_max_off_diag_activation_cosine_handles_singleton_layer():
    torch.manual_seed(0)
    net = _make_net(hidden=4, latent=1)
    probe = torch.randn(32, 6)
    m = max_off_diag_activation_cosine(net, layer_idx=2, probe_batch=probe)
    assert m == float("-inf")


def test_compress_activation_signal_requires_probe_batch():
    torch.manual_seed(0)
    net = _make_net(hidden=4)
    try:
        compress(net, redundancy_signal="activation")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "compress(redundancy_signal='activation') without probe_batch "
            "should raise ValueError"
        )


def test_compress_rejects_unknown_signal():
    torch.manual_seed(0)
    net = _make_net(hidden=4)
    try:
        compress(net, redundancy_signal="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown redundancy_signal")


def test_compress_activation_merges_planted_duplicate():
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)
    probe = torch.randn(128, 6)
    n0_before = net.layers[0].n_nodes
    # Restrict to layer 0 so we measure only the planted pair's effect —
    # downstream layers can show activation correlations on random probes
    # that aren't relevant to this assertion.
    events = compress(
        net, redundancy_signal="activation",
        probe_batch=probe, ac_threshold=0.95,
        layer_idxs=[0], skip_output_layer=False,
    )
    assert len(events) >= 1, "activation compress missed planted duplicate"
    assert net.layers[0].n_nodes == n0_before - len(events)
    for ev in events:
        assert ev.layer_idx == 0
        assert ev.cos_sim >= 0.95
    # First (highest-similarity) event should capture the planted pair.
    first = events[0]
    assert {first.peer_idx, first.victim_idx} == {0, 4}, (
        f"first merge should be the planted (0,4), got "
        f"({first.peer_idx},{first.victim_idx})"
    )


def test_dreaming_block_activation_signal_populates_probe_field():
    """End-to-end: redundancy_signal='activation' with a planted duplicate
    on a hidden layer should fire at least one merge AND populate
    pre_compress_max_activation_cosines (and leave pre_compress_max_cosines
    populated too, for diagnostic comparison)."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)

    pair_fn = _make_pair_fn(state_dim=6, rng_seed=2)
    rep = dreaming_block(
        net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
        past_pair_names=["alpha", "beta"], replay_fraction=1.0,
        replay_steps_per_pair=2, replay_batch=4, ewc_strength=0.0,
        redundancy_signal="activation", ac_threshold=0.95,
        probe_batch_size=64, u_threshold=1e9,
        rng=random.Random(0),
    )
    assert isinstance(rep, DreamingReport)
    assert len(rep.pre_compress_max_activation_cosines) >= 1, (
        "activation probe field should be populated"
    )
    assert len(rep.pre_compress_max_cosines) >= 1, (
        "weight probe field should still be populated for cross-check"
    )
    # Plant survived through 2 replay steps with EWC=0 — duplicate weight
    # state should still produce duplicate activations → at least one merge.
    assert len(rep.merges) >= 1, (
        "planted duplicate should be merged via activation signal"
    )


def test_dreaming_block_activation_falls_back_when_no_past_pairs():
    """First-task case: past_pair_names=[] → no probe batch can be built;
    block must complete (falling back to the weight signal) without
    raising, and the activation-probe field stays empty."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)

    pair_fn = _make_pair_fn(state_dim=6, rng_seed=2)
    rep = dreaming_block(
        net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
        past_pair_names=[],  # ← the trigger
        replay_fraction=1.0, replay_steps_per_pair=2, replay_batch=4,
        ewc_strength=0.0, redundancy_signal="activation",
        ac_threshold=0.95, probe_batch_size=64, u_threshold=1e9,
        rng=random.Random(0),
    )
    assert isinstance(rep, DreamingReport)
    assert rep.pre_compress_max_activation_cosines == []


def test_dreaming_block_rejects_unknown_signal():
    torch.manual_seed(0)
    net = _make_net()
    pair_fn = _make_pair_fn(state_dim=6)
    try:
        dreaming_block(
            net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
            past_pair_names=["alpha"], replay_steps_per_pair=1,
            replay_batch=4, redundancy_signal="bogus",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown redundancy_signal")


# --------------------------------------------------------------------------- #
# synaptic_downscale (substrate-preserving compression)                       #
# --------------------------------------------------------------------------- #


def _snapshot_layer(layer):
    """Capture per-node state at a layer so we can assert preservation."""
    return {
        "W": layer.W.data.clone(),
        "b": layer.b.data.clone(),
        "W_anchor": layer.W_anchor.clone(),
        "b_anchor": layer.b_anchor.clone(),
        "lam": layer.lam.clone(),
        "fisher_W": layer.fisher_W.clone(),
        "fisher_b": layer.fisher_b.clone(),
        "u": layer.u.clone(),
        "n_nodes": layer.n_nodes,
        "fan_in": layer.fan_in,
    }


def test_downscale_preserves_victim_substrate_at_layer_L():
    """Layer L is the layer being compressed; victim's row in W, b,
    anchor, lam, fisher must be unchanged."""
    torch.manual_seed(0)
    net = _make_net(hidden=5, fan_in=6, latent=2)
    L = 0
    layer = net.layers[L]
    snap = _snapshot_layer(layer)

    synaptic_downscale(net, layer_idx=L, peer_idx=0, victim_idx=2)

    assert layer.n_nodes == snap["n_nodes"], (
        "downscale must NOT change architecture"
    )
    for key in ("W", "b", "W_anchor", "b_anchor", "lam",
                "fisher_W", "fisher_b", "u"):
        before = snap[key]
        after = getattr(layer, key)
        if hasattr(after, "data"):
            after = after.data
        err = (after - before).abs().max().item()
        assert err < 1e-9, (
            f"layer {L} {key} drifted under downscale (max err {err}) — "
            f"substrate must be preserved"
        )


def test_downscale_zeros_victim_outgoing_at_next_layer():
    torch.manual_seed(0)
    net = _make_net(hidden=5, fan_in=6, latent=2)
    L = 0
    victim = 2
    synaptic_downscale(net, layer_idx=L, peer_idx=0, victim_idx=victim)
    nxt = net.layers[L + 1]
    assert nxt.W.data[:, victim].abs().max().item() < 1e-9
    assert nxt.W_anchor[:, victim].abs().max().item() < 1e-9
    assert nxt.fisher_W[:, victim].abs().max().item() < 1e-9


def test_downscale_peer_absorbs_victim_outgoing():
    torch.manual_seed(0)
    net = _make_net(hidden=5, fan_in=6, latent=2)
    L = 0
    nxt = net.layers[L + 1]
    with torch.no_grad():
        peer_W_before = nxt.W.data[:, 0].clone()
        victim_W_before = nxt.W.data[:, 3].clone()
        peer_anchor_before = nxt.W_anchor[:, 0].clone()
        victim_anchor_before = nxt.W_anchor[:, 3].clone()
        peer_fisher_before = nxt.fisher_W[:, 0].clone()
        victim_fisher_before = nxt.fisher_W[:, 3].clone()

    synaptic_downscale(net, layer_idx=L, peer_idx=0, victim_idx=3)

    err_w = (nxt.W.data[:, 0]
             - (peer_W_before + victim_W_before)).abs().max().item()
    err_anchor = (nxt.W_anchor[:, 0]
                  - (peer_anchor_before + victim_anchor_before)
                  ).abs().max().item()
    err_fisher = (nxt.fisher_W[:, 0]
                  - (peer_fisher_before + victim_fisher_before)
                  ).abs().max().item()
    assert err_w < 1e-6, f"peer outgoing W not absorbed (err {err_w})"
    assert err_anchor < 1e-6, f"peer outgoing anchor not absorbed (err {err_anchor})"
    assert err_fisher < 1e-6, f"peer outgoing fisher not absorbed (err {err_fisher})"


def test_downscale_self_or_oob_rejected():
    torch.manual_seed(0)
    net = _make_net(hidden=4)
    try:
        synaptic_downscale(net, layer_idx=0, peer_idx=1, victim_idx=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for peer == victim")
    try:
        synaptic_downscale(net, layer_idx=0, peer_idx=0, victim_idx=99)
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError for OOB victim")


def test_downscale_on_output_layer_is_noop():
    """No L+1 to redirect into → must not raise, must change nothing."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, latent=3)
    L = len(net.layers) - 1  # output layer
    snap = _snapshot_layer(net.layers[L])
    synaptic_downscale(net, layer_idx=L, peer_idx=0, victim_idx=1)
    layer = net.layers[L]
    for key in ("W", "b", "W_anchor", "b_anchor"):
        after = getattr(layer, key)
        if hasattr(after, "data"):
            after = after.data
        err = (after - snap[key]).abs().max().item()
        assert err < 1e-9, f"output-layer downscale changed {key} (err {err})"


def test_downscale_victim_can_be_re_recruited():
    """Fisher on victim's outgoing is zero → no EWC penalty pinning the
    column at zero. A subsequent training step should be able to drive
    the column away from zero (re-recruitment available)."""
    torch.manual_seed(0)
    net = _make_net(hidden=5, fan_in=6, latent=2)
    victim = 2
    synaptic_downscale(net, layer_idx=0, peer_idx=0, victim_idx=victim)
    nxt = net.layers[1]
    # Confirm zeros first.
    assert nxt.W.data[:, victim].abs().max().item() < 1e-9
    assert nxt.fisher_W[:, victim].abs().max().item() < 1e-9

    pair_fn = _make_pair_fn(state_dim=6, rng_seed=3)
    opt = optim.Adam(net.parameters(), lr=3e-2)
    for _ in range(20):
        a, b = pair_fn("rho", 8)
        l_task = _contrastive_loss(net(a), net(b))
        # Add EWC penalty so we're testing that EWC doesn't pin the
        # zeroed column (fisher_W there is 0, lam at layer L+1 is
        # whatever it was). EWC contribution on victim column should
        # therefore be zero, leaving the column free to move.
        l = l_task + 1000.0 * net.ewc_penalty()
        opt.zero_grad(); l.backward(); opt.step()

    moved = nxt.W.data[:, victim].abs().max().item()
    assert moved > 1e-3, (
        f"victim's outgoing column stuck near zero after 20 training steps "
        f"(max abs {moved}) — re-recruitment should be possible"
    )


def test_compress_downscale_with_planted_pair_records_action():
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)
    arch_before = tuple(net.n_nodes_per_layer())

    events = compress(
        net, redundancy_signal="weight", cos_threshold=0.95,
        compression_action="downscale", layer_idxs=[0],
        skip_output_layer=False,
    )

    assert len(events) >= 1, "downscale compress missed planted duplicate"
    assert tuple(net.n_nodes_per_layer()) == arch_before, (
        "downscale must preserve architecture"
    )
    for ev in events:
        assert ev.action == "downscale"
        assert ev.layer_idx == 0
    # First event should be the planted (0, 4) — highest similarity.
    first = events[0]
    assert {first.peer_idx, first.victim_idx} == {0, 4}


def test_compress_downscale_terminates_on_static_victim():
    """With downscale the victim stays in the layer; compress must
    not infinite-loop. Verified by running with a planted pair and
    confirming we exit after exactly one event on the picked pair."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)

    events = compress(
        net, redundancy_signal="weight", cos_threshold=0.95,
        compression_action="downscale", layer_idxs=[0],
        skip_output_layer=False, max_merges=10,
    )
    # The planted pair fires once; subsequent iterations exclude the
    # victim from new candidate pairs, and the only above-threshold
    # pair was the planted one, so the loop exits with 1 event.
    assert len(events) == 1, (
        f"expected exactly 1 downscale event, got {len(events)} — "
        f"loop may not be excluding downscaled victims"
    )


def test_compress_downscale_skips_output_layer_implicitly():
    """When the only candidate layer is the output (which has no L+1),
    downscale skips it entirely — no events, no error."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, latent=4)
    _plant_duplicate(net, layer_idx=2, src=0, dst=1)
    events = compress(
        net, redundancy_signal="weight", cos_threshold=0.95,
        compression_action="downscale", layer_idxs=[2],
        skip_output_layer=False,
    )
    assert events == [], (
        "downscale on output layer should be skipped (no L+1 to absorb into)"
    )


def test_compress_rejects_unknown_action():
    torch.manual_seed(0)
    net = _make_net(hidden=4)
    try:
        compress(net, compression_action="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown compression_action")


# --------------------------------------------------------------------------- #
# Phase 4.5 Experiment 2 — max_downscales_per_layer cap                       #
# --------------------------------------------------------------------------- #


def test_compress_downscale_cap_one_stops_after_first_event():
    """Two planted pairs on one layer; cap=1 fires exactly one event then
    breaks the inner loop without touching the second pair."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)
    _plant_duplicate(net, layer_idx=0, src=1, dst=5)

    events = compress(
        net, redundancy_signal="weight", cos_threshold=0.95,
        compression_action="downscale", layer_idxs=[0],
        skip_output_layer=False, max_downscales_per_layer=1,
    )
    assert len(events) == 1, (
        f"cap=1 should stop after one downscale, got {len(events)}"
    )


def test_compress_downscale_cap_zero_blocks_all_events():
    """cap=0 prevents any downscale even when planted pairs exist."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)

    events = compress(
        net, redundancy_signal="weight", cos_threshold=0.95,
        compression_action="downscale", layer_idxs=[0],
        skip_output_layer=False, max_downscales_per_layer=0,
    )
    assert events == [], (
        f"cap=0 should fire no events, got {len(events)}"
    )


def test_compress_downscale_cap_none_preserves_uncapped_behavior():
    """cap=None matches the prior 1-pair behavior on a single planted pair
    (this exists primarily as a sanity baseline against drift in the cap
    branch — the existing terminates_on_static_victim test is the real
    uncapped contract)."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)

    events = compress(
        net, redundancy_signal="weight", cos_threshold=0.95,
        compression_action="downscale", layer_idxs=[0],
        skip_output_layer=False, max_downscales_per_layer=None,
    )
    assert len(events) == 1, (
        f"uncapped on 1 planted pair should fire exactly once, "
        f"got {len(events)}"
    )


def test_compress_downscale_cap_is_per_layer_not_global():
    """Plant a duplicate on layer 0 and another on layer 1; cap=1 should
    fire one event PER LAYER (total 2), not one event globally."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)
    _plant_duplicate(net, layer_idx=1, src=0, dst=4)

    events = compress(
        net, redundancy_signal="weight", cos_threshold=0.95,
        compression_action="downscale", layer_idxs=[0, 1],
        skip_output_layer=False, max_downscales_per_layer=1,
    )
    assert len(events) == 2, (
        f"cap=1 per layer with 2 candidate layers should fire 2 events, "
        f"got {len(events)}"
    )
    layers_hit = {ev.layer_idx for ev in events}
    assert layers_hit == {0, 1}, (
        f"both layers should fire one event each, got layers {layers_hit}"
    )


def test_compress_downscale_cap_does_not_affect_merge_action():
    """The cap is downscale-specific. With compression_action='merge' and
    cap=1, multiple planted pairs should still all merge (merge has its
    own natural termination via victim deletion + max_merges)."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)
    _plant_duplicate(net, layer_idx=0, src=1, dst=5)

    events = compress(
        net, redundancy_signal="weight", cos_threshold=0.95,
        compression_action="merge", layer_idxs=[0],
        skip_output_layer=False, max_downscales_per_layer=1,
    )
    assert len(events) >= 2, (
        f"merge action should ignore the downscale cap and merge both "
        f"planted pairs; got {len(events)} events"
    )
    for ev in events:
        assert ev.action == "merge"


def test_compress_downscale_cap_negative_raises():
    torch.manual_seed(0)
    net = _make_net(hidden=4)
    try:
        compress(net, compression_action="downscale",
                 max_downscales_per_layer=-1)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected ValueError for negative max_downscales_per_layer"
        )


def test_dreaming_block_threads_max_downscales_per_layer():
    """End-to-end via dreaming_block with activation signal: planted
    duplicate on layer 0 fires only once when cap=1, and the report
    records exactly one downscale event."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)
    _plant_duplicate(net, layer_idx=0, src=1, dst=5)

    pair_fn = _make_pair_fn(state_dim=6, rng_seed=2)
    rep = dreaming_block(
        net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
        past_pair_names=["alpha", "beta"], replay_fraction=1.0,
        replay_steps_per_pair=5, replay_batch=8, ewc_strength=10.0,
        cos_threshold=0.95, ac_threshold=0.95, u_threshold=1e-3,
        rng=random.Random(0),
        redundancy_signal="weight",          # weight signal so the planted
                                             # W-duplicates are reliably found
        compression_action="downscale",
        max_downscales_per_layer=1,
    )
    # Layer 0 has two planted pairs; cap=1 should stop after the first.
    layer_0_events = [m for m in rep.merges if m.layer_idx == 0]
    assert len(layer_0_events) == 1, (
        f"cap=1 via dreaming_block should produce 1 layer-0 event, "
        f"got {len(layer_0_events)}"
    )


def test_dreaming_block_downscale_preserves_param_count():
    """End-to-end: dreaming_block(compression_action='downscale') with a
    planted activation-correlated pair runs an event but leaves the
    param count exactly equal to before-block."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)
    n_before = net.n_parameters()

    pair_fn = _make_pair_fn(state_dim=6, rng_seed=2)
    rep = dreaming_block(
        net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
        past_pair_names=["alpha", "beta"], replay_fraction=1.0,
        replay_steps_per_pair=2, replay_batch=4, ewc_strength=0.0,
        redundancy_signal="activation", ac_threshold=0.95,
        probe_batch_size=64,
        u_threshold=0.0,  # purge drops u < threshold; 0.0 → never fires.
        compression_action="downscale",
        rng=random.Random(0),
    )
    assert isinstance(rep, DreamingReport)
    assert rep.n_params_before == n_before
    assert rep.n_params_after == n_before, (
        f"downscale should leave param count unchanged, "
        f"got {rep.n_params_before} → {rep.n_params_after}"
    )
    assert len(rep.merges) >= 1
    for ev in rep.merges:
        assert ev.action == "downscale"


def test_dreaming_block_rejects_unknown_action():
    torch.manual_seed(0)
    net = _make_net()
    pair_fn = _make_pair_fn(state_dim=6)
    try:
        dreaming_block(
            net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
            past_pair_names=["alpha"], replay_steps_per_pair=1,
            replay_batch=4, compression_action="bogus",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected ValueError for unknown compression_action"
        )


# --------------------------------------------------------------------------- #
# dreaming_block end-to-end                                                   #
# --------------------------------------------------------------------------- #


def test_dreaming_block_runs_and_returns_report():
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    # Plant a duplicate on a hidden layer so compress has work.
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)

    pair_fn = _make_pair_fn(state_dim=6, rng_seed=2)
    past = ["alpha", "beta", "gamma"]
    rng = random.Random(0)

    n_before = net.n_parameters()
    rep = dreaming_block(
        net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
        past_pair_names=past, replay_fraction=1.0,
        replay_steps_per_pair=20, replay_batch=8, ewc_strength=10.0,
        cos_threshold=0.95, u_threshold=1e-3, rng=rng,
    )
    assert isinstance(rep, DreamingReport)
    assert rep.replay_steps == 3 * 20
    assert rep.replay_pairs_sampled == 3
    assert len(rep.merges) >= 1, "planted duplicate should produce a merge"
    assert rep.n_params_before == n_before
    assert rep.n_params_after < rep.n_params_before, (
        "compression should reduce params"
    )

    # Net is still trainable post-block (optimizer rebuild from scratch).
    opt = optim.Adam(net.parameters(), lr=1e-3)
    a, b = pair_fn("alpha", 8)
    h_a, h_b = net(a), net(b)
    loss = _contrastive_loss(h_a, h_b)
    opt.zero_grad(); loss.backward(); opt.step()


# --------------------------------------------------------------------------- #
# Routing starvation (Phase 4.5 Experiment 3)                                 #
# --------------------------------------------------------------------------- #


def test_routing_scale_buffer_exists_and_init_one():
    """Every layer is born with routing_scale=1.0 per node and unlatched."""
    net = _make_net(hidden=4, fan_in=6, latent=2)
    for L_i, layer in enumerate(net.layers):
        assert torch.allclose(
            layer.routing_scale, torch.ones(layer.n_nodes)
        ), f"layer {L_i}: routing_scale not all-ones at init"
        assert not bool(layer.routing_latched.any().item()), (
            f"layer {L_i}: nothing should be latched at init"
        )
        assert torch.equal(
            layer.task_of_origin,
            torch.zeros(layer.n_nodes, dtype=torch.long),
        ), f"layer {L_i}: task_of_origin not zero at init"


def test_forward_with_routing_scale_one_matches_unscaled():
    """Default routing_scale=1.0 → forward identical to plain F.linear."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    x = torch.randn(8, 6)
    with torch.no_grad():
        out = net(x)
    # Compare against the un-scaled compute path manually.
    h = x
    for layer in net.layers:
        z = F.linear(h, layer.W, layer.b)
        if layer.activation == "relu":
            h = F.relu(z)
        elif layer.activation == "tanh":
            h = torch.tanh(z)
        else:
            h = z
    assert torch.allclose(out, h, atol=1e-6), (
        "forward with routing_scale=1.0 must match unscaled path"
    )


def test_forward_attenuates_when_routing_scale_zero():
    """A zeroed routing_scale collapses the unit's pre-activation to bias."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    with torch.no_grad():
        layer.b.data.zero_()           # bias=0 so the dead unit produces 0
        layer.routing_scale[2] = 0.0    # kill node 2

    x = torch.randn(8, 6)
    with torch.no_grad():
        # First-layer activation only.
        z = F.linear(x, layer.W * layer.routing_scale.unsqueeze(1), layer.b)
        h = F.relu(z)
    assert torch.allclose(h[:, 2], torch.zeros(8)), (
        f"node-2 activation should be zero with scale=0 + bias=0, got {h[:,2]}"
    )


def test_grow_node_records_task_of_origin_and_routing_scale():
    """grow_node(task_idx=5) appends task_of_origin=5 and routing_scale=1.0."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    new_idx = net.grow_layer(layer_idx=0, task_idx=5)
    layer = net.layers[0]
    assert int(layer.task_of_origin[new_idx].item()) == 5
    assert float(layer.routing_scale[new_idx].item()) == 1.0
    assert not bool(layer.routing_latched[new_idx].item())


def test_prune_node_drops_routing_buffers():
    """prune_node removes the corresponding routing_scale / task_of_origin entry."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    with torch.no_grad():
        layer.routing_scale[1] = 0.5  # mark node 1 distinctly
        layer.task_of_origin[2] = 7
    layer.prune_node(1)
    # After dropping idx=1, the layer has 3 nodes; the original node 2
    # (task_of_origin=7) is now at index 1.
    assert layer.routing_scale.shape == (3,)
    assert layer.task_of_origin.shape == (3,)
    assert int(layer.task_of_origin[1].item()) == 7, (
        f"task_of_origin not preserved across prune: {layer.task_of_origin}"
    )


def test_routing_starve_multiplies_scale_and_returns_value():
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    scale_after, latched = routing_starve(
        net, 0, victim_idx=1, alpha=0.5, floor=1e-3,
    )
    assert latched is False
    assert math.isclose(scale_after, 0.5, abs_tol=1e-7)
    assert math.isclose(
        float(net.layers[0].routing_scale[1].item()), 0.5, abs_tol=1e-7,
    )


def test_routing_starve_does_not_touch_peer_or_other_layers():
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    routing_starve(net, 0, victim_idx=1, alpha=0.5, floor=1e-3)
    # Node 0 on layer 0 untouched.
    assert float(net.layers[0].routing_scale[0].item()) == 1.0
    # Other layers entirely untouched.
    for L_i in (1, 2):
        assert torch.allclose(
            net.layers[L_i].routing_scale,
            torch.ones(net.layers[L_i].n_nodes),
        )


def test_routing_starve_latches_below_floor():
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    with torch.no_grad():
        layer.routing_scale[2] = 0.005
    scale_after, latched = routing_starve(
        net, 0, victim_idx=2, alpha=0.5, floor=0.01,
    )
    assert latched is True
    assert scale_after == 0.0
    assert bool(layer.routing_latched[2].item())
    assert float(layer.routing_scale[2].item()) == 0.0


def test_routing_starve_already_latched_is_noop():
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    with torch.no_grad():
        layer.routing_latched[3] = True
        layer.routing_scale[3] = 0.0
    scale_after, latched = routing_starve(
        net, 0, victim_idx=3, alpha=0.5, floor=1e-3,
    )
    assert (scale_after, latched) == (0.0, True)
    assert float(layer.routing_scale[3].item()) == 0.0


def test_routing_starve_validates_alpha_and_floor():
    net = _make_net(hidden=4, fan_in=6, latent=2)
    for bad_alpha in (-0.1, 0.0, 1.0, 1.5):
        try:
            routing_starve(net, 0, victim_idx=0, alpha=bad_alpha, floor=1e-3)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for alpha={bad_alpha}")
    try:
        routing_starve(net, 0, victim_idx=0, alpha=0.5, floor=-0.1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for negative floor")


def test_routing_regrow_multiplies_by_inv_alpha_capped_at_one():
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    with torch.no_grad():
        layer.routing_scale[1] = 0.49
    scale = routing_regrow(net, 0, victim_idx=1, alpha=0.7)
    # 0.49 / 0.7 = 0.7 — still under 1.0
    assert math.isclose(scale, 0.7, abs_tol=1e-6)
    # Run again — would push above 1.0, must cap.
    scale = routing_regrow(net, 0, victim_idx=1, alpha=0.7)
    assert math.isclose(scale, 1.0, abs_tol=1e-6)


def test_routing_regrow_skips_latched():
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    with torch.no_grad():
        layer.routing_latched[2] = True
        layer.routing_scale[2] = 0.0
    scale = routing_regrow(net, 0, victim_idx=2, alpha=0.7)
    assert scale == 0.0
    assert float(layer.routing_scale[2].item()) == 0.0


def test_pick_starvation_victim_older_keeps():
    """Older task_of_origin → primary."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    with torch.no_grad():
        layer.task_of_origin[1] = 0
        layer.task_of_origin[3] = 7
    peer, victim = _pick_starvation_victim(net, 0, 1, 3)
    assert (peer, victim) == (1, 3), (
        f"expected older(idx=1, age=0) primary; got peer={peer} victim={victim}"
    )
    # Order of args reversed → same answer.
    peer, victim = _pick_starvation_victim(net, 0, 3, 1)
    assert (peer, victim) == (1, 3)


def test_pick_starvation_victim_tiebreak_outgoing_norm():
    """Same age → larger outgoing-norm on layer L+1 = primary."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    nxt = net.layers[1]
    with torch.no_grad():
        layer.task_of_origin[1] = 5
        layer.task_of_origin[2] = 5
        nxt.W_anchor[:, 1] = 10.0
        nxt.W_anchor[:, 2] = 0.5
    peer, victim = _pick_starvation_victim(net, 0, 1, 2)
    assert (peer, victim) == (1, 2)


def test_pick_starvation_victim_tied_falls_back_to_index():
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    nxt = net.layers[1]
    with torch.no_grad():
        layer.task_of_origin[2] = 0
        layer.task_of_origin[3] = 0
        nxt.W_anchor[:, 2] = 0.0
        nxt.W_anchor[:, 3] = 0.0
    peer, victim = _pick_starvation_victim(net, 0, 2, 3)
    assert (peer, victim) == (2, 3), (
        f"with everything tied, smaller idx is primary; got {peer},{victim}"
    )


def test_compress_starve_picks_younger_victim_and_records_event():
    """Plant W-cosine pair on layer 0 between an OLD (origin=0) and NEW
    (origin=7) node. compress(action='starve') must starve the new one."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=3)
    layer = net.layers[0]
    with torch.no_grad():
        layer.task_of_origin[0] = 0
        layer.task_of_origin[3] = 7
    events = compress(
        net, redundancy_signal="weight", cos_threshold=0.95,
        compression_action="starve", starvation_alpha=0.5,
        starvation_floor=1e-3, max_downscales_per_layer=1,
        layer_idxs=[0],
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.action == "starve"
    assert ev.peer_idx == 0
    assert ev.victim_idx == 3
    assert math.isclose(ev.victim_routing_scale_after, 0.5, abs_tol=1e-6)
    assert ev.victim_latched is False
    assert math.isclose(
        float(layer.routing_scale[3].item()), 0.5, abs_tol=1e-6,
    )
    # Peer untouched.
    assert float(layer.routing_scale[0].item()) == 1.0


def test_compress_starve_caps_events_per_layer():
    """Two planted pairs, cap=1 → only one starve event fires on layer 0."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)
    _plant_duplicate(net, layer_idx=0, src=1, dst=5)
    events = compress(
        net, redundancy_signal="weight", cos_threshold=0.95,
        compression_action="starve", starvation_alpha=0.7,
        starvation_floor=1e-3, max_downscales_per_layer=1,
    )
    layer_0 = [e for e in events if e.layer_idx == 0]
    assert len(layer_0) == 1, (
        f"cap=1 should yield exactly one layer-0 event, got {len(layer_0)}"
    )


def test_compress_starve_regrows_non_victim_below_one():
    """A node with scale<1.0 that ISN'T selected as victim this call regrows."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    # Start node 2 mid-ramp. Plant no redundancy elsewhere.
    with torch.no_grad():
        layer.routing_scale[2] = 0.49
    events = compress(
        net, redundancy_signal="weight", cos_threshold=0.95,
        compression_action="starve", starvation_alpha=0.7,
        starvation_floor=1e-3, layer_idxs=[0],
    )
    assert events == [], "no redundancy planted; no events expected"
    # 0.49 / 0.7 = 0.7
    assert math.isclose(
        float(layer.routing_scale[2].item()), 0.7, abs_tol=1e-6,
    ), f"expected regrow to 0.7, got {layer.routing_scale[2]}"


def test_compress_starve_latched_does_not_regrow():
    """A latched unit (scale=0, latched=True) stays at 0 across compress()."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    with torch.no_grad():
        layer.routing_latched[3] = True
        layer.routing_scale[3] = 0.0
    compress(
        net, redundancy_signal="weight", cos_threshold=0.95,
        compression_action="starve", starvation_alpha=0.7,
        starvation_floor=1e-3,
    )
    assert float(layer.routing_scale[3].item()) == 0.0
    assert bool(layer.routing_latched[3].item())


def test_compress_starve_repeated_calls_compound_to_latch():
    """Repeated compress calls keep ramping the same redundant pair until it
    crosses the floor and latches. After latching, no further changes."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=3)
    layer = net.layers[0]
    with torch.no_grad():
        layer.task_of_origin[3] = 9   # younger → victim
    # alpha=0.5, floor=0.05 → ramp 1 → 0.5 → 0.25 → 0.125 → 0.0625 → latch.
    last_scale = 1.0
    latched_seen = False
    for _ in range(10):
        compress(
            net, redundancy_signal="weight", cos_threshold=0.95,
            compression_action="starve", starvation_alpha=0.5,
            starvation_floor=0.05, max_downscales_per_layer=1,
        )
        s = float(layer.routing_scale[3].item())
        if bool(layer.routing_latched[3].item()):
            latched_seen = True
            assert s == 0.0
            break
        assert s < last_scale, f"scale should monotonically decrease: {s}"
        last_scale = s
    assert latched_seen, "victim should have latched within 10 iterations"


def test_compress_starve_validates_alpha_and_floor():
    net = _make_net(hidden=4, fan_in=6, latent=2)
    for bad_alpha in (-0.1, 0.0, 1.0, 1.5):
        try:
            compress(
                net, compression_action="starve",
                starvation_alpha=bad_alpha, starvation_floor=1e-3,
            )
        except ValueError:
            continue
        raise AssertionError(
            f"compress should reject starvation_alpha={bad_alpha}"
        )
    try:
        compress(
            net, compression_action="starve",
            starvation_alpha=0.5, starvation_floor=-0.1,
        )
    except ValueError:
        return
    raise AssertionError("compress should reject negative starvation_floor")


def test_compress_starve_skips_pair_when_both_latched():
    """If the only redundant pair has both nodes already latched, no events."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=3)
    layer = net.layers[0]
    with torch.no_grad():
        layer.routing_latched[0] = True
        layer.routing_latched[3] = True
        layer.routing_scale[0] = 0.0
        layer.routing_scale[3] = 0.0
    events = compress(
        net, redundancy_signal="weight", cos_threshold=0.95,
        compression_action="starve", starvation_alpha=0.5,
        starvation_floor=1e-3, layer_idxs=[0],
    )
    assert events == []


def test_compress_starve_one_latched_forces_other_as_victim():
    """If exactly one node in a redundant pair is latched, it MUST be primary
    (no further starve possible) and the un-latched node becomes victim
    regardless of age."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=3)
    layer = net.layers[0]
    # Make node 0 OLDER than node 3 — normally node 3 would be victim.
    with torch.no_grad():
        layer.task_of_origin[0] = 0
        layer.task_of_origin[3] = 9
        # Latch node 3 first — ordinary victim is now ineligible.
        layer.routing_latched[3] = True
        layer.routing_scale[3] = 0.0
    events = compress(
        net, redundancy_signal="weight", cos_threshold=0.95,
        compression_action="starve", starvation_alpha=0.5,
        starvation_floor=1e-3, max_downscales_per_layer=1,
        layer_idxs=[0],
    )
    assert len(events) == 1
    assert events[0].victim_idx == 0  # forced — node 3 is latched
    assert events[0].peer_idx == 3
    assert math.isclose(
        float(layer.routing_scale[0].item()), 0.5, abs_tol=1e-6,
    )


def test_compress_starve_preserves_param_count():
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=3)
    n_before = net.n_parameters()
    compress(
        net, redundancy_signal="weight", cos_threshold=0.95,
        compression_action="starve", starvation_alpha=0.5,
        starvation_floor=1e-3, max_downscales_per_layer=1,
    )
    assert net.n_parameters() == n_before, (
        "starve must not change parameter count"
    )


def test_dreaming_block_starve_threads_starvation_params():
    """End-to-end: dreaming_block(compression_action='starve') ramps the
    victim's routing_scale by the requested alpha and records the action."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)
    pair_fn = _make_pair_fn(state_dim=6, rng_seed=2)
    rep = dreaming_block(
        net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
        past_pair_names=["alpha", "beta"], replay_fraction=1.0,
        replay_steps_per_pair=2, replay_batch=4, ewc_strength=0.0,
        cos_threshold=0.95, ac_threshold=0.95, u_threshold=0.0,
        rng=random.Random(0),
        redundancy_signal="weight",
        compression_action="starve",
        starvation_alpha=0.6, starvation_floor=1e-3,
        max_downscales_per_layer=1,
    )
    assert rep.n_params_after == rep.n_params_before
    starves = [m for m in rep.merges if m.action == "starve"]
    assert len(starves) >= 1
    ev = starves[0]
    assert ev.layer_idx == 0
    assert math.isclose(ev.victim_routing_scale_after, 0.6, abs_tol=1e-6)


# --------------------------------------------------------------------------- #
# Apoptosis spike (Phase 4.5 Experiment 5)                                    #
# --------------------------------------------------------------------------- #


def test_apoptosis_pulse_buffer_init_zero():
    net = _make_net(hidden=4, fan_in=6, latent=2)
    for layer in net.layers:
        assert torch.allclose(
            layer.apoptosis_pulse, torch.zeros(layer.n_nodes)
        )


def test_apoptosis_pulse_extends_with_grow_node():
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    new_idx = net.grow_layer(layer_idx=0, task_idx=3)
    layer = net.layers[0]
    assert layer.apoptosis_pulse.shape == (layer.n_nodes,)
    assert float(layer.apoptosis_pulse[new_idx].item()) == 0.0


def test_apoptosis_pulse_drops_with_prune_node():
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    with torch.no_grad():
        layer.apoptosis_pulse[2] = 0.7
    layer.prune_node(0)
    # After dropping idx=0, the original idx=2 becomes idx=1 — pulse=0.7
    # should have moved with it.
    assert layer.apoptosis_pulse.shape == (3,)
    assert math.isclose(
        float(layer.apoptosis_pulse[1].item()), 0.7, abs_tol=1e-7,
    )


def test_ewc_penalty_scales_with_apoptosis_pulse():
    """ewc_penalty uses lambda * (1 - pulse) as effective stiffness."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    # Drift weights so penalty is nonzero.
    with torch.no_grad():
        layer.W.data += 0.5
        layer.lam.fill_(1.0)
    pen_baseline = float(layer.ewc_penalty().item())
    with torch.no_grad():
        layer.apoptosis_pulse.fill_(0.5)  # halve effective lambda everywhere
    pen_halved = float(layer.ewc_penalty().item())
    assert math.isclose(pen_halved, 0.5 * pen_baseline, rel_tol=1e-5), (
        f"pulse=0.5 should halve penalty; baseline={pen_baseline} "
        f"halved={pen_halved}"
    )
    # pulse=1.0 → effective lambda = 0 → penalty = 0
    with torch.no_grad():
        layer.apoptosis_pulse.fill_(1.0)
    pen_zero = float(layer.ewc_penalty().item())
    assert math.isclose(pen_zero, 0.0, abs_tol=1e-6)


def test_ewc_penalty_pulse_above_one_clamps_to_zero():
    """A pulse value > 1 (shouldn't happen but defensively) clamps to 0
    effective lambda, never producing negative penalties."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    with torch.no_grad():
        layer.W.data += 0.5
        layer.lam.fill_(1.0)
        layer.apoptosis_pulse.fill_(1.5)
    pen = float(layer.ewc_penalty().item())
    assert pen >= 0.0
    assert math.isclose(pen, 0.0, abs_tol=1e-6)


def test_apoptosis_redistribute_uniform_transfer():
    """Dead cell's outgoing column on L+1 splits uniformly across all
    surviving (non-latched) peers; dead cell's outgoing zeroed."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    nxt = net.layers[1]
    # Plant a known outgoing column on the dead-cell-to-be (idx=2).
    with torch.no_grad():
        for k in range(nxt.fan_in):
            nxt.W.data[:, k] = 1.0
            nxt.W_anchor[:, k] = 1.0
            nxt.fisher_W[:, k] = 0.0
        nxt.W.data[:, 2] = 4.0
        nxt.W_anchor[:, 2] = 4.0
        nxt.fisher_W[:, 2] = 0.4
    # All four nodes alive → 3 survivors share 4/3 each.
    n_survivors = apoptosis_redistribute(net, layer_idx=0, dead_idx=2)
    assert n_survivors == 3
    expected = 1.0 + 4.0 / 3.0
    for k in (0, 1, 3):
        assert torch.allclose(
            nxt.W.data[:, k],
            torch.full_like(nxt.W.data[:, k], expected),
        ), f"peer {k} did not absorb 4/3 share: {nxt.W.data[:, k]}"
        assert torch.allclose(
            nxt.W_anchor[:, k],
            torch.full_like(nxt.W_anchor[:, k], expected),
        )
    assert torch.allclose(nxt.W.data[:, 2], torch.zeros(4))
    assert torch.allclose(nxt.W_anchor[:, 2], torch.zeros(4))
    assert torch.allclose(nxt.fisher_W[:, 2], torch.zeros(4))


def test_apoptosis_redistribute_excludes_already_latched():
    """Already-latched peers don't get a share."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    nxt = net.layers[1]
    with torch.no_grad():
        layer.routing_latched[1] = True   # peer 1 already dead
        nxt.W.data[:, 2] = 6.0
        nxt.W.data[:, 0] = 0.0
        nxt.W.data[:, 1] = 0.0   # already dead, should stay 0
        nxt.W.data[:, 3] = 0.0
    n_survivors = apoptosis_redistribute(net, layer_idx=0, dead_idx=2)
    # Survivors: nodes 0 and 3 (node 1 is latched, node 2 is the dead cell)
    assert n_survivors == 2
    expected = 6.0 / 2
    assert torch.allclose(nxt.W.data[:, 0],
                          torch.full_like(nxt.W.data[:, 0], expected))
    assert torch.allclose(nxt.W.data[:, 3],
                          torch.full_like(nxt.W.data[:, 3], expected))
    assert torch.allclose(nxt.W.data[:, 1], torch.zeros(4)), (
        "already-latched peer must NOT receive a share"
    )


def test_apoptosis_redistribute_output_layer_is_noop():
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    out_layer_idx = len(net.layers) - 1
    n_survivors = apoptosis_redistribute(net, out_layer_idx, dead_idx=0)
    assert n_survivors == 0


def test_apoptosis_redistribute_no_survivors_returns_zero():
    """All-but-the-dead-cell already latched → no-op (returns 0)."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    nxt = net.layers[1]
    with torch.no_grad():
        for k in (0, 1, 3):
            layer.routing_latched[k] = True
        nxt.W.data[:, 2] = 5.0
    n = apoptosis_redistribute(net, layer_idx=0, dead_idx=2)
    assert n == 0
    # Dead cell's outgoing preserved (no peers to absorb it).
    assert torch.allclose(nxt.W.data[:, 2],
                          torch.full_like(nxt.W.data[:, 2], 5.0))


def test_apoptosis_spike_raises_pulse_on_survivors():
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    n = apoptosis_spike(net, layer_idx=0, dead_idx=2, spike_init=0.8)
    assert n == 3
    for k in (0, 1, 3):
        assert math.isclose(
            float(layer.apoptosis_pulse[k].item()), 0.8, abs_tol=1e-6,
        )
    # Dead cell's own pulse untouched (it's about to be irrelevant anyway).
    assert float(layer.apoptosis_pulse[2].item()) == 0.0


def test_apoptosis_spike_uses_max_with_existing_pulse():
    """Repeated deaths don't drive a peer's pulse above the spike value."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    with torch.no_grad():
        layer.apoptosis_pulse[0] = 0.9   # already higher than the spike
    apoptosis_spike(net, 0, dead_idx=2, spike_init=0.5)
    assert math.isclose(
        float(layer.apoptosis_pulse[0].item()), 0.9, abs_tol=1e-6,
    )
    # A new HIGHER spike raises the existing pulse to the new value.
    apoptosis_spike(net, 0, dead_idx=2, spike_init=0.95)
    assert math.isclose(
        float(layer.apoptosis_pulse[0].item()), 0.95, abs_tol=1e-6,
    )


def test_apoptosis_spike_validates_init_range():
    net = _make_net(hidden=4, fan_in=6, latent=2)
    for bad in (-0.1, 1.1, 2.0):
        try:
            apoptosis_spike(net, 0, dead_idx=0, spike_init=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for spike_init={bad}")


def test_apoptosis_decay_multiplies_all_layers():
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    with torch.no_grad():
        for layer in net.layers:
            layer.apoptosis_pulse.fill_(0.8)
    apoptosis_decay(net, decay_rate=0.5)
    for layer in net.layers:
        assert torch.allclose(
            layer.apoptosis_pulse,
            torch.full_like(layer.apoptosis_pulse, 0.4),
        )


def test_apoptosis_decay_validates_rate_range():
    net = _make_net(hidden=4, fan_in=6, latent=2)
    for bad in (-0.1, 1.1):
        try:
            apoptosis_decay(net, decay_rate=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for decay_rate={bad}")


def test_routing_starve_apoptosis_off_no_side_effects():
    """When apoptosis_on=False (default), latching has no effect on
    L+1 outgoing or apoptosis_pulse."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    nxt = net.layers[1]
    with torch.no_grad():
        layer.routing_scale[2] = 0.005
        nxt.W.data[:, 2] = 7.0
    routing_starve(
        net, 0, victim_idx=2, alpha=0.5, floor=0.01,
    )
    # Latched, but no apoptosis fired.
    assert bool(layer.routing_latched[2].item())
    assert torch.allclose(layer.apoptosis_pulse,
                          torch.zeros(layer.n_nodes))
    assert torch.allclose(nxt.W.data[:, 2],
                          torch.full_like(nxt.W.data[:, 2], 7.0))


def test_routing_starve_apoptosis_on_fires_on_latch_only():
    """apoptosis_on=True: non-latching ramp does NOT fire apoptosis;
    only the latch-transition step does."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    nxt = net.layers[1]
    with torch.no_grad():
        nxt.W.data[:, 2] = 8.0
    # First call: scale 1.0 → 0.5, no latch. No apoptosis side effects.
    routing_starve(
        net, 0, victim_idx=2, alpha=0.5, floor=1e-3,
        apoptosis_on=True, apoptosis_spike_init=0.8,
    )
    assert not bool(layer.routing_latched[2].item())
    assert torch.allclose(layer.apoptosis_pulse,
                          torch.zeros(layer.n_nodes))
    assert torch.allclose(nxt.W.data[:, 2],
                          torch.full_like(nxt.W.data[:, 2], 8.0))


def test_routing_starve_apoptosis_on_fires_redistribute_and_spike():
    """When the ramp crosses the floor, latch fires both A and C."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    layer = net.layers[0]
    nxt = net.layers[1]
    with torch.no_grad():
        layer.routing_scale[2] = 0.005   # next ramp will cross 0.01 floor
        nxt.W.data[:, 2] = 9.0
        nxt.W.data[:, 0] = 0.0
        nxt.W.data[:, 1] = 0.0
        nxt.W.data[:, 3] = 0.0
    scale_after, latched_now = routing_starve(
        net, 0, victim_idx=2, alpha=0.5, floor=0.01,
        apoptosis_on=True, apoptosis_spike_init=0.8,
    )
    assert latched_now is True
    # Mechanism A: surviving peers spiked.
    for k in (0, 1, 3):
        assert math.isclose(
            float(layer.apoptosis_pulse[k].item()), 0.8, abs_tol=1e-6,
        )
    # Mechanism C: dead cell's outgoing redistributed uniformly.
    expected = 9.0 / 3
    for k in (0, 1, 3):
        assert torch.allclose(
            nxt.W.data[:, k],
            torch.full_like(nxt.W.data[:, k], expected),
        )
    assert torch.allclose(nxt.W.data[:, 2], torch.zeros(4))


def test_dreaming_block_apoptosis_decay_runs_each_block():
    """Even when nothing latches this block, the pulse decays."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    with torch.no_grad():
        net.layers[0].apoptosis_pulse.fill_(0.8)
    pair_fn = _make_pair_fn(state_dim=6, rng_seed=2)
    dreaming_block(
        net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
        past_pair_names=["alpha", "beta"], replay_fraction=1.0,
        replay_steps_per_pair=1, replay_batch=4, ewc_strength=0.0,
        cos_threshold=0.95, u_threshold=0.0, rng=random.Random(0),
        compression_action="merge",
        apoptosis_on=True, apoptosis_decay_rate=0.5,
    )
    assert torch.allclose(
        net.layers[0].apoptosis_pulse,
        torch.full_like(net.layers[0].apoptosis_pulse, 0.4),
    )


def test_dreaming_block_apoptosis_off_skips_decay():
    """apoptosis_on=False leaves pulse untouched (even if non-zero)."""
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    with torch.no_grad():
        net.layers[0].apoptosis_pulse.fill_(0.8)
    pair_fn = _make_pair_fn(state_dim=6, rng_seed=2)
    dreaming_block(
        net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
        past_pair_names=["alpha", "beta"], replay_fraction=1.0,
        replay_steps_per_pair=1, replay_batch=4, ewc_strength=0.0,
        cos_threshold=0.95, u_threshold=0.0, rng=random.Random(0),
        compression_action="merge",
        apoptosis_on=False, apoptosis_decay_rate=0.5,
    )
    assert torch.allclose(
        net.layers[0].apoptosis_pulse,
        torch.full_like(net.layers[0].apoptosis_pulse, 0.8),
    )


def test_dreaming_block_consolidate_false_skips_compress_and_purge():
    """consolidate=False: even with a planted duplicate, no merges/purges fire.
    Replay still runs (loss probe present)."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)
    pair_fn = _make_pair_fn(state_dim=6, rng_seed=2)
    n_before = net.n_parameters()

    rep = dreaming_block(
        net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
        past_pair_names=["alpha", "beta"], replay_fraction=1.0,
        replay_steps_per_pair=2, replay_batch=4, ewc_strength=0.0,
        cos_threshold=0.95, u_threshold=0.0,
        rng=random.Random(0),
        compression_action="merge",
        consolidate=False,
    )
    assert rep.merges == [], (
        f"consolidate=False must produce no merges, got {len(rep.merges)}"
    )
    assert rep.purges == []
    assert rep.n_params_after == n_before, (
        "consolidate=False must leave param count unchanged"
    )
    # Replay still runs: pair count should equal what we asked for.
    assert rep.replay_pairs_sampled == 2
    assert rep.replay_steps == 2 * 2


def test_dreaming_block_consolidate_true_default_unchanged():
    """consolidate=True (default) keeps prior behavior — planted duplicate
    still produces a merge."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)
    pair_fn = _make_pair_fn(state_dim=6, rng_seed=2)
    rep = dreaming_block(
        net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
        past_pair_names=["alpha", "beta"], replay_fraction=1.0,
        replay_steps_per_pair=2, replay_batch=4, ewc_strength=0.0,
        cos_threshold=0.95, u_threshold=1e-3, rng=random.Random(0),
        compression_action="merge",
        # consolidate omitted → True
    )
    assert len(rep.merges) >= 1


def test_dreaming_block_consolidate_false_starve_no_routing_change():
    """consolidate=False with starve action: routing_scale stays at 1.0 even
    on a planted activation-correlated pair."""
    torch.manual_seed(0)
    net = _make_net(hidden=6, fan_in=6, latent=2)
    _plant_duplicate(net, layer_idx=0, src=0, dst=4)
    pair_fn = _make_pair_fn(state_dim=6, rng_seed=2)
    rep = dreaming_block(
        net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
        past_pair_names=["alpha", "beta"], replay_fraction=1.0,
        replay_steps_per_pair=2, replay_batch=4, ewc_strength=0.0,
        cos_threshold=0.95, u_threshold=0.0, rng=random.Random(0),
        compression_action="starve",
        starvation_alpha=0.5, starvation_floor=1e-3,
        consolidate=False,
    )
    assert rep.merges == []
    layer = net.layers[0]
    assert torch.allclose(
        layer.routing_scale, torch.ones(layer.n_nodes)
    ), "routing_scale must stay at 1.0 when consolidate=False"


def test_dreaming_block_starve_invalid_alpha_raises():
    torch.manual_seed(0)
    net = _make_net(hidden=4, fan_in=6, latent=2)
    pair_fn = _make_pair_fn(state_dim=6)
    try:
        dreaming_block(
            net, sample_pair_fn=pair_fn, loss_fn=_contrastive_loss,
            past_pair_names=["alpha"], replay_steps_per_pair=1,
            replay_batch=4, compression_action="starve",
            starvation_alpha=1.5,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for starvation_alpha=1.5")


# --------------------------------------------------------------------------- #
# Runner                                                                      #
# --------------------------------------------------------------------------- #


def main() -> int:
    print("test_dreaming.py")
    print("-" * 60)
    _run("find_redundant_pairs_empty_when_no_duplicates",
         test_find_redundant_pairs_empty_when_no_duplicates)
    _run("find_redundant_pairs_finds_planted_duplicate",
         test_find_redundant_pairs_finds_planted_duplicate)
    _run("find_redundant_pairs_threshold_excludes_low_sim",
         test_find_redundant_pairs_threshold_excludes_low_sim)
    _run("merge_preserves_forward_when_duplicate",
         test_merge_preserves_forward_when_duplicate)
    _run("merge_averages_per_node_state", test_merge_averages_per_node_state)
    _run("merge_sums_next_layer_outgoing", test_merge_sums_next_layer_outgoing)
    _run("merge_drops_victim_and_input_column",
         test_merge_drops_victim_and_input_column)
    _run("merge_rejects_self_or_oob", test_merge_rejects_self_or_oob)
    _run("compress_reduces_n_nodes_with_planted_duplicate",
         test_compress_reduces_n_nodes_with_planted_duplicate)
    _run("compress_no_op_when_no_redundancy",
         test_compress_no_op_when_no_redundancy)
    _run("compress_skips_output_layer_by_default",
         test_compress_skips_output_layer_by_default)
    _run("compress_includes_output_when_explicitly_listed",
         test_compress_includes_output_when_explicitly_listed)
    _run("replay_handles_empty_past_pairs",
         test_replay_handles_empty_past_pairs)
    _run("replay_decreases_loss_on_familiar_pairs",
         test_replay_decreases_loss_on_familiar_pairs)
    _run("purge_drops_low_utility_nodes", test_purge_drops_low_utility_nodes)
    _run("purge_no_op_when_all_above_threshold",
         test_purge_no_op_when_all_above_threshold)
    _run("purge_skips_output_layer_by_default",
         test_purge_skips_output_layer_by_default)
    _run("purge_never_drops_last_node", test_purge_never_drops_last_node)
    _run("max_off_diag_cosine_random_init_is_low",
         test_max_off_diag_cosine_random_init_is_low)
    _run("max_off_diag_cosine_planted_duplicate_is_one",
         test_max_off_diag_cosine_planted_duplicate_is_one)
    _run("max_off_diag_cosine_handles_singleton_layer",
         test_max_off_diag_cosine_handles_singleton_layer)
    _run("dreaming_report_pre_compress_max_cosines_populated",
         test_dreaming_report_pre_compress_max_cosines_populated)
    _run("dreaming_report_probe_includes_output_when_unskipped",
         test_dreaming_report_probe_includes_output_when_unskipped)
    _run("dreaming_block_runs_and_returns_report",
         test_dreaming_block_runs_and_returns_report)
    _run("find_activation_pairs_empty_when_no_correlation",
         test_find_activation_pairs_empty_when_no_correlation)
    _run("find_activation_pairs_finds_planted_duplicate",
         test_find_activation_pairs_finds_planted_duplicate)
    _run("find_activation_pairs_orthogonal_W_correlated_activations",
         test_find_activation_pairs_orthogonal_W_correlated_activations)
    _run("max_off_diag_activation_cosine_handles_singleton_layer",
         test_max_off_diag_activation_cosine_handles_singleton_layer)
    _run("compress_activation_signal_requires_probe_batch",
         test_compress_activation_signal_requires_probe_batch)
    _run("compress_rejects_unknown_signal",
         test_compress_rejects_unknown_signal)
    _run("compress_activation_merges_planted_duplicate",
         test_compress_activation_merges_planted_duplicate)
    _run("dreaming_block_activation_signal_populates_probe_field",
         test_dreaming_block_activation_signal_populates_probe_field)
    _run("dreaming_block_activation_falls_back_when_no_past_pairs",
         test_dreaming_block_activation_falls_back_when_no_past_pairs)
    _run("dreaming_block_rejects_unknown_signal",
         test_dreaming_block_rejects_unknown_signal)
    _run("downscale_preserves_victim_substrate_at_layer_L",
         test_downscale_preserves_victim_substrate_at_layer_L)
    _run("downscale_zeros_victim_outgoing_at_next_layer",
         test_downscale_zeros_victim_outgoing_at_next_layer)
    _run("downscale_peer_absorbs_victim_outgoing",
         test_downscale_peer_absorbs_victim_outgoing)
    _run("downscale_self_or_oob_rejected",
         test_downscale_self_or_oob_rejected)
    _run("downscale_on_output_layer_is_noop",
         test_downscale_on_output_layer_is_noop)
    _run("downscale_victim_can_be_re_recruited",
         test_downscale_victim_can_be_re_recruited)
    _run("compress_downscale_with_planted_pair_records_action",
         test_compress_downscale_with_planted_pair_records_action)
    _run("compress_downscale_terminates_on_static_victim",
         test_compress_downscale_terminates_on_static_victim)
    _run("compress_downscale_skips_output_layer_implicitly",
         test_compress_downscale_skips_output_layer_implicitly)
    _run("compress_rejects_unknown_action",
         test_compress_rejects_unknown_action)
    _run("dreaming_block_downscale_preserves_param_count",
         test_dreaming_block_downscale_preserves_param_count)
    _run("dreaming_block_rejects_unknown_action",
         test_dreaming_block_rejects_unknown_action)
    _run("compress_downscale_cap_one_stops_after_first_event",
         test_compress_downscale_cap_one_stops_after_first_event)
    _run("compress_downscale_cap_zero_blocks_all_events",
         test_compress_downscale_cap_zero_blocks_all_events)
    _run("compress_downscale_cap_none_preserves_uncapped_behavior",
         test_compress_downscale_cap_none_preserves_uncapped_behavior)
    _run("compress_downscale_cap_is_per_layer_not_global",
         test_compress_downscale_cap_is_per_layer_not_global)
    _run("compress_downscale_cap_does_not_affect_merge_action",
         test_compress_downscale_cap_does_not_affect_merge_action)
    _run("compress_downscale_cap_negative_raises",
         test_compress_downscale_cap_negative_raises)
    _run("dreaming_block_threads_max_downscales_per_layer",
         test_dreaming_block_threads_max_downscales_per_layer)
    # Phase 4.5 Experiment 3 — routing starvation.
    _run("routing_scale_buffer_exists_and_init_one",
         test_routing_scale_buffer_exists_and_init_one)
    _run("forward_with_routing_scale_one_matches_unscaled",
         test_forward_with_routing_scale_one_matches_unscaled)
    _run("forward_attenuates_when_routing_scale_zero",
         test_forward_attenuates_when_routing_scale_zero)
    _run("grow_node_records_task_of_origin_and_routing_scale",
         test_grow_node_records_task_of_origin_and_routing_scale)
    _run("prune_node_drops_routing_buffers",
         test_prune_node_drops_routing_buffers)
    _run("routing_starve_multiplies_scale_and_returns_value",
         test_routing_starve_multiplies_scale_and_returns_value)
    _run("routing_starve_does_not_touch_peer_or_other_layers",
         test_routing_starve_does_not_touch_peer_or_other_layers)
    _run("routing_starve_latches_below_floor",
         test_routing_starve_latches_below_floor)
    _run("routing_starve_already_latched_is_noop",
         test_routing_starve_already_latched_is_noop)
    _run("routing_starve_validates_alpha_and_floor",
         test_routing_starve_validates_alpha_and_floor)
    _run("routing_regrow_multiplies_by_inv_alpha_capped_at_one",
         test_routing_regrow_multiplies_by_inv_alpha_capped_at_one)
    _run("routing_regrow_skips_latched",
         test_routing_regrow_skips_latched)
    _run("pick_starvation_victim_older_keeps",
         test_pick_starvation_victim_older_keeps)
    _run("pick_starvation_victim_tiebreak_outgoing_norm",
         test_pick_starvation_victim_tiebreak_outgoing_norm)
    _run("pick_starvation_victim_tied_falls_back_to_index",
         test_pick_starvation_victim_tied_falls_back_to_index)
    _run("compress_starve_picks_younger_victim_and_records_event",
         test_compress_starve_picks_younger_victim_and_records_event)
    _run("compress_starve_caps_events_per_layer",
         test_compress_starve_caps_events_per_layer)
    _run("compress_starve_regrows_non_victim_below_one",
         test_compress_starve_regrows_non_victim_below_one)
    _run("compress_starve_latched_does_not_regrow",
         test_compress_starve_latched_does_not_regrow)
    _run("compress_starve_repeated_calls_compound_to_latch",
         test_compress_starve_repeated_calls_compound_to_latch)
    _run("compress_starve_validates_alpha_and_floor",
         test_compress_starve_validates_alpha_and_floor)
    _run("compress_starve_skips_pair_when_both_latched",
         test_compress_starve_skips_pair_when_both_latched)
    _run("compress_starve_one_latched_forces_other_as_victim",
         test_compress_starve_one_latched_forces_other_as_victim)
    _run("compress_starve_preserves_param_count",
         test_compress_starve_preserves_param_count)
    _run("dreaming_block_starve_threads_starvation_params",
         test_dreaming_block_starve_threads_starvation_params)
    _run("dreaming_block_consolidate_false_skips_compress_and_purge",
         test_dreaming_block_consolidate_false_skips_compress_and_purge)
    _run("dreaming_block_consolidate_true_default_unchanged",
         test_dreaming_block_consolidate_true_default_unchanged)
    _run("dreaming_block_consolidate_false_starve_no_routing_change",
         test_dreaming_block_consolidate_false_starve_no_routing_change)
    _run("dreaming_block_starve_invalid_alpha_raises",
         test_dreaming_block_starve_invalid_alpha_raises)
    # Phase 4.5 Experiment 5 — apoptosis spike.
    _run("apoptosis_pulse_buffer_init_zero",
         test_apoptosis_pulse_buffer_init_zero)
    _run("apoptosis_pulse_extends_with_grow_node",
         test_apoptosis_pulse_extends_with_grow_node)
    _run("apoptosis_pulse_drops_with_prune_node",
         test_apoptosis_pulse_drops_with_prune_node)
    _run("ewc_penalty_scales_with_apoptosis_pulse",
         test_ewc_penalty_scales_with_apoptosis_pulse)
    _run("ewc_penalty_pulse_above_one_clamps_to_zero",
         test_ewc_penalty_pulse_above_one_clamps_to_zero)
    _run("apoptosis_redistribute_uniform_transfer",
         test_apoptosis_redistribute_uniform_transfer)
    _run("apoptosis_redistribute_excludes_already_latched",
         test_apoptosis_redistribute_excludes_already_latched)
    _run("apoptosis_redistribute_output_layer_is_noop",
         test_apoptosis_redistribute_output_layer_is_noop)
    _run("apoptosis_redistribute_no_survivors_returns_zero",
         test_apoptosis_redistribute_no_survivors_returns_zero)
    _run("apoptosis_spike_raises_pulse_on_survivors",
         test_apoptosis_spike_raises_pulse_on_survivors)
    _run("apoptosis_spike_uses_max_with_existing_pulse",
         test_apoptosis_spike_uses_max_with_existing_pulse)
    _run("apoptosis_spike_validates_init_range",
         test_apoptosis_spike_validates_init_range)
    _run("apoptosis_decay_multiplies_all_layers",
         test_apoptosis_decay_multiplies_all_layers)
    _run("apoptosis_decay_validates_rate_range",
         test_apoptosis_decay_validates_rate_range)
    _run("routing_starve_apoptosis_off_no_side_effects",
         test_routing_starve_apoptosis_off_no_side_effects)
    _run("routing_starve_apoptosis_on_fires_on_latch_only",
         test_routing_starve_apoptosis_on_fires_on_latch_only)
    _run("routing_starve_apoptosis_on_fires_redistribute_and_spike",
         test_routing_starve_apoptosis_on_fires_redistribute_and_spike)
    _run("dreaming_block_apoptosis_decay_runs_each_block",
         test_dreaming_block_apoptosis_decay_runs_each_block)
    _run("dreaming_block_apoptosis_off_skips_decay",
         test_dreaming_block_apoptosis_off_skips_decay)
    print("-" * 60)
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print(f"  {n_pass} passed, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
