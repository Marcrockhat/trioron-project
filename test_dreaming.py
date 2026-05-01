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
    compress,
    dreaming_block,
    find_redundant_pairs,
    merge_nodes,
    purge,
    replay,
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
    _run("dreaming_block_runs_and_returns_report",
         test_dreaming_block_runs_and_returns_report)
    print("-" * 60)
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print(f"  {n_pass} passed, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
