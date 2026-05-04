"""Phase 4.5: the Dreaming Phase — offline consolidation, compression, purge.

Implements three offline mechanisms that run between active training tasks
on a grown TrioronNetwork. The architectural bet (per
phase_4_5_dreaming_phase.md) is that growth + dreaming gives a structural
advantage over freeze-based CL methods (HAT, PackNet) whose per-task
frozen units are inviolable. Without dreaming, growth pays the bloat tax
without collecting the compaction reward.

Three mechanisms, in order:

  1. Generative replay (`replay`):
     Rehearses past pairs with EWC active, frustration off. Substitutes
     `train_cur.sample_pair(name, batch=...)` for the spec's "Teacher LLM
     low-fidelity stimulation". Keeps foundational pathways warm and
     prevents λ from drifting.

  2. Topological compression (`compress`):
     Identifies redundant nodes via cosine similarity of W_anchor rows
     (post-consolidation — captures the committed function, not transient
     state). Pairs above `cos_threshold` are merged: incoming-mean,
     outgoing-sum (function-preserving on the linear part), bias/anchor/
     λ/Fisher averaged. The peer survives, the victim is pruned.

  3. VRAM purge (`purge`):
     Drops nodes whose utility score is below `u_threshold`. Reuses the
     existing prune_layer_node primitive (which handles cross-layer
     fan_in cleanup). Distinct from the in-training PruningController:
     this is a structural sweep at task end, not an event-clock check
     during training.

All three modify the network in place. After any merge or purge, the
optimizer must be rebuilt by the caller — TrioronLayer's W and b
Parameter objects are replaced by prune_node.

Function-preservation note for `compress`: the merge math is exact on
the linear pre-activation only when w_i == w_j and b_i == b_j. At
cos_threshold=0.95 the activation a_merged ≈ a_i ≈ a_j only
approximately; the next layer's output drift is bounded by
(1 - cos_sim) plus whatever magnitude difference the two nodes carry.
This is the source of the compression's residual error — replay
afterwards cleans up most of it.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.optim as optim

from .network import TrioronNetwork


@dataclass
class MergeEvent:
    layer_idx: int
    peer_idx: int
    victim_idx: int
    cos_sim: float
    arch_after: Tuple[int, ...]
    # "merge" (legacy: incoming-mean + outgoing-sum + delete victim),
    # "downscale" (synaptic downscale: peer.outgoing += victim.outgoing,
    # victim.outgoing = 0, victim's substrate at layer L preserved), or
    # "starve" (routing starvation: victim.routing_scale *= alpha, peer
    # untouched). The field is named `action` not `mechanism` because we
    # expect more to be added as the dreaming-phase mechanism evolves.
    # Defaults to "merge" so existing tests / callers see no change.
    action: str = "merge"
    # Routing-starvation only: the routing_scale on the victim AFTER the
    # ramp (informational). NaN for non-starve actions.
    victim_routing_scale_after: float = float("nan")
    # Routing-starvation only: True when this event pushed the victim's
    # routing_scale below the starvation floor and latched it to 0.
    victim_latched: bool = False


@dataclass
class PurgeEvent:
    layer_idx: int
    node_idx: int
    u_at_purge: float
    arch_after: Tuple[int, ...]


@dataclass
class DreamingReport:
    replay_loss_before: float
    replay_loss_after: float
    replay_pairs_sampled: int
    replay_steps: int
    merges: List[MergeEvent]
    purges: List[PurgeEvent]
    n_params_before: int
    n_params_after: int
    # Per-layer (layer_idx, max off-diagonal W_anchor cosine) measured
    # BEFORE compress runs. Lets the caller see what threshold would
    # have fired without lowering the actual cos_threshold. The probe
    # respects skip_output_layer the same way compress does.
    pre_compress_max_cosines: List[Tuple[int, float]]
    # Activation-correlation analog of `pre_compress_max_cosines`:
    # max off-diagonal Pearson cosine of post-activation outputs across
    # the probe batch. Empty list when the activation signal isn't in
    # use (redundancy_signal == "weight" or no probe batch).
    pre_compress_max_activation_cosines: List[Tuple[int, float]] = (
        None  # type: ignore[assignment]
    )

    def __post_init__(self) -> None:
        if self.pre_compress_max_activation_cosines is None:
            self.pre_compress_max_activation_cosines = []


# ---------------------------------------------------------------------
# Mechanism 1 — replay
# ---------------------------------------------------------------------


def replay(
    net: TrioronNetwork,
    sample_pair_fn: Callable[[str, int], Tuple[torch.Tensor, torch.Tensor]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    past_pair_names: Sequence[str],
    *,
    fraction: float = 0.25,
    n_steps_per_pair: int = 200,
    batch: int = 32,
    ewc_strength: float = 1000.0,
    lr: float = 3e-3,
    rng: Optional[random.Random] = None,
) -> Tuple[float, float, int, int]:
    """Run K rehearsal steps on a random subset of past pairs.

    Frustration is intentionally not threaded through — the spec disables
    it inside dreaming blocks. EWC is on (`ewc_strength` > 0) so the
    rehearsal pulls the weights toward their consolidated state rather
    than away from it.

    Returns:
        (avg_loss_before, avg_loss_after, n_pairs_sampled, total_steps)

    avg_loss_before/after are the mean contrastive loss across the
    sampled subset, before and after the replay block. A no-op-on-noise
    sanity check: avg_loss_after should be <= avg_loss_before within
    the noise of the per-batch sampling.
    """
    if rng is None:
        rng = random.Random()
    if not past_pair_names:
        return (0.0, 0.0, 0, 0)

    k = max(1, int(math.ceil(fraction * len(past_pair_names))))
    sampled = rng.sample(list(past_pair_names), k=min(k, len(past_pair_names)))

    def _avg_loss() -> float:
        net.eval()
        total = 0.0
        with torch.no_grad():
            for name in sampled:
                a, b = sample_pair_fn(name, batch)
                total += float(loss_fn(net(a), net(b)).item())
        net.train()
        return total / max(len(sampled), 1)

    loss_before = _avg_loss()

    opt = optim.Adam(net.parameters(), lr=lr)
    total_steps = 0
    for name in sampled:
        for _ in range(n_steps_per_pair):
            a, b = sample_pair_fn(name, batch)
            l_task = loss_fn(net(a), net(b))
            l = l_task + ewc_strength * net.ewc_penalty() if ewc_strength > 0 else l_task
            opt.zero_grad()
            l.backward()
            opt.step()
            total_steps += 1

    loss_after = _avg_loss()
    return (loss_before, loss_after, len(sampled), total_steps)


# ---------------------------------------------------------------------
# Mechanism 2 — topological compression
# ---------------------------------------------------------------------


def max_off_diag_cosine(net: TrioronNetwork, layer_idx: int) -> float:
    """Return the maximum off-diagonal W_anchor cosine similarity within a
    layer. Returns -inf if the layer has fewer than 2 nodes (no pair).

    Cheap diagnostic — used by callers to probe what compress threshold
    WOULD have fired without actually changing cos_threshold.
    """
    layer = net.layers[layer_idx]
    A = layer.W_anchor.detach()  # (n_nodes, fan_in)
    n = A.shape[0]
    if n < 2:
        return float("-inf")
    norms = A.norm(dim=1).clamp_min(1e-12)
    A_n = A / norms.unsqueeze(1)
    sim = A_n @ A_n.T
    sim.fill_diagonal_(float("-inf"))
    return float(sim.max().item())


def find_redundant_pairs(
    net: TrioronNetwork,
    layer_idx: int,
    *,
    cos_threshold: float = 0.95,
) -> List[Tuple[int, int, float]]:
    """Find node pairs whose W_anchor rows have cosine similarity >= threshold.

    Returns a list of (i, j, sim) with i < j, sorted by similarity desc.
    Only pairs strictly above threshold are returned.
    """
    layer = net.layers[layer_idx]
    A = layer.W_anchor.detach()  # (n_nodes, fan_in)
    n = A.shape[0]
    if n < 2:
        return []
    norms = A.norm(dim=1).clamp_min(1e-12)
    A_n = A / norms.unsqueeze(1)
    sim = A_n @ A_n.T  # (n, n) cosine similarity matrix
    pairs: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j].item())
            if s >= cos_threshold:
                pairs.append((i, j, s))
    pairs.sort(key=lambda t: -t[2])
    return pairs


# ---- activation-correlation redundancy detector --------------------------
#
# Phase 4.5 redesign (per next_session_plan.md, dated 2026-05-01): the
# W_anchor cosine signal can't find redundancy in this architecture.
# grow_layer produces orthogonal latent dims by construction (PCA top-1
# of contrastive residuals, then EWC-pinned), and high hidden-layer
# W cosines are partial-direction-sharing rather than functional
# redundancy. The principled signal is "do these two nodes produce the
# same activation pattern across the data distribution" — Pearson
# (centered) cosine of post-activation column vectors over a probe batch.


def _layer_activations(
    net: TrioronNetwork, layer_idx: int, probe_batch: torch.Tensor,
) -> torch.Tensor:
    """Return the post-activation output of `layer_idx` on `probe_batch`.
    Shape (batch, n_nodes_at_layer_idx)."""
    if not (0 <= layer_idx < len(net.layers)):
        raise IndexError(
            f"layer_idx {layer_idx} out of range [0, {len(net.layers)})"
        )
    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            x = probe_batch
            for L_i, layer in enumerate(net.layers):
                x = layer(x)
                if L_i == layer_idx:
                    return x.detach()
    finally:
        if was_training:
            net.train()
    raise RuntimeError("unreachable")  # pragma: no cover


def _activation_cosine_matrix(act: torch.Tensor) -> torch.Tensor:
    """Pearson-style cosine of activation columns: center per-node, then
    cosine-similarity column-wise. Returns (n_nodes, n_nodes).

    Two columns that are constant (zero variance) cosine to NaN under
    the centering — clamped to 0 here so they don't fire the threshold.
    """
    # act: (batch, n_nodes); we want similarity between columns (nodes),
    # so transpose: cols become rows of a (n_nodes, batch) matrix.
    A = act.t().contiguous()  # (n_nodes, batch)
    A = A - A.mean(dim=1, keepdim=True)  # center per node
    norms = A.norm(dim=1).clamp_min(1e-12)
    A_n = A / norms.unsqueeze(1)
    sim = A_n @ A_n.t()  # (n_nodes, n_nodes)
    sim = torch.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    return sim


def max_off_diag_activation_cosine(
    net: TrioronNetwork, layer_idx: int, probe_batch: torch.Tensor,
) -> float:
    """Probe analog of `max_off_diag_cosine` using activation-correlation.
    Returns -inf if the layer has fewer than 2 nodes.
    """
    layer = net.layers[layer_idx]
    if layer.n_nodes < 2:
        return float("-inf")
    act = _layer_activations(net, layer_idx, probe_batch)
    sim = _activation_cosine_matrix(act)
    sim.fill_diagonal_(float("-inf"))
    return float(sim.max().item())


def find_activation_redundant_pairs(
    net: TrioronNetwork,
    layer_idx: int,
    *,
    probe_batch: torch.Tensor,
    ac_threshold: float = 0.95,
) -> List[Tuple[int, int, float]]:
    """Find node pairs whose activation patterns across `probe_batch` are
    Pearson-cosine-similar at or above `ac_threshold`.

    Returns (i, j, sim) with i < j, sorted by similarity descending. Only
    pairs strictly at/above threshold are returned.
    """
    layer = net.layers[layer_idx]
    if layer.n_nodes < 2:
        return []
    act = _layer_activations(net, layer_idx, probe_batch)
    sim = _activation_cosine_matrix(act)
    n = sim.shape[0]
    pairs: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j].item())
            if s >= ac_threshold:
                pairs.append((i, j, s))
    pairs.sort(key=lambda t: -t[2])
    return pairs


def merge_nodes(
    net: TrioronNetwork,
    layer_idx: int,
    peer_idx: int,
    victim_idx: int,
) -> None:
    """Merge `victim_idx` into `peer_idx` on `layer_idx`. Function-preserving
    on the linear pre-activation when peer.W ≈ victim.W; approximate
    otherwise.

    Operations:
      peer's incoming weights, anchor, bias, b_anchor, λ, fisher_W,
      fisher_b: averaged with victim's.
      peer's u: max(peer.u, victim.u) — utility tracks contribution; the
      merged node inherits the higher of the two.

      next layer's outgoing column at peer: SUM(peer_col, victim_col)
      on W, W_anchor, fisher_W. This preserves the next layer's input
      function under the assumption a_peer ≈ a_victim ≈ a_merged.

    Then victim is pruned (TrioronLayer.prune_node), and if a next layer
    exists, victim's input column is dropped (prune_input).

    Caller MUST rebuild the optimizer after this returns — W and b
    Parameter objects are replaced.
    """
    if peer_idx == victim_idx:
        raise ValueError("peer_idx == victim_idx")
    layer = net.layers[layer_idx]
    if not (0 <= peer_idx < layer.n_nodes and 0 <= victim_idx < layer.n_nodes):
        raise IndexError(
            f"peer/victim ({peer_idx},{victim_idx}) out of range "
            f"[0,{layer.n_nodes})"
        )
    if layer.n_nodes <= 1:
        raise ValueError(f"Cannot merge in layer {layer_idx} with n_nodes={layer.n_nodes}")

    # --- 1. average peer's per-node state with victim's ---
    with torch.no_grad():
        layer.W.data[peer_idx] = 0.5 * (layer.W.data[peer_idx] + layer.W.data[victim_idx])
        layer.b.data[peer_idx] = 0.5 * (layer.b.data[peer_idx] + layer.b.data[victim_idx])
        layer.W_anchor[peer_idx] = 0.5 * (
            layer.W_anchor[peer_idx] + layer.W_anchor[victim_idx]
        )
        layer.b_anchor[peer_idx] = 0.5 * (
            layer.b_anchor[peer_idx] + layer.b_anchor[victim_idx]
        )
        layer.lam[peer_idx] = 0.5 * (layer.lam[peer_idx] + layer.lam[victim_idx])
        layer.fisher_W[peer_idx] = 0.5 * (
            layer.fisher_W[peer_idx] + layer.fisher_W[victim_idx]
        )
        layer.fisher_b[peer_idx] = 0.5 * (
            layer.fisher_b[peer_idx] + layer.fisher_b[victim_idx]
        )
        layer.u[peer_idx] = torch.max(layer.u[peer_idx], layer.u[victim_idx])

    # --- 2. SUM victim's outgoing column into peer's on the next layer ---
    has_next = layer_idx + 1 < len(net.layers)
    if has_next:
        nxt = net.layers[layer_idx + 1]
        with torch.no_grad():
            nxt.W.data[:, peer_idx] = nxt.W.data[:, peer_idx] + nxt.W.data[:, victim_idx]
            nxt.W_anchor[:, peer_idx] = (
                nxt.W_anchor[:, peer_idx] + nxt.W_anchor[:, victim_idx]
            )
            nxt.fisher_W[:, peer_idx] = (
                nxt.fisher_W[:, peer_idx] + nxt.fisher_W[:, victim_idx]
            )

    # --- 3. drop the victim row + corresponding next-layer input column ---
    layer.prune_node(victim_idx)
    if has_next:
        net.layers[layer_idx + 1].prune_input(victim_idx)


def synaptic_downscale(
    net: TrioronNetwork,
    layer_idx: int,
    peer_idx: int,
    victim_idx: int,
) -> None:
    """Substrate-preserving compression: redirect victim's downstream
    contribution into peer, zero victim's outgoing, but DON'T touch
    victim's substrate at layer L.

    Phase 4.5 redesign (Rocky 2026-05-01): "brain cells are too precious
    to destroy." Biologically faithful — synaptic homeostasis re-weights
    connections, the cell body stays. The starved victim is functionally
    dormant but available for re-recruitment by future tasks (gradient
    descent on layer L+1's column at victim_idx is unconstrained because
    fisher_W there is now zero).

    Operations on the next layer (L+1):
      - W[:, peer]      += W[:, victim]      (peer takes over downstream load)
      - W_anchor[:, peer] += W_anchor[:, victim]
      - fisher_W[:, peer] += fisher_W[:, victim]   (peer carries the
        consolidation weight too — otherwise the post-downscale peer would
        look "fresh" to EWC and drift unprotected)
      - W[:, victim]      = 0  (victim no longer routes)
      - W_anchor[:, victim] = 0
      - fisher_W[:, victim] = 0  (so EWC doesn't pin zero in place,
        leaving the substrate available for re-recruitment)

    Layer L (the layer being compressed) is NOT touched. Victim's W_in,
    bias, anchor, lam, fisher all stay. Architecture (n_nodes per layer)
    is unchanged — VRAM does not free up immediately.

    The output layer has no L+1; downscale is a no-op there. (compress's
    skip_output_layer default still applies.)

    Caller does NOT need to rebuild the optimizer — no Parameter objects
    are replaced (only their .data is mutated).
    """
    if peer_idx == victim_idx:
        raise ValueError("peer_idx == victim_idx")
    layer = net.layers[layer_idx]
    if not (0 <= peer_idx < layer.n_nodes and 0 <= victim_idx < layer.n_nodes):
        raise IndexError(
            f"peer/victim ({peer_idx},{victim_idx}) out of range "
            f"[0,{layer.n_nodes})"
        )
    if layer_idx + 1 >= len(net.layers):
        # Output layer — no downstream to redirect. No-op (caller's
        # responsibility to skip; compress's skip_output_layer covers it
        # by default).
        return

    nxt = net.layers[layer_idx + 1]
    with torch.no_grad():
        nxt.W.data[:, peer_idx] = (
            nxt.W.data[:, peer_idx] + nxt.W.data[:, victim_idx]
        )
        nxt.W_anchor[:, peer_idx] = (
            nxt.W_anchor[:, peer_idx] + nxt.W_anchor[:, victim_idx]
        )
        nxt.fisher_W[:, peer_idx] = (
            nxt.fisher_W[:, peer_idx] + nxt.fisher_W[:, victim_idx]
        )
        nxt.W.data[:, victim_idx] = 0.0
        nxt.W_anchor[:, victim_idx] = 0.0
        nxt.fisher_W[:, victim_idx] = 0.0


def apoptosis_redistribute(
    net: TrioronNetwork,
    layer_idx: int,
    dead_idx: int,
) -> int:
    """Phase 4.5 Experiment 5 — outgoing-role transfer at the moment of
    cell death. Uniformly redistribute the dead cell's outgoing column
    on layer L+1 across all SURVIVING (non-latched) peers in layer L,
    then zero the dead cell's outgoing.

    Returns the number of surviving peers that received a share. 0
    means there were no survivors and the call was a no-op (the dead
    cell's outgoing is preserved so something still routes downstream).

    No-op when layer_idx is the output layer (no L+1 to redistribute
    into) or when there are no surviving peers (single-cell layers,
    or every peer already latched).

    Caller does NOT need to rebuild the optimizer — only buffer .data
    on layer L+1 is mutated.
    """
    if layer_idx + 1 >= len(net.layers):
        return 0
    layer = net.layers[layer_idx]
    nxt = net.layers[layer_idx + 1]
    survivors = [
        k for k in range(layer.n_nodes)
        if k != dead_idx
        and not bool(layer.routing_latched[k].item())
    ]
    if not survivors:
        return 0
    n = len(survivors)
    with torch.no_grad():
        share_W = nxt.W.data[:, dead_idx] / n
        share_anchor = nxt.W_anchor[:, dead_idx] / n
        share_fisher = nxt.fisher_W[:, dead_idx] / n
        for k in survivors:
            nxt.W.data[:, k] = nxt.W.data[:, k] + share_W
            nxt.W_anchor[:, k] = nxt.W_anchor[:, k] + share_anchor
            nxt.fisher_W[:, k] = nxt.fisher_W[:, k] + share_fisher
        # Zero the dead cell's outgoing — its bias-driven constant no
        # longer routes anywhere. The role it carried lives on in the
        # peers that absorbed its share.
        nxt.W.data[:, dead_idx] = 0.0
        nxt.W_anchor[:, dead_idx] = 0.0
        nxt.fisher_W[:, dead_idx] = 0.0
    return n


def apoptosis_spike(
    net: TrioronNetwork,
    layer_idx: int,
    dead_idx: int,
    *,
    spike_init: float = 0.8,
) -> int:
    """Phase 4.5 Experiment 5 — strong-signal-that-dies-slowly. At the
    moment unit `dead_idx` latches, raise apoptosis_pulse on every
    surviving (non-latched) peer in the same layer to at least
    `spike_init`. Subsequent dream blocks decay the pulse.

    Uses max() so cumulative deaths don't drive pulse above 1.0 — the
    signal saturates rather than compounding into negative effective
    lambda.

    Returns the number of peers spiked.
    """
    if not (0.0 <= spike_init <= 1.0):
        raise ValueError(
            f"spike_init must be in [0,1], got {spike_init!r}"
        )
    layer = net.layers[layer_idx]
    survivors = [
        k for k in range(layer.n_nodes)
        if k != dead_idx
        and not bool(layer.routing_latched[k].item())
        and not bool(layer.archived[k].item())
    ]
    if not survivors:
        return 0
    with torch.no_grad():
        for k in survivors:
            cur = float(layer.apoptosis_pulse[k].item())
            if spike_init > cur:
                layer.apoptosis_pulse[k] = spike_init
    return len(survivors)


def apoptosis_decay(
    net: TrioronNetwork,
    decay_rate: float = 0.7,
) -> None:
    """Multiply apoptosis_pulse by `decay_rate` on every layer. Called
    once per dream block so the spike fades over a handful of blocks.

    decay_rate=0.7 → spike halves in ~2 blocks, drops below 10% in ~7.
    decay_rate=0.0 → pulse cleared each block (one-shot effect).
    decay_rate=1.0 → pulse never decays (permanent — usually wrong).
    """
    if not (0.0 <= decay_rate <= 1.0):
        raise ValueError(
            f"decay_rate must be in [0,1], got {decay_rate!r}"
        )
    with torch.no_grad():
        for layer in net.layers:
            layer.apoptosis_pulse.mul_(decay_rate)


def routing_starve(
    net: TrioronNetwork,
    layer_idx: int,
    victim_idx: int,
    *,
    alpha: float = 0.7,
    floor: float = 1e-3,
    apoptosis_on: bool = False,
    apoptosis_spike_init: float = 0.8,
) -> Tuple[float, bool]:
    """Routing starvation: multiply victim's routing_scale by `alpha`. If the
    new scale crosses `floor`, set it to 0 and latch (permanent — no regrow).

    Substrate-preserving in the same spirit as `synaptic_downscale`, but
    asymmetric: the victim's INPUTS are ramped down (forward applies
    F.linear(x, W * routing_scale.unsqueeze(1), b)) while its OUTGOING
    weights are untouched. Bias is untouched too — so as routing_scale → 0
    the victim continues producing a bias-only constant downstream, and
    downstream layers learn to absorb that constant via gradient descent
    on their own weights / biases. The unit "dies slowly" rather than
    being deleted.

    `apoptosis_on` (Phase 4.5 Experiment 5): when True, the moment a
    victim transitions from non-latched to latched (scale crosses
    `floor`) fires both apoptosis_redistribute (uniform outgoing
    transfer to surviving peers; victim's outgoing zeroed) and
    apoptosis_spike (raise apoptosis_pulse on surviving peers to
    `apoptosis_spike_init`). Subsequent dream blocks must apply
    apoptosis_decay to fade the spike.

    Returns (routing_scale_after, latched_now).

    Caller does NOT need to rebuild the optimizer — only buffer .data is
    mutated (and L+1's W .data when apoptosis fires).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha!r}")
    if floor < 0.0:
        raise ValueError(f"floor must be >= 0, got {floor!r}")
    layer = net.layers[layer_idx]
    if not (0 <= victim_idx < layer.n_nodes):
        raise IndexError(
            f"victim_idx {victim_idx} out of range [0,{layer.n_nodes})"
        )
    if bool(layer.archived[victim_idx].item()):
        # Archived rows are off-limits to starvation. Caller bug if we
        # got here — return current scale unchanged (no latch transition).
        return (float(layer.routing_scale[victim_idx].item()), False)
    with torch.no_grad():
        if bool(layer.routing_latched[victim_idx].item()):
            return (0.0, True)
        cur = float(layer.routing_scale[victim_idx].item())
        new = cur * alpha
        latched_now = False
        if new < floor:
            new = 0.0
            layer.routing_latched[victim_idx] = True
            latched_now = True
        layer.routing_scale[victim_idx] = new
    if latched_now and apoptosis_on:
        # Set latched=True is already in place above; helpers will
        # exclude this index because it's now latched. Do this OUTSIDE
        # the `with torch.no_grad()` block for symmetry — the helpers
        # take their own no_grad context.
        apoptosis_redistribute(net, layer_idx, victim_idx)
        apoptosis_spike(
            net, layer_idx, victim_idx, spike_init=apoptosis_spike_init,
        )
    return (new, latched_now)


def routing_regrow(
    net: TrioronNetwork,
    layer_idx: int,
    victim_idx: int,
    *,
    alpha: float = 0.7,
) -> float:
    """Inverse of routing_starve: multiply victim's routing_scale by 1/alpha
    (capped at 1.0). Latched units are NOT regrown (the latch is permanent).

    Returns the routing_scale after the regrow.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha!r}")
    layer = net.layers[layer_idx]
    with torch.no_grad():
        if bool(layer.routing_latched[victim_idx].item()):
            return 0.0
        cur = float(layer.routing_scale[victim_idx].item())
        new = min(1.0, cur / alpha)
        layer.routing_scale[victim_idx] = new
    return new


def _pick_starvation_victim(
    net: TrioronNetwork,
    layer_idx: int,
    idx_a: int,
    idx_b: int,
) -> Tuple[int, int]:
    """Return (peer_idx, victim_idx) — primary keeps, victim gets starved.

    Primary criterion: older task_of_origin = primary. Tiebreak: larger
    outgoing-norm on layer L+1 = primary. Last-resort tiebreak when both
    ages and outgoing norms are identical (or no L+1 exists): smaller
    index = primary, so the choice is deterministic for tests.
    """
    layer = net.layers[layer_idx]
    age_a = int(layer.task_of_origin[idx_a].item())
    age_b = int(layer.task_of_origin[idx_b].item())
    if age_a < age_b:
        return idx_a, idx_b
    if age_b < age_a:
        return idx_b, idx_a
    has_next = layer_idx + 1 < len(net.layers)
    if has_next:
        nxt = net.layers[layer_idx + 1]
        norm_a = float(nxt.W_anchor[:, idx_a].norm().item())
        norm_b = float(nxt.W_anchor[:, idx_b].norm().item())
    else:
        # Output layer fallback — use the unit's own incoming W norm. The
        # default skip_output_layer policy means we usually don't reach
        # this path.
        norm_a = float(layer.W_anchor[idx_a].norm().item())
        norm_b = float(layer.W_anchor[idx_b].norm().item())
    if norm_a > norm_b:
        return idx_a, idx_b
    if norm_b > norm_a:
        return idx_b, idx_a
    return (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)


def compress(
    net: TrioronNetwork,
    *,
    cos_threshold: float = 0.95,
    layer_idxs: Optional[Sequence[int]] = None,
    skip_output_layer: bool = True,
    max_merges: int = 64,
    redundancy_signal: str = "weight",
    probe_batch: Optional[torch.Tensor] = None,
    ac_threshold: float = 0.95,
    compression_action: str = "merge",
    max_downscales_per_layer: Optional[int] = None,
    starvation_alpha: float = 0.7,
    starvation_floor: float = 1e-3,
    apoptosis_on: bool = False,
    apoptosis_spike_init: float = 0.8,
) -> List[MergeEvent]:
    """Greedy topological compression: repeatedly merge the highest-similarity
    pair on each candidate layer until none exceed the threshold.

    `redundancy_signal`:
      - "weight"     — cosine of W_anchor rows (cheap, the original
                       Phase 4.5 mechanism). Uses `cos_threshold`.
      - "activation" — Pearson cosine of post-activation columns over
                       `probe_batch`. Uses `ac_threshold`. `probe_batch`
                       is required.

    `compression_action`:
      - "merge"     — legacy: incoming-mean + outgoing-sum + delete
                      victim. Destructive. Optimizer rebuild required
                      after returning if events fired (Parameter
                      objects of any modified layer are replaced).
      - "downscale" — synaptic downscale: peer's outgoing column +=
                      victim's, victim's outgoing zeroed, victim's
                      substrate at layer L preserved. Non-destructive,
                      VRAM unchanged. Optimizer rebuild NOT required.
                      Re-recruitment is possible (fisher pinned at zero
                      on victim's outgoing column → no EWC penalty).
      - "starve"    — routing starvation (Phase 4.5 Experiment 3,
                      2026-05-01): victim's routing_scale *= alpha each
                      event; below `starvation_floor` it latches to 0
                      permanently. Peer untouched. Asymmetric and
                      reversible until latched: at the END of each
                      compress() call, any non-latched node in a
                      processed layer with 0 < scale < 1.0 that was
                      NOT selected as a victim this call is regrown
                      (scale ← min(1.0, scale / alpha)). The "primary"
                      kept-side is older task_of_origin (tiebreak: larger
                      outgoing-norm). Optimizer rebuild NOT required.

    `max_downscales_per_layer` (Experiment 2 of Phase 4.5, 2026-05-01):
      Hard cap on the number of compression events per layer per call.
      Applies to BOTH "downscale" and "starve" (merge has natural
      termination via victim deletion). None = uncapped (the original
      behavior). Set to small integers (1-2) to let replay absorb each
      consolidation before the next one drifts on top of it; dampens
      runaway behavior on seeds with high natural activation-cosine
      ceilings (see dreaming_synaptic_sweep_result).

    `starvation_alpha`, `starvation_floor` (Experiment 3): per-event
      multiplicative ramp factor and latch threshold for "starve".
      Defaults: alpha=0.7, floor=1e-3 — hits floor in ~21 events from
      scale=1.0, hits 0.1 in ~7 events.

    By default the OUTPUT layer is skipped — collapsing two latent dims
    silently changes the representational capacity of the network in a
    way that's distinct from compressing redundant hidden features.
    Override with `layer_idxs` to compress everything.

    The MergeEvent.cos_sim field carries whichever similarity score the
    chosen signal produced (W cosine for "weight", activation cosine
    for "activation"). MergeEvent.action records which compression
    action was applied.

    Returns the sequence of compression events. For "merge", caller MUST
    rebuild any optimizer afterwards if events occurred. For "downscale",
    optimizer rebuild is not required.
    """
    if redundancy_signal not in ("weight", "activation"):
        raise ValueError(
            f"redundancy_signal must be 'weight' or 'activation', "
            f"got {redundancy_signal!r}"
        )
    if compression_action not in ("merge", "downscale", "starve"):
        raise ValueError(
            f"compression_action must be 'merge', 'downscale', or "
            f"'starve', got {compression_action!r}"
        )
    if redundancy_signal == "activation" and probe_batch is None:
        raise ValueError(
            "redundancy_signal='activation' requires probe_batch"
        )
    if (max_downscales_per_layer is not None
            and max_downscales_per_layer < 0):
        raise ValueError(
            f"max_downscales_per_layer must be >= 0 or None, "
            f"got {max_downscales_per_layer!r}"
        )
    if compression_action == "starve":
        if not (0.0 < starvation_alpha < 1.0):
            raise ValueError(
                f"starvation_alpha must be in (0,1), "
                f"got {starvation_alpha!r}"
            )
        if starvation_floor < 0.0:
            raise ValueError(
                f"starvation_floor must be >= 0, got {starvation_floor!r}"
            )

    if layer_idxs is None:
        last = len(net.layers) - 1
        layer_idxs = list(range(len(net.layers)))
        if skip_output_layer and last >= 0:
            layer_idxs = [i for i in layer_idxs if i != last]

    events: List[MergeEvent] = []
    for L in layer_idxs:
        if L < 0 or L >= len(net.layers):
            continue
        # Per-layer set of indices that have already been chosen as victim
        # this call. With "merge" the victim is physically removed so re-
        # detection naturally stops; with "downscale"/"starve" the victim
        # is still in the layer, so without an explicit exclusion the same
        # pair would be re-picked. We only exclude the victim — a node
        # that became someone else's PEER on a later iteration is still a
        # valid load-bearer.
        excluded_victims: set = set()
        n_events_this_layer = 0
        # For "downscale" on the output layer, synaptic_downscale is a
        # no-op (no L+1 to redirect into) — drop the layer to keep the
        # event log honest. "starve" doesn't need L+1, but the dreaming
        # spec keeps the output layer off-limits as a representational
        # invariant.
        if compression_action == "downscale" and L >= len(net.layers) - 1:
            continue
        # Iterate inside this layer until no eligible pair remains, or we
        # hit max_merges (paranoia bound — prevents pathological loops on
        # an all-collinear layer). For "activation", recompute the probe
        # forward each iteration so post-action activations drive the
        # next candidate decision (downscale/starve change L+1 weights or
        # routing_scale, not L's W_anchor, so the layer-L activation
        # cosine of (peer, victim) wouldn't change without the explicit
        # exclusion above).
        while len(events) < max_merges:
            if (compression_action in ("downscale", "starve")
                    and max_downscales_per_layer is not None
                    and n_events_this_layer >= max_downscales_per_layer):
                break
            if net.layers[L].n_nodes <= 1:
                break
            if redundancy_signal == "weight":
                pairs = find_redundant_pairs(
                    net, L, cos_threshold=cos_threshold,
                )
            else:
                pairs = find_activation_redundant_pairs(
                    net, L, probe_batch=probe_batch, ac_threshold=ac_threshold,
                )
            if not pairs:
                break
            # Pick first eligible pair (highest similarity).
            #   merge:     no exclusion needed — victim is deleted.
            #   downscale: exclude any previously-victimized index (it
            #              has zero outgoing already).
            #   starve:    exclude any node already starved this call AND
            #              skip pairs where BOTH nodes are latched (no
            #              eligible victim left in the pair).
            picked = None
            peer_for_pair = None
            layer_at_L = net.layers[L]
            for i, j, s in pairs:
                # Archive guard: archived rows are developmentally closed
                # and off-limits to every compression action (merge,
                # downscale, starve). If either side of the candidate
                # pair is archived, skip the pair entirely — its
                # contribution is locked at consolidated state and must
                # not be mutated by dreaming.
                if (bool(layer_at_L.archived[i].item())
                        or bool(layer_at_L.archived[j].item())):
                    continue
                if (compression_action == "downscale"
                        and (i in excluded_victims
                             or j in excluded_victims)):
                    continue
                if compression_action == "starve":
                    if i in excluded_victims or j in excluded_victims:
                        continue
                    a_lat = bool(layer_at_L.routing_latched[i].item())
                    b_lat = bool(layer_at_L.routing_latched[j].item())
                    if a_lat and b_lat:
                        continue
                    # If exactly one is latched, the latched node is forced
                    # to be the primary (we can't starve it further). The
                    # un-latched node is the victim.
                    if a_lat:
                        peer_for_pair = i
                        picked = (i, j, s)
                        break
                    if b_lat:
                        peer_for_pair = j
                        picked = (i, j, s)
                        break
                    p_idx, _ = _pick_starvation_victim(net, L, i, j)
                    peer_for_pair = p_idx
                    picked = (i, j, s)
                    break
                picked = (i, j, s)
                break
            if picked is None:
                break
            i, j, s = picked
            if compression_action == "merge":
                merge_nodes(net, L, peer_idx=i, victim_idx=j)
                events.append(MergeEvent(
                    layer_idx=L, peer_idx=i, victim_idx=j, cos_sim=s,
                    arch_after=tuple(net.n_nodes_per_layer()),
                    action=compression_action,
                ))
            elif compression_action == "downscale":
                synaptic_downscale(net, L, peer_idx=i, victim_idx=j)
                excluded_victims.add(j)
                n_events_this_layer += 1
                events.append(MergeEvent(
                    layer_idx=L, peer_idx=i, victim_idx=j, cos_sim=s,
                    arch_after=tuple(net.n_nodes_per_layer()),
                    action=compression_action,
                ))
            else:  # starve
                victim_idx = j if peer_for_pair == i else i
                scale_after, latched_now = routing_starve(
                    net, L, victim_idx,
                    alpha=starvation_alpha, floor=starvation_floor,
                    apoptosis_on=apoptosis_on,
                    apoptosis_spike_init=apoptosis_spike_init,
                )
                excluded_victims.add(victim_idx)
                n_events_this_layer += 1
                events.append(MergeEvent(
                    layer_idx=L, peer_idx=peer_for_pair,
                    victim_idx=victim_idx, cos_sim=s,
                    arch_after=tuple(net.n_nodes_per_layer()),
                    action="starve",
                    victim_routing_scale_after=scale_after,
                    victim_latched=latched_now,
                ))

        # Routing-starvation regrow pass: any non-latched node at this
        # layer with 0 < routing_scale < 1.0 that was NOT chosen as a
        # victim this call regrows by 1/alpha (capped at 1.0). This is
        # the "reversible if not yet latched to zero" half of Rocky's
        # spec — units that drifted away from their redundant peer
        # recover their routing.
        if compression_action == "starve":
            layer = net.layers[L]
            for k in range(layer.n_nodes):
                if k in excluded_victims:
                    continue
                if bool(layer.routing_latched[k].item()):
                    continue
                if bool(layer.archived[k].item()):
                    continue
                cur = float(layer.routing_scale[k].item())
                if 0.0 < cur < 1.0:
                    routing_regrow(
                        net, L, k, alpha=starvation_alpha,
                    )
    return events


# ---------------------------------------------------------------------
# Mechanism 3 — VRAM purge
# ---------------------------------------------------------------------


def purge(
    net: TrioronNetwork,
    *,
    u_threshold: float = 1e-3,
    layer_idxs: Optional[Sequence[int]] = None,
    skip_output_layer: bool = True,
    max_purges: int = 64,
) -> List[PurgeEvent]:
    """Drop nodes whose utility u < `u_threshold`. Reuses
    network.prune_layer_node (which handles cross-layer fan_in cleanup
    and redistributes the outgoing column to the cosine-nearest peer
    per §3.3).

    By default the OUTPUT layer is skipped (same rationale as compress).
    The last node in any layer is never pruned.

    Returns purge events. Caller MUST rebuild any optimizer afterwards
    if events occurred.
    """
    if layer_idxs is None:
        last = len(net.layers) - 1
        layer_idxs = list(range(len(net.layers)))
        if skip_output_layer and last >= 0:
            layer_idxs = [i for i in layer_idxs if i != last]

    events: List[PurgeEvent] = []
    for L in layer_idxs:
        if L < 0 or L >= len(net.layers):
            continue
        while len(events) < max_purges:
            layer = net.layers[L]
            if layer.n_nodes <= 1:
                break
            u = layer.u.detach()
            below = (u < u_threshold).nonzero(as_tuple=False).flatten().tolist()
            # Archived rows are off-limits to purge — locked at
            # consolidated state, dropping them would lose protected
            # structure.
            if layer.archived.any():
                archived_set = set(int(i) for i in
                                   layer.archived.nonzero(as_tuple=False).flatten().tolist())
                below = [i for i in below if i not in archived_set]
            if not below:
                break
            # Drop the lowest-utility node first.
            idx = int(min(below, key=lambda i: float(u[i].item())))
            u_val = float(u[idx].item())
            net.prune_layer_node(L, idx, redistribute=True)
            events.append(PurgeEvent(
                layer_idx=L, node_idx=idx, u_at_purge=u_val,
                arch_after=tuple(net.n_nodes_per_layer()),
            ))
    return events


# ---------------------------------------------------------------------
# Archive — Phase 1 of dream-archive stage.
#
# Triggers (per memory/dream_archive_stage.md): a row gets archived when
# its λ has been in the layer's top-percentile band for N consecutive
# consolidations AND its recent gradient magnitude is low AND no
# apoptosis pulse is on it (archive is "this is GOOD, lock it"; pulse
# means "a neighbor just died, you may need to retrain"). The streak is
# updated each call by comparing this consolidation's λ against the
# layer-percentile threshold.
#
# Phase 1 only marks rows; it does NOT yet drop W_anchor / fisher_W /
# Adam moments. The structural mechanism (developmentally-closed rows
# that growth/apoptosis/purge skip and that don't drift) lands first;
# the int8-quantization RAM payoff is Phase 2.
# ---------------------------------------------------------------------


def archive_block(
    net: TrioronNetwork,
    *,
    streak_threshold: int = 3,
    lam_top_percentile: float = 0.75,
    grad_mag_floor: float = 0.1,
    pulse_max: float = 0.1,
    layer_idxs: Optional[Sequence[int]] = None,
    skip_output_layer: bool = True,
    max_archives_per_layer: int = 8,
) -> List[Tuple[int, int]]:
    """Update lam_high_streak counters and archive eligible rows.

    Call AFTER `consolidate_task` (so λ is freshly refreshed) and AFTER
    Fisher EMA is updated (so fisher_W reflects recent gradient
    magnitude). Idempotent: archived rows stay archived.

    streak_threshold: archive a row only after this many consecutive
        consolidations with λ in the top percentile. Default 3.
    lam_top_percentile: per-layer percentile boundary on λ. Default 0.75
        means "row's λ must be ≥ the 75th-percentile λ across non-
        archived rows." A row in the top quartile counts as a "high λ"
        consolidation; otherwise the streak resets.
    grad_mag_floor: archive only if sqrt(fisher_W[row].sum()) ≤ this
        threshold. Fisher_W is the EMA of squared gradients, so its
        sqrt-sum is a magnitude of recent gradient pull. Low magnitude
        = row is settled (not contested by current task). Default 0.1
        is a conservative starting point at fan_in=128 scales; calibrate
        from per-task fisher_W magnitudes when wiring into the bench.
    pulse_max: archive only if apoptosis_pulse[row] ≤ this. A row near
        a recent death needs plasticity, not closure.
    skip_output_layer: archive doesn't fire on the head (its rows are
        per-class outputs; closing one would freeze that class's logits
        forever). Default True.
    max_archives_per_layer: per-call cap. Defaults to 8 to avoid bulk
        archiving in a single consolidation step.

    Returns a list of (layer_idx, row_idx) tuples for newly-archived
    rows. Streak counters are updated in-place on every call.
    """
    if layer_idxs is None:
        last = len(net.layers) - 1
        layer_idxs = list(range(len(net.layers)))
        if skip_output_layer and last >= 0:
            layer_idxs = [i for i in layer_idxs if i != last]

    archived_now: List[Tuple[int, int]] = []
    for L in layer_idxs:
        if L < 0 or L >= len(net.layers):
            continue
        layer = net.layers[L]
        if layer.n_nodes <= 1:
            continue

        # Compute per-layer top-percentile λ threshold over NON-archived
        # rows. If all rows are already archived (degenerate edge case),
        # nothing to do.
        active_mask = ~layer.archived
        if int(active_mask.sum().item()) == 0:
            continue
        active_lam = layer.lam[active_mask]
        if active_lam.numel() == 0:
            continue
        threshold = float(
            torch.quantile(active_lam.float(), lam_top_percentile).item()
        )

        # Update streak: increment for active rows whose λ ≥ threshold,
        # reset for active rows below. Archived rows' streaks stay 0.
        with torch.no_grad():
            high = (layer.lam >= threshold) & active_mask
            layer.lam_high_streak[high] += 1
            layer.lam_high_streak[active_mask & ~high] = 0

        # Find archive candidates: streak ≥ threshold, low grad mag,
        # low apoptosis pulse, and not already archived.
        grad_mag = layer.fisher_W.sum(dim=1).sqrt()
        pulse = layer.apoptosis_pulse
        candidates = (
            (layer.lam_high_streak >= streak_threshold)
            & (grad_mag <= grad_mag_floor)
            & (pulse <= pulse_max)
            & active_mask
        )
        candidate_idxs = candidates.nonzero(as_tuple=False).flatten().tolist()

        # Cap per-call. Among candidates, archive the rows with the
        # HIGHEST λ first (most-stable-most-stiff first).
        candidate_idxs.sort(key=lambda i: -float(layer.lam[i].item()))
        for idx in candidate_idxs[:max_archives_per_layer]:
            layer.archive_row(int(idx))
            archived_now.append((L, int(idx)))

    return archived_now


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------


def _build_probe_batch(
    sample_pair_fn: Callable[[str, int], Tuple[torch.Tensor, torch.Tensor]],
    past_pair_names: Sequence[str],
    probe_batch_size: int,
    rng: Optional[random.Random],
) -> Optional[torch.Tensor]:
    """Concatenate `probe_batch_size` rows drawn evenly from past pairs.

    Returns None if `past_pair_names` is empty. Each pair contributes
    roughly `probe_batch_size / len(past_pair_names)` samples (at least
    1), capped at probe_batch_size in total. Both halves of the
    contrastive pair are stacked — the network sees the same data the
    detector cares about.
    """
    if not past_pair_names:
        return None
    n = len(past_pair_names)
    per_pair = max(1, probe_batch_size // (2 * n))  # halved because we
    # take BOTH a and b from each pair below.
    chunks: List[torch.Tensor] = []
    total = 0
    order = list(past_pair_names)
    if rng is not None:
        rng.shuffle(order)
    for name in order:
        a, b = sample_pair_fn(name, per_pair)
        chunks.append(a.detach())
        chunks.append(b.detach())
        total += a.shape[0] + b.shape[0]
        if total >= probe_batch_size:
            break
    if not chunks:
        return None
    out = torch.cat(chunks, dim=0)
    if out.shape[0] > probe_batch_size:
        out = out[:probe_batch_size]
    return out


def dreaming_block(
    net: TrioronNetwork,
    *,
    sample_pair_fn: Callable[[str, int], Tuple[torch.Tensor, torch.Tensor]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    past_pair_names: Sequence[str],
    replay_fraction: float = 0.25,
    replay_steps_per_pair: int = 200,
    replay_batch: int = 32,
    replay_lr: float = 3e-3,
    ewc_strength: float = 1000.0,
    cos_threshold: float = 0.95,
    u_threshold: float = 1e-3,
    skip_output_layer: bool = True,
    rng: Optional[random.Random] = None,
    redundancy_signal: str = "weight",
    ac_threshold: float = 0.95,
    probe_batch_size: int = 128,
    compression_action: str = "merge",
    max_downscales_per_layer: Optional[int] = None,
    starvation_alpha: float = 0.7,
    starvation_floor: float = 1e-3,
    consolidate: bool = True,
    apoptosis_on: bool = False,
    apoptosis_spike_init: float = 0.8,
    apoptosis_decay_rate: float = 0.7,
) -> DreamingReport:
    """Run replay → compress → purge in sequence. Returns a DreamingReport.

    `redundancy_signal`:
      - "weight" (default for back-compat): use W_anchor cosine and
        `cos_threshold` to find redundant pairs.
      - "activation": build a probe batch of size `probe_batch_size`
        from `past_pair_names` after replay and BEFORE compress; pass
        that probe to `compress` with `ac_threshold`. The probe is
        sampled once per dreaming block (consistent detection / merge
        decisions across the layer sweep). When past_pair_names is
        empty, falls back to the weight signal silently — no past
        activations are available to correlate.

    `compression_action`:
      - "merge" (default): destructive merge_nodes (averaging incoming +
        summing outgoing + deleting victim).
      - "downscale": synaptic downscale (substrate-preserving — peer
        absorbs victim's outgoing, victim's outgoing zeroed, victim's
        row at layer L untouched). The architecture / param count does
        NOT change.
      - "starve": routing starvation (Phase 4.5 Experiment 3 — see
        `routing_starve` docstring). Asymmetric: victim's incoming W is
        scaled by `starvation_alpha` per event; below
        `starvation_floor` the scale latches to 0 permanently.
        Reversible while not latched (regrow pass at end of compress).
        Architecture and param count unchanged.

    `max_downscales_per_layer` (Phase 4.5 Experiment 2): hard cap on
    the number of compression events per layer per call. None = uncapped
    (the original behavior). Applies to BOTH 'downscale' and 'starve'
    — merge has natural termination via victim deletion.

    `starvation_alpha`, `starvation_floor` (Phase 4.5 Experiment 3):
    only consulted when compression_action='starve'. Defaults
    alpha=0.7, floor=1e-3.

    `consolidate` (Phase 4.5 Experiment 4 — developmental window,
    2026-05-02): when False, skip both compress() and purge() — but
    replay and pre-compress probes still run. The block becomes a
    pure-rehearsal-and-measure pass. Lets the caller turn off
    structural consolidation past a developmental window (e.g.,
    `task_idx < N`) while keeping past-pair memories warm. Default
    True preserves prior behavior.

    `apoptosis_on` (Phase 4.5 Experiment 5 — apoptosis spike,
    2026-05-02): when True (and compression_action='starve'), every
    full-latch transition during compress fires:
      - apoptosis_redistribute: uniform transfer of the dead cell's
        outgoing column on layer L+1 across all surviving non-latched
        peers; victim's outgoing zeroed.
      - apoptosis_spike: raise apoptosis_pulse on every surviving
        non-latched peer to `apoptosis_spike_init`.
    Each dream block also applies `apoptosis_pulse *= apoptosis_decay_rate`
    on every layer (whether or not consolidate is True), so the spike
    fades over a few blocks. ewc_penalty uses (1 - pulse).clamp_min(0)
    as effective lambda — neighbors of fresh deaths train with reduced
    EWC stiffness so they can adjust to absorb the dead cell's role.
    Defaults: spike_init=0.8, decay_rate=0.7.

    Caller is responsible for rebuilding the optimizer after this returns
    if `report.merges` or `report.purges` is non-empty under
    `compression_action='merge'` (the W/b Parameter objects of any
    modified layer have been replaced). For 'downscale' / 'starve',
    only purges require rebuild.
    """
    if redundancy_signal not in ("weight", "activation"):
        raise ValueError(
            f"redundancy_signal must be 'weight' or 'activation', "
            f"got {redundancy_signal!r}"
        )
    if compression_action not in ("merge", "downscale", "starve"):
        raise ValueError(
            f"compression_action must be 'merge', 'downscale', or "
            f"'starve', got {compression_action!r}"
        )
    n_before = net.n_parameters()

    # Apoptosis pulse decay — applied at the START of every dream block
    # (whether or not consolidate is True) so the spike from any
    # previous block's deaths fades over time. Skipped entirely when
    # apoptosis is OFF (the pulse stays 0 anyway, but keeping the call
    # gated avoids touching buffers in the no-apoptosis path).
    if apoptosis_on:
        apoptosis_decay(net, decay_rate=apoptosis_decay_rate)

    rep = replay(
        net,
        sample_pair_fn=sample_pair_fn,
        loss_fn=loss_fn,
        past_pair_names=past_pair_names,
        fraction=replay_fraction,
        n_steps_per_pair=replay_steps_per_pair,
        batch=replay_batch,
        ewc_strength=ewc_strength,
        lr=replay_lr,
        rng=rng,
    )

    # Probe per-layer max W_anchor cosine BEFORE compress, on the same
    # set of layers compress would consider. Lets the bench observe
    # what cos_threshold would fire without lowering the threshold.
    last = len(net.layers) - 1
    probe_layers = list(range(len(net.layers)))
    if skip_output_layer and last >= 0:
        probe_layers = [i for i in probe_layers if i != last]
    pre_cos = [(L, max_off_diag_cosine(net, L)) for L in probe_layers]

    # Build the probe batch up-front so the activation-cosine probe and
    # compress see the SAME activations (no resampling drift between
    # detection and merge).
    probe_batch: Optional[torch.Tensor] = None
    if redundancy_signal == "activation":
        probe_batch = _build_probe_batch(
            sample_pair_fn, past_pair_names, probe_batch_size, rng,
        )

    pre_act_cos: List[Tuple[int, float]] = []
    if redundancy_signal == "activation" and probe_batch is not None:
        pre_act_cos = [
            (L, max_off_diag_activation_cosine(net, L, probe_batch))
            for L in probe_layers
        ]

    if not consolidate:
        # Developmental-window OFF: replay + probes only, no structural
        # changes. Empty merge/purge lists let downstream callers see
        # the cosine trajectory without reacting to it.
        merges: List[MergeEvent] = []
        purges: List[PurgeEvent] = []
    elif redundancy_signal == "activation" and probe_batch is not None:
        merges = compress(
            net,
            redundancy_signal="activation",
            probe_batch=probe_batch,
            ac_threshold=ac_threshold,
            skip_output_layer=skip_output_layer,
            compression_action=compression_action,
            max_downscales_per_layer=max_downscales_per_layer,
            starvation_alpha=starvation_alpha,
            starvation_floor=starvation_floor,
            apoptosis_on=apoptosis_on,
            apoptosis_spike_init=apoptosis_spike_init,
        )
        purges = purge(
            net, u_threshold=u_threshold,
            skip_output_layer=skip_output_layer,
        )
    else:
        # Either signal=='weight', or signal=='activation' but no past
        # pairs to build a probe from (first task) — fall through to
        # the weight signal so the block still completes. The
        # pre_compress_max_activation_cosines field stays empty so
        # the caller can see no activation probe was taken.
        merges = compress(
            net,
            cos_threshold=cos_threshold,
            skip_output_layer=skip_output_layer,
            compression_action=compression_action,
            max_downscales_per_layer=max_downscales_per_layer,
            starvation_alpha=starvation_alpha,
            starvation_floor=starvation_floor,
            apoptosis_on=apoptosis_on,
            apoptosis_spike_init=apoptosis_spike_init,
        )
        purges = purge(
            net, u_threshold=u_threshold,
            skip_output_layer=skip_output_layer,
        )
    n_after = net.n_parameters()

    return DreamingReport(
        replay_loss_before=rep[0],
        replay_loss_after=rep[1],
        replay_pairs_sampled=rep[2],
        replay_steps=rep[3],
        merges=merges,
        purges=purges,
        n_params_before=n_before,
        n_params_after=n_after,
        pre_compress_max_cosines=pre_cos,
        pre_compress_max_activation_cosines=pre_act_cos,
    )
