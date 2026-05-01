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


def compress(
    net: TrioronNetwork,
    *,
    cos_threshold: float = 0.95,
    layer_idxs: Optional[Sequence[int]] = None,
    skip_output_layer: bool = True,
    max_merges: int = 64,
) -> List[MergeEvent]:
    """Greedy topological compression: repeatedly merge the highest-cosine
    pair on each candidate layer until none exceed `cos_threshold`.

    By default the OUTPUT layer is skipped — merging two latent dims
    silently changes the representational capacity of the network in a
    way that's distinct from compressing redundant hidden features.
    Override with `layer_idxs` to compress everything.

    Returns the sequence of merge events. Caller MUST rebuild any
    optimizer afterwards if any events occurred.
    """
    if layer_idxs is None:
        last = len(net.layers) - 1
        layer_idxs = list(range(len(net.layers)))
        if skip_output_layer and last >= 0:
            layer_idxs = [i for i in layer_idxs if i != last]

    events: List[MergeEvent] = []
    for L in layer_idxs:
        if L < 0 or L >= len(net.layers):
            continue
        # Iterate inside this layer until no eligible pair remains, or we
        # hit max_merges (paranoia bound — prevents pathological loops on
        # an all-collinear layer).
        while len(events) < max_merges:
            if net.layers[L].n_nodes <= 1:
                break
            pairs = find_redundant_pairs(net, L, cos_threshold=cos_threshold)
            if not pairs:
                break
            i, j, s = pairs[0]
            merge_nodes(net, L, peer_idx=i, victim_idx=j)
            events.append(MergeEvent(
                layer_idx=L, peer_idx=i, victim_idx=j, cos_sim=s,
                arch_after=tuple(net.n_nodes_per_layer()),
            ))
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
# Orchestrator
# ---------------------------------------------------------------------


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
) -> DreamingReport:
    """Run replay → compress → purge in sequence. Returns a DreamingReport.

    Caller is responsible for rebuilding the optimizer after this returns
    if `report.merges` or `report.purges` is non-empty (the W/b Parameter
    objects of any modified layer have been replaced).
    """
    n_before = net.n_parameters()
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

    merges = compress(
        net, cos_threshold=cos_threshold, skip_output_layer=skip_output_layer,
    )
    purges = purge(
        net, u_threshold=u_threshold, skip_output_layer=skip_output_layer,
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
    )
