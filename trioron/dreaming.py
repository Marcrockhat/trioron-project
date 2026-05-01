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
    # "merge" (legacy: incoming-mean + outgoing-sum + delete victim) or
    # "downscale" (synaptic downscale: peer.outgoing += victim.outgoing,
    # victim.outgoing = 0, victim's substrate at layer L preserved). The
    # field is named `action` not `mechanism` because we expect more to
    # be added as the dreaming-phase mechanism evolves. Defaults to
    # "merge" so existing tests / callers see no change.
    action: str = "merge"


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

    `max_downscales_per_layer` (Experiment 2 of Phase 4.5, 2026-05-01):
      Hard cap on the number of downscale events per layer per call.
      Only applies to `compression_action='downscale'` (merge has
      natural termination via victim deletion). None = uncapped (the
      original behavior). Set to small integers (1-2) to let replay
      absorb each consolidation before the next one drifts on top of
      it; dampens runaway behavior on seeds with high natural
      activation-cosine ceilings (see dreaming_synaptic_sweep_result).

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
    if compression_action not in ("merge", "downscale"):
        raise ValueError(
            f"compression_action must be 'merge' or 'downscale', "
            f"got {compression_action!r}"
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

    if layer_idxs is None:
        last = len(net.layers) - 1
        layer_idxs = list(range(len(net.layers)))
        if skip_output_layer and last >= 0:
            layer_idxs = [i for i in layer_idxs if i != last]

    events: List[MergeEvent] = []
    for L in layer_idxs:
        if L < 0 or L >= len(net.layers):
            continue
        # Per-layer set of indices that have already been downscaled this
        # call. With "merge" the victim is physically removed so re-
        # detection naturally stops; with "downscale" the victim is
        # still in the layer, so without an explicit exclusion the pair
        # would be re-found on every iteration. We only exclude the
        # victim — a downscaled node that became someone else's PEER is
        # still a valid load-bearer.
        downscaled_victims: set = set()
        n_downscales_this_layer = 0
        # For "downscale" on the output layer, synaptic_downscale is a
        # no-op (no L+1 to redirect into) — drop the layer to keep the
        # event log honest.
        if compression_action == "downscale" and L >= len(net.layers) - 1:
            continue
        # Iterate inside this layer until no eligible pair remains, or we
        # hit max_merges (paranoia bound — prevents pathological loops on
        # an all-collinear layer). For "activation", recompute the probe
        # forward each iteration so post-action activations drive the
        # next candidate decision (downscale changes layer L+1, not L,
        # so the layer-L activation cosine of (peer, victim) wouldn't
        # change without the explicit exclusion above).
        while len(events) < max_merges:
            if (compression_action == "downscale"
                    and max_downscales_per_layer is not None
                    and n_downscales_this_layer >= max_downscales_per_layer):
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
            # Pick first eligible pair (highest similarity) where neither
            # index has been downscaled-as-victim already.
            picked = None
            for i, j, s in pairs:
                if (compression_action == "downscale"
                        and (i in downscaled_victims
                             or j in downscaled_victims)):
                    continue
                picked = (i, j, s)
                break
            if picked is None:
                break
            i, j, s = picked
            if compression_action == "merge":
                merge_nodes(net, L, peer_idx=i, victim_idx=j)
            else:
                synaptic_downscale(net, L, peer_idx=i, victim_idx=j)
                downscaled_victims.add(j)
                n_downscales_this_layer += 1
            events.append(MergeEvent(
                layer_idx=L, peer_idx=i, victim_idx=j, cos_sim=s,
                arch_after=tuple(net.n_nodes_per_layer()),
                action=compression_action,
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

    `max_downscales_per_layer` (Phase 4.5 Experiment 2): hard cap on
    the number of downscale events per layer per call. None = uncapped
    (the original behavior). Only applies when
    compression_action='downscale' — merge has natural termination.

    Caller is responsible for rebuilding the optimizer after this returns
    if `report.merges` or `report.purges` is non-empty under
    `compression_action='merge'` (the W/b Parameter objects of any
    modified layer have been replaced). For 'downscale', only purges
    require rebuild.
    """
    if redundancy_signal not in ("weight", "activation"):
        raise ValueError(
            f"redundancy_signal must be 'weight' or 'activation', "
            f"got {redundancy_signal!r}"
        )
    if compression_action not in ("merge", "downscale"):
        raise ValueError(
            f"compression_action must be 'merge' or 'downscale', "
            f"got {compression_action!r}"
        )
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

    if redundancy_signal == "activation" and probe_batch is not None:
        merges = compress(
            net,
            redundancy_signal="activation",
            probe_batch=probe_batch,
            ac_threshold=ac_threshold,
            skip_output_layer=skip_output_layer,
            compression_action=compression_action,
            max_downscales_per_layer=max_downscales_per_layer,
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
        pre_compress_max_activation_cosines=pre_act_cos,
    )
