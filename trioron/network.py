"""Trioron network — feedforward stack of TrioronLayers.

Implements the multi-layer composition of TrioronLayer, with the
continual-learning aggregations (Fisher estimation, λ refresh, anchoring,
EWC penalty) lifted to whole-network operations.

Used in §8 step 2 to verify EWC works at network scale and in §8 step 5
for coordinated cellular division (grow_layer): when a node is added to
layer i, the corresponding input column is added to layer i+1.
"""

from __future__ import annotations
from typing import Callable, Iterable, Optional, Sequence, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .node import TrioronLayer, _ACTIVATIONS


LayerSpec = Tuple[int, int, str]  # (fan_in, n_nodes, activation)


class TrioronNetwork(nn.Module):
    """Feedforward stack of TrioronLayers."""

    def __init__(self, layer_specs: Sequence[LayerSpec]):
        super().__init__()
        if not layer_specs:
            raise ValueError("Need at least one layer spec.")

        # Validate dimensional consistency between layers.
        for i in range(1, len(layer_specs)):
            prev_n_nodes = layer_specs[i - 1][1]
            this_fan_in = layer_specs[i][0]
            if prev_n_nodes != this_fan_in:
                raise ValueError(
                    f"Layer {i} fan_in={this_fan_in} != layer {i-1} "
                    f"n_nodes={prev_n_nodes}"
                )

        self.layers = nn.ModuleList(
            [
                TrioronLayer(fan_in, n_nodes, activation=act)
                for (fan_in, n_nodes, act) in layer_specs
            ]
        )

    # ----- forward -----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_with_anchors(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using each layer's anchored triparametric state
        (W_anchor, b_anchor, routing_scale_anchor) instead of live state.
        No gradients tracked.

        Used for Parasitic-Dream / LwF distillation: at the start of each
        task, the network is in its just-consolidated state (W ≈ W_anchor
        for all anchored layers). As live state drifts during the new
        task's training — including via dream-rescue mutating routing
        mid-task — this method re-creates the consolidated network's
        response on any input — supplying the "old network's view of new
        data" supervisory signal that distills past-task decision
        boundaries forward into the new task.

        Reads routing_scale_anchor (not live routing_scale): the trioron
        node is triparametric (w, b, u-via-routing). Mixing anchored W
        with live routing produces a fictional network that never existed
        at any point in training history, which silently corrupts the
        LwF target whenever dream-rescue purges fire mid-task.
        """
        with torch.no_grad():
            return self._forward_with_anchors_inner(x)

    def forward_with_anchors_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient-tracking sister of `forward_with_anchors`. Used by
        Engram-Replay consolidation, which runs gradient ascent on the
        input through the anchored network to find per-class engram
        prototypes. The anchored W / b / routing_scale buffers are not
        leaves of the autograd graph (they're registered buffers, not
        Parameters), so backward will only populate `x.grad`, never
        modify the anchored state.
        """
        return self._forward_with_anchors_inner(x)

    def _forward_with_anchors_inner(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            if h.dtype != layer.W_anchor.dtype:
                h = h.to(layer.W_anchor.dtype)
            scale = layer.routing_scale_anchor.unsqueeze(1).to(layer.W_anchor.dtype)
            W_eff = layer.W_anchor * scale
            z = F.linear(h, W_eff, layer.b_anchor)
            h = _ACTIVATIONS[layer.activation](z)
        return h

    def forward_from_layer(
        self, h: torch.Tensor, start_layer: int,
    ) -> torch.Tensor:
        """Forward starting from `start_layer` with live W. h is the input
        to layer `start_layer` (i.e., the post-activation output of layer
        `start_layer - 1`).

        Used for Brainstem-Spark latent rehearsal: synthetic activations
        sampled at a bottleneck (e.g., L1 output) get fed directly into
        the head, skipping the upstream layers entirely.
        """
        if start_layer < 0 or start_layer >= len(self.layers):
            raise ValueError(
                f"start_layer={start_layer} out of range "
                f"[0, {len(self.layers)})"
            )
        for layer in self.layers[start_layer:]:
            h = layer(h)
        return h

    # ----- aggregations across layers -----

    def ewc_penalty(self) -> torch.Tensor:
        """Sum of per-layer EWC penalties.

        Returns a scalar autograd-attached tensor; add it to your task loss
        with a strength multiplier:
            L = L_task + ewc_strength * net.ewc_penalty()
        """
        total = self.layers[0].ewc_penalty()
        for layer in self.layers[1:]:
            total = total + layer.ewc_penalty()
        return total

    def anchor_all(self) -> None:
        for layer in self.layers:
            layer.anchor_weights()

    def reanchor_routing_only(self) -> None:
        """Copy live routing_scale into routing_scale_anchor on every layer
        WITHOUT touching W_anchor or b_anchor. Used after a dream-rescue
        purge mutates routing on grown arms: feature-distillation losses
        (engram L1-MSE, differential δL1) compare live activations against
        anchored activations, and if anchor still references pre-purge
        routing while live uses post-purge routing, the loss fights the
        purge mutation. Re-anchoring routing only (not W) keeps EWC
        anchoring intact (which depends on W_anchor) while letting the
        feature-distillation losses see consistent routing on both sides.
        """
        with torch.no_grad():
            for layer in self.layers:
                layer.routing_scale_anchor.copy_(layer.routing_scale)

    def mask_archived_grads_all(self) -> None:
        """Zero W.grad / b.grad at archived rows across every layer.
        Call AFTER .backward() and BEFORE update_fisher_all() and
        optimizer.step(), so archived rows neither contribute to Fisher
        EMA nor receive optimizer updates."""
        for layer in self.layers:
            layer.mask_archived_grads()

    def n_archived_per_layer(self) -> List[int]:
        """Number of archived rows in each layer (diagnostic)."""
        return [int(layer.archived.sum().item()) for layer in self.layers]

    def update_fisher_all(self) -> None:
        """Call after loss.backward() and before optimizer.step()."""
        for layer in self.layers:
            layer.update_fisher()

    def update_utilities_from_saliency(self) -> None:
        """EMA-update each layer's per-node utility u from the most recent
        forward+backward via |y · ∂L/∂y| saliency (the OBD signal).

        Call after loss.backward() and before optimizer.step(), exactly
        like update_fisher_all(). Replaces the older |W|·|grad_W|
        heuristic that was biased toward weight-magnitude rather than
        functional contribution.
        """
        for layer in self.layers:
            layer.update_utility(layer.saliency_utility())

    def reset_utilities_all(self) -> None:
        """Zero each layer's per-node utility u. Used to reset the
        utility signal at the start of a dream-rescue replay, so the
        post-replay u reflects exclusively saliency on this round of
        replayed past tasks (and not stale current-task or
        previous-block contributions)."""
        with torch.no_grad():
            for layer in self.layers:
                layer.u.zero_()

    def update_lambda_all(self) -> None:
        for layer in self.layers:
            layer.update_lambda()

    def set_lambda_all(
        self,
        signals: Sequence[torch.Tensor],
        mode: str = "absolute",
    ) -> None:
        """Per-layer counterpart to TrioronLayer.set_lambda.

        Writes the per-node plasticity gate λ on every layer from an
        externally-supplied signal. Use for any non-Fisher source:
        environmental sensors on a device deployment, reward magnitudes,
        attention masks, manually-injected freeze/wake priors, etc.

        signals: sequence of per-layer 1-D tensors, one per layer, each
            of shape (layer.n_nodes,). Must match len(self.layers).
        mode: forwarded to TrioronLayer.set_lambda — "absolute"
            (replace), "additive" (mix on top of existing λ), or
            "multiplicative" (scale, e.g. a global sleep-cycle factor).

        The result is clamped to ≥0 per-layer for the same reason as
        TrioronLayer.set_lambda.
        """
        if len(signals) != len(self.layers):
            raise ValueError(
                f"signals length {len(signals)} != n_layers {len(self.layers)}"
            )
        for layer, sig in zip(self.layers, signals):
            layer.set_lambda(sig, mode=mode)

    def reset_fisher_all(self) -> None:
        with torch.no_grad():
            for layer in self.layers:
                layer.fisher_W.zero_()
                layer.fisher_b.zero_()

    def estimate_fisher(
        self,
        batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n_batches: int = 20,
    ) -> None:
        """Reset Fisher buffers and re-estimate at current weights.

        batches: iterable yielding (x, y) batches. Should produce at least
            n_batches batches; iteration stops if it runs dry.
        loss_fn: callable (pred, y) -> scalar loss.
        n_batches: how many batches to consume.

        Use this AFTER training on a task converges, BEFORE anchoring.
        Per Kirkpatrick (2017), Fisher should be estimated at the final
        weights of the task you want to consolidate, not as an EMA across
        the whole training trajectory.

        After this returns, call update_lambda_all() to refresh λ from the
        new Fisher estimate, then anchor_all().
        """
        self.reset_fisher_all()

        # Temporarily lower fisher_decay so this acts more like a uniform
        # mean over the n_batches samples (rather than an EMA dominated by
        # the last few batches).
        saved = [layer.fisher_decay for layer in self.layers]
        for layer in self.layers:
            layer.fisher_decay = 0.5

        try:
            iterator = iter(batches)
            for _ in range(n_batches):
                try:
                    x, y = next(iterator)
                except StopIteration:
                    break
                for p in self.parameters():
                    p.grad = None
                pred = self(x)
                loss = loss_fn(pred, y)
                loss.backward()
                self.update_fisher_all()
        finally:
            for layer, d in zip(self.layers, saved):
                layer.fisher_decay = d

    def populate_lambda(
        self,
        batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n_batches: int = 20,
        rescale_mean: bool = True,
    ) -> None:
        """One-shot consolidation for traditional training loops.

        Use this when the network was trained without the standard
        per-task cycle (joint training, plain SGD/Adam loops, no task
        boundaries) and you now want EWC protection for downstream
        fine-tuning. Wraps estimate_fisher → update_lambda_all →
        (optional rescale) → anchor_all into a single call. Equivalent
        to:

            net.estimate_fisher(batches, loss_fn, n_batches=n_batches)
            net.update_lambda_all()
            if rescale_mean:
                for layer in net.layers:
                    m = layer.lam.mean()
                    if m > 0:
                        layer.lam.div_(m)
            net.anchor_all()

        rescale_mean (default True): per-layer normalize λ to mean 1.0
        so β becomes a stiffness knob independent of the optimizer's
        gradient-magnitude regime. Adam at convergence gives near-zero
        raw Fisher (gradients vanish at the optimum), so without
        rescaling callers need β in the 1e5–1e7 range to get a useful
        penalty. Rescaling loses absolute-importance information across
        layers but keeps relative per-node selectivity within each layer.
        Set to False to preserve raw Fisher magnitudes.

        Clears stale gradients on return so the network is safe to hand
        back to a downstream optimizer immediately.
        """
        self.estimate_fisher(batches, loss_fn, n_batches=n_batches)
        self.update_lambda_all()
        if rescale_mean:
            with torch.no_grad():
                for layer in self.layers:
                    m = layer.lam.mean()
                    if m > 0:
                        layer.lam.div_(m)
        self.anchor_all()
        for p in self.parameters():
            p.grad = None

    # ----- structural plasticity (§8 step 5: cellular division) -----

    def grow_layer(
        self,
        layer_idx: int,
        init_vec: Optional[torch.Tensor] = None,
        peer_init_for_next: Optional[torch.Tensor] = None,
        task_idx: int = 0,
    ) -> int:
        """Coordinated cellular division: add one node to `layer_idx` and,
        if a downstream layer exists, extend its fan_in by 1 to accept the
        new input.

        Per blueprint §4.1:
          - The new node's incoming weight `w` ← init_vec (PCA of residuals
            in the caller's task; zero/random fallback if init_vec is None).
          - λ_new = 0 (fully plastic — handled by TrioronLayer.grow_node).
          - u_new = 0 (neutral start — same).
          - The next layer's NEW INPUT COLUMN ← peer_init_for_next, which
            should be utility-weighted across the next layer's existing
            nodes (§4.1.4: "connect it to all nodes whose u is currently
            elevated"). If None, zeros — the network learns by gradient.

        Returns the new node index in `layer_idx`.

        Caveat: any optimizer holding references to this network's
        parameters MUST be rebuilt after this call.
        """
        if not (0 <= layer_idx < len(self.layers)):
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {len(self.layers)})"
            )
        target = self.layers[layer_idx]
        new_idx = target.grow_node(init_vec=init_vec, task_idx=task_idx)

        # Cross-layer coordination: extend next layer's fan_in by 1.
        if layer_idx + 1 < len(self.layers):
            next_layer = self.layers[layer_idx + 1]
            if peer_init_for_next is not None:
                if peer_init_for_next.shape != (next_layer.n_nodes,):
                    raise ValueError(
                        f"peer_init_for_next shape {tuple(peer_init_for_next.shape)} "
                        f"!= (next_layer.n_nodes={next_layer.n_nodes},)"
                    )
            next_layer.grow_input(init_col=peer_init_for_next)

        return new_idx

    def prune_layer_node(
        self,
        layer_idx: int,
        node_idx: int,
        redistribute: bool = True,
    ) -> None:
        """Coordinated cellular pruning per §3.3.

        Removes node `node_idx` from layer `layer_idx`. If a downstream
        layer exists and `redistribute=True`, the pruned node's outgoing
        column is added to its cosine-similarity-nearest peer's outgoing
        column BEFORE the structural removal — preserving the network's
        approximate input-output behavior.

        Refuses to prune the last remaining node in any layer (would
        zero a layer and break the forward pass).

        Caveat: any optimizer holding references to this network's
        parameters MUST be rebuilt after this call.
        """
        if not (0 <= layer_idx < len(self.layers)):
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {len(self.layers)})"
            )
        target = self.layers[layer_idx]
        if not (0 <= node_idx < target.n_nodes):
            raise IndexError(
                f"node_idx {node_idx} out of range [0, {target.n_nodes})"
            )
        if target.n_nodes <= 1:
            raise ValueError(
                f"Refusing to prune the last node in layer {layer_idx}"
            )

        has_next = layer_idx + 1 < len(self.layers)

        # §3.3 redistribution: peer absorbs the pruned node's downstream role.
        if redistribute and has_next:
            next_layer = self.layers[layer_idx + 1]
            with torch.no_grad():
                W = target.W.data
                victim = W[node_idx]
                victim_norm = victim.norm()
                if victim_norm < 1e-12:
                    # Degenerate node — pick any peer.
                    peer_idx = 0 if node_idx != 0 else 1
                else:
                    sims = torch.zeros(target.n_nodes, device=W.device)
                    for j in range(target.n_nodes):
                        if j == node_idx:
                            sims[j] = -float("inf")
                            continue
                        peer_norm = W[j].norm()
                        if peer_norm < 1e-12:
                            sims[j] = -float("inf")
                            continue
                        sims[j] = torch.dot(victim, W[j]) / (victim_norm * peer_norm)
                    peer_idx = int(sims.argmax().item())

                next_layer.W.data[:, peer_idx] += next_layer.W.data[:, node_idx]
                next_layer.W_anchor[:, peer_idx] += next_layer.W_anchor[:, node_idx]
                # fisher is per-weight; absorbing fisher mass is approximate
                # but better than zeroing it.
                next_layer.fisher_W[:, peer_idx] += next_layer.fisher_W[:, node_idx]

        # Remove the node from this layer.
        target.prune_node(node_idx)

        # Drop the matching input column on the next layer.
        if has_next:
            self.layers[layer_idx + 1].prune_input(node_idx)

    # ----- introspection -----

    def to_mixed_precision(
        self,
        weights_dtype: torch.dtype = torch.float16,
    ) -> "TrioronNetwork":
        """Convert W and b Parameters to `weights_dtype` (default FP16) on
        every layer; keep ALL buffers at their current dtype (FP32 for
        the float buffers, untouched for bool/long ones).

        The point is *mixed* precision: weights ride at the requested
        narrow type for fast hardware-friendly forward / backward, but
        the EWC anchors, Fisher accumulator, lambda, routing-scale,
        apoptosis-pulse all stay FP32 so the consolidation math doesn't
        suffer from FP16 underflow.

        forward / ewc_penalty / update_fisher / grow_node already
        cast across the boundary cleanly (see node.py). Returns self
        for chaining.

        Caller MUST rebuild any optimizer afterwards — the W/b
        Parameter objects are replaced with new dtypes.
        """
        for layer in self.layers:
            new_W = layer.W.detach().to(weights_dtype)
            new_b = layer.b.detach().to(weights_dtype)
            layer._replace_parameter("W", new_W)
            layer._replace_parameter("b", new_b)
        return self

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_nodes_per_layer(self) -> List[int]:
        return [layer.n_nodes for layer in self.layers]

    def __repr__(self) -> str:
        layers_repr = " → ".join(
            f"{layer.fan_in}->{layer.n_nodes}({layer.activation})"
            for layer in self.layers
        )
        return f"TrioronNetwork({layers_repr})"
