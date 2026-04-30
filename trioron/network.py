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

from .node import TrioronLayer


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

    def update_fisher_all(self) -> None:
        """Call after loss.backward() and before optimizer.step()."""
        for layer in self.layers:
            layer.update_fisher()

    def update_lambda_all(self) -> None:
        for layer in self.layers:
            layer.update_lambda()

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

    # ----- structural plasticity (§8 step 5: cellular division) -----

    def grow_layer(
        self,
        layer_idx: int,
        init_vec: Optional[torch.Tensor] = None,
        peer_init_for_next: Optional[torch.Tensor] = None,
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
        new_idx = target.grow_node(init_vec=init_vec)

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

    # ----- introspection -----

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
