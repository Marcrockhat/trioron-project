"""Trioron network — feedforward stack of TrioronLayers.

Implements the multi-layer composition of TrioronLayer, with the
continual-learning aggregations (Fisher estimation, λ refresh, anchoring,
EWC penalty) lifted to whole-network operations.

This is the architecture used in §8 step 2 of the blueprint to verify EWC
works at network scale, not just per-layer.

Note on growth: TrioronLayer.grow_node only resizes that layer's output
dimension. Coordinated cross-layer growth (where adding a node in layer i
also adds an input column to layer i+1) is implemented in step 5 of the
build plan, not here.
"""

from __future__ import annotations
from typing import Callable, Iterable, Sequence, Tuple, List

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
