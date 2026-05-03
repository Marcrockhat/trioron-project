"""Trioron node — the fundamental unit of the dynamic network.

Implements the Trioron node specification from blueprint §3.

A "node" in this architecture is characterized by three coupled state variables:
    w  — the row of incoming weights (a vector of length fan_in)
    λ  — per-node plasticity coefficient (scalar). Higher = stiffer.
    u  — per-node utility score (scalar). Tracks contribution to good outputs.

For efficiency, many nodes are stored together inside a TrioronLayer, where
each row of the weight matrix W corresponds to one node's incoming weights,
and λ / u are vectors with one entry per node.

The layer supports:
    - Forward pass with a configurable activation
    - Per-node Fisher-information estimation (running EMA of squared gradients)
    - Lambda derived from Fisher info (per-node mean across incoming weights)
    - Utility update from an externally-supplied contribution signal
    - EWC quadratic penalty term against an anchor snapshot
    - Growth (add a new node) and pruning (remove a node)

Notes for callers:
    - After grow_node() or prune_node() you MUST rebuild any optimizer that
      held references to the old W or b parameters. The Parameter objects are
      replaced.
    - update_fisher() must be called after .backward() but before optimizer.step()
      so that .grad is still populated.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_ACTIVATIONS = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "linear": lambda x: x,
}


class TrioronLayer(nn.Module):
    """A layer of Trioron nodes with shared incoming dimensionality."""

    def __init__(
        self,
        fan_in: int,
        n_nodes: int,
        activation: str = "relu",
        lambda_init: float = 0.0,
        u_init: float = 0.0,
        u_decay: float = 0.9,
        fisher_decay: float = 0.99,
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Supported: {list(_ACTIVATIONS.keys())}"
            )

        self.fan_in = int(fan_in)
        self.activation = activation
        self.u_decay = float(u_decay)
        self.fisher_decay = float(fisher_decay)

        # Kaiming-style init scaled for the chosen activation.
        gain = 2.0 if activation == "relu" else 1.0
        std = (gain / fan_in) ** 0.5
        self.W = nn.Parameter(torch.randn(n_nodes, fan_in) * std)
        self.b = nn.Parameter(torch.zeros(n_nodes))

        # Per-node state — buffers, not learned via gradient.
        self.register_buffer("lam", torch.full((n_nodes,), float(lambda_init)))
        self.register_buffer("u", torch.full((n_nodes,), float(u_init)))

        # EWC anchor snapshot (last consolidated weights).
        self.register_buffer("W_anchor", self.W.detach().clone())
        self.register_buffer("b_anchor", self.b.detach().clone())

        # Running EMA of squared gradients ≈ diagonal Fisher information.
        self.register_buffer("fisher_W", torch.zeros_like(self.W))
        self.register_buffer("fisher_b", torch.zeros_like(self.b))

        # Routing-starvation per-node ramp on incoming weights.
        # forward applies F.linear(x, W * routing_scale.unsqueeze(1), b),
        # so as scale → 0 the unit's pre-activation collapses to its bias
        # and gradient flow into W on that row also collapses. 1.0 = full
        # routing. routing_latched marks units whose scale has crossed
        # the starvation floor (permanent — no regrow).
        self.register_buffer("routing_scale", torch.ones(n_nodes))
        self.register_buffer(
            "routing_latched", torch.zeros(n_nodes, dtype=torch.bool),
        )

        # Task index at which each node was born. Layer-init nodes are 0
        # (no prior tasks have run). grow_node accepts the current
        # task_idx so the dreaming phase can pick the YOUNGER of two
        # redundant nodes as the starvation victim.
        self.register_buffer(
            "task_of_origin", torch.zeros(n_nodes, dtype=torch.long),
        )

        # Apoptosis pulse — per-node float in [0, 1] tracking the
        # remaining "dying neighbor" signal received since the last
        # decay step. Spiked at the moment a sibling cell latches
        # (Phase 4.5 Experiment 5). Decays each dream block. Used by
        # ewc_penalty to scale this node's effective lambda by
        # (1 - pulse) — neighbors of a fresh death train at lower
        # EWC stiffness so they can adjust to fill the lost role.
        self.register_buffer("apoptosis_pulse", torch.zeros(n_nodes))

        # Saliency caches for |a · g| utility (Mozer & Smolensky 1989,
        # OBD/blueprint §3.2). _last_y is stashed on each grad-enabled
        # forward; _last_upstream is captured by a backward hook on y
        # when .backward() propagates through. saliency_utility()
        # combines them. Both are transient — NOT registered as
        # buffers so they don't pollute state_dict / serialization.
        self._last_y: Optional[torch.Tensor] = None
        self._last_upstream: Optional[torch.Tensor] = None

    # ----- properties -----
    @property
    def n_nodes(self) -> int:
        return self.W.shape[0]

    # ----- forward -----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # routing_scale modulates each row's incoming contribution. When
        # all entries are 1.0 (the default) this is identical to the
        # plain linear; a starved unit's row is multiplied toward zero,
        # so its pre-activation collapses to its bias.
        # Cast routing_scale to W's dtype so mixed-precision (FP16/BF16 W
        # + FP32 buffer) keeps W_eff in W's dtype.
        # Cast x to W's dtype too so callers using FP32 inputs against a
        # mixed-precision network don't need to know the layer's dtype —
        # F.linear requires matching dtypes and would error otherwise.
        if x.dtype != self.W.dtype:
            x = x.to(self.W.dtype)
        scale = self.routing_scale.unsqueeze(1).to(self.W.dtype)
        W_eff = self.W * scale
        z = F.linear(x, W_eff, self.b)
        y = _ACTIVATIONS[self.activation](z)
        # Stash post-activation y + register a backward hook for
        # upstream gradient ∂L/∂y. Together these enable
        # saliency_utility() to compute |y · ∂L/∂y| (the OBD-style
        # per-node utility). Only fires when y is part of a backward
        # graph (training forwards), so eval/no-grad forwards don't
        # overwrite the most recent saliency state.
        if y.requires_grad:
            self._last_y = y.detach()
            y.register_hook(self._capture_upstream)
        return y

    def _capture_upstream(self, grad: torch.Tensor) -> None:
        """Backward hook: record ∂L/∂y for saliency_utility()."""
        self._last_upstream = grad.detach()

    # ----- state updates (call in this order during training) -----

    def update_fisher(self) -> None:
        """Update running estimate of diagonal Fisher info from current gradients.

        Call AFTER loss.backward() and BEFORE optimizer.step().
        """
        if self.W.grad is None:
            return
        with torch.no_grad():
            self.fisher_W.mul_(self.fisher_decay).add_(
                self.W.grad.detach().pow(2),
                alpha=1.0 - self.fisher_decay,
            )
            if self.b.grad is not None:
                self.fisher_b.mul_(self.fisher_decay).add_(
                    self.b.grad.detach().pow(2),
                    alpha=1.0 - self.fisher_decay,
                )

    def update_lambda(self) -> None:
        """Refresh per-node plasticity λ from current Fisher estimate.

        λ = sum of Fisher mass across the node's incoming weights. The
        blueprint §3.2 said "mean," but the chained-15 Fisher probe
        (2026-05-03) showed the mean collapse divides typical fisher_W
        magnitudes (~1e-3) by fan_in (32–128), pushing per-node λ to
        1e-4 to 1e-6 — orders of magnitude below any usable floor. Sum
        recovers the magnitude so real Fisher can express selectivity
        rather than being drowned by an external λ_floor clamp.
        Higher λ ⇒ this node's weights matter a lot for current loss ⇒ stiffer.
        """
        with torch.no_grad():
            self.lam.copy_(self.fisher_W.sum(dim=1))

    def update_utility(self, contributions: torch.Tensor) -> None:
        """EMA update of per-node utility scores.

        contributions: tensor of shape (n_nodes,). The network is responsible
        for computing this — typically sign(reward) * |activation * gradient|
        averaged over the batch.
        """
        if contributions.shape != (self.n_nodes,):
            raise ValueError(
                f"contributions shape {tuple(contributions.shape)} "
                f"!= (n_nodes={self.n_nodes},)"
            )
        with torch.no_grad():
            self.u.mul_(self.u_decay).add_(
                contributions.detach().to(self.u.device),
                alpha=1.0 - self.u_decay,
            )

    def saliency_utility(self) -> torch.Tensor:
        """Per-node saliency |y · ∂L/∂y| from the most recent grad-enabled
        forward + backward.

        This is the standard OBD / Mozer-Smolensky pruning saliency: a
        first-order approximation of "how much would the loss change if
        I clamped this node's output to zero." The right signal for
        purge victim selection in dreaming.

        For relu activations, dead nodes (y=0 always) score 0 here even
        if their incoming weights are large, which is the desired
        behavior: their removal genuinely costs nothing.

        Returns a (n_nodes,) tensor on self.W.device. Returns zeros if
        no grad-enabled forward+backward has run yet (e.g., right after
        construction, or after only no_grad/eval forwards).
        """
        if self._last_y is None or self._last_upstream is None:
            return torch.zeros(self.n_nodes, device=self.W.device)
        # _last_y, _last_upstream: (batch, n_nodes). Mean across batch.
        contrib = (self._last_y.abs() * self._last_upstream.abs()).mean(dim=0)
        return contrib.to(device=self.W.device, dtype=torch.float32)

    # ----- EWC anchor / penalty -----

    def anchor_weights(self) -> None:
        """Snapshot current weights as the EWC anchor.

        Call after a successful learning plateau to lock current state in.
        Subsequent EWC penalty drags W back toward this snapshot.
        """
        with torch.no_grad():
            self.W_anchor.copy_(self.W.detach())
            self.b_anchor.copy_(self.b.detach())

    def ewc_penalty(self) -> torch.Tensor:
        """Quadratic penalty pulling weights toward the anchor.

        Per-node λ scales the strength. Total task loss should be:
            L = L_task + ewc_strength * layer.ewc_penalty()

        The apoptosis_pulse buffer (Phase 4.5 Experiment 5) lowers each
        node's effective λ by (1 - pulse) clamped to ≥0 — when a
        neighbor cell latches dead, surviving peers get a temporary
        plasticity boost that fades as the pulse decays.

        Returns a scalar tensor (autograd-attached when W requires grad).
        """
        # Per-node effective stiffness = λ · (1 - apoptosis_pulse). When
        # all pulses are 0 (no recent deaths), this equals plain λ.
        eff_lam = self.lam * (1.0 - self.apoptosis_pulse).clamp_min(0.0)
        stiffness = eff_lam.unsqueeze(1)  # (n_nodes, 1)
        pen_W = (stiffness * (self.W - self.W_anchor).pow(2)).sum()
        pen_b = (eff_lam * (self.b - self.b_anchor).pow(2)).sum()
        return pen_W + pen_b

    # ----- structural plasticity -----

    def grow_node(
        self,
        init_vec: Optional[torch.Tensor] = None,
        task_idx: int = 0,
    ) -> int:
        """Add one new node to the layer. Returns the index of the new node.

        init_vec: optional length-fan_in tensor for the new node's incoming
            weights. If None, uses the same Kaiming-style init as construction.
        task_idx: current task index (used by dreaming-phase routing
            starvation to pick the YOUNGER of two redundant nodes as
            victim). Defaults to 0 so legacy callers / tests are unaffected.

        After calling this, REBUILD any optimizer that referenced this layer's
        parameters — the W and b Parameter objects are replaced.
        """
        device = self.W.device
        with torch.no_grad():
            if init_vec is None:
                gain = 2.0 if self.activation == "relu" else 1.0
                std = (gain / self.fan_in) ** 0.5
                new_row = torch.randn(1, self.fan_in, device=device) * std
            else:
                new_row = init_vec.detach().to(device).reshape(1, self.fan_in)

            # Mixed-precision support: cast the new row to each
            # destination's dtype before concat. W may be FP16 while
            # W_anchor / fisher_W are kept FP32 by to_mixed_precision().
            new_row_W = new_row.to(self.W.dtype)
            new_row_anchor = new_row.to(self.W_anchor.dtype)
            new_W = torch.cat([self.W.data, new_row_W], dim=0)
            new_b = torch.cat(
                [self.b.data, torch.zeros(1, dtype=self.b.dtype, device=device)],
                dim=0,
            )
            new_lam = torch.cat(
                [self.lam, torch.zeros(1, dtype=self.lam.dtype, device=device)]
            )
            new_u = torch.cat(
                [self.u, torch.zeros(1, dtype=self.u.dtype, device=device)]
            )
            new_W_anchor = torch.cat(
                [self.W_anchor, new_row_anchor.clone()], dim=0
            )
            new_b_anchor = torch.cat(
                [self.b_anchor,
                 torch.zeros(1, dtype=self.b_anchor.dtype, device=device)]
            )
            new_fisher_W = torch.cat(
                [self.fisher_W,
                 torch.zeros(1, self.fan_in, dtype=self.fisher_W.dtype,
                             device=device)],
                dim=0,
            )
            new_fisher_b = torch.cat(
                [self.fisher_b,
                 torch.zeros(1, dtype=self.fisher_b.dtype, device=device)]
            )
            new_routing_scale = torch.cat(
                [self.routing_scale, torch.ones(1, device=device)]
            )
            new_routing_latched = torch.cat(
                [self.routing_latched,
                 torch.zeros(1, dtype=torch.bool, device=device)]
            )
            new_task_of_origin = torch.cat(
                [self.task_of_origin,
                 torch.tensor([int(task_idx)], dtype=torch.long, device=device)]
            )
            new_apoptosis_pulse = torch.cat(
                [self.apoptosis_pulse, torch.zeros(1, device=device)]
            )

        # Re-register parameters and buffers with new shapes.
        self._replace_parameter("W", new_W)
        self._replace_parameter("b", new_b)
        self._replace_buffer("lam", new_lam)
        self._replace_buffer("u", new_u)
        self._replace_buffer("W_anchor", new_W_anchor)
        self._replace_buffer("b_anchor", new_b_anchor)
        self._replace_buffer("fisher_W", new_fisher_W)
        self._replace_buffer("fisher_b", new_fisher_b)
        self._replace_buffer("routing_scale", new_routing_scale)
        self._replace_buffer("routing_latched", new_routing_latched)
        self._replace_buffer("task_of_origin", new_task_of_origin)
        self._replace_buffer("apoptosis_pulse", new_apoptosis_pulse)

        return self.n_nodes - 1

    def grow_input(self, init_col: Optional[torch.Tensor] = None) -> None:
        """Extend fan_in by 1, adding a new column to W. Used for cross-layer
        growth: when the previous layer adds a node, this layer must accept
        the extra input.

        init_col: optional length-n_nodes tensor for the new column. If None,
            zeros — the new input contributes nothing initially, and the
            network learns to use it via gradient descent.

        Per blueprint §4.1.4 ("Connecting it to all nodes whose `u` is currently
        elevated"), callers can pass a utility-weighted column to bias toward
        high-relevance peers.

        Same optimizer-rebuild caveat as grow_node: the W Parameter object
        is replaced.
        """
        device = self.W.device
        with torch.no_grad():
            if init_col is None:
                new_col = torch.zeros(
                    self.n_nodes, 1, dtype=self.W.dtype, device=device,
                )
            else:
                new_col = init_col.detach().to(device).reshape(self.n_nodes, 1)
            # Mixed-precision: cast to each destination's dtype.
            new_col_W = new_col.to(self.W.dtype)
            new_col_anchor = new_col.to(self.W_anchor.dtype)
            new_W = torch.cat([self.W.data, new_col_W], dim=1)
            new_W_anchor = torch.cat(
                [self.W_anchor, new_col_anchor.clone()], dim=1
            )
            new_fisher_W = torch.cat(
                [self.fisher_W,
                 torch.zeros(self.n_nodes, 1,
                             dtype=self.fisher_W.dtype, device=device)],
                dim=1,
            )

        self.fan_in += 1
        self._replace_parameter("W", new_W)
        self._replace_buffer("W_anchor", new_W_anchor)
        self._replace_buffer("fisher_W", new_fisher_W)

    def prune_input(self, col_idx: int) -> None:
        """Remove the input column at `col_idx`. Inverse of grow_input.

        Used for cross-layer pruning: when the previous layer removes a
        node, this layer must drop the corresponding input.

        Same optimizer-rebuild caveat as grow_node: the W Parameter object
        is replaced.
        """
        if not (0 <= col_idx < self.fan_in):
            raise IndexError(f"col_idx {col_idx} out of range [0, {self.fan_in})")
        if self.fan_in == 1:
            raise ValueError("Cannot prune the last remaining input column.")

        keep = [i for i in range(self.fan_in) if i != col_idx]
        keep_t = torch.tensor(keep, device=self.W.device, dtype=torch.long)

        with torch.no_grad():
            new_W = self.W.data.index_select(1, keep_t)
            new_W_anchor = self.W_anchor.index_select(1, keep_t)
            new_fisher_W = self.fisher_W.index_select(1, keep_t)

        self.fan_in -= 1
        self._replace_parameter("W", new_W)
        self._replace_buffer("W_anchor", new_W_anchor)
        self._replace_buffer("fisher_W", new_fisher_W)

    def prune_node(self, idx: int) -> None:
        """Remove the node at index `idx`. Same optimizer rebuild caveat as grow_node."""
        if not (0 <= idx < self.n_nodes):
            raise IndexError(f"node idx {idx} out of range [0, {self.n_nodes})")
        if self.n_nodes == 1:
            raise ValueError("Cannot prune the last remaining node.")

        keep = [i for i in range(self.n_nodes) if i != idx]
        keep_t = torch.tensor(keep, device=self.W.device, dtype=torch.long)

        with torch.no_grad():
            new_W = self.W.data.index_select(0, keep_t)
            new_b = self.b.data.index_select(0, keep_t)
            new_lam = self.lam.index_select(0, keep_t)
            new_u = self.u.index_select(0, keep_t)
            new_W_anchor = self.W_anchor.index_select(0, keep_t)
            new_b_anchor = self.b_anchor.index_select(0, keep_t)
            new_fisher_W = self.fisher_W.index_select(0, keep_t)
            new_fisher_b = self.fisher_b.index_select(0, keep_t)
            new_routing_scale = self.routing_scale.index_select(0, keep_t)
            new_routing_latched = self.routing_latched.index_select(0, keep_t)
            new_task_of_origin = self.task_of_origin.index_select(0, keep_t)
            new_apoptosis_pulse = self.apoptosis_pulse.index_select(0, keep_t)

        self._replace_parameter("W", new_W)
        self._replace_parameter("b", new_b)
        self._replace_buffer("lam", new_lam)
        self._replace_buffer("u", new_u)
        self._replace_buffer("W_anchor", new_W_anchor)
        self._replace_buffer("b_anchor", new_b_anchor)
        self._replace_buffer("fisher_W", new_fisher_W)
        self._replace_buffer("fisher_b", new_fisher_b)
        self._replace_buffer("routing_scale", new_routing_scale)
        self._replace_buffer("routing_latched", new_routing_latched)
        self._replace_buffer("task_of_origin", new_task_of_origin)
        self._replace_buffer("apoptosis_pulse", new_apoptosis_pulse)

    # ----- low-level helpers -----

    def _replace_parameter(self, name: str, new_tensor: torch.Tensor) -> None:
        if name in self._parameters:
            del self._parameters[name]
        setattr(self, name, nn.Parameter(new_tensor))

    def _replace_buffer(self, name: str, new_tensor: torch.Tensor) -> None:
        if name in self._buffers:
            del self._buffers[name]
        self.register_buffer(name, new_tensor)

    def __repr__(self) -> str:
        return (
            f"TrioronLayer(fan_in={self.fan_in}, n_nodes={self.n_nodes}, "
            f"activation='{self.activation}')"
        )
