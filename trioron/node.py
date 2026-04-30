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

    # ----- properties -----
    @property
    def n_nodes(self) -> int:
        return self.W.shape[0]

    # ----- forward -----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.linear(x, self.W, self.b)
        return _ACTIVATIONS[self.activation](z)

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

        Per blueprint §3.2: λ ≈ mean Fisher info across the node's incoming weights.
        Higher λ ⇒ this node's weights matter a lot for current loss ⇒ stiffer.
        """
        with torch.no_grad():
            self.lam.copy_(self.fisher_W.mean(dim=1))

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

        Returns a scalar tensor (autograd-attached when W requires grad).
        """
        # Per-node λ broadcasts across that node's incoming weights.
        stiffness = self.lam.unsqueeze(1)  # (n_nodes, 1)
        pen_W = (stiffness * (self.W - self.W_anchor).pow(2)).sum()
        pen_b = (self.lam * (self.b - self.b_anchor).pow(2)).sum()
        return pen_W + pen_b

    # ----- structural plasticity -----

    def grow_node(self, init_vec: Optional[torch.Tensor] = None) -> int:
        """Add one new node to the layer. Returns the index of the new node.

        init_vec: optional length-fan_in tensor for the new node's incoming
            weights. If None, uses the same Kaiming-style init as construction.

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

            new_W = torch.cat([self.W.data, new_row], dim=0)
            new_b = torch.cat([self.b.data, torch.zeros(1, device=device)], dim=0)
            new_lam = torch.cat([self.lam, torch.zeros(1, device=device)])
            new_u = torch.cat([self.u, torch.zeros(1, device=device)])
            new_W_anchor = torch.cat([self.W_anchor, new_row.clone()], dim=0)
            new_b_anchor = torch.cat([self.b_anchor, torch.zeros(1, device=device)])
            new_fisher_W = torch.cat(
                [self.fisher_W, torch.zeros(1, self.fan_in, device=device)], dim=0
            )
            new_fisher_b = torch.cat([self.fisher_b, torch.zeros(1, device=device)])

        # Re-register parameters and buffers with new shapes.
        self._replace_parameter("W", new_W)
        self._replace_parameter("b", new_b)
        self._replace_buffer("lam", new_lam)
        self._replace_buffer("u", new_u)
        self._replace_buffer("W_anchor", new_W_anchor)
        self._replace_buffer("b_anchor", new_b_anchor)
        self._replace_buffer("fisher_W", new_fisher_W)
        self._replace_buffer("fisher_b", new_fisher_b)

        return self.n_nodes - 1

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

        self._replace_parameter("W", new_W)
        self._replace_parameter("b", new_b)
        self._replace_buffer("lam", new_lam)
        self._replace_buffer("u", new_u)
        self._replace_buffer("W_anchor", new_W_anchor)
        self._replace_buffer("b_anchor", new_b_anchor)
        self._replace_buffer("fisher_W", new_fisher_W)
        self._replace_buffer("fisher_b", new_fisher_b)

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
