"""Trioron — pruning controller and utility-contribution helper.

Implements §3.3 + §8 step 6: a slower-clock pruning loop that removes
nodes whose utility u has been below `u_threshold` for `T_prune` steps,
with cosine-similarity redistribution handled by
TrioronNetwork.prune_layer_node.

Per blueprint §3.2 the contribution is:
    contribution_t = sign(reward_t) · |activation_t · gradient_t|

For the contrastive incubation task there is no external scalar reward
AND once the task converges the gradient term collapses to zero across
all nodes (used and unused alike) — the |a·g| signal can no longer tell
them apart. So this module supports three contribution modes:

    "act_grad"   — |activation · gradient|, the literal blueprint form.
                   Best during active learning; degenerate at convergence.
    "act_var"    — variance of activation across the batch. Structural
                   "is this node alive" signal independent of task
                   progress; non-zero whenever the node fires differently
                   for different inputs.
    "combined"   — max of the two. Default. A node is "used" if either
                   signal says so.

The controller intentionally re-derives streak counters from the running
u buffer, not from per-step cumulative dictionaries, so that structural
changes (which shift node indices) don't corrupt tracking.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import torch

from .network import TrioronNetwork


# ---------------------------------------------------------------------
# Activation × gradient contribution (§3.2)
# ---------------------------------------------------------------------


_VALID_MODES = ("act_grad", "act_var", "combined")


class _ContribCapture:
    """Forward-hook helper that retains each layer's outputs so we can
    read .grad on them after backward(). One capture per training step."""

    def __init__(self, net: TrioronNetwork, mode: str = "combined"):
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self.net = net
        self.mode = mode
        self.captured: List[List[torch.Tensor]] = [[] for _ in net.layers]
        self._handles = []

    def __enter__(self):
        for i, layer in enumerate(self.net.layers):
            def make_hook(idx):
                def hook(_mod, _inp, out):
                    if out.requires_grad:
                        out.retain_grad()
                        self.captured[idx].append(out)
                return hook
            self._handles.append(layer.register_forward_hook(make_hook(i)))
        return self

    def __exit__(self, *_exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def update_layer_utilities(self) -> None:
        """After loss.backward(), aggregate captured |h * h.grad| across
        every forward pass and feed each layer's update_utility().

        Safe to call when no captures occurred (e.g. the layer was not
        invoked this step) — those layers' u is left unchanged.
        """
        for layer_idx, layer in enumerate(self.net.layers):
            contribs: List[torch.Tensor] = []
            for h in self.captured[layer_idx]:
                h_d = h.detach()
                if h_d.dim() != 2:
                    continue

                if self.mode in ("act_grad", "combined"):
                    if h.grad is not None:
                        ag = (h_d * h.grad.detach()).abs().mean(dim=0)
                    else:
                        ag = torch.zeros(layer.n_nodes, device=h_d.device)
                else:
                    ag = None

                if self.mode in ("act_var", "combined"):
                    if h_d.shape[0] >= 2:
                        av = h_d.var(dim=0, unbiased=False)
                    else:
                        av = h_d.abs().squeeze(0)
                else:
                    av = None

                if self.mode == "act_grad":
                    c = ag
                elif self.mode == "act_var":
                    c = av
                else:  # combined
                    c = torch.maximum(ag, av)
                contribs.append(c)

            if contribs:
                stacked = torch.stack(contribs)
                contribution = stacked.mean(dim=0)
                if contribution.shape == (layer.n_nodes,):
                    layer.update_utility(contribution)


def utility_capture(net: TrioronNetwork, mode: str = "combined") -> _ContribCapture:
    """Context manager. Use as:

        with utility_capture(net) as cap:
            loss = compute_loss(net, ...)
            loss.backward()
            cap.update_layer_utilities()   # before optimizer.step()
        optimizer.step()

    `mode` selects the contribution signal:
        "act_grad"  — |activation · gradient| (blueprint §3.2 literal form)
        "act_var"   — variance of activation across the batch
        "combined"  — elementwise max of the above (default; robust at
                      convergence when act_grad collapses)
    """
    return _ContribCapture(net, mode=mode)


# ---------------------------------------------------------------------
# Pruning controller (§3.3 + §8 step 6)
# ---------------------------------------------------------------------


class PruningController:
    """Runs on a slower clock than growth; removes chronically low-utility
    nodes with cosine-similarity redistribution.

    Behavior per call to maybe_prune:
    - Every step: update internal "consecutive-low-u" streak counters per
      (layer_idx, node_idx) from the layer's current u buffer.
    - On clock ticks (step % prune_clock == 0, step > 0): scan streaks,
      identify nodes whose streak has reached T_prune, prune them in
      descending node-index order so deletions don't invalidate the
      remaining indices in the same layer.
    - After any pruning happens, clear all streak counters (indices have
      shifted, peer relationships changed — start fresh).

    Safety:
    - Refuses to prune the last remaining node in any layer.
    - `protect_layers` set lets callers exclude specific layers (default:
      none — both first and last layer nodes are eligible by design).
    """

    def __init__(
        self,
        u_threshold: float = 1e-3,
        T_prune: int = 2000,
        prune_clock: int = 500,
        protect_layers: Optional[List[int]] = None,
    ):
        if T_prune < 1:
            raise ValueError("T_prune must be >= 1")
        if prune_clock < 1:
            raise ValueError("prune_clock must be >= 1")
        self.u_threshold = u_threshold
        self.T_prune = T_prune
        self.prune_clock = prune_clock
        self.protect_layers = set(protect_layers or [])
        self._streak: dict[Tuple[int, int], int] = {}

    # ----- per-step bookkeeping -----

    def step(self, net: TrioronNetwork, step_idx: int) -> List[Tuple[int, int]]:
        """Update streaks; if at a clock tick, prune candidates and return
        the list of (layer_idx, node_idx) pruned this call."""
        self._update_streaks(net)
        if step_idx == 0 or step_idx % self.prune_clock != 0:
            return []
        return self._prune_candidates(net)

    # ----- helpers -----

    def _update_streaks(self, net: TrioronNetwork) -> None:
        active_keys: set = set()
        for L_idx, layer in enumerate(net.layers):
            if L_idx in self.protect_layers:
                continue
            with torch.no_grad():
                u_vec = layer.u.detach().cpu()
            for n_idx in range(layer.n_nodes):
                key = (L_idx, n_idx)
                active_keys.add(key)
                if u_vec[n_idx].item() < self.u_threshold:
                    self._streak[key] = self._streak.get(key, 0) + 1
                else:
                    self._streak[key] = 0
        # Drop streaks for any key that doesn't exist anymore (safety).
        for stale in list(self._streak.keys()):
            if stale not in active_keys:
                del self._streak[stale]

    def _prune_candidates(self, net: TrioronNetwork) -> List[Tuple[int, int]]:
        candidates = [
            (L_idx, n_idx)
            for (L_idx, n_idx), streak in self._streak.items()
            if streak >= self.T_prune
        ]
        # Descending node-index within each layer so deletions don't
        # invalidate larger indices in the same layer mid-loop.
        candidates.sort(key=lambda x: (x[0], -x[1]))

        pruned: List[Tuple[int, int]] = []
        for L_idx, n_idx in candidates:
            layer = net.layers[L_idx]
            if layer.n_nodes <= 1:
                continue
            net.prune_layer_node(L_idx, n_idx, redistribute=True)
            pruned.append((L_idx, n_idx))

        if pruned:
            # Indices have shifted across the network; reset all tracking.
            self._streak.clear()
        return pruned

    # ----- diagnostics -----

    def streak_snapshot(self) -> dict:
        return dict(self._streak)

    def __repr__(self) -> str:
        return (
            f"PruningController(u_threshold={self.u_threshold}, "
            f"T_prune={self.T_prune}, prune_clock={self.prune_clock}, "
            f"protect_layers={sorted(self.protect_layers)})"
        )
