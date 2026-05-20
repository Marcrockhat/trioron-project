"""Trioron node — the fundamental unit of the dynamic network.

Implements the Trioron node specification from blueprint §3.

A "node" in this architecture is characterized by three coupled state variables:
    w  — the row of incoming weights (a vector of length fan_in)
    λ  — per-node plasticity gate (scalar, ≥0). Higher = stiffer. The paper
         formulation treats λ as a stand-in for biological gating mechanisms
         (BDNF methylation, perineuronal-net maturation) — environmentally
         regulated, NOT exclusively cognitive. Any upstream signal can write
         it: Fisher information (the canonical EWC channel via
         update_fisher → update_lambda), but equally environmental sensors
         (temperature, light, stress proxies), reward magnitudes, attention
         masks, hand-injected priors for cells you want frozen, or anything
         else the deployment context provides. Use `set_lambda()` to write
         from arbitrary signals. The substrate doesn't care where λ came
         from — only that it gates how rigid each cell is.
    u  — per-node utility score (scalar). Tracks contribution to good outputs.

For efficiency, many nodes are stored together inside a TrioronLayer, where
each row of the weight matrix W corresponds to one node's incoming weights,
and λ / u are vectors with one entry per node.

The layer supports:
    - Forward pass with a configurable activation
    - Per-node Fisher-information estimation (running EMA of squared gradients)
    - λ derived from Fisher info (one signal source among many)
    - λ written from arbitrary external signals via set_lambda()
    - Utility update from an externally-supplied contribution signal
    - EWC quadratic penalty term against an anchor snapshot
    - Growth (add a new node) and pruning (remove a node)

Notes for callers:
    - After grow_node() or prune_node() you MUST rebuild any optimizer that
      held references to the old W or b parameters. The Parameter objects are
      replaced.
    - update_fisher() must be called after .backward() but before optimizer.step()
      so that .grad is still populated.
    - The trioron node has three coupled state variables (w, λ, u). λ and u
      are initialized to zero. Most users populate λ via the EWC consolidation
      cycle:
          update_fisher()       per batch (after .backward(), before .step())
          update_lambda()       at task end
          anchor_weights()      at task end
      A training loop that skips this cycle AND doesn't write λ from another
      source leaves λ at zero — ewc_penalty() is then mathematically zero
      regardless of the caller's β / ewc_strength, growth triggers have no
      Fisher signal, and dream consolidation has nothing to consolidate.
      The network keeps training and looks healthy, but the substrate has
      silently degraded into a regular MLP wearing a Trioron shell. After
      training, sanity-check `layer.lam.max() > 0`; ewc_penalty() also emits
      a one-shot RuntimeWarning when called against an all-zero λ.

      Fisher is NOT the only valid signal for λ. The (w, λ, u) formulation
      models λ as a per-cell plasticity gate, equivalent to environmentally
      regulated biological mechanisms — environmental sensors, reward
      magnitudes, attention masks, or any externally-derived per-cell
      importance signal are all legitimate sources. Use set_lambda() to
      write λ from any such signal; the substrate behaves identically
      regardless of where the values came from.
"""

from __future__ import annotations
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_ACTIVATIONS = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "linear": lambda x: x,
}


# Per-branch local nonlinearity (σ_branch, Trioron 2.0 Axis 5).
# Applied to each branch's pre-pool sum in cells with K>1 only — the K=1
# fast path bypasses σ_branch entirely. "quad" is the live default for
# newly-constructed layers (NMDA-style supralinear, Poirazi & Mel 2003);
# "identity" is the v1-donor default (point-neuron-equivalent even at K>1).
_BRANCH_ACTIVATIONS = {
    "quad": lambda z: z * z,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "identity": lambda z: z,
}


# One-shot warning state for ewc_penalty's silent-zero check. Process-wide
# so a single training run doesn't get spammed once per layer per call.
# Test hook: _EwcZeroWarning.reset() re-arms the warning between cases.
class _EwcZeroWarning:
    _warned: bool = False

    @classmethod
    def reset(cls) -> None:
        cls._warned = False


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
        branch_activation: Optional[str] = None,
        B_max: Optional[int] = None,
    ):
        super().__init__()
        # Axis 5 defaults consult the currently-active TrioronProfile
        # when the caller leaves them None. Explicit kwargs always
        # override. The default active profile is OPEN, which
        # reproduces 1.0-era construction defaults byte-for-byte
        # (branch_activation="quad", B_max=8). Imported lazily to
        # avoid a circular dependency at module init.
        if branch_activation is None or B_max is None:
            from trioron.profile import TrioronProfile
            active = TrioronProfile.active()
            if branch_activation is None:
                branch_activation = active.branch_activation
            if B_max is None:
                B_max = active.B_max
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Supported: {list(_ACTIVATIONS.keys())}"
            )
        if branch_activation not in _BRANCH_ACTIVATIONS:
            raise ValueError(
                f"Unknown branch_activation '{branch_activation}'. "
                f"Supported: {list(_BRANCH_ACTIVATIONS.keys())}"
            )
        if B_max < 1:
            raise ValueError(f"B_max must be >= 1, got {B_max}")

        self.fan_in = int(fan_in)
        self.activation = activation
        # Trioron 2.0 Axis 5 (dendritic compartmentalization). σ_branch is
        # the per-branch local nonlinearity applied inside cells with K>1.
        # "quad" is the live default for fresh substrates; v1-loaded donors
        # are auto-flipped to "identity" by _load_from_state_dict so they
        # remain point-neuron-equivalent even if later grown to K>1.
        self.branch_activation = branch_activation
        self.B_max = int(B_max)
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
        # Anchored routing_scale: snapshot of routing_scale at the most
        # recent consolidate. The trioron node is triparametric (w, λ, u);
        # u feeds forward via routing_scale, so reconstructing the
        # consolidated network requires anchoring routing_scale alongside
        # W and b. Without this, forward_with_anchors mixes anchored W
        # with live routing_scale — which diverges from the consolidated
        # state whenever dream-rescue mutates routing mid-task. Used by
        # forward_with_anchors (LwF distillation target).
        self.register_buffer("routing_scale_anchor", torch.ones(n_nodes))

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

        # Archive flag (Phase 1 of dream-archive stage). A row marked
        # archived is "developmentally closed": its W is locked at the
        # consolidated state, gradients don't accumulate, EWC and Fisher
        # skip it, growth and apoptosis don't target it. Maps onto
        # cortical long-term consolidation — settled memories move from
        # active machinery to durable read-only state. Phase 2 will
        # additionally drop W_anchor / fisher / Adam moments and
        # quantize W to int8 for actual RAM savings.
        self.register_buffer(
            "archived", torch.zeros(n_nodes, dtype=torch.bool),
        )
        # Streak counter: number of consecutive consolidations during
        # which this row's λ was in the layer's top percentile. Used by
        # archive_block as one of the triggers (high λ across N
        # consolidations = candidate for archive). Reset on percentile
        # miss; incremented on hit by the archive trigger logic.
        self.register_buffer(
            "lam_high_streak", torch.zeros(n_nodes, dtype=torch.long),
        )

        # Per-column input provenance (Trioron 2.0 axis 1: long-range
        # reach). Shape (fan_in, 2) long. Entry [j] = (src_layer_idx,
        # src_node_idx) declares which earlier node feeds column j of
        # W. The sentinel value (-1, -1) means "use the immediate
        # sequential predecessor's row j" — at construction every
        # column is sentinel, preserving 1.0 byte-identical forward
        # behavior. TrioronNetwork.forward interprets the buffer; the
        # layer itself just stores it and keeps it aligned with
        # grow_input / prune_input. See trioron_2_0.md §3.1.
        self.register_buffer(
            "input_sources",
            torch.full((fan_in, 2), -1, dtype=torch.long),
        )

        # Per-column archive flag (Trioron 2.0 axis 2: plastic fanout
        # sparsity). Column-side mirror of the row-level `archived`
        # buffer. A column flagged here is "developmentally closed":
        # W column is locked at the anchored state, gradient is
        # masked via mask_archived_input_grads, Fisher is zeroed.
        # Source-side apoptosis (severing an archived source's
        # outgoing fanout) is implemented at the network level by
        # calling archive_input on every column whose input_sources
        # points at the archived source. See trioron_2_0.md §3.2.
        self.register_buffer(
            "input_archived", torch.zeros(fan_in, dtype=torch.bool),
        )

        # Per-source axonal gain (Trioron 2.0 axis 4). Source-side
        # mirror of `routing_scale` — scales each NODE's outgoing
        # contribution into every downstream destination it reaches.
        # Default 1.0 preserves byte-identical 1.0 forward behavior.
        # The network's gather step multiplies y[src_layer][src_node]
        # by axonal_gain[src_node] before feeding it as input to any
        # destination column tagged with that source. Slow time-scale,
        # plastic; write via set_axonal_gain from any external signal
        # (reward magnitude, attention mask, emotional-tag broadcast,
        # manual prior). axonal_gain_anchor preserves the triparametric
        # anchored-state contract analogous to routing_scale_anchor.
        # See trioron_2_0.md §3.4.
        self.register_buffer("axonal_gain", torch.ones(n_nodes))
        self.register_buffer("axonal_gain_anchor", torch.ones(n_nodes))

        # Dendritic compartmentalization (Trioron 2.0 Axis 5). Each cell
        # gains an internal flat partition of its fan_in columns into
        # branches, each branch pooled with its own soma-side weight.
        # At K=1 (every cell starts here), the forward bypasses σ_branch
        # entirely and reduces to F.linear — byte-identical to 1.0. The
        # K>1 path is exercised once grow_branch lands (Phase 2.5).
        #
        #   branch_id[i, j]  — column j of cell i belongs to branch_id[i, j]
        #                       ∈ [0, B_per_node[i]). Init all-zero ⇒ every
        #                       column on branch 0 ⇒ single-branch point neuron.
        #   branch_weight    — soma pooling weight per (cell, branch). Plastic
        #                       via gradient. Init [1, 0, ..., 0] per cell so
        #                       the K=1 pool is exactly the branch-0 sum.
        #   branch_weight_anchor / fisher_branch_weight — anchored-state
        #                       mirror + per-branch Fisher; participate in
        #                       ewc_penalty alongside W and b.
        #   B_per_node[i]    — current branch count for cell i. Increments
        #                       on grow_branch, decrements on prune_branch.
        #   internal_stress[i] — EMA of |∂L/∂y_i| · engaged(y_i). Per-cell
        #                       within-niche frustration signal. Phase 2.5
        #                       uses it to trigger grow_branch.
        #   branch_utility[i, b] — EMA of |branch_weight[i,b] · y_{i,b}|.
        #                       Mozer/Smolensky saliency at branch granularity.
        #                       Phase 2.5 uses it to trigger prune_branch.
        # See trioron_2_0.md §3.5.
        self.register_buffer(
            "branch_id",
            torch.zeros(n_nodes, fan_in, dtype=torch.long),
        )
        self.branch_weight = nn.Parameter(
            torch.zeros(n_nodes, self.B_max)
        )
        with torch.no_grad():
            self.branch_weight.data[:, 0] = 1.0
        self.register_buffer(
            "branch_weight_anchor", self.branch_weight.detach().clone(),
        )
        self.register_buffer(
            "fisher_branch_weight", torch.zeros(n_nodes, self.B_max),
        )
        self.register_buffer(
            "B_per_node", torch.ones(n_nodes, dtype=torch.long),
        )
        self.register_buffer("internal_stress", torch.zeros(n_nodes))
        self.register_buffer(
            "branch_utility", torch.zeros(n_nodes, self.B_max),
        )
        # Per-(cell, column) orphan mask (Trioron 2.0 Axis 5 — Phase 2.5).
        # Flipped True when prune_branch removes a branch: the columns
        # that belonged to the pruned branch are "orphaned" for THIS
        # cell — its forward zeros them out — but other cells reading
        # the same upstream source remain unaffected (the cell-level
        # mask is finer-grained than `input_archived`, which is
        # column-wide). Default all-False = no orphaned columns =
        # byte-identical fast path.
        self.register_buffer(
            "dendrite_orphan",
            torch.zeros(n_nodes, fan_in, dtype=torch.bool),
        )

        # Saliency caches for |a · g| utility (Mozer & Smolensky 1989,
        # OBD/blueprint §3.2). _last_y is stashed on each grad-enabled
        # forward; _last_upstream is captured by a backward hook on y
        # when .backward() propagates through. saliency_utility()
        # combines them. Both are transient — NOT registered as
        # buffers so they don't pollute state_dict / serialization.
        self._last_y: Optional[torch.Tensor] = None
        self._last_upstream: Optional[torch.Tensor] = None
        # Per-branch caches from the K>1 dendritic forward (Trioron 2.0
        # Phase 2.5). _last_y_branches is (batch, n_nodes, B_max), used
        # by update_branch_utility for per-(cell, branch) saliency EMA.
        # Only populated when the dendritic path runs (any cell K>1);
        # the K=1 fast path leaves both at None. Transient, no buffer.
        self._last_y_branches: Optional[torch.Tensor] = None
        self._last_z_branches: Optional[torch.Tensor] = None

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

        # Axis 5 two-stage forward. All-K=1 is the entire installed base
        # (every 1.0 donor + every freshly-constructed substrate before
        # grow_branch fires) and takes the F.linear fast path —
        # byte-identical to 1.0. The .all().item() is a single-scalar
        # CPU sync per forward; acceptable until Phase 2.5 introduces
        # branch mutation, at which point we'll cache the flag.
        if bool((self.B_per_node == 1).all().item()):
            z = F.linear(x, W_eff, self.b)
        else:
            z = self._forward_dendritic(x, W_eff)

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

    def _forward_dendritic(
        self, x: torch.Tensor, W_eff: torch.Tensor,
    ) -> torch.Tensor:
        """K>1 dendritic forward (Trioron 2.0 Axis 5).

        Returns the soma pre-activation (n_nodes, batch) → batch-first
        (batch, n_nodes), which the caller then runs through σ_soma.

        Per-cell K=1 shortcut: cells with B_per_node[i] == 1 still take
        σ_branch=identity inside this path (matching the fast-path
        semantics), so a K=1 cell embedded in a mixed-K population stays
        point-neuron-equivalent. Only K>1 cells see σ_branch.
        """
        batch_size = x.shape[0]
        n_nodes = self.n_nodes
        B_max = self.branch_weight.shape[1]

        # contrib[batch, i, j] = x[batch, j] · W_eff[i, j].
        # (batch, 1, fan_in) × (1, n_nodes, fan_in) → (batch, n_nodes, fan_in).
        # This materializes a (batch, n_nodes, fan_in) tensor — fine at
        # typical layer sizes, and only hit when at least one cell has
        # grown to K>1 (Phase 2.5).
        contrib = x.unsqueeze(1) * W_eff.unsqueeze(0)

        # Phase 2.5: zero out orphaned (cell, column) pairs before the
        # scatter. dendrite_orphan[i, j] = True means cell i's pruned
        # branch carried column j and the cell now skips it. Other cells
        # reading the same upstream source are unaffected.
        if self.dendrite_orphan.any():
            keep = (~self.dendrite_orphan).unsqueeze(0).to(contrib.dtype)
            contrib = contrib * keep

        # scatter_add per-branch sum: z_branches[batch, i, b] =
        #   Σ_{j : branch_id[i, j] == b} contrib[batch, i, j].
        z_branches = contrib.new_zeros(batch_size, n_nodes, B_max)
        idx = self.branch_id.unsqueeze(0).expand(batch_size, -1, -1)
        z_branches = z_branches.scatter_add(2, idx, contrib)

        # σ_branch on all cells uniformly, then per-cell shortcut
        # restores identity for K=1 cells. is_k1 is a (1, n_nodes, 1)
        # float mask broadcast across (batch, n_nodes, B_max).
        y_branches_raw = _BRANCH_ACTIVATIONS[self.branch_activation](z_branches)
        is_k1 = (
            (self.B_per_node == 1).view(1, n_nodes, 1).to(z_branches.dtype)
        )
        y_branches = is_k1 * z_branches + (1.0 - is_k1) * y_branches_raw

        # Cache per-branch tensors for update_branch_utility / debugging.
        # Detach so the EMA path doesn't pull gradient back through here.
        # Only populated when this dendritic path actually runs.
        self._last_z_branches = z_branches.detach()
        self._last_y_branches = y_branches.detach()

        # Soma pool: Σ_b branch_weight[i, b] · y_branches[batch, i, b].
        bw = self.branch_weight.to(self.W.dtype).unsqueeze(0)
        soma_input = (bw * y_branches).sum(dim=2)
        return soma_input + self.b

    def _capture_upstream(self, grad: torch.Tensor) -> None:
        """Backward hook: record ∂L/∂y for saliency_utility()."""
        self._last_upstream = grad.detach()

    # ----- state updates (call in this order during training) -----

    def update_fisher(self) -> None:
        """Update running estimate of diagonal Fisher info from current gradients.

        Call AFTER loss.backward() and BEFORE optimizer.step(). Archived
        rows are skipped — they're locked at consolidated state and have
        no gradient signal worth tracking.

        Trioron 2.0 Axis 5: fisher_branch_weight tracks per-(cell,
        branch) squared gradient on the dendritic pool weights, mirroring
        fisher_W's treatment of W. Enters ewc_penalty at per-branch
        granularity (unlike fisher_W, which is pre-collapsed into per-cell
        λ via update_lambda).
        """
        if self.W.grad is None:
            return
        with torch.no_grad():
            grad_W_sq = self.W.grad.detach().pow(2)
            if self.archived.any():
                grad_W_sq = grad_W_sq.clone()
                grad_W_sq[self.archived] = 0.0
            self.fisher_W.mul_(self.fisher_decay).add_(
                grad_W_sq, alpha=1.0 - self.fisher_decay,
            )
            if self.b.grad is not None:
                grad_b_sq = self.b.grad.detach().pow(2)
                if self.archived.any():
                    grad_b_sq = grad_b_sq.clone()
                    grad_b_sq[self.archived] = 0.0
                self.fisher_b.mul_(self.fisher_decay).add_(
                    grad_b_sq, alpha=1.0 - self.fisher_decay,
                )
            if self.branch_weight.grad is not None:
                grad_bw_sq = self.branch_weight.grad.detach().pow(2)
                if self.archived.any():
                    grad_bw_sq = grad_bw_sq.clone()
                    grad_bw_sq[self.archived] = 0.0
                self.fisher_branch_weight.mul_(self.fisher_decay).add_(
                    grad_bw_sq, alpha=1.0 - self.fisher_decay,
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

        Fisher is the canonical EWC source for λ but not the only one;
        see set_lambda() for writing λ from arbitrary external signals.
        """
        with torch.no_grad():
            self.lam.copy_(self.fisher_W.sum(dim=1))

    def set_lambda(
        self,
        signal: torch.Tensor,
        mode: str = "absolute",
    ) -> None:
        """Write the per-node plasticity gate λ from an arbitrary signal.

        λ is the per-cell plasticity coefficient — high λ stiffens the
        cell against EWC drift, low λ leaves it plastic. The (w, λ, u)
        formulation treats λ as a stand-in for environmentally regulated
        biological gating (BDNF methylation, perineuronal-net
        maturation); Fisher information is one signal that can drive it,
        but any upstream source is legitimate:

          - environmental sensors (temperature, light, stress proxies on
            an edge device) — λ becomes a literal "environment sense"
          - reward magnitudes — high-reward cells get protected
          - attention masks — task-salient cells stiffen
          - hand-injected priors — freeze specific cells (set to large λ)
            or wake them (set to 0)

        signal: shape (n_nodes,). Cast to layer dtype/device.
        mode:
          "absolute"       λ ← signal             (replace)
          "additive"       λ ← λ + signal         (layer on top of Fisher)
          "multiplicative" λ ← λ * signal         (scale, e.g. a sleep
                                                  cycle in [0, 1])

        The result is clamped to ≥0 — ewc_penalty() interprets λ as a
        non-negative stiffness, and negative values would invert the
        penalty into a push AWAY from the anchor.
        """
        if signal.shape != (self.n_nodes,):
            raise ValueError(
                f"signal shape {tuple(signal.shape)} "
                f"!= (n_nodes={self.n_nodes},)"
            )
        if mode not in ("absolute", "additive", "multiplicative"):
            raise ValueError(
                f"mode '{mode}' not in 'absolute' / 'additive' / 'multiplicative'"
            )
        sig = signal.detach().to(device=self.lam.device, dtype=self.lam.dtype)
        with torch.no_grad():
            if mode == "absolute":
                self.lam.copy_(sig)
            elif mode == "additive":
                self.lam.add_(sig)
            else:  # multiplicative
                self.lam.mul_(sig)
            self.lam.clamp_(min=0.0)

    def set_axonal_gain(
        self,
        signal: torch.Tensor,
        mode: str = "absolute",
    ) -> None:
        """Write the per-source axonal gain from an arbitrary signal.

        Trioron 2.0 axis 4: axonal_gain is the source-side
        multiplicative gain on each node's downstream broadcast. It is
        the substrate-level analog of attention / emotional /
        neuromodulatory state — a small set of "broadcaster" nodes can
        scale their entire outgoing influence without touching any edge
        weight. Slow time-scale; written from any external signal:

          - reward magnitudes — high-reward nodes get amplified outgoing
          - attention masks — task-salient nodes broadcast louder
          - emotional-tag node outputs (when output_sinks land in a
            future tweak) — emotion modulates other regions' inputs
          - hand-injected priors — silence a node by setting gain to 0

        signal: shape (n_nodes,). Cast to layer dtype/device.
        mode:
          "absolute"       axonal_gain ← signal     (replace)
          "additive"       axonal_gain ← axonal_gain + signal
          "multiplicative" axonal_gain ← axonal_gain * signal

        Clamped to ≥0 — negative gain would flip the source's outgoing
        contribution sign, which is not the modulatory semantics this
        buffer models. (Inhibitory edges are a separate future tweak.)
        """
        if signal.shape != (self.n_nodes,):
            raise ValueError(
                f"signal shape {tuple(signal.shape)} "
                f"!= (n_nodes={self.n_nodes},)"
            )
        if mode not in ("absolute", "additive", "multiplicative"):
            raise ValueError(
                f"mode '{mode}' not in 'absolute' / 'additive' / 'multiplicative'"
            )
        sig = signal.detach().to(
            device=self.axonal_gain.device, dtype=self.axonal_gain.dtype,
        )
        with torch.no_grad():
            if mode == "absolute":
                self.axonal_gain.copy_(sig)
            elif mode == "additive":
                self.axonal_gain.add_(sig)
            else:
                self.axonal_gain.mul_(sig)
            self.axonal_gain.clamp_(min=0.0)

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
        Also snapshots routing_scale, axonal_gain, and branch_weight
        (Trioron 2.0 Axis 5) so forward_with_anchors can reconstruct
        the consolidated state across all per-node and per-branch gains.
        """
        with torch.no_grad():
            self.W_anchor.copy_(self.W.detach())
            self.b_anchor.copy_(self.b.detach())
            self.routing_scale_anchor.copy_(self.routing_scale.detach())
            self.axonal_gain_anchor.copy_(self.axonal_gain.detach())
            self.branch_weight_anchor.copy_(self.branch_weight.detach())

    def archive_row(self, idx: int) -> None:
        """Mark row `idx` as archived: snap W and b to their anchored
        values, zero its Fisher row, set λ to 0, mark archived. The row
        becomes "developmentally closed" — gradients still flow through
        it (so upstream layers can adapt to its fixed contribution) but
        do not accumulate to it (mask_archived_grads zeros W.grad and
        b.grad at archived rows after backward, before optimizer.step).

        Idempotent: archiving an already-archived row is a no-op.
        """
        if not (0 <= idx < self.n_nodes):
            raise IndexError(
                f"archive_row idx {idx} out of range [0, {self.n_nodes})"
            )
        if bool(self.archived[idx]):
            return
        with torch.no_grad():
            # Lock W at consolidated state.
            self.W.data[idx].copy_(self.W_anchor[idx])
            self.b.data[idx].copy_(self.b_anchor[idx])
            # Drop Fisher / λ contributions for this row.
            self.fisher_W[idx].zero_()
            self.fisher_b[idx].zero_()
            self.lam[idx] = 0.0
            # Reset utility / streak / apoptosis pulse — archive is a
            # terminal state; these signals don't apply anymore.
            self.u[idx] = 0.0
            self.lam_high_streak[idx] = 0
            self.apoptosis_pulse[idx] = 0.0
            self.archived[idx] = True

    def mask_archived_grads(self) -> None:
        """Zero W.grad / b.grad at archived rows. Call AFTER .backward()
        and BEFORE optimizer.step() (and before update_fisher, so Fisher
        sees the masked grads consistently). No-op if no rows archived.
        """
        if not self.archived.any():
            return
        with torch.no_grad():
            if self.W.grad is not None:
                self.W.grad[self.archived] = 0.0
            if self.b.grad is not None:
                self.b.grad[self.archived] = 0.0

    def archive_input(self, col_idx: int) -> None:
        """Mark input column `col_idx` as archived (Trioron 2.0 axis 2).
        Column-side mirror of `archive_row`. The column is locked at
        the anchored state: W[:, col_idx] snaps to W_anchor[:, col_idx],
        fisher_W[:, col_idx] is zeroed, input_archived[col_idx] flips
        True. Gradient flow into the column is masked by
        mask_archived_input_grads (call after .backward(), before
        optimizer.step()).

        b is per-row, so this method does not touch it. Likewise λ
        and u are per-row (per-destination-node); they aggregate
        Fisher across fan_in but the zeroed fisher column drops out
        of the sum automatically on the next update_lambda.

        Idempotent: archiving an already-archived column is a no-op.
        """
        if not (0 <= col_idx < self.fan_in):
            raise IndexError(
                f"archive_input col_idx {col_idx} out of range "
                f"[0, {self.fan_in})"
            )
        if bool(self.input_archived[col_idx]):
            return
        with torch.no_grad():
            self.W.data[:, col_idx].copy_(self.W_anchor[:, col_idx])
            self.fisher_W[:, col_idx].zero_()
            self.input_archived[col_idx] = True

    def mask_archived_input_grads(self) -> None:
        """Zero W.grad at archived input columns. Call AFTER .backward()
        and BEFORE optimizer.step() (and before update_fisher). No-op
        if no columns archived. Column-side mirror of
        mask_archived_grads.
        """
        if not self.input_archived.any():
            return
        with torch.no_grad():
            if self.W.grad is not None:
                self.W.grad[:, self.input_archived] = 0.0

    def ewc_penalty(self) -> torch.Tensor:
        """Quadratic penalty pulling weights toward the anchor.

        Per-node λ scales the strength. Total task loss should be:
            L = L_task + ewc_strength * layer.ewc_penalty()

        The apoptosis_pulse buffer (Phase 4.5 Experiment 5) lowers each
        node's effective λ by (1 - pulse) clamped to ≥0 — when a
        neighbor cell latches dead, surviving peers get a temporary
        plasticity boost that fades as the pulse decays.

        Returns a scalar tensor (autograd-attached when W requires grad).

        Silent-zero guard: if λ is all zero we emit a one-shot
        RuntimeWarning. λ stays at zero when the consolidation cycle
        (update_fisher → update_lambda → anchor_weights) is skipped
        during training; in that case the penalty is mathematically
        zero regardless of the caller's ewc_strength and the substrate
        is silently a plain MLP. The warning fires once per process.
        """
        if not _EwcZeroWarning._warned and bool((self.lam == 0).all().item()):
            _EwcZeroWarning._warned = True
            warnings.warn(
                "TrioronLayer.ewc_penalty(): layer.lam is all zero — the "
                "EWC penalty is silently zero. The consolidation cycle "
                "(update_fisher → update_lambda → anchor_weights) was "
                "likely skipped during training; without it the node is "
                "just a regular MLP. See the trioron.node module docstring.",
                RuntimeWarning, stacklevel=2,
            )
        # Per-node effective stiffness = λ · (1 - apoptosis_pulse). When
        # all pulses are 0 (no recent deaths), this equals plain λ.
        # Archived rows contribute zero — they're locked at consolidated
        # state by gradient masking, so their EWC term would be 0 anyway,
        # but skipping explicitly keeps the penalty term clean and saves
        # the squared-difference compute.
        eff_lam = self.lam * (1.0 - self.apoptosis_pulse).clamp_min(0.0)
        if self.archived.any():
            eff_lam = eff_lam * (~self.archived).to(eff_lam.dtype)
        stiffness = eff_lam.unsqueeze(1)  # (n_nodes, 1)
        pen_W = (stiffness * (self.W - self.W_anchor).pow(2)).sum()
        pen_b = (eff_lam * (self.b - self.b_anchor).pow(2)).sum()
        # Trioron 2.0 Axis 5: branch_weight enters EWC at per-branch
        # granularity, scaled by both fisher_branch_weight (per-branch)
        # and per-cell eff_lam. Unlike W (whose Fisher is pre-collapsed
        # into λ), branch_weight uses fisher_branch_weight directly so
        # individual branches that drift can be selectively protected
        # without stiffening every branch in the cell uniformly.
        pen_bw = (
            stiffness * self.fisher_branch_weight
            * (self.branch_weight - self.branch_weight_anchor).pow(2)
        ).sum()
        return pen_W + pen_b + pen_bw

    # ----- structural plasticity -----

    def grow_node(
        self,
        init_vec: Optional[torch.Tensor] = None,
        task_idx: int = 0,
        parent_idx: Optional[int] = None,
    ) -> int:
        """Add one new node to the layer. Returns the index of the new node.

        init_vec: optional length-fan_in tensor for the new node's incoming
            weights. If None, uses the same Kaiming-style init as construction.
        task_idx: current task index (used by dreaming-phase routing
            starvation to pick the YOUNGER of two redundant nodes as
            victim). Defaults to 0 so legacy callers / tests are unaffected.
        parent_idx: optional Trioron 2.0 Axis 5 sister-specialist seed.
            When provided, the new cell inherits the parent's dendritic
            structure (branch_id row, branch_weight row, B_per_node) via
            inherit_dendrite, with a 5% ε column-reassignment perturbation.
            Default None preserves 1.0 behavior: blank-slate K=1 child.

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
            new_routing_scale_anchor = torch.cat(
                [self.routing_scale_anchor, torch.ones(1, device=device)]
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
            new_archived = torch.cat(
                [self.archived,
                 torch.zeros(1, dtype=torch.bool, device=device)]
            )
            new_lam_high_streak = torch.cat(
                [self.lam_high_streak,
                 torch.zeros(1, dtype=torch.long, device=device)]
            )
            new_axonal_gain = torch.cat(
                [self.axonal_gain, torch.ones(1, device=device)]
            )
            new_axonal_gain_anchor = torch.cat(
                [self.axonal_gain_anchor, torch.ones(1, device=device)]
            )
            # Trioron 2.0 Axis 5: dendritic state for the new cell. Born
            # at K=1 (every column on branch 0, branch_weight = [1, 0..]),
            # no internal stress / utility / Fisher yet. Phase 2.5's
            # inherit_dendrite will overwrite branch_id and branch_weight
            # with a parent's pattern + ε perturbation when parent_idx
            # is supplied; for now the new cell is a blank-slate point
            # neuron.
            new_branch_id = torch.cat(
                [self.branch_id,
                 torch.zeros(1, self.fan_in, dtype=torch.long, device=device)],
                dim=0,
            )
            new_bw_row = torch.zeros(
                1, self.B_max, dtype=self.branch_weight.dtype, device=device,
            )
            new_bw_row[0, 0] = 1.0
            new_branch_weight = torch.cat([self.branch_weight.data, new_bw_row], dim=0)
            new_branch_weight_anchor = torch.cat(
                [self.branch_weight_anchor, new_bw_row.clone().to(self.branch_weight_anchor.dtype)],
                dim=0,
            )
            new_fisher_branch_weight = torch.cat(
                [self.fisher_branch_weight,
                 torch.zeros(1, self.B_max,
                             dtype=self.fisher_branch_weight.dtype, device=device)],
                dim=0,
            )
            new_B_per_node = torch.cat(
                [self.B_per_node,
                 torch.ones(1, dtype=torch.long, device=device)],
            )
            new_internal_stress = torch.cat(
                [self.internal_stress, torch.zeros(1, device=device)]
            )
            new_branch_utility = torch.cat(
                [self.branch_utility,
                 torch.zeros(1, self.B_max,
                             dtype=self.branch_utility.dtype, device=device)],
                dim=0,
            )
            new_dendrite_orphan = torch.cat(
                [self.dendrite_orphan,
                 torch.zeros(1, self.fan_in,
                             dtype=torch.bool, device=device)],
                dim=0,
            )

        # Re-register parameters and buffers with new shapes.
        self._replace_parameter("W", new_W)
        self._replace_parameter("b", new_b)
        self._replace_parameter("branch_weight", new_branch_weight)
        self._replace_buffer("lam", new_lam)
        self._replace_buffer("u", new_u)
        self._replace_buffer("W_anchor", new_W_anchor)
        self._replace_buffer("b_anchor", new_b_anchor)
        self._replace_buffer("fisher_W", new_fisher_W)
        self._replace_buffer("fisher_b", new_fisher_b)
        self._replace_buffer("routing_scale", new_routing_scale)
        self._replace_buffer("routing_scale_anchor", new_routing_scale_anchor)
        self._replace_buffer("routing_latched", new_routing_latched)
        self._replace_buffer("task_of_origin", new_task_of_origin)
        self._replace_buffer("apoptosis_pulse", new_apoptosis_pulse)
        self._replace_buffer("archived", new_archived)
        self._replace_buffer("lam_high_streak", new_lam_high_streak)
        self._replace_buffer("axonal_gain", new_axonal_gain)
        self._replace_buffer("axonal_gain_anchor", new_axonal_gain_anchor)
        self._replace_buffer("branch_id", new_branch_id)
        self._replace_buffer("branch_weight_anchor", new_branch_weight_anchor)
        self._replace_buffer("fisher_branch_weight", new_fisher_branch_weight)
        self._replace_buffer("B_per_node", new_B_per_node)
        self._replace_buffer("internal_stress", new_internal_stress)
        self._replace_buffer("branch_utility", new_branch_utility)
        self._replace_buffer("dendrite_orphan", new_dendrite_orphan)

        # Axis 5 sister-specialist inheritance. inherit_dendrite
        # validates parent_idx against the post-grow n_nodes and refuses
        # parent_idx == child_idx, so the validation happens there.
        if parent_idx is not None:
            self.inherit_dendrite(parent_idx, self.n_nodes - 1)

        return self.n_nodes - 1

    def grow_input(
        self,
        init_col: Optional[torch.Tensor] = None,
        source: Optional[tuple] = None,
    ) -> None:
        """Extend fan_in by 1, adding a new column to W. Used for cross-layer
        growth: when the previous layer adds a node, this layer must accept
        the extra input.

        init_col: optional length-n_nodes tensor for the new column. If None,
            zeros — the new input contributes nothing initially, and the
            network learns to use it via gradient descent.

        source: optional (src_layer_idx, src_node_idx) tuple recording the
            new column's provenance for Trioron 2.0 long-range reach. If
            None (default), the sentinel (-1, -1) is recorded, meaning
            "this column reads from the immediate sequential predecessor."
            Sequential-default behavior is preserved byte-identically.

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
            if source is None:
                src_row = torch.tensor(
                    [[-1, -1]], dtype=torch.long, device=device,
                )
            else:
                src_row = torch.tensor(
                    [[int(source[0]), int(source[1])]],
                    dtype=torch.long, device=device,
                )
            new_input_sources = torch.cat(
                [self.input_sources, src_row], dim=0,
            )
            new_input_archived = torch.cat(
                [self.input_archived,
                 torch.zeros(1, dtype=torch.bool, device=device)],
            )
            # Trioron 2.0 Axis 5: new column lands on branch 0 (default
            # niche). Phase 2.5's grow_branch may later reassign it. New
            # column starts un-orphaned for every existing cell.
            new_branch_id = torch.cat(
                [self.branch_id,
                 torch.zeros(self.n_nodes, 1, dtype=torch.long, device=device)],
                dim=1,
            )
            new_dendrite_orphan = torch.cat(
                [self.dendrite_orphan,
                 torch.zeros(self.n_nodes, 1, dtype=torch.bool, device=device)],
                dim=1,
            )

        self.fan_in += 1
        self._replace_parameter("W", new_W)
        self._replace_buffer("W_anchor", new_W_anchor)
        self._replace_buffer("fisher_W", new_fisher_W)
        self._replace_buffer("input_sources", new_input_sources)
        self._replace_buffer("input_archived", new_input_archived)
        self._replace_buffer("branch_id", new_branch_id)
        self._replace_buffer("dendrite_orphan", new_dendrite_orphan)

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
            new_input_sources = self.input_sources.index_select(0, keep_t)
            new_input_archived = self.input_archived.index_select(0, keep_t)
            # Trioron 2.0 Axis 5: drop the matching branch_id /
            # dendrite_orphan columns.
            new_branch_id = self.branch_id.index_select(1, keep_t)
            new_dendrite_orphan = self.dendrite_orphan.index_select(1, keep_t)

        self.fan_in -= 1
        self._replace_parameter("W", new_W)
        self._replace_buffer("W_anchor", new_W_anchor)
        self._replace_buffer("fisher_W", new_fisher_W)
        self._replace_buffer("input_sources", new_input_sources)
        self._replace_buffer("input_archived", new_input_archived)
        self._replace_buffer("branch_id", new_branch_id)
        self._replace_buffer("dendrite_orphan", new_dendrite_orphan)

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
            new_routing_scale_anchor = self.routing_scale_anchor.index_select(0, keep_t)
            new_routing_latched = self.routing_latched.index_select(0, keep_t)
            new_task_of_origin = self.task_of_origin.index_select(0, keep_t)
            new_apoptosis_pulse = self.apoptosis_pulse.index_select(0, keep_t)
            new_archived = self.archived.index_select(0, keep_t)
            new_lam_high_streak = self.lam_high_streak.index_select(0, keep_t)
            new_axonal_gain = self.axonal_gain.index_select(0, keep_t)
            new_axonal_gain_anchor = self.axonal_gain_anchor.index_select(0, keep_t)
            new_branch_id = self.branch_id.index_select(0, keep_t)
            new_branch_weight = self.branch_weight.data.index_select(0, keep_t)
            new_branch_weight_anchor = self.branch_weight_anchor.index_select(0, keep_t)
            new_fisher_branch_weight = self.fisher_branch_weight.index_select(0, keep_t)
            new_B_per_node = self.B_per_node.index_select(0, keep_t)
            new_internal_stress = self.internal_stress.index_select(0, keep_t)
            new_branch_utility = self.branch_utility.index_select(0, keep_t)
            new_dendrite_orphan_row = self.dendrite_orphan.index_select(0, keep_t)

        self._replace_parameter("W", new_W)
        self._replace_parameter("b", new_b)
        self._replace_parameter("branch_weight", new_branch_weight)
        self._replace_buffer("lam", new_lam)
        self._replace_buffer("u", new_u)
        self._replace_buffer("W_anchor", new_W_anchor)
        self._replace_buffer("b_anchor", new_b_anchor)
        self._replace_buffer("fisher_W", new_fisher_W)
        self._replace_buffer("fisher_b", new_fisher_b)
        self._replace_buffer("routing_scale", new_routing_scale)
        self._replace_buffer("routing_scale_anchor", new_routing_scale_anchor)
        self._replace_buffer("routing_latched", new_routing_latched)
        self._replace_buffer("task_of_origin", new_task_of_origin)
        self._replace_buffer("apoptosis_pulse", new_apoptosis_pulse)
        self._replace_buffer("archived", new_archived)
        self._replace_buffer("lam_high_streak", new_lam_high_streak)
        self._replace_buffer("axonal_gain", new_axonal_gain)
        self._replace_buffer("axonal_gain_anchor", new_axonal_gain_anchor)
        self._replace_buffer("branch_id", new_branch_id)
        self._replace_buffer("branch_weight_anchor", new_branch_weight_anchor)
        self._replace_buffer("fisher_branch_weight", new_fisher_branch_weight)
        self._replace_buffer("B_per_node", new_B_per_node)
        self._replace_buffer("internal_stress", new_internal_stress)
        self._replace_buffer("branch_utility", new_branch_utility)
        self._replace_buffer("dendrite_orphan", new_dendrite_orphan_row)

    # ----- dendritic plasticity (Trioron 2.0 Axis 5, Phase 2.5) -----

    def grow_branch(
        self,
        node_idx: int,
        source_cols,
    ) -> int:
        """Split a cell's flat fan-in into a new dendritic branch.

        Reassigns the columns in `source_cols` from their current
        branches on cell `node_idx` to a freshly-allocated branch. The
        new branch's soma-side weight is initialized to
        0.1 · mean(branch_weight[node_idx, :B_per_node[node_idx]]) —
        small enough not to perturb the cell's forward output, large
        enough that gradient routes through it immediately. Increments
        B_per_node[node_idx] and returns the new branch index.

        Refuses if B_per_node[node_idx] == B_max (lifetime budget) or
        if source_cols is empty / out-of-range / duplicated. No
        optimizer rebuild required: branch_weight retains its Parameter
        identity (the new branch slot was already allocated at
        construction). Adam moments for the new slot are stale at
        initialization (zero), which aligns with the convention that
        unused slots carry zero moment anyway.
        """
        if not (0 <= node_idx < self.n_nodes):
            raise IndexError(
                f"grow_branch node_idx {node_idx} out of range "
                f"[0, {self.n_nodes})"
            )
        cols = torch.as_tensor(
            source_cols, dtype=torch.long, device=self.W.device,
        ).reshape(-1)
        if cols.numel() == 0:
            raise ValueError("grow_branch source_cols must be non-empty")
        if (cols < 0).any() or (cols >= self.fan_in).any():
            raise IndexError(
                f"grow_branch source_cols out of range [0, {self.fan_in})"
            )
        if cols.unique().numel() != cols.numel():
            raise ValueError("grow_branch source_cols must be unique")

        K = int(self.B_per_node[node_idx].item())
        if K >= self.B_max:
            raise ValueError(
                f"grow_branch on node {node_idx}: B_per_node already at "
                f"B_max={self.B_max}"
            )

        new_idx = K
        with torch.no_grad():
            self.branch_id[node_idx, cols] = new_idx
            # Orphan flag clears for any reassigned column — joining a
            # live branch un-prunes that cell's connection to the source.
            self.dendrite_orphan[node_idx, cols] = False
            mean_w = self.branch_weight.data[node_idx, :K].mean()
            self.branch_weight.data[node_idx, new_idx] = 0.1 * mean_w
            self.branch_weight_anchor[node_idx, new_idx] = (
                self.branch_weight.data[node_idx, new_idx]
            )
            self.fisher_branch_weight[node_idx, new_idx] = 0.0
            self.branch_utility[node_idx, new_idx] = 0.0
            self.B_per_node[node_idx] = K + 1
        return new_idx

    def prune_branch(self, node_idx: int, branch_idx: int) -> None:
        """Retract a dendritic branch from a cell.

        Removes branch `branch_idx` from cell `node_idx`:
          - columns assigned to it become orphaned for this cell
            (dendrite_orphan[node_idx, cols] = True). The cell's
            forward zeros their contribution; other cells reading the
            same upstream source remain unaffected.
          - higher-numbered branches are compacted down by one
            (branch_id renumbered, branch_weight / anchor / fisher /
            utility columns shifted left at branch_idx).
          - B_per_node[node_idx] decrements.

        Refuses if B_per_node[node_idx] == 1 (cell apoptosis handles
        the last-branch case) or if branch_idx is out of range. After
        a prune, Adam moments for the compacted slots are stale
        (they're still indexed by the old branch number); callers that
        care should reset the optimizer for the affected cell.
        """
        if not (0 <= node_idx < self.n_nodes):
            raise IndexError(
                f"prune_branch node_idx {node_idx} out of range "
                f"[0, {self.n_nodes})"
            )
        K = int(self.B_per_node[node_idx].item())
        if K == 1:
            raise ValueError(
                f"prune_branch on node {node_idx}: cannot prune at K=1 — "
                f"use cell-level apoptosis instead"
            )
        if not (0 <= branch_idx < K):
            raise IndexError(
                f"prune_branch branch_idx {branch_idx} out of range "
                f"[0, {K})"
            )

        with torch.no_grad():
            row_branch_id = self.branch_id[node_idx]
            # 1. Orphan the columns this branch held.
            orphan_cols = (row_branch_id == branch_idx)
            self.dendrite_orphan[node_idx, orphan_cols] = True
            # 2. Renumber higher branches down by 1.
            higher = row_branch_id > branch_idx
            self.branch_id[node_idx, higher] = (
                row_branch_id[higher] - 1
            )
            # Orphaned columns' branch_id resets to 0 (their contrib is
            # masked out via dendrite_orphan; the canonical value
            # doesn't matter functionally but 0 keeps the buffer tidy).
            self.branch_id[node_idx, orphan_cols] = 0
            # 3. Shift branch_weight / anchor / fisher / utility columns
            #    branch_idx+1..K-1 → branch_idx..K-2; zero the tail.
            if branch_idx < K - 1:
                shift = slice(branch_idx, K - 1)
                source = slice(branch_idx + 1, K)
                self.branch_weight.data[node_idx, shift] = (
                    self.branch_weight.data[node_idx, source]
                )
                self.branch_weight_anchor[node_idx, shift] = (
                    self.branch_weight_anchor[node_idx, source]
                )
                self.fisher_branch_weight[node_idx, shift] = (
                    self.fisher_branch_weight[node_idx, source]
                )
                self.branch_utility[node_idx, shift] = (
                    self.branch_utility[node_idx, source]
                )
            # Vacated tail slot resets to defaults.
            self.branch_weight.data[node_idx, K - 1] = 0.0
            self.branch_weight_anchor[node_idx, K - 1] = 0.0
            self.fisher_branch_weight[node_idx, K - 1] = 0.0
            self.branch_utility[node_idx, K - 1] = 0.0
            # 4. Decrement branch count.
            self.B_per_node[node_idx] = K - 1

    def inherit_dendrite(
        self,
        parent_idx: int,
        child_idx: int,
        perturb_frac: float = 0.05,
    ) -> None:
        """Seed a child cell's dendritic structure from a parent.

        Called automatically by grow_node when a parent_idx is supplied.
        Copies the parent's branch_id row, branch_weight row, and
        B_per_node entry, then randomly reassigns `perturb_frac` of the
        child's columns to OTHER existing branches (the ε structural
        perturbation that gives the child a sister-specialist identity
        rather than a literal clone).

        If the parent is K=1, perturbation is a no-op (only one branch
        to choose from) — the child is born a faithful K=1 clone and
        will diverge through subsequent grow_branch events.

        Resets the child's fisher_branch_weight, branch_utility, and
        dendrite_orphan to defaults — the child inherits structure but
        not gradient history.
        """
        if not (0 <= parent_idx < self.n_nodes):
            raise IndexError(
                f"inherit_dendrite parent_idx {parent_idx} out of range "
                f"[0, {self.n_nodes})"
            )
        if not (0 <= child_idx < self.n_nodes):
            raise IndexError(
                f"inherit_dendrite child_idx {child_idx} out of range "
                f"[0, {self.n_nodes})"
            )
        if parent_idx == child_idx:
            raise ValueError("inherit_dendrite: parent_idx == child_idx")
        if not (0.0 <= perturb_frac <= 1.0):
            raise ValueError(
                f"inherit_dendrite perturb_frac {perturb_frac} not in [0, 1]"
            )

        K_parent = int(self.B_per_node[parent_idx].item())
        with torch.no_grad():
            self.branch_id[child_idx].copy_(self.branch_id[parent_idx])
            self.branch_weight.data[child_idx].copy_(
                self.branch_weight.data[parent_idx]
            )
            self.branch_weight_anchor[child_idx].copy_(
                self.branch_weight.data[parent_idx]
            )
            self.fisher_branch_weight[child_idx].zero_()
            self.branch_utility[child_idx].zero_()
            self.dendrite_orphan[child_idx].zero_()
            self.B_per_node[child_idx] = K_parent

            if K_parent > 1 and perturb_frac > 0:
                n_perturb = max(1, int(round(perturb_frac * self.fan_in)))
                n_perturb = min(n_perturb, self.fan_in)
                perm = torch.randperm(
                    self.fan_in, device=self.W.device,
                )[:n_perturb]
                for j in perm.tolist():
                    current = int(self.branch_id[child_idx, j].item())
                    others = [b for b in range(K_parent) if b != current]
                    pick = torch.randint(
                        0, len(others), (1,), device=self.W.device,
                    ).item()
                    self.branch_id[child_idx, j] = others[pick]

    def select_parent(self) -> int:
        """Pick the existing cell with the highest mean activation
        across the most recent grad-enabled forward as the parent for
        a grow_node call.

        Ties are broken by lowest index (torch.argmax convention).
        Raises if no forward has run yet (no _last_y cached).
        """
        if self._last_y is None:
            raise RuntimeError(
                "select_parent: no cached forward yet — run a "
                "grad-enabled forward first"
            )
        return int(self._last_y.mean(dim=0).argmax().item())

    def update_internal_stress(self) -> None:
        """EMA update of per-cell internal stress (within-niche signal).

            internal_stress[i] = EMA( |∂L/∂y_i| · engaged(y_i) )

        engaged(y) = 1(y > 0) for ReLU; 1(|y| > 0.05) for tanh / linear.

        Call AFTER loss.backward() (so _last_upstream is populated).
        No-op if no grad-enabled forward + backward has run yet.
        """
        if self._last_y is None or self._last_upstream is None:
            return
        with torch.no_grad():
            if self.activation == "relu":
                engaged = (self._last_y > 0).to(self._last_y.dtype)
            else:
                engaged = (self._last_y.abs() > 0.05).to(self._last_y.dtype)
            stress = (self._last_upstream.abs() * engaged).mean(dim=0)
            stress = stress.to(
                device=self.internal_stress.device,
                dtype=self.internal_stress.dtype,
            )
            self.internal_stress.mul_(self.fisher_decay).add_(
                stress, alpha=1.0 - self.fisher_decay,
            )

    def update_branch_utility(self) -> None:
        """EMA update of per-(cell, branch) saliency.

            branch_utility[i, b] = EMA( |branch_weight[i, b] · y_{i, b}| )

        Mozer & Smolensky utility at branch granularity. Only does work
        when the K>1 dendritic forward ran (so y_branches is cached);
        otherwise returns silently — the K=1 fast path doesn't compute
        per-branch outputs and there's no branch structure to track.
        """
        if self._last_y_branches is None:
            return
        with torch.no_grad():
            bw = (
                self.branch_weight.data.unsqueeze(0)
                .to(self._last_y_branches.dtype)
            )
            contrib = (bw * self._last_y_branches).abs().mean(dim=0)
            contrib = contrib.to(
                device=self.branch_utility.device,
                dtype=self.branch_utility.dtype,
            )
            self.branch_utility.mul_(self.fisher_decay).add_(
                contrib, alpha=1.0 - self.fisher_decay,
            )

    def reset_dendritic_state(self) -> None:
        """Reset all Axis 5 dendritic state to K=1 / point-neuron form.

        Used by donor absorption (spec §5.2): R·S factorizes W's column
        space and does not depend on branch_id, so a donor's
        post-training dendritic structure (which branch carries which
        column, branch_weight per branch, etc.) is not portable. After
        absorption, the absorbed substrate joins the host's dendritic
        regime at K=1 and re-grows branches under the host's
        internal-stress signals.

        Idempotent: calling on an already-K=1 layer is a no-op modulo
        clearing utility / stress EMAs.

        Buffers reset:
          branch_id            → all 0 (every column on branch 0)
          branch_weight        → [1.0, 0.0, ..., 0.0] per cell
          branch_weight_anchor → mirror of branch_weight
          fisher_branch_weight → all 0
          B_per_node           → all 1
          internal_stress      → all 0
          branch_utility       → all 0
          dendrite_orphan      → all False
        """
        with torch.no_grad():
            self.branch_id.zero_()
            self.branch_weight.data.zero_()
            self.branch_weight.data[:, 0] = 1.0
            self.branch_weight_anchor.copy_(self.branch_weight.data)
            self.fisher_branch_weight.zero_()
            self.B_per_node.fill_(1)
            self.internal_stress.zero_()
            self.branch_utility.zero_()
            self.dendrite_orphan.zero_()

    def standardized_column_mask(self) -> torch.Tensor:
        """Return bool[fan_in] True for columns at sequential-default
        provenance (input_sources[j] == (-1, -1)).

        Used by the R·S handshake (composition.translator) per spec
        §5.2: cross-donor factorization operates only on the
        standardized subset of columns; long-range columns are
        per-donor private and excluded from the handshake. 1.0 donors
        trivially have all-standardized columns.
        """
        return (self.input_sources < 0).all(dim=1)

    def internal_frustration_candidates(
        self,
        threshold: float = 0.05,
        overall_saliency_ceiling: float = 0.1,
    ) -> list[int]:
        """Cells eligible for grow_branch (specialist trying but
        failing to discriminate within its niche).

        Returns cell indices sorted by internal_stress descending,
        keeping only those where:
          - internal_stress[i] > threshold  (within-niche failure)
          - saliency_utility[i] < overall_saliency_ceiling  (NOT a
            cell with strong overall contribution — those are handled
            by population-level frustration, not within-niche growth)

        For Phase 2.5 the saliency-utility proxy for 'overall stress'
        is intentional: if a cell is loudly contributing to good
        outputs, its high internal_stress is noise — keep its current
        flat structure. Phase 3+ may refine this.

        Returns an empty list when the active TrioronProfile has
        `allow_grow_branch=False` (e.g., under CLASSIFICATION or EDGE
        regimes). grow_branch itself remains callable for explicit
        manual use; this is the policy-layer gate.
        """
        from trioron.profile import TrioronProfile
        if not TrioronProfile.active().allow_grow_branch:
            return []
        with torch.no_grad():
            high_internal = (
                self.internal_stress > threshold
            ).nonzero(as_tuple=True)[0]
            if high_internal.numel() == 0:
                return []
            overall = self.saliency_utility()
            mask = overall[high_internal] < overall_saliency_ceiling
            keep = high_internal[mask]
            if keep.numel() == 0:
                return []
            order = self.internal_stress[keep].argsort(descending=True)
            return keep[order].tolist()

    # ----- low-level helpers -----

    def _replace_parameter(self, name: str, new_tensor: torch.Tensor) -> None:
        if name in self._parameters:
            del self._parameters[name]
        setattr(self, name, nn.Parameter(new_tensor))

    def _replace_buffer(self, name: str, new_tensor: torch.Tensor) -> None:
        if name in self._buffers:
            del self._buffers[name]
        self.register_buffer(name, new_tensor)

    # ----- state-dict back-compat (Trioron 2.0 Phase 4) -----

    # New buffers / parameters introduced in Trioron 2.0. Pre-2.0 donor
    # checkpoints don't have these keys; in strict-mode load_state_dict
    # the missing keys would raise. We override _load_from_state_dict
    # to inject the layer's current default value for any 2.0 key not
    # present in the incoming state_dict — preserving back-compat with
    # the entire pre-2.0 shipped-donor zoo. branch_weight is included
    # here because it's a 2.0-introduced Parameter; PyTorch's strict
    # load treats parameters and buffers uniformly in the missing-keys
    # check, so the same default-injection trick works for both.
    _TRIORON_2_0_KEYS = (
        # Phase 1 (Axes 1, 2, 4):
        "input_sources",
        "input_archived",
        "axonal_gain",
        "axonal_gain_anchor",
        # Phase 1.5 (Axis 5 — dendritic compartmentalization):
        "branch_id",
        "branch_weight",
        "branch_weight_anchor",
        "fisher_branch_weight",
        "B_per_node",
        "internal_stress",
        "branch_utility",
        # Phase 2.5 (orphan mask for prune_branch):
        "dendrite_orphan",
    )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # v1 donor detection: state-dicts written before Axis 5 don't
        # carry `branch_id`. When that key is absent we treat the load
        # as a v1 donor and flip branch_activation to "identity" so the
        # absorbed substrate remains point-neuron-equivalent even if it
        # later grows branches under the host's machinery (Phase 2.5+).
        # See trioron_2_0.md §5.1.
        #
        # Profile override (re_apply_after_donor_load): when the active
        # TrioronProfile sets re_apply_after_donor_load=True (default),
        # the v1 flip is undone after load — the profile's chosen
        # branch_activation wins. Set False on the profile to honor the
        # v1 silent-override and freeze the loaded layer as
        # point-neuron-equivalent regardless of regime.
        if (prefix + "branch_id") not in state_dict:
            from trioron.profile import TrioronProfile
            active = TrioronProfile.active()
            if active.re_apply_after_donor_load:
                self.branch_activation = active.branch_activation
            else:
                self.branch_activation = "identity"

        # Inject defaults for any 2.0 buffer / parameter the incoming
        # state_dict doesn't carry. The layer was constructed with the
        # current 2.0 defaults; we just push them into the dict so
        # PyTorch's strict-mode load doesn't trip on missing keys.
        for key_name in self._TRIORON_2_0_KEYS:
            full_key = prefix + key_name
            if full_key not in state_dict:
                state_dict[full_key] = getattr(self, key_name).detach().clone()
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __repr__(self) -> str:
        return (
            f"TrioronLayer(fan_in={self.fan_in}, n_nodes={self.n_nodes}, "
            f"activation='{self.activation}')"
        )
