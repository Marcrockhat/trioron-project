"""HAT baseline (Serra et al. 2018) — Hard Attention to the Task.

Task-aware continual-learning baseline. Per-task LEARNED sigmoid
attention masks over hidden-layer outputs, temperature-annealed during
training. The cumulative-mask of past tasks gates how much each weight
can change during the current task; weights tied to past-task units
are protected by gradient scaling. A sparsity loss encourages the
current task to overlap as much as possible with already-claimed
capacity (rather than always allocating fresh).

Why HAT not DEN: per next_session_plan.md, literature consensus
(Parisi 2019, van de Ven 2019, Mai 2022) places HAT ≈ DEN on standard
benchmarks. HAT is also less ambiguous to implement.

Mapping to TrioronNetwork:

  - Last layer (the contrastive head, tanh) is treated as the shared
    output head: no per-task mask, but its weights are protected via
    the input-side cumulative mask of the layer beneath.
  - All non-final layers get per-task sigmoid masks on their output.
    For our 3-layer architecture this means the two ReLU hidden layers
    are masked and the latent tanh layer is shared.
  - Forward masking is applied via PyTorch forward_hooks; no surgery
    on TrioronNetwork.forward.

Departures from the paper, all noted:

  - Embedding-gradient stability trick (Serra Algorithm 1's
    cosh^2 ratio rescale on `e.grad` to prevent saturation collapse):
    omitted in the initial implementation. We rely on a hard
    embedding-magnitude clip after each step. If divergence shows up
    the rescale is the next thing to add.
  - Optimizer state for the active embedding is held by the
    controller (it owns nn.Parameter); the caller is expected to
    include `ctrl.parameters()` when constructing the optimizer.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .network import TrioronNetwork


class HATController(nn.Module):
    """Per-task sigmoid-attention manager for a TrioronNetwork.

    Lifecycle per task t in 1..n_total_tasks:

        ctrl.begin_task(t)
        opt = Adam(list(net.parameters()) + list(ctrl.parameters()), lr=...)
        for step in range(n_steps):
            ctrl.set_temperature(s_step)            # anneal from s_min → s_max
            opt.zero_grad()
            l_task = task_loss(...)
            l = l_task + ctrl.sparsity_coef * ctrl.sparsity_loss()
            l.backward()
            ctrl.scale_grads()                      # protect prior-task weights
            opt.step()
            ctrl.clip_embeddings()
        ctrl.end_task(t)                            # snapshot e^l_t, update cumulative

    For inference of task t:

        snap = ctrl.apply_inference_mask(t)         # installs hooks for task t
        eval_loss = ...
        ctrl.restore(snap)                          # removes hooks
    """

    def __init__(
        self,
        net: TrioronNetwork,
        n_total_tasks: int,
        s_min: float = 1.0 / 400.0,
        s_max: float = 400.0,
        sparsity_coef: float = 0.75,
        emb_clip: float = 6.0,
    ):
        super().__init__()
        if n_total_tasks < 1:
            raise ValueError("n_total_tasks must be >= 1")
        if s_min <= 0 or s_max <= 0 or s_min >= s_max:
            raise ValueError("require 0 < s_min < s_max")
        if sparsity_coef < 0:
            raise ValueError("sparsity_coef must be >= 0")
        if emb_clip <= 0:
            raise ValueError("emb_clip must be > 0")

        # Hold net as a non-submodule so ctrl.parameters() yields only
        # the controller's own embeddings, not the net's weights — caller
        # is responsible for combining both lists when constructing the
        # optimizer (and not double-counting).
        object.__setattr__(self, "net", net)
        self.n_total_tasks = int(n_total_tasks)
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.sparsity_coef = float(sparsity_coef)
        self.emb_clip = float(emb_clip)

        # Layers that get per-task masks: every layer except the last (head).
        if len(net.layers) < 2:
            raise ValueError("net needs at least 2 layers (one masked + head)")
        self.masked_layer_idxs: List[int] = list(range(len(net.layers) - 1))
        self.task_dims: List[int] = [
            net.layers[i].n_nodes for i in self.masked_layer_idxs
        ]

        # Active per-task embeddings (Parameter list; reset each begin_task).
        # Layer i embedding has shape (n_nodes_i,).
        self.active_embeddings = nn.ParameterList(
            [nn.Parameter(torch.zeros(d)) for d in self.task_dims]
        )

        # Cumulative mask per masked layer (buffer; in [0,1], grows with tasks).
        for li, d in enumerate(self.task_dims):
            self.register_buffer(f"cum_mask_{li}", torch.zeros(d))

        # Per-task saved embeddings: task_id -> list of tensors (no grad).
        self.task_embeddings: Dict[int, List[torch.Tensor]] = {}

        # Per-task input-side cumulative used when scaling the head's grads.
        # Initialized to zero, refreshed as tasks complete.
        self.tasks_done: int = 0

        # Mutable hook state — accessed by the closures registered as forward hooks.
        self._mode: str = "off"          # "train" | "inference" | "off"
        self._inference_task: Optional[int] = None
        self._current_temperature: float = self.s_max
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------
    # Mask access
    # ------------------------------------------------------------------

    def _cum_mask(self, layer_idx_in_masked: int) -> torch.Tensor:
        return getattr(self, f"cum_mask_{layer_idx_in_masked}")

    def _set_cum_mask(self, layer_idx_in_masked: int, m: torch.Tensor) -> None:
        getattr(self, f"cum_mask_{layer_idx_in_masked}").copy_(m)

    def _active_mask(self, layer_idx_in_masked: int) -> torch.Tensor:
        e = self.active_embeddings[layer_idx_in_masked]
        return torch.sigmoid(self._current_temperature * e)

    def _saved_mask(self, task_id: int, layer_idx_in_masked: int) -> torch.Tensor:
        e = self.task_embeddings[task_id][layer_idx_in_masked]
        return torch.sigmoid(self.s_max * e)

    # ------------------------------------------------------------------
    # Forward hooks
    # ------------------------------------------------------------------

    def _make_hook(self, layer_idx_in_masked: int):
        def hook(module, inputs, output):
            if self._mode == "off":
                return output
            if self._mode == "train":
                mask = self._active_mask(layer_idx_in_masked)
            elif self._mode == "inference":
                tid = self._inference_task
                if tid is None or tid not in self.task_embeddings:
                    return output
                mask = self._saved_mask(tid, layer_idx_in_masked)
            else:
                return output
            return output * mask

        return hook

    def _install_hooks(self) -> None:
        if self._hook_handles:
            return
        for slot, layer_idx in enumerate(self.masked_layer_idxs):
            layer = self.net.layers[layer_idx]
            h = layer.register_forward_hook(self._make_hook(slot))
            self._hook_handles.append(h)

    def _remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    # ------------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------------

    def set_temperature(self, s: float) -> None:
        if s <= 0:
            raise ValueError("temperature must be > 0")
        self._current_temperature = float(s)

    def temperature_for_step(self, step: int, n_steps: int) -> float:
        """Linear anneal from s_min to s_max over n_steps (inclusive of last)."""
        if n_steps <= 1:
            return self.s_max
        frac = step / (n_steps - 1)
        return self.s_min + frac * (self.s_max - self.s_min)

    # ------------------------------------------------------------------
    # Per-task lifecycle
    # ------------------------------------------------------------------

    def begin_task(self, task_id: int) -> None:
        if task_id < 1 or task_id > self.n_total_tasks:
            raise ValueError(
                f"task_id {task_id} outside [1, {self.n_total_tasks}]"
            )
        if task_id != self.tasks_done + 1:
            raise ValueError(
                f"begin_task({task_id}) called when tasks_done={self.tasks_done}; "
                "tasks must run in order 1, 2, …"
            )
        with torch.no_grad():
            for p in self.active_embeddings:
                p.zero_()
        self._mode = "train"
        self._current_temperature = self.s_min
        self._install_hooks()

    def end_task(self, task_id: int) -> None:
        if task_id != self.tasks_done + 1:
            raise ValueError(
                f"end_task({task_id}) inconsistent with tasks_done={self.tasks_done}"
            )
        # Snapshot current embeddings.
        snap = [p.detach().clone() for p in self.active_embeddings]
        self.task_embeddings[task_id] = snap
        # Update cumulative mask = max(cum, σ(s_max · e)) per masked layer.
        with torch.no_grad():
            for slot in range(len(self.masked_layer_idxs)):
                cum = self._cum_mask(slot)
                new = torch.sigmoid(self.s_max * snap[slot])
                self._set_cum_mask(slot, torch.maximum(cum, new))
        self.tasks_done += 1
        self._mode = "off"
        self._remove_hooks()

    # ------------------------------------------------------------------
    # Loss + grad surgery
    # ------------------------------------------------------------------

    def sparsity_loss(self) -> torch.Tensor:
        """Regularizer that pushes the current task to share with prior tasks.

        Per Serra: for each masked layer, R_l = sum(a^l_t · (1 - a^l_<t)) /
        max(sum(1 - a^l_<t), eps).  We average over masked layers.
        """
        if not self.masked_layer_idxs:
            return torch.zeros((), device=self.active_embeddings[0].device)
        terms: List[torch.Tensor] = []
        for slot in range(len(self.masked_layer_idxs)):
            a_t = self._active_mask(slot)             # autograd-attached
            a_prev = self._cum_mask(slot)             # buffer, no grad
            free = 1.0 - a_prev
            denom = free.sum().clamp_min(1e-6)
            terms.append((a_t * free).sum() / denom)
        return torch.stack(terms).mean()

    def scale_grads(self) -> None:
        """Scale the gradients on net weights so prior-task weights are
        protected. Call after .backward() and before optimizer.step().

        For weight W^l_{ij} (W shape: n_nodes × fan_in):
            scale = min(1 - a^l_<t-1[i],  1 - a^{l-1}_<t-1[j])
        For bias b^l_i:
            scale = (1 - a^l_<t-1[i])

        Layer 0 (first masked layer) has no prior-layer mask: input-side
        scale = 1. Layer N-1 (the head) has no own mask, only input side.
        """
        n_masked = len(self.masked_layer_idxs)
        # Pre-cache 1 - cumulative per masked slot.
        free_per_slot: List[torch.Tensor] = [
            (1.0 - self._cum_mask(slot)).detach() for slot in range(n_masked)
        ]
        with torch.no_grad():
            for slot, layer_idx in enumerate(self.masked_layer_idxs):
                layer = self.net.layers[layer_idx]
                free_self = free_per_slot[slot]                 # (n_nodes,)
                if slot == 0:
                    free_in = None                              # no prior masked layer
                else:
                    free_in = free_per_slot[slot - 1]           # (fan_in,)
                if layer.W.grad is not None:
                    if free_in is None:
                        layer.W.grad.mul_(free_self.unsqueeze(1))
                    else:
                        scale = torch.minimum(
                            free_self.unsqueeze(1),             # (n_nodes,1)
                            free_in.unsqueeze(0),               # (1,fan_in)
                        )
                        layer.W.grad.mul_(scale)
                if layer.b.grad is not None:
                    layer.b.grad.mul_(free_self)
            # Head layer (last in net.layers, no own mask, input-side from last masked).
            head_idx = len(self.net.layers) - 1
            if head_idx not in self.masked_layer_idxs:
                head = self.net.layers[head_idx]
                free_in = free_per_slot[-1]                     # last masked layer's free
                if head.W.grad is not None:
                    head.W.grad.mul_(free_in.unsqueeze(0))      # broadcast over rows

    def clip_embeddings(self) -> None:
        """Clamp |e| ≤ emb_clip to keep sigmoid in a learnable regime."""
        with torch.no_grad():
            for p in self.active_embeddings:
                p.clamp_(min=-self.emb_clip, max=self.emb_clip)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def apply_inference_mask(self, eval_task_id: int) -> Dict[str, object]:
        if eval_task_id < 1 or eval_task_id > self.tasks_done:
            raise ValueError(
                f"eval_task_id {eval_task_id} outside [1, {self.tasks_done}]"
            )
        prior_mode = self._mode
        prior_task = self._inference_task
        self._mode = "inference"
        self._inference_task = eval_task_id
        self._install_hooks()
        return {"prior_mode": prior_mode, "prior_task": prior_task}

    def restore(self, snapshot: Dict[str, object]) -> None:
        prior_mode = snapshot.get("prior_mode", "off")
        prior_task = snapshot.get("prior_task", None)
        self._mode = prior_mode  # type: ignore[assignment]
        self._inference_task = prior_task  # type: ignore[assignment]
        if prior_mode == "off":
            self._remove_hooks()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def cumulative_mask_density(self) -> List[float]:
        """Mean of (σ(s_max · cum_logit_proxy)) per masked layer. Since cum is
        already in [0,1], we just report the mean directly. Useful for
        eyeballing how 'full' the network is."""
        out: List[float] = []
        for slot in range(len(self.masked_layer_idxs)):
            out.append(float(self._cum_mask(slot).mean().item()))
        return out

    def per_task_mask_overlap(self, t1: int, t2: int) -> List[float]:
        """Per-masked-layer mean overlap (a^t1 ∧ a^t2) for diagnostics."""
        if t1 not in self.task_embeddings or t2 not in self.task_embeddings:
            raise ValueError("both tasks must have completed end_task")
        out: List[float] = []
        for slot in range(len(self.masked_layer_idxs)):
            a1 = self._saved_mask(t1, slot)
            a2 = self._saved_mask(t2, slot)
            out.append(float(torch.minimum(a1, a2).mean().item()))
        return out

    def __repr__(self) -> str:
        return (
            f"HATController(n_total_tasks={self.n_total_tasks}, "
            f"tasks_done={self.tasks_done}, "
            f"masked_layers={self.masked_layer_idxs}, "
            f"task_dims={self.task_dims}, "
            f"s∈[{self.s_min},{self.s_max}], "
            f"sparsity_coef={self.sparsity_coef})"
        )
