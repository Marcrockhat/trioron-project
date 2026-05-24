"""PackNet baseline (Mallya & Lazebnik 2018) — simplified faithful variant.

A continual-learning reference baseline that the §11 risk register
(*"basically Cascade Correlation / DEN with extra steps"*) requires us
to compare against. PackNet is structurally different from the
EWC-anchored fixed baselines we already run: instead of soft anchoring,
each task is allocated a disjoint slice of the network's weights, and
inference for task t uses only the union of slices for tasks 1..t.

Differences from the original paper:

  - Original: 50% iterative magnitude pruning per task, designed for
    a small number of tasks (typically 4) on wide image-classification
    networks. Fragments to nothing if applied to 20 tasks.
  - Here: uniform per-task allocation. Task t claims fraction
    1 / (n_total_tasks - tasks_done) of the currently-free weights,
    so each task ends up with ~1/n_total_tasks of the network. Same
    spirit (disjoint per-task subnets, magnitude-based selection),
    different scaling math.

Caveats this code accepts:

  - Hinge contrastive loss has a zero-gradient region. Pruned-but-not-
    frozen weights set to zero will stay at zero unless we re-initialize
    them at each task start. We do that (random Kaiming-style init for
    free weights only) before each new task; original PackNet did not.
    Without it, ReLU layers develop dead neurons in the free pool.

  - Optimizer state (Adam moments) is reset at each task boundary.
    Pre-task masks are stale across structural changes; rebuild is the
    safe play and matches what we do in division/pruning paths
    elsewhere.

  - Inference: at end-of-curriculum eval for task j, we apply
    union(masks[1..j]) and zero everything else. Caller is responsible
    for snapshotting/restoring weights around the eval call (helper
    methods provided).
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import torch

from .network import TrioronNetwork


class PackNetController:
    """Per-task disjoint-subnet manager for a TrioronNetwork.

    Lifecycle per task t in 1..n_total_tasks:

        ctrl.begin_task(t)            # re-init free weights
        for step in range(...):
            ...
            opt.zero_grad()
            loss.backward()
            ctrl.freeze_grads()       # zero gradient on past-task weights
            opt.step()
        ctrl.end_task(t)              # claim fraction of free weights
                                      # by magnitude, zero the rest

    For inference of task t (where t ≤ tasks_done):

        snap = ctrl.apply_inference_mask(t)
        loss = compute_eval_loss(net, ...)
        ctrl.restore(snap)

    All weight masks are torch.bool tensors with the same shape as the
    corresponding parameter (W is (n_nodes, fan_in); b is (n_nodes,)).
    """

    def __init__(
        self,
        net: TrioronNetwork,
        n_total_tasks: int,
        frozen_layer_ids: List[int] = None,
    ):
        """
        frozen_layer_ids: list of layer indices that PackNet should leave
        ALONE — their weights are not re-initialized at begin_task, not
        claimed at end_task, and not zeroed at apply_inference_mask. Used
        when an upstream layer (e.g., a warmed-L0 random projection) must
        be shared across all tasks rather than partitioned.
        """
        if n_total_tasks < 1:
            raise ValueError("n_total_tasks must be >= 1")
        self.net = net
        self.n_total_tasks = int(n_total_tasks)
        self.tasks_done: int = 0
        self.frozen_layer_ids = sorted(set(int(i) for i in (frozen_layer_ids or [])))
        for li in self.frozen_layer_ids:
            if li < 0 or li >= len(net.layers):
                raise ValueError(
                    f"frozen_layer_ids contains {li} outside [0, {len(net.layers)})"
                )

        # Per-layer cumulative-frozen masks. True = belongs to some past task,
        # gradient must be zeroed and value preserved.
        # For frozen_layer_ids: mark ALL weights as frozen up front so
        # begin_task / end_task / apply_inference_mask all leave them alone.
        self.frozen: List[Tuple[torch.Tensor, torch.Tensor]] = self._init_zero_masks()
        for li in self.frozen_layer_ids:
            layer = self.net.layers[li]
            self.frozen[li] = (
                torch.ones_like(layer.W, dtype=torch.bool),
                torch.ones_like(layer.b, dtype=torch.bool),
            )

        # task_id -> [(W_mask, b_mask) per layer]. Stored at end_task time.
        self.task_masks: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

    # ----- internal helpers -----

    def _init_zero_masks(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.net.layers:
            out.append(
                (
                    torch.zeros_like(layer.W, dtype=torch.bool),
                    torch.zeros_like(layer.b, dtype=torch.bool),
                )
            )
        return out

    # ----- per-task lifecycle -----

    def begin_task(self, task_id: int) -> None:
        """Re-initialize free weights at task start.

        Without this, free weights left at 0 from end_task() of the previous
        task stay at 0 under our hinge-loss zero-gradient regime. ReLU layers
        develop dead neurons; the new task can't use any of its allocated
        capacity. Frozen weights are preserved verbatim.
        """
        if task_id < 1 or task_id > self.n_total_tasks:
            raise ValueError(
                f"task_id {task_id} outside [1, {self.n_total_tasks}]"
            )
        with torch.no_grad():
            for li, layer in enumerate(self.net.layers):
                Wm, bm = self.frozen[li]
                free_W = ~Wm
                free_b = ~bm

                gain = 2.0 if layer.activation == "relu" else 1.0
                std = (gain / layer.fan_in) ** 0.5
                fresh_W = torch.randn_like(layer.W) * std
                layer.W.data[free_W] = fresh_W[free_W]
                layer.b.data[free_b] = 0.0

    def freeze_grads(self) -> None:
        """Zero gradients on frozen weights so the optimizer can't update
        them. Call after .backward() and before .step()."""
        for li, layer in enumerate(self.net.layers):
            Wm, bm = self.frozen[li]
            if layer.W.grad is not None:
                layer.W.grad[Wm] = 0
            if layer.b.grad is not None:
                layer.b.grad[bm] = 0

    def end_task(self, task_id: int) -> None:
        """Claim a fraction of currently-free weights by magnitude as this
        task's mask. The non-claimed free weights are zeroed (they'll be
        re-used for future tasks)."""
        if task_id != self.tasks_done + 1:
            raise ValueError(
                f"end_task({task_id}) called after tasks_done={self.tasks_done}; "
                "tasks must be ended in order 1, 2, 3, …"
            )

        self.tasks_done += 1
        remaining = self.n_total_tasks - self.tasks_done + 1
        keep_fraction = 1.0 / remaining

        layer_masks: List[Tuple[torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            for li, layer in enumerate(self.net.layers):
                Wm_frozen, bm_frozen = self.frozen[li]
                free_W_mask = ~Wm_frozen
                free_b_mask = ~bm_frozen

                W_abs = layer.W.data.abs()
                free_W_vals = W_abs[free_W_mask]
                if free_W_vals.numel() > 0:
                    n_keep_W = max(1, int(round(keep_fraction * free_W_vals.numel())))
                    n_keep_W = min(n_keep_W, free_W_vals.numel())
                    threshold_W = torch.kthvalue(
                        free_W_vals, free_W_vals.numel() - n_keep_W + 1
                    ).values
                    keep_W = (W_abs >= threshold_W) & free_W_mask
                else:
                    keep_W = torch.zeros_like(Wm_frozen)

                b_abs = layer.b.data.abs()
                free_b_vals = b_abs[free_b_mask]
                if free_b_vals.numel() > 0:
                    n_keep_b = max(1, int(round(keep_fraction * free_b_vals.numel())))
                    n_keep_b = min(n_keep_b, free_b_vals.numel())
                    threshold_b = torch.kthvalue(
                        free_b_vals, free_b_vals.numel() - n_keep_b + 1
                    ).values
                    keep_b = (b_abs >= threshold_b) & free_b_mask
                else:
                    keep_b = torch.zeros_like(bm_frozen)

                drop_W = free_W_mask & ~keep_W
                drop_b = free_b_mask & ~keep_b
                layer.W.data[drop_W] = 0
                layer.b.data[drop_b] = 0

                self.frozen[li] = (Wm_frozen | keep_W, bm_frozen | keep_b)
                layer_masks.append((keep_W, keep_b))

        self.task_masks[task_id] = layer_masks

    # ----- inference -----

    def apply_inference_mask(self, eval_task_id: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Mask the network for inference at task `eval_task_id`. Returns
        snapshots that the caller MUST pass to restore() afterwards."""
        if eval_task_id < 1 or eval_task_id > self.tasks_done:
            raise ValueError(
                f"eval_task_id {eval_task_id} outside [1, {self.tasks_done}]"
            )
        snapshots: List[Tuple[torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            for li, layer in enumerate(self.net.layers):
                snap_W = layer.W.data.clone()
                snap_b = layer.b.data.clone()
                snapshots.append((snap_W, snap_b))

                # Layers in frozen_layer_ids are shared across all tasks —
                # don't zero anything in them, just snapshot+restore.
                if li in self.frozen_layer_ids:
                    continue

                W_union = torch.zeros_like(layer.W, dtype=torch.bool)
                b_union = torch.zeros_like(layer.b, dtype=torch.bool)
                for tid in range(1, eval_task_id + 1):
                    if tid not in self.task_masks:
                        continue
                    kW, kb = self.task_masks[tid][li]
                    W_union |= kW
                    b_union |= kb

                layer.W.data[~W_union] = 0
                layer.b.data[~b_union] = 0
        return snapshots

    def restore(
        self, snapshots: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        with torch.no_grad():
            for li, layer in enumerate(self.net.layers):
                W, b = snapshots[li]
                layer.W.data.copy_(W)
                layer.b.data.copy_(b)

    # ----- introspection -----

    def per_task_capacity(self) -> List[int]:
        """Returns total weights claimed per task (sum across layers)."""
        out: List[int] = []
        for tid in range(1, self.tasks_done + 1):
            n = 0
            for kW, kb in self.task_masks[tid]:
                n += int(kW.sum().item()) + int(kb.sum().item())
            out.append(n)
        return out

    def cumulative_frozen_count(self) -> int:
        n = 0
        for Wm, bm in self.frozen:
            n += int(Wm.sum().item()) + int(bm.sum().item())
        return n

    def __repr__(self) -> str:
        return (
            f"PackNetController(n_total_tasks={self.n_total_tasks}, "
            f"tasks_done={self.tasks_done}, "
            f"frozen={self.cumulative_frozen_count()})"
        )
