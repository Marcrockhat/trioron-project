"""Scheduler — rank-batched forward execution.  See spec §2.7."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import torch

from .epigenome import PERCEPTION, OUTPUT, has_gene
from .state import CellState

if TYPE_CHECKING:
    from .arena import Arena

ForwardFn = Callable[["torch.Tensor", "Bucket", "Arena"], torch.Tensor]


@dataclass
class Bucket:
    """One (rank, phenotype) group ready for batched dispatch."""

    rank: int
    phenotype: int
    cell_ids: torch.Tensor
    edge_indices: torch.Tensor
    edge_src: torch.Tensor
    edge_dst_local: torch.Tensor
    bias_ids: torch.Tensor  # same as cell_ids — kept for clarity


@dataclass
class DispatchPlan:
    """Pre-compiled execution plan, rebuilt at task boundaries."""

    buckets: list[Bucket] = field(default_factory=list)
    perception_ids: torch.Tensor = field(
        default_factory=lambda: torch.tensor([], dtype=torch.int32)
    )
    output_ids: torch.Tensor = field(
        default_factory=lambda: torch.tensor([], dtype=torch.int32)
    )


class Scheduler:
    """Rank-batched forward dispatch.  See spec §2.7 and §6.3.

    The scheduler receives a *dispatch_table* mapping phenotype int
    to a batched forward function.  Phenotype implementations live
    in ``trioron/phenotype/``; the scheduler has no hard dependency
    on them (commitment 6 — no phenotype logic in the hot loop,
    only dispatch).
    """

    def __init__(
        self,
        arena: Arena,
        dispatch_table: dict[int, ForwardFn] | None = None,
    ) -> None:
        self._arena = arena
        self._dispatch: dict[int, ForwardFn] = dispatch_table or {}
        self._plan = DispatchPlan()
        self._last_activations: torch.Tensor | None = None

    def set_dispatch_table(self, table: dict[int, ForwardFn]) -> None:
        self._dispatch = table

    # ── Compile ───────────────────────────────────────────────────

    def compile(self) -> None:
        """Build (or rebuild) the dispatch plan from current arena state."""
        a = self._arena

        # Schedulable = alive + forward_inclusion (dormant cells
        # still participate in the forward pass — §2.8).
        sched_mask = a.alive & a.forward_inclusion
        cell_ids = sched_mask.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)

        if cell_ids.numel() == 0:
            self._plan = DispatchPlan()
            return

        ranks = a.rank[cell_ids.long()]
        phenos = a.phenotype_cache[cell_ids.long()]
        epis = a.epigenome[cell_ids.long()]

        # Perception cells: set from input, skip in dispatch
        is_perception = has_gene(epis, PERCEPTION).bool()
        perception_ids = cell_ids[is_perception]
        dispatch_ids = cell_ids[~is_perception]

        # Output cells
        is_output = has_gene(epis, OUTPUT).bool()
        output_ids = cell_ids[is_output]

        # Group dispatch_ids by (rank, phenotype)
        if dispatch_ids.numel() == 0:
            self._plan = DispatchPlan([], perception_ids, output_ids)
            return

        d_ranks = a.rank[dispatch_ids.long()]
        d_phenos = a.phenotype_cache[dispatch_ids.long()]

        combo = torch.stack([d_ranks, d_phenos], dim=1)
        unique = torch.unique(combo, dim=0)

        # Pre-index edge arrays once
        active_src = a.edge_src[: a.edge_cursor]
        active_dst = a.edge_dst[: a.edge_cursor]

        buckets: list[Bucket] = []
        for row in unique:
            r, p = int(row[0].item()), int(row[1].item())
            mask = (d_ranks == r) & (d_phenos == p)
            b_cells = dispatch_ids[mask]

            # Build a local-index map:  global_cell_id → 0..len-1
            id_set = set(b_cells.tolist())
            id_to_local = {int(cid): idx for idx, cid in enumerate(b_cells.tolist())}

            # Gather edges whose dst is in this bucket
            dst_match = torch.zeros(a.edge_cursor, dtype=torch.bool, device=a.device)
            for cid in b_cells.tolist():
                dst_match |= active_dst[: a.edge_cursor] == cid
            edge_idx = dst_match.nonzero(as_tuple=False).squeeze(-1)

            if edge_idx.numel() == 0:
                e_idx = torch.tensor([], dtype=torch.long, device=a.device)
                e_src = torch.tensor([], dtype=torch.int32, device=a.device)
                e_dst_local = torch.tensor([], dtype=torch.long, device=a.device)
            else:
                e_idx = edge_idx.long()
                e_src = active_src[e_idx]
                e_dst_local = torch.tensor(
                    [id_to_local[int(active_dst[i].item())] for i in e_idx],
                    dtype=torch.long,
                    device=a.device,
                )

            buckets.append(
                Bucket(
                    rank=r,
                    phenotype=p,
                    cell_ids=b_cells,
                    edge_indices=e_idx,
                    edge_src=e_src,
                    edge_dst_local=e_dst_local,
                    bias_ids=b_cells,
                )
            )

        buckets.sort(key=lambda b: b.rank)
        self._plan = DispatchPlan(buckets, perception_ids, output_ids)

    # ── Forward ───────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the rank-batched forward pass.

        Args:
            x: Input tensor ``[batch, input_dim]``.

        Returns:
            Logit tensor ``[batch, n_output_cells]``.
        """
        plan = self._plan
        a = self._arena
        batch = x.shape[0]
        act = torch.zeros(batch, a.capacity, device=x.device)

        # Write input into perception cells
        n_perc = plan.perception_ids.numel()
        if n_perc > 0:
            act[:, plan.perception_ids.long()] = x[:, :n_perc]

        # Dispatch each bucket
        for bucket in plan.buckets:
            fn = self._dispatch.get(bucket.phenotype)
            if fn is None:
                continue
            bucket_act = fn(act, bucket, a)
            act[:, bucket.cell_ids.long()] = bucket_act

        self._last_activations = act.detach()

        # Collect output logits
        if plan.output_ids.numel() == 0:
            return torch.zeros(batch, 0, device=x.device)
        return act[:, plan.output_ids.long()]

    # ── Gradient mask (commitment 5) ──────────────────────────────

    def zero_dormant_grads(self) -> None:
        """Zero out gradients on dormant cells' parameters.

        Protects edges INTO dormant cells (their input weights)
        AND edges FROM dormant cells (their output connections).
        Both directions must be frozen to prevent corruption of
        locked pathways.
        """
        a = self._arena
        dormant = a.state == CellState.DORMANT

        if a.bias.grad is not None:
            a.bias.grad[dormant] = 0.0

        if a.edge_weight.grad is not None and a.edge_cursor > 0:
            src = a.edge_src[: a.edge_cursor].long()
            dst = a.edge_dst[: a.edge_cursor].long()
            frozen_edges = dormant[dst] | dormant[src]
            a.edge_weight.grad[: a.edge_cursor][frozen_edges] = 0.0
