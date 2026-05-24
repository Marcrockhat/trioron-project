"""Linear phenotype — batched linear combine.  See spec §3.2."""
from __future__ import annotations

import torch

from trioron.core.arena import Arena
from trioron.core.scheduler import Bucket


def forward_batch(
    act: torch.Tensor,
    bucket: Bucket,
    arena: Arena,
) -> torch.Tensor:
    """Batched linear combine for all cells in *bucket*.

    y_v = b_v + sum_{u in inputs(v)} w_{vu} * a_u

    Args:
        act: ``[batch, capacity]`` — current activations for all cells.
        bucket: compiled bucket with edge index mappings.
        arena: SoA storage (read-only during forward).

    Returns:
        ``[batch, n_cells_in_bucket]`` — activations for this bucket's cells.
    """
    n_cells = bucket.cell_ids.numel()
    batch = act.shape[0]

    out = arena.bias[bucket.bias_ids.long()].unsqueeze(0).expand(batch, -1).clone()

    if bucket.edge_indices.numel() == 0:
        return out

    src_act = act[:, bucket.edge_src.long()]
    weights = arena.edge_weight[bucket.edge_indices]

    weighted = src_act * weights.unsqueeze(0)

    out.scatter_add_(1, bucket.edge_dst_local.unsqueeze(0).expand(batch, -1), weighted)

    return out


GENE_BIT = 0
NAME = "linear"
