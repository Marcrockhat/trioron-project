"""Saliency scoring — composite per-cell importance measure.  See spec §5.5."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from trioron.core.arena import Arena


@dataclass
class SaliencyConfig:
    w_utility: float = 0.6
    w_engagement: float = 0.3
    w_downstream: float = 0.1


def compute_saliency(
    arena: Arena,
    cfg: SaliencyConfig | None = None,
) -> torch.Tensor:
    """Return per-cell saliency scores [capacity].

    saliency(c) = w_u * utility(c) + w_e * engagement(c) + w_d * downstream(c)
    """
    cfg = cfg or SaliencyConfig()
    a = arena
    sal = torch.zeros(a.capacity, device=a.device)

    alive_mask = a.alive
    if not alive_mask.any():
        return sal

    downstream = _downstream_impact(a)

    sal[alive_mask] = (
        cfg.w_utility * a.utility[alive_mask]
        + cfg.w_engagement * a.engagement[alive_mask]
        + cfg.w_downstream * downstream[alive_mask]
    )
    return sal


def _downstream_impact(arena: Arena) -> torch.Tensor:
    """Sum of (utility + engagement) of cells that consume this cell's output."""
    a = arena
    impact = torch.zeros(a.capacity, device=a.device)

    if a.edge_cursor == 0:
        return impact

    src = a.edge_src[:a.edge_cursor].long()
    dst = a.edge_dst[:a.edge_cursor].long()
    consumer_sal = a.utility + a.engagement
    impact.scatter_add_(0, src, consumer_sal[dst])

    return impact
