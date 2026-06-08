"""Return-filter: turn a list of episodes into a (X, y) classification
dataset suitable for trioron.api.

Two filtering modes:
  - "median": keep episodes with return >= median(returns). Adaptive,
    works in early training when there are no successes yet (median is
    still some return).
  - float threshold: keep episodes with return >= threshold. Used when
    the caller has a good prior (e.g., for Pong, threshold=-15 keeps
    episodes that lost less badly than -21:0).

Optional class-balancing: cap the number of (state, action) tuples
per action class. Without this, NOOP can dominate the buffer because
a passive agent collects long quiet episodes — the trioron then
collapses to "always NOOP". Capping is the cheap fix.
"""
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch

from .rollout import Episode


def filter_by_return(
    episodes: List[Episode],
    *,
    threshold: float | str = "median",
    top_k: int | None = None,
    per_class_cap: int | None = None,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Returns (X, y, stats).

    X: (N, OBS_DIM) float32, concatenated states from surviving episodes.
    y: (N,) int64, the action taken at each surviving step.
    stats: diagnostic dict — return distribution, retention, class balance.

    Selection precedence: `top_k` (if set) wins over `threshold`. Top-k
    is the sharper-signal filter — it keeps only the K best-returning
    episodes regardless of the rest of the distribution. Useful when
    returns concentrate near the random-policy floor and a quantile
    cutoff degenerates to "keep almost everything".
    """
    rets = np.array([e.return_ for e in episodes], dtype=np.float32)

    if top_k is not None and top_k > 0:
        # Argsort descending, keep top_k. Tie-break is order-stable.
        order = np.argsort(-rets, kind="stable")
        keep_idx = order[:int(top_k)]
        keep = [episodes[i] for i in keep_idx]
        cutoff = float(rets[keep_idx[-1]]) if len(keep_idx) else 0.0
    else:
        if isinstance(threshold, str):
            if threshold == "median":
                cutoff = float(np.median(rets))
            elif threshold == "max":
                cutoff = float(rets.max())
            else:
                raise ValueError(f"unknown threshold spec: {threshold!r}")
        else:
            cutoff = float(threshold)

        keep = [e for e in episodes if e.return_ >= cutoff]
        if not keep:
            # Pathological: every episode below cutoff. Fall back to
            # the single best so the loop can still produce a signal.
            best = max(episodes, key=lambda e: e.return_)
            keep = [best]

    X = torch.cat([e.states for e in keep], dim=0)
    y = torch.cat([e.actions for e in keep], dim=0)

    # Class balancing: cap per-action sample count.
    if per_class_cap is not None and per_class_cap > 0:
        rng = np.random.default_rng(0)
        keep_idx = []
        for c in y.unique().tolist():
            cls_idx = (y == c).nonzero(as_tuple=True)[0].numpy()
            if len(cls_idx) > per_class_cap:
                cls_idx = rng.choice(cls_idx, size=per_class_cap,
                                     replace=False)
            keep_idx.extend(cls_idx.tolist())
        keep_idx = np.array(sorted(keep_idx))
        X = X[keep_idx]
        y = y[keep_idx]

    counts = {int(c): int((y == c).sum()) for c in y.unique().tolist()}
    stats = {
        "n_episodes_total": len(episodes),
        "n_episodes_kept": len(keep),
        "cutoff_return": cutoff,
        "return_min": float(rets.min()),
        "return_max": float(rets.max()),
        "return_mean": float(rets.mean()),
        "return_median": float(np.median(rets)),
        "n_samples": int(X.shape[0]),
        "class_counts": counts,
    }
    if verbose:
        print(f"  [filter] kept {len(keep)}/{len(episodes)} eps "
              f"(cutoff={cutoff:+.2f}, return range "
              f"{rets.min():+.1f}..{rets.max():+.1f}); "
              f"{X.shape[0]} samples; class counts={counts}")
    return X, y, stats
