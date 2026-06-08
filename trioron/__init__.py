"""Trioron — a dynamically growing neural architecture.

v2.0 modules live under trioron/core/, trioron/phenotype/, etc.
v1 modules live under trioron/legacy/ and are re-exported here
for backward compatibility.
"""

from .legacy.node import TrioronLayer
from .legacy.network import TrioronNetwork
from .legacy.ceilings import (
    CeilingsController,
    DivisionDelta,
    PreflightDecision,
    division_param_delta,
)
from .legacy.classification import (
    SplitClassificationTask,
    SplitClassificationReport,
    split_mnist_tasks,
    split_cifar100_tasks,
    extend_output_head,
    masked_cross_entropy,
    predict_full,
    accuracy,
    summarize,
)
from .legacy.profile import (
    TrioronProfile,
    OPEN,
    REASONING,
    CLASSIFICATION,
    EDGE,
    PRESETS,
)
from .legacy.spatial import (
    grid_positions_2d,
    pixel_positions_2d,
    pool_id,
    pool_centroid,
    build_lcn_mask,
    locality_metric,
)
from .legacy.api import (
    cosine_logits,
    track_engagement,
    update_credit,
    apply_credit_mask,
)

__all__ = [
    "TrioronLayer",
    "TrioronNetwork",
    "CeilingsController",
    "DivisionDelta",
    "PreflightDecision",
    "division_param_delta",
    "SplitClassificationTask",
    "SplitClassificationReport",
    "split_mnist_tasks",
    "split_cifar100_tasks",
    "extend_output_head",
    "masked_cross_entropy",
    "predict_full",
    "accuracy",
    "summarize",
    "TrioronProfile",
    "OPEN",
    "REASONING",
    "CLASSIFICATION",
    "EDGE",
    "PRESETS",
    "grid_positions_2d",
    "pixel_positions_2d",
    "pool_id",
    "pool_centroid",
    "build_lcn_mask",
    "locality_metric",
    "cosine_logits",
    "track_engagement",
    "update_credit",
    "apply_credit_mask",
]
# Single source of truth = pyproject.toml (exposed via installed metadata).
# Fallback is only hit when running from raw source with no install present;
# keep it in sync with pyproject's version on release.
try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError

    __version__ = _pkg_version("trioron")
except PackageNotFoundError:  # pragma: no cover - raw source / not installed
    __version__ = "0.2.2"
