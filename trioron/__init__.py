"""Trioron — a dynamically growing neural architecture.

See trioron_blueprint.md for the design document.
"""

from .node import TrioronLayer
from .network import TrioronNetwork
from .ceilings import (
    CeilingsController,
    DivisionDelta,
    PreflightDecision,
    division_param_delta,
)
from .classification import (
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
from .profile import (
    TrioronProfile,
    OPEN,
    REASONING,
    CLASSIFICATION,
    EDGE,
    PRESETS,
)
from .spatial import (
    grid_positions_2d,
    pixel_positions_2d,
    pool_id,
    pool_centroid,
    build_lcn_mask,
    locality_metric,
)
from .api import (
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
__version__ = "1.1.0"
