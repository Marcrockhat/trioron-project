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

__all__ = [
    "TrioronLayer",
    "TrioronNetwork",
    "CeilingsController",
    "DivisionDelta",
    "PreflightDecision",
    "division_param_delta",
]
__version__ = "0.0.2"
