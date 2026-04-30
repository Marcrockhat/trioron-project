"""Trioron — a dynamically growing neural architecture.

See trioron_blueprint.md for the design document.
"""

from .node import TrioronLayer
from .network import TrioronNetwork

__all__ = ["TrioronLayer", "TrioronNetwork"]
__version__ = "0.0.2"
