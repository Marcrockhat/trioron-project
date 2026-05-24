"""Lifecycle — growth, evolution, deployment.  See spec §5."""
from .grow import divide, check_growth_trigger, GrowthConfig, GrowthEvent
from .saliency import compute_saliency, SaliencyConfig
from .compact import compact, CompactConfig, CompactResult
from .ship import ship
from .wake import wake, wake_for_training, get_optimizer_state
from .extend import extend_envelope
from .graft import graft, GraftResult, compose_heads
