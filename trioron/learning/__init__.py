"""Learning subsystem — anchor, stress, sleep, replay.  See spec §4."""
from .credit import CreditTracker, CreditConfig
from .frustration import FrustrationDetector, FrustrationConfig
from .manifold import ManifoldArchive, ManifoldConfig, get_code_boundary
from .rejuvenate import rejuvenate, find_rejuvenation_candidates
from .dream import dream_cycle, DreamConfig, DreamResult, interleaved_replay_batch
