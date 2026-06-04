"""Learning subsystem — anchor, stress, sleep, replay.  See spec §4."""
from .credit import CreditTracker, CreditConfig
from .frustration import FrustrationDetector, FrustrationConfig
from .manifold import ManifoldArchive, ManifoldConfig, get_code_boundary
from .rejuvenate import rejuvenate, find_rejuvenation_candidates
from .dream import dream_cycle, DreamConfig, DreamResult, interleaved_replay_batch
from .epigenetic_lock import (  # the namesake λ (restored s014; |w·g| driver s015)
    accumulate_saliency, accumulate_fisher, refresh_lambda, set_lambda,
    anchor, ewc_penalty, modulated_scale, fisher_loss,
    DEFAULT_DECAY, LAMBDA_FLOOR,
)
