"""Learning subsystem — anchor, stress, sleep, replay.  See spec §4."""
from .credit import CreditTracker, CreditConfig
from .frustration import FrustrationDetector, FrustrationConfig
from .manifold import ManifoldArchive, ManifoldConfig, get_code_boundary
from .router import (  # H-space task selector (promoted from bench_chained_15_v2)
    ManifoldRouter, build_h_archive_from_data, build_h_archive_from_manifold,
)
from .shift import (  # streaming context-shift detection (§10.2.1 aperiodic cut)
    SurpriseBaseline, ShiftDetector, ShiftEvent,
)
from .rejuvenate import rejuvenate, find_rejuvenation_candidates
from .dream import dream_cycle, DreamConfig, DreamResult, interleaved_replay_batch
from .epigenetic_lock import (  # the namesake λ (restored s014; |w·g| driver s015)
    accumulate_saliency, accumulate_fisher, refresh_lambda, set_lambda,
    anchor, ewc_penalty, modulated_scale, fisher_loss,
    DEFAULT_DECAY, LAMBDA_FLOOR,
)
