"""trioron.pcll — Phase-Coherent Lock-in Learning.  See spec §10 / §9.15.

The streaming, single-pass, label-free alternative to gradient-trained
depth: magnitudes phase-coded into 1000 quanta, coherently integrated
over a period (real class ∝N, noise ∝√N), matched-filter readout with
parameter-free floors. Method spec of record:
docs/design/receptor_period_frustration.md; integration design:
docs/design/pcll_substrate_integration.md.

Typical wiring (the driver's one step — constructing the controller
attaches it; thereafter Substrate.end_task() runs the boundary meeting):

    controller = PCLLController(substrate)
    for batch in period_stream:
        controller.observe(batch)
    report = substrate.end_task()   # MeetingReport
"""
from .controller import RETIRE_PATIENCE, MeetingReport, PCLLController, PCLLResolution
from .division import circ_2means, try_divide
from .labels import LabelTapBank
from .lockin import LockInView, TrigBank, deposit, evidence_mask, matched_k, reset
from .mixed import MixedStreamController
from .nest import (PhasecyteLeaf, PhasecyteNest, dream_distill,
                   dreamed_predict)
from .progenitor import (
    K_DISCRETE,
    FirstSittingReport,
    Germline,
    PerceptionGenesis,
    germline_base,
)
from .receptor import N_QUANTA, phase, quantize, theta_discrete
from .resolve import EMPTY, FRUSTRATED, RESOLVED, Resolution, Resolver, template
from .signature import LearnedClass, ScheduleLearner
from .stress import (
    DISCRIMINATION,
    DRIVE_FLOOR,
    HABITUATION_DECAY,
    SENSATION,
    StressRouter,
)

__all__ = [
    "RETIRE_PATIENCE", "MeetingReport", "PCLLController", "PCLLResolution",
    "circ_2means", "try_divide", "MixedStreamController", "LabelTapBank",
    "PhasecyteLeaf", "PhasecyteNest", "dream_distill", "dreamed_predict",
    "LockInView", "TrigBank", "deposit", "evidence_mask", "matched_k", "reset",
    "K_DISCRETE", "FirstSittingReport", "Germline", "PerceptionGenesis",
    "germline_base",
    "N_QUANTA", "phase", "quantize", "theta_discrete",
    "EMPTY", "FRUSTRATED", "RESOLVED", "Resolution", "Resolver", "template",
    "LearnedClass", "ScheduleLearner",
    "DISCRIMINATION", "DRIVE_FLOOR", "HABITUATION_DECAY", "SENSATION",
    "StressRouter",
]
