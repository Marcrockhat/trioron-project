"""trioron.api — the public surface (spec §9.1: "v2 public API").

Three ways in, one import path:

1. **Continual classification / donors (v1 flows, unchanged).**
   ``TaskData`` -> ``build_donor`` -> ``absorb`` / ``extend`` ->
   ``deploy_agent``.  These are the v1 API functions, now implemented
   in ``trioron.legacy.api``; every example in MANUAL.md / QUICKSTART.md
   imports them from here.

2. **The v2 substrate (gradient learner, triparametric (w, λ, u)).**
   ``construct(base=seeded(...), envelope=Envelope(...),
   dispatch_table=default_dispatch_table())`` -> ``compile()`` ->
   ``prepare_training()`` -> ``trainable_tensors()``.  See spec §2–§6
   and docs/TRIORON_MANUAL.md.

3. **Phasecyte (gradient-free, phase-coherent learner) — nesting and
   the wake/dream loop.**  ``PhasecyteNest`` / ``PhasecyteLeaf`` /
   ``dream_distill`` / ``dreamed_predict``.  See spec §10.

The embodied "organism" recipe (drives -> primitives -> router ->
structural dreaming; tour/phasecyte.html) is NOT yet a package API — it
lives in archive/experiments/world/ and is being reduced to a
"declare your drives" contract (handoff s050).
"""
from __future__ import annotations

# 1. v1 flows (the documented donor API)
from .legacy.api import *  # noqa: F401,F403
from .legacy.api import __all__ as _legacy_all

# 2. v2 substrate
from .core import Envelope, construct  # noqa: F401
from .bases import seeded, minimal, frozen, compose  # noqa: F401
from .phenotype import default_dispatch_table  # noqa: F401
from .lifecycle import export_dense, verify_export  # noqa: F401  (spec §5.4 dense export)

# 3. Phasecyte
from .pcll import (PhasecyteLeaf, PhasecyteNest, dream_distill,  # noqa: F401
                   dreamed_predict, PCLLController)

__all__ = list(_legacy_all) + [
    "Envelope", "construct", "seeded", "minimal", "frozen", "compose",
    "default_dispatch_table", "export_dense", "verify_export",
    "PhasecyteLeaf", "PhasecyteNest", "dream_distill", "dreamed_predict",
    "PCLLController",
]
