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
from .legacy.api import absorb as _absorb_v1, pool_matched_absorb as _pma_v1

# 2. v2 substrate
from .core import Envelope, construct  # noqa: F401
from .bases import seeded, minimal, frozen, compose  # noqa: F401
from .phenotype import default_dispatch_table  # noqa: F401
from .lifecycle import export_dense, verify_export  # noqa: F401  (spec §5.4 dense export)
from .lifecycle import graft, GraftResult  # noqa: F401  (spec §5.3 grafting)
from .core.construct import Substrate as _Substrate

# 3. Phasecyte
from .pcll import (PhasecyteLeaf, PhasecyteNest, dream_distill,  # noqa: F401
                   dreamed_predict, PCLLController)



def absorb(*substrates, donor_paths=None, out_path=None, freeze=False,
           merge_output=True, wiring="none"):
    """Absorb — one name, two substrates (spec §5.3 / §9.1).

    **v2 (positional ``Substrate`` objects):**
    ``absorb(recipient, donor1, donor2, ...)`` grafts every donor INTO the
    recipient in place (head-merged by default: donor interior cells are
    transplanted with their dendritic state, donor edges into output cell
    *j* land on the recipient's output cell *j*, biases summed) so that
    ``recipient(x) == old_recipient(x) + Σ donor(x)`` exactly. Requires
    equal input and output widths (siblings: same task grown from
    different seeds; the same drive leaf of two organisms). ``freeze=True``
    makes the absorbed cells dormant (Protocol A); default False keeps
    them trainable (Protocol B). ``merge_output=False`` keeps donor
    output cells (routed heads, ``compose_heads``); ``wiring`` is then
    the cross-edge policy. Returns the list of ``GraftResult``.
    Do NOT raw-TD/fine-tune a k-fold Q-sum right after absorbing without
    rescaling or locking the transplanted cells (s051: −92 survival).

    **v1 (keyword ``donor_paths=[...], out_path=...``):** the branch-
    granularity donor composition of the 1.0 API, unchanged — returns the
    organism path.
    """
    if donor_paths is not None or out_path is not None:
        if substrates:
            raise TypeError("absorb: pass EITHER v2 Substrate objects "
                            "positionally OR v1 donor_paths/out_path")
        return _absorb_v1(donor_paths=donor_paths, out_path=out_path)
    if len(substrates) < 2:
        raise TypeError("absorb(recipient, donor, ...): need a recipient "
                        "and at least one donor")
    if not all(isinstance(x, _Substrate) for x in substrates):
        raise TypeError("absorb: positional arguments must be v2 "
                        "Substrate objects (v1 donors go through "
                        "donor_paths=/out_path=)")
    rec, *donors = substrates
    out = [graft(rec, d, freeze=freeze, wiring=wiring,
                 merge_output=merge_output) for d in donors]
    if rec.arena.bias.requires_grad:
        rec.compile()
    return out


absorb.__doc__ = (absorb.__doc__ or "") + "\n\nv1 docstring:\n" + (
    _absorb_v1.__doc__ or "")


def pool_matched_absorb(recipient, donor, **kw):
    """Cell-granularity absorption. v1 ``TrioronNetwork`` pair → the 1.0
    pool-matched absorb (all its kwargs). v2 ``Substrate`` pair → the
    head-merged graft (``absorb(recipient, donor)``; kwargs ``freeze`` /
    ``wiring`` / ``merge_output`` honoured, v1-only kwargs rejected)."""
    if isinstance(recipient, _Substrate):
        bad = set(kw) - {"freeze", "wiring", "merge_output"}
        if bad:
            raise TypeError(f"pool_matched_absorb (v2 substrate): "
                            f"unsupported kwargs {sorted(bad)}")
        return absorb(recipient, donor, **kw)[0]
    return _pma_v1(recipient, donor, **kw)


pool_matched_absorb.__doc__ += "\n\nv1 docstring:\n" + (_pma_v1.__doc__ or "")


__all__ = list(_legacy_all) + [
    "Envelope", "construct", "seeded", "minimal", "frozen", "compose",
    "default_dispatch_table", "export_dense", "verify_export",
    "graft", "GraftResult",
    "PhasecyteLeaf", "PhasecyteNest", "dream_distill", "dreamed_predict",
    "PCLLController",
]
