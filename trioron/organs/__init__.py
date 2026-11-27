"""Organs — what a body is made of, kept apart from the learners (s072).

Rocky, s072: "Rebuild the code, and make sure we clearly separate the organs
and the triorons core." The rule of the partition (spec §9.16):

  * `trioron/core` and `trioron/pcll` are the learners. They import nothing
    from here and name no organ rule; `pcll/mixed.py` only keeps empty places
    (`sitting`, `numa`, `court`) that a body may fill on an organ.
  * `trioron/organs` holds what belongs to an organ or a body of organs: how
    an organ sees (blur), how it is judged (sitting), how its growth is paid
    (numa), and the fourth organ, which tries divisions for the others.
  * `experiments/incubator` holds the world, the drivers, the gates and the
    probes, and nothing an organism is made of.

Bodies saved before the move name the rules by their old modules; the aliases
below keep those bodies readable. They live here, not in the core.
"""
import sys as _sys

from . import blur, fourth, numa, sitting
from .blur import Blur
from .body import Body
from .fourth import FourthOrgan
from .numa import NumaLedger
from .sitting import SittingCap

_sys.modules.setdefault("trioron.pcll.numa", numa)
_sys.modules.setdefault("trioron.pcll.sitting", sitting)

__all__ = ["Blur", "Body", "FourthOrgan", "NumaLedger", "SittingCap"]
