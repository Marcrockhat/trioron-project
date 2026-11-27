"""Where a body is kept, and the guards on it.

HOME is the organism's home (``VEE_HOME``; the first organism's by default).  The stage
file HOME/stage names the stage it is in; a driver may only run the stage it is written
for — a stage is entered by Rocky's sign-off, never promoted by code (matron rule 3).
A body is not fed above 85% of host RAM, swap never counted (incubator design §5.1).
"""
from __future__ import annotations
import os

OUT = "outputs/incubator"
HOME = os.environ.get("VEE_HOME", f"{OUT}/vee")
RAM_CEILING = 0.85


def stage() -> str:
    p = f"{HOME}/stage"
    return open(p).read().strip() if os.path.exists(p) else ""


def require_stage(name: str) -> None:
    if stage() != name:
        raise SystemExit(f"stage file {HOME}/stage says '{stage()}', this driver runs '{name}' only — "
                         f"a stage is entered by Rocky's sign-off, not by code")


def ram_used() -> float:
    m = {l.split(":")[0]: float(l.split()[1]) for l in open("/proc/meminfo")}
    return 1 - m["MemAvailable"] / m["MemTotal"]
