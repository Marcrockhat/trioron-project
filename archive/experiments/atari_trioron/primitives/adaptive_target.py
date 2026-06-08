"""Adaptive target-return scheduler — Vygotsky-style curriculum dial.

Implements the "target heating" idea: the satisfaction threshold sits
just above current-best achievement, raising as the donor plateaus and
cooling if the donor strains. Avoids both burnout (target unreachable
forever → unstable wandering) and premature lock-in (target so low
the donor settles immediately).

State machine per episode:

    if ret >= current_target:
        plateau_count += 1
        strain_count = 0
        if plateau_count >= plateau_K:
            current_target += raise_delta   # heat
            plateau_count = 0
    else:
        strain_count += 1
        plateau_count = 0
        if strain_count >= strain_M:
            current_target -= cool_delta    # cool
            current_target = max(current_target, running_best - 1)
            strain_count = 0

The cool-floor at `running_best - 1` ensures we never lower the bar
below an already-achieved level: progress is monotone-or-better.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AdaptiveTargetConfig:
    """Tunable knobs for the curriculum scheduler.

    `stretch` is the "ZPD margin" above ep1's baseline — how far above
    the agent's first-episode performance to set the initial target.
    Slightly positive means ep1 is just-frustrating, triggering at
    least one retrain. Set to 0 to satisfy ep1 (skip first retrain).

    `raise_delta` and `cool_delta` should be positive. `plateau_K` is
    the number of consecutive satisfied eps before heating; `strain_M`
    is consecutive frustrated eps before cooling.
    """
    stretch: float = 2.0
    raise_delta: float = 2.0
    cool_delta: float = 2.0
    plateau_K: int = 2
    strain_M: int = 3


@dataclass
class AdaptiveTarget:
    """Mutable state for the scheduler. Initialize once before the
    episode loop; call `update_after_episode` after each ep."""
    config: AdaptiveTargetConfig = field(default_factory=AdaptiveTargetConfig)
    current_target: Optional[float] = None
    running_best: float = float("-inf")
    plateau_count: int = 0
    strain_count: int = 0

    def initialize_from_first(self, ret: float) -> None:
        """Set initial target from ep1's return + stretch."""
        if self.current_target is None:
            self.current_target = ret + self.config.stretch
        if ret > self.running_best:
            self.running_best = ret

    def is_satisfied(self, ret: float) -> bool:
        """Per-ep ratchet check. Caller decides what to do with this
        (skip retrain, log, etc.)."""
        if self.current_target is None:
            return False
        return ret >= self.current_target

    def update_after_episode(self, ret: float) -> dict:
        """Advance state machine. Returns a small dict with the
        post-update target + counts so the caller can log/print."""
        if self.current_target is None:
            self.initialize_from_first(ret)
            return {
                "target": self.current_target,
                "running_best": self.running_best,
                "plateau_count": 0,
                "strain_count": 0,
                "transition": "init",
            }
        if ret > self.running_best:
            self.running_best = ret
        transition = "hold"
        if ret >= self.current_target:
            self.plateau_count += 1
            self.strain_count = 0
            if self.plateau_count >= self.config.plateau_K:
                self.current_target += self.config.raise_delta
                self.plateau_count = 0
                transition = "heat"
        else:
            self.strain_count += 1
            self.plateau_count = 0
            if self.strain_count >= self.config.strain_M:
                self.current_target -= self.config.cool_delta
                # Floor: never drop the bar below known-achievable.
                if self.current_target < self.running_best - 1.0:
                    self.current_target = self.running_best - 1.0
                self.strain_count = 0
                transition = "cool"
        return {
            "target": self.current_target,
            "running_best": self.running_best,
            "plateau_count": self.plateau_count,
            "strain_count": self.strain_count,
            "transition": transition,
        }


__all__ = ["AdaptiveTarget", "AdaptiveTargetConfig"]
