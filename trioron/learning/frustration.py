"""Frustration signal — per-task plateau detector.  See spec §4.2."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque


@dataclass
class FrustrationConfig:
    window_size: int = 50
    epsilon_improvement: float = 0.01
    epsilon_reset: float = 0.1
    hinge: int = 2
    gain: float = 0.5
    ceiling: float = 4.0


class FrustrationDetector:
    """Accumulates task loss in windows and produces a frustration multiplier."""

    def __init__(self, cfg: FrustrationConfig | None = None) -> None:
        self.cfg = cfg or FrustrationConfig()
        self._window: list[float] = []
        self._prev_mean: float | None = None
        self._stuck: int = 0
        self._step_count: int = 0

    @property
    def multiplier(self) -> float:
        cfg = self.cfg
        if self._stuck < cfg.hinge:
            return 1.0
        return min(cfg.ceiling, 1.0 + cfg.gain * (self._stuck - cfg.hinge + 1))

    @property
    def is_frustrated(self) -> bool:
        return self._stuck >= self.cfg.hinge

    def step(self, loss_value: float) -> float:
        """Record one batch loss and return the current multiplier."""
        self._window.append(loss_value)
        self._step_count += 1

        if len(self._window) >= self.cfg.window_size:
            self._evaluate_window()

        return self.multiplier

    def _evaluate_window(self) -> None:
        mean = sum(self._window) / len(self._window)
        cfg = self.cfg

        if self._prev_mean is not None:
            improvement = self._prev_mean - mean
            if improvement < cfg.epsilon_improvement:
                self._stuck += 1
            elif improvement > cfg.epsilon_reset:
                self._stuck = 0

        self._prev_mean = mean
        self._window = []

    def reset(self) -> None:
        """Reset at task boundary."""
        self._window = []
        self._prev_mean = None
        self._stuck = 0
        self._step_count = 0
