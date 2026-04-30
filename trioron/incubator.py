"""Trioron — scripted incubation environment + contrastive curriculum.

Implements §5 of the blueprint:
- §5.1  An 8-dim sensory/physical state vector.
- §5.2  A scripted simulator (NOT an LLM) that emits state tuples on a
        clock, accepts an action vector, updates state per simple rules,
        and provides a reward.
- §5.3  A contrastive curriculum: 5 concept pairs (Hungry/Stuffed,
        Cold/Hot, Threat/Safe, Reachable/Unreachable, Owned/Foreign)
        and a margin-based contrastive loss for opposite states.

The contrastive-loss plateau is one of the three growth-trigger
conditions in §4 (the others — effective-rank saturation and
gradient-norm stability — live in triggers.py, step 4 of §8).

Design note: the blueprint specifies the 5 pairs but not the exact
state-dim layout. The mapping below is committed convention for steps
4–8; change it here and the rest of the system follows.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch


# ---------------------------------------------------------------------
# State dimension layout (k = 8, blueprint §5.1)
# ---------------------------------------------------------------------

DIM_ENERGY      = 0
DIM_TEMPERATURE = 1
DIM_SATIETY     = 2
DIM_THREAT      = 3
DIM_POSITION_X  = 4
DIM_TARGET_X    = 5
DIM_OWNED       = 6
DIM_TIME_PHASE  = 7

STATE_DIM = 8

# Action vector matches state shape: each component is a per-dim delta
# that the scripted physics will clip and apply. Kept minimal on purpose.
ACTION_DIM = STATE_DIM


# ---------------------------------------------------------------------
# Scripted environment (§5.2)
# ---------------------------------------------------------------------


@dataclass
class EnvConfig:
    action_scale: float = 0.1       # how much an action delta moves state
    hunger_rate: float = 0.005      # satiety drift per step toward 0
    temp_drift: float = 0.01        # temperature drift toward ambient
    ambient_temp: float = 0.5
    energy_cost_per_move: float = 0.002
    energy_recovery: float = 0.001
    noise_std: float = 0.005        # tiny Gaussian noise on transitions
    target_drift: float = 0.0       # set > 0 to make the target wander
    seed: Optional[int] = None


class ScriptedEnvironment:
    """Closed numeric environment for the incubation phase.

    The environment is a deterministic-ish ODE-like simulator:
    state at t+1 is state at t plus (action delta) plus a small
    drift term (hunger over time, temperature toward ambient,
    energy regeneration when not moving) plus light Gaussian noise.

    All state components are clipped to [0, 1] except DIM_OWNED which
    is held in {0, 1}. The simulator does NOT pretend to be physically
    realistic — its job is to produce a stream of grounded numeric
    tuples that the network can learn to represent.
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config or EnvConfig()
        self._rng = torch.Generator()
        if self.cfg.seed is not None:
            self._rng.manual_seed(self.cfg.seed)
        self.state: torch.Tensor = torch.zeros(STATE_DIM)
        self.t: int = 0
        self.reset()

    def reset(self) -> torch.Tensor:
        s = torch.rand(STATE_DIM, generator=self._rng)
        s[DIM_OWNED] = (s[DIM_OWNED] > 0.5).float()
        s[DIM_TIME_PHASE] = 0.0
        self.state = s
        self.t = 0
        return self.state.clone()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, dict]:
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action must be shape ({ACTION_DIM},), got {tuple(action.shape)}")

        s = self.state.clone()
        s = s + self.cfg.action_scale * action

        # Drift terms.
        s[DIM_SATIETY] -= self.cfg.hunger_rate
        s[DIM_TEMPERATURE] += self.cfg.temp_drift * (self.cfg.ambient_temp - s[DIM_TEMPERATURE])

        movement = abs(action[DIM_POSITION_X].item())
        s[DIM_ENERGY] -= self.cfg.energy_cost_per_move * movement
        s[DIM_ENERGY] += self.cfg.energy_recovery * (1.0 - movement)

        if self.cfg.target_drift > 0:
            s[DIM_TARGET_X] += self.cfg.target_drift * (torch.rand(1, generator=self._rng).item() - 0.5)

        s += self.cfg.noise_std * torch.randn(STATE_DIM, generator=self._rng)

        # Clip / quantize.
        s = torch.clamp(s, 0.0, 1.0)
        s[DIM_OWNED] = (s[DIM_OWNED] > 0.5).float()
        self.t += 1
        s[DIM_TIME_PHASE] = (self.t % 100) / 100.0

        self.state = s
        reward = self._reward(s)
        info = {"t": self.t}
        return s.clone(), reward, info

    def _reward(self, s: torch.Tensor) -> float:
        """Comfort-band reward: rewards mid-range satiety/energy/temp,
        penalizes proximity to extremes and high threat."""
        comfort = -(
            (s[DIM_SATIETY] - 0.6).abs()
            + (s[DIM_ENERGY] - 0.6).abs()
            + (s[DIM_TEMPERATURE] - 0.5).abs()
        )
        threat_pen = -s[DIM_THREAT]
        return float(comfort + 0.5 * threat_pen)


# ---------------------------------------------------------------------
# Contrastive curriculum (§5.3)
# ---------------------------------------------------------------------

PAIR_HUNGRY_STUFFED       = "hungry_stuffed"
PAIR_COLD_HOT             = "cold_hot"
PAIR_THREAT_SAFE          = "threat_safe"
PAIR_REACHABLE_UNREACHABLE = "reachable_unreachable"
PAIR_OWNED_FOREIGN        = "owned_foreign"

PAIR_NAMES: List[str] = [
    PAIR_HUNGRY_STUFFED,
    PAIR_COLD_HOT,
    PAIR_THREAT_SAFE,
    PAIR_REACHABLE_UNREACHABLE,
    PAIR_OWNED_FOREIGN,
]


class ContrastiveCurriculum:
    """Generates batches of opposite state pairs for the contrastive loss.

    Each call to sample_pair(name, batch) returns (states_low, states_high),
    two batches of states that differ primarily in the dimension(s) the
    pair targets, with all other dims drawn iid uniform so the network
    can't shortcut on incidental correlations.
    """

    def __init__(self, seed: Optional[int] = None, low: float = 0.1, high: float = 0.9):
        if not (0.0 <= low < high <= 1.0):
            raise ValueError("low and high must satisfy 0 <= low < high <= 1")
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)
        self.low = low
        self.high = high

    def sample_pair(self, name: str, batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if name not in PAIR_NAMES:
            raise ValueError(f"Unknown pair {name!r}; valid: {PAIR_NAMES}")
        a = torch.rand(batch, STATE_DIM, generator=self._rng)
        b = torch.rand(batch, STATE_DIM, generator=self._rng)
        a[:, DIM_OWNED] = (a[:, DIM_OWNED] > 0.5).float()
        b[:, DIM_OWNED] = (b[:, DIM_OWNED] > 0.5).float()

        if name == PAIR_HUNGRY_STUFFED:
            a[:, DIM_SATIETY] = self.low
            b[:, DIM_SATIETY] = self.high
        elif name == PAIR_COLD_HOT:
            a[:, DIM_TEMPERATURE] = self.low
            b[:, DIM_TEMPERATURE] = self.high
        elif name == PAIR_THREAT_SAFE:
            a[:, DIM_THREAT] = self.high   # threat = high
            b[:, DIM_THREAT] = self.low    # safe  = low
        elif name == PAIR_REACHABLE_UNREACHABLE:
            # Reachable: pos and target close. Unreachable: far apart.
            pos = torch.rand(batch, generator=self._rng)
            a[:, DIM_POSITION_X] = pos
            a[:, DIM_TARGET_X] = pos + 0.05 * (torch.rand(batch, generator=self._rng) - 0.5)
            a[:, DIM_TARGET_X] = torch.clamp(a[:, DIM_TARGET_X], 0.0, 1.0)
            b[:, DIM_POSITION_X] = torch.full((batch,), self.low)
            b[:, DIM_TARGET_X] = torch.full((batch,), self.high)
        elif name == PAIR_OWNED_FOREIGN:
            a[:, DIM_OWNED] = 1.0
            b[:, DIM_OWNED] = 0.0
        return a, b

    def sample_all(self, batch_per_pair: int) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        return [(n, *self.sample_pair(n, batch_per_pair)) for n in PAIR_NAMES]


# ---------------------------------------------------------------------
# Contrastive loss (§5.3)
# ---------------------------------------------------------------------


def contrastive_loss(h_a: torch.Tensor, h_b: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Hinge loss on Euclidean distance: penalize whenever opposite-pair
    representations come closer than `margin` apart.

        L = mean( max(0, margin - ||h_a - h_b||_2) )

    Returns a scalar autograd-attached tensor. This is the loss whose
    plateau is condition (1) of the three-condition growth trigger (§4).
    """
    if h_a.shape != h_b.shape:
        raise ValueError(f"shape mismatch: {tuple(h_a.shape)} vs {tuple(h_b.shape)}")
    dist = torch.linalg.vector_norm(h_a - h_b, dim=-1)
    return torch.clamp(margin - dist, min=0.0).mean()
