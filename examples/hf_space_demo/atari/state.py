"""Pong + Breakout state builders + skill gates + action mappings.

Self-contained port of the inference-time bits of:
  experiments/atari_trioron/primitives/synthetic_env.py
  experiments/atari_trioron/primitives/breakout_state.py
  experiments/atari_trioron/primitives/pong_inference.py
  experiments/atari_trioron/primitives/pong_skill_inference.py
  experiments/atari_trioron/primitives/breakout_skill_inference.py

Both games use STATE_DIM = 9 so a single trioron L0 layer can serve
either game; Breakout pads `opp_y` to a constant since it has no
opponent paddle. Class IDs are disjoint across games (Pong 210..232,
Breakout 300..312) so a multi-branch organism can host both without
collision.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .features import extract_pong, extract_breakout


STATE_DIM = 9
FRAME_HW = 84

# ---------------------------------------------------------------------
# Pong-side constants
# ---------------------------------------------------------------------

PADDLE_HEIGHT_84 = 8.15
SMASH_TRIGGER_X = 70.0
PREPOS_ANCHOR_Y = 42.0
PONG_ACTION_DEAD_ZONE = 4.0
PADDLE_LOOKAHEAD_PONG = 1.0
_PRED_DX_EPS = 1e-3

PONG_FIELDS = ["ball_x", "ball_y", "ball_dx", "ball_dy",
               "paddle_y", "paddle_dy", "ball_speed", "pred_y", "opp_y"]
PONG_STANDARDIZE: Dict[str, Tuple[float, float]] = {
    "ball_x":     (42.0, 24.25),
    "ball_y":     (42.0, 24.25),
    "ball_dx":    (0.0,  2.89),
    "ball_dy":    (0.0,  2.89),
    "paddle_y":   (42.0, 24.25),
    "paddle_dy":  (0.0,  4.62),
    "ball_speed": (2.50, 1.30),
    "pred_y":     (42.0, 24.25),
    "opp_y":      (42.0, 24.25),
}

# Pong RAM ranges (empirical, calibrated 2026-05-07).
RAM_BALL_X_RANGE_PONG = (68.0, 205.0)
RAM_BALL_Y_RANGE_PONG = (38.0, 203.0)
RAM_PADDLE_Y_RANGE_PONG = (38.0, 203.0)
ALE_FRAME_SKIP = 1

# Pong skill class IDs.
SKILL_CATCH_UP, SKILL_CATCH_DOWN, SKILL_CATCH_HOLD = 210, 211, 212
SKILL_SMASH_UP, SKILL_SMASH_DOWN = 220, 221
SKILL_PREPOS_UP, SKILL_PREPOS_DOWN, SKILL_PREPOS_HOLD = 230, 231, 232

# Pong ALE actions.
ALE_HOLD, ALE_UP, ALE_DOWN = 0, 2, 3

# Pong: skill-class → ALE action.
PONG_SKILL_CLASS_TO_ALE: Dict[int, int] = {
    SKILL_CATCH_UP:    ALE_UP,
    SKILL_CATCH_DOWN:  ALE_DOWN,
    SKILL_CATCH_HOLD:  ALE_HOLD,
    SKILL_SMASH_UP:    ALE_UP,
    SKILL_SMASH_DOWN:  ALE_DOWN,
    SKILL_PREPOS_UP:   ALE_UP,
    SKILL_PREPOS_DOWN: ALE_DOWN,
    SKILL_PREPOS_HOLD: ALE_HOLD,
}

PONG_SKILL_CLASS_SETS: Dict[str, set] = {
    "CATCH":  {SKILL_CATCH_UP, SKILL_CATCH_DOWN, SKILL_CATCH_HOLD},
    "SMASH":  {SKILL_SMASH_UP, SKILL_SMASH_DOWN},
    "PREPOS": {SKILL_PREPOS_UP, SKILL_PREPOS_DOWN, SKILL_PREPOS_HOLD},
}

# Pong: paddle edge clamps (empirical).
PADDLE_TOP_84_PONG = 5.0
PADDLE_BOTTOM_84_PONG = 87.0


# ---------------------------------------------------------------------
# Breakout-side constants
# ---------------------------------------------------------------------

PADDLE_Y_84 = 76.0
SETTLE_ANCHOR_X = 42.0
ACTION_DEAD_ZONE = 1.0
PADDLE_LOOKAHEAD_BREAKOUT = 0.0
OPP_Y_PAD_RAW = 42.0
_PRED_DY_EPS = 1e-3

# Breakout RAM ranges.
RAM_PADDLE_X_RANGE = (57.0, 201.0)
RAM_BALL_X_RANGE_BO = (49.0, 200.0)
RAM_BALL_Y_RANGE_BO = (80.0, 200.0)

# Breakout skill class IDs.
SKILL_RECEIVE_LEFT, SKILL_RECEIVE_RIGHT, SKILL_RECEIVE_HOLD = 300, 301, 302
SKILL_SETTLE_LEFT, SKILL_SETTLE_RIGHT, SKILL_SETTLE_HOLD = 310, 311, 312

# Breakout ALE actions.
ALE_NOOP, ALE_FIRE, ALE_RIGHT, ALE_LEFT = 0, 1, 2, 3

BREAKOUT_SKILL_CLASS_TO_ALE: Dict[int, int] = {
    SKILL_RECEIVE_LEFT:  ALE_LEFT,
    SKILL_RECEIVE_RIGHT: ALE_RIGHT,
    SKILL_RECEIVE_HOLD:  ALE_NOOP,
    SKILL_SETTLE_LEFT:   ALE_LEFT,
    SKILL_SETTLE_RIGHT:  ALE_RIGHT,
    SKILL_SETTLE_HOLD:   ALE_NOOP,
}

BREAKOUT_SKILL_CLASS_SETS: Dict[str, set] = {
    "RECEIVE": {SKILL_RECEIVE_LEFT, SKILL_RECEIVE_RIGHT, SKILL_RECEIVE_HOLD},
    "SETTLE":  {SKILL_SETTLE_LEFT,  SKILL_SETTLE_RIGHT,  SKILL_SETTLE_HOLD},
}

PADDLE_LEFT_84_BO = 20.0
PADDLE_RIGHT_84_BO = 76.0


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _ram_to_84(value: float, lo: float, hi: float) -> float:
    return (float(value) - lo) * 84.0 / (hi - lo)


def _fold_to_range(x: float, lo: float = 0.0, hi: float = FRAME_HW) -> float:
    span = hi - lo
    rel = (x - lo) % (2.0 * span)
    if rel < 0.0:
        rel += 2.0 * span
    if rel > span:
        rel = 2.0 * span - rel
    return lo + rel


def predict_ball_y_at_impact(ball_x: float, ball_y: float,
                             ball_dx: float, ball_dy: float,
                             paddle_x: float = 84.0) -> float:
    if ball_dx <= _PRED_DX_EPS:
        return float(ball_y)
    dx_to_impact = paddle_x - ball_x
    if dx_to_impact <= 0.0:
        return float(ball_y)
    t = dx_to_impact / ball_dx
    return _fold_to_range(ball_y + ball_dy * t)


def predict_ball_x_at_impact(ball_x: float, ball_y: float,
                             ball_dx: float, ball_dy: float,
                             paddle_y: float = PADDLE_Y_84) -> float:
    if ball_dy <= _PRED_DY_EPS:
        return float(ball_x)
    dy_to_impact = paddle_y - ball_y
    if dy_to_impact <= 0.0:
        return float(ball_x)
    t = dy_to_impact / ball_dy
    return _fold_to_range(ball_x + ball_dx * t)


def standardize_pong(state: np.ndarray) -> np.ndarray:
    out = state.astype(np.float32, copy=True)
    flat = out.reshape(-1, STATE_DIM)
    for i, k in enumerate(PONG_FIELDS):
        m, s = PONG_STANDARDIZE[k]
        flat[:, i] = (flat[:, i] - m) / s
    return flat.reshape(state.shape)


# Breakout uses the same per-axis stats; the layout differs in field
# semantics (paddle_x at idx 4, pred_x at idx 7) but the (mean, std)
# values per index are identical because all spatial axes share
# (42, 24.25) and velocity axes share (0, 2.89). Reusing the Pong
# function works.
standardize_breakout = standardize_pong


# ---------------------------------------------------------------------
# Pong state builder
# ---------------------------------------------------------------------


class PongSkillStateBuilder:
    """Differences ALE RAM frame-to-frame to produce the standardized
    9-d state vector the Pong skill donors consume."""

    def __init__(self) -> None:
        self.prev_ball_x = 0.0
        self.prev_ball_y = 0.0
        self.prev_paddle_y = 0.0
        self.has_prev = False

    def reset(self) -> None:
        self.has_prev = False

    def step(self, ale):
        s = extract_pong(ale)
        if not s.ball_in_play:
            self.has_prev = False
            return None, None
        ball_x84 = _ram_to_84(s.ball_x, *RAM_BALL_X_RANGE_PONG)
        ball_y84 = _ram_to_84(s.ball_y, *RAM_BALL_Y_RANGE_PONG)
        paddle_y84 = (_ram_to_84(s.my_paddle_y, *RAM_PADDLE_Y_RANGE_PONG)
                      + PADDLE_HEIGHT_84 / 2.0)
        opp_y84 = (_ram_to_84(s.opp_paddle_y, *RAM_PADDLE_Y_RANGE_PONG)
                   + PADDLE_HEIGHT_84 / 2.0)
        if not self.has_prev:
            self.prev_ball_x = ball_x84
            self.prev_ball_y = ball_y84
            self.prev_paddle_y = paddle_y84
            self.has_prev = True
            return None, None
        ball_dx = (ball_x84 - self.prev_ball_x) / ALE_FRAME_SKIP
        ball_dy = (ball_y84 - self.prev_ball_y) / ALE_FRAME_SKIP
        paddle_dy = (paddle_y84 - self.prev_paddle_y) / ALE_FRAME_SKIP
        ball_speed = float(np.sqrt(ball_dx ** 2 + ball_dy ** 2))
        pred_y = predict_ball_y_at_impact(ball_x84, ball_y84, ball_dx, ball_dy)
        raw = np.array([
            ball_x84, ball_y84, ball_dx, ball_dy,
            paddle_y84, paddle_dy,
            ball_speed, pred_y, opp_y84,
        ], dtype=np.float32)
        self.prev_ball_x = ball_x84
        self.prev_ball_y = ball_y84
        self.prev_paddle_y = paddle_y84
        self.has_prev = True
        std = standardize_pong(raw[None, :])
        return torch.from_numpy(std), raw


def select_skill_pong(raw_state: np.ndarray, prev_skill: str = "CATCH") -> str:
    ball_dx = float(raw_state[2])
    ball_x = float(raw_state[0])
    if ball_dx <= _PRED_DX_EPS:
        return "PREPOS"
    if ball_x < SMASH_TRIGGER_X:
        return "CATCH"
    return "SMASH"


@torch.no_grad()
def select_action_pong(donor, state_vec: torch.Tensor, skill: str,
                       paddle_y84: float = 42.0,
                       fallback_action: int = ALE_HOLD) -> int:
    eligible = PONG_SKILL_CLASS_SETS[skill]
    logits = donor(state_vec, routing="soft")
    if isinstance(logits, tuple):
        logits = logits[0]
    union = list(donor.union_classes)
    masked = torch.full_like(logits, float("-inf"))
    for j, c in enumerate(union):
        if int(c) in eligible:
            masked[:, j] = logits[:, j]
    if torch.isinf(masked).all():
        return fallback_action
    pred_idx = int(masked[0].argmax())
    pred_class = int(union[pred_idx])
    action = PONG_SKILL_CLASS_TO_ALE[pred_class]
    if action == ALE_UP and paddle_y84 < PADDLE_TOP_84_PONG:
        return ALE_HOLD
    if action == ALE_DOWN and paddle_y84 > PADDLE_BOTTOM_84_PONG:
        return ALE_HOLD
    return action


# ---------------------------------------------------------------------
# Breakout state builder
# ---------------------------------------------------------------------


class BreakoutStateBuilder:
    """9-d Breakout state. opp_y is padded to OPP_Y_PAD_RAW since
    Breakout has no opponent paddle — keeps shape compatibility with
    Pong donors."""

    def __init__(self) -> None:
        self.prev_ball_x = 0.0
        self.prev_ball_y = 0.0
        self.prev_paddle_x = 0.0
        self.has_prev = False

    def reset(self) -> None:
        self.has_prev = False

    def step(self, ale):
        s = extract_breakout(ale)
        if not s.ball_in_play:
            self.has_prev = False
            return None, None
        ball_x84 = _ram_to_84(s.ball_x, *RAM_BALL_X_RANGE_BO)
        ball_y84 = _ram_to_84(s.ball_y, *RAM_BALL_Y_RANGE_BO)
        paddle_x84 = _ram_to_84(s.paddle_x, *RAM_PADDLE_X_RANGE)
        if not self.has_prev:
            self.prev_ball_x = ball_x84
            self.prev_ball_y = ball_y84
            self.prev_paddle_x = paddle_x84
            self.has_prev = True
            return None, None
        ball_dx = (ball_x84 - self.prev_ball_x) / ALE_FRAME_SKIP
        ball_dy = (ball_y84 - self.prev_ball_y) / ALE_FRAME_SKIP
        paddle_dx = (paddle_x84 - self.prev_paddle_x) / ALE_FRAME_SKIP
        ball_speed = float(np.sqrt(ball_dx ** 2 + ball_dy ** 2))
        pred_x = predict_ball_x_at_impact(ball_x84, ball_y84, ball_dx, ball_dy)
        raw = np.array([
            ball_x84, ball_y84, ball_dx, ball_dy,
            paddle_x84, paddle_dx,
            ball_speed, pred_x, OPP_Y_PAD_RAW,
        ], dtype=np.float32)
        self.prev_ball_x = ball_x84
        self.prev_ball_y = ball_y84
        self.prev_paddle_x = paddle_x84
        self.has_prev = True
        std = standardize_breakout(raw[None, :])
        return torch.from_numpy(std), raw


def select_skill_breakout(raw_state: np.ndarray) -> str:
    ball_dy = float(raw_state[3])
    return "RECEIVE" if ball_dy > _PRED_DY_EPS else "SETTLE"


@torch.no_grad()
def select_action_breakout(donor, state_vec: torch.Tensor, skill: str,
                           paddle_x84: float = 42.0,
                           fallback_action: int = ALE_NOOP) -> int:
    eligible = BREAKOUT_SKILL_CLASS_SETS[skill]
    logits = donor(state_vec, routing="soft")
    if isinstance(logits, tuple):
        logits = logits[0]
    union = list(donor.union_classes)
    masked = torch.full_like(logits, float("-inf"))
    for j, c in enumerate(union):
        if int(c) in eligible:
            masked[:, j] = logits[:, j]
    if torch.isinf(masked).all():
        return fallback_action
    pred_idx = int(masked[0].argmax())
    pred_class = int(union[pred_idx])
    action = BREAKOUT_SKILL_CLASS_TO_ALE[pred_class]
    if action == ALE_LEFT and paddle_x84 < PADDLE_LEFT_84_BO:
        return ALE_NOOP
    if action == ALE_RIGHT and paddle_x84 > PADDLE_RIGHT_84_BO:
        return ALE_NOOP
    return action
