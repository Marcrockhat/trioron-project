"""Synthetic primitive-env framework — vector-state version.

This module emits 8-d float32 state vectors matching the format
emitted by `experiments/atari_trioron/features.py` at inference time:

    state = (ball_x, ball_y, ball_dx, ball_dy,
             paddle_y, paddle_dy, ball_speed, pred_y)

Class IDs match docs/atari_pong_primitives.md:
  100-102  ball vertical position    (HIGH, MID, LOW)
  103-105  ball horizontal position  (LEFT, CENTER, RIGHT)
  110-112  paddle vertical position  (HIGH, MID, LOW)
  120-123  ball motion direction     (UP, DOWN, LEFT, RIGHT)
  130-131  ball speed                (FAST, SLOW)
  140-142  ball-paddle vertical rel  (ABOVE, ALIGNED, BELOW)
  150-151  ball-paddle approach      (APPROACHING, RECEDING)

This replaces an earlier pixel-input version that hit the random L0
projection's spatial-blindness ceiling — the design pivot is logged
in docs/atari_pong_primitives.md.

Trioron does not consume raw pixels in this design. A perception
module (e.g. RAM extraction) emits the state vector; trioron operates
on those vectors. See `device_conscience_pattern.md` and
`learn_to_use_not_from_principle.md` for the framing.

CLI:
  python3 -m experiments.atari_trioron.primitives.synthetic_env --probe
  python3 -m experiments.atari_trioron.primitives.synthetic_env --class-id 100
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------
# State-vector format
# ---------------------------------------------------------------------

STATE_DIM = 9
# (ball_x, ball_y, ball_dx, ball_dy, paddle_y, paddle_dy, ball_speed,
#  pred_y, opp_y)
# ball_speed = ‖(ball_dx, ball_dy)‖ and pred_y = predicted ball-y at the
# moment the ball would reach the agent paddle's x (with wall reflection)
# are pre-computed in the perception module so the substrate doesn't
# have to recover a norm or a division through a linear-plus-ReLU
# projection. Receding balls fall back to pred_y = ball_y so the
# feature stays in-distribution.
#
# opp_y added 2026-05-08 for the multi-skill curriculum: the SMASH
# skill needs opponent paddle position to choose deflection direction
# (aim ball away from where opp is). Sampled independently from
# my-paddle_y in synthetic data.
FRAME_HW = 84

# Right-edge agent paddle x (synthetic env convention; the perception
# module rescales ALE RAM coords into this same [0, 84] frame).
PADDLE_X = 84.0

# Paddle full height in frame-84 units (16 RAM units / 165 RAM-units-
# per-field × 84 frame-84-per-field ≈ 8.15). Used by the perception
# module to convert ALE Pong's paddle RAM byte (which stores the
# paddle's TOP edge) into a "paddle center" coordinate, which is what
# the synthetic env's paddle_y treats it as. Without this shift the
# teacher targets the paddle's top, not its catching center, and the
# ball misses by ~half a paddle height every rally.
PADDLE_HEIGHT_84 = 8.15

# Default L0 width for vector-state input. Random projection
# 8 → 16 preserves linear separability for all standardized
# linear-partition primitives.
DEFAULT_L0_DIM = 16

# Per-axis natural ranges, used by samplers. paddle_dy widened from
# (-3, 3) → (-8, 8) to match real ALE: under sustained UP/DOWN, the
# paddle reaches |dy| up to ~8 frame-84 per step (frame_skip=4).
RANGES: Dict[str, Tuple[float, float]] = {
    "ball_x":   (0.0, 84.0),
    "ball_y":   (0.0, 84.0),
    "ball_dx":  (-5.0, 5.0),
    "ball_dy":  (-5.0, 5.0),
    "paddle_y": (0.0, 84.0),
    "paddle_dy": (-8.0, 8.0),
    "opp_y":    (0.0, 84.0),
}

# Per-axis (mean, std) for standardization before L0. Position axes
# uniform over [0, 84] → mean 42, std ≈ 24.25. Velocity axes uniform
# over [-V, V] → mean 0, std ≈ V/sqrt(3). Speed is half-bounded
# (computed from velocities), use empirical std ≈ 2.5.
STANDARDIZE: Dict[str, Tuple[float, float]] = {
    "ball_x":    (42.0, 24.25),
    "ball_y":    (42.0, 24.25),
    "ball_dx":   (0.0,  2.89),
    "ball_dy":   (0.0,  2.89),
    "paddle_y":  (42.0, 24.25),
    "paddle_dy": (0.0,  4.62),
    "ball_speed":(2.50, 1.30),
    # pred_y shares ball_y's range/distribution: uniform on [0, 84]
    # under reflective folding from approaching states; ball_y itself
    # for receding states.
    "pred_y":    (42.0, 24.25),
    "opp_y":     (42.0, 24.25),
}

# Tier-1 thirds boundaries on the [0, 84] axis.
THIRD_LO = 84.0 / 3.0     # 28.0
THIRD_HI = 84.0 * 2.0 / 3.0  # 56.0


# ---------------------------------------------------------------------
# Class IDs (match docs/atari_pong_primitives.md)
# ---------------------------------------------------------------------

# Tier 1 — position
BALL_HIGH = 100
BALL_MID = 101
BALL_LOW = 102
BALL_LEFT = 103
BALL_CENTER = 104
BALL_RIGHT = 105
PADDLE_HIGH = 110
PADDLE_MID = 111
PADDLE_LOW = 112

# Tier 2 — motion direction
BALL_GOING_UP = 120
BALL_GOING_DOWN = 121
BALL_GOING_LEFT = 122
BALL_GOING_RIGHT = 123

# Tier 3 — speed
BALL_FAST = 130
BALL_SLOW = 131

# Tier 4 — relational
BALL_ABOVE_PADDLE = 140
BALL_ALIGNED_WITH_PADDLE = 141
BALL_BELOW_PADDLE = 142
BALL_APPROACHING_PADDLE = 150
BALL_RECEDING_FROM_PADDLE = 151

# Pong action classes (target-task, not part of the primitive vocabulary).
PONG_ACTION_UP = 200    # ALE action 2 (paddle up)
PONG_ACTION_DOWN = 201  # ALE action 3 (paddle down)
PONG_ACTION_HOLD = 202  # ALE action 0 (NOOP)
PONG_ACTION_DEAD_ZONE = 4.0  # |pred_y - effective_paddle_y| ≤ this → HOLD

# Multi-skill curriculum (2026-05-08). Three action-producing skills,
# each gated by ball state at inference. Per-skill class IDs, trained
# as separate primitive donors with shared L0 seed.
#
#   CATCH  — bdx > 0 and bx84 < SMASH_TRIGGER_X (track pred_y)
#   SMASH  — bdx > 0 and bx84 ≥ SMASH_TRIGGER_X (commit deflection)
#   PREPOS — bdx ≤ 0 (drift to mid-screen anchor)
SMASH_TRIGGER_X = 70.0
PREPOS_ANCHOR_Y = 42.0

SKILL_CATCH_UP = 210
SKILL_CATCH_DOWN = 211
SKILL_CATCH_HOLD = 212
SKILL_SMASH_UP = 220
SKILL_SMASH_DOWN = 221
SKILL_PREPOS_UP = 230
SKILL_PREPOS_DOWN = 231
SKILL_PREPOS_HOLD = 232

# Body-dynamics lookahead. The agent paddle in ALE Pong has continuous
# velocity that persists across env.step boundaries (frame_skip=4 plus
# residual drift under HOLD). The teacher predicts where the paddle
# will be one step from now if the agent does nothing — that's the
# "effective" paddle position the action targets.
#     effective_paddle_y = paddle_y + paddle_dy * PADDLE_LOOKAHEAD
# Empirically the paddle drifts ~1 step's worth of velocity under HOLD
# before braking, so LOOKAHEAD=1.0 is a reasonable first cut.
PADDLE_LOOKAHEAD = 1.0

CLASS_NAMES: Dict[int, str] = {
    BALL_HIGH: "BALL_HIGH",
    BALL_MID: "BALL_MID",
    BALL_LOW: "BALL_LOW",
    BALL_LEFT: "BALL_LEFT",
    BALL_CENTER: "BALL_CENTER",
    BALL_RIGHT: "BALL_RIGHT",
    PADDLE_HIGH: "PADDLE_HIGH",
    PADDLE_MID: "PADDLE_MID",
    PADDLE_LOW: "PADDLE_LOW",
    BALL_GOING_UP: "BALL_GOING_UP",
    BALL_GOING_DOWN: "BALL_GOING_DOWN",
    BALL_GOING_LEFT: "BALL_GOING_LEFT",
    BALL_GOING_RIGHT: "BALL_GOING_RIGHT",
    BALL_FAST: "BALL_FAST",
    BALL_SLOW: "BALL_SLOW",
    BALL_ABOVE_PADDLE: "BALL_ABOVE_PADDLE",
    BALL_ALIGNED_WITH_PADDLE: "BALL_ALIGNED_WITH_PADDLE",
    BALL_BELOW_PADDLE: "BALL_BELOW_PADDLE",
    BALL_APPROACHING_PADDLE: "BALL_APPROACHING_PADDLE",
    BALL_RECEDING_FROM_PADDLE: "BALL_RECEDING_FROM_PADDLE",
    PONG_ACTION_UP: "PONG_ACTION_UP",
    PONG_ACTION_DOWN: "PONG_ACTION_DOWN",
    PONG_ACTION_HOLD: "PONG_ACTION_HOLD",
    SKILL_CATCH_UP:    "SKILL_CATCH_UP",
    SKILL_CATCH_DOWN:  "SKILL_CATCH_DOWN",
    SKILL_CATCH_HOLD:  "SKILL_CATCH_HOLD",
    SKILL_SMASH_UP:    "SKILL_SMASH_UP",
    SKILL_SMASH_DOWN:  "SKILL_SMASH_DOWN",
    SKILL_PREPOS_UP:   "SKILL_PREPOS_UP",
    SKILL_PREPOS_DOWN: "SKILL_PREPOS_DOWN",
    SKILL_PREPOS_HOLD: "SKILL_PREPOS_HOLD",
}

ALL_CLASSES: List[int] = sorted(CLASS_NAMES.keys())


# ---------------------------------------------------------------------
# Random-state sampling
# ---------------------------------------------------------------------

FIELDS = ["ball_x", "ball_y", "ball_dx", "ball_dy",
          "paddle_y", "paddle_dy", "ball_speed", "pred_y", "opp_y"]


# ---------------------------------------------------------------------
# Kinematic ball-y prediction (predict-where-the-ball-will-be teacher)
# ---------------------------------------------------------------------

def _fold_to_range(y: float, lo: float = 0.0, hi: float = FRAME_HW) -> float:
    """Reflect y back into [lo, hi] across both walls. Models the ball
    bouncing off top and bottom edges any number of times."""
    span = hi - lo
    rel = (y - lo) % (2.0 * span)
    if rel < 0.0:
        rel += 2.0 * span
    if rel > span:
        rel = 2.0 * span - rel
    return lo + rel


# Trigger threshold for kinematic prediction. Decoupled from
# _DIRECTION_EPS (which buffers BALL_GOING_X label boundaries): we want
# the predictor to fire for *any* positive ball_dx, including the slow
# approaching frames that show up in real ALE Pong after frame-skip.
_PRED_DX_EPS = 1e-3


def predict_ball_y_at_impact(
    ball_x: float, ball_y: float,
    ball_dx: float, ball_dy: float,
    paddle_x: float = PADDLE_X,
) -> float:
    """Predicted ball-y when ball_x reaches paddle_x, with wall
    reflection in [0, FRAME_HW]. Returns ball_y unchanged when the ball
    is receding, stationary in x, or already past the paddle — those
    states have no kinematic prediction, and falling back to ball_y
    keeps the feature in-distribution at inference."""
    if ball_dx <= _PRED_DX_EPS:
        return float(ball_y)
    dx_to_impact = paddle_x - ball_x
    if dx_to_impact <= 0.0:
        return float(ball_y)
    t = dx_to_impact / ball_dx
    return _fold_to_range(ball_y + ball_dy * t)


def standardize(state: np.ndarray) -> np.ndarray:
    """Apply per-axis (mean, std) standardization. Operates in-place
    semantically but returns a new array to avoid surprising callers.
    Shape: (..., STATE_DIM) → same shape."""
    out = state.astype(np.float32, copy=True)
    flat = out.reshape(-1, STATE_DIM)
    for i, k in enumerate(FIELDS):
        m, s = STANDARDIZE[k]
        flat[:, i] = (flat[:, i] - m) / s
    return flat.reshape(state.shape)


def _sample_raw(rng: np.random.Generator,
                overrides: Optional[Dict[str, Tuple[float, float]]] = None
                ) -> np.ndarray:
    """Sample one STATE_DIM state vector in RAW (un-standardized) units.
    Samplers operate in raw units for human-readable rejection
    thresholds; `generate_dataset` standardizes before returning."""
    raw = np.zeros(STATE_DIM, dtype=np.float32)
    for i, k in enumerate(FIELDS):
        if k in ("ball_speed", "pred_y"):
            continue
        lo, hi = (overrides or {}).get(k, RANGES[k])
        raw[i] = float(rng.uniform(lo, hi))
    raw[6] = float(np.sqrt(raw[2] ** 2 + raw[3] ** 2))
    raw[7] = predict_ball_y_at_impact(
        ball_x=raw[0], ball_y=raw[1],
        ball_dx=raw[2], ball_dy=raw[3],
    )
    return raw


# ---------------------------------------------------------------------
# Per-class samplers
# ---------------------------------------------------------------------

def _sample_ball_y_third(third: str, rng: np.random.Generator) -> np.ndarray:
    if third == "high":
        return _sample_raw(rng, {"ball_y": (0.0, THIRD_LO)})
    if third == "mid":
        return _sample_raw(rng, {"ball_y": (THIRD_LO, THIRD_HI)})
    if third == "low":
        return _sample_raw(rng, {"ball_y": (THIRD_HI, 84.0)})
    raise ValueError(third)


def _sample_ball_x_third(third: str, rng: np.random.Generator) -> np.ndarray:
    if third == "left":
        return _sample_raw(rng, {"ball_x": (0.0, THIRD_LO)})
    if third == "center":
        return _sample_raw(rng, {"ball_x": (THIRD_LO, THIRD_HI)})
    if third == "right":
        return _sample_raw(rng, {"ball_x": (THIRD_HI, 84.0)})
    raise ValueError(third)


def _sample_paddle_y_third(third: str, rng: np.random.Generator) -> np.ndarray:
    if third == "high":
        return _sample_raw(rng, {"paddle_y": (0.0, THIRD_LO)})
    if third == "mid":
        return _sample_raw(rng, {"paddle_y": (THIRD_LO, THIRD_HI)})
    if third == "low":
        return _sample_raw(rng, {"paddle_y": (THIRD_HI, 84.0)})
    raise ValueError(third)


_DIRECTION_EPS = 0.5


def _sample_motion_direction(direction: str,
                             rng: np.random.Generator) -> np.ndarray:
    if direction == "up":
        return _sample_raw(rng, {"ball_dy": (-5.0, -_DIRECTION_EPS)})
    if direction == "down":
        return _sample_raw(rng, {"ball_dy": (_DIRECTION_EPS, 5.0)})
    if direction == "left":
        return _sample_raw(rng, {"ball_dx": (-5.0, -_DIRECTION_EPS)})
    if direction == "right":
        return _sample_raw(rng, {"ball_dx": (_DIRECTION_EPS, 5.0)})
    raise ValueError(direction)


_FAST_THRESH = 3.0
_STATIONARY_THRESH = 0.5


def _sample_ball_fast(rng: np.random.Generator) -> np.ndarray:
    """Sample a state with ball-velocity norm ≥ FAST_THRESH."""
    while True:
        s = _sample_raw(rng)
        if (s[2] ** 2 + s[3] ** 2) ** 0.5 >= _FAST_THRESH:
            return s


def _sample_ball_slow(rng: np.random.Generator) -> np.ndarray:
    """Sample a state with STATIONARY_THRESH ≤ norm < FAST_THRESH."""
    while True:
        s = _sample_raw(rng)
        n = (s[2] ** 2 + s[3] ** 2) ** 0.5
        if _STATIONARY_THRESH <= n < _FAST_THRESH:
            return s


def _sample_ball_above_paddle(rng: np.random.Generator) -> np.ndarray:
    """ball_y < paddle_y - 4."""
    while True:
        s = _sample_raw(rng)
        if s[1] < s[4] - 4.0:
            return s


def _sample_ball_aligned(rng: np.random.Generator) -> np.ndarray:
    """|ball_y - paddle_y| ≤ 4."""
    while True:
        s = _sample_raw(rng)
        if abs(s[1] - s[4]) <= 4.0:
            return s


def _sample_ball_below_paddle(rng: np.random.Generator) -> np.ndarray:
    """ball_y > paddle_y + 4."""
    while True:
        s = _sample_raw(rng)
        if s[1] > s[4] + 4.0:
            return s


def _sample_ball_approaching_paddle(rng: np.random.Generator) -> np.ndarray:
    """Pong: agent paddle on right edge, so ball approaches if dx > 0."""
    return _sample_raw(rng, {"ball_dx": (_DIRECTION_EPS, 5.0)})


def _sample_ball_receding_from_paddle(rng: np.random.Generator) -> np.ndarray:
    return _sample_raw(rng, {"ball_dx": (-5.0, -_DIRECTION_EPS)})


# ---------------------------------------------------------------------
# Pong-action samplers (target task) — predict-y + dynamics-aware rule
#
# Geometry alone (predict-y vs paddle_y) loses on real ALE because the
# paddle has momentum: a HOLD command after UP still lets the paddle
# drift up for one more step. The teacher anticipates this drift by
# comparing pred_y against the paddle's projected position one step
# from now under HOLD:
#
#     eff_py = paddle_y + paddle_dy * PADDLE_LOOKAHEAD
#
#     receding (ball_dx ≤ 0)             → HOLD  (no kinematic info)
#     pred_y < eff_py - DEAD_ZONE        → UP    (ball will land above)
#     pred_y > eff_py + DEAD_ZONE        → DOWN  (ball will land below)
#     else                               → HOLD  (paddle's drift is on target)
#
# Each class is sampled by reject-sampling so the dataset stays
# balanced across UP/DOWN/HOLD.
# ---------------------------------------------------------------------


def _effective_paddle_y(s: np.ndarray) -> float:
    return float(s[4] + s[5] * PADDLE_LOOKAHEAD)


def _sample_pong_action_up(rng: np.random.Generator) -> np.ndarray:
    """Approaching ball with pred_y < eff_paddle_y - DEAD_ZONE."""
    while True:
        s = _sample_raw(rng)
        if s[2] <= _PRED_DX_EPS:
            continue
        if s[7] < _effective_paddle_y(s) - PONG_ACTION_DEAD_ZONE:
            return s


def _sample_pong_action_down(rng: np.random.Generator) -> np.ndarray:
    """Approaching ball with pred_y > eff_paddle_y + DEAD_ZONE."""
    while True:
        s = _sample_raw(rng)
        if s[2] <= _PRED_DX_EPS:
            continue
        if s[7] > _effective_paddle_y(s) + PONG_ACTION_DEAD_ZONE:
            return s


def _sample_pong_action_hold(rng: np.random.Generator) -> np.ndarray:
    """Receding ball OR approaching ball with |pred_y - eff_paddle_y| ≤ DZ."""
    while True:
        s = _sample_raw(rng)
        if s[2] <= _PRED_DX_EPS:
            return s
        if abs(s[7] - _effective_paddle_y(s)) <= PONG_ACTION_DEAD_ZONE:
            return s


# ---------------------------------------------------------------------
# Multi-skill curriculum samplers (2026-05-08)
#
# Each skill covers a disjoint region of state space. Combined with a
# hand-coded gate at inference (see pong_inference.py), the three
# skills tile the full state space without overlap:
#
#   if bdx ≤ 0:                           PREPOS skill
#   elif bdx > 0 and bx84 < TRIGGER:      CATCH skill
#   else (bdx > 0 and bx84 ≥ TRIGGER):    SMASH skill
#
# Per-skill samplers use range overrides on _sample_raw to stay in
# their region, then enforce the action-class condition by reject
# sampling.
# ---------------------------------------------------------------------


def _sample_skill_catch_up(rng: np.random.Generator) -> np.ndarray:
    """Approaching ball, far from paddle, pred_y < eff_paddle_y - DZ."""
    while True:
        s = _sample_raw(rng, {
            "ball_x": (0.0, SMASH_TRIGGER_X),
            "ball_dx": (_PRED_DX_EPS, 5.0),
        })
        if s[7] < _effective_paddle_y(s) - PONG_ACTION_DEAD_ZONE:
            return s


def _sample_skill_catch_down(rng: np.random.Generator) -> np.ndarray:
    while True:
        s = _sample_raw(rng, {
            "ball_x": (0.0, SMASH_TRIGGER_X),
            "ball_dx": (_PRED_DX_EPS, 5.0),
        })
        if s[7] > _effective_paddle_y(s) + PONG_ACTION_DEAD_ZONE:
            return s


def _sample_skill_catch_hold(rng: np.random.Generator) -> np.ndarray:
    while True:
        s = _sample_raw(rng, {
            "ball_x": (0.0, SMASH_TRIGGER_X),
            "ball_dx": (_PRED_DX_EPS, 5.0),
        })
        if abs(s[7] - _effective_paddle_y(s)) <= PONG_ACTION_DEAD_ZONE:
            return s


def _sample_skill_smash_up(rng: np.random.Generator) -> np.ndarray:
    """Ball close to paddle and approaching, opp ABOVE us → smash UP
    (paddle moves up to put ball into paddle's bottom edge → reflects
    DOWN... wait, see below).

    Direction convention (matches pong_oracle):
      opp_y > my_y (opp BELOW us)  → action DOWN  (deflect ball UP toward empty top)
      opp_y < my_y (opp ABOVE us)  → action UP    (deflect ball DOWN toward empty bottom)
    """
    while True:
        s = _sample_raw(rng, {
            "ball_x": (SMASH_TRIGGER_X, 84.0),
            "ball_dx": (_PRED_DX_EPS, 5.0),
        })
        if s[8] < s[4]:  # opp_y < my_y
            return s


def _sample_skill_smash_down(rng: np.random.Generator) -> np.ndarray:
    while True:
        s = _sample_raw(rng, {
            "ball_x": (SMASH_TRIGGER_X, 84.0),
            "ball_dx": (_PRED_DX_EPS, 5.0),
        })
        if s[8] > s[4]:  # opp_y > my_y
            return s


def _sample_skill_prepos_up(rng: np.random.Generator) -> np.ndarray:
    """Receding ball, paddle below anchor → move UP (decrease y)."""
    while True:
        s = _sample_raw(rng, {"ball_dx": (-5.0, -_PRED_DX_EPS)})
        if s[4] > PREPOS_ANCHOR_Y + PONG_ACTION_DEAD_ZONE:
            return s


def _sample_skill_prepos_down(rng: np.random.Generator) -> np.ndarray:
    while True:
        s = _sample_raw(rng, {"ball_dx": (-5.0, -_PRED_DX_EPS)})
        if s[4] < PREPOS_ANCHOR_Y - PONG_ACTION_DEAD_ZONE:
            return s


def _sample_skill_prepos_hold(rng: np.random.Generator) -> np.ndarray:
    while True:
        s = _sample_raw(rng, {"ball_dx": (-5.0, -_PRED_DX_EPS)})
        if abs(s[4] - PREPOS_ANCHOR_Y) <= PONG_ACTION_DEAD_ZONE:
            return s


# ---------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------

GENERATORS: Dict[int, Callable[[np.random.Generator], np.ndarray]] = {
    BALL_HIGH:   lambda rng: _sample_ball_y_third("high", rng),
    BALL_MID:    lambda rng: _sample_ball_y_third("mid", rng),
    BALL_LOW:    lambda rng: _sample_ball_y_third("low", rng),
    BALL_LEFT:   lambda rng: _sample_ball_x_third("left", rng),
    BALL_CENTER: lambda rng: _sample_ball_x_third("center", rng),
    BALL_RIGHT:  lambda rng: _sample_ball_x_third("right", rng),
    PADDLE_HIGH: lambda rng: _sample_paddle_y_third("high", rng),
    PADDLE_MID:  lambda rng: _sample_paddle_y_third("mid", rng),
    PADDLE_LOW:  lambda rng: _sample_paddle_y_third("low", rng),
    BALL_GOING_UP:    lambda rng: _sample_motion_direction("up", rng),
    BALL_GOING_DOWN:  lambda rng: _sample_motion_direction("down", rng),
    BALL_GOING_LEFT:  lambda rng: _sample_motion_direction("left", rng),
    BALL_GOING_RIGHT: lambda rng: _sample_motion_direction("right", rng),
    BALL_FAST: _sample_ball_fast,
    BALL_SLOW: _sample_ball_slow,
    BALL_ABOVE_PADDLE: _sample_ball_above_paddle,
    BALL_ALIGNED_WITH_PADDLE: _sample_ball_aligned,
    BALL_BELOW_PADDLE: _sample_ball_below_paddle,
    BALL_APPROACHING_PADDLE: _sample_ball_approaching_paddle,
    BALL_RECEDING_FROM_PADDLE: _sample_ball_receding_from_paddle,
    PONG_ACTION_UP:   _sample_pong_action_up,
    PONG_ACTION_DOWN: _sample_pong_action_down,
    PONG_ACTION_HOLD: _sample_pong_action_hold,
    SKILL_CATCH_UP:    _sample_skill_catch_up,
    SKILL_CATCH_DOWN:  _sample_skill_catch_down,
    SKILL_CATCH_HOLD:  _sample_skill_catch_hold,
    SKILL_SMASH_UP:    _sample_skill_smash_up,
    SKILL_SMASH_DOWN:  _sample_skill_smash_down,
    SKILL_PREPOS_UP:   _sample_skill_prepos_up,
    SKILL_PREPOS_DOWN: _sample_skill_prepos_down,
    SKILL_PREPOS_HOLD: _sample_skill_prepos_hold,
}


def generate_dataset(
    class_id: int,
    n_samples: int,
    seed: int = 42,
    standardize_output: bool = True,
) -> torch.Tensor:
    """Returns (n_samples, STATE_DIM) float32 for the given class.
    Output is standardized (per-axis zero-mean, unit-variance) by
    default, matching what the perception module emits at inference."""
    if class_id not in GENERATORS:
        raise ValueError(
            f"unknown class_id {class_id}; valid: {ALL_CLASSES}"
        )
    rng = np.random.default_rng(seed)
    gen = GENERATORS[class_id]
    raw = np.stack([gen(rng) for _ in range(n_samples)], axis=0)
    if standardize_output:
        raw = standardize(raw)
    return torch.from_numpy(raw)


# ---------------------------------------------------------------------
# Clustering verification probe
# ---------------------------------------------------------------------

def clustering_probe(
    class_id_pos: int,
    class_id_neg: int,
    *,
    n_per_class: int = 200,
    l0_seed: int = 42,
    l0_dim: int = DEFAULT_L0_DIM,
    seed: int = 0,
) -> Dict[str, float]:
    """Project samples through a fresh frozen L0 layer and report
    L0-space clustering metrics + linear-probe accuracy on a held-out
    split.

    Decision threshold (per docs/atari_pong_primitives.md §6):
        probe_acc ≥ 0.95 → OK
        0.85 ≤ probe_acc < 0.95 → YELLOW
        probe_acc < 0.85 → RED
    """
    from trioron.node import TrioronLayer

    X_pos = generate_dataset(class_id_pos, n_per_class, seed=seed)
    X_neg = generate_dataset(class_id_neg, n_per_class, seed=seed + 1_000_003)

    torch.manual_seed(l0_seed)
    l0 = TrioronLayer(fan_in=STATE_DIM, n_nodes=l0_dim, activation="relu")
    for p in l0.parameters():
        p.requires_grad_(False)
    l0.eval()
    with torch.no_grad():
        Z_pos = l0(X_pos)
        Z_neg = l0(X_neg)

    def _mean_pairwise(Z: torch.Tensor) -> float:
        n = Z.shape[0]
        if n < 2:
            return 0.0
        d = torch.cdist(Z, Z)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        return float(d[mask].mean())

    d_intra_pos = _mean_pairwise(Z_pos)
    d_intra_neg = _mean_pairwise(Z_neg)
    d_inter = float(torch.cdist(Z_pos, Z_neg).mean())
    max_intra = max(d_intra_pos, d_intra_neg, 1e-9)
    ratio = d_inter / max_intra

    Z = torch.cat([Z_pos, Z_neg], dim=0)
    y = torch.cat([torch.zeros(n_per_class, dtype=torch.long),
                   torch.ones(n_per_class,  dtype=torch.long)], dim=0)
    perm = torch.randperm(Z.shape[0],
                          generator=torch.Generator().manual_seed(seed))
    Z = Z[perm]
    y = y[perm]
    n_tr = int(0.8 * Z.shape[0])
    Z_tr, Z_te = Z[:n_tr], Z[n_tr:]
    y_tr, y_te = y[:n_tr], y[n_tr:]
    probe = torch.nn.Linear(l0_dim, 2)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        logits = probe(Z_tr)
        loss = torch.nn.functional.cross_entropy(logits, y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = probe(Z_te).argmax(dim=-1)
        probe_acc = float((pred == y_te).float().mean())

    return {
        "d_intra_pos": d_intra_pos,
        "d_intra_neg": d_intra_neg,
        "d_inter": d_inter,
        "ratio": ratio,
        "probe_acc": probe_acc,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

_PROBE_PAIRS: List[Tuple[int, int]] = [
    (BALL_HIGH, BALL_LOW),
    (BALL_HIGH, BALL_MID),
    (BALL_LEFT, BALL_RIGHT),
    (BALL_LEFT, BALL_CENTER),
    (PADDLE_HIGH, PADDLE_LOW),
    (PADDLE_HIGH, PADDLE_MID),
    (BALL_GOING_UP, BALL_GOING_DOWN),
    (BALL_GOING_LEFT, BALL_GOING_RIGHT),
    (BALL_FAST, BALL_SLOW),
    (BALL_ABOVE_PADDLE, BALL_BELOW_PADDLE),
    (BALL_ABOVE_PADDLE, BALL_ALIGNED_WITH_PADDLE),
    (BALL_APPROACHING_PADDLE, BALL_RECEDING_FROM_PADDLE),
]


def _verdict(probe_acc: float) -> str:
    if probe_acc >= 0.95:
        return "OK    "
    if probe_acc >= 0.85:
        return "YELLOW"
    return "RED   "


def _main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true",
                    help="Run clustering probe across all primitive pairs")
    ap.add_argument("--class-id", type=int,
                    help="Generate one batch of this class for sanity check")
    ap.add_argument("--n", type=int, default=200,
                    help="Samples per class for probe / generation")
    ap.add_argument("--l0-dim", type=int, default=DEFAULT_L0_DIM,
                    help="L0 projection width")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.probe:
        print(f"Clustering probe — n={args.n} per class, l0_seed=42, "
              f"l0_dim={args.l0_dim}, STATE_DIM={STATE_DIM}")
        print(f"  {'pos':>26s} vs {'neg':>26s}  "
              f"{'ratio':>5s}  {'probe':>5s}  verdict")
        print("  " + "-" * 80)
        for pos, neg in _PROBE_PAIRS:
            r = clustering_probe(pos, neg, n_per_class=args.n,
                                 l0_dim=args.l0_dim, seed=args.seed)
            verdict = _verdict(r["probe_acc"])
            print(f"  {CLASS_NAMES[pos]:>26s} vs {CLASS_NAMES[neg]:>26s}  "
                  f"{r['ratio']:5.2f}  {r['probe_acc']:5.2f}  [{verdict}]")
        return

    if args.class_id is not None:
        X = generate_dataset(args.class_id, args.n, seed=args.seed)
        print(f"class {args.class_id} ({CLASS_NAMES.get(args.class_id, '?')}): "
              f"shape={tuple(X.shape)} dtype={X.dtype}")
        print(f"  per-axis mean: {X.mean(0).tolist()}")
        print(f"  per-axis  min: {X.min(0).values.tolist()}")
        print(f"  per-axis  max: {X.max(0).values.tolist()}")
        return

    ap.print_help()


if __name__ == "__main__":
    _main()
