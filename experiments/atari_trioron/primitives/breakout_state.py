"""Breakout perception module — RAM-to-state-vector for the
multi-skill organism.

Mirrors `pong_skill_inference.PongSkillStateBuilder` but for Breakout's
horizontal-paddle geometry. Where Pong's state vector is 9-d (paddle
moves vertically against an opponent), Breakout collapses to 8-d:
paddle moves horizontally only, and there is no opponent paddle. The
analogous prediction task swaps ball-y-at-paddle-x for ball-x-at-
paddle-y.

State vector (raw, before standardization):
    (ball_x, ball_y, ball_dx, ball_dy,
     paddle_x, paddle_dx, ball_speed, pred_x)

Coordinate convention: 84-coords, [0, 84] linear rescaling from RAM.
PADDLE_Y_84 is the constant y-coord of the paddle row (paddle never
moves vertically). All ball/paddle spatial axes share the same scale
so distances and predictions are interpretable without per-axis hacks.

Class IDs for the synthetic curriculum (action-class space, parallel
to Pong's SKILL_*) are defined here; per-skill samplers live in this
module too rather than in a shared synthetic_env, because Breakout's
state shape is different and mixing the two would invite silent shape
errors.

The probe_ram() helper at module bottom dumps observed RAM ranges
during a heuristic-driven warm-up — use it to calibrate the constants
below if oracle performance is poor.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from experiments.atari_trioron.features import extract_breakout, get_ale
from experiments.atari_trioron.primitives.pong_inference import (
    _ram_to_84, ALE_FRAME_SKIP,
)


# ---------------------------------------------------------------------
# RAM range calibration (empirical; refine with --probe-ram)
# ---------------------------------------------------------------------
# Initial guesses from AtariARI annotation tables; the probe routine
# below dumps observed mins/maxes during a heuristic warm-up so these
# can be re-tuned if the oracle underperforms.
RAM_PADDLE_X_RANGE = (57.0, 201.0)
RAM_BALL_X_RANGE   = (49.0, 200.0)
RAM_BALL_Y_RANGE   = (80.0, 200.0)

# Paddle row in 84-coords. Paddle is a fixed-y horizontal bar near the
# bottom of the playfield; in RAM ball_y terms the paddle row sits at
# ~189, which under the (80, 200) mapping above lands at:
#     (189 - 80) * 84 / (200 - 80) = 76.3
# Recalibrate jointly with RAM_BALL_Y_RANGE if probe observations
# diverge.
PADDLE_Y_84 = 76.0

# Frame-stack convention shared with Pong primitives.
FRAME_HW = 84
# 9-d state, matching Pong's STATE_DIM=9 for absorb/extend compatibility.
# Field 8 is `opp_y` — for Pong it's the opponent paddle's y; for
# Breakout there is no opponent, so we pad it with a constant
# (`OPP_Y_PAD_RAW`, ~mid-screen) which standardizes to ~0 and adds
# zero variance through the L0 layer. The L0 random projection does
# not interpret semantics; identical input shapes are sufficient for
# cross-game absorb / api.extend.
STATE_DIM = 9
FIELDS = [
    "ball_x", "ball_y", "ball_dx", "ball_dy",
    "paddle_x", "paddle_dx", "ball_speed", "pred_x",
    "opp_y",
]
OPP_Y_PAD_RAW = 42.0  # 84-coord midpoint; standardizes to 0.0

# Per-axis natural ranges for the synthetic samplers.
RANGES: Dict[str, Tuple[float, float]] = {
    "ball_x":    (0.0, 84.0),
    "ball_y":    (0.0, 84.0),
    "ball_dx":   (-5.0, 5.0),
    "ball_dy":   (-5.0, 5.0),
    "paddle_x":  (0.0, 84.0),
    "paddle_dx": (-8.0, 8.0),
    # opp_y not sampled — pinned to OPP_Y_PAD_RAW for Breakout
}

# Per-axis (mean, std) for standardization. Same conventions as
# synthetic_env.STANDARDIZE — zero-mean unit-variance under the
# uniform-sampling distribution.
STANDARDIZE: Dict[str, Tuple[float, float]] = {
    "ball_x":     (42.0, 24.25),
    "ball_y":     (42.0, 24.25),
    "ball_dx":    (0.0,  2.89),
    "ball_dy":    (0.0,  2.89),
    "paddle_x":   (42.0, 24.25),
    "paddle_dx":  (0.0,  4.62),
    "ball_speed": (2.50, 1.30),
    "pred_x":     (42.0, 24.25),
    # opp_y matches Pong's standardization (mean=42, std=24.25); for
    # Breakout the raw value is constant 42 → standardized 0 every
    # frame, contributing zero variance to L0 output.
    "opp_y":      (42.0, 24.25),
}


# ---------------------------------------------------------------------
# Class IDs — Breakout skill curriculum
# ---------------------------------------------------------------------
# IDs are disjoint from Pong's (200..232) so a single absorbed
# organism could in principle host both games' skill donors without
# class-id collision. We start the Breakout block at 300.
#
# Three skills (LAUNCH is handled out-of-band by the gate, since it
# fires on serve frames where there is no in-play state vector):
#
#   RECEIVE — ball_dy > 0  (ball approaching the paddle row from above)
#             classes: RECEIVE_LEFT / RECEIVE_RIGHT / RECEIVE_HOLD
#   SETTLE  — ball_dy ≤ 0  (ball receding upward toward bricks)
#             classes: SETTLE_LEFT / SETTLE_RIGHT / SETTLE_HOLD
#
# LAUNCH is a degenerate single-action "skill" — it always emits FIRE.
# We give it a class ID for completeness but never train a donor on
# it; the gate just emits ALE_FIRE directly when ball_in_play is False.

SKILL_RECEIVE_LEFT  = 300
SKILL_RECEIVE_RIGHT = 301
SKILL_RECEIVE_HOLD  = 302
SKILL_SETTLE_LEFT   = 310
SKILL_SETTLE_RIGHT  = 311
SKILL_SETTLE_HOLD   = 312
SKILL_LAUNCH_FIRE   = 320  # symbolic; never a donor target

CLASS_NAMES_BREAKOUT: Dict[int, str] = {
    SKILL_RECEIVE_LEFT:  "BO_RECEIVE_LEFT",
    SKILL_RECEIVE_RIGHT: "BO_RECEIVE_RIGHT",
    SKILL_RECEIVE_HOLD:  "BO_RECEIVE_HOLD",
    SKILL_SETTLE_LEFT:   "BO_SETTLE_LEFT",
    SKILL_SETTLE_RIGHT:  "BO_SETTLE_RIGHT",
    SKILL_SETTLE_HOLD:   "BO_SETTLE_HOLD",
    SKILL_LAUNCH_FIRE:   "BO_LAUNCH_FIRE",
}

# Target paddle anchor x for the SETTLE skill — drift to mid-playfield
# while ball is receding so worst-case distance to the next predicted
# bounce is minimized.
SETTLE_ANCHOR_X = 42.0

# Action dead-zone in 84-coords: |target_x - paddle_x| ≤ DZ → HOLD.
# Without it the paddle jitters around the target pixel. Tuned 2026-
# 05-08 against Breakout: 3.0 was too generous (paddle held when ball
# was already past its edge), 1.0 keeps ~80% of catches at the cost
# of mild end-of-episode jitter.
ACTION_DEAD_ZONE = 1.0

# Paddle lookahead for momentum-aware action selection. In Pong this
# was 1.0 — paddle drifts ~1 step's worth of velocity under HOLD
# before braking. In Breakout the paddle decelerates faster under
# NOOP (less drift) and the inertia correction overshoots; 0.0
# (track raw position) catches noticeably more balls in the diagnostic.
PADDLE_LOOKAHEAD = 0.0


# ---------------------------------------------------------------------
# Kinematic ball-x prediction with L/R wall folding
# ---------------------------------------------------------------------

_PRED_DY_EPS = 1e-3


def _fold_to_range(x: float, lo: float = 0.0, hi: float = FRAME_HW) -> float:
    """Reflect x back into [lo, hi] across both walls — models the
    ball bouncing off left/right edges any number of times before
    reaching the paddle row."""
    span = hi - lo
    rel = (x - lo) % (2.0 * span)
    if rel < 0.0:
        rel += 2.0 * span
    if rel > span:
        rel = 2.0 * span - rel
    return lo + rel


def predict_ball_x_at_impact(
    ball_x: float, ball_y: float,
    ball_dx: float, ball_dy: float,
    paddle_y: float = PADDLE_Y_84,
) -> float:
    """Predicted ball-x when ball_y reaches paddle_y, with wall
    reflection in [0, FRAME_HW]. Returns ball_x unchanged when the
    ball is rising (receding from paddle), already past the paddle
    row, or stationary in y."""
    if ball_dy <= _PRED_DY_EPS:
        return float(ball_x)
    dy_to_impact = paddle_y - ball_y
    if dy_to_impact <= 0.0:
        return float(ball_x)
    t = dy_to_impact / ball_dy
    return _fold_to_range(ball_x + ball_dx * t)


def standardize(state: np.ndarray) -> np.ndarray:
    """Per-axis zero-mean unit-variance standardization. Shape
    (..., STATE_DIM) → same shape."""
    out = state.astype(np.float32, copy=True)
    flat = out.reshape(-1, STATE_DIM)
    for i, k in enumerate(FIELDS):
        m, s = STANDARDIZE[k]
        flat[:, i] = (flat[:, i] - m) / s
    return flat.reshape(state.shape)


# ---------------------------------------------------------------------
# Stateful state builder for real-ALE inference
# ---------------------------------------------------------------------


class BreakoutStateBuilder:
    """Reads ALE Breakout RAM each step, builds the standardized 8-d
    vector by differencing against the previous frame's stored state.

    Returns (None, None) on serve frames (ball not in play) and on the
    first valid frame after a serve (no velocity yet — same convention
    as Pong's builder, fixes the "first-frame routes to wrong skill"
    bug). Callers should treat None as "let the LAUNCH gate decide"
    (FIRE on serve, HOLD on first-valid).
    """

    def __init__(self) -> None:
        self.prev_ball_x: float = 0.0
        self.prev_ball_y: float = 0.0
        self.prev_paddle_x: float = 0.0
        self.has_prev: bool = False

    def reset(self) -> None:
        self.has_prev = False

    def step(self, ale) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        s = extract_breakout(ale)
        if not s.ball_in_play:
            self.has_prev = False
            return None, None
        ball_x84 = _ram_to_84(s.ball_x, *RAM_BALL_X_RANGE)
        ball_y84 = _ram_to_84(s.ball_y, *RAM_BALL_Y_RANGE)
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
        pred_x = predict_ball_x_at_impact(
            ball_x=ball_x84, ball_y=ball_y84,
            ball_dx=ball_dx, ball_dy=ball_dy,
        )
        raw = np.array([
            ball_x84, ball_y84,
            ball_dx, ball_dy,
            paddle_x84, paddle_dx,
            ball_speed, pred_x,
            OPP_Y_PAD_RAW,
        ], dtype=np.float32)
        self.prev_ball_x = ball_x84
        self.prev_ball_y = ball_y84
        self.prev_paddle_x = paddle_x84
        self.has_prev = True
        std_vec = standardize(raw[None, :])
        return torch.from_numpy(std_vec), raw


# ---------------------------------------------------------------------
# Skill gate (hand-coded, parallel to Pong's select_skill)
# ---------------------------------------------------------------------


def select_skill_breakout(raw_state: np.ndarray) -> str:
    """Three skills tile the in-play state space disjointly:
        ball_dy > 0  → RECEIVE  (approaching)
        else         → SETTLE   (receding or stationary)
    LAUNCH is gated on ball_in_play=False at the inference loop level,
    not here — by the time this function is called, raw_state already
    represents an in-play frame.
    """
    ball_dy = float(raw_state[3])
    if ball_dy > _PRED_DY_EPS:
        return "RECEIVE"
    return "SETTLE"


# ---------------------------------------------------------------------
# RAM range probe — instruments a short heuristic episode and prints
# observed RAM mins/maxes so the constants above can be tuned.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Synthetic samplers for the donor curriculum
# ---------------------------------------------------------------------
# Each skill class is sampled by reject-sampling under the right
# action condition. Two skills (RECEIVE / SETTLE) tile the in-play
# state space disjointly via the sign of ball_dy. LAUNCH is gated
# out-of-band on ball_in_play and is a single-action degenerate
# "skill" — no donor needed.


def _sample_raw_breakout(
    rng: np.random.Generator,
    overrides: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """Sample one 9-d Breakout state in RAW (un-standardized) units.
    ball_speed (idx 6) and pred_x (idx 7) are derived from sampled
    velocities/positions; opp_y (idx 8) is pinned to OPP_Y_PAD_RAW
    since Breakout has no opponent paddle."""
    raw = np.zeros(STATE_DIM, dtype=np.float32)
    for i, k in enumerate(FIELDS):
        if k in ("ball_speed", "pred_x", "opp_y"):
            continue
        lo, hi = (overrides or {}).get(k, RANGES[k])
        raw[i] = float(rng.uniform(lo, hi))
    raw[6] = float(np.sqrt(raw[2] ** 2 + raw[3] ** 2))
    raw[7] = predict_ball_x_at_impact(
        ball_x=raw[0], ball_y=raw[1],
        ball_dx=raw[2], ball_dy=raw[3],
    )
    raw[8] = OPP_Y_PAD_RAW
    return raw


def _eff_paddle_x(s: np.ndarray) -> float:
    """Paddle's projected x one step out under HOLD — same role as
    `_effective_paddle_y` in synthetic_env. Reduces to s[4] when
    PADDLE_LOOKAHEAD=0 (Breakout default)."""
    return float(s[4] + s[5] * PADDLE_LOOKAHEAD)


def _sample_skill_receive_left(rng: np.random.Generator) -> np.ndarray:
    """ball_dy > 0 (approaching) AND pred_x < eff_paddle_x - DZ."""
    while True:
        s = _sample_raw_breakout(rng, {"ball_dy": (_PRED_DY_EPS, 5.0)})
        if s[7] < _eff_paddle_x(s) - ACTION_DEAD_ZONE:
            return s


def _sample_skill_receive_right(rng: np.random.Generator) -> np.ndarray:
    while True:
        s = _sample_raw_breakout(rng, {"ball_dy": (_PRED_DY_EPS, 5.0)})
        if s[7] > _eff_paddle_x(s) + ACTION_DEAD_ZONE:
            return s


def _sample_skill_receive_hold(rng: np.random.Generator) -> np.ndarray:
    while True:
        s = _sample_raw_breakout(rng, {"ball_dy": (_PRED_DY_EPS, 5.0)})
        if abs(s[7] - _eff_paddle_x(s)) <= ACTION_DEAD_ZONE:
            return s


def _sample_skill_settle_left(rng: np.random.Generator) -> np.ndarray:
    """ball_dy ≤ 0 (receding) AND paddle_x > anchor + DZ."""
    while True:
        s = _sample_raw_breakout(rng, {"ball_dy": (-5.0, -_PRED_DY_EPS)})
        if s[4] > SETTLE_ANCHOR_X + ACTION_DEAD_ZONE:
            return s


def _sample_skill_settle_right(rng: np.random.Generator) -> np.ndarray:
    while True:
        s = _sample_raw_breakout(rng, {"ball_dy": (-5.0, -_PRED_DY_EPS)})
        if s[4] < SETTLE_ANCHOR_X - ACTION_DEAD_ZONE:
            return s


def _sample_skill_settle_hold(rng: np.random.Generator) -> np.ndarray:
    while True:
        s = _sample_raw_breakout(rng, {"ball_dy": (-5.0, -_PRED_DY_EPS)})
        if abs(s[4] - SETTLE_ANCHOR_X) <= ACTION_DEAD_ZONE:
            return s


GENERATORS: Dict[int, Callable[[np.random.Generator], np.ndarray]] = {
    SKILL_RECEIVE_LEFT:  _sample_skill_receive_left,
    SKILL_RECEIVE_RIGHT: _sample_skill_receive_right,
    SKILL_RECEIVE_HOLD:  _sample_skill_receive_hold,
    SKILL_SETTLE_LEFT:   _sample_skill_settle_left,
    SKILL_SETTLE_RIGHT:  _sample_skill_settle_right,
    SKILL_SETTLE_HOLD:   _sample_skill_settle_hold,
}


def generate_dataset(
    class_id: int,
    n_samples: int,
    seed: int = 42,
    standardize_output: bool = True,
) -> torch.Tensor:
    """Returns (n_samples, STATE_DIM) float32 for a Breakout skill
    class. Output is standardized by default to match what the
    BreakoutStateBuilder emits at inference time."""
    if class_id not in GENERATORS:
        raise ValueError(
            f"unknown breakout class_id {class_id}; valid: "
            f"{sorted(GENERATORS.keys())}"
        )
    rng = np.random.default_rng(seed)
    gen = GENERATORS[class_id]
    raw = np.stack([gen(rng) for _ in range(n_samples)], axis=0)
    if standardize_output:
        raw = standardize(raw)
    return torch.from_numpy(raw)


# ---------------------------------------------------------------------
# RAM range probe (referenced from breakout_oracle for empirical
# calibration; the helper there does the actual env-driving)
# ---------------------------------------------------------------------


def probe_ram(ale, n_steps: int = 200) -> Dict[str, Tuple[int, int]]:
    """Read RAM each step over the next n_steps env frames (caller
    drives env.step). Returns observed (min, max) per relevant byte
    across all in-play frames. Use this to calibrate the RAM_*_RANGE
    constants.

    Caller pattern:
        observed = {"paddle_x": (255, 0), ...}
        for _ in range(n_steps):
            s = extract_breakout(ale)
            if s.ball_in_play:
                update each min/max
            env.step(action)
    """
    raise NotImplementedError(
        "probe_ram() is a documentation stub; the breakout_oracle "
        "script implements the probe inline so it can drive env.step."
    )


__all__ = [
    "STATE_DIM", "FIELDS", "RANGES", "STANDARDIZE",
    "PADDLE_Y_84", "PADDLE_LOOKAHEAD", "ACTION_DEAD_ZONE", "SETTLE_ANCHOR_X",
    "RAM_PADDLE_X_RANGE", "RAM_BALL_X_RANGE", "RAM_BALL_Y_RANGE",
    "SKILL_RECEIVE_LEFT", "SKILL_RECEIVE_RIGHT", "SKILL_RECEIVE_HOLD",
    "SKILL_SETTLE_LEFT", "SKILL_SETTLE_RIGHT", "SKILL_SETTLE_HOLD",
    "SKILL_LAUNCH_FIRE", "CLASS_NAMES_BREAKOUT",
    "predict_ball_x_at_impact", "standardize", "generate_dataset",
    "BreakoutStateBuilder", "select_skill_breakout",
    "GENERATORS", "_PRED_DY_EPS",
]
