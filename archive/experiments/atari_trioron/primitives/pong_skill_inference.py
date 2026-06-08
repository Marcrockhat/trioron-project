"""Gated multi-skill Pong inference.

Three trioron primitive donors (CATCH, SMASH, PREPOS) cover the
state space disjointly. A hand-coded gate routes each frame to the
appropriate donor based on (bdx sign, bx84 vs SMASH_TRIGGER_X). This
isolates "can trioron *learn* each motor skill?" from "can trioron
*route* between skills?" — routing can be learned later.

Donor paths:
    outputs/atari_primitive_donors/SKILL_CATCH.pt
    outputs/atari_primitive_donors/SKILL_SMASH.pt
    outputs/atari_primitive_donors/SKILL_PREPOS.pt

Each donor's union_classes covers only its skill's class IDs; we
mask logits to those classes before argmax.

Usage:
    python3 -m experiments.atari_trioron.primitives.pong_skill_inference
    python3 -m experiments.atari_trioron.primitives.pong_skill_inference --seed 0 --max-steps 5000
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo

PROJ = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from trioron.api import load_organism  # noqa: E402

from experiments.atari_trioron.features import extract_pong, get_ale  # noqa: E402
from experiments.atari_trioron.eval_render import _FlatRGBObsWrapper, _resolve_env_id  # noqa: E402
from experiments.atari_trioron.env import FRAME_HW, FRAME_STACK  # noqa: E402
from experiments.atari_trioron.primitives.synthetic_env import (  # noqa: E402
    STATE_DIM, standardize, predict_ball_y_at_impact, PADDLE_HEIGHT_84,
    SMASH_TRIGGER_X, _PRED_DX_EPS,
    SKILL_CATCH_UP, SKILL_CATCH_DOWN, SKILL_CATCH_HOLD,
    SKILL_SMASH_UP, SKILL_SMASH_DOWN,
    SKILL_PREPOS_UP, SKILL_PREPOS_DOWN, SKILL_PREPOS_HOLD,
)
from experiments.atari_trioron.primitives.pong_inference import (  # noqa: E402
    _ram_to_84, RAM_BALL_X_RANGE, RAM_BALL_Y_RANGE, RAM_PADDLE_Y_RANGE,
    ALE_FRAME_SKIP,
)

gym.register_envs(ale_py)


# Map skill class ID → ALE action.
ALE_HOLD, ALE_UP, ALE_DOWN = 0, 2, 3
SKILL_CLASS_TO_ALE: Dict[int, int] = {
    SKILL_CATCH_UP:    ALE_UP,
    SKILL_CATCH_DOWN:  ALE_DOWN,
    SKILL_CATCH_HOLD:  ALE_HOLD,
    SKILL_SMASH_UP:    ALE_UP,
    SKILL_SMASH_DOWN:  ALE_DOWN,
    SKILL_PREPOS_UP:   ALE_UP,
    SKILL_PREPOS_DOWN: ALE_DOWN,
    SKILL_PREPOS_HOLD: ALE_HOLD,
}

# Per-skill class membership for masking.
SKILL_CLASS_SETS: Dict[str, set] = {
    "CATCH":  {SKILL_CATCH_UP, SKILL_CATCH_DOWN, SKILL_CATCH_HOLD},
    "SMASH":  {SKILL_SMASH_UP, SKILL_SMASH_DOWN},
    "PREPOS": {SKILL_PREPOS_UP, SKILL_PREPOS_DOWN, SKILL_PREPOS_HOLD},
}

DONOR_ROOT = PROJ / "outputs" / "atari_primitive_donors"


class PongSkillStateBuilder:
    """Stateful 9-d perception. Same as the 8-d builder but appends
    opp_y as a 9th field. ALE Pong stores opp paddle's TOP edge in RAM
    byte 50; we shift by half-paddle-height for symmetry with my_y."""

    def __init__(self) -> None:
        self.prev_ball_x: float = 0.0
        self.prev_ball_y: float = 0.0
        self.prev_paddle_y: float = 0.0
        self.has_prev: bool = False

    def reset(self) -> None:
        self.has_prev = False

    def step(self, ale):
        s = extract_pong(ale)
        if not s.ball_in_play:
            self.has_prev = False
            return None, None
        ball_x84 = _ram_to_84(s.ball_x, *RAM_BALL_X_RANGE)
        ball_y84 = _ram_to_84(s.ball_y, *RAM_BALL_Y_RANGE)
        paddle_y84 = _ram_to_84(s.my_paddle_y, *RAM_PADDLE_Y_RANGE) + PADDLE_HEIGHT_84 / 2.0
        opp_y84 = _ram_to_84(s.opp_paddle_y, *RAM_PADDLE_Y_RANGE) + PADDLE_HEIGHT_84 / 2.0
        # Fix 2026-05-08: first valid frame has no velocity → previously
        # bdx=0 routed to PREPOS, pulling paddle off the incoming ball.
        # Now we cache prev fields and return None so the inference loop
        # HOLDs until next frame produces real differenced velocities.
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
        pred_y = predict_ball_y_at_impact(
            ball_x=ball_x84, ball_y=ball_y84,
            ball_dx=ball_dx, ball_dy=ball_dy,
        )
        raw = np.array([
            ball_x84, ball_y84,
            ball_dx, ball_dy,
            paddle_y84, paddle_dy,
            ball_speed, pred_y,
            opp_y84,
        ], dtype=np.float32)
        self.prev_ball_x = ball_x84
        self.prev_ball_y = ball_y84
        self.prev_paddle_y = paddle_y84
        self.has_prev = True
        std_vec = standardize(raw[None, :])
        # Return both raw and standardized — the gate uses raw values
        # (bdx sign, bx84 threshold), the donor consumes standardized.
        return torch.from_numpy(std_vec), raw


def select_skill(raw_state: np.ndarray, prev_skill: str = "CATCH") -> str:
    """Hand-coded gate. raw_state is the unstandardized 9-d vector."""
    ball_dx = float(raw_state[2])
    ball_x = float(raw_state[0])
    if ball_dx <= _PRED_DX_EPS:
        return "PREPOS"
    if ball_x < SMASH_TRIGGER_X:
        return "CATCH"
    return "SMASH"


# Paddle range in 84-coords after center-shift: [4.075, 88.075].
# Override actions that would push the paddle past these limits — the
# motor donor never saw "paddle stuck at edge" during training, and a
# wasted action is worse than HOLD (it locks the chosen direction
# instead of letting tracking adapt).
PADDLE_TOP_84 = 5.0     # paddle_y84 below this → suppress UP
PADDLE_BOTTOM_84 = 87.0  # paddle_y84 above this → suppress DOWN


@torch.no_grad()
def select_action_for_skill(donor, state_vec: torch.Tensor,
                            skill: str,
                            paddle_y84: float = 42.0,
                            fallback_action: int = ALE_HOLD) -> int:
    eligible = SKILL_CLASS_SETS[skill]
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
    action = SKILL_CLASS_TO_ALE[pred_class]
    # Edge override: don't waste a UP/DOWN at the screen limits.
    if action == ALE_UP and paddle_y84 < PADDLE_TOP_84:
        return ALE_HOLD
    if action == ALE_DOWN and paddle_y84 > PADDLE_BOTTOM_84:
        return ALE_HOLD
    return action


def evaluate_pong_skills(
    *,
    out_dir: Path,
    seed: int = 0,
    max_steps: int = 5000,
    name: str = "pong_skill_eval",
    verbose: bool = True,
    donor_root: Path = DONOR_ROOT,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_id = _resolve_env_id("Pong")
    env = gym.make(
        env_id, render_mode="rgb_array",
        frameskip=1, repeat_action_probability=0.0,
        full_action_space=False,
    )
    env = RecordVideo(env, video_folder=str(out_dir),
                      episode_trigger=lambda i: i == 0,
                      name_prefix=name, disable_logger=True)
    env = AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=FRAME_HW,
        terminal_on_life_loss=False,
        grayscale_obs=True, grayscale_newaxis=False, scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    env = _FlatRGBObsWrapper(env)

    donors = {
        "CATCH":  load_organism(donor_root / "SKILL_CATCH.pt"),
        "SMASH":  load_organism(donor_root / "SKILL_SMASH.pt"),
        "PREPOS": load_organism(donor_root / "SKILL_PREPOS.pt"),
    }
    ale = get_ale(env)
    state_builder = PongSkillStateBuilder()

    obs, _info = env.reset(seed=seed)
    state_builder.reset()
    ret = 0.0
    n_steps = 0
    action_counts: Dict[int, int] = {ALE_HOLD: 0, ALE_UP: 0, ALE_DOWN: 0}
    skill_counts: Dict[str, int] = {"CATCH": 0, "SMASH": 0, "PREPOS": 0,
                                    "NONE": 0}
    prev_skill = "CATCH"
    t0 = time.time()
    for _ in range(max_steps):
        sv_raw = state_builder.step(ale)
        if sv_raw[0] is None:
            action = ALE_HOLD
            skill_counts["NONE"] += 1
        else:
            sv, raw = sv_raw
            skill = select_skill(raw, prev_skill)
            paddle_y84 = float(raw[4])
            action = select_action_for_skill(donors[skill], sv, skill,
                                             paddle_y84=paddle_y84)
            skill_counts[skill] += 1
            prev_skill = skill
        action_counts[action] = action_counts.get(action, 0) + 1
        obs, r, term, trunc, _ = env.step(action)
        ret += float(r)
        n_steps += 1
        if term or trunc:
            break
    env.close()
    elapsed = time.time() - t0
    video_path = out_dir / f"{name}-episode-0.mp4"
    if verbose:
        print(f"[skill] return={ret:+.1f}  length={n_steps}  ({elapsed:.1f}s)")
        print(f"  action counts: {action_counts}")
        print(f"  skill counts:  {skill_counts}")
        print(f"  video: {video_path}")
    return {
        "return": ret, "length": n_steps,
        "action_counts": action_counts,
        "skill_counts": skill_counts,
        "video_path": str(video_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=str,
                    default=str(PROJ / "outputs" / "atari_primitive_donors"
                                / "skill_eval"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=5000)
    args = ap.parse_args()
    res = evaluate_pong_skills(out_dir=Path(args.out_dir), seed=args.seed,
                               max_steps=args.max_steps)
    print(f"\nfinal: return={res['return']:+.1f} length={res['length']}")


if __name__ == "__main__":
    main()
