"""Predict-x oracle for Breakout — no donor, hand-coded policy.

Three-state hand policy (parallel to pong_oracle's catch+prepos
structure, minus SMASH):

    ball not in play   → FIRE (re-launch)
    ball_dy > 0        → track pred_x with paddle (LEFT/RIGHT/NOOP)
    ball_dy ≤ 0        → drift toward SETTLE_ANCHOR_X (LEFT/RIGHT/NOOP)

The oracle reads ALE RAM directly via extract_breakout and consumes
the same rescaled state vector the trioron donor will eventually
consume. RAM ranges in breakout_state are first-cut estimates; this
script also instruments observed RAM mins/maxes per byte and dumps
them at end-of-episode for calibration.

Usage:
    python3 -m experiments.atari_trioron.primitives.breakout_oracle
    python3 -m experiments.atari_trioron.primitives.breakout_oracle --seed 0 --max-steps 5000
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.wrappers import (
    AtariPreprocessing, FrameStackObservation, RecordVideo,
)

PROJ = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from experiments.atari_trioron.features import extract_breakout, get_ale  # noqa: E402
from experiments.atari_trioron.eval_render import _FlatRGBObsWrapper, _resolve_env_id  # noqa: E402
from experiments.atari_trioron.env import FRAME_HW, FRAME_STACK  # noqa: E402
from experiments.atari_trioron.primitives.breakout_state import (  # noqa: E402
    BreakoutStateBuilder, select_skill_breakout,
    PADDLE_Y_84, PADDLE_LOOKAHEAD, ACTION_DEAD_ZONE, SETTLE_ANCHOR_X,
    _PRED_DY_EPS,
)

gym.register_envs(ale_py)


# ALE Breakout action set: NOOP, FIRE, RIGHT, LEFT.
ALE_NOOP, ALE_FIRE, ALE_RIGHT, ALE_LEFT = 0, 1, 2, 3


def _track_target_x(target_x: float, paddle_x: float,
                    paddle_dx: float) -> int:
    """Momentum-aware paddle action toward target_x.

    Compares target against the paddle's projected position one step
    out under HOLD; HOLDs inside the dead zone to prevent jitter.
    """
    eff_px = paddle_x + paddle_dx * PADDLE_LOOKAHEAD
    if target_x < eff_px - ACTION_DEAD_ZONE:
        return ALE_LEFT
    if target_x > eff_px + ACTION_DEAD_ZONE:
        return ALE_RIGHT
    return ALE_NOOP


def evaluate_oracle(*, out_dir: Path, seed: int = 0, max_steps: int = 5000,
                    name: str = "breakout_oracle_eval",
                    verbose: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_id = _resolve_env_id("Breakout")
    env = gym.make(env_id, render_mode="rgb_array",
                   frameskip=1, repeat_action_probability=0.0,
                   full_action_space=False)
    env = RecordVideo(env, video_folder=str(out_dir),
                      episode_trigger=lambda i: i == 0,
                      name_prefix=name, disable_logger=True)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=FRAME_HW,
                             terminal_on_life_loss=False,
                             grayscale_obs=True, grayscale_newaxis=False,
                             scale_obs=False)
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    env = _FlatRGBObsWrapper(env)

    ale = get_ale(env)
    state_builder = BreakoutStateBuilder()

    obs, _ = env.reset(seed=seed)
    state_builder.reset()
    ret = 0.0
    n_steps = 0
    action_counts = {ALE_NOOP: 0, ALE_FIRE: 0, ALE_RIGHT: 0, ALE_LEFT: 0}
    skill_counts = {"LAUNCH": 0, "RECEIVE": 0, "SETTLE": 0, "BUILDUP": 0}

    # Observed RAM byte ranges (for post-hoc calibration of the
    # RAM_*_RANGE constants in breakout_state).
    ram_min = {"paddle_x": 255, "ball_x": 255, "ball_y": 255}
    ram_max = {"paddle_x": 0,   "ball_x": 0,   "ball_y": 0}
    in_play_frames = 0
    # Track ball_y values seen near the moment the ball is at paddle
    # row, for jointly calibrating PADDLE_Y_84 with RAM_BALL_Y_RANGE.
    ball_y_near_paddle: list = []

    t0 = time.time()
    for _ in range(max_steps):
        sv_raw = state_builder.step(ale)
        s = extract_breakout(ale)
        if s.ball_in_play:
            in_play_frames += 1
            ram_min["paddle_x"] = min(ram_min["paddle_x"], s.paddle_x)
            ram_max["paddle_x"] = max(ram_max["paddle_x"], s.paddle_x)
            ram_min["ball_x"]   = min(ram_min["ball_x"], s.ball_x)
            ram_max["ball_x"]   = max(ram_max["ball_x"], s.ball_x)
            ram_min["ball_y"]   = min(ram_min["ball_y"], s.ball_y)
            ram_max["ball_y"]   = max(ram_max["ball_y"], s.ball_y)

        if sv_raw[0] is None:
            # Either ball not in play (LAUNCH) or first valid frame
            # post-launch (no velocity yet → BUILDUP).
            if not s.ball_in_play:
                action = ALE_FIRE
                skill_counts["LAUNCH"] += 1
            else:
                action = ALE_NOOP
                skill_counts["BUILDUP"] += 1
        else:
            sv, raw = sv_raw
            ball_y84 = float(raw[1])
            paddle_x84 = float(raw[4])
            paddle_dx = float(raw[5])
            pred_x = float(raw[7])
            # Sample ball_y for paddle-row calibration when ball is
            # within ~5 84-units of where we think the paddle is.
            if abs(ball_y84 - PADDLE_Y_84) <= 5.0:
                ball_y_near_paddle.append(s.ball_y)
            skill = select_skill_breakout(raw)
            skill_counts[skill] += 1
            if skill == "RECEIVE":
                action = _track_target_x(pred_x, paddle_x84, paddle_dx)
            else:  # SETTLE
                action = _track_target_x(SETTLE_ANCHOR_X, paddle_x84,
                                         paddle_dx)
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
        print(f"[oracle] return={ret:+.1f}  length={n_steps}  ({elapsed:.1f}s)")
        print(f"  action counts: {action_counts}")
        print(f"  skill counts:  {skill_counts}")
        print(f"  in-play frames: {in_play_frames}")
        print(f"  observed RAM (min..max):")
        print(f"    paddle_x [byte 72]: {ram_min['paddle_x']}..{ram_max['paddle_x']}")
        print(f"    ball_x   [byte 99]: {ram_min['ball_x']}..{ram_max['ball_x']}")
        print(f"    ball_y   [byte 101]: {ram_min['ball_y']}..{ram_max['ball_y']}")
        if ball_y_near_paddle:
            arr = np.array(ball_y_near_paddle)
            print(f"  ball_y RAM near paddle row: "
                  f"n={len(arr)} mean={arr.mean():.1f} "
                  f"min={arr.min()} max={arr.max()}")
        print(f"  video: {video_path}")
    return {
        "return": ret, "length": n_steps,
        "action_counts": action_counts,
        "skill_counts": skill_counts,
        "ram_min": ram_min, "ram_max": ram_max,
        "in_play_frames": in_play_frames,
        "ball_y_near_paddle_mean": (
            float(np.mean(ball_y_near_paddle)) if ball_y_near_paddle else None
        ),
        "video_path": str(video_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=str,
                    default=str(PROJ / "outputs" / "atari_primitive_donors"
                                / "breakout_eval"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=5000)
    args = ap.parse_args()
    res = evaluate_oracle(out_dir=Path(args.out_dir), seed=args.seed,
                          max_steps=args.max_steps)
    print(f"\nfinal: return={res['return']:+.1f} length={res['length']}")


if __name__ == "__main__":
    main()
