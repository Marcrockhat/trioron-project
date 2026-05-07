"""Real-ALE Pong inference for the curriculum donor.

Reads ALE RAM, builds the standardized 7-d state vector matching what
the donor was trained on, predicts a Pong action class, maps to an
ALE action, runs the episode, and records video.

The mapping from ALE RAM coords to the synthetic env's [0, 84] frame
coords is approximate but adequate — the standardization absorbs the
absolute scale. RAM ranges below were measured empirically from a
sample episode; recalibrate if the donor underperforms vs synthetic.

Usage:
    python3 -m experiments.atari_trioron.primitives.pong_inference
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

import gymnasium as gym
import ale_py
from gymnasium.wrappers import (
    AtariPreprocessing, FrameStackObservation, RecordVideo,
)

PROJ = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from trioron.api import load_organism   # noqa: E402
from experiments.atari_trioron.features import extract_pong, get_ale   # noqa: E402
from experiments.atari_trioron.eval_render import _FlatRGBObsWrapper, _resolve_env_id   # noqa: E402
from experiments.atari_trioron.env import FRAME_HW, FRAME_STACK   # noqa: E402
from experiments.atari_trioron.primitives.synthetic_env import (   # noqa: E402
    STATE_DIM, standardize, predict_ball_y_at_impact, PADDLE_HEIGHT_84,
    PONG_ACTION_UP, PONG_ACTION_DOWN, PONG_ACTION_HOLD, CLASS_NAMES,
)

gym.register_envs(ale_py)


# ALE Pong RAM coordinates (empirical ranges measured under random
# play, 2026-05-07 probe). Translation to synthetic env's [0, 84]
# frame coords is linear:
#     coord_84 = (coord_ram - lo) * 84 / (hi - lo)
# ball_x's true playfield range is 68..205 (opponent paddle column to
# agent paddle column). Earlier 0..205 mapping compressed bx84 into
# [27.9, 84] and made the kinematic predictor reflect off phantom walls.
RAM_BALL_X_RANGE = (68.0, 205.0)
RAM_BALL_Y_RANGE = (38.0, 203.0)
RAM_PADDLE_Y_RANGE = (38.0, 203.0)

# AtariPreprocessing applies frame_skip=4. Velocities computed as
# (now - prev) span 4 game frames — i.e. one preprocessed step. The
# synthetic env's RANGES["ball_dx"] = (-5, 5) is unit-free per-step,
# and an empirical probe of real ALE Pong shows per-step ball_dx
# distributes with std≈4.9 — well-matched to synthetic. Earlier code
# divided by ALE_FRAME_SKIP, which crushed real ball_dx into a band
# where 97% of frames fell below the kinematic-prediction threshold.
# Per-step delta is the right convention; no rescale needed.
ALE_FRAME_SKIP = 1


def _ram_to_84(value: float, lo: float, hi: float) -> float:
    return (float(value) - lo) * 84.0 / (hi - lo)


# Action-class → ALE action.
PONG_CLASS_TO_ALE_ACTION = {
    PONG_ACTION_UP:   2,
    PONG_ACTION_DOWN: 3,
    PONG_ACTION_HOLD: 0,
}


class PongStateBuilder:
    """Stateful wrapper: each step produces the standardized 7-d
    vector by reading current RAM and differencing against the
    previous frame's stored state. Returns None when ball isn't in
    play (those frames have no useful state)."""

    def __init__(self) -> None:
        self.prev_ball_x: float = 0.0
        self.prev_ball_y: float = 0.0
        self.prev_paddle_y: float = 0.0
        self.has_prev: bool = False

    def reset(self) -> None:
        self.has_prev = False

    def step(self, ale) -> "torch.Tensor | None":
        s = extract_pong(ale)
        if not s.ball_in_play:
            self.has_prev = False
            return None
        ball_x84 = _ram_to_84(s.ball_x, *RAM_BALL_X_RANGE)
        ball_y84 = _ram_to_84(s.ball_y, *RAM_BALL_Y_RANGE)
        # Byte 51 is paddle TOP edge; shift to paddle center for parity
        # with synthetic env (which treats paddle_y as the catching center).
        paddle_y84 = _ram_to_84(s.my_paddle_y, *RAM_PADDLE_Y_RANGE) + PADDLE_HEIGHT_84 / 2.0
        if self.has_prev:
            # Divide by frame_skip to match the donor's per-frame
            # training distribution.
            ball_dx = (ball_x84 - self.prev_ball_x) / ALE_FRAME_SKIP
            ball_dy = (ball_y84 - self.prev_ball_y) / ALE_FRAME_SKIP
            paddle_dy = (paddle_y84 - self.prev_paddle_y) / ALE_FRAME_SKIP
        else:
            ball_dx = 0.0
            ball_dy = 0.0
            paddle_dy = 0.0
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
        ], dtype=np.float32)
        # Cache for next frame.
        self.prev_ball_x = ball_x84
        self.prev_ball_y = ball_y84
        self.prev_paddle_y = paddle_y84
        self.has_prev = True
        std_vec = standardize(raw[None, :])
        return torch.from_numpy(std_vec)  # (1, STATE_DIM)


@torch.no_grad()
def select_action(organism, state_vec: torch.Tensor,
                  rng: np.random.Generator,
                  fallback_action: int = 0,
                  eps: float = 0.0) -> int:
    """Run the state vector through the organism, mask to Pong-action
    classes, argmax → ALE action. With prob `eps`, take a random
    valid action (epsilon-greedy)."""
    if rng.random() < eps:
        return int(rng.choice(list(PONG_CLASS_TO_ALE_ACTION.values())))
    logits = organism(state_vec, routing="soft")
    if isinstance(logits, tuple):
        logits = logits[0]
    union = list(organism.union_classes)
    eligible = set(PONG_CLASS_TO_ALE_ACTION.keys())
    masked = torch.full_like(logits, float("-inf"))
    for j, c in enumerate(union):
        if int(c) in eligible:
            masked[:, j] = logits[:, j]
    if torch.isinf(masked).all():
        return fallback_action
    pred_idx = int(masked[0].argmax())
    pred_class = int(union[pred_idx])
    return PONG_CLASS_TO_ALE_ACTION[pred_class]


def evaluate_pong(
    *,
    organism_path: Path,
    out_dir: Path,
    seed: int = 0,
    eps: float = 0.0,
    max_steps: int = 5000,
    name: str = "pong_curriculum_eval",
    verbose: bool = True,
) -> dict:
    """Play one Pong episode, record video, return summary."""
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

    organism = load_organism(organism_path)
    ale = get_ale(env)
    state_builder = PongStateBuilder()
    rng = np.random.default_rng(seed)

    obs, _info = env.reset(seed=seed)
    state_builder.reset()
    ret = 0.0
    n_steps = 0
    action_counts: dict = {0: 0, 2: 0, 3: 0}
    t0 = time.time()
    for _ in range(max_steps):
        sv = state_builder.step(ale)
        if sv is None:
            action = 0  # HOLD when ball not in play
        else:
            action = select_action(organism, sv, rng, eps=eps)
        action_counts[action] = action_counts.get(action, 0) + 1
        obs, r, term, trunc, _info = env.step(action)
        ret += float(r)
        n_steps += 1
        if term or trunc:
            break
    env.close()
    elapsed = time.time() - t0
    video_path = out_dir / f"{name}-episode-0.mp4"
    if verbose:
        print(f"[pong] return={ret:+.1f}  length={n_steps}  "
              f"({elapsed:.1f}s)")
        print(f"  action counts: {action_counts}")
        v_status = "OK" if video_path.exists() else "(missing)"
        print(f"  video: {video_path} {v_status}")
    return {
        "return": ret,
        "length": n_steps,
        "action_counts": action_counts,
        "video_path": str(video_path),
        "wallclock": elapsed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organism", type=str,
                    default=str(PROJ / "outputs" / "atari_primitive_donors"
                                / "pong_curriculum_donor.pt"))
    ap.add_argument("--out-dir", type=str,
                    default=str(PROJ / "outputs" / "atari_primitive_donors"
                                / "pong_eval"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=5000)
    args = ap.parse_args()

    res = evaluate_pong(
        organism_path=Path(args.organism),
        out_dir=Path(args.out_dir),
        seed=args.seed, eps=args.eps,
        max_steps=args.max_steps,
    )
    print(f"\nfinal: return={res['return']:+.1f} length={res['length']}")


if __name__ == "__main__":
    main()
