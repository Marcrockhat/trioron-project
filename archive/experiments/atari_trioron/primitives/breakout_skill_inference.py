"""Gated multi-skill Breakout inference.

Two trioron primitive donors (RECEIVE, SETTLE) cover the in-play
state space; LAUNCH is gated out-of-band whenever ball_in_play=False
(no donor — there's only one action and nothing to learn). A hand-
coded gate on `ball_dy` routes each frame to the appropriate donor.
This isolates "can trioron *learn* each motor skill?" from "can
trioron *route* between skills?" — routing can be learned later via
absorb.

Donor paths:
    outputs/atari_primitive_donors/BO_SKILL_RECEIVE.pt
    outputs/atari_primitive_donors/BO_SKILL_SETTLE.pt

Each donor's union_classes covers only its skill's class IDs; we
mask logits to those classes before argmax.

Usage:
    python3 -m experiments.atari_trioron.primitives.breakout_skill_inference
    python3 -m experiments.atari_trioron.primitives.breakout_skill_inference --seed 0 --max-steps 5000
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
from gymnasium.wrappers import (
    AtariPreprocessing, FrameStackObservation, RecordVideo,
)

PROJ = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from trioron.api import load_organism  # noqa: E402

from experiments.atari_trioron.features import extract_breakout, get_ale  # noqa: E402
from experiments.atari_trioron.eval_render import _FlatRGBObsWrapper, _resolve_env_id  # noqa: E402
from experiments.atari_trioron.env import FRAME_HW, FRAME_STACK  # noqa: E402
from experiments.atari_trioron.primitives.breakout_state import (  # noqa: E402
    BreakoutStateBuilder, select_skill_breakout, STATE_DIM,
    SKILL_RECEIVE_LEFT, SKILL_RECEIVE_RIGHT, SKILL_RECEIVE_HOLD,
    SKILL_SETTLE_LEFT, SKILL_SETTLE_RIGHT, SKILL_SETTLE_HOLD,
)

gym.register_envs(ale_py)


# ALE Breakout actions.
ALE_NOOP, ALE_FIRE, ALE_RIGHT, ALE_LEFT = 0, 1, 2, 3

SKILL_CLASS_TO_ALE: Dict[int, int] = {
    SKILL_RECEIVE_LEFT:  ALE_LEFT,
    SKILL_RECEIVE_RIGHT: ALE_RIGHT,
    SKILL_RECEIVE_HOLD:  ALE_NOOP,
    SKILL_SETTLE_LEFT:   ALE_LEFT,
    SKILL_SETTLE_RIGHT:  ALE_RIGHT,
    SKILL_SETTLE_HOLD:   ALE_NOOP,
}

SKILL_CLASS_SETS: Dict[str, set] = {
    "RECEIVE": {SKILL_RECEIVE_LEFT, SKILL_RECEIVE_RIGHT, SKILL_RECEIVE_HOLD},
    "SETTLE":  {SKILL_SETTLE_LEFT, SKILL_SETTLE_RIGHT, SKILL_SETTLE_HOLD},
}

DONOR_ROOT = PROJ / "outputs" / "atari_primitive_donors"

# Empirical paddle-x84 limits from oracle's RAM probe (RAM 88..190
# under RAM_PADDLE_X_RANGE=(57,201) → paddle_x84 ∈ [18, 78]). Suppress
# LEFT/RIGHT actions when paddle is already against the rail — the
# motor donor never saw "paddle stuck at edge" during training, and a
# wasted action locks the chosen direction instead of letting tracking
# adapt.
PADDLE_LEFT_84 = 20.0
PADDLE_RIGHT_84 = 76.0


@torch.no_grad()
def select_action_for_skill(donor, state_vec: torch.Tensor,
                            skill: str,
                            paddle_x84: float = 42.0,
                            fallback_action: int = ALE_NOOP) -> int:
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
    if action == ALE_LEFT and paddle_x84 < PADDLE_LEFT_84:
        return ALE_NOOP
    if action == ALE_RIGHT and paddle_x84 > PADDLE_RIGHT_84:
        return ALE_NOOP
    return action


def evaluate_breakout_skills(
    *,
    out_dir: Path,
    seed: int = 0,
    max_steps: int = 5000,
    name: str = "breakout_skill_eval",
    verbose: bool = True,
    donor_root: Path = DONOR_ROOT,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_id = _resolve_env_id("Breakout")
    env = gym.make(env_id, render_mode="rgb_array",
                   frameskip=1, repeat_action_probability=0.0,
                   full_action_space=False)
    env = RecordVideo(env, video_folder=str(out_dir),
                      episode_trigger=lambda i: i == 0,
                      name_prefix=name, disable_logger=True)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4,
                             screen_size=FRAME_HW,
                             terminal_on_life_loss=False,
                             grayscale_obs=True, grayscale_newaxis=False,
                             scale_obs=False)
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    env = _FlatRGBObsWrapper(env)

    donors = {
        "RECEIVE": load_organism(donor_root / "BO_SKILL_RECEIVE.pt"),
        "SETTLE":  load_organism(donor_root / "BO_SKILL_SETTLE.pt"),
    }
    ale = get_ale(env)
    state_builder = BreakoutStateBuilder()

    obs, _ = env.reset(seed=seed)
    state_builder.reset()
    ret = 0.0
    n_steps = 0
    action_counts: Dict[int, int] = {a: 0 for a in (ALE_NOOP, ALE_FIRE,
                                                    ALE_RIGHT, ALE_LEFT)}
    skill_counts: Dict[str, int] = {"LAUNCH": 0, "BUILDUP": 0,
                                    "RECEIVE": 0, "SETTLE": 0}
    t0 = time.time()
    for _ in range(max_steps):
        sv_raw = state_builder.step(ale)
        s = extract_breakout(ale)
        if sv_raw[0] is None:
            if not s.ball_in_play:
                action = ALE_FIRE
                skill_counts["LAUNCH"] += 1
            else:
                action = ALE_NOOP
                skill_counts["BUILDUP"] += 1
        else:
            sv, raw = sv_raw
            paddle_x84 = float(raw[4])
            skill = select_skill_breakout(raw)
            skill_counts[skill] += 1
            action = select_action_for_skill(donors[skill], sv, skill,
                                             paddle_x84=paddle_x84)
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
                                / "breakout_skill_eval"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=5000)
    args = ap.parse_args()
    res = evaluate_breakout_skills(out_dir=Path(args.out_dir),
                                   seed=args.seed,
                                   max_steps=args.max_steps)
    print(f"\nfinal: return={res['return']:+.1f} length={res['length']}")


if __name__ == "__main__":
    main()
