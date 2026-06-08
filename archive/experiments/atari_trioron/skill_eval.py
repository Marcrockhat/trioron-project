"""Skill-policy evaluation + video render.

Inference path: project the framestack through the organism, get
union-class logits, mask to the eligible skill-class set for the
target game, argmax → skill class → SKILL_TO_ACTION → game action.

Masking matters because an absorbed organism (arm 4) has BOTH
games' skill classes in its head; on Breakout we mask the Pong
classes to -inf so the agent doesn't try to "press UP" on a
Breakout frame.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import gymnasium as gym
import ale_py
from gymnasium.wrappers import (
    AtariPreprocessing, FrameStackObservation, RecordVideo,
)

from trioron.api import load_organism

from .env import FRAME_HW, FRAME_STACK, OBS_DIM, GAME_ACTION_MASK
from .eval_render import _FlatRGBObsWrapper, _resolve_env_id
from .skill_curriculum import (
    SKILL_TO_ACTION, PONG_SKILL_CLASSES, BREAKOUT_SKILL_CLASSES,
)

gym.register_envs(ale_py)


def _eligible_skill_classes(game: str):
    if game == "Pong":
        return PONG_SKILL_CLASSES
    if game == "Breakout":
        return BREAKOUT_SKILL_CLASSES
    raise ValueError(game)


@torch.no_grad()
def _select_skill_action(
    organism, obs_tensor, eligible_classes, eps, rng,
    fallback_action_set,
):
    if rng.random() < eps:
        return int(rng.choice(fallback_action_set))
    logits = organism(obs_tensor, routing="soft").squeeze(0)
    union = list(organism.union_classes)
    eligible_set = set(eligible_classes)
    # Mask out any class not in eligible_set.
    masked = torch.full_like(logits, float("-inf"))
    for i, c in enumerate(union):
        if int(c) in eligible_set:
            masked[i] = logits[i]
    if torch.isinf(masked).all():
        # Organism doesn't cover this game's skill classes — fallback.
        return int(rng.choice(fallback_action_set))
    idx = int(masked.argmax())
    skill_class = int(union[idx])
    action = SKILL_TO_ACTION.get(skill_class, 0)
    return action


def evaluate_skill_organism(
    *,
    organism_path: Path,
    game: str,
    out_dir: Path,
    name: str = "eval",
    seed: int = 0,
    eps: float = 0.0,
    max_steps: int = 30_000,
    verbose: bool = True,
) -> dict:
    """Play one episode in `game` driven by skill-class argmax.
    Records MP4 to {out_dir}/{name}-episode-0.mp4."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_id = _resolve_env_id(game)

    env = gym.make(
        env_id, render_mode="rgb_array",
        frameskip=1, repeat_action_probability=0.0,
        full_action_space=False,
    )
    env = RecordVideo(
        env, video_folder=str(out_dir),
        episode_trigger=lambda i: i == 0,
        name_prefix=name,
        disable_logger=True,
    )
    env = AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=FRAME_HW,
        terminal_on_life_loss=False,
        grayscale_obs=True, grayscale_newaxis=False, scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    env = _FlatRGBObsWrapper(env)

    organism = load_organism(organism_path)
    eligible = _eligible_skill_classes(game)
    valid_action_set = np.flatnonzero(GAME_ACTION_MASK[game])
    rng = np.random.default_rng(seed)

    obs, _info = env.reset(seed=seed)
    ret = 0.0
    n_steps = 0
    t0 = time.time()
    for _t in range(max_steps):
        obs_t = torch.from_numpy(obs).unsqueeze(0).float()
        a = _select_skill_action(
            organism, obs_t, eligible, eps, rng, valid_action_set,
        )
        obs, r, term, trunc, _info = env.step(a)
        ret += float(r)
        n_steps += 1
        if term or trunc:
            break
    env.close()
    elapsed = time.time() - t0

    video_path = out_dir / f"{name}-episode-0.mp4"
    if verbose:
        v_status = "OK" if video_path.exists() else "(missing)"
        print(f"[skill-eval] {game} return={ret:+.1f} length={n_steps} "
              f"({elapsed:.1f}s) → {video_path} {v_status}")
    return {
        "return": ret,
        "length": n_steps,
        "video_path": str(video_path),
        "wallclock": elapsed,
    }
