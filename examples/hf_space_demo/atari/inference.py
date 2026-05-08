"""Run one Pong / Breakout match with the chosen organism, record
the gameplay to MP4, return summary stats."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.wrappers import (
    AtariPreprocessing, FrameStackObservation, RecordVideo,
)

from .features import extract_breakout, get_ale
from .state import (
    PongSkillStateBuilder, BreakoutStateBuilder,
    select_skill_pong, select_skill_breakout,
    select_action_pong, select_action_breakout,
    ALE_HOLD, ALE_NOOP, ALE_FIRE,
)


gym.register_envs(ale_py)
FRAME_HW = 84
FRAME_STACK = 4


class _FlatRGBObsWrapper(gym.ObservationWrapper):
    """Flatten the (FRAME_STACK, FRAME_HW, FRAME_HW) framestack into
    a (28224,) float32 vector in [0,1]. Trioron's L0 takes a 2D
    tensor (N, input_dim); this flattens one row of it."""
    def __init__(self, env):
        super().__init__(env)
        n = FRAME_STACK * FRAME_HW * FRAME_HW
        self.observation_space = gym.spaces.Box(
            low=np.zeros(n, dtype=np.float32),
            high=np.ones(n, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, obs):
        return (obs.astype(np.float32) / 255.0).reshape(-1)


def _resolve_env_id(game: str) -> str:
    if "/" in game:
        return game
    return f"ALE/{game}-v5"


def _make_env(game: str, *, seed: int, record_dir: Path, record_name: str):
    env_id = _resolve_env_id(game)
    env = gym.make(
        env_id, render_mode="rgb_array",
        frameskip=1, repeat_action_probability=0.0,
        full_action_space=False,
    )
    env = RecordVideo(
        env, video_folder=str(record_dir),
        episode_trigger=lambda i: i == 0,
        name_prefix=record_name, disable_logger=True,
    )
    env = AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=FRAME_HW,
        terminal_on_life_loss=False,
        grayscale_obs=True, grayscale_newaxis=False, scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    env = _FlatRGBObsWrapper(env)
    return env


def play_pong(organism, *, seed: int, max_steps: int = 8000,
              record_dir: Path,
              record_name: str = "pong_match") -> dict:
    env = _make_env("Pong", seed=seed, record_dir=record_dir,
                    record_name=record_name)
    ale = get_ale(env)
    sb = PongSkillStateBuilder()
    obs, _ = env.reset(seed=seed)
    sb.reset()
    ret = 0.0
    n_steps = 0
    prev_skill = "CATCH"
    donors = {"CATCH": organism, "SMASH": organism, "PREPOS": organism}
    for _ in range(max_steps):
        sv_raw = sb.step(ale)
        if sv_raw[0] is None:
            action = ALE_HOLD
        else:
            sv, raw = sv_raw
            skill = select_skill_pong(raw, prev_skill)
            paddle_y84 = float(raw[4])
            action = select_action_pong(
                donors[skill], sv, skill,
                paddle_y84=paddle_y84, fallback_action=ALE_HOLD,
            )
            prev_skill = skill
        obs, r, term, trunc, _ = env.step(action)
        ret += float(r)
        n_steps += 1
        if term or trunc:
            break
    env.close()
    video = record_dir / f"{record_name}-episode-0.mp4"
    return {"return": ret, "length": n_steps, "video_path": str(video)}


def play_breakout(organism, *, seed: int, max_steps: int = 8000,
                  record_dir: Path,
                  record_name: str = "breakout_match") -> dict:
    env = _make_env("Breakout", seed=seed, record_dir=record_dir,
                    record_name=record_name)
    ale = get_ale(env)
    sb = BreakoutStateBuilder()
    obs, _ = env.reset(seed=seed)
    sb.reset()
    ret = 0.0
    n_steps = 0
    donors = {"RECEIVE": organism, "SETTLE": organism}
    for _ in range(max_steps):
        sv_raw = sb.step(ale)
        s = extract_breakout(ale)
        if sv_raw[0] is None:
            action = ALE_FIRE if not s.ball_in_play else ALE_NOOP
        else:
            sv, raw = sv_raw
            paddle_x84 = float(raw[4])
            skill = select_skill_breakout(raw)
            action = select_action_breakout(
                donors[skill], sv, skill,
                paddle_x84=paddle_x84, fallback_action=ALE_NOOP,
            )
        obs, r, term, trunc, _ = env.step(action)
        ret += float(r)
        n_steps += 1
        if term or trunc:
            break
    env.close()
    video = record_dir / f"{record_name}-episode-0.mp4"
    return {"return": ret, "length": n_steps, "video_path": str(video)}


def play_match(organism, *, game: str, seed: int,
               record_dir: Path, record_name: str) -> dict:
    """Dispatch by game name. Returns dict with return/length/video_path."""
    record_dir.mkdir(parents=True, exist_ok=True)
    if game == "Pong":
        return play_pong(organism, seed=seed,
                         record_dir=record_dir, record_name=record_name)
    if game == "Breakout":
        return play_breakout(organism, seed=seed,
                             record_dir=record_dir, record_name=record_name)
    raise ValueError(f"unknown game {game!r}; use 'Pong' or 'Breakout'")
