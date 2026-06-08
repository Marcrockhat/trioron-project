"""Evaluate an organism on ALE and record an MP4 of one episode.

Used for the three "display" arms — the user-visible artifact is the
video of the trioron-controlled paddle.

The video uses the unwrapped (210, 160, 3) RGB frames so the result
is recognisable Atari, not the 84x84 preprocessed feed. The
organism still sees the preprocessed feed for action selection.

True-score eval: terminal_on_life_loss=False so a Breakout episode
runs through all 5 lives and the printed return is the real game
score, not per-life.
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

from .env import (
    FRAME_HW, FRAME_STACK, OBS_DIM, GAME_ACTION_MASK, N_ACTIONS,
)
from .rollout import _select_action

gym.register_envs(ale_py)


class _FlatRGBObsWrapper(gym.ObservationWrapper):
    """Like FlatObsWrapper but RGB-rendered video is preserved by the
    underlying RecordVideo wrapper — we only flatten the *observation*
    feed the agent sees, not the rendered frames."""
    def __init__(self, env):
        super().__init__(env)
        low = np.zeros(OBS_DIM, dtype=np.float32)
        high = np.ones(OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high,
                                                dtype=np.float32)

    def observation(self, obs):
        return (obs.astype(np.float32) / 255.0).reshape(-1)


def _resolve_env_id(game: str) -> str:
    return game if "/" in game else f"ALE/{game}-v5"


def evaluate_and_record(
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
    """Play one episode in `game` driven by the organism at
    `organism_path`. Save MP4 to {out_dir}/{name}-episode-0.mp4.

    Returns:
        {"return": episode_return, "length": steps,
         "video_path": path_to_mp4}
    """
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
        terminal_on_life_loss=False,           # true-score eval
        grayscale_obs=True, grayscale_newaxis=False, scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    env = _FlatRGBObsWrapper(env)

    organism = load_organism(organism_path)
    game_short = game.split("/")[-1].split("-")[0]
    mask = GAME_ACTION_MASK[game_short]
    rng = np.random.default_rng(seed)

    obs, _info = env.reset(seed=seed)
    ret = 0.0
    n_steps = 0
    t0 = time.time()
    for _t in range(max_steps):
        obs_t = torch.from_numpy(obs).unsqueeze(0).float()
        a = _select_action(organism, obs_t, mask, eps, rng)
        obs, r, term, trunc, _info = env.step(a)
        ret += float(r)
        n_steps += 1
        if term or trunc:
            break
    env.close()
    elapsed = time.time() - t0

    # RecordVideo writes to "{name}-episode-0.mp4" by default.
    video_path = out_dir / f"{name}-episode-0.mp4"
    if verbose:
        v_status = "OK" if video_path.exists() else "(missing — check log)"
        print(f"[eval] {game} return={ret:+.1f} length={n_steps} "
              f"({elapsed:.1f}s) → {video_path} {v_status}")
    return {
        "return": ret,
        "length": n_steps,
        "video_path": str(video_path),
        "wallclock": elapsed,
    }
