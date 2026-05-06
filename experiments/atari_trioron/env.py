"""ALE → trioron-input wrapper.

Standard Atari preprocessing (84×84 grayscale, max-skip-4, frame-
stack-4, episode-on-life-loss for Breakout) but emits a flat
float32 vector in [0,1] for trioron consumption.

We use Pong's full 6-action space as the union: {NOOP, FIRE, RIGHT,
LEFT, RIGHTFIRE, LEFTFIRE}. Breakout's native 4 actions are the
prefix; the trailing 2 are masked to -inf at action-selection time
on Breakout. This keeps a single trioron output head usable across
both games without an embedding-space re-mapping.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import torch

import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)


# Standard Atari preprocessing constants.
FRAME_HW = 84
FRAME_STACK = 4
OBS_DIM = FRAME_HW * FRAME_HW * FRAME_STACK   # 28224

# Action-space union = Pong's full set. Breakout's native 4-action set
# is the prefix; the trailing 2 are masked in BREAKOUT_MASK below.
N_ACTIONS = 6
PONG_MASK = np.array([1, 1, 1, 1, 1, 1], dtype=bool)
BREAKOUT_MASK = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
GAME_ACTION_MASK = {
    "Pong": PONG_MASK,
    "Breakout": BREAKOUT_MASK,
}


class FlatObsWrapper(gym.ObservationWrapper):
    """Flatten the (4, 84, 84) uint8 framestack into a (28224,) float32
    vector in [0,1]. Trioron's L0 expects a 2D tensor (N, input_dim)
    of float32; this is the per-step row that builds into one."""
    def __init__(self, env):
        super().__init__(env)
        low = np.zeros(OBS_DIM, dtype=np.float32)
        high = np.ones(OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high,
                                                dtype=np.float32)

    def observation(self, obs):
        # obs: (FRAME_STACK, FRAME_HW, FRAME_HW) uint8 -> (OBS_DIM,) float32
        return (obs.astype(np.float32) / 255.0).reshape(-1)


def _resolve_env_id(game: str) -> str:
    """Accept short names ('Pong', 'Breakout') or full IDs."""
    if "/" in game:
        return game
    return f"ALE/{game}-v5"


def make_env(
    game: str,
    seed: int = 0,
    render_mode: str | None = None,
    terminal_on_life_loss: bool = True,
    full_action_space: bool = False,
) -> gym.Env:
    """Returns a gym Env emitting (28224,) float32 obs in [0,1].

    Args:
        game: "Pong", "Breakout", or a full ALE/* env id.
        seed: passed to env.reset(seed=...).
        render_mode: "rgb_array" for video recording; None for headless.
        terminal_on_life_loss: standard Atari trick — treat life loss
            as episode end during training so the agent learns each
            life is precious. False for eval (true scoring requires
            full-episode returns).
        full_action_space: when True, expose the full 18-action ALE
            space (useful for running a model trained with one game's
            6-action space against another). Default False so each
            game's native action space is preserved (Pong=6, Breakout=4).
    """
    env_id = _resolve_env_id(game)
    env = gym.make(
        env_id, render_mode=render_mode,
        frameskip=1,                # AtariPreprocessing handles the skip
        repeat_action_probability=0.0,
        full_action_space=full_action_space,
    )
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=FRAME_HW,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,            # we do float32 + /255 ourselves below
    )
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    env = FlatObsWrapper(env)
    env.reset(seed=seed)
    return env


def obs_to_tensor(obs: np.ndarray) -> torch.Tensor:
    """One-row tensor (1, OBS_DIM). Used at action-selection time."""
    return torch.from_numpy(obs).unsqueeze(0).float()
