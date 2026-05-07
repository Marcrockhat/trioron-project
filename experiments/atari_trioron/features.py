"""Privileged-information state extraction for Pong + Breakout.

Reads the underlying ALE 128-byte RAM directly via
``env.unwrapped.ale.getRAM()`` and pulls out paddle / ball positions
at well-documented offsets (cross-checked against the AtariARI
project's RAM-annotation tables).

Important honesty distinction:
  - This is **labeling-time** information. The skill labeler uses
    paddle/ball coordinates to assign a deterministic action to each
    frame.
  - The trioron policy itself **never sees the RAM**. It sees the
    standard 84×84×4 grayscale framestack — same input as a vanilla
    DQN agent. The RAM is privileged information used only at data-
    generation time, like a teacher demonstrator with X-ray vision
    showing what to imitate.

This split is the cleanest version of "expert demonstrator + clean
substrate test": the labels are noiseless, but the policy still
learns from raw pixels.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class BreakoutState:
    paddle_x: int
    ball_x: int
    ball_y: int
    blocks_left: int
    score: int
    lives: int
    ball_in_play: bool

    @property
    def is_serve(self) -> bool:
        """True when the ball hasn't been launched yet (fresh life or
        episode start). RAM values for ball_x/ball_y are both 0 in
        this state — the launcher resets them to (0, 0) until FIRE
        kicks the ball into play."""
        return not self.ball_in_play


@dataclass
class PongState:
    my_paddle_y: int
    opp_paddle_y: int
    ball_x: int
    ball_y: int
    my_score: int
    opp_score: int
    ball_in_play: bool

    @property
    def is_serve(self) -> bool:
        return not self.ball_in_play


def extract_breakout(ale) -> BreakoutState:
    """ALE Breakout RAM offsets:
        72:  paddle_x        (0..160)
        99:  ball_x          (0 when not in play, ~ 0..160 otherwise)
        101: ball_y          (0 when not in play, ~ 80..210 otherwise)
        77:  score           (BCD-coded; treat as raw 0..99)
        57:  lives           (typically 5)
        77,76,75:  packed score digits across 3 bytes (LSD ordered)
    """
    ram = ale.getRAM()
    paddle_x = int(ram[72])
    ball_x = int(ram[99])
    ball_y = int(ram[101])
    blocks_left = 0  # not consistently single-byte; skip
    score = int(ram[77])
    lives = int(ram[57])
    # ball_in_play: nonzero ball_y means ball has been launched.
    # Right after a brick hit ball_y can briefly read 0; treat that
    # as in-play for stability by also gating on a recent nonzero
    # score field, but the simple heuristic is fine for skill rules.
    ball_in_play = (ball_x != 0 or ball_y != 0)
    return BreakoutState(
        paddle_x=paddle_x, ball_x=ball_x, ball_y=ball_y,
        blocks_left=blocks_left, score=score, lives=lives,
        ball_in_play=ball_in_play,
    )


def extract_pong(ale) -> PongState:
    """ALE Pong RAM offsets:
        51:  my paddle_y     (right paddle, agent-controlled)
        50:  opp paddle_y    (left paddle, scripted opponent)
        49:  ball_x          (0 when not in play)
        54:  ball_y          (0 when not in play)
        13:  my score        (right side)
        14:  opp score       (left side)
    """
    ram = ale.getRAM()
    my_y = int(ram[51])
    opp_y = int(ram[50])
    ball_x = int(ram[49])
    ball_y = int(ram[54])
    my_score = int(ram[13])
    opp_score = int(ram[14])
    ball_in_play = (ball_x != 0 or ball_y != 0)
    return PongState(
        my_paddle_y=my_y, opp_paddle_y=opp_y,
        ball_x=ball_x, ball_y=ball_y,
        my_score=my_score, opp_score=opp_score,
        ball_in_play=ball_in_play,
    )


def get_ale(env):
    """Walk through env wrapper layers to find the underlying ALE."""
    inner = env
    while hasattr(inner, "env") or hasattr(inner, "unwrapped"):
        if hasattr(inner, "ale"):
            return inner.ale
        if hasattr(inner, "unwrapped") and inner.unwrapped is not inner:
            inner = inner.unwrapped
        elif hasattr(inner, "env"):
            inner = inner.env
        else:
            break
    if hasattr(inner, "ale"):
        return inner.ale
    raise RuntimeError(f"could not find ALE on {type(env).__name__}")
