"""ALE RAM extraction — local copy for the HF Space.

Mirrors `experiments/atari_trioron/features.py`. RAM offsets cross-
checked against AtariARI annotation tables. The ball_in_play
heuristic for Breakout uses ball_y alone (the OR-with-ball_x
variant in earlier versions stuck True after life loss because
ball_x persisted at its last nonzero RAM byte).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class BreakoutState:
    paddle_x: int
    ball_x: int
    ball_y: int
    score: int
    lives: int
    ball_in_play: bool

    @property
    def is_serve(self) -> bool:
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
    ram = ale.getRAM()
    paddle_x = int(ram[72])
    ball_x = int(ram[99])
    ball_y = int(ram[101])
    score = int(ram[77])
    lives = int(ram[57])
    ball_in_play = (ball_y != 0)
    return BreakoutState(
        paddle_x=paddle_x, ball_x=ball_x, ball_y=ball_y,
        score=score, lives=lives, ball_in_play=ball_in_play,
    )


def extract_pong(ale) -> PongState:
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
    """Walk wrappers to find the underlying ALE handle."""
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
