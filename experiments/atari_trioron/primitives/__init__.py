"""Pong primitive vocabulary — synthetic data generation + clustering probe.

See docs/atari_pong_primitives.md for the vocabulary design and class-ID
layout. This package owns:

  synthetic_env.py — frame generators per primitive class + clustering probe.
"""
from .synthetic_env import (
    STATE_DIM, FRAME_HW, DEFAULT_L0_DIM,
    ALL_CLASSES, CLASS_NAMES,
    BALL_HIGH, BALL_MID, BALL_LOW,
    BALL_LEFT, BALL_CENTER, BALL_RIGHT,
    PADDLE_HIGH, PADDLE_MID, PADDLE_LOW,
    BALL_GOING_UP, BALL_GOING_DOWN, BALL_GOING_LEFT, BALL_GOING_RIGHT,
    BALL_FAST, BALL_SLOW,
    BALL_ABOVE_PADDLE, BALL_ALIGNED_WITH_PADDLE, BALL_BELOW_PADDLE,
    BALL_APPROACHING_PADDLE, BALL_RECEDING_FROM_PADDLE,
    generate_dataset,
    clustering_probe,
    standardize,
    STANDARDIZE,
    FIELDS,
)

__all__ = [
    "STATE_DIM", "FRAME_HW", "DEFAULT_L0_DIM",
    "ALL_CLASSES", "CLASS_NAMES",
    "BALL_HIGH", "BALL_MID", "BALL_LOW",
    "BALL_LEFT", "BALL_CENTER", "BALL_RIGHT",
    "PADDLE_HIGH", "PADDLE_MID", "PADDLE_LOW",
    "BALL_GOING_UP", "BALL_GOING_DOWN", "BALL_GOING_LEFT", "BALL_GOING_RIGHT",
    "BALL_FAST", "BALL_SLOW",
    "BALL_ABOVE_PADDLE", "BALL_ALIGNED_WITH_PADDLE", "BALL_BELOW_PADDLE",
    "BALL_APPROACHING_PADDLE", "BALL_RECEDING_FROM_PADDLE",
    "generate_dataset", "clustering_probe",
    "standardize", "STANDARDIZE", "FIELDS",
]
