"""Trioron self-imitation on Atari (Pong, Breakout).

Architecture:
  - The trioron substrate IS the policy. No critic, no advantage,
    no replay buffer — just rollout → return-filter → api.build_donor
    / api.extend on the filtered (state, action) tuples. The
    frustration → growth → consolidation cycle that drives trioron's
    classification benches drives RL too, when paired with reward-
    filtered self-imitation data.

Three arms (Rocky's spec):
  1. Breakout learning only — cold-start trioron, self-imitation on
     Breakout until the return-filter plateaus.
  2. Breakout after Pong — train Pong first, api.extend into a
     Breakout self-imitation loop.
  3. Pong-only, eval on Breakout — measures pure feature transfer
     through the manifold archive's routing.
"""
from .env import make_env, OBS_DIM, N_ACTIONS, GAME_ACTION_MASK  # noqa: F401
