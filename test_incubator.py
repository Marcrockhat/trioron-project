"""Self-contained tests for trioron.incubator.

Run with:    python3 test_incubator.py
"""
from __future__ import annotations
import sys
import traceback
import torch

from trioron.incubator import (
    STATE_DIM,
    ACTION_DIM,
    DIM_SATIETY,
    DIM_TEMPERATURE,
    DIM_THREAT,
    DIM_POSITION_X,
    DIM_TARGET_X,
    DIM_OWNED,
    EnvConfig,
    ScriptedEnvironment,
    ContrastiveCurriculum,
    PAIR_NAMES,
    PAIR_HUNGRY_STUFFED,
    PAIR_COLD_HOT,
    PAIR_THREAT_SAFE,
    PAIR_REACHABLE_UNREACHABLE,
    PAIR_OWNED_FOREIGN,
    contrastive_loss,
)


_RESULTS: list[tuple[str, bool, str]] = []


def _run(name, fn):
    try:
        fn()
        _RESULTS.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as e:
        _RESULTS.append((name, False, str(e)))
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc(limit=3)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def test_env_reset_shape_and_range():
    env = ScriptedEnvironment(EnvConfig(seed=0))
    s = env.reset()
    assert s.shape == (STATE_DIM,), f"got {tuple(s.shape)}"
    assert (s >= 0).all() and (s <= 1).all(), "state out of [0,1]"
    assert s[DIM_OWNED].item() in (0.0, 1.0), f"owned not binary: {s[DIM_OWNED].item()}"


def test_env_step_shape_and_range():
    env = ScriptedEnvironment(EnvConfig(seed=1))
    env.reset()
    action = torch.zeros(ACTION_DIM)
    s, r, info = env.step(action)
    assert s.shape == (STATE_DIM,)
    assert (s >= 0).all() and (s <= 1).all()
    assert isinstance(r, float)
    assert info["t"] == 1


def test_env_step_reacts_to_action():
    env = ScriptedEnvironment(EnvConfig(seed=2, noise_std=0.0))
    s0 = env.reset()
    a = torch.zeros(ACTION_DIM)
    a[DIM_SATIETY] = 0.5  # large positive delta
    s1, _, _ = env.step(a)
    # Satiety should rise, even after the hunger drift subtracts a tiny bit.
    assert s1[DIM_SATIETY] > s0[DIM_SATIETY], (
        f"satiety did not rise: {s0[DIM_SATIETY].item()} -> {s1[DIM_SATIETY].item()}"
    )


def test_env_action_wrong_shape_raises():
    env = ScriptedEnvironment(EnvConfig(seed=3))
    env.reset()
    try:
        env.step(torch.zeros(ACTION_DIM + 1))
    except ValueError:
        return
    raise AssertionError("expected ValueError on wrong action shape")


def test_env_seed_reproducible():
    env_a = ScriptedEnvironment(EnvConfig(seed=42))
    env_b = ScriptedEnvironment(EnvConfig(seed=42))
    sa = env_a.reset()
    sb = env_b.reset()
    assert torch.allclose(sa, sb), "reset not reproducible under same seed"
    a = torch.randn(ACTION_DIM)
    s_a1, _, _ = env_a.step(a)
    s_b1, _, _ = env_b.step(a)
    assert torch.allclose(s_a1, s_b1), "step not reproducible under same seed"


# --------------------------------------------------------------------------- #
# Curriculum
# --------------------------------------------------------------------------- #


def test_curriculum_pair_shapes():
    cur = ContrastiveCurriculum(seed=0)
    a, b = cur.sample_pair(PAIR_HUNGRY_STUFFED, batch=7)
    assert a.shape == (7, STATE_DIM) and b.shape == (7, STATE_DIM)


def test_curriculum_hungry_stuffed_separates():
    cur = ContrastiveCurriculum(seed=0)
    a, b = cur.sample_pair(PAIR_HUNGRY_STUFFED, batch=64)
    assert (a[:, DIM_SATIETY] < b[:, DIM_SATIETY]).all(), "hungry side should have lower satiety"


def test_curriculum_cold_hot_separates():
    cur = ContrastiveCurriculum(seed=0)
    a, b = cur.sample_pair(PAIR_COLD_HOT, batch=64)
    assert (a[:, DIM_TEMPERATURE] < b[:, DIM_TEMPERATURE]).all()


def test_curriculum_threat_safe_separates():
    cur = ContrastiveCurriculum(seed=0)
    a, b = cur.sample_pair(PAIR_THREAT_SAFE, batch=64)
    # a is the threat side (high), b is safe (low).
    assert (a[:, DIM_THREAT] > b[:, DIM_THREAT]).all()


def test_curriculum_reachable_unreachable_separates():
    cur = ContrastiveCurriculum(seed=0)
    a, b = cur.sample_pair(PAIR_REACHABLE_UNREACHABLE, batch=64)
    d_a = (a[:, DIM_POSITION_X] - a[:, DIM_TARGET_X]).abs()
    d_b = (b[:, DIM_POSITION_X] - b[:, DIM_TARGET_X]).abs()
    assert (d_a < d_b).all(), "reachable side should have smaller pos-target distance"


def test_curriculum_owned_foreign_separates():
    cur = ContrastiveCurriculum(seed=0)
    a, b = cur.sample_pair(PAIR_OWNED_FOREIGN, batch=32)
    assert (a[:, DIM_OWNED] == 1.0).all()
    assert (b[:, DIM_OWNED] == 0.0).all()


def test_curriculum_sample_all_covers_pairs():
    cur = ContrastiveCurriculum(seed=0)
    items = cur.sample_all(batch_per_pair=4)
    assert len(items) == len(PAIR_NAMES)
    names = [n for (n, _, _) in items]
    assert names == PAIR_NAMES
    for _, a, b in items:
        assert a.shape == (4, STATE_DIM) and b.shape == (4, STATE_DIM)


def test_curriculum_unknown_pair_raises():
    cur = ContrastiveCurriculum(seed=0)
    try:
        cur.sample_pair("not_a_real_pair", batch=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError on unknown pair name")


# --------------------------------------------------------------------------- #
# Contrastive loss
# --------------------------------------------------------------------------- #


def test_contrastive_loss_zero_when_far():
    h_a = torch.zeros(8, 16)
    h_b = torch.zeros(8, 16)
    h_b[:, 0] = 5.0  # distance = 5, margin = 1 → zero loss
    loss = contrastive_loss(h_a, h_b, margin=1.0)
    assert loss.item() == 0.0, f"expected 0, got {loss.item()}"


def test_contrastive_loss_positive_when_close():
    h_a = torch.zeros(8, 16)
    h_b = torch.zeros(8, 16)
    h_b[:, 0] = 0.1  # distance = 0.1, margin = 1 → loss ≈ 0.9
    loss = contrastive_loss(h_a, h_b, margin=1.0)
    assert 0.85 < loss.item() < 0.95, f"expected ≈0.9, got {loss.item()}"


def test_contrastive_loss_has_grad():
    h_a = torch.zeros(4, 8, requires_grad=True)
    h_b = torch.zeros(4, 8, requires_grad=True)
    loss = contrastive_loss(h_a, h_b, margin=1.0)
    loss.backward()
    assert h_a.grad is not None and h_b.grad is not None


def test_contrastive_loss_shape_mismatch_raises():
    try:
        contrastive_loss(torch.zeros(4, 8), torch.zeros(4, 16), margin=1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


# --------------------------------------------------------------------------- #


def main():
    print("Running incubator tests...")
    print(f"  torch version: {torch.__version__}")

    tests = [
        ("env_reset_shape_and_range",                test_env_reset_shape_and_range),
        ("env_step_shape_and_range",                 test_env_step_shape_and_range),
        ("env_step_reacts_to_action",                test_env_step_reacts_to_action),
        ("env_action_wrong_shape_raises",            test_env_action_wrong_shape_raises),
        ("env_seed_reproducible",                    test_env_seed_reproducible),
        ("curriculum_pair_shapes",                   test_curriculum_pair_shapes),
        ("curriculum_hungry_stuffed_separates",      test_curriculum_hungry_stuffed_separates),
        ("curriculum_cold_hot_separates",            test_curriculum_cold_hot_separates),
        ("curriculum_threat_safe_separates",         test_curriculum_threat_safe_separates),
        ("curriculum_reachable_unreachable_separates", test_curriculum_reachable_unreachable_separates),
        ("curriculum_owned_foreign_separates",       test_curriculum_owned_foreign_separates),
        ("curriculum_sample_all_covers_pairs",       test_curriculum_sample_all_covers_pairs),
        ("curriculum_unknown_pair_raises",           test_curriculum_unknown_pair_raises),
        ("contrastive_loss_zero_when_far",           test_contrastive_loss_zero_when_far),
        ("contrastive_loss_positive_when_close",     test_contrastive_loss_positive_when_close),
        ("contrastive_loss_has_grad",                test_contrastive_loss_has_grad),
        ("contrastive_loss_shape_mismatch_raises",   test_contrastive_loss_shape_mismatch_raises),
    ]

    for name, fn in tests:
        _run(name, fn)

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print()
    print(f"  Total: {len(_RESULTS)}    PASS: {n_pass}    FAIL: {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
