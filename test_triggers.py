"""Self-contained tests for trioron.triggers.

Run with:    python3 test_triggers.py
"""
from __future__ import annotations
import sys
import traceback
import torch

from trioron.triggers import (
    GrowthTrigger,
    effective_rank,
    total_gradient_norm,
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
# effective_rank
# --------------------------------------------------------------------------- #


def test_effective_rank_orthonormal_is_full():
    # Orthonormal columns => all singular values equal => rank_eff ≈ d.
    d = 4
    Q, _ = torch.linalg.qr(torch.randn(64, d))
    H = Q  # (64, 4) with orthonormal columns
    r = effective_rank(H)
    assert abs(r - d) < 0.01, f"expected ~{d}, got {r}"


def test_effective_rank_rank_one_is_one():
    # All rows are scaled copies of the same vector -> rank 1.
    base = torch.randn(8)
    scales = torch.linspace(0.1, 1.0, 32).unsqueeze(1)
    H = scales * base.unsqueeze(0)
    r = effective_rank(H)
    assert abs(r - 1.0) < 0.01, f"expected ~1, got {r}"


def test_effective_rank_rejects_non_2d():
    try:
        effective_rank(torch.randn(4, 4, 4))
    except ValueError:
        return
    raise AssertionError("expected ValueError on 3D input")


# --------------------------------------------------------------------------- #
# total_gradient_norm
# --------------------------------------------------------------------------- #


def test_total_gradient_norm_matches_manual():
    p1 = torch.randn(3, 4, requires_grad=True)
    p2 = torch.randn(2, requires_grad=True)
    loss = (p1 ** 2).sum() + (p2 ** 2).sum()
    loss.backward()
    g = total_gradient_norm([p1, p2])
    expected = (p1.grad.pow(2).sum() + p2.grad.pow(2).sum()).sqrt().item()
    assert abs(g - expected) < 1e-5, f"{g} vs {expected}"


def test_total_gradient_norm_skips_no_grad():
    # Should not raise on parameters with no .grad set.
    p = torch.randn(4, requires_grad=True)
    g = total_gradient_norm([p])
    assert g == 0.0


# --------------------------------------------------------------------------- #
# GrowthTrigger — warmup behavior
# --------------------------------------------------------------------------- #


def test_trigger_warmup_blocks_fire():
    trig = GrowthTrigger(latent_dim=2, window=10)
    H = torch.eye(8, 2)  # rank 2 — would otherwise saturate
    for _ in range(15):  # only 15 < 2*window=20 observations
        s = trig.observe(loss=0.05, hidden=H, grad_norm=0.5)
    assert s.warmup, "should still be in warmup before 2*W observations"
    assert not s.fire, "must not fire during warmup"


def test_trigger_clears_warmup_after_2W():
    trig = GrowthTrigger(latent_dim=2, window=10)
    H = torch.eye(8, 2)
    for _ in range(20):
        s = trig.observe(loss=0.05, hidden=H, grad_norm=0.5)
    assert not s.warmup, "should have left warmup after 2*W observations"


# --------------------------------------------------------------------------- #
# GrowthTrigger — independent per-condition behavior
# --------------------------------------------------------------------------- #


def test_loss_plateau_true_when_flat():
    trig = GrowthTrigger(latent_dim=2, window=10, eps_loss=1e-3)
    H = torch.randn(8, 2)
    for _ in range(20):
        trig.observe(loss=0.05, hidden=H, grad_norm=0.5)
    s = trig.observe(loss=0.05, hidden=H, grad_norm=0.5)
    assert s.loss_plateau, f"flat loss should plateau; improvement={s.loss_improvement}"


def test_loss_plateau_false_when_decreasing():
    trig = GrowthTrigger(latent_dim=2, window=10, eps_loss=1e-3)
    H = torch.randn(8, 2)
    # First W steps: high loss. Next W: lower loss. Improvement >> ε_loss.
    for _ in range(10):
        trig.observe(loss=1.0, hidden=H, grad_norm=0.5)
    for _ in range(10):
        s = trig.observe(loss=0.1, hidden=H, grad_norm=0.5)
    assert not s.loss_plateau, f"decreasing loss should NOT plateau; improvement={s.loss_improvement}"


def test_rank_saturated_at_full_dim():
    trig = GrowthTrigger(latent_dim=2, window=10, eps_rank=0.05)
    # Rank-2 matrix saturates dim=2 by construction.
    Q, _ = torch.linalg.qr(torch.randn(16, 2))
    H = Q  # (16, 2), orthonormal cols => rank_eff ~ 2
    for _ in range(20):
        s = trig.observe(loss=0.05, hidden=H, grad_norm=0.5)
    assert s.rank_saturated, f"rank should saturate; recent_mean={s.rank_recent_mean}"


def test_rank_not_saturated_when_rank_one():
    trig = GrowthTrigger(latent_dim=2, window=10, eps_rank=0.05)
    # Rank-1 matrix in d=2 latent => rank_eff ~ 1, gap = 1 >> 0.05.
    base = torch.randn(8)[:2]
    H = torch.linspace(0.1, 1.0, 16).unsqueeze(1) * base.unsqueeze(0)
    for _ in range(20):
        s = trig.observe(loss=0.05, hidden=H, grad_norm=0.5)
    assert not s.rank_saturated, f"rank-1 H should not saturate dim 2; mean={s.rank_recent_mean}"


def test_grad_stable_in_range():
    trig = GrowthTrigger(latent_dim=2, window=10, g_min=0.1, g_max=10.0)
    H = torch.randn(8, 2)
    for _ in range(20):
        s = trig.observe(loss=0.05, hidden=H, grad_norm=1.0)
    assert s.grad_stable


def test_grad_unstable_when_vanishing():
    trig = GrowthTrigger(latent_dim=2, window=10, g_min=0.1, g_max=10.0)
    H = torch.randn(8, 2)
    for _ in range(20):
        s = trig.observe(loss=0.05, hidden=H, grad_norm=1e-6)
    assert not s.grad_stable


def test_grad_unstable_when_exploding():
    trig = GrowthTrigger(latent_dim=2, window=10, g_min=0.1, g_max=10.0)
    H = torch.randn(8, 2)
    for _ in range(20):
        s = trig.observe(loss=0.05, hidden=H, grad_norm=1000.0)
    assert not s.grad_stable


# --------------------------------------------------------------------------- #
# GrowthTrigger — conjunction
# --------------------------------------------------------------------------- #


def test_fire_only_on_full_conjunction():
    trig = GrowthTrigger(latent_dim=2, window=10, eps_loss=1e-3, eps_rank=0.05)
    Q, _ = torch.linalg.qr(torch.randn(16, 2))
    H = Q
    for _ in range(20):
        s = trig.observe(loss=0.05, hidden=H, grad_norm=1.0)
    assert s.loss_plateau and s.rank_saturated and s.grad_stable
    assert s.fire


def test_fire_blocked_by_grad_pathology():
    trig = GrowthTrigger(latent_dim=2, window=10, eps_loss=1e-3, eps_rank=0.05)
    Q, _ = torch.linalg.qr(torch.randn(16, 2))
    H = Q
    for _ in range(20):
        s = trig.observe(loss=0.05, hidden=H, grad_norm=1e-6)  # vanishing grads
    assert s.loss_plateau and s.rank_saturated
    assert not s.grad_stable
    assert not s.fire, "grad instability must veto growth"


def test_fire_blocked_by_unsaturated_rank():
    trig = GrowthTrigger(latent_dim=2, window=10, eps_loss=1e-3, eps_rank=0.05)
    base = torch.randn(8)[:2]
    H = torch.linspace(0.1, 1.0, 16).unsqueeze(1) * base.unsqueeze(0)
    for _ in range(20):
        s = trig.observe(loss=0.05, hidden=H, grad_norm=1.0)
    assert s.loss_plateau and s.grad_stable
    assert not s.rank_saturated
    assert not s.fire


# --------------------------------------------------------------------------- #
# GrowthTrigger — reset / set_latent_dim
# --------------------------------------------------------------------------- #


def test_reset_clears_warmup():
    trig = GrowthTrigger(latent_dim=2, window=10)
    H = torch.eye(8, 2)
    for _ in range(20):
        trig.observe(loss=0.05, hidden=H, grad_norm=0.5)
    trig.reset()
    s = trig.observe(loss=0.05, hidden=H, grad_norm=0.5)
    assert s.warmup, "reset should put trigger back into warmup"


def test_set_latent_dim_changes_saturation_target():
    trig = GrowthTrigger(latent_dim=2, window=10, eps_rank=0.05)
    Q, _ = torch.linalg.qr(torch.randn(16, 2))
    H = Q  # rank 2
    for _ in range(20):
        s = trig.observe(loss=0.05, hidden=H, grad_norm=1.0)
    assert s.rank_saturated  # saturates dim=2
    # Now bump target to 3; same H is no longer saturated relative to d=3.
    trig.set_latent_dim(3)
    # We need the running stats to reflect the new dim, but the rank
    # window contents (rank values) don't change; only the comparison does.
    s2 = trig.observe(loss=0.05, hidden=H, grad_norm=1.0)
    assert not s2.rank_saturated, "saturation must be relative to current latent_dim"


# --------------------------------------------------------------------------- #


def main():
    print("Running trigger tests...")
    print(f"  torch version: {torch.__version__}")

    tests = [
        ("effective_rank_orthonormal_is_full", test_effective_rank_orthonormal_is_full),
        ("effective_rank_rank_one_is_one",     test_effective_rank_rank_one_is_one),
        ("effective_rank_rejects_non_2d",      test_effective_rank_rejects_non_2d),
        ("total_gradient_norm_matches_manual", test_total_gradient_norm_matches_manual),
        ("total_gradient_norm_skips_no_grad",  test_total_gradient_norm_skips_no_grad),
        ("trigger_warmup_blocks_fire",         test_trigger_warmup_blocks_fire),
        ("trigger_clears_warmup_after_2W",     test_trigger_clears_warmup_after_2W),
        ("loss_plateau_true_when_flat",        test_loss_plateau_true_when_flat),
        ("loss_plateau_false_when_decreasing", test_loss_plateau_false_when_decreasing),
        ("rank_saturated_at_full_dim",         test_rank_saturated_at_full_dim),
        ("rank_not_saturated_when_rank_one",   test_rank_not_saturated_when_rank_one),
        ("grad_stable_in_range",               test_grad_stable_in_range),
        ("grad_unstable_when_vanishing",       test_grad_unstable_when_vanishing),
        ("grad_unstable_when_exploding",       test_grad_unstable_when_exploding),
        ("fire_only_on_full_conjunction",      test_fire_only_on_full_conjunction),
        ("fire_blocked_by_grad_pathology",     test_fire_blocked_by_grad_pathology),
        ("fire_blocked_by_unsaturated_rank",   test_fire_blocked_by_unsaturated_rank),
        ("reset_clears_warmup",                test_reset_clears_warmup),
        ("set_latent_dim_changes_saturation_target", test_set_latent_dim_changes_saturation_target),
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
