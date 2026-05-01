"""Self-contained tests for trioron.frustration.

Run with:    python test_frustration.py

Verifies the per-pair plateau-counter contract:
  - Multiplier is 1.0 during warmup (no closed window) and below threshold.
  - Stuck counter increments only when window-mean improvement < eps_loss.
  - Stuck counter resets when a window shows real improvement.
  - Multiplier ramps as 1 + gain * (stuck - threshold + 1), capped at max_mult.
  - Per-pair state is isolated (one pair's stuck count doesn't bleed into another).
  - reset_pair / reset_all clear state correctly.
  - peak_stuck / total_boosted_windows / boosted_pairs report correctly.
"""
from __future__ import annotations
import sys
import traceback

from trioron.frustration import FrustrationTracker


_RESULTS: list[tuple[str, bool, str]] = []


def _run(name: str, fn) -> None:
    try:
        fn()
        _RESULTS.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as e:
        _RESULTS.append((name, False, str(e)))
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc(limit=3)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_warmup_returns_one():
    f = FrustrationTracker(window=10, threshold=1, gain=1.0, max_mult=4.0)
    for _ in range(9):
        assert f.observe("p1", 1.0) == 1.0
    # 10th observation closes the first window but stuck stays 0
    # (no previous window to compare against yet).
    assert f.observe("p1", 1.0) == 1.0


def test_two_flat_windows_increments_stuck():
    f = FrustrationTracker(window=10, threshold=1, eps_loss=0.001,
                           gain=1.0, max_mult=4.0)
    for _ in range(10):
        f.observe("p1", 1.0)
    # First window closed; prev_mean recorded but no stuck increment.
    assert f.stuck_count("p1") == 0
    for _ in range(10):
        f.observe("p1", 1.0)
    # Second window closed: improvement = 0 < eps → stuck → 1
    assert f.stuck_count("p1") == 1
    # threshold=1 → multiplier is 1 + gain*1 = 2.0
    assert f.multiplier("p1") == 2.0


def test_real_improvement_resets_stuck():
    f = FrustrationTracker(window=10, threshold=1, eps_loss=0.001,
                           gain=1.0, max_mult=4.0)
    # Two flat windows → stuck=1
    for _ in range(20):
        f.observe("p1", 1.0)
    assert f.stuck_count("p1") == 1
    # Now 10 losses much lower than the previous window mean.
    for _ in range(10):
        f.observe("p1", 0.1)
    # Improvement = 1.0 - 0.1 = 0.9 ≥ eps → stuck resets to 0
    assert f.stuck_count("p1") == 0
    assert f.multiplier("p1") == 1.0


def test_ramp_clamps_at_max_mult():
    f = FrustrationTracker(window=4, threshold=1, eps_loss=0.001,
                           gain=1.0, max_mult=2.5)
    # Drive many flat windows
    for _ in range(40):  # 10 closed windows
        f.observe("p1", 1.0)
    # 10 windows; first establishes prev_mean, next 9 increment stuck.
    assert f.stuck_count("p1") == 9
    # Without cap: mult = 1 + 1 * (9 - 1 + 1) = 10. With cap 2.5: 2.5.
    assert f.multiplier("p1") == 2.5


def test_threshold_gates_multiplier():
    # threshold=3 means stuck must be >= 3 before multiplier > 1.
    f = FrustrationTracker(window=2, threshold=3, eps_loss=0.001,
                           gain=1.0, max_mult=10.0)
    # Window 1: prev = 1.0
    f.observe("p1", 1.0); f.observe("p1", 1.0)
    # Window 2: stuck → 1
    f.observe("p1", 1.0); f.observe("p1", 1.0)
    assert f.stuck_count("p1") == 1
    assert f.multiplier("p1") == 1.0
    # Window 3: stuck → 2, still below threshold
    f.observe("p1", 1.0); f.observe("p1", 1.0)
    assert f.stuck_count("p1") == 2
    assert f.multiplier("p1") == 1.0
    # Window 4: stuck → 3, multiplier = 1 + (3-3+1) = 2.0
    f.observe("p1", 1.0); f.observe("p1", 1.0)
    assert f.stuck_count("p1") == 3
    assert f.multiplier("p1") == 2.0


def test_per_pair_isolation():
    f = FrustrationTracker(window=10, threshold=1, eps_loss=0.001,
                           gain=1.0, max_mult=4.0)
    for _ in range(20):
        f.observe("stuck_pair", 1.0)
    for _ in range(10):
        f.observe("fresh_pair", 1.0)
    assert f.stuck_count("stuck_pair") == 1
    assert f.multiplier("stuck_pair") == 2.0
    assert f.stuck_count("fresh_pair") == 0
    assert f.multiplier("fresh_pair") == 1.0


def test_reset_pair_clears_only_one_pair():
    f = FrustrationTracker(window=10, threshold=1, eps_loss=0.001,
                           gain=1.0, max_mult=4.0)
    for _ in range(20):
        f.observe("p1", 1.0)
        f.observe("p2", 1.0)
    assert f.stuck_count("p1") == 1
    assert f.stuck_count("p2") == 1
    f.reset_pair("p1")
    assert f.stuck_count("p1") == 0
    assert f.stuck_count("p2") == 1
    assert f.multiplier("p1") == 1.0
    assert f.multiplier("p2") == 2.0


def test_reset_all_clears_everything():
    f = FrustrationTracker(window=10, threshold=1)
    for _ in range(20):
        f.observe("p1", 1.0)
        f.observe("p2", 1.0)
    f.reset_all()
    assert f.stuck_count("p1") == 0
    assert f.stuck_count("p2") == 0
    assert f.peak_stuck("p1") == 0


def test_peak_stuck_persists_across_recovery():
    f = FrustrationTracker(window=4, threshold=1, eps_loss=0.001,
                           gain=1.0, max_mult=10.0)
    # Drive 5 closed windows of plateau (stuck reaches 4)
    for _ in range(20):
        f.observe("p1", 1.0)
    assert f.stuck_count("p1") == 4
    assert f.peak_stuck("p1") == 4
    # Real improvement → stuck resets
    for _ in range(4):
        f.observe("p1", 0.0)
    assert f.stuck_count("p1") == 0
    # peak_stuck remembers we were once at 4
    assert f.peak_stuck("p1") == 4


def test_total_boosted_windows_diagnostic():
    f = FrustrationTracker(window=4, threshold=2, eps_loss=0.001,
                           gain=1.0, max_mult=10.0)
    # Pair p1 reaches stuck=3 → boosted windows = 3-2+1 = 2
    for _ in range(16):
        f.observe("p1", 1.0)
    assert f.peak_stuck("p1") == 3
    assert f.total_boosted_windows() == 2
    # Pair p2 only stuck=1 → never crossed threshold → contributes 0
    for _ in range(8):
        f.observe("p2", 1.0)
    assert f.peak_stuck("p2") == 1
    assert f.total_boosted_windows() == 2  # unchanged


def test_boosted_pairs_returns_only_crossing_threshold():
    f = FrustrationTracker(window=4, threshold=2, eps_loss=0.001,
                           gain=1.0, max_mult=10.0)
    # p1 stuck=2 (crosses threshold)
    for _ in range(12):
        f.observe("p1", 1.0)
    # p2 stuck=1 (doesn't cross)
    for _ in range(8):
        f.observe("p2", 1.0)
    assert f.boosted_pairs() == ["p1"]


def test_invalid_constructor_args_raise():
    try:
        FrustrationTracker(window=1)
        assert False, "expected ValueError for window<2"
    except ValueError:
        pass
    try:
        FrustrationTracker(threshold=0)
        assert False, "expected ValueError for threshold<1"
    except ValueError:
        pass
    try:
        FrustrationTracker(max_mult=0.5)
        assert False, "expected ValueError for max_mult<1"
    except ValueError:
        pass
    try:
        FrustrationTracker(gain=-0.1)
        assert False, "expected ValueError for gain<0"
    except ValueError:
        pass


def test_eps_loss_distinguishes_progress_from_plateau():
    # improvement comfortably ABOVE eps_loss should NOT count as plateau
    # improvement comfortably BELOW eps_loss SHOULD count.
    f = FrustrationTracker(window=4, threshold=1, eps_loss=0.05)
    for _ in range(4):
        f.observe("p1", 1.0)
    for _ in range(4):
        f.observe("p1", 0.5)  # improvement 0.5 ≫ eps → not stuck
    assert f.stuck_count("p1") == 0
    g = FrustrationTracker(window=4, threshold=1, eps_loss=0.05)
    for _ in range(4):
        g.observe("p1", 1.0)
    for _ in range(4):
        g.observe("p1", 0.99)  # improvement 0.01 < eps → stuck
    assert g.stuck_count("p1") == 1


# --------------------------------------------------------------------------- #
# Runner                                                                      #
# --------------------------------------------------------------------------- #


def main() -> int:
    print("test_frustration.py")
    print("-" * 60)
    _run("warmup_returns_one", test_warmup_returns_one)
    _run("two_flat_windows_increments_stuck", test_two_flat_windows_increments_stuck)
    _run("real_improvement_resets_stuck", test_real_improvement_resets_stuck)
    _run("ramp_clamps_at_max_mult", test_ramp_clamps_at_max_mult)
    _run("threshold_gates_multiplier", test_threshold_gates_multiplier)
    _run("per_pair_isolation", test_per_pair_isolation)
    _run("reset_pair_clears_only_one_pair", test_reset_pair_clears_only_one_pair)
    _run("reset_all_clears_everything", test_reset_all_clears_everything)
    _run("peak_stuck_persists_across_recovery", test_peak_stuck_persists_across_recovery)
    _run("total_boosted_windows_diagnostic", test_total_boosted_windows_diagnostic)
    _run("boosted_pairs_returns_only_crossing_threshold",
         test_boosted_pairs_returns_only_crossing_threshold)
    _run("invalid_constructor_args_raise", test_invalid_constructor_args_raise)
    _run("eps_loss_distinguishes_progress_from_plateau",
         test_eps_loss_distinguishes_progress_from_plateau)
    print("-" * 60)
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print(f"  {n_pass} passed, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
