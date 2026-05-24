"""Per-pair plateau counter that scales the contrastive loss when stuck.

The next_session_plan.md spec ("frustration multiplier"): a per-pair
plateau counter that scales the per-pair gradient when loss has been
stuck for N windows. Mechanically equivalent to focal-loss /
hard-example-mining: amplify the signal on examples the optimizer
isn't making progress on.

Important: the multiplier is applied to the *contrastive task loss
only*, not to regularizers (EWC penalty, sparsity loss, etc.). The
intent is to boost the per-pair learning signal, not the anchor pull
or regularization weight. Callers are responsible for splitting their
loss accordingly:

    mult = frustration.observe(pair_name, l_task.item())
    l = mult * l_task + ewc_strength * net.ewc_penalty()
    l.backward()

Granularity is *per-window*, not per-step. The tracker accumulates W
losses for the current pair, then closes the window: it compares this
window's mean to the previous window's mean, and increments a stuck
counter if the improvement is below eps_loss. The multiplier holds
constant within a window and may step up at window boundaries.

Why per-window: a per-step rolling check would re-trigger almost every
step once the rolling window contains a plateau, which doesn't match
the "stuck for N windows" phrasing in the spec.

Plain-task ablation: the tracker has no effect when no pair ever
plateaus, so wiring it in unconditionally with a high threshold is
safe — runs without frustration look identical.
"""

from __future__ import annotations
from typing import Dict, List, Optional


class FrustrationTracker:
    """Per-pair plateau counter with a hinge-then-ramp multiplier.

    Lifecycle, called from the training loop:

        for step in range(n_steps):
            l_task = contrastive_loss(...)
            mult = frustration.observe(pair_name, l_task.item())
            l = mult * l_task + reg_term
            opt.zero_grad(); l.backward(); opt.step()

    `observe` returns the current scalar multiplier for `pair_name`.
    It is 1.0 until the pair's stuck-window count reaches `threshold`,
    then ramps as 1 + gain * (stuck - threshold + 1), capped at
    `max_mult`.

    Per-pair state is keyed by pair_name, so multiple pairs share the
    same tracker without interfering. If a pair is revisited later in
    the curriculum and its stuck state should NOT carry over, call
    `reset_pair(name)` at the task boundary.
    """

    def __init__(
        self,
        window: int = 400,
        threshold: int = 2,
        eps_loss: float = 0.001,
        gain: float = 1.0,
        max_mult: float = 4.0,
    ):
        if window < 2:
            raise ValueError("window must be >= 2")
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        if max_mult < 1.0:
            raise ValueError("max_mult must be >= 1.0")
        if gain < 0.0:
            raise ValueError("gain must be >= 0")

        self.window = int(window)
        self.threshold = int(threshold)
        self.eps_loss = float(eps_loss)
        self.gain = float(gain)
        self.max_mult = float(max_mult)

        self._buffers: Dict[str, List[float]] = {}
        self._prev_means: Dict[str, Optional[float]] = {}
        self._stuck: Dict[str, int] = {}
        self._peak_stuck: Dict[str, int] = {}
        self._windows_seen: Dict[str, int] = {}

    # ----- main step -----

    def observe(self, pair: str, loss: float) -> float:
        buf = self._buffers.setdefault(pair, [])
        if pair not in self._stuck:
            self._stuck[pair] = 0
            self._prev_means[pair] = None
            self._peak_stuck[pair] = 0
            self._windows_seen[pair] = 0

        buf.append(float(loss))
        if len(buf) >= self.window:
            this_mean = sum(buf) / len(buf)
            prev = self._prev_means[pair]
            if prev is not None:
                improvement = prev - this_mean
                if improvement < self.eps_loss:
                    self._stuck[pair] += 1
                    if self._stuck[pair] > self._peak_stuck[pair]:
                        self._peak_stuck[pair] = self._stuck[pair]
                else:
                    self._stuck[pair] = 0
            self._prev_means[pair] = this_mean
            self._windows_seen[pair] += 1
            buf.clear()

        return self.multiplier(pair)

    # ----- queries -----

    def multiplier(self, pair: str) -> float:
        s = self._stuck.get(pair, 0)
        if s < self.threshold:
            return 1.0
        excess = s - self.threshold + 1
        return min(self.max_mult, 1.0 + self.gain * excess)

    def stuck_count(self, pair: str) -> int:
        return self._stuck.get(pair, 0)

    def peak_stuck(self, pair: str) -> int:
        return self._peak_stuck.get(pair, 0)

    def windows_seen(self, pair: str) -> int:
        return self._windows_seen.get(pair, 0)

    def total_boosted_windows(self) -> int:
        """Across all pairs, count how many closed windows had the
        multiplier >1.0 in effect during them. Useful as a diagnostic:
        if this is zero across a whole bench, frustration never
        engaged and any difference vs the no-frustration baseline is
        seed noise."""
        return sum(
            max(0, peak - self.threshold + 1)
            for peak in self._peak_stuck.values()
        )

    def boosted_pairs(self) -> List[str]:
        """Names of pairs that ever crossed the threshold."""
        return [p for p, peak in self._peak_stuck.items() if peak >= self.threshold]

    # ----- lifecycle -----

    def reset_pair(self, pair: str) -> None:
        """Forget all state for one pair. Call at task boundary if a
        pair is being revisited and the previous visit's stuck state
        should not carry into the new visit."""
        self._buffers.pop(pair, None)
        self._prev_means.pop(pair, None)
        self._stuck.pop(pair, None)
        self._peak_stuck.pop(pair, None)
        self._windows_seen.pop(pair, None)

    def reset_all(self) -> None:
        self._buffers.clear()
        self._prev_means.clear()
        self._stuck.clear()
        self._peak_stuck.clear()
        self._windows_seen.clear()

    def __repr__(self) -> str:
        return (
            f"FrustrationTracker(W={self.window}, threshold={self.threshold}, "
            f"eps_loss={self.eps_loss}, gain={self.gain}, "
            f"max_mult={self.max_mult}, "
            f"pairs_tracked={len(self._stuck)}, "
            f"boosted_windows={self.total_boosted_windows()})"
        )
