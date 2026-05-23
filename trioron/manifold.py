"""Per-class manifold archive over L0 codes.

The manifold is trioron's in-network rehearsal channel. After a task
trains, the substrate computes the per-class mean and standard
deviation of L0 codes (the frozen perception layer's outputs) on that
task's training data and stashes them in a ``ManifoldStore``. During
subsequent task training, the store samples synthetic L0 codes drawn
from each archived class's diagonal Gaussian and feeds them through
the current L1 + head — providing a rehearsal signal that doesn't
require keeping any pixels on disk.

The archive lives at the L0 layer's output (post-activation, frozen-
perception) for two reasons:

  - L0 is frozen across the curriculum, so its outputs for a given
    image are deterministic and stable. Stored ``(μ, σ)`` taken now
    will be statistically equivalent to L0 outputs drawn at eval-time
    on the same class.
  - Synthetic codes still pass through the CURRENT L1 + head when
    they replay — so cells that spawned after a class was archived
    still participate in its representation. The head learns a
    consistent decision over all classes' representations under the
    live L1.

Storage is tiny: 128 × 4 × 2 bytes per class ≈ 1 KB; a 30-class
curriculum is ~30 KB. The ``Manifold Replay`` paper-grade result
([[manifold_replay_result]] in memory) pegs this at 30 KB on
grown_uncapped_dream — Pareto-dominating hippo K=10, matching K=20
at 1/10 storage.

This module also surfaces ``settle_head_via_manifold``: a short
head-only training pass that re-calibrates the head logits across the
union of archived classes. The standard use case is post-absorption:
after pool-matched absorption combines two donors' cells, the head's
per-class biases drift; a 200-step replay over the combined manifold
re-balances them without touching L1.

``cosine_logits`` is the cosine-similarity head used by the chained-15
benchmarks (no bias term, head rows are class prototypes in L1
feature space) — promoted here so ``settle_head_via_manifold`` can
default to it without depending on an experiments module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .network import TrioronNetwork


COSINE_DEFAULT_TEMPERATURE = 16.0


def cosine_logits(
    l1_features: torch.Tensor,
    head_W: torch.Tensor,
    temperature: float = COSINE_DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """Cosine-similarity logits.

    ``outs[b, k] = τ · cos(l1_features[b], head_W[k])`` — no bias term.
    Each class row of ``head_W`` is its prototype vector in L1 feature
    space; both inputs and prototypes are L2-normalised so the inner
    product gives bounded similarities in [-1, 1] and the temperature
    τ sets the softmax sharpness.

    Why no bias: linear+softmax heads in the streamlined chained-15
    setup (no EWC, fixed-size head, credit-frozen L1) exhibit a
    bias-accumulation pathology under synthetic replay — stored-class
    logits get pumped to large positive bias values during replay
    steps and drown out current-task gradients. The cosine head is
    bounded by construction, so the pathology can't manifest.
    """
    h_norm = F.normalize(l1_features, dim=-1, eps=1e-8)
    w_norm = F.normalize(head_W, dim=-1, eps=1e-8)
    return temperature * (h_norm @ w_norm.t())


# ---------------------------------------------------------------------
# ManifoldStore
# ---------------------------------------------------------------------


@dataclass
class ManifoldStore:
    """Stores per-class diagonal Gaussian (μ, σ) over L0 OUTPUT codes.

    Fields:
      mu_per_class:    class id → mean tensor of shape (n_l0,).
      sigma_per_class: class id → std tensor of shape (n_l0,),
                       clamped at ``sigma_floor`` to avoid degenerate
                       zero-variance dimensions.
      n_l0:            L0 width. Set automatically by ``store_task``;
                       construct with the L0 width when the layer is
                       known (recommended).
      sigma_floor:     Minimum allowed σ per dimension. 1e-3 by
                       default; raise for noisier substrates.
    """

    mu_per_class: Dict[int, torch.Tensor] = field(default_factory=dict)
    sigma_per_class: Dict[int, torch.Tensor] = field(default_factory=dict)
    n_l0: int = 0
    sigma_floor: float = 1e-3

    def has_classes(self) -> bool:
        return len(self.mu_per_class) > 0

    def store_task(
        self,
        net: TrioronNetwork,
        view,
        task_global_classes: Sequence[int],
        batch_size: int = 1024,
    ) -> None:
        """Compute and store per-class (μ, σ) of L0 codes on a task.

        ``view`` is duck-typed: it must expose ``all_examples() ->
        (X, y)`` returning the task's full training set. Adapters for
        explicit tensor inputs can call this method with a small
        wrapper or call :meth:`store_codes` directly.

        Stored ``after`` credit-freezing for symmetry with the rest of
        the chained-15 pipeline, but the call site is up to the
        caller — the store is purely a per-class statistics record.
        """
        x, y = view.all_examples()
        self.store_codes(net, x, y, task_global_classes, batch_size=batch_size)

    def store_codes(
        self,
        net: TrioronNetwork,
        x: torch.Tensor,
        y: torch.Tensor,
        task_global_classes: Sequence[int],
        batch_size: int = 1024,
    ) -> None:
        """Compute (μ, σ) of L0 codes from explicit tensor inputs.

        Lower-level entry point used by store_task; call directly when
        you have ``(X, y)`` tensors in hand rather than a TaskDataView.
        """
        L0 = net.layers[0]
        with torch.no_grad():
            n = x.shape[0]
            feats_chunks: List[torch.Tensor] = []
            label_chunks: List[torch.Tensor] = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                h0 = L0(x[start:end])
                feats_chunks.append(h0.detach().cpu())
                label_chunks.append(y[start:end])
            feats = torch.cat(feats_chunks, dim=0)
            labels = torch.cat(label_chunks, dim=0)
            for c in task_global_classes:
                m = (labels == c)
                if int(m.sum().item()) > 0:
                    cf = feats[m]
                    self.mu_per_class[int(c)] = cf.mean(dim=0)
                    self.sigma_per_class[int(c)] = (
                        cf.std(dim=0, unbiased=False).clamp(
                            min=self.sigma_floor,
                        )
                    )
            self.n_l0 = L0.n_nodes

    def sample_synthetic(
        self,
        n_total: int,
        generator: Optional[torch.Generator] = None,
        noise_scale: float = 1.0,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Draw ``n_total`` synthetic L0 codes, each from a uniformly
        chosen archived class.

        Returns ``(codes, labels)`` or None when no classes are
        archived yet. Codes are clamped at ≥0 to match L0's ReLU
        activation (otherwise a synthetic code can land on the
        non-physical side of the ReLU and produce spurious gradients
        for L1).

        Matches the bench's ``ManifoldBuffer.sample`` pattern: a
        fixed total budget shared uniformly across classes, so each
        class gets sparser samples as the curriculum grows. Use
        ``noise_scale`` < 1.0 to tighten the synthetic distribution
        around the per-class mean (less variance) or > 1.0 to
        broaden it.
        """
        if not self.mu_per_class:
            return None
        classes_sorted = sorted(self.mu_per_class.keys())
        choice = torch.randint(
            0, len(classes_sorted), (n_total,), generator=generator,
        )
        d = self.mu_per_class[classes_sorted[0]].shape[0]
        feats = torch.zeros(n_total, d)
        labels = torch.zeros(n_total, dtype=torch.long)
        for i in range(n_total):
            c = classes_sorted[int(choice[i])]
            mu = self.mu_per_class[c]
            sigma = self.sigma_per_class[c] * noise_scale
            noise = torch.randn(d, generator=generator)
            feats[i] = (mu + sigma * noise).clamp(min=0.0)
            labels[i] = c
        return feats, labels


# ---------------------------------------------------------------------
# Head-settle pass
# ---------------------------------------------------------------------


def settle_head_via_manifold(
    net: TrioronNetwork,
    manifold: ManifoldStore,
    seen_classes: Sequence[int],
    *,
    n_steps: int = 200,
    batch_size: int = 64,
    lr: float = 0.01,
    noise_scale: float = 1.0,
    head_layer_idx: int = 2,
    head_logits_fn: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
) -> float:
    """Brief head-only training pass under manifold replay.

    Updates ONLY the head's W parameter using synthetic L1 features
    sampled from the manifold's per-class (μ, σ). L1 (and everything
    upstream) stays at no_grad — only the head re-calibrates. This is
    the standard post-absorption head-settle: after pool-matched
    absorption stitches two donors' cells into one substrate, the
    head's per-class logits inherit each donor's pre-absorption
    calibration and need to be re-balanced over the union of classes
    the combined organism now covers.

    Args:
        net: The trioron network. Layer ``head_layer_idx`` is the
            head; its ``W`` is the only trainable parameter.
        manifold: The combined manifold archive (use ``merge_manifold``
            first if you're combining donor archives).
        seen_classes: Global class IDs the head should now cover.
            Used to build the masked cross-entropy active-classes set.
        n_steps: Number of optimizer steps. 200 is the chained-15
            default; smaller substrates settle in 50-100.
        batch_size: Synthetic batch size per step.
        lr: Adam learning rate over the head's W only.
        noise_scale: Forwarded to ``manifold.sample_synthetic``.
        head_layer_idx: Which layer of ``net`` to treat as the head
            (default 2 — i.e. ``net.layers[2]`` in a [L0, L1, head]
            trio).
        head_logits_fn: Callable mapping ``(l1_features, head_W) ->
            logits``. Defaults to :func:`cosine_logits`. Pass a
            linear-head adapter when the head isn't a cosine prototype
            head.

    Returns the final loss value, or NaN when the manifold has no
    archived classes.
    """
    from .classification import masked_cross_entropy

    if not manifold.has_classes():
        return float("nan")
    if head_logits_fn is None:
        head_logits_fn = cosine_logits

    head = net.layers[head_layer_idx]
    opt = torch.optim.Adam([head.W], lr=lr)
    last_loss = float("nan")
    for _ in range(n_steps):
        sample = manifold.sample_synthetic(batch_size, noise_scale=noise_scale)
        if sample is None:
            break
        synth_feats, synth_labels = sample
        with torch.no_grad():
            h0 = synth_feats
            h1 = net.layers[head_layer_idx - 1](h0)
        logits = head_logits_fn(h1, head.W)
        loss = masked_cross_entropy(
            logits, synth_labels, active_classes=list(seen_classes),
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
    return last_loss


def score_via_manifold(
    net: TrioronNetwork,
    manifold: ManifoldStore,
    seen_classes: Sequence[int],
    *,
    n_samples: int = 256,
    noise_scale: float = 1.0,
    head_layer_idx: int = 2,
    head_logits_fn: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Synthetic-sample accuracy proxy used as the retry-and-pick signal
    for :func:`settle_head_with_retry`.

    Draws ``n_samples`` L0 codes from the manifold's per-class
    (μ, σ), forwards them through L1 + head, and returns the fraction
    of argmax-over-active-classes predictions matching the true class.

    Pass a dedicated ``generator`` so the scoring batch is independent
    from the settle-time draws — that way we measure generalisation to
    fresh synthetic samples, not memorisation of the training draws.
    """
    if not manifold.has_classes():
        return float("nan")
    if head_logits_fn is None:
        head_logits_fn = cosine_logits

    sample = manifold.sample_synthetic(
        n_samples, generator=generator, noise_scale=noise_scale,
    )
    if sample is None:
        return float("nan")
    synth_feats, synth_labels = sample
    head = net.layers[head_layer_idx]
    with torch.no_grad():
        h1 = net.layers[head_layer_idx - 1](synth_feats)
        logits = head_logits_fn(h1, head.W)
    active = sorted({int(c) for c in seen_classes})
    if not active:
        return float("nan")
    masked = torch.full_like(logits, float("-inf"))
    idx = torch.tensor(active, dtype=torch.long, device=logits.device)
    masked.index_copy_(1, idx, logits.index_select(1, idx))
    preds = masked.argmax(dim=1)
    return float((preds == synth_labels.to(preds.device)).float().mean().item())


def settle_head_with_retry(
    net: TrioronNetwork,
    manifold: ManifoldStore,
    seen_classes: Sequence[int],
    *,
    n_attempts: int = 10,
    score_samples: int = 256,
    score_noise_scale: float = 1.0,
    seed_offset: int = 0,
    settle_kwargs: Optional[Dict] = None,
    head_layer_idx: int = 2,
    head_logits_fn: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
) -> Tuple[int, float]:
    """Run ``n_attempts`` independent head-settle passes; keep the one
    with the highest held-out synthetic-sample accuracy.

    Strategy:
      1. Snapshot head's pre-settle W and b.
      2. For each trial:
         - restore head from snapshot
         - seed the global RNG (``seed_offset + trial``)
         - run :func:`settle_head_via_manifold`
         - score against a *fresh* synthetic batch drawn from a
           per-trial Generator that is independent of the settle-time
           draws (so we score generalisation, not memorisation)
         - record (W, b) snapshots for the best score so far
      3. Commit the best (W, b) onto the head.

    Returns ``(best_trial_index, best_score)``.

    Motivation: post-absorption, ``settle_head_via_manifold``'s yield
    is highly RNG-dependent — same absorbed substrate can land in a
    high-accuracy basin or a chance-level one depending on which
    synthetic-batch draws gradient descent happens to see (see
    [[head_provenance_mask_result]] for the n=84 σ characterisation).
    Retry-and-pick directly attacks that variance source by trading a
    few extra seconds of absorption-time compute for a near-best-case
    final head.

    The retry is opt-in and absorption-time-only — it costs zero at
    inference and zero persistent storage.
    """
    if settle_kwargs is None:
        settle_kwargs = {}
    head = net.layers[head_layer_idx]
    W_snap = head.W.detach().clone()
    b_snap = head.b.detach().clone()

    best_score = -float("inf")
    best_trial = -1
    best_W: Optional[torch.Tensor] = None
    best_b: Optional[torch.Tensor] = None

    for trial in range(n_attempts):
        with torch.no_grad():
            head.W.data.copy_(W_snap)
            head.b.data.copy_(b_snap)

        torch.manual_seed(seed_offset + trial)
        settle_head_via_manifold(
            net, manifold, seen_classes,
            head_layer_idx=head_layer_idx,
            head_logits_fn=head_logits_fn,
            **settle_kwargs,
        )

        score_gen = torch.Generator().manual_seed(
            seed_offset + 100_000 + trial,
        )
        score = score_via_manifold(
            net, manifold, seen_classes,
            n_samples=score_samples,
            noise_scale=score_noise_scale,
            head_layer_idx=head_layer_idx,
            head_logits_fn=head_logits_fn,
            generator=score_gen,
        )

        if score > best_score:
            best_score = score
            best_trial = trial
            best_W = head.W.detach().clone()
            best_b = head.b.detach().clone()

    with torch.no_grad():
        if best_W is not None:
            head.W.data.copy_(best_W)
        if best_b is not None:
            head.b.data.copy_(best_b)
    return best_trial, best_score


__all__ = [
    "cosine_logits",
    "ManifoldStore",
    "settle_head_via_manifold",
    "score_via_manifold",
    "settle_head_with_retry",
]
