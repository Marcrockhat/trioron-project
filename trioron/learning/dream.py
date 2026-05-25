"""Dream cycle — offline consolidation between tasks.  See spec §4.3.

Sequence: replay (all classes) → consolidate → rejuvenate-check → compact
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from trioron.core.construct import Substrate
from trioron.core.epigenome import PERCEPTION, has_gene
from trioron.core.state import CellState
from .credit import CreditTracker
from .manifold import ManifoldArchive
from .rejuvenate import find_rejuvenation_candidates, rejuvenate, DEFAULT_COOLDOWN_TASKS


@dataclass
class DreamConfig:
    replay_batch_size: int = 32
    replay_lr: float = 0.01
    rejuvenate_base_rate: float = 0.1
    rejuvenate_floor: int = 1
    recycle_base_rate: float = 0.1
    recycle_floor: int = 1
    stability_alpha: float = 0.7
    stability_beta: float = 0.3


@dataclass
class DreamResult:
    replay_loss: float = 0.0
    n_locked: int = 0
    n_rejuvenated: int = 0
    n_recycled: int = 0
    locked_ids: list[int] = field(default_factory=list)
    rejuvenated_ids: list[int] = field(default_factory=list)


def stability_factor(
    frustration_pressure: float,
    growth_rate: float,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> float:
    """Compute the stability modulator phi(t) in [0.1, 1.0]."""
    return max(0.1, min(1.0, 1.0 - alpha * frustration_pressure - beta * growth_rate))


def dream_cycle(
    substrate: Substrate,
    credit: CreditTracker,
    archive: ManifoldArchive,
    *,
    current_classes: list[int] | int | None = None,
    current_class: int | None = None,
    frustration_pressure: float = 0.0,
    growth_rate: float = 0.0,
    cfg: DreamConfig | None = None,
) -> DreamResult:
    """Run one full dream cycle: replay ALL classes → consolidate → rejuvenate → compact.

    Replay includes all archived classes (current + past) so the
    manifold signal for the just-learned task is preserved alongside
    past tasks.
    """
    cfg = cfg or DreamConfig()
    result = DreamResult()
    phi = stability_factor(frustration_pressure, growth_rate, cfg.stability_alpha, cfg.stability_beta)

    # Stage 1: Replay ALL classes (no exclusion)
    replay_batches = archive.replay_batches(cfg.replay_batch_size)
    if replay_batches:
        result.replay_loss = _run_replay(substrate, replay_batches, cfg)

    # Stage 2: Consolidate
    locked = credit.consolidate(stability_factor=phi)
    result.locked_ids = locked
    result.n_locked = len(locked)

    # Stage 3: Rejuvenate-check
    n_active = int((substrate.arena.alive & (substrate.arena.state == CellState.ACTIVE)).sum().item())
    reju_cap = max(cfg.rejuvenate_floor, math.ceil(math.sqrt(n_active) * cfg.rejuvenate_base_rate * phi))

    candidates = find_rejuvenation_candidates(substrate.arena)
    for cid in candidates[:reju_cap]:
        if rejuvenate(substrate.arena, cid):
            credit.set_cooldown(cid, DEFAULT_COOLDOWN_TASKS)
            result.rejuvenated_ids.append(cid)
    result.n_rejuvenated = len(result.rejuvenated_ids)

    # Stage 4: Compact (placeholder — full compaction deferred to lifecycle/compact.py)
    result.n_recycled = 0

    # Recompile after structural changes
    if result.n_locked > 0 or result.n_rejuvenated > 0:
        substrate.compile()

    return result


def _run_replay(
    substrate: Substrate,
    replay_batches: list[tuple[torch.Tensor, int]],
    cfg: DreamConfig,
) -> float:
    """Run replay forward/backward passes and return mean loss."""
    a = substrate.arena
    perc_ids = (a.alive & has_gene(a.epigenome, PERCEPTION).bool()).nonzero(as_tuple=False).squeeze(-1)
    n_perc = perc_ids.numel()

    params = substrate.trainable_tensors()
    opt = torch.optim.SGD(params, lr=cfg.replay_lr)

    total_loss = 0.0
    n_batches = 0

    for samples, class_id in replay_batches:
        for start in range(0, samples.shape[0], cfg.replay_batch_size):
            batch = samples[start:start + cfg.replay_batch_size]
            if batch.shape[0] == 0:
                continue

            x = batch[:, :n_perc] if batch.shape[1] >= n_perc else torch.zeros(batch.shape[0], n_perc, device=a.device)

            logits = substrate(x)
            if logits.shape[1] == 0:
                continue

            target_idx = min(class_id, logits.shape[1] - 1)
            target = torch.full((batch.shape[0],), target_idx, dtype=torch.long, device=a.device)

            loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            substrate.zero_dormant_grads()
            opt.step()
            opt.zero_grad()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def interleaved_replay_batch(
    archive: ManifoldArchive,
    current_classes: list[int],
    batch_size: int,
    n_perc: int,
    code_batch: torch.Tensor | None = None,
    top_k: int = 4,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Astrocyte-gated replay: only replay the K most at-risk past classes.

    Each astrocyte evaluates how much the current input overlaps its
    manifold. Only the top-K highest-overlap classes fire protective
    replay. Cost is O(K) regardless of total class count.

    Args:
        archive: the manifold archive.
        current_classes: classes being trained now (excluded from replay).
        batch_size: total replay samples to generate.
        n_perc: number of perception cells (input dim).
        code_batch: current training batch's perception codes [B, code_dim].
            If None, falls back to uniform replay over all past classes.
        top_k: max number of past classes to replay per batch.

    Returns:
        (x, y) tensors for cross_entropy, or None if no past classes.
    """
    past_classes = [
        cid for cid in archive.class_ids
        if cid not in current_classes
        and archive.arena.state[archive.get(cid).cell_id] == CellState.DORMANT
    ]

    if not past_classes:
        return None

    if code_batch is not None and len(past_classes) > top_k:
        scores = []
        for cid in past_classes:
            astro = archive.get(cid)
            ll = astro.log_likelihood(code_batch)
            scores.append(ll.mean().item())

        ranked = sorted(zip(scores, past_classes), reverse=True)
        past_classes = [cid for _, cid in ranked[:top_k]]

    per_class = max(1, batch_size // len(past_classes))
    xs = []
    ys = []

    for cid in past_classes:
        astro = archive.get(cid)
        samples = astro.sample(per_class)
        x = samples[:, :n_perc] if samples.shape[1] >= n_perc else torch.zeros(per_class, n_perc, device=archive.arena.device)
        xs.append(x)
        ys.append(torch.full((x.shape[0],), cid, dtype=torch.long, device=archive.arena.device))

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)
