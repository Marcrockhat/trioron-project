"""Self-imitation outer loop.

One iteration = rollout → return-filter → trioron API train. The
policy used for rollout is the donor saved at the end of the
previous iteration (or random at iter 0).

The loop is deliberately stateless across iterations from trioron's
perspective: each iteration the bench's growth + dream cycle runs
on the iteration's filtered data, treating it as a fresh task. Past
manifold archives carry the prior knowledge forward — no replay
buffer of raw frames, no Q-target bookkeeping, no PPO machinery.

The frustration → growth signal fires when an iteration's data is
hard enough that the existing substrate plateaus on it. That's the
RL learning signal: "this iteration brought in trajectories the old
policy couldn't recreate, so grow capacity for them."
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from trioron.api import (
    TaskData, TrioronConfig, AdvancedConfig,
    build_donor, extend, load_organism,
)

from .env import N_ACTIONS
from .filter import filter_by_return
from .rollout import collect_episodes


@dataclass
class IterationLog:
    iteration: int
    return_mean: float
    return_median: float
    return_max: float
    return_min: float
    n_episodes: int
    n_kept: int
    n_samples: int
    cutoff: float
    donor_path: str


@dataclass
class TrainResult:
    final_donor: Path
    iterations: List[IterationLog] = field(default_factory=list)


def _to_taskdata(
    X: torch.Tensor, y: torch.Tensor, name: str,
) -> TaskData:
    """80/20 train/test split. test set is required by trioron.api;
    on tiny early iters (n<10) we just clone train as test to satisfy
    the contract."""
    n = X.shape[0]
    n_tr = max(1, int(0.8 * n))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    X_tr, X_te = X[perm[:n_tr]], X[perm[n_tr:]]
    y_tr, y_te = y[perm[:n_tr]], y[perm[n_tr:]]
    if X_te.shape[0] == 0:
        X_te, y_te = X_tr.clone(), y_tr.clone()
    return TaskData(
        name=name,
        X_train=X_tr, y_train=y_tr,
        X_test=X_te, y_test=y_te,
        classes=list(range(N_ACTIONS)),
    )


def self_imitation_train(
    *,
    game: str,
    out_dir: Path,
    n_iterations: int = 8,
    n_episodes_per_iter: int = 16,
    eps_schedule: Union[float, List[float]] = 0.5,
    epochs_per_task: int = 4,
    cap_bytes: int = 32_000,
    seed: int = 42,
    initial_donor: Optional[Path] = None,
    threshold: Union[float, str] = "median",
    top_k: Optional[int] = 3,
    per_class_cap: Optional[int] = None,
    verbose: bool = True,
) -> TrainResult:
    """Run a multi-iteration self-imitation loop.

    Args:
        initial_donor: If given, iteration 0 starts from this organism
            (use case: arm 2 — Pong→Breakout extension). Otherwise
            iteration 0 uses uniform-random rollouts.
        threshold: passed to filter_by_return per-iter.
        eps_schedule: ε for exploration; pass a float for constant or
            a list[n_iterations] for per-iter schedule.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log: List[IterationLog] = []

    # Resolve eps schedule.
    if isinstance(eps_schedule, (int, float)):
        eps_list = [float(eps_schedule)] * n_iterations
    else:
        if len(eps_schedule) != n_iterations:
            raise ValueError(
                f"eps_schedule length {len(eps_schedule)} != "
                f"n_iterations {n_iterations}"
            )
        eps_list = list(eps_schedule)

    cfg = TrioronConfig(
        cap_bytes=cap_bytes,
        dream_replay_steps=50,
        advanced=AdvancedConfig(
            h_init=32,
            n_grow_per_task=4,
            l0_width=128,
            freeze_l0=True,
        ),
    )

    current_donor: Optional[Path] = initial_donor
    for it in range(n_iterations):
        eps = eps_list[it]
        if verbose:
            print(f"\n=== {game} iter {it+1}/{n_iterations} "
                  f"(eps={eps:.2f}) ===")

        organism = load_organism(current_donor) if current_donor else None
        episodes = collect_episodes(
            game=game,
            organism=organism,
            n_episodes=n_episodes_per_iter,
            eps=eps,
            seed=seed + it * 100,
            verbose=verbose,
        )

        X, y, stats = filter_by_return(
            episodes,
            threshold=threshold,
            top_k=top_k,
            per_class_cap=per_class_cap,
            verbose=verbose,
        )
        task = _to_taskdata(X, y, name=f"{game}_iter{it}")

        donor_path = out_dir / f"donor_iter{it}.pt"
        if current_donor is None:
            # First iter on a cold start: build_donor.
            build_donor(
                label=f"{game}_iter{it}",
                tasks=[task],
                seed=seed,
                epochs_per_task=epochs_per_task,
                config=cfg,
                out_path=donor_path,
            )
        else:
            # Subsequent iters: extend the existing organism. Each
            # extension treats the new iter as one more task in the
            # curriculum; growth + dream fire as designed.
            extend(
                donor_path=current_donor,
                base_tasks=[],            # intra-game continuation; no
                                          # need to re-thread base.
                new_tasks=[task],
                out_path=donor_path,
                extension_cap_bytes=cap_bytes * 2,
                epochs_per_task=epochs_per_task,
                permanent_int8=False,
            )
        current_donor = donor_path
        log.append(IterationLog(
            iteration=it,
            return_mean=stats["return_mean"],
            return_median=stats["return_median"],
            return_max=stats["return_max"],
            return_min=stats["return_min"],
            n_episodes=stats["n_episodes_total"],
            n_kept=stats["n_episodes_kept"],
            n_samples=stats["n_samples"],
            cutoff=stats["cutoff_return"],
            donor_path=str(donor_path),
        ))

    return TrainResult(final_donor=current_donor, iterations=log)
