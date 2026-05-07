"""Population-based search over trioron donors.

Each generation produces N donors with different `build_donor` seeds.
Different seeds → different L0 random projections → different feature
spaces. Some projections happen to cluster game-relevant features
(ball-near-paddle, brick-cluster) usefully; others don't. Selection
picks the elites by rollout return; the next generation trains on
the elites' trajectories.

This is the trioron-API-only version of pop search:
  - No PPO, no gradient-based weight perturbation, no advantage
    estimation. The "mutation" is "different random seed in
    build_donor"; the "selection" is "rollout return"; the "breeding"
    is "next gen trains on elite rollouts".
  - All trioron behaviour (frustration → growth, dream consolidation,
    archive routing) runs unchanged inside each pop member's
    build_donor call.

Caveat: varying seed across pop members varies the L0 projection,
which breaks the shared-L0 invariant `api.absorb` requires for graft
(arm4). For arm4, post-hoc the chosen elites must share L0 — handled
by enforcing seed equality across the two single-arm finals when
those are produced from the SAME seed pool. The simplest safe
recipe: accept that pop-search winners across arms 1 & 3 will not
be graft-compatible; document the limitation and address in a
separate canonicalisation pass if arm4 turns out to be load-bearing.
"""
from __future__ import annotations
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from trioron.api import (
    TaskData, TrioronConfig, AdvancedConfig, build_donor, load_organism,
)

from .env import N_ACTIONS
from .filter import filter_by_return
from .rollout import collect_episodes, Episode


@dataclass
class GenLog:
    generation: int
    pop_returns: List[float]
    elite_returns: List[float]
    elite_seeds: List[int]
    n_samples_trained: int
    wallclock_s: float
    best_donor: str


@dataclass
class PopResult:
    final_donor: Path
    generations: List[GenLog] = field(default_factory=list)


def _to_taskdata(X: torch.Tensor, y: torch.Tensor, name: str) -> TaskData:
    n = X.shape[0]
    n_tr = max(1, int(0.8 * n))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    X_tr, X_te = X[perm[:n_tr]], X[perm[n_tr:]]
    y_tr, y_te = y[perm[:n_tr]], y[perm[n_tr:]]
    if X_te.shape[0] == 0:
        X_te, y_te = X_tr.clone(), y_tr.clone()
    return TaskData(
        name=name, X_train=X_tr, y_train=y_tr,
        X_test=X_te, y_test=y_te,
        classes=list(range(N_ACTIONS)),
    )


def _mean_return(rollouts: List[Episode]) -> float:
    if not rollouts:
        return float("-inf")
    return sum(e.return_ for e in rollouts) / len(rollouts)


def pop_search_train(
    *,
    game: str,
    out_dir: Path,
    n_generations: int = 6,
    n_population: int = 8,
    n_elites: int = 2,
    n_episodes_per_eval: int = 4,
    epochs_per_task: int = 4,
    cap_bytes: int = 32_000,
    base_seed: int = 42,
    eval_eps: float = 0.10,
    cleanup_non_elites: bool = True,
    initial_donor: Optional[Path] = None,
    verbose: bool = True,
) -> PopResult:
    """Run N_gens × N_pop pop search.

    Generation 0: N random rollouts (no donor needed). Pick elites,
    use their merged trajectories as the dataset for gen 1's N
    builds. Each gen ≥ 1: build N donors with seeds = base_seed+gen*N+i,
    evaluate each, pick elites, repeat.

    initial_donor: when set (e.g., for arm2 phase B = continue from
    arm3's Pong donor), skip the random gen-0 step and use the
    initial donor's rollouts as gen-0 elites. Each subsequent gen's
    builds api.extend the initial donor with the new game's elite
    rollouts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log: List[GenLog] = []

    cfg = TrioronConfig(
        cap_bytes=cap_bytes,
        dream_replay_steps=50,
        advanced=AdvancedConfig(
            h_init=32, n_grow_per_task=4, l0_width=128, freeze_l0=True,
        ),
    )

    # ---------- Generation 0: collect baseline rollouts -----------
    if verbose:
        print(f"\n=== {game} pop-search generation 0 "
              f"({'random' if initial_donor is None else 'initial donor'} × "
              f"{n_population}) ===")
    t0 = time.time()
    pop = []
    if initial_donor is None:
        for i in range(n_population):
            eps = collect_episodes(
                game=game, organism=None, eps=1.0,
                n_episodes=n_episodes_per_eval,
                seed=base_seed + i * 1000,
                verbose=False,
            )
            pop.append({"donor": None, "rollouts": eps,
                        "mean_return": _mean_return(eps),
                        "seed": base_seed + i * 1000})
    else:
        org = load_organism(initial_donor)
        for i in range(n_population):
            eps = collect_episodes(
                game=game, organism=org, eps=eval_eps,
                n_episodes=n_episodes_per_eval,
                seed=base_seed + i * 1000,
                verbose=False,
            )
            pop.append({"donor": initial_donor, "rollouts": eps,
                        "mean_return": _mean_return(eps),
                        "seed": base_seed + i * 1000})
    pop.sort(key=lambda x: x["mean_return"], reverse=True)
    rets = [p["mean_return"] for p in pop]
    if verbose:
        print(f"  gen 0: returns top={rets[0]:+.2f} bot={rets[-1]:+.2f} "
              f"spread={rets[0]-rets[-1]:+.2f}")
    log.append(GenLog(
        generation=0,
        pop_returns=rets,
        elite_returns=rets[:n_elites],
        elite_seeds=[p["seed"] for p in pop[:n_elites]],
        n_samples_trained=0,
        wallclock_s=time.time() - t0,
        best_donor=str(initial_donor) if initial_donor else "(random)",
    ))
    elite_eps: List[Episode] = []
    for p in pop[:n_elites]:
        elite_eps.extend(p["rollouts"])

    # ---------- Generations 1..G -----------
    final_donor: Optional[Path] = None
    for gen in range(1, n_generations + 1):
        gen_t0 = time.time()
        if verbose:
            print(f"\n=== {game} pop-search generation {gen}/"
                  f"{n_generations}: train {n_population} from "
                  f"{n_elites}-elite trajectories ===")

        X, y, stats = filter_by_return(
            elite_eps, top_k=n_elites, verbose=False,
        )
        task = _to_taskdata(X, y, name=f"{game}_gen{gen}")
        n_samples = int(X.shape[0])

        # Build N donors with diverse seeds. The seed varies the L0
        # random projection AND SGD order; with frozen L0 the policy
        # head trains over a different feature space per pop member.
        new_pop: List[dict] = []
        for i in range(n_population):
            seed_i = base_seed + gen * 1000 + i
            donor_path = out_dir / f"gen{gen}_pop{i}.pt"
            build_donor(
                label=f"{game}_g{gen}_p{i}",
                tasks=[task],
                seed=seed_i,
                epochs_per_task=epochs_per_task,
                config=cfg,
                out_path=donor_path,
            )
            new_pop.append({"donor": donor_path, "seed": seed_i})

        # Evaluate each pop member.
        for member in new_pop:
            org = load_organism(member["donor"])
            eps = collect_episodes(
                game=game, organism=org, eps=eval_eps,
                n_episodes=n_episodes_per_eval,
                seed=base_seed + gen * 100_000 + member["seed"] % 1000,
                verbose=False,
            )
            member["rollouts"] = eps
            member["mean_return"] = _mean_return(eps)

        new_pop.sort(key=lambda x: x["mean_return"], reverse=True)
        rets = [m["mean_return"] for m in new_pop]
        if verbose:
            print(f"  gen {gen}: returns top={rets[0]:+.2f} "
                  f"med={np.median(rets):+.2f} bot={rets[-1]:+.2f} "
                  f"spread={rets[0]-rets[-1]:+.2f}")

        elite_eps = []
        elites = new_pop[:n_elites]
        for m in elites:
            elite_eps.extend(m["rollouts"])

        # Cleanup non-elite donors to keep disk under control. Each
        # donor is ~43 MB; without cleanup, n_pop × n_gen × 43 MB
        # accumulates fast.
        if cleanup_non_elites:
            elite_paths = {str(m["donor"]) for m in elites}
            for m in new_pop:
                p = m["donor"]
                if str(p) not in elite_paths and p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass

        log.append(GenLog(
            generation=gen,
            pop_returns=rets,
            elite_returns=[m["mean_return"] for m in elites],
            elite_seeds=[m["seed"] for m in elites],
            n_samples_trained=n_samples,
            wallclock_s=time.time() - gen_t0,
            best_donor=str(elites[0]["donor"]),
        ))
        final_donor = elites[0]["donor"]

    return PopResult(final_donor=final_donor, generations=log)
