"""Genetic optimization of v2 hyperparameters on chained-15.

Fitness = mean full-softmax accuracy after all 15 tasks.
Genome = ~30 tunable hyperparameters with bounded ranges.

Usage:
    python3 -m experiments.evolve_hyperparams [--pop 8] [--gen 3] [--seed 42] [--workers 1]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from experiments.datasets import DatasetBundle, chained_15_specs, IMAGE_DIM
from trioron.core import Envelope, construct
from trioron.core.epigenome import OUTPUT, PERCEPTION, has_gene
from trioron.core.state import CellState
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from trioron.learning import (
    CreditTracker, FrustrationDetector, ManifoldArchive,
    dream_cycle, DreamConfig, interleaved_replay_batch,
)
from trioron.learning.credit import CreditConfig
from trioron.learning.frustration import FrustrationConfig
from trioron.learning.manifold import ManifoldConfig
from trioron.lifecycle import divide, GrowthConfig


# ── Gene definitions ────────────────────────────────────────────
# Each gene: (name, min, max, dtype, log_scale)
# log_scale=True means mutation/crossover happen in log space

GENOME_SPEC: list[tuple[str, float, float, str, bool]] = [
    # Script-level
    ("h_init",              8,    64,   "int",   False),
    ("batch",              16,   128,   "int",   False),
    ("lr",              1e-4,  1e-1, "float",    True),
    ("epochs",              2,     6,   "int",   False),
    ("n_grow_per_task",     0,    12,   "int",   False),

    # Credit (locking)
    ("theta_e",          0.05,   0.9, "float",  False),
    ("consecutive_tasks",   1,     5,   "int",   False),
    ("g_min",           1e-6,  1e-2, "float",    True),
    ("engagement_decay", 1e-3,   0.1, "float",    True),
    ("lock_base_rate",   0.05,   1.0, "float",  False),

    # Frustration
    ("frust_window",       10,   200,   "int",   False),
    ("frust_eps_improve",1e-4,   0.1, "float",    True),
    ("frust_eps_reset",  0.01,   0.5, "float",  False),
    ("frust_hinge",         1,     5,   "int",   False),
    ("frust_gain",        0.1,   2.0, "float",  False),
    ("frust_ceiling",     1.5,   8.0, "float",  False),

    # Dream cycle
    ("dream_replay_bs",    16,    64,   "int",   False),
    ("dream_replay_lr", 1e-4,  5e-2, "float",    True),
    ("dream_reju_rate",  0.01,   0.5, "float",  False),
    ("dream_recycle_rate",0.01,  0.5, "float",  False),
    ("dream_stab_alpha",  0.1,   0.9, "float",  False),
    ("dream_stab_beta",   0.05,  0.5, "float",  False),

    # Manifold
    ("manifold_replay_steps", 5,  50,  "int",   False),

    # Growth
    ("growth_inherit_frac", 0.1,  0.9, "float", False),
    ("growth_new_edges",     1,   12,   "int",   False),
    ("growth_pos_jitter",0.001,  0.1, "float",    True),
    ("growth_frust_threshold",1.2, 4.0,"float", False),
    ("growth_frust_window",  20, 200,  "int",   False),
    ("growth_frust_frac",  0.2,  0.8, "float",  False),
]


def random_genome(rng: random.Random) -> dict[str, float]:
    genome: dict[str, float] = {}
    for name, lo, hi, dtype, log_scale in GENOME_SPEC:
        if log_scale:
            val = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        else:
            val = rng.uniform(lo, hi)
        if dtype == "int":
            val = round(val)
        genome[name] = val
    return genome


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    child: dict[str, float] = {}
    for name, lo, hi, dtype, log_scale in GENOME_SPEC:
        if rng.random() < 0.5:
            child[name] = a[name]
        else:
            child[name] = b[name]
        if dtype == "int":
            child[name] = round(child[name])
    return child


def mutate(genome: dict, rng: random.Random, rate: float = 0.2, sigma: float = 0.15) -> dict:
    g = dict(genome)
    for name, lo, hi, dtype, log_scale in GENOME_SPEC:
        if rng.random() > rate:
            continue
        if log_scale:
            log_val = math.log(g[name])
            log_range = math.log(hi) - math.log(lo)
            log_val += rng.gauss(0, sigma * log_range)
            g[name] = math.exp(max(math.log(lo), min(math.log(hi), log_val)))
        else:
            span = hi - lo
            g[name] += rng.gauss(0, sigma * span)
            g[name] = max(lo, min(hi, g[name]))
        if dtype == "int":
            g[name] = round(g[name])
    return g


# ── Evaluation ──────────────────────────────────────────────────

def evaluate_genome(genome: dict, seed: int, bundle: DatasetBundle, specs: list) -> float:
    """Run chained-15 with the given genome. Returns mean full-softmax accuracy."""
    torch.manual_seed(seed)

    h_init = int(genome["h_init"])
    epochs = int(genome["epochs"])
    batch = int(genome["batch"])
    lr = genome["lr"]
    n_grow = int(genome["n_grow_per_task"])

    sub = construct(
        base=seeded(IMAGE_DIM, 30, interior_cells=h_init),
        envelope=Envelope(max_parameter_bytes=200_000),
        dispatch_table=default_dispatch_table(),
        capacity=2048,
    )

    credit_cfg = CreditConfig(
        theta_e=genome["theta_e"],
        consecutive_tasks=int(genome["consecutive_tasks"]),
        g_min=genome["g_min"],
        engagement_decay=genome["engagement_decay"],
        lock_base_rate=genome["lock_base_rate"],
    )
    credit = CreditTracker(sub.arena, credit_cfg)

    frust_cfg = FrustrationConfig(
        window_size=int(genome["frust_window"]),
        epsilon_improvement=genome["frust_eps_improve"],
        epsilon_reset=genome["frust_eps_reset"],
        hinge=int(genome["frust_hinge"]),
        gain=genome["frust_gain"],
        ceiling=genome["frust_ceiling"],
    )
    frust = FrustrationDetector(frust_cfg)

    manifold_cfg = ManifoldConfig(
        replay_steps_per_class=int(genome["manifold_replay_steps"]),
    )
    archive = ManifoldArchive(sub.arena, manifold_cfg)

    dream_cfg = DreamConfig(
        replay_batch_size=int(genome["dream_replay_bs"]),
        replay_lr=genome["dream_replay_lr"],
        rejuvenate_base_rate=genome["dream_reju_rate"],
        recycle_base_rate=genome["dream_recycle_rate"],
        stability_alpha=genome["dream_stab_alpha"],
        stability_beta=genome["dream_stab_beta"],
    )

    growth_cfg = GrowthConfig(
        inherit_frac=genome["growth_inherit_frac"],
        new_edges=int(genome["growth_new_edges"]),
        position_jitter=genome["growth_pos_jitter"],
        frustration_threshold=genome["growth_frust_threshold"],
        frustration_window=int(genome["growth_frust_window"]),
        frustration_sustained_frac=genome["growth_frust_frac"],
    )

    for task_idx, spec in enumerate(specs):
        train_view = bundle.task_view(
            spec.dataset_name, spec.local_classes, spec.global_classes,
            split="train", task_name=spec.name,
        )

        sub.prepare_training()
        opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)

        code_boundary = []
        for cid in sub.arena.alive_ids().tolist():
            if has_gene(int(sub.arena.epigenome[cid].item()), PERCEPTION):
                code_boundary.append(cid)

        growth_budget = n_grow
        frust_steps = 0

        for epoch in range(epochs):
            for x_batch, y_batch in train_view.iter_epoch(batch):
                logits = sub(x_batch)
                loss = torch.nn.functional.cross_entropy(logits, y_batch)

                m = frust.step(loss.item())
                if frust.is_frustrated:
                    frust_steps += 1

                scaled_loss = loss * m
                scaled_loss.backward()
                sub.zero_dormant_grads()
                credit.update_utility()

                if sub.last_activations is not None:
                    credit.update_engagement(sub.last_activations)

                opt.step()
                opt.zero_grad()

                # Interleaved manifold replay of past tasks
                replay = interleaved_replay_batch(
                    archive, spec.global_classes,
                    batch_size=batch, n_perc=len(code_boundary),
                )
                if replay is not None:
                    rx, ry = replay
                    r_logits = sub(rx)
                    r_loss = torch.nn.functional.cross_entropy(r_logits, ry)
                    r_loss.backward()
                    sub.zero_dormant_grads()
                    opt.step()
                    opt.zero_grad()

                code = x_batch[:, :len(code_boundary)]
                for gc in spec.global_classes:
                    mask = y_batch == gc
                    if mask.any():
                        archive.update_class(gc, code[mask])

                if growth_budget > 0 and frust.is_frustrated and frust_steps >= 25:
                    interior = [cid for cid in sub.arena.alive_ids().tolist()
                                if not has_gene(int(sub.arena.epigenome[cid].item()), PERCEPTION)
                                and not has_gene(int(sub.arena.epigenome[cid].item()), OUTPUT)
                                and sub.arena.state[cid] == CellState.ACTIVE]
                    if interior:
                        parent = interior[torch.randint(0, len(interior), (1,)).item()]
                        event = divide(sub.arena, parent, growth_cfg)
                        if event:
                            growth_budget -= 1
                            sub.compile()
                            opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
                            frust_steps = 0

        archive.finalize_all()
        dream_result = dream_cycle(
            sub, credit, archive,
            current_classes=spec.global_classes,
            cfg=dream_cfg,
        )
        frust.reset()
        sub.end_task()
        sub.compile()

        for gc in spec.global_classes:
            astro = archive.get(gc)
            if astro and sub.arena.state[astro.cell_id] == CellState.DORMANT:
                sub.arena.state[astro.cell_id] = CellState.ACTIVE

    # Final evaluation: mean full-softmax accuracy
    total_correct = 0
    total_samples = 0
    for t_idx in range(len(specs)):
        spec = specs[t_idx]
        test_view = bundle.task_view(
            spec.dataset_name, spec.local_classes, spec.global_classes,
            split="test", task_name=spec.name,
        )
        x, y = test_view.all_examples()
        with torch.no_grad():
            logits = sub(x)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_samples += y.numel()

    return total_correct / total_samples


# ── GA loop ─────────────────────────────────────────────────────

def run_evolution(
    pop_size: int = 8,
    n_gen: int = 3,
    seed: int = 42,
    elite_frac: float = 0.25,
    out_dir: str = "runs/evolve",
):
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs = chained_15_specs()
    print("Loading datasets...")
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"])

    population = [random_genome(rng) for _ in range(pop_size)]
    n_elite = max(1, int(pop_size * elite_frac))

    best_ever: dict | None = None
    best_fitness = -1.0

    for gen in range(n_gen):
        print(f"\n{'='*60}")
        print(f"Generation {gen}/{n_gen-1}  (pop={pop_size})")
        print(f"{'='*60}")

        fitnesses: list[float] = []
        for i, genome in enumerate(population):
            t0 = time.time()
            try:
                fitness = evaluate_genome(genome, seed, bundle, specs)
            except Exception as e:
                print(f"  [{i}] CRASHED: {e}")
                fitness = 0.0
            elapsed = time.time() - t0
            fitnesses.append(fitness)
            print(f"  [{i}] full={fitness:.4f}  ({elapsed:.1f}s)  "
                  f"h={int(genome['h_init'])} lr={genome['lr']:.1e} "
                  f"θ={genome['theta_e']:.2f} ep={int(genome['epochs'])}")

        ranked = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        elite = [population[i] for i in ranked[:n_elite]]
        elite_fit = [fitnesses[i] for i in ranked[:n_elite]]

        if elite_fit[0] > best_fitness:
            best_fitness = elite_fit[0]
            best_ever = copy.deepcopy(elite[0])

        print(f"\n  Best this gen: {elite_fit[0]:.4f}")
        print(f"  Best ever:     {best_fitness:.4f}")

        gen_log = {
            "generation": gen,
            "fitnesses": fitnesses,
            "best_genome": elite[0],
            "best_fitness": elite_fit[0],
        }
        with open(out / f"gen_{gen:03d}.json", "w") as f:
            json.dump(gen_log, f, indent=2, default=str)

        if gen < n_gen - 1:
            next_pop = list(elite)
            while len(next_pop) < pop_size:
                p1 = elite[rng.randint(0, n_elite - 1)]
                p2 = elite[rng.randint(0, n_elite - 1)]
                child = crossover(p1, p2, rng)
                child = mutate(child, rng)
                next_pop.append(child)
            population = next_pop

    print(f"\n{'='*60}")
    print(f"EVOLUTION COMPLETE — best fitness: {best_fitness:.4f}")
    print(f"{'='*60}")

    with open(out / "best_genome.json", "w") as f:
        json.dump({"fitness": best_fitness, "genome": best_ever}, f, indent=2)
    print(f"Best genome saved to {out / 'best_genome.json'}")

    print("\nBest genome values:")
    for name, _, _, _, _ in GENOME_SPEC:
        print(f"  {name:28s} = {best_ever[name]}")


def main():
    parser = argparse.ArgumentParser(description="Evolve v2 hyperparameters")
    parser.add_argument("--pop", type=int, default=8)
    parser.add_argument("--gen", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1, help="(reserved, sequential for now)")
    parser.add_argument("--out", type=str, default="runs/evolve")
    args = parser.parse_args()

    run_evolution(
        pop_size=args.pop,
        n_gen=args.gen,
        seed=args.seed,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
