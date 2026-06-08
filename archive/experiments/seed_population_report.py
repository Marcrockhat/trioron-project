"""Aggregate the K-seed donor population for selection signal.

Loads each donor in outputs/seed_population_emnist_kt/seed_*/ and reports
task-aware accuracy mean/σ/min/max. The decision rule is loose:
σ > 0.01 means selection has meaningful signal; σ ≪ 0.01 means seeds
collapse to the same fitness and the inoculation paradigm degenerates
to "any seed will do" (still cheap to compose, just no selection edge).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def load_donors(root: str, label: str = "emnist_kt"):
    """Yield (seed, payload_dict) for every donor in the population dir."""
    pattern = os.path.join(root, "seed_*", f"poc_donor_{label}.pt")
    for path in sorted(glob.glob(pattern)):
        # Path looks like outputs/.../seed_42/poc_donor_emnist_kt.pt
        seed_dir = os.path.basename(os.path.dirname(path))
        seed = int(seed_dir.split("_")[-1])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        yield seed, payload, path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/seed_population_emnist_kt")
    p.add_argument("--label", default="emnist_kt")
    args = p.parse_args(argv)

    rows = list(load_donors(args.root, args.label))
    if not rows:
        print(f"No donors found at {args.root}/seed_*/poc_donor_{args.label}.pt")
        return 1

    print(f"Loaded {len(rows)} donors from {args.root}")
    print()

    # Pull task-aware as the fitness metric. Also collect domain/full
    # for context (these are usually flat across seeds because each
    # donor only covers its own slice).
    scores = []
    print(f"{'seed':>5}  {'task-aware':>11}  {'full':>8}  {'domain':>8}  "
          f"{'arch':>20}  {'protocol_seed':>15}")
    print("-" * 80)
    for seed, payload, path in rows:
        ta = payload["final_accuracy_aware"]
        f  = payload["final_accuracy"]
        d  = payload["final_accuracy_domain"]
        arch = payload["n_nodes_per_layer"]
        ps = payload.get("protocol_seed")
        ps_str = f"0x{ps:08X}" if isinstance(ps, int) else str(ps)
        scores.append(ta)
        print(f"{seed:>5}  {ta:>11.4f}  {f:>8.4f}  {d:>8.4f}  "
              f"{str(arch):>20}  {ps_str:>15}")

    t = torch.tensor(scores, dtype=torch.float64)
    mean = float(t.mean()); std = float(t.std(unbiased=True))
    lo = float(t.min()); hi = float(t.max())
    rng = hi - lo
    print("-" * 80)
    print(f"  task-aware  mean = {mean:.4f}   σ = {std:.4f}   "
          f"min = {lo:.4f}   max = {hi:.4f}   range = {rng:.4f}")
    print()

    # Decision rule.
    if std > 0.01:
        print(f"  → σ = {std:.4f} > 0.01  ::  SELECTION HAS SIGNAL")
        # Identify candidates.
        sorted_idx = sorted(range(len(rows)), key=lambda i: -scores[i])
        top3 = [rows[i][0] for i in sorted_idx[:3]]
        bot3 = [rows[i][0] for i in sorted_idx[-3:]]
        print(f"     top-3 seeds: {top3}  (scores: "
              f"{[f'{scores[i]:.4f}' for i in sorted_idx[:3]]})")
        print(f"     bot-3 seeds: {bot3}  (scores: "
              f"{[f'{scores[i]:.4f}' for i in sorted_idx[-3:]]})")
    elif std > 0.003:
        print(f"  → σ = {std:.4f} ∈ (0.003, 0.01]  ::  WEAK SIGNAL  "
              f"(committee likely beats single, selection marginal)")
    else:
        print(f"  → σ = {std:.4f} ≤ 0.003  ::  SEEDS ARE FUNGIBLE  "
              f"(inoculation collapses to 'any seed', no selection edge)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
