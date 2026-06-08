"""Diagnostic probe: confirm finite-space mechanic actually displaces cells.

Builds a small TrioronNetwork with 4 L1 cells on a tight 2x2 grid in [0,1]²,
calls axis6_spawn 3 times to introduce new cells near existing ones, runs
relax_after_spawn each time, and dumps positions before/after. Should
show: (a) new cells push existing neighbors outward, (b) positions stay
in the unit box, (c) max displacement is non-trivial.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# Force finite-space on so module-level constants are set correctly.
os.environ["AXIS6_FINITE_SPACE"] = "1"
os.environ["AXIS6_REPULSE_STEPS"] = "8"
os.environ["AXIS6_REPULSE_MIN_SEP"] = "0.20"  # bigger than default to force visible motion
os.environ["AXIS6_REPULSE_STRENGTH"] = "0.4"
os.environ["AXIS6_BOUND_REGION"] = "unit_box"
os.environ["AXIS6_REPULSE_DIMS"] = "2"

from trioron.network import TrioronNetwork  # noqa: E402
from experiments.axis6_credit_chained15 import (  # noqa: E402
    relax_after_spawn,
    REPULSE_STEPS, REPULSE_MIN_SEP, REPULSE_STRENGTH,
    BOUND_REGION, REPULSE_DIMS,
)


def dump_positions(label: str, L1) -> None:
    n = L1.n_nodes
    print(f"\n[{label}]  L1.n_nodes={n}")
    for i in range(n):
        p = L1.cell_position[i, :2].tolist()
        print(f"  cell {i:2d}  pos=({p[0]:.3f}, {p[1]:.3f})")


def main() -> int:
    torch.manual_seed(0)
    net = TrioronNetwork([
        (16, 4, "relu"),     # tiny L0
        (4, 4, "relu"),      # L1 with 4 cells
        (4, 3, "linear"),
    ])
    L1 = net.layers[1]
    L1.enable_axis6_field(field_sigma=0.15)

    # Place L1 cells on a tight 2x2 grid so they're close together.
    with torch.no_grad():
        L1.cell_position[0] = torch.tensor([0.3, 0.3, 0.0])
        L1.cell_position[1] = torch.tensor([0.7, 0.3, 0.0])
        L1.cell_position[2] = torch.tensor([0.3, 0.7, 0.0])
        L1.cell_position[3] = torch.tensor([0.7, 0.7, 0.0])

    dump_positions("INITIAL  (2x2 grid)", L1)

    print(f"\n[config]  min_sep={REPULSE_MIN_SEP}  strength={REPULSE_STRENGTH}  "
          f"steps={REPULSE_STEPS}  region={BOUND_REGION}  dims={REPULSE_DIMS}")

    # Spawn 3 times near cell 0 → expect cells 0, 1, 2 pushed outward.
    for spawn_i in range(3):
        before_positions = L1.cell_position[: L1.n_nodes].clone()
        new_idx = net.axis6_spawn(1, candidate_idx=0, position_jitter=0.02)
        # The 'before' snapshot above was BEFORE grow_layer extended the buffer;
        # axis6_spawn places the new cell at parent+tiny_jitter. Now relax:
        max_disp = relax_after_spawn(L1)
        n = L1.n_nodes
        # Compare positions of pre-existing cells (excluding the new one) to before.
        max_existing_drift = float(
            (L1.cell_position[:before_positions.shape[0], :2]
             - before_positions[:, :2]).abs().max().item()
        )
        print(f"\n[spawn {spawn_i+1}]  parent=0  new_idx={new_idx}  "
              f"max_relax_disp={max_disp:.3f}  "
              f"max_existing_cell_drift={max_existing_drift:.3f}")
        dump_positions(f"after spawn {spawn_i+1}", L1)

    print("\n[verdict]")
    in_box = (
        (L1.cell_position[: L1.n_nodes, :2] >= 0.0).all().item()
        and (L1.cell_position[: L1.n_nodes, :2] <= 1.0).all().item()
    )
    pairwise = L1.cell_position[: L1.n_nodes, :2].unsqueeze(1) - \
        L1.cell_position[: L1.n_nodes, :2].unsqueeze(0)
    dists = pairwise.norm(dim=-1)
    n_cells = dists.shape[0]
    dists.fill_diagonal_(float("inf"))
    min_actual_sep = float(dists.min().item())
    print(f"  all cells in unit box:      {in_box}")
    print(f"  min pairwise separation:    {min_actual_sep:.3f}  "
          f"(target ≥ {REPULSE_MIN_SEP})")
    print(f"  cells:                      {n_cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
