"""Trioron 2.0 risks §8.6 — lifetime dendritic-budget sizing exercise.

The spec (trioron_2_0.md §8 risk #6) requires a back-of-envelope
parameter envelope before dendrites are turned on outside paper benches:

  "B_max=8 per cell combines multiplicatively with n_nodes and
   insert_layer's K_insert. A 70–80 yr deployment that grows cells,
   layers, *and* branches has a parameter envelope substantially
   larger than the current width-only growth budget."

This script computes the envelope for a few canonical deployment
scenarios and recommends cap defaults that keep the substrate
trainable on commodity edge hardware while leaving headroom for the
reasoning regime.

Model assumptions (kept deliberately simple — order-of-magnitude only):

  - Initial substrate: 3 layers (L0 / L1 / head).
  - Growth concentrated in L1 (paper convention).
  - Per-task growth events: n_grow_per_task new cells in L1.
  - Insertions: at most K_insert per original slot over the lifetime.
  - Dendrites: each cell at K_max ≤ B_max branches once mature.
  - No pruning (worst-case ceiling — actual deployments will prune,
    so reported numbers are upper bounds).
  - fp32 parameters, 4 bytes each.

Output: outputs/lifetime_budget_sizing.csv + a run log.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------- model ----------

@dataclass
class Scenario:
    name: str
    n_layers_init: int = 3        # L0, L1, head
    fan_in_init: int = 128        # L1's fan_in (typical post-L0 width)
    n_nodes_init: int = 32        # cells per layer at construction
    n_classes: int = 100          # head width
    tasks_per_year: int = 100     # continual-learning tempo
    years: int = 80               # lifetime horizon
    n_grow_per_task: int = 4      # cells added per task (L1)
    enable_insert: bool = False
    K_insert: int = 0
    enable_dendrites: bool = False
    B_max: int = 1


def total_params(s: Scenario) -> dict:
    """Compute the upper-bound parameter count under the scenario's
    growth model. Returns a breakdown dict (param counts, not bytes)."""
    total_tasks = s.tasks_per_year * s.years

    # ---- L1 width growth ----
    l1_cells_init = s.n_nodes_init
    l1_cells_grown = total_tasks * s.n_grow_per_task
    l1_cells = l1_cells_init + l1_cells_grown

    # Per L1 cell, point-neuron-equivalent params:
    #   W row (fan_in_init) + b (1)
    # Plus dendritic params if enabled:
    #   branch_weight (B_max) + branch_weight_anchor (B_max) +
    #   fisher_branch_weight (B_max) — only branch_weight is
    #   trainable; the rest are buffers but still RAM.
    base_per_cell = s.fan_in_init + 1
    dend_per_cell = (3 * s.B_max) if s.enable_dendrites else 0
    l1_params_per_cell = base_per_cell + dend_per_cell

    l1_params = l1_cells * l1_params_per_cell

    # Head: fan_in = l1_cells. n_classes nodes. Each: W row + b.
    head_params = l1_cells * s.n_classes + s.n_classes

    # ---- Inserted layers (Axis 3) ----
    inserted_params = 0
    if s.enable_insert and s.K_insert > 0:
        # Two original slots in a 3-layer net (between L0/L1, L1/head).
        n_slots = s.n_layers_init - 1
        total_inserted_layers = n_slots * s.K_insert
        # Each inserted layer: n_nodes_init cells with fan_in =
        # n_nodes_init (the prev layer's width at insertion time).
        # Plus its own dendrite cost if enabled.
        per_layer = s.n_nodes_init * (s.n_nodes_init + 1)
        per_layer += s.n_nodes_init * dend_per_cell
        inserted_params = total_inserted_layers * per_layer

    # ---- L0 stays fixed (frozen substrate) ----
    # fan_in = e.g. 784 for MNIST. We don't size it explicitly because
    # L0 is shared across donors and lives in the protocol subspace —
    # not part of the lifetime growth envelope.
    l0_params = s.fan_in_init * 784  # rough estimate; not tracked

    return {
        "l1_cells": l1_cells,
        "l1_params": l1_params,
        "head_params": head_params,
        "inserted_params": inserted_params,
        "total_growth_params": l1_params + head_params + inserted_params,
        "total_growth_bytes_fp32": (l1_params + head_params + inserted_params) * 4,
    }


def format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    for u in units:
        if n < 1024 or u == units[-1]:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} {units[-1]}"


# ---------- scenarios ----------

def build_scenarios() -> list[Scenario]:
    common = dict(
        n_layers_init=3,
        fan_in_init=128,
        n_nodes_init=32,
        n_classes=100,
        tasks_per_year=100,
        n_grow_per_task=4,
    )

    out: list[Scenario] = []

    # Three horizons × four configs.
    horizons = [("1yr", 1), ("10yr", 10), ("80yr", 80)]
    configs = [
        ("1.0 baseline (no Axes 3/5)",
         dict(enable_insert=False, K_insert=0, enable_dendrites=False, B_max=1)),
        ("Axis 3 only (insert_layer K=3)",
         dict(enable_insert=True, K_insert=3, enable_dendrites=False, B_max=1)),
        ("Axis 5 only (dendrites B_max=8)",
         dict(enable_insert=False, K_insert=0, enable_dendrites=True, B_max=8)),
        ("Both axes (REASONING profile)",
         dict(enable_insert=True, K_insert=3, enable_dendrites=True, B_max=8)),
    ]
    for label, years in horizons:
        for cname, cfg in configs:
            s = Scenario(name=f"{cname} @ {label}", years=years, **common, **cfg)
            out.append(s)

    # Plus an edge-hardware scenario at 80yr horizon.
    out.append(Scenario(
        name="EDGE profile @ 80yr (constrained growth)",
        n_layers_init=3,
        fan_in_init=128,
        n_nodes_init=32,
        n_classes=100,
        tasks_per_year=20,         # slower lifetime tempo
        years=80,
        n_grow_per_task=1,         # one cell per task
        enable_insert=False,
        K_insert=0,
        enable_dendrites=False,
        B_max=1,
    ))
    return out


# ---------- driver ----------

def main() -> None:
    scenarios = build_scenarios()
    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        print(msg, flush=True)

    log("trioron 2.0 risks §8.6 — lifetime dendritic-budget sizing")
    log("model: 3 init layers, L1 width growth, optional Axis 3/5 enabled")
    log(
        "  base per-cell: fan_in + 1 (W row + bias)"
    )
    log(
        "  dendrite per-cell: 3 × B_max "
        "(branch_weight + anchor + fisher_branch_weight)"
    )
    log("  inserted layers: n_slots × K_insert × (n_nodes² + n_nodes + dend)")
    log("  upper bound — no pruning; fp32 (4 B/param)")
    log("")
    log(f"{'scenario':<55}  {'L1 cells':>10}  {'growth params':>14}  {'fp32 bytes':>14}")
    log("-" * 100)

    rows: list[dict] = []
    for s in scenarios:
        r = total_params(s)
        log(
            f"{s.name:<55}  "
            f"{r['l1_cells']:>10,d}  "
            f"{r['total_growth_params']:>14,d}  "
            f"{format_bytes(r['total_growth_bytes_fp32']):>14}"
        )
        rows.append({
            "scenario": s.name,
            "years": s.years,
            "tasks_per_year": s.tasks_per_year,
            "n_grow_per_task": s.n_grow_per_task,
            "enable_insert": s.enable_insert,
            "K_insert": s.K_insert,
            "enable_dendrites": s.enable_dendrites,
            "B_max": s.B_max,
            "l1_cells": r["l1_cells"],
            "growth_params": r["total_growth_params"],
            "fp32_bytes": r["total_growth_bytes_fp32"],
            "fp32_human": format_bytes(r["total_growth_bytes_fp32"]),
        })

    log("")
    log("findings:")

    # Pull headline numbers for the analysis block.
    by_label = {row["scenario"]: row for row in rows}
    baseline_80 = by_label["1.0 baseline (no Axes 3/5) @ 80yr"]
    insert_80 = by_label["Axis 3 only (insert_layer K=3) @ 80yr"]
    dendrite_80 = by_label["Axis 5 only (dendrites B_max=8) @ 80yr"]
    both_80 = by_label["Both axes (REASONING profile) @ 80yr"]
    edge_80 = by_label["EDGE profile @ 80yr (constrained growth)"]

    def ratio(a, b):
        return a["growth_params"] / b["growth_params"]

    log(
        f"  Axis 5 only adds {ratio(dendrite_80, baseline_80):.3f}× params "
        f"vs 1.0 baseline at 80yr (3·B_max per cell is small vs head)"
    )
    log(
        f"  Axis 3 only adds {ratio(insert_80, baseline_80):.3f}× params "
        f"(insertions are bounded by n_slots × K_insert × n_nodes²)"
    )
    log(
        f"  Both axes together: {ratio(both_80, baseline_80):.3f}× — "
        f"multiplicative envelope is not catastrophic"
    )
    log("")

    log("cap recommendations:")
    log(
        f"  REASONING @ 80yr {format_bytes(both_80['fp32_bytes'])}: "
        "feasible on a workstation / Orange Pi 5B with 8 GB RAM; "
        "no profile-level memory cap needed for the substrate alone"
    )
    log(
        f"  EDGE @ 80yr {format_bytes(edge_80['fp32_bytes'])}: "
        "well under the 256 MB cap currently set on the EDGE preset; "
        "the existing cap is the right floor"
    )
    log(
        "  The dominant cost is the HEAD (n_l1_cells × n_classes); "
        "dendrites are a rounding error against head growth. "
        "Cap policy should target n_l1_cells (or equivalently "
        "tasks_per_year × n_grow_per_task × years), not B_max."
    )
    log("")
    log(
        "verdict: Axis 5's lifetime contribution is small relative "
        "to L1 width growth. B_max=8 is safe to ship live by default; "
        "lifetime sizing is gated on n_grow_per_task and head width, "
        "not on dendritic structure."
    )

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "lifetime_budget_sizing.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    log_path = out_dir / "lifetime_budget_sizing_run.log"
    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"\nwrote {csv_path}")
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()
