"""I2 gate — period-1 perception generation REPLACES genesis (integration
design, phase I2).

Same testbed as run_taxonomy_genesis: the 10-species taxonomy planted in a
64-column aperture (7 real features + 5 randn noise + 52 empty). Baseline =
genesis.discover_input_layer (variance apoptosis + a 21-step GRADIENT
saliency probe). Challenger = the PCLL developmental flow with ZERO
gradient steps anywhere:

  organism = germline only (progenitor + attached council)
  period 1 = perception generation: tick-1 spawn (one PERCEPTION cell per
             column, progenitor-parented), tick-2 receptor equip, census;
             first sitting STARVES census-constant columns (the variance-
             apoptosis equivalent — under signed data a constant is NOT at
             the q=0 floor, so starvation is structural: receptor gene
             withdrawn, cell dormant, column mapping preserved), assigns
             DISCRETE labeled lines (D7 census), seeds the importance
             ranking, hands over to the controller.
  periods 2+ = species periods; habituation retires variance-bearing-but-
             incoherent receptors (the saliency-probe equivalent) after
             RETIRE_PATIENCE consecutive testifying-incoherent periods.

Gate asserts (PASS required before I3):
  - starved set == the empty columns exactly (every seed);
  - the 5 binary feature columns (4× cover one-hot + eggs) get discrete
    k=2 verdicts; weight/height stay continuous;
  - every noise column retires by habituation; no real column ever does;
  - final read set == the 7 real columns == the genesis baseline kept-set
    (kept-set quality ≥ genesis, with zero gradient probes);
  - the germline (progenitor + 20 council cells) stays active, never locks;
  - no gradients: no optimizer exists, edge_weight.grad is None, 0 edges.

Run: python3 -m experiments.progenitor.run_i2_genesis   (from project root)
"""
from __future__ import annotations

import torch

from trioron.core import construct
from trioron.core.state import CellState
from trioron.pcll import PerceptionGenesis, germline_base

from .data_taxonomy import make_taxonomy
from .genesis import build_aperture, discover_input_layer

SEEDS = 3
APERTURE_N = 64
N_NOISE = 5
N_PER_CLASS = 256
GENESIS_SEED = 1


def run_seed(seed: int) -> None:
    tr = make_taxonomy(seed=seed, n_per_class=N_PER_CLASS)
    n_out = len(tr.names)
    X, real_cols, noise_cols = build_aperture(
        tr.x, APERTURE_N, N_NOISE, seed=GENESIS_SEED + seed)
    realset, noiseset = set(real_cols), set(noise_cols)
    emptyset = set(range(APERTURE_N)) - realset - noiseset
    # feature j sits at aperture column real_cols[j] (build_aperture sorts);
    # features 2..6 are the binary ones (4× cover one-hot + eggs)
    binary_cols = {real_cols[j] for j in range(2, 7)}

    # ── baseline: genesis with its gradient probe ──
    base = discover_input_layer(X, tr.y, n_out, real_cols, noise_cols, seed=0)
    base_kept = set(base.kept)

    # ── challenger: the PCLL developmental flow ──
    sub = construct(germline_base, capacity=128)
    pg = PerceptionGenesis(sub)
    per_class = [X[tr.y == c] for c in range(n_out)]

    pg.feed(per_class[0])
    sitting = sub.end_task()

    assert set(sitting.starved) == emptyset, (
        f"starved {sorted(set(sitting.starved) ^ emptyset)} mismatch")
    assert set(sitting.discrete) == binary_cols, (
        f"discrete verdicts {sorted(sitting.discrete)} != binary {sorted(binary_cols)}")
    assert all(k == 2 for k in sitting.discrete.values()), "non-binary k"
    cont = {c for c, v in sitting.verdicts.items() if v.kind == "continuous"}
    assert cont == (realset - binary_cols) | noiseset, "continuous verdicts wrong"

    ctrl = pg.controller
    plan_cols = sub.scheduler._plan.receptor_cols.tolist()  # receptor idx → aperture col
    retired_at: dict[int, int] = {}
    for c in range(1, n_out):
        ctrl.observe(per_class[c])
        report = sub.end_task()
        for r in report.retired:
            retired_at[plan_cols[r]] = c + 1   # organism period number

    kept = {plan_cols[i] for i in range(len(plan_cols)) if ctrl.read_mask[i]}

    assert kept == realset, f"kept {sorted(kept)} != real {sorted(realset)}"
    assert set(retired_at) == noiseset, (
        f"retired {sorted(retired_at)} != noise {sorted(noiseset)}")
    assert kept == base_kept, "PCLL kept-set != genesis baseline kept-set"

    # germline integrity: progenitor + council active, never locked
    a = sub.arena
    g = pg.germline
    germ = [g.progenitor_id] + [c for ids in g.council_ids.values() for c in ids]
    assert all(int(a.state[c].item()) == CellState.ACTIVE for c in germ), \
        "germline cell locked/dormant"
    assert abs(sum(g.votes.values()) - 20.0) < 1e-9, "vote economy not conserved"

    # zero gradients anywhere
    assert a.edge_weight.grad is None and a.edge_cursor == 0

    top = [f"col{c}:{m:.1f}" for c, m in sitting.importance[:4]]
    print(f"seed={seed}: starved {len(sitting.starved)} empties at the first "
          f"sitting; discrete k=2 on {sorted(sitting.discrete)}; noise retired "
          f"at periods {sorted(set(retired_at.values()))}; kept == real == "
          f"genesis-kept ({sorted(kept)}); importance top: {', '.join(top)}")


def main() -> None:
    for seed in range(SEEDS):
        run_seed(seed)
    print("\nI2 gate PASS — period-1 perception generation matches the genesis "
          "kept-set with zero gradient probes")


if __name__ == "__main__":
    main()
