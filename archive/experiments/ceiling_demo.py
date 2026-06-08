"""§8 step 7 verification: hard ceilings as preflight checks before division.

Two scenarios, both run against the same starter network:

  Scenario A — MEMORY ceiling.
    M_max set just below the predicted footprint of the next division.
    The very first preflight DENIES with reason=memory_ceiling and
    flips arrested=True. A subsequent attempt returns reason=arrested.

  Scenario B — TIME ceiling.
    M_max generous; T_div_max=10s but the (faked) stabilization phase
    takes 11s. The first preflight ALLOWS, division proceeds, then the
    NEXT preflight DENIES with reason=time_ceiling.

Both scenarios use injected memory/time providers — the point of this
demo is to exercise the decision paths deterministically, not to chase
real RSS noise. Per §4.2: "Once arrested, the network is mature and may
only update via plasticity (weights change, topology does not)."

Outputs:
  outputs/ceiling_demo_log.csv  — per-attempt decision trace.
"""
from __future__ import annotations
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import TrioronNetwork
from trioron.ceilings import (
    CeilingsController,
    REASON_ARRESTED,
    REASON_MEMORY_CEILING,
    REASON_OK,
    REASON_TIME_CEILING,
    division_param_delta,
)


def _make_net() -> TrioronNetwork:
    return TrioronNetwork(
        [
            (8, 16, "relu"),
            (16, 16, "relu"),
            (16, 4, "tanh"),
        ]
    )


class _FakeClock:
    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _row(scenario: str, attempt: int, decision, ceilings) -> list:
    return [
        scenario,
        attempt,
        decision.layer_idx,
        int(decision.allowed),
        decision.reason,
        decision.current_bytes,
        decision.delta_bytes,
        decision.projected_bytes,
        decision.M_max_bytes,
        decision.last_stab_seconds if decision.last_stab_seconds is not None else "",
        decision.T_div_max_seconds,
        int(ceilings.arrested),
    ]


def run_scenario_memory(rows: list) -> bool:
    """M_max sized to deny the first attempt. Expect MEMORY → ARRESTED."""
    print("-" * 78)
    print("Scenario A — MEMORY ceiling")
    print("-" * 78)
    net = _make_net()
    layer_idx = 1
    delta = division_param_delta(net, layer_idx, optimizer_state_per_param=2)
    delta_bytes = delta.bytes(dtype_bytes=4)
    # Set M_max so projected = current(0) + delta is over by 1 byte.
    M_max = delta_bytes - 1
    print(f"  predicted Δ = {delta.total_floats} floats "
          f"= {delta_bytes} B   M_max = {M_max} B (over by 1 B)")

    c = CeilingsController(
        M_max_bytes=M_max,
        T_div_max_seconds=60.0,
        memory_provider=lambda: 0,
    )

    d1 = c.preflight(net, layer_idx)
    print(f"  attempt 1: {d1}")
    rows.append(_row("memory", 1, d1, c))
    ok1 = (not d1.allowed) and d1.reason == REASON_MEMORY_CEILING and c.arrested

    d2 = c.preflight(net, layer_idx)
    print(f"  attempt 2: {d2}")
    rows.append(_row("memory", 2, d2, c))
    ok2 = (not d2.allowed) and d2.reason == REASON_ARRESTED

    print(f"  controller: {c}")
    print(f"  result:     {'PASS' if (ok1 and ok2) else 'FAIL'}")
    return ok1 and ok2


def run_scenario_time(rows: list) -> bool:
    """First division allowed; stabilization takes 11s > T_div_max=10s.
    Next preflight should arrest with TIME_CEILING."""
    print("-" * 78)
    print("Scenario B — TIME ceiling")
    print("-" * 78)
    net = _make_net()
    clock = _FakeClock()
    c = CeilingsController(
        M_max_bytes=1024 ** 3,             # generous
        T_div_max_seconds=10.0,
        memory_provider=lambda: 0,
        time_provider=clock,
    )

    layer_idx = 0

    d1 = c.preflight(net, layer_idx)
    print(f"  attempt 1: {d1}")
    rows.append(_row("time", 1, d1, c))
    if not d1.allowed:
        print("  unexpected: first preflight should have allowed.")
        return False

    # Real division.
    net.grow_layer(layer_idx)
    print(f"  divided: layer 0 → {net.layers[0].n_nodes} nodes; net = {net}")

    # Faked stabilization phase: pretend it took 11s of wall clock.
    c.mark_stabilization_start()
    clock.advance(11.0)
    elapsed = c.mark_stabilization_end()
    print(f"  stabilization elapsed: {elapsed:.2f}s "
          f"(budget {c.T_div_max_seconds:.2f}s — over)")

    d2 = c.preflight(net, layer_idx)
    print(f"  attempt 2: {d2}")
    rows.append(_row("time", 2, d2, c))
    ok2 = (not d2.allowed) and d2.reason == REASON_TIME_CEILING and c.arrested

    d3 = c.preflight(net, layer_idx)
    print(f"  attempt 3: {d3}")
    rows.append(_row("time", 3, d3, c))
    ok3 = (not d3.allowed) and d3.reason == REASON_ARRESTED

    print(f"  controller: {c}")
    print(f"  result:     {'PASS' if (ok2 and ok3) else 'FAIL'}")
    return ok2 and ok3


def run_scenario_happy(rows: list) -> bool:
    """Generous budget, fast stabilization, two divisions allowed."""
    print("-" * 78)
    print("Scenario C — happy path (control)")
    print("-" * 78)
    net = _make_net()
    clock = _FakeClock()
    c = CeilingsController(
        M_max_bytes=1024 ** 3,
        T_div_max_seconds=10.0,
        memory_provider=lambda: 0,
        time_provider=clock,
    )

    for attempt in range(1, 3):
        d = c.preflight(net, layer_idx=0)
        print(f"  attempt {attempt}: {d}")
        rows.append(_row("happy", attempt, d, c))
        if not d.allowed:
            print("  unexpected: should have been allowed.")
            return False
        net.grow_layer(layer_idx=0)
        c.mark_stabilization_start()
        clock.advance(2.0)               # well under 10s budget
        c.mark_stabilization_end()

    print(f"  controller: {c}")
    print(f"  divisions_allowed = {c.divisions_allowed}")
    return c.divisions_allowed == 2 and not c.arrested


def main() -> int:
    print("=" * 78)
    print("Trioron — Step 7 verification: ceilings preflight")
    print("=" * 78)

    rows: list = []

    a = run_scenario_memory(rows)
    print()
    b = run_scenario_time(rows)
    print()
    c = run_scenario_happy(rows)
    print()

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ceiling_demo_log.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "scenario", "attempt", "layer_idx", "allowed", "reason",
            "current_bytes", "delta_bytes", "projected_bytes",
            "M_max_bytes", "last_stab_seconds", "T_div_max_seconds",
            "arrested",
        ])
        w.writerows(rows)
    print(f"  log: {csv_path}")

    all_pass = a and b and c
    print()
    print("=" * 78)
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 78)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
