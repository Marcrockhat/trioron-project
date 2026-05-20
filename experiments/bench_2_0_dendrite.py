"""Trioron 2.0 — Phase 6 dendrite delta on concentric rings.

The Axis 5 falsification gate (per trioron_2_0.md §6): does dendritic
growth lift a within-niche fine-discrimination task above K=1?

Task: 2D concentric rings binary classification.
  - Class 0: points sampled from a thin ring at radius 1.0 + noise.
  - Class 1: points sampled from a thin ring at radius 1.3 + noise.
  - Optimal decision boundary: r² > ~1.15 — a quadratic in (x, y).

Substrate: single trioron cell (fan_in=2, n_nodes=1), activation="linear".
The cell's scalar output is treated directly as the binary logit
(BCEWithLogitsLoss). With this architecture:

  K=1 (point neuron): output = W[0,0]·x + W[0,1]·y + b
       → linear classifier on (x, y) → cannot separate concentric rings.

  K=2 forced (col 0 → branch 0, col 1 → branch 1), σ_branch=quad:
       output = bw_0·(W[0,0]·x)² + bw_1·(W[0,1]·y)² + b
       → α·x² + β·y² + b → quadratic boundary → solves rings.

This is the cleanest falsification target the column-partition design
admits: K=1 provably cannot, K=2 closed-form can.

Three arms, n=5 seeds:

  K=1            — substrate stays K=1 throughout. Expected: ~50–70% acc
                   (best linear bisection of radially symmetric data).
  K=2_forced     — pre-grow to K=2 via grow_branch(0, [1]) before training.
                   Expected: high 90s.
  K=1->grown     — start K=1, call grow_branch at a fixed mid-training
                   step (proxy for the internal_stress trigger window).
                   Tests the structural K=1 → K=2 transition end-to-end.
                   Expected: matches K=2_forced asymptotically.

Output: outputs/bench_2_0_dendrite.csv + outputs/bench_2_0_dendrite_run.log.

Verdict criterion (the falsification gate): mean K=2_forced accuracy
must exceed mean K=1 accuracy by at least 20 percentage points across
all 5 seeds. If not, Axis 5 ships dormant per spec §6.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trioron.node import TrioronLayer, _EwcZeroWarning


# Disarm silent-zero EWC warning — this bench doesn't run a
# consolidation cycle, so λ stays zero by design. The warning would
# spam the log otherwise.
_EwcZeroWarning._warned = True


# ---------- task ----------

def make_rings(
    n_per_class: int,
    r_inner: float = 1.0,
    r_outer: float = 1.3,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample 2D points from two concentric rings. Returns (X, y) where
    y ∈ {0.0, 1.0} for BCEWithLogitsLoss."""
    gen = torch.Generator().manual_seed(seed)
    theta_in = torch.rand(n_per_class, generator=gen) * 2 * math.pi
    r_in = r_inner + torch.randn(n_per_class, generator=gen) * noise
    x_in = torch.stack(
        [r_in * torch.cos(theta_in), r_in * torch.sin(theta_in)], dim=1,
    )

    theta_out = torch.rand(n_per_class, generator=gen) * 2 * math.pi
    r_out = r_outer + torch.randn(n_per_class, generator=gen) * noise
    x_out = torch.stack(
        [r_out * torch.cos(theta_out), r_out * torch.sin(theta_out)], dim=1,
    )

    X = torch.cat([x_in, x_out], dim=0)
    y = torch.cat(
        [torch.zeros(n_per_class), torch.ones(n_per_class)], dim=0,
    )
    perm = torch.randperm(2 * n_per_class, generator=gen)
    return X[perm], y[perm]


# ---------- training loop ----------

def train_arm(
    layer: TrioronLayer,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    steps: int = 600,
    lr: float = 0.05,
    grow_at_step: int | None = None,
) -> tuple[float, float]:
    """Train a single-cell substrate to binary-classify rings. Optionally
    grow_branch at `grow_at_step`. Returns (final_test_acc, final_train_loss)."""
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for step in range(steps):
        logit = layer(X_train).squeeze(-1)
        loss = loss_fn(logit, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if grow_at_step is not None and step == grow_at_step:
            # Natural column partition: col 1 → branch 1.
            layer.grow_branch(node_idx=0, source_cols=[1])
            # Rebuild optimizer — branch_weight is the same Parameter
            # object (slot was preallocated at construction), so this
            # rebuild is conservative rather than required. Doing it
            # anyway in case the user adds insert_layer / grow_node
            # later inside the trigger window.
            opt = torch.optim.Adam(layer.parameters(), lr=lr)

    with torch.no_grad():
        logits = layer(X_test).squeeze(-1)
        preds = (logits > 0).long()
        acc = (preds == y_test.long()).float().mean().item()
    return acc, float(loss.item())


# ---------- driver ----------

def run_seed(seed: int) -> dict[str, dict[str, float]]:
    torch.manual_seed(seed)
    X_train, y_train = make_rings(n_per_class=300, seed=seed)
    X_test, y_test = make_rings(n_per_class=100, seed=seed + 10_000)

    out: dict[str, dict[str, float]] = {}

    # Arm 1: K=1 baseline. Output stays a linear function of (x, y);
    # quadratic boundary unattainable.
    torch.manual_seed(seed)
    layer_k1 = TrioronLayer(
        fan_in=2, n_nodes=1, activation="linear",
        branch_activation="quad",  # set but inert at K=1 (fast path)
    )
    acc, loss = train_arm(layer_k1, X_train, y_train, X_test, y_test)
    out["K=1"] = {"acc": acc, "loss": loss, "K_final": 1}

    # Arm 2: K=2 forced. Col 1 → branch 1 before training starts.
    torch.manual_seed(seed)
    layer_k2 = TrioronLayer(
        fan_in=2, n_nodes=1, activation="linear",
        branch_activation="quad",
    )
    layer_k2.grow_branch(node_idx=0, source_cols=[1])
    acc, loss = train_arm(layer_k2, X_train, y_train, X_test, y_test)
    out["K=2_forced"] = {"acc": acc, "loss": loss, "K_final": 2}

    # Arm 3: K=1 → grown. Start K=1, grow at step 200 (well after the
    # K=1 plateau has formed). Remaining 400 steps exercise the K>1
    # dendritic path.
    torch.manual_seed(seed)
    layer_grown = TrioronLayer(
        fan_in=2, n_nodes=1, activation="linear",
        branch_activation="quad",
    )
    acc, loss = train_arm(
        layer_grown, X_train, y_train, X_test, y_test,
        grow_at_step=200,
    )
    out["K=1->grown"] = {
        "acc": acc, "loss": loss,
        "K_final": int(layer_grown.B_per_node[0].item()),
    }
    return out


def main() -> None:
    t0 = time.time()
    n_seeds = 5
    all_results: dict[int, dict[str, dict[str, float]]] = {}

    log_lines: list[str] = []
    def log(msg: str) -> None:
        log_lines.append(msg)
        print(msg, flush=True)

    log("trioron 2.0 — Phase 6 dendrite delta on concentric rings")
    log(f"n_seeds={n_seeds}  task=2D rings (r_inner=1.0, r_outer=1.3, noise=0.05)")
    log(f"substrate=single trioron cell (fan_in=2, n_nodes=1, activation=linear)")
    log(f"arms: K=1 / K=2_forced / K=1->grown  steps=600 lr=0.05  σ_branch=quad")
    log("")

    for seed in range(n_seeds):
        all_results[seed] = run_seed(seed)
        r = all_results[seed]
        log(
            f"seed {seed}: "
            f"K=1 acc={r['K=1']['acc']:.4f} | "
            f"K=2_forced acc={r['K=2_forced']['acc']:.4f} | "
            f"K=1->grown acc={r['K=1->grown']['acc']:.4f} "
            f"(K_final={r['K=1->grown']['K_final']})"
        )

    log("")
    log("aggregate (mean ± std across seeds):")

    summary_rows: list[dict[str, str]] = []
    for arm in ("K=1", "K=2_forced", "K=1->grown"):
        accs = [all_results[s][arm]["acc"] for s in range(n_seeds)]
        mean = statistics.mean(accs)
        std = statistics.pstdev(accs)
        log(
            f"  {arm:14s}: mean={mean:.4f} ± {std:.4f}  "
            f"per-seed={[f'{a:.3f}' for a in accs]}"
        )
        summary_rows.append({
            "arm": arm,
            "mean_acc": f"{mean:.6f}",
            "std_acc": f"{std:.6f}",
            "per_seed_acc": ",".join(f"{a:.4f}" for a in accs),
        })

    # Falsification verdict.
    k1_mean = statistics.mean(
        all_results[s]["K=1"]["acc"] for s in range(n_seeds)
    )
    k2_mean = statistics.mean(
        all_results[s]["K=2_forced"]["acc"] for s in range(n_seeds)
    )
    grown_mean = statistics.mean(
        all_results[s]["K=1->grown"]["acc"] for s in range(n_seeds)
    )
    delta_forced = k2_mean - k1_mean
    delta_grown = grown_mean - k1_mean
    log("")
    log("falsification gate (per spec §6):")
    log(f"  ΔK=2_forced  - K=1 = {delta_forced:+.4f}  (need ≥ +0.20 to PASS)")
    log(f"  ΔK=1->grown  - K=1 = {delta_grown:+.4f}  (sanity: should ~match forced)")
    verdict = "PASS" if delta_forced >= 0.20 else "FAIL — Axis 5 ships dormant per §6"
    log(f"  verdict: {verdict}")
    log(f"")
    log(f"elapsed: {time.time() - t0:.1f}s")

    # Write outputs.
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "bench_2_0_dendrite.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["arm", "mean_acc", "std_acc", "per_seed_acc"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    log_path = out_dir / "bench_2_0_dendrite_run.log"
    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"\nwrote {csv_path}")
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()
