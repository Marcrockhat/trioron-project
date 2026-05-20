"""Trioron 2.0 — multi-class within-niche bench, the harder Phase 6 cousin.

bench_2_0_dendrite.py (2-class concentric rings) was the closed-form
falsification gate: K=1 provably cannot separate radially-symmetric
data, K=2 quad recovers the optimal r² boundary. This bench escalates
to 4 concentric rings with a 4-cell L1 substrate + linear head — a
multi-class within-niche task where cell-specialization variance enters
the picture alongside the K-vs-K signal.

Hypothesis: K=2 quad lifts K=1 noticeably (the +0.49 abs from the
2-class bench should hold or strengthen at 4 classes), AND the
K=1 → trigger-grown arm approaches K=2 forced when the
internal_frustration_candidates trigger fires correctly.

If K=1 → trigger-grown matches K=2 forced within noise, Phase 2.5's
trigger machinery is empirically validated end-to-end on a multi-cell
substrate with real cross-cell competition for branch-growth budget.
If K=1 → trigger-grown lags K=2 forced, the trigger thresholds or the
forced-partition heuristic need work.

Substrate:
  L1: TrioronLayer(fan_in=2, n_nodes=4, activation="linear",
                   branch_activation="quad")
  head: nn.Linear(4, 4)

Loss: cross-entropy on 4-class logits.

Arms, n=10 seeds each:
  K=1            — substrate stays K=1 throughout. Each L1 cell is a
                   linear function of (x, y); the head linearly
                   combines 4 such features. Expected: poor (single
                   hyperplane per cell, no quadratic features).
  K=2_forced     — pre-grow every L1 cell to K=2 via grow_branch(i, [1]).
                   With σ_branch=quad and natural column partition,
                   each cell computes α·x² + β·y² (a quadratic
                   feature). Expected: near-ceiling.
  K=1->trigger   — start K=1. Every PROBE_EVERY steps, call
                   internal_frustration_candidates() on L1 and
                   grow_branch on returned cells. Tests Phase 2.5's
                   trigger machinery on a real multi-cell substrate.

Output: outputs/bench_2_0_dendrite_real.csv +
        outputs/bench_2_0_dendrite_real_run.log
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trioron.node import TrioronLayer, _EwcZeroWarning


_EwcZeroWarning._warned = True


# ---------- task ----------

def make_4rings(
    n_per_class: int,
    radii: tuple[float, ...] = (1.0, 1.3, 1.6, 1.9),
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    X_chunks: list[torch.Tensor] = []
    y_chunks: list[torch.Tensor] = []
    for c, r in enumerate(radii):
        theta = torch.rand(n_per_class, generator=gen) * 2 * math.pi
        rr = r + torch.randn(n_per_class, generator=gen) * noise
        X_chunks.append(torch.stack(
            [rr * torch.cos(theta), rr * torch.sin(theta)], dim=1,
        ))
        y_chunks.append(torch.full((n_per_class,), c, dtype=torch.long))
    X = torch.cat(X_chunks, dim=0)
    y = torch.cat(y_chunks, dim=0)
    perm = torch.randperm(len(X), generator=gen)
    return X[perm], y[perm]


# ---------- substrate ----------

class FourRingClassifier(nn.Module):
    """L1 trioron layer + linear head."""

    def __init__(self, branch_activation: str = "quad"):
        super().__init__()
        self.l1 = TrioronLayer(
            fan_in=2, n_nodes=4, activation="linear",
            branch_activation=branch_activation,
        )
        self.head = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.l1(x))


# ---------- training loop ----------

def train_arm(
    model: FourRingClassifier,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    steps: int = 1000,
    lr: float = 0.05,
    grow_mode: str = "none",
    probe_every: int = 100,
    trigger_threshold: float = 0.01,
    trigger_ceiling: float = 1e9,
) -> dict:
    """Train end-to-end. grow_mode ∈ {none, forced, trigger}.

    - "none":     no dendritic growth; L1 stays K=1.
    - "forced":   pre-grow every L1 cell to K=2 with col 1 → branch 1
                  (already done before this call by the caller).
    - "trigger":  every `probe_every` steps, call
                  internal_frustration_candidates() and grow_branch on
                  the returned cells (column partition: col 1 → branch 1).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    grow_events: list[int] = []

    for step in range(steps):
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        opt.zero_grad()
        loss.backward()
        # Always update internal_stress so the trigger arm has data
        # to consult. Mode-independent — cheap when no growth follows.
        model.l1.update_internal_stress()
        opt.step()

        if grow_mode == "trigger" and step > 0 and step % probe_every == 0:
            cands = model.l1.internal_frustration_candidates(
                threshold=trigger_threshold,
                overall_saliency_ceiling=trigger_ceiling,
            )
            for cell_idx in cands:
                # Each cell can only grow once in this bench (K=2 cap is
                # enforced naturally because we always use col 1 → branch 1).
                if int(model.l1.B_per_node[cell_idx].item()) == 1:
                    try:
                        model.l1.grow_branch(node_idx=cell_idx, source_cols=[1])
                        grow_events.append(step)
                    except ValueError:
                        pass
            # No optimizer rebuild: grow_branch is buffer-only (the
            # branch_weight Parameter object is unchanged), so Adam
            # state stays valid. Newly-activated branch_weight slots
            # naturally cold-start from zero momentum (their gradient
            # was zero pre-grow, so Adam state was zero anyway).
            # Rebuilding would handicap the trigger arm by resetting
            # W / b / head momentum vs the forced arm.

    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
    K_final_per_cell = model.l1.B_per_node.tolist()
    return {
        "acc": acc,
        "loss": float(loss.item()),
        "K_final_per_cell": K_final_per_cell,
        "n_grow_events": len(grow_events),
        "grow_steps": grow_events,
    }


# ---------- driver ----------

def run_seed(seed: int) -> dict:
    torch.manual_seed(seed)
    X_train, y_train = make_4rings(n_per_class=200, seed=seed)
    X_test, y_test = make_4rings(n_per_class=100, seed=seed + 10_000)

    out: dict = {}

    # Arm 1: K=1.
    torch.manual_seed(seed)
    m1 = FourRingClassifier(branch_activation="quad")
    out["K=1"] = train_arm(m1, X_train, y_train, X_test, y_test,
                           grow_mode="none")

    # Arm 2: K=2 forced.
    torch.manual_seed(seed)
    m2 = FourRingClassifier(branch_activation="quad")
    for i in range(4):
        m2.l1.grow_branch(node_idx=i, source_cols=[1])
    out["K=2_forced"] = train_arm(m2, X_train, y_train, X_test, y_test,
                                  grow_mode="forced")

    # Arm 3: K=1 → trigger-grown.
    torch.manual_seed(seed)
    m3 = FourRingClassifier(branch_activation="quad")
    # Threshold tuned to fire on K=1 cells that are clearly engaged
    # (high upstream grad through linear σ_soma → engaged gate triggers
    # on |y| > 0.05; with random init most cells qualify by step 100).
    out["K=1->trigger"] = train_arm(
        m3, X_train, y_train, X_test, y_test,
        grow_mode="trigger",
        probe_every=100,
        trigger_threshold=0.001,   # any nonzero internal stress qualifies
        trigger_ceiling=1e9,       # never block on saliency_utility (always 0
                                    # here — no graph captured for the
                                    # linear-activation engagement)
    )
    return out


def main() -> None:
    t0 = time.time()
    n_seeds = 10
    all_results: dict[int, dict] = {}
    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        print(msg, flush=True)

    log("trioron 2.0 — Phase 6 multi-class within-niche bench")
    log("task: 4 concentric rings (radii 1.0/1.3/1.6/1.9, noise=0.05)")
    log("substrate: L1 TrioronLayer(fan_in=2, n_nodes=4, linear) + Linear(4,4) head")
    log(f"arms: K=1 / K=2_forced / K=1->trigger    n_seeds={n_seeds}")
    log("")

    for seed in range(n_seeds):
        all_results[seed] = run_seed(seed)
        r = all_results[seed]
        log(
            f"seed {seed:2d}: K=1 acc={r['K=1']['acc']:.4f} | "
            f"K=2_forced acc={r['K=2_forced']['acc']:.4f} | "
            f"K=1->trigger acc={r['K=1->trigger']['acc']:.4f} "
            f"(K_final={r['K=1->trigger']['K_final_per_cell']}, "
            f"grow_events={r['K=1->trigger']['n_grow_events']})"
        )

    log("")
    log("aggregate (mean ± std across seeds):")

    summary_rows: list[dict] = []
    for arm in ("K=1", "K=2_forced", "K=1->trigger"):
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

    k1_mean = statistics.mean(
        all_results[s]["K=1"]["acc"] for s in range(n_seeds)
    )
    k2_mean = statistics.mean(
        all_results[s]["K=2_forced"]["acc"] for s in range(n_seeds)
    )
    trig_mean = statistics.mean(
        all_results[s]["K=1->trigger"]["acc"] for s in range(n_seeds)
    )
    delta_forced = k2_mean - k1_mean
    delta_trigger = trig_mean - k1_mean
    delta_trigger_vs_forced = trig_mean - k2_mean

    log("")
    log("findings:")
    log(f"  Δ(K=2_forced  − K=1)        = {delta_forced:+.4f}")
    log(f"  Δ(K=1->trigger − K=1)       = {delta_trigger:+.4f}")
    log(f"  Δ(K=1->trigger − K=2_forced) = {delta_trigger_vs_forced:+.4f}")

    log("")
    log("interpretation:")
    if delta_forced >= 0.15:
        log("  K=2 forced beats K=1 by ≥0.15 abs → architectural lift "
            "confirmed at 4-class within-niche scale.")
    elif delta_forced >= 0.05:
        log("  K=2 forced beats K=1 by 0.05-0.15 → real but modest "
            "architectural lift; gap may close with longer training "
            "or more cells.")
    else:
        log("  K=2 forced ≤ K=1 + 0.05 → no architectural lift at this "
            "task scale. The K=1 baseline already saturates the rings "
            "(4 ReLU cells + linear head may suffice).")

    if abs(delta_trigger_vs_forced) <= 0.05:
        log("  K=1->trigger matches K=2_forced within ±0.05 → Phase 2.5 "
            "trigger machinery validates end-to-end.")
    elif delta_trigger_vs_forced < -0.05:
        log("  K=1->trigger lags K=2_forced by >0.05 → trigger thresholds "
            "or growth heuristic need tuning. Forced partition (col 1 → "
            "branch 1) is reaching cells the trigger misses.")
    else:
        log("  K=1->trigger beats K=2_forced (unexpected) → growth "
            "schedule may help convergence; investigate seed-by-seed.")

    log(f"")
    log(f"elapsed: {time.time() - t0:.1f}s")

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "bench_2_0_dendrite_real.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["arm", "mean_acc", "std_acc", "per_seed_acc"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    log_path = out_dir / "bench_2_0_dendrite_real_run.log"
    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"\nwrote {csv_path}")
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()
