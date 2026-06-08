"""Cell-first CL PoC — scaled up: 5 tasks, 6x6 grid, longer curriculum.

Extends cell_first_cl_poc.py to test:
    1. Does retention hold across 5 sequential tasks?
    2. Does the substrate keep recruiting hidden cells as tasks arrive?
    3. How does the per-task hidden-cell budget evolve?

Setup:
    6x6 locus grid (36 cells).
    2 inputs at (0, 0), (0, 5).
    5 outputs at (5, 1), (5, 2), (5, 3), (5, 4), (5, 5) — clustered
        along the bottom edge so frustration fields fan out from a
        shared row.
    29 cells start potential.

Curriculum:
    Task 1: linear      y = sign(x_0 + x_1)
    Task 2: XOR         y = sign(x_0) XOR sign(x_1)
    Task 3: rings       y = sign(x_0² + x_1² − 0.5)
    Task 4: diamond     y = sign(|x_0| + |x_1| − 0.7)
    Task 5: stripe      y = sign(sin(πx_0) − 0)   — nonmonotonic, multi-stripe

The 4th and 5th tasks are progressively harder geometric boundaries.
Stripe is genuinely hard: alternating-sign decision boundary, hard for
a 1-hidden-cell net to approximate.
"""
from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class CellSubstrate(torch.nn.Module):
    def __init__(
        self,
        grid_size: int = 6,
        field_sigma: float = 1.5,
        input_positions: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 5)),
        output_positions: Tuple[Tuple[int, int], ...] = (
            (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
        ),
    ):
        super().__init__()
        self.grid_size = grid_size
        n_cells = grid_size * grid_size
        positions = torch.tensor(
            [(i, j) for i in range(grid_size) for j in range(grid_size)],
            dtype=torch.float32,
        )
        self.register_buffer("positions", positions)
        def _flatten(coord):
            return coord[0] * grid_size + coord[1]
        self.input_idx: List[int] = [_flatten(c) for c in input_positions]
        self.output_idx: List[int] = [_flatten(c) for c in output_positions]
        self.roles: List[str] = ["potential"] * n_cells
        for i in self.input_idx:
            self.roles[i] = "input"
        for o in self.output_idx:
            self.roles[o] = "output"
        self.incoming: Dict[int, List[int]] = {i: [] for i in range(n_cells)}
        for o in self.output_idx:
            for i in self.input_idx:
                self.incoming[o].append(i)
        self.W = torch.nn.ParameterDict()
        self.b = torch.nn.ParameterDict()
        for o in self.output_idx:
            fan = len(self.incoming[o])
            W_init = torch.empty(fan)
            torch.nn.init.uniform_(W_init, -0.5, 0.5)
            self.W[f"c{o}"] = torch.nn.Parameter(W_init)
            self.b[f"c{o}"] = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("internal_stress", torch.zeros(n_cells))
        self.register_buffer("activity", torch.zeros(n_cells))
        self.register_buffer("epi_A", torch.zeros(n_cells))
        self.register_buffer("epi_B", torch.zeros(n_cells))
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        dist = diff.norm(dim=-1)
        kernel = torch.exp(-(dist / field_sigma) ** 2)
        kernel.fill_diagonal_(0.0)
        kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("kernel", kernel)
        self._cell_outputs: Optional[Dict[int, torch.Tensor]] = None

    def n_cells(self) -> int:
        return self.grid_size ** 2

    def _params(self, cell: int):
        key = f"c{cell}"
        if key in self.W:
            return self.W[key], self.b[key]
        return None, None

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        outputs = self._compute_all(x)
        self._cell_outputs = outputs
        idx = self.output_idx[task_id if task_id is not None else 0]
        return outputs[idx]

    def forward_all_outputs(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        outputs = self._compute_all(x)
        self._cell_outputs = outputs
        return {o: outputs[o] for o in self.output_idx}

    def _compute_all(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        batch = x.shape[0]
        device = x.device
        outputs: Dict[int, torch.Tensor] = {}
        for k, i in enumerate(self.input_idx):
            outputs[i] = x[:, k]
        for c in range(self.n_cells()):
            if c in outputs:
                continue
            W, _ = self._params(c)
            if W is None and not self.incoming[c]:
                outputs[c] = torch.zeros(batch, device=device)
        pending = set(range(self.n_cells())) - set(outputs.keys())
        for _ in range(self.n_cells() + 2):
            if not pending:
                break
            ready = [c for c in pending
                     if all(s in outputs for s in self.incoming[c])]
            if not ready:
                for c in pending:
                    outputs[c] = torch.zeros(batch, device=device)
                pending.clear()
                break
            for c in ready:
                W, b = self._params(c)
                if W is None:
                    outputs[c] = torch.zeros(batch, device=device)
                    continue
                upstream = torch.stack([outputs[s] for s in self.incoming[c]], dim=1)
                z = upstream @ W + b
                if self.roles[c] == "hidden":
                    outputs[c] = F.relu(z.squeeze(-1) if z.dim() > 1 else z)
                else:
                    outputs[c] = z.squeeze(-1) if z.dim() > 1 else z
            pending -= set(ready)
        return outputs

    def update_signals(self, loss: torch.Tensor, task_id: int) -> None:
        with torch.no_grad():
            current_loss = float(loss.detach().item())
            active_output = self.output_idx[task_id]
            for c in range(self.n_cells()):
                if c == active_output:
                    self.internal_stress[c] = (
                        0.8 * self.internal_stress[c].item() + 0.2 * current_loss
                    )
                else:
                    self.internal_stress[c] *= 0.9
            if self._cell_outputs is not None:
                for c, y in self._cell_outputs.items():
                    if c in self.input_idx:
                        self.activity[c] = 0.0
                        continue
                    W, _ = self._params(c)
                    if W is None:
                        self.activity[c] = 0.0
                        continue
                    a = y.detach().abs().mean().item()
                    self.activity[c] = (
                        0.7 * self.activity[c].item() + 0.3 * a
                    )

    def diffuse_and_accumulate(
        self,
        dt: float = 0.1,
        stress_tolerance: float = 0.1,
    ) -> None:
        with torch.no_grad():
            ambient_A = self.kernel @ self.internal_stress
            self.epi_A += dt * (ambient_A - stress_tolerance)
            self.epi_A.clamp_(min=0.0)
            self.epi_B += dt * self.activity

    def differentiate(
        self,
        threshold_A: float = 0.5,
        threshold_B: float = 0.2,
        mode: str = "absolute",
        k: float = 2.0,
        stress_floor: float = 0.05,
    ) -> List[int]:
        """Latch potential cells to hidden when neighborhood stress is high
        and self-activity is low.

        Two modes:
            absolute: epi_A > threshold_A. Fixed bar. Used by 4x4 PoCs.
            relative: epi_A > mean(epi_A over potentials) + k*std, AND
                epi_A > stress_floor. Auto-scales to grid size; the floor
                suppresses noise-firing when the substrate is quiet.
        """
        new_hidden: List[int] = []
        potentials = [c for c in range(self.n_cells()) if self.roles[c] == "potential"]
        if not potentials:
            return new_hidden
        if mode == "relative":
            epi_A_pool = self.epi_A[potentials]
            mu = float(epi_A_pool.mean().item())
            sd = float(epi_A_pool.std(unbiased=False).item())
            if len(potentials) < 3 or sd < 1e-6:
                return new_hidden
            bar = mu + k * sd
            for c in potentials:
                a = self.epi_A[c].item()
                b = self.epi_B[c].item()
                if a > bar and a > stress_floor and b < threshold_B:
                    self._make_hidden(c)
                    new_hidden.append(c)
            return new_hidden
        for c in potentials:
            if (self.epi_A[c].item() > threshold_A
                    and self.epi_B[c].item() < threshold_B):
                self._make_hidden(c)
                new_hidden.append(c)
        return new_hidden

    def _make_hidden(self, idx: int) -> None:
        my_pos = self.positions[idx]
        n = self.n_cells()
        in_scores: List[Tuple[float, int]] = []
        for j in range(n):
            if j == idx:
                continue
            is_input = j in self.input_idx
            has_W = f"c{j}" in self.W
            if not (is_input or has_W):
                continue
            d = (self.positions[j] - my_pos).norm().item()
            score = 1.0 / (d + 0.5)
            if is_input:
                score += 0.5
            in_scores.append((score, j))
        in_scores.sort(reverse=True)
        incoming = [j for _, j in in_scores[:2]]
        out_scores: List[Tuple[float, int]] = []
        for j in range(n):
            if j == idx or j in incoming:
                continue
            if f"c{j}" not in self.W:
                continue
            d = (self.positions[j] - my_pos).norm().item()
            score = self.internal_stress[j].item() / (d + 0.5)
            out_scores.append((score, j))
        out_scores.sort(reverse=True)
        target = out_scores[0][1] if out_scores else None
        self.roles[idx] = "hidden"
        self.incoming[idx] = incoming
        W_new = torch.empty(len(incoming))
        torch.nn.init.uniform_(W_new, -0.5, 0.5)
        self.W[f"c{idx}"] = torch.nn.Parameter(W_new)
        self.b[f"c{idx}"] = torch.nn.Parameter(torch.zeros(1))
        if target is not None:
            old_W = self.W[f"c{target}"].detach()
            new_target_W = torch.cat([old_W, torch.tensor([0.1])])
            self.W[f"c{target}"] = torch.nn.Parameter(new_target_W)
            self.incoming[target].append(idx)

    def summarize_per_output(self, tasks: List[Tuple[str, ...]]) -> str:
        """Per-output: how many hidden cells are nearest to each output."""
        lines = []
        for k, o in enumerate(self.output_idx):
            name = tasks[k][0]
            my_hidden = []
            for c in range(self.n_cells()):
                if self.roles[c] != "hidden":
                    continue
                my_pos = self.positions[c]
                dists = [(self.positions[oo] - my_pos).norm().item() for oo in self.output_idx]
                nearest = self.output_idx[int(torch.tensor(dists).argmin())]
                if nearest == o:
                    my_hidden.append(c)
            inc = self.incoming[o]
            lines.append(
                f"  output {o:>2} (task '{name}', pos={tuple(self.positions[o].int().tolist())}) "
                f"hidden_neighbors={my_hidden} "
                f"incoming={inc}"
            )
        return "\n".join(lines)


def make_curriculum(n: int = 300, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 2, generator=g) * 2.0 - 1.0  # [-1, 1]^2
    tasks = [
        ("linear", ((X[:, 0] + X[:, 1]) > 0).float() * 2.0 - 1.0),
        ("XOR",    ((X[:, 0] > 0) ^ (X[:, 1] > 0)).float() * 2.0 - 1.0),
        ("rings",  ((X[:, 0] ** 2 + X[:, 1] ** 2) > 0.5).float() * 2.0 - 1.0),
        ("diamond", ((X[:, 0].abs() + X[:, 1].abs()) > 0.7).float() * 2.0 - 1.0),
        ("stripe",  (torch.sin(math.pi * X[:, 0]) > 0).float() * 2.0 - 1.0),
    ]
    return X, tasks


def train_task(
    sub: CellSubstrate,
    X: torch.Tensor,
    y: torch.Tensor,
    task_id: int,
    n_epochs: int,
    lr: float,
    label: str,
    threshold_A: float,
    threshold_B: float,
    diff_mode: str = "absolute",
    diff_k: float = 2.0,
    diff_stress_floor: float = 0.05,
) -> None:
    opt = torch.optim.Adam(sub.parameters(), lr=lr)
    for epoch in range(n_epochs):
        opt.zero_grad()
        pred = sub(X, task_id=task_id)
        loss = F.mse_loss(pred, y)
        loss.backward()
        sub.update_signals(loss, task_id=task_id)
        opt.step()
        sub.diffuse_and_accumulate()
        new_hidden = sub.differentiate(
            threshold_A=threshold_A,
            threshold_B=threshold_B,
            mode=diff_mode,
            k=diff_k,
            stress_floor=diff_stress_floor,
        )
        if new_hidden:
            opt = torch.optim.Adam(sub.parameters(), lr=lr)
            print(
                f"  [{label}] epoch {epoch:>3}: NEW HIDDEN cells {new_hidden} "
                f"@ {[tuple(sub.positions[c].int().tolist()) for c in new_hidden]}"
            )
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            n_hidden = sum(1 for r in sub.roles if r == "hidden")
            acc = ((pred.sign() == y.sign()).float().mean().item())
            print(
                f"  [{label}] epoch {epoch:>3}: loss={loss.item():.4f}  "
                f"acc={acc:.3f}  hidden_total={n_hidden}"
            )


def evaluate_all(
    sub: CellSubstrate, X: torch.Tensor, tasks: List[Tuple[str, torch.Tensor]],
) -> List[float]:
    with torch.no_grad():
        outs = sub.forward_all_outputs(X)
    accs = []
    for k, (label, y) in enumerate(tasks):
        pred = outs[sub.output_idx[k]]
        accs.append((pred.sign() == y.sign()).float().mean().item())
    return accs


def main() -> int:
    import os
    torch.manual_seed(0)
    diff_mode = os.environ.get("CELL_DIFF_MODE", "absolute")
    diff_k = float(os.environ.get("CELL_DIFF_K", "2.0"))
    diff_floor = float(os.environ.get("CELL_DIFF_FLOOR", "0.05"))
    print("=" * 88)
    print("Cell-first CL PoC — 5 tasks, 6x6 grid, retention test")
    print(f"diff_mode={diff_mode}  diff_k={diff_k}  diff_floor={diff_floor}")
    print("=" * 88)
    sub = CellSubstrate(grid_size=6, field_sigma=1.5)
    X, tasks = make_curriculum(n=300, seed=0)
    print(f"Setup:")
    print(f"  grid:         {sub.grid_size}×{sub.grid_size} = {sub.n_cells()} cells")
    print(f"  inputs:       {sub.input_idx} at {[tuple(sub.positions[i].int().tolist()) for i in sub.input_idx]}")
    print(f"  outputs:      {sub.output_idx} at {[tuple(sub.positions[o].int().tolist()) for o in sub.output_idx]}")
    print(f"  tasks:        {[t[0] for t in tasks]}")
    print()

    epochs_per_task = [200, 400, 400, 400, 400]
    accs_after_phase: List[Tuple[str, List[float]]] = []
    hidden_count_after_phase: List[int] = []
    for task_id, ((name, y), n_ep) in enumerate(zip(tasks, epochs_per_task)):
        print(f"[Phase {task_id+1}: training task '{name}']")
        train_task(
            sub, X, y, task_id=task_id, n_epochs=n_ep, lr=0.05, label=name,
            threshold_A=0.5, threshold_B=0.2,
            diff_mode=diff_mode, diff_k=diff_k, diff_stress_floor=diff_floor,
        )
        accs = evaluate_all(sub, X, tasks)
        accs_after_phase.append((name, accs))
        hidden_count_after_phase.append(sum(1 for r in sub.roles if r == "hidden"))
        print(f"  >> after phase {task_id+1} ({name}): "
              f"acc = {[f'{a:.3f}' for a in accs]}  "
              f"hidden_total={hidden_count_after_phase[-1]}\n")

    print("=" * 88)
    print("Retention matrix (rows = after phase, cols = per task)")
    print("=" * 88)
    header = f"  {'phase':<16}" + "".join(f" {tasks[i][0]:>8}" for i in range(len(tasks)))
    print(header)
    for k, (name, accs) in enumerate(accs_after_phase):
        line = f"  after {name:<10} " + "".join(f" {a:>8.3f}" for a in accs)
        print(line)
    print(f"  {'hidden_total':<16}  {'':>40}      counts: {hidden_count_after_phase}")

    print()
    print("=" * 88)
    print("Per-output hidden-cell distribution")
    print("=" * 88)
    print(sub.summarize_per_output(tasks))

    final_accs = accs_after_phase[-1][1]
    initial_accs_per_task: List[float] = [None] * len(tasks)
    for k in range(len(tasks)):
        initial_accs_per_task[k] = accs_after_phase[k][1][k]
    drift = [initial_accs_per_task[k] - final_accs[k] for k in range(len(tasks))]
    print()
    print(f"Per-task drift (acc when first learned − acc at end):")
    for k, (name, _) in enumerate(tasks):
        print(f"  {name:<10}: learned at {initial_accs_per_task[k]:.3f}, final {final_accs[k]:.3f},  drift = {drift[k]:+.3f}")
    max_drift = max(abs(d) for d in drift)
    print()
    if max_drift < 0.05:
        print(f"VERDICT: no observable forgetting across 5 tasks (max drift {max_drift:.3f})")
    elif max_drift < 0.15:
        print(f"VERDICT: minor drift (max {max_drift:.3f})")
    else:
        print(f"VERDICT: significant forgetting (max drift {max_drift:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
