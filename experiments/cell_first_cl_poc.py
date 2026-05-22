"""Cell-first CL PoC: multi-task substrate with per-task output cells.

Extends cell_first_poc.py to a 3-task curriculum:
    Task 1: linear separable      sign(x[0] + x[1])
    Task 2: XOR                   sign(x[0]) * sign(x[1])
    Task 3: concentric rings      sign(x[0]^2 + x[1]^2 − 0.5)

Same inputs (x in [-1, 1]^2), three different labelings. Each task
uses a different output cell. The substrate's frustration field
localizes per output — so different tasks should grow hidden cells
in different spatial neighborhoods — natural task isolation by
construction.

Setup:
    4x4 locus grid (16 cells). 2 inputs at (0, 0) and (0, 3); 3
    outputs at (3, 0), (3, 1), (3, 2); remaining 11 cells potential.

CL behavior under test:
    1. Per-task accuracy stays high across the full curriculum (no
       catastrophic forgetting).
    2. Distinct hidden-cell neighborhoods per task (spatial isolation
       from frustration-field locality).
    3. Total hidden cell count scales with task nonlinearity (Task 1
       linear → 0 hidden, Tasks 2 & 3 nonlinear → some hidden).

This is the existence proof for cell-first CL: if it works, the
substrate handles continual learning natively without any of the
manifold-replay machinery propping up trioron 1.0.
"""
from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Substrate
# ---------------------------------------------------------------------


class CellSubstrate(torch.nn.Module):
    def __init__(
        self,
        grid_size: int = 4,
        field_sigma: float = 1.2,
        input_positions: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 3)),
        output_positions: Tuple[Tuple[int, int], ...] = ((3, 0), (3, 1), (3, 2)),
    ):
        super().__init__()
        self.grid_size = grid_size
        n_cells = grid_size * grid_size
        positions = torch.tensor(
            [(i, j) for i in range(grid_size) for j in range(grid_size)],
            dtype=torch.float32,
        )
        self.register_buffer("positions", positions)
        # Designate roles by grid position.
        def _flatten(coord):
            return coord[0] * grid_size + coord[1]
        self.input_idx: List[int] = [_flatten(c) for c in input_positions]
        self.output_idx: List[int] = [_flatten(c) for c in output_positions]
        self.roles: List[str] = ["potential"] * n_cells
        for i in self.input_idx:
            self.roles[i] = "input"
        for o in self.output_idx:
            self.roles[o] = "output"
        # Connectivity. Initial: output cells read from all input cells
        # (the dumb single-layer baseline).
        self.incoming: Dict[int, List[int]] = {i: [] for i in range(n_cells)}
        for o in self.output_idx:
            for i in self.input_idx:
                self.incoming[o].append(i)
        # Per-cell W, b. Only cells whose role admits a forward (output,
        # hidden) get parameters. Input cells get their value clamped
        # from data; potential cells output zero until they latch.
        self.W = torch.nn.ParameterDict()
        self.b = torch.nn.ParameterDict()
        for o in self.output_idx:
            fan = len(self.incoming[o])
            W_init = torch.empty(fan)
            torch.nn.init.uniform_(W_init, -0.5, 0.5)
            self.W[f"c{o}"] = torch.nn.Parameter(W_init)
            self.b[f"c{o}"] = torch.nn.Parameter(torch.zeros(1))
        # Per-cell signal state.
        self.register_buffer("internal_stress", torch.zeros(n_cells))
        self.register_buffer("activity", torch.zeros(n_cells))
        self.register_buffer("epi_A", torch.zeros(n_cells))
        self.register_buffer("epi_B", torch.zeros(n_cells))
        # Field kernel — Gaussian over locus distance, with the self-
        # diagonal zeroed so a cell doesn't diffuse its own signal back
        # to itself (self-stress already drives the cell's own behavior;
        # the field is for telling NEIGHBORS something). Row-normalized
        # after zeroing so rows still sum to 1.
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        dist = diff.norm(dim=-1)
        kernel = torch.exp(-(dist / field_sigma) ** 2)
        kernel.fill_diagonal_(0.0)
        kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("kernel", kernel)
        # Cache last forward's outputs (cell_idx -> (batch,) tensor).
        self._cell_outputs: Optional[Dict[int, torch.Tensor]] = None

    def n_cells(self) -> int:
        return self.grid_size ** 2

    def _params(self, cell: int) -> Tuple[Optional[torch.nn.Parameter], Optional[torch.nn.Parameter]]:
        key = f"c{cell}"
        if key in self.W:
            return self.W[key], self.b[key]
        return None, None

    # ----- forward (dynamic topo-sort) -----

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """x: (batch, len(input_idx)). Returns the active task's output cell value.

        task_id: index into self.output_idx; None = use 0 (first output).
        For all-tasks eval, call .forward_all_outputs().
        """
        outputs = self._compute_all(x)
        self._cell_outputs = outputs
        idx = self.output_idx[task_id if task_id is not None else 0]
        return outputs[idx]

    def forward_all_outputs(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Compute all cells once and return a dict of {output_cell_idx: value}."""
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
        max_iters = self.n_cells() + 2
        for _ in range(max_iters):
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

    # ----- signals & field -----

    def update_signals(self, loss: torch.Tensor, task_id: int) -> None:
        """Update per-cell internal_stress and activity.

        Stress is broadcast ONLY from the active task's output cell.
        Other output cells decay their stress (their task isn't being
        trained right now). This localizes the frustration field to
        the active task's neighborhood, which is the mechanism that
        gives per-task spatial isolation of hidden cells.
        """
        with torch.no_grad():
            current_loss = float(loss.detach().item())
            active_output = self.output_idx[task_id]
            for c in range(self.n_cells()):
                if c == active_output:
                    self.internal_stress[c] = (
                        0.8 * self.internal_stress[c].item() + 0.2 * current_loss
                    )
                else:
                    # Inactive outputs and all non-output cells decay
                    # their stress. Frustration is a per-task signal.
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
        """Diffuse per-cell signals to ambient fields, accumulate epi-state.

        epi_A: integrates (ambient_A - stress_tolerance). Stays flat when
               ambient is below tolerance (the substrate is doing OK in
               this neighborhood); grows when ambient exceeds tolerance
               (chronic frustration in the neighborhood). Clamped at >=0.
        epi_B: integrates SELF activity (not ambient). A potential cell
               with no W has activity=0 always, so its epi_B stays 0.
               This decouples the gate from neighborhood activity, which
               was contaminating the "stressed-but-quiet" signal: every
               cell near the output was being bathed in the output's
               activity broadcast and never qualified as quiet.
        """
        with torch.no_grad():
            ambient_A = self.kernel @ self.internal_stress
            self.epi_A += dt * (ambient_A - stress_tolerance)
            self.epi_A.clamp_(min=0.0)
            self.epi_B += dt * self.activity

    # ----- differentiation -----

    def differentiate(
        self,
        threshold_A: float = 0.5,
        threshold_B: float = 0.2,
    ) -> List[int]:
        """Transition potential cells to hidden when (epi_A high, epi_B low)."""
        new_hidden: List[int] = []
        for c in range(self.n_cells()):
            if self.roles[c] != "potential":
                continue
            if (self.epi_A[c].item() > threshold_A
                    and self.epi_B[c].item() < threshold_B):
                self._make_hidden(c)
                new_hidden.append(c)
        return new_hidden

    def _make_hidden(self, idx: int) -> None:
        """Form connections and parameters for a newly-hidden cell."""
        my_pos = self.positions[idx]
        n = self.n_cells()
        # Incoming: eligible sources are (a) input cells (always — they're
        # the natural sensors), and (b) other cells that have a W and so
        # produce a computed value. Score = 1/(distance + 0.5); inputs
        # get a constant +0.5 bonus so they're not down-weighted simply
        # because the locus put them far from the new cell. Pick top-2.
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
                score += 0.5    # bias toward sensors
            in_scores.append((score, j))
        in_scores.sort(reverse=True)
        incoming = [j for _, j in in_scores[:2]]
        # Outgoing target: top-1 by frustration / distance, restricted
        # to cells that already have a W (output + previously-hidden).
        # A potential cell has no W to extend; we'd need to also make
        # IT hidden first, which complicates ordering. Restricting to
        # W-having cells means the first wave of hidden cells all wire
        # into the output, the second wave can wire into the first
        # wave, and so on.
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
        # Apply.
        self.roles[idx] = "hidden"
        self.incoming[idx] = incoming
        W_new = torch.empty(len(incoming))
        torch.nn.init.uniform_(W_new, -0.5, 0.5)
        self.W[f"c{idx}"] = torch.nn.Parameter(W_new)
        self.b[f"c{idx}"] = torch.nn.Parameter(torch.zeros(1))
        # Extend target's incoming and W by one slot for the new hidden cell.
        if target is not None:
            old_W = self.W[f"c{target}"].detach()
            new_target_W = torch.cat([old_W, torch.tensor([0.1])])
            self.W[f"c{target}"] = torch.nn.Parameter(new_target_W)
            self.incoming[target].append(idx)

    # ----- diagnostics -----

    def summarize(self, include_potential: bool = False) -> str:
        lines = []
        for c in range(self.n_cells()):
            r = self.roles[c]
            if r == "potential" and not include_potential:
                continue
            pos = tuple(self.positions[c].int().tolist())
            inc = self.incoming.get(c, [])
            lines.append(
                f"  cell {c:>2} ({r:<9}) pos={pos} "
                f"epi_A={self.epi_A[c].item():>6.3f} "
                f"epi_B={self.epi_B[c].item():>6.3f} "
                f"stress={self.internal_stress[c].item():>6.3f} "
                f"activity={self.activity[c].item():>5.3f} "
                f"incoming={inc}"
            )
        n_hidden = sum(1 for r in self.roles if r == "hidden")
        lines.append(f"  total hidden cells: {n_hidden}")
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------


def make_curriculum(n: int = 200, seed: int = 0):
    """Returns shared X, and a list of (label, y) per task.

    All three tasks use the SAME inputs — different labelings only.
    This isolates the "task identity" signal entirely in the labels
    and the output cell routing.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 2, generator=g) * 2.0 - 1.0  # [-1, 1]^2
    tasks = [
        ("linear", ((X[:, 0] + X[:, 1]) > 0).float() * 2.0 - 1.0),
        ("XOR", ((X[:, 0] > 0) ^ (X[:, 1] > 0)).float() * 2.0 - 1.0),
        ("rings", ((X[:, 0] ** 2 + X[:, 1] ** 2) > 0.5).float() * 2.0 - 1.0),
    ]
    return X, tasks


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train_substrate_on_task(
    sub: CellSubstrate,
    X: torch.Tensor,
    y: torch.Tensor,
    task_id: int,
    n_epochs: int,
    lr: float,
    label: str,
    threshold_A: float,
    threshold_B: float,
) -> List[float]:
    opt = torch.optim.Adam(sub.parameters(), lr=lr)
    losses: List[float] = []
    for epoch in range(n_epochs):
        opt.zero_grad()
        pred = sub(X, task_id=task_id)
        loss = F.mse_loss(pred, y)
        loss.backward()
        sub.update_signals(loss, task_id=task_id)
        opt.step()
        sub.diffuse_and_accumulate()
        new_hidden = sub.differentiate(threshold_A=threshold_A, threshold_B=threshold_B)
        if new_hidden:
            opt = torch.optim.Adam(sub.parameters(), lr=lr)
            print(
                f"  [{label}] epoch {epoch:>3}: NEW HIDDEN cells {new_hidden} "
                f"@ positions {[tuple(sub.positions[c].int().tolist()) for c in new_hidden]}"
            )
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            losses.append(loss.item())
            n_hidden = sum(1 for r in sub.roles if r == "hidden")
            acc = ((pred.sign() == y.sign()).float().mean().item())
            print(
                f"  [{label}] epoch {epoch:>3}: loss={loss.item():.4f}  "
                f"acc={acc:.3f}  hidden={n_hidden}  "
                f"max(epi_A)={sub.epi_A.max().item():.3f}"
            )
    return losses


def evaluate_all_tasks(
    sub: CellSubstrate, X: torch.Tensor, tasks: List[Tuple[str, torch.Tensor]],
) -> List[float]:
    """Per-task accuracy on the canonical X. All outputs read off ONE forward."""
    with torch.no_grad():
        outs = sub.forward_all_outputs(X)
    accs = []
    for k, (label, y) in enumerate(tasks):
        pred = outs[sub.output_idx[k]]
        accs.append((pred.sign() == y.sign()).float().mean().item())
    return accs


def main() -> int:
    torch.manual_seed(0)
    print("=" * 72)
    print("Cell-first CL PoC — 3 tasks, 3 outputs, retention test")
    print("=" * 72)
    sub = CellSubstrate(grid_size=4, field_sigma=1.2)
    X, tasks = make_curriculum(n=200, seed=0)
    print(f"Setup:")
    print(f"  input cells:  {sub.input_idx} at {[tuple(sub.positions[i].int().tolist()) for i in sub.input_idx]}")
    print(f"  output cells: {sub.output_idx} at {[tuple(sub.positions[o].int().tolist()) for o in sub.output_idx]}")
    print(f"  tasks:        {[name for name, _ in tasks]}")
    print(f"  total cells:  {sub.n_cells()}")
    print()

    # Track per-task accuracy across the curriculum so we can see
    # whether earlier tasks survive later training.
    accs_after_phase: List[Tuple[str, List[float]]] = []

    # Curriculum: train on each task in sequence. Re-evaluate ALL
    # tasks after each phase.
    epochs_per_task = [150, 400, 400]
    for task_id, ((name, y), n_ep) in enumerate(zip(tasks, epochs_per_task)):
        print(f"[Phase {task_id+1}: training task '{name}']")
        train_substrate_on_task(
            sub, X, y, task_id=task_id, n_epochs=n_ep, lr=0.05, label=name,
            threshold_A=0.5, threshold_B=0.2,
        )
        accs = evaluate_all_tasks(sub, X, tasks)
        accs_after_phase.append((name, accs))
        print(f"  >> after phase {task_id+1} ({name}): per-task acc = "
              f"{[f'{a:.3f}' for a in accs]}\n")

    print("=" * 72)
    print("Retention matrix (rows = phase after, cols = task)")
    print("=" * 72)
    print(f"  {'phase':<14} {'linear':>8} {'XOR':>8} {'rings':>8}")
    for name, accs in accs_after_phase:
        print(f"  after {name:<8}  {accs[0]:>8.3f} {accs[1]:>8.3f} {accs[2]:>8.3f}")

    print()
    print("=" * 72)
    print("Final substrate state (all non-potential cells)")
    print("=" * 72)
    print(sub.summarize())
    n_hidden = sum(1 for r in sub.roles if r == "hidden")
    print()

    # Group hidden cells by which output they're nearest to — gives a
    # spatial check on whether tasks are isolated.
    if n_hidden > 0:
        print("Spatial distribution of hidden cells (nearest output):")
        for c in range(sub.n_cells()):
            if sub.roles[c] != "hidden":
                continue
            my_pos = sub.positions[c]
            dists = [(sub.positions[o] - my_pos).norm().item() for o in sub.output_idx]
            nearest = sub.output_idx[int(torch.tensor(dists).argmin())]
            print(f"  cell {c:>2} at pos={tuple(sub.positions[c].int().tolist())} "
                  f"nearest output = cell {nearest} "
                  f"(task '{tasks[sub.output_idx.index(nearest)][0]}')")

    # Verdicts.
    final_accs = accs_after_phase[-1][1]
    min_acc = min(final_accs)
    print()
    if min_acc >= 0.7:
        print(f"VERDICT: no catastrophic forgetting — min per-task acc {min_acc:.3f}")
    elif min_acc >= 0.5:
        print(f"VERDICT: partial forgetting — min per-task acc {min_acc:.3f}")
    else:
        print(f"VERDICT: catastrophic forgetting — min per-task acc {min_acc:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
