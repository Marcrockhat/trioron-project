"""Cell-first substrate PoC: layer-emergence from field-driven differentiation.

The substrate is a 4x4 grid of cells. Two cells are designated "input"
(read data), one "output" (read for loss). The remaining 13 start as
"potential" — they have positions but no incoming connections, no W,
no role.

Two field channels broadcast across the grid via a Gaussian kernel
on locus distance:
    A (frustration) = sRNA-analog, per-cell internal_stress
    B (activity)    = per-cell forward output magnitude

Per-cell epi_A and epi_B integrate ambient (neighbor-diffused) exposure
monotonically. A potential cell transitions to "hidden" role when
epi_A > threshold_A AND epi_B < threshold_B (the "stressed-but-quiet
neighborhood needs recruitment" signature). The transition forms:
    - 2 incoming connections from cells with high activity (the input
      stream)
    - 1 outgoing connection to the cell with the highest frustration
      (the output stream)

No top-level insert_layer() call. Layer-shaped structure should emerge
from cells deciding their own role from local field readings.

Curriculum:
    Task 1: linearly separable binary classification — direct input→
            output connection should suffice; potential cells stay
            potential.
    Task 2: XOR — direct linear connection cannot solve. Frustration
            spikes. Cells near the output should latch into hidden,
            forming an emergent hidden layer that lets the substrate
            solve XOR.

Success criterion:
    1. ≥1 potential cell transitions to hidden during Task 2.
    2. Task 2 loss decreases below the linear-only ceiling.
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
        output_positions: Tuple[Tuple[int, int], ...] = ((3, 1),),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, len(input_idx)). Returns (batch,) output value."""
        batch = x.shape[0]
        device = x.device
        outputs: Dict[int, torch.Tensor] = {}
        # Inputs get clamped values.
        for k, i in enumerate(self.input_idx):
            outputs[i] = x[:, k]
        # Cells with no W and no incoming output zero.
        for c in range(self.n_cells()):
            if c in outputs:
                continue
            W, _ = self._params(c)
            if W is None and not self.incoming[c]:
                outputs[c] = torch.zeros(batch, device=device)
        # Topo loop: any pending cell whose all-incoming are computed
        # becomes ready. Bound iterations to avoid infinite loop on
        # cycles (this substrate is DAG by construction).
        pending = set(range(self.n_cells())) - set(outputs.keys())
        max_iters = self.n_cells() + 2
        for _ in range(max_iters):
            if not pending:
                break
            ready = [c for c in pending
                     if all(s in outputs for s in self.incoming[c])]
            if not ready:
                # Disconnected stragglers — output zero, drop them.
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
        self._cell_outputs = outputs
        return outputs[self.output_idx[0]]

    # ----- signals & field -----

    def update_signals(self, loss: torch.Tensor) -> None:
        """Update per-cell internal_stress and activity.

        internal_stress: at the output cell, set to current loss value.
                         W.grad magnitude isn't useful — once Adam
                         converges to a local min, gradients are small
                         even when the model is failing (e.g., XOR
                         plateau). Loss itself is the honest signal:
                         "how much is this cell still wrong."
        activity:        per-cell mean |output|, but ONLY for cells
                         that actually compute (W-having). Input cells'
                         clamped data values are not "activity" — they're
                         transduction. Excluding them stops the activity
                         field from being bathed by input magnitudes,
                         which previously kept epi_B above threshold
                         everywhere.
        """
        with torch.no_grad():
            # Stress = loss, broadcast from the output cell. Slow decay
            # toward current loss so chronic-high-loss stays high and
            # converged-low-loss drops.
            current_loss = float(loss.detach().item())
            for c in range(self.n_cells()):
                if c in self.output_idx:
                    self.internal_stress[c] = (
                        0.8 * self.internal_stress[c].item() + 0.2 * current_loss
                    )
                else:
                    # Non-output cells don't generate stress directly;
                    # they receive it via the field.
                    self.internal_stress[c] *= 0.95
            if self._cell_outputs is not None:
                for c, y in self._cell_outputs.items():
                    if c in self.input_idx:
                        # Input cells are data clamps, not computing
                        # — they don't broadcast activity.
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


def make_linear_task(n: int = 200, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 2, generator=g) * 2.0 - 1.0  # [-1, 1]^2
    y = ((X[:, 0] + X[:, 1]) > 0).float() * 2.0 - 1.0
    return X, y


def make_xor_task(n: int = 200, seed: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 2, generator=g) * 2.0 - 1.0  # [-1, 1]^2
    a = X[:, 0] > 0
    b = X[:, 1] > 0
    y = (a ^ b).float() * 2.0 - 1.0
    return X, y


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train_substrate(
    sub: CellSubstrate,
    X: torch.Tensor,
    y: torch.Tensor,
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
        pred = sub(X)
        loss = F.mse_loss(pred, y)
        loss.backward()
        sub.update_signals(loss)
        opt.step()
        sub.diffuse_and_accumulate()
        new_hidden = sub.differentiate(threshold_A=threshold_A, threshold_B=threshold_B)
        if new_hidden:
            opt = torch.optim.Adam(sub.parameters(), lr=lr)
            print(
                f"  [{label}] epoch {epoch:>3}: NEW HIDDEN cells {new_hidden} "
                f"(epi_A {[round(sub.epi_A[c].item(), 2) for c in new_hidden]}, "
                f"epi_B {[round(sub.epi_B[c].item(), 2) for c in new_hidden]})"
            )
        if epoch % 25 == 0 or epoch == n_epochs - 1:
            losses.append(loss.item())
            n_hidden = sum(1 for r in sub.roles if r == "hidden")
            print(
                f"  [{label}] epoch {epoch:>3}: loss={loss.item():.4f}  "
                f"acc={((pred.sign() == y.sign()).float().mean().item()):.3f}  "
                f"hidden={n_hidden}  "
                f"max(epi_A)={sub.epi_A.max().item():.3f}  "
                f"max(stress)={sub.internal_stress.max().item():.3f}"
            )
    return losses


def main() -> int:
    torch.manual_seed(0)
    print("=" * 64)
    print("Cell-first substrate PoC")
    print("=" * 64)
    sub = CellSubstrate(grid_size=4, field_sigma=1.2)
    print(f"Initial setup:")
    print(f"  input cells:  {sub.input_idx}")
    print(f"  output cells: {sub.output_idx}")
    print(f"  potential:    {[c for c in range(sub.n_cells()) if sub.roles[c]=='potential']}")
    print(f"  total cells:  {sub.n_cells()}")
    print()

    # Task 1: linear binary classification. Should stay one-layer.
    X1, y1 = make_linear_task(n=200, seed=0)
    print("[Task 1: linear binary classification]")
    train_substrate(
        sub, X1, y1, n_epochs=150, lr=0.05, label="task1",
        threshold_A=0.5, threshold_B=0.2,
    )

    print("\nState after Task 1:")
    print(sub.summarize())

    # Task 2: XOR. Single linear layer cannot solve. We expect
    # frustration to spike and hidden cells to emerge.
    X2, y2 = make_xor_task(n=200, seed=1)
    print("\n[Task 2: XOR]")
    train_substrate(
        sub, X2, y2, n_epochs=400, lr=0.05, label="task2",
        threshold_A=0.5, threshold_B=0.2,
    )

    print("\n=== Final substrate (all cells) ===")
    print(sub.summarize(include_potential=True))
    n_hidden = sum(1 for r in sub.roles if r == "hidden")
    # Final acc on task 2.
    with torch.no_grad():
        pred = sub(X2)
        acc = ((pred.sign() == y2.sign()).float().mean().item())
    print(f"\nFinal Task 2 (XOR) accuracy: {acc:.3f}  (chance=0.5)")
    print(f"Hidden cells emerged: {n_hidden}")
    if n_hidden >= 1 and acc > 0.7:
        print("\nVERDICT: emergence + capability lift confirmed")
    elif n_hidden >= 1:
        print("\nVERDICT: emergence happened but XOR not solved — recruitment "
              "wiring or learning not yet sufficient")
    else:
        print("\nVERDICT: NO emergence — field/epi/threshold tuning needed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
