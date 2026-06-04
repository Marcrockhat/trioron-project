"""Epigenetic lock λ — the namesake third node variable, restored to v2 (canonical form).

`tri·oron` = "three coupled state variables per node" (blueprint §3): weight `w`,
**epigenetic lock `λ`**, utility `u`. v2.0 dropped λ (spec §8.6) for cell-level credit-
locking — and that loss is the root of the catastrophic forgetting the world experiments
keep fighting (the lam=0 unanchored drift). This module restores it.

WHAT λ ACTUALLY IS (paper §A.1, legacy/node.py:919 `set_lambda`) — easy to get wrong:

  * λ is **PER-NODE**, not per-weight: `node_lambda ∈ ℝⁿ`, one gate per cell.
  * Its default driver is **Fisher**, but `λ_i = Σ_j F_{i,j}` is the **ROW-SUM** of the
    cell's incoming Fisher, NOT the row-mean — the mean divides Fisher below any usable
    floor at high fan-in, giving a silent-zero EWC penalty (the exact washout bug fixed
    historically; see commits 81d3785 / bba9420). A **LAMBDA_FLOOR** keeps λ ≥ ε.
  * **Fisher is only ONE driver.** λ is a general plasticity GATE standing in for
    environmentally-regulated biological gating (BDNF / methylation). `set_lambda` can
    drive it from reward magnitudes, an environment sense (temperature/stress), attention
    masks, or hand-injected priors (freeze = big λ, wake = 0). That generality is λ's
    intrinsic value — for the embodied organism, drive λ from survival-reward.

EWC pull (composes with any optimizer — add to the loss before backward):
    L_total = L_task + strength · 0.5 · Σ_i λ_i · ( Σ_j (w_ij − ŵ_ij)²  +  (b_i − b̂_i)² )

Storage lives in the arena (`edge_fisher`, `edge_anchor`, `node_lambda`, `bias_fisher`,
`bias_anchor`), so λ rides through ship/wake with the weights.
"""
from __future__ import annotations

import torch

from trioron.core.arena import Arena

DEFAULT_DECAY = 0.95     # Fisher EMA decay (blueprint α ∈ [0.95, 0.99])
LAMBDA_FLOOR = 1e-3      # ε-floor so λ never silently zeros (commit 81d3785)


def fisher_loss(logits: torch.Tensor) -> torch.Tensor:
    """Canonical **model-distribution** Fisher target: sample y ~ softmax(logits) and
    return CE(logits, y). Backprop of this is the TRUE Fisher diagonal in expectation.

    Why not the true label? The empirical (true-label) Fisher VANISHES once the model is
    correct and confident — at convergence CE(true) → 0, grad² → 0, every node's row-sum
    falls below LAMBDA_FLOOR, and λ flattens to a uniform floor (no differentiation; the
    exact washout that left λ dead — session 014/015). Sampling the target from the
    model's own softmax keeps the borderline samples producing real gradients, so λ
    differentiates stiff cells from plastic ones even at high confidence. Standard EWC
    'true Fisher' protocol. Call under autograd; backward(), then accumulate_fisher()."""
    with torch.no_grad():
        y = torch.multinomial(torch.softmax(logits, dim=1), 1).squeeze(1)
    return torch.nn.functional.cross_entropy(logits, y)


@torch.no_grad()
def accumulate_fisher(arena: Arena, decay: float = DEFAULT_DECAY) -> None:
    """Per-PARAM Fisher EMA: F ← α·F + (1−α)·grad². Call after backward(), before
    step() and before zeroing grads. Roll up to node_lambda via refresh_lambda()."""
    cur = arena.edge_cursor
    if arena.edge_weight.grad is not None and cur > 0:
        arena.edge_fisher[:cur].mul_(decay).add_(arena.edge_weight.grad[:cur] ** 2,
                                                 alpha=(1.0 - decay))
    if arena.bias.grad is not None:
        arena.bias_fisher.mul_(decay).add_(arena.bias.grad ** 2, alpha=(1.0 - decay))


@torch.no_grad()
def accumulate_saliency(arena: Arena, decay: float = DEFAULT_DECAY) -> None:
    """Per-PARAM saliency EMA: S ← α·S + (1−α)·|w·g| — trioron's NATIVE importance
    signal (KIBRA edge-tagging in dream.py, the utility `u`, and the pruner all use the
    |·grad| family), as opposed to Fisher's grad². Two reasons it's the better λ driver
    on the substrate: (1) it's trioron's own dialect, not imported academic EWC; (2) the
    |w| factor keeps large settled weights important even as g→0 at convergence, so λ
    does NOT collapse to the floor the way empirical Fisher does (a first-order Taylor/
    OBD saliency). Shares the edge_fisher/bias_fisher importance buffers; roll up with
    refresh_lambda(). Call after backward(), before step()."""
    cur = arena.edge_cursor
    if arena.edge_weight.grad is not None and cur > 0:
        sal = arena.edge_weight[:cur].abs() * arena.edge_weight.grad[:cur].abs()
        arena.edge_fisher[:cur].mul_(decay).add_(sal, alpha=(1.0 - decay))
    if arena.bias.grad is not None:
        sal_b = arena.bias.abs() * arena.bias.grad.abs()
        arena.bias_fisher.mul_(decay).add_(sal_b, alpha=(1.0 - decay))


@torch.no_grad()
def refresh_lambda(arena: Arena, floor: float = LAMBDA_FLOOR) -> None:
    """node_lambda_i = ROW-SUM of cell i's incoming-edge Fisher + its bias Fisher,
    clamped to ≥ floor. The canonical Fisher→λ map (paper §A.1: row-sum, NOT mean)."""
    cur = arena.edge_cursor
    lam = arena.bias_fisher.clone()
    if cur > 0:
        dst = arena.edge_dst[:cur].long()
        lam.scatter_add_(0, dst, arena.edge_fisher[:cur])   # Σ_j F over incoming edges
    arena.node_lambda.copy_(lam.clamp_(min=floor))


@torch.no_grad()
def set_lambda(arena: Arena, signal: torch.Tensor, mode: str = "absolute",
               floor: float = LAMBDA_FLOOR) -> None:
    """Drive λ from an arbitrary per-cell signal (reward / environment / attention /
    prior), bypassing or layering on Fisher. signal: [capacity]. mode:
      absolute       λ ← signal          (replace)
      additive       λ ← λ + signal      (layer on Fisher)
      multiplicative λ ← λ * signal      (gate, e.g. a sleep cycle in [0,1])
    Clamped to ≥ floor (negative λ would invert the EWC pull)."""
    if mode not in ("absolute", "additive", "multiplicative"):
        raise ValueError(f"mode {mode!r} not in absolute/additive/multiplicative")
    s = signal.detach().to(arena.node_lambda)
    if mode == "absolute":
        arena.node_lambda.copy_(s)
    elif mode == "additive":
        arena.node_lambda.add_(s)
    else:
        arena.node_lambda.mul_(s)
    arena.node_lambda.clamp_(min=floor)


@torch.no_grad()
def anchor(arena: Arena) -> None:
    """Snapshot current weights/biases as the anchors the EWC pull defends. Call at a
    task boundary / stable plateau (then refresh_lambda to lock in the importance)."""
    cur = arena.edge_cursor
    if cur > 0:
        arena.edge_anchor[:cur].copy_(arena.edge_weight[:cur].detach())
    arena.bias_anchor.copy_(arena.bias.detach())


def ewc_penalty(arena: Arena) -> torch.Tensor:
    """0.5 · Σ_i λ_i · ( Σ_j (w_ij − ŵ_ij)² + (b_i − b̂_i)² ) — per-node λ gating each
    cell's incoming edges and bias. Scalar autograd tensor; add strength·this to loss."""
    cur = arena.edge_cursor
    lam = arena.node_lambda
    pen = torch.zeros((), device=arena.edge_weight.device)
    if cur > 0:
        dst = arena.edge_dst[:cur].long()
        dw2 = (arena.edge_weight[:cur] - arena.edge_anchor[:cur]) ** 2
        pen = pen + (lam[dst] * dw2).sum()
    db2 = (arena.bias - arena.bias_anchor) ** 2
    pen = pen + (lam * db2).sum()
    return 0.5 * pen


@torch.no_grad()
def modulated_scale(arena: Arena) -> None:
    """Optional blueprint §3.2 refinement: scale grads by 1/(1+λ) so stiff (high-λ)
    cells learn slower. Call after backward(), before step()."""
    cur = arena.edge_cursor
    lam = arena.node_lambda
    if arena.edge_weight.grad is not None and cur > 0:
        arena.edge_weight.grad[:cur].div_(1.0 + lam[arena.edge_dst[:cur].long()])
    if arena.bias.grad is not None:
        arena.bias.grad.div_(1.0 + lam)
