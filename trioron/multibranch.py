"""Multi-branch organism — parallel L1 specialty branches off a shared
frozen L0, with soft routing driven by each branch's manifold archive.

Architecture (per memory/absorption_mechanism_design.md):

    L0 (shared frozen random projection, same seed across donors)
      │
      ├── Branch 0  (L1₀ → head₀)   archive₀: {c: (μ_c, σ_c)} for c ∈ C₀
      ├── Branch 1  (L1₁ → head₁)   archive₁: {c: (μ_c, σ_c)} for c ∈ C₁
      └── ...

A donor checkpoint produced by `experiments/poc_dual_organism.py` is
self-contained — state_dict + manifold archive + class layout — and can
be transplanted as a frozen branch into any recipient that was born with
the SAME L0 seed. This module is the receiver side: instantiate `Branch`
from each donor's checkpoint, hand them to a `MultiBranchOrganism`, and
the resulting forward pass produces a soft-gated logit over the union of
all branches' covered classes.

Routing — for input x, project z = L0(x). Each branch scores z against
its own per-class diagonal Gaussian archive; the per-branch
log-likelihood is logsumexp over the branch's classes. Branch gates are
softmax(log_lik / T) — soft routing by default (T=1.0), with bleed
explicitly allowed (Rocky's bet: bleed adds new function rather than
destroying it). Hard routing (argmax over branches) is available for
ablation. Logits are assembled as Σ_b g_b · pad_to_union(logits_b);
non-covered slots from each branch are 0.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import TrioronNetwork


# ---------------------------------------------------------------------
# Branch — one absorbed skill pack
# ---------------------------------------------------------------------


@dataclass
class Branch:
    """One absorbed donor: frozen TrioronNetwork (L0 + L1 + head) + per-
    class manifold archive over L0 code-space + the subset of global
    class IDs this branch was trained on.

    Only L1+head are used at inference (the organism owns the canonical
    L0 and runs that once per input). The branch's own L0 weights are
    retained for L0-match validation when the branch is added to an
    organism.
    """

    label: str
    classes_covered: List[int]
    net: TrioronNetwork
    manifold_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    l0_seed: Optional[int] = None
    arm: Optional[str] = None
    # Cached stacked archive tensors for vectorized log-pdf. Built lazily.
    _archive_classes: Optional[List[int]] = field(default=None, repr=False)
    _archive_mu: Optional[torch.Tensor] = field(default=None, repr=False)
    _archive_sigma: Optional[torch.Tensor] = field(default=None, repr=False)

    @classmethod
    def from_checkpoint(cls, path: str, *, label: Optional[str] = None) -> "Branch":
        """Load a poc_donor_*.pt payload and instantiate a frozen Branch.

        The payload format is what `experiments/poc_dual_organism.py`
        writes: state_dict + n_nodes_per_layer + manifold_stats +
        classes_covered + l0_seed + arm. The reconstructed network is
        moved to eval mode and all parameters are frozen.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        n_nodes = payload["n_nodes_per_layer"]
        # Layer specs: L0 frozen relu, L1 relu, head linear (matches
        # make_classifier in bench_chained_15task.py).
        layer_specs: List[Tuple[int, int, str]] = []
        prev = payload["input_dim"]
        for i, n in enumerate(n_nodes):
            if i == len(n_nodes) - 1:
                act = "linear"
            else:
                act = "relu"
            layer_specs.append((prev, n, act))
            prev = n
        net = TrioronNetwork(layer_specs)
        net.load_state_dict(payload["state_dict"])
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return cls(
            label=label or payload.get("label", "donor"),
            classes_covered=list(payload["classes_covered"]),
            net=net,
            manifold_stats={
                int(c): (mu, sg) for c, (mu, sg) in payload["manifold_stats"].items()
            },
            l0_seed=payload.get("l0_seed"),
            arm=payload.get("arm"),
        )

    # ----- L0 access (for organism-level L0-match validation) -----

    def l0_W(self) -> torch.Tensor:
        return self.net.layers[0].W.detach()

    def l0_b(self) -> torch.Tensor:
        return self.net.layers[0].b.detach()

    # ----- archive likelihood -----

    def _ensure_archive_tensors(self, device: torch.device) -> None:
        if self._archive_mu is None or self._archive_mu.device != device:
            classes = sorted(self.manifold_stats.keys())
            mus = torch.stack([self.manifold_stats[c][0] for c in classes]).to(device)
            sgs = torch.stack([self.manifold_stats[c][1] for c in classes]).to(device)
            self._archive_classes = classes
            self._archive_mu = mus
            self._archive_sigma = sgs

    def archive_log_likelihood(
        self, z: torch.Tensor, eps: float = 1e-6,
    ) -> torch.Tensor:
        """Per-row log p(z | branch) under a mixture-of-equally-weighted
        per-class diagonal Gaussians (logsumexp over classes).

        z shape: (B, l0_width). Returns (B,).
        """
        self._ensure_archive_tensors(z.device)
        mu = self._archive_mu             # (C, d)
        sg = self._archive_sigma.clamp_min(eps)   # (C, d)
        d = z.shape[-1]
        # (B, 1, d) - (1, C, d) → (B, C, d)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        norm = ((diff / sg.unsqueeze(0)) ** 2).sum(-1)         # (B, C)
        logdet = sg.log().sum(-1)                              # (C,)
        log_pdfs = -0.5 * norm - logdet.unsqueeze(0) - 0.5 * d * math.log(2 * math.pi)
        # uniform mixture: log( (1/C) Σ exp(log_pdf_c) ) = logsumexp - log C
        return torch.logsumexp(log_pdfs, dim=-1) - math.log(log_pdfs.shape[-1])

    # ----- forward over donor's own head namespace -----

    def forward_from_l0(self, z: torch.Tensor) -> torch.Tensor:
        """z = shared L0 projection. Returns logits over the donor's full
        head (shape (B, head_size)). Caller picks out trained slots via
        `classes_covered`."""
        return self.net.forward_from_layer(z, start_layer=1)


# ---------------------------------------------------------------------
# Organism — shared L0 + parallel branches
# ---------------------------------------------------------------------


class MultiBranchOrganism(nn.Module):
    """Shared frozen L0 + N parallel Branches with soft archive routing.

    The organism's canonical L0 is taken from the first branch added (or
    set explicitly via `l0_W`/`l0_b`). All subsequent branches must have
    byte-identical L0 weights — the shared-seed invariant is what makes
    instantaneous transplant valid (per absorption_mechanism_design.md).

    Forward modes:
      - "soft" (default): gates = softmax(branch_log_lik / temperature).
        Multiple branches contribute simultaneously, weighted by gate.
      - "hard": gates = one-hot(argmax_b log_lik). Single branch fires.
      - "uniform": gates = 1/N. Pure ablation — disables routing.
    """

    def __init__(
        self,
        l0_W: Optional[torch.Tensor] = None,
        l0_b: Optional[torch.Tensor] = None,
        l0_seed: Optional[int] = None,
        l0_activation: str = "relu",
    ):
        super().__init__()
        self.l0_seed = l0_seed
        self.l0_activation = l0_activation
        if l0_W is not None and l0_b is not None:
            self.register_buffer("l0_W", l0_W.detach().clone())
            self.register_buffer("l0_b", l0_b.detach().clone())
        else:
            self.l0_W = None
            self.l0_b = None
        self._branches: List[Branch] = []
        self._union_classes: List[int] = []
        self._class_to_union: Dict[int, int] = {}

    # ----- assembly -----

    @classmethod
    def from_branches(
        cls,
        branches: Sequence[Branch],
        *,
        l0_seed: Optional[int] = None,
    ) -> "MultiBranchOrganism":
        if not branches:
            raise ValueError("Need at least one branch to build the organism.")
        first = branches[0]
        org = cls(
            l0_W=first.l0_W(),
            l0_b=first.l0_b(),
            l0_seed=l0_seed if l0_seed is not None else first.l0_seed,
            l0_activation=first.net.layers[0].activation,
        )
        for b in branches:
            org.add_branch(b)
        return org

    def add_branch(self, branch: Branch) -> None:
        """Append a branch. Validates the shared-L0 invariant (W and b
        byte-identical) and forbids class overlap with existing branches.
        """
        if self.l0_W is None:
            self.register_buffer("l0_W", branch.l0_W().clone())
            self.register_buffer("l0_b", branch.l0_b().clone())
            self.l0_activation = branch.net.layers[0].activation
        else:
            if not torch.equal(self.l0_W, branch.l0_W()):
                raise ValueError(
                    f"Branch '{branch.label}' L0 W mismatch — "
                    "shared-seed invariant violated; absorption requires "
                    "donor and recipient share the L0 random projection."
                )
            if not torch.equal(self.l0_b, branch.l0_b()):
                raise ValueError(
                    f"Branch '{branch.label}' L0 b mismatch — "
                    "shared-seed invariant violated."
                )
        existing = set(self._class_to_union)
        overlap = sorted(set(branch.classes_covered) & existing)
        if overlap:
            raise ValueError(
                f"Branch '{branch.label}' classes {overlap} already covered "
                "by an earlier branch; class-namespace collision."
            )
        self._branches.append(branch)
        for c in branch.classes_covered:
            self._class_to_union[c] = len(self._union_classes)
            self._union_classes.append(c)

    @property
    def branches(self) -> List[Branch]:
        return list(self._branches)

    @property
    def union_classes(self) -> List[int]:
        return list(self._union_classes)

    # ----- forward -----

    def project_l0(self, x: torch.Tensor) -> torch.Tensor:
        """x → z = activation(L0_W·x + L0_b). Frozen, no_grad. Same path
        as TrioronLayer.forward but bypasses routing_scale (donor's L0
        was frozen at init so routing_scale stayed 1.0 anyway)."""
        if x.dtype != self.l0_W.dtype:
            x = x.to(self.l0_W.dtype)
        z = F.linear(x, self.l0_W, self.l0_b)
        if self.l0_activation == "relu":
            return F.relu(z)
        if self.l0_activation == "linear":
            return z
        raise ValueError(f"Unsupported L0 activation: {self.l0_activation}")

    def gate_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Per-input, per-branch log-likelihoods. Shape (B, N_branches)."""
        cols = [b.archive_log_likelihood(z) for b in self._branches]
        return torch.stack(cols, dim=-1)

    def gates(
        self,
        z: torch.Tensor,
        *,
        mode: str = "soft",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Branch routing weights, shape (B, N_branches), rows sum to 1."""
        if mode == "uniform":
            n = len(self._branches)
            return z.new_full((z.shape[0], n), 1.0 / n)
        log_lik = self.gate_logits(z)
        if mode == "hard":
            idx = log_lik.argmax(dim=-1)
            g = torch.zeros_like(log_lik)
            g.scatter_(1, idx.unsqueeze(1), 1.0)
            return g
        if mode == "soft":
            return F.softmax(log_lik / max(temperature, 1e-6), dim=-1)
        raise ValueError(f"Unknown routing mode: {mode}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        routing: str = "soft",
        temperature: float = 1.0,
        normalize_per_branch: bool = False,
        return_extras: bool = False,
    ):
        """Run x through the organism. Returns logits over `union_classes`
        in the order they appear in that list.

        normalize_per_branch=False (default) — combine raw head logits
            weighted by gates: combined[c] = Σ_b g_b · pad(logits_b)[c].
            Preserves task-aware accuracy, but full-union argmax is
            biased by per-donor calibration mismatch.

        normalize_per_branch=True — each branch's logits are passed
            through log_softmax restricted to its OWN covered classes
            before combining; gate weight enters in log space:
                combined[c] = log(g_b(c)) + log P_b(c | x)
            where b(c) is the branch covering c (non-overlap is enforced
            in add_branch). This is the principled mixture-of-experts
            log-probability and the right argmax target for full-union
            without retraining. Inference-only fix.

        With `return_extras=True`, also returns a dict with `z` (the
        shared L0 projection), `gates` (B, N_branches), and
        `branch_logits_padded` (B, N_branches, n_union_classes).
        """
        z = self.project_l0(x)
        gates = self.gates(z, mode=routing, temperature=temperature)
        n_union = len(self._union_classes)
        B = z.shape[0]
        if normalize_per_branch:
            # Combined log-probability over the union. Slots not covered
            # by any branch (impossible by construction — union_classes
            # is the union of branch coverage) would stay at -inf.
            combined = z.new_full((B, n_union), float("-inf"))
        else:
            combined = z.new_zeros(B, n_union)
        branch_padded = z.new_zeros(B, len(self._branches), n_union) \
            if return_extras else None
        log_g = None
        if normalize_per_branch:
            log_g = torch.log(gates.clamp_min(1e-30))      # (B, N_branches)
        for bi, b in enumerate(self._branches):
            head_logits = b.forward_from_l0(z)              # (B, head_size_b)
            cov = b.classes_covered
            cols = head_logits[:, cov]                      # (B, |C_b|)
            if normalize_per_branch:
                cols = F.log_softmax(cols, dim=-1)          # log P_b over C_b
                cols = cols + log_g[:, bi:bi + 1]           # + log(g_b)
            else:
                cols = gates[:, bi:bi + 1] * cols           # g_b · logits_b
            for j, c in enumerate(cov):
                ui = self._class_to_union[c]
                # Non-overlap is enforced in add_branch, so each union
                # slot is covered by exactly one branch — direct write.
                combined[:, ui] = cols[:, j]
                if branch_padded is not None:
                    branch_padded[:, bi, ui] = cols[:, j]
        if return_extras:
            return combined, {
                "z": z,
                "gates": gates,
                "branch_logits_padded": branch_padded,
            }
        return combined

    # ----- diagnostics -----

    def storage_bytes(self) -> Dict[str, int]:
        """Rough byte breakdown across the shared L0, branch substrates,
        and manifold archives. Useful for the storage-vs-accuracy story
        in the paper."""
        l0 = self.l0_W.numel() * self.l0_W.element_size() + \
             self.l0_b.numel() * self.l0_b.element_size()
        branch_substrate = 0
        archive = 0
        for b in self._branches:
            for layer in b.net.layers[1:]:
                branch_substrate += layer.W.numel() * layer.W.element_size()
                branch_substrate += layer.b.numel() * layer.b.element_size()
            for (mu, sg) in b.manifold_stats.values():
                archive += (mu.numel() + sg.numel()) * mu.element_size()
        return {
            "l0_bytes": l0,
            "branch_substrate_bytes": branch_substrate,
            "archive_bytes": archive,
            "total_bytes": l0 + branch_substrate + archive,
        }


__all__ = ["Branch", "MultiBranchOrganism"]
