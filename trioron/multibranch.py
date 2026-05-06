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
# Random-projection adapter (Phase C handoff item 2, 2026-05-06)
# ---------------------------------------------------------------------


def _pick_canonical_seed(branches: Sequence["Branch"]) -> Optional[int]:
    """Most-common seed wins; tie-break by first appearance order.
    Returns None only if every branch's l0_seed is None."""
    counts: Dict[Optional[int], int] = {}
    first_seen: Dict[Optional[int], int] = {}
    for i, b in enumerate(branches):
        s = b.l0_seed
        counts[s] = counts.get(s, 0) + 1
        if s not in first_seen:
            first_seen[s] = i
    # max by (count, -first_seen) so ties prefer earlier branches
    return max(counts.keys(), key=lambda s: (counts[s], -first_seen[s]))


def _build_random_projection(
    *,
    canon_seed: int,
    donor_seed: int,
    canon_dim: int,
    donor_dim: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic Gaussian random projection canon→donor.

    Std = sqrt(1/canon_dim) so E[||z @ A||^2] ≈ ||z||^2 (Johnson-
    Lindenstrauss scaling). Seeded by mixing the two L0 seeds so the
    same (canonical, donor) pair always produces the same matrix.

    NOTE: this is the FALLBACK path. The donor's L1 was trained on its
    own L0's outputs; a Gaussian random projection of canonical-z
    won't reproduce that distribution, so accuracy degrades. See
    MANUAL §3 for the recommended shared-L0 invariant.
    """
    mixed = (int(canon_seed) * 1_000_003) ^ int(donor_seed)
    g = torch.Generator(device="cpu").manual_seed(mixed & 0x7FFF_FFFF)
    std = (1.0 / canon_dim) ** 0.5
    A = torch.randn(canon_dim, donor_dim, generator=g) * std
    return A.to(dtype)


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
    # Random-projection adapter for non-canonical-L0 branches (Phase C
    # handoff item 2, 2026-05-06). Shape (canonical_dim, donor_dim).
    # When set, the organism projects shared canonical z through this
    # before passing to the branch's L1 and to its archive scoring.
    # None for branches whose L0 matches the canonical (shared-seed
    # path, no projection needed). UNTESTED accuracy hit; see
    # MANUAL §3.
    projection: Optional[torch.Tensor] = None
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

    def _project_to_donor_space(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the random-projection adapter if this branch's L0 didn't
        match the canonical L0 at absorb time. No-op for canonical
        branches."""
        if self.projection is None:
            return z
        proj = self.projection.to(device=z.device, dtype=z.dtype)
        return z @ proj

    @property
    def archive_classes(self) -> List[int]:
        """Sorted list of global class IDs covered by this branch's
        manifold archive. The order matches the column order of
        :meth:`per_class_log_likelihood`. Useful for callers that want
        to index per-class scores back to class IDs (e.g. routing,
        novelty detection, confidence inspection)."""
        if self._archive_classes is None:
            # Materialize without requiring a forward pass.
            self._archive_classes = sorted(self.manifold_stats.keys())
        return list(self._archive_classes)

    def per_class_log_likelihood(
        self, z: torch.Tensor, eps: float = 1e-6,
    ) -> torch.Tensor:
        """Per-row, per-class log-pdf of z under the branch's manifold
        archive. Each archived class c contributes an independent
        diagonal-Gaussian log-pdf computed from its (μ_c, σ_c).

        z shape: (B, canonical_dim). For non-canonical branches the
        random-projection adapter rotates z into donor space first.
        Returns: (B, C) where C = len(archive_classes).

        The logsumexp aggregate (mixture-of-equally-weighted form) is
        provided by :meth:`archive_log_likelihood`; use this method
        directly when you need the per-class breakdown — for example
        a routing argmax, a novelty gate based on top-vs-runner-up
        gap, or an inspect panel showing per-class confidence.
        """
        z = self._project_to_donor_space(z)
        self._ensure_archive_tensors(z.device)
        mu = self._archive_mu             # (C, d)
        sg = self._archive_sigma.clamp_min(eps)   # (C, d)
        d = z.shape[-1]
        diff = z.unsqueeze(1) - mu.unsqueeze(0)                 # (B, C, d)
        norm = ((diff / sg.unsqueeze(0)) ** 2).sum(-1)          # (B, C)
        logdet = sg.log().sum(-1)                               # (C,)
        return -0.5 * norm - logdet.unsqueeze(0) - 0.5 * d * math.log(2 * math.pi)

    def archive_log_likelihood(
        self, z: torch.Tensor, eps: float = 1e-6,
    ) -> torch.Tensor:
        """Per-row log p(z | branch) under a mixture-of-equally-weighted
        per-class diagonal Gaussians (logsumexp aggregate). Shape:
        (B, canonical_dim) → (B,).

        Thin wrapper over :meth:`per_class_log_likelihood` for callers
        that only want a single scalar confidence per row.
        """
        log_pdfs = self.per_class_log_likelihood(z, eps=eps)
        # uniform mixture: log( (1/C) Σ exp(log_pdf_c) ) = logsumexp - log C
        return torch.logsumexp(log_pdfs, dim=-1) - math.log(log_pdfs.shape[-1])

    # ----- forward over donor's own head namespace -----

    def forward_from_l0(self, z: torch.Tensor) -> torch.Tensor:
        """z = shared canonical L0 projection. For non-canonical
        branches it gets random-projected into donor space first.
        Returns logits over the donor's full head (shape
        (B, head_size)). Caller picks out trained slots via
        `classes_covered`."""
        z = self._project_to_donor_space(z)
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
        # Pick the canonical L0 by majority seed; tie-break by first
        # appearance. Reorder so canonical-seed branches are added
        # first (so the organism installs the right L0 before any
        # mismatched branch builds its random-projection adapter).
        canon_seed = _pick_canonical_seed(branches) if l0_seed is None else l0_seed
        canon_first, others = [], []
        for b in branches:
            (canon_first if b.l0_seed == canon_seed else others).append(b)
        ordered = canon_first + others
        first = ordered[0]
        org = cls(
            l0_W=first.l0_W(),
            l0_b=first.l0_b(),
            l0_seed=canon_seed,
            l0_activation=first.net.layers[0].activation,
        )
        for b in ordered:
            org.add_branch(b)
        return org

    def add_branch(self, branch: Branch) -> None:
        """Append a branch. If the branch's L0 matches the canonical
        L0 (shared-seed invariant), it is plugged in directly. If it
        does NOT match, a deterministic random-projection adapter is
        materialized (untested fallback path — see MANUAL §3).
        """
        if self.l0_W is None:
            self.register_buffer("l0_W", branch.l0_W().clone())
            self.register_buffer("l0_b", branch.l0_b().clone())
            self.l0_activation = branch.net.layers[0].activation
        else:
            shapes_match = (
                self.l0_W.shape == branch.l0_W().shape
                and self.l0_b.shape == branch.l0_b().shape
            )
            weights_match = shapes_match and torch.equal(
                self.l0_W, branch.l0_W()
            ) and torch.equal(self.l0_b, branch.l0_b())
            if not weights_match:
                canon_dim = self.l0_W.shape[0]
                donor_dim = branch.l0_W().shape[0]
                branch.projection = _build_random_projection(
                    canon_seed=int(self.l0_seed) if self.l0_seed is not None else 0,
                    donor_seed=int(branch.l0_seed) if branch.l0_seed is not None else 0,
                    canon_dim=canon_dim,
                    donor_dim=donor_dim,
                    dtype=self.l0_W.dtype,
                )
                print(
                    f"[trioron absorb] WARNING: branch '{branch.label}' L0 "
                    f"(seed={branch.l0_seed}, dim={donor_dim}) does not match "
                    f"canonical L0 (seed={self.l0_seed}, dim={canon_dim}). "
                    f"Built random-projection adapter A[{canon_dim}→{donor_dim}] "
                    f"from seeds ({self.l0_seed}, {branch.l0_seed}). "
                    "This path is UNTESTED; per-branch accuracy may degrade "
                    "10-30% (see MANUAL §3). Use at your own risk."
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

    def forward_from_z(
        self,
        z: torch.Tensor,
        *,
        routing: str = "soft",
        temperature: float = 1.0,
        normalize_per_branch: bool = False,
        bias_offset: Optional[torch.Tensor] = None,
        return_extras: bool = False,
    ):
        """Combine path starting from a code-space tensor z (skips L0).
        Used by the dream-cycle calibration trainer, which samples z
        directly from the union manifold archive — no need to round-trip
        through pixel space. `bias_offset` (shape (n_union,)) is added
        to combined logits if supplied; that's the calibrator parameter.
        """
        gates = self.gates(z, mode=routing, temperature=temperature)
        n_union = len(self._union_classes)
        B = z.shape[0]
        if normalize_per_branch:
            combined = z.new_full((B, n_union), float("-inf"))
            log_g = torch.log(gates.clamp_min(1e-30))
        else:
            combined = z.new_zeros(B, n_union)
            log_g = None
        branch_padded = z.new_zeros(B, len(self._branches), n_union) \
            if return_extras else None
        for bi, b in enumerate(self._branches):
            head_logits = b.forward_from_l0(z)
            cov = b.classes_covered
            cols = head_logits[:, cov]
            if normalize_per_branch:
                cols = F.log_softmax(cols, dim=-1)
                cols = cols + log_g[:, bi:bi + 1]
            else:
                cols = gates[:, bi:bi + 1] * cols
            for j, c in enumerate(cov):
                ui = self._class_to_union[c]
                combined[:, ui] = cols[:, j]
                if branch_padded is not None:
                    branch_padded[:, bi, ui] = cols[:, j]
        if bias_offset is not None:
            combined = combined + bias_offset
        if return_extras:
            return combined, {
                "z": z,
                "gates": gates,
                "branch_logits_padded": branch_padded,
            }
        return combined

    def forward(
        self,
        x: torch.Tensor,
        *,
        routing: str = "soft",
        temperature: float = 1.0,
        normalize_per_branch: bool = False,
        bias_offset: Optional[torch.Tensor] = None,
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
        return self.forward_from_z(
            z, routing=routing, temperature=temperature,
            normalize_per_branch=normalize_per_branch,
            bias_offset=bias_offset,
            return_extras=return_extras,
        )

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
