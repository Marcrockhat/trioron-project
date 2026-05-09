"""SensoryOrganism — multi-modal sibling of MultiBranchOrganism.

Where MultiBranchOrganism enforces a shared canonical L0 across all
branches (so their L0-space z's are directly comparable and the
shared-seed instantaneous-transplant story holds), SensoryOrganism
holds a list of sense-specialized branches each with their OWN
preprocessing pipeline:

    image ──► apply_sense ──► standardize ──► L0  ──► z_b
                                                     │
                                                     ├── archive_b log-lik   (routing score)
                                                     └── L1_b → head_b → logits_b

Each sense reads the same image but extracts a different feature
vector, so the per-branch z lives in a *different* random-projection
space — the canonical-shared-L0 invariant doesn't apply, and the
no-overlap-classes invariant doesn't apply either (every sense donor
typically covers the full class set). This module is the natural home
for those cases.

Routing: per-input, per-branch archive log-likelihood under the
branch's own per-class diagonal-Gaussian archive (mixture-of-
equally-weighted form, identical math to ``Branch.archive_log_likelihood``
just evaluated on the branch's own z). Gates are softmax across
branches; combined logits are gate-weighted sums of per-branch head
logits, padded to the union-class layout.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import apply_sense
from .standardizer import Standardizer
from ..multibranch import Branch
from ..network import TrioronNetwork


# ---------------------------------------------------------------------
# SenseBranch — Branch + sense pipeline
# ---------------------------------------------------------------------


@dataclass
class SenseBranch(Branch):
    """A Branch that owns its full input pipeline.

    Adds ``sense_name`` (registry key into ``trioron.senses.SENSES``)
    and ``standardizer`` (mean/std over the sense's training-set
    readings). ``compute_z(images)`` runs the full image-to-z path
    using the branch's OWN L0 weights.
    """

    sense_name: str = ""
    standardizer: Optional[Standardizer] = None

    @classmethod
    def from_sense_donor(
        cls, path: Union[str, Path], *, label: Optional[str] = None,
    ) -> "SenseBranch":
        """Load a sense_donor_*.pt checkpoint produced by
        ``experiments/cifar/train_donor.py``. The checkpoint embeds
        the sense name and standardizer state alongside the standard
        donor payload (state_dict + manifold_stats + classes_covered
        + l0_seed)."""
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        sense_name = payload.get("sense")
        if sense_name is None:
            raise ValueError(
                f"{path}: missing 'sense' field — not a sense-donor checkpoint"
            )
        std_dict = payload.get("standardizer")
        if std_dict is None:
            raise ValueError(f"{path}: missing 'standardizer' field")
        std = Standardizer.from_dict(std_dict)

        n_nodes = list(payload["n_nodes_per_layer"])
        input_dim = int(payload["input_dim"])
        layer_specs: List[Tuple[int, int, str]] = []
        prev = input_dim
        for i, n in enumerate(n_nodes):
            act = "linear" if i == len(n_nodes) - 1 else "relu"
            layer_specs.append((prev, int(n), act))
            prev = int(n)
        net = TrioronNetwork(layer_specs)
        net.load_state_dict(payload["state_dict"])
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

        manifold_stats = {
            int(c): (mu, sg) for c, (mu, sg) in payload["manifold_stats"].items()
        }
        return cls(
            label=label or payload.get("label", f"sense_{sense_name}"),
            classes_covered=list(payload["classes_covered"]),
            net=net,
            manifold_stats=manifold_stats,
            l0_seed=payload.get("l0_seed"),
            arm=payload.get("arm"),
            sense_name=sense_name,
            standardizer=std,
        )

    # ----- per-branch input pipeline -----

    def compute_z(self, images: torch.Tensor) -> torch.Tensor:
        """images (N, 3, H, W) in [0, 1] → z (N, l0_dim) under THIS
        branch's sense + standardizer + L0 path. Pure inference, no
        gradients required."""
        device = images.device
        x_raw = apply_sense(self.sense_name, images)
        mean = self.standardizer.mean.to(device=device, dtype=x_raw.dtype)
        std = self.standardizer.std.to(device=device, dtype=x_raw.dtype)
        x = (x_raw - mean) / std
        # The branch's own L0 = layers[0]. Run only L0 (live-W path),
        # not the whole net. forward_from_layer(start_layer=1) consumes
        # post-L0 activations, so we still need an explicit L0 forward.
        l0 = self.net.layers[0]
        return l0(x)


# ---------------------------------------------------------------------
# SensoryOrganism
# ---------------------------------------------------------------------


class SensoryOrganism(nn.Module):
    """Multi-modal organism: a list of SenseBranches, each running its
    own sense pipeline, fused by archive-routed soft gating.

    Surface mirrors ``MultiBranchOrganism`` where applicable:
      - ``forward(images, routing, temperature, normalize_per_branch,
        bias_offset, return_extras)``
      - ``gates(images, mode, temperature)``
      - ``gate_logits(images)``
      - ``union_classes``, ``branches``

    Departures from MultiBranchOrganism:
      - No canonical L0 — each branch carries its own.
      - Class overlap across branches IS allowed; the typical case
        for sense donors is every branch covering the full label set.
        Combined logits sum gate-weighted contributions; if multiple
        branches cover the same class, their logits add (with
        ``normalize_per_branch=False``) or their log-probabilities mix
        (with ``normalize_per_branch=True``).
    """

    def __init__(self) -> None:
        super().__init__()
        self._branches: List[SenseBranch] = []
        self._union_classes: List[int] = []
        self._class_to_union: Dict[int, int] = {}
        # nn.ModuleList holds the underlying TrioronNetworks so .to()
        # propagates device moves through the branches' parameters.
        self.branch_nets = nn.ModuleList()

    # ----- assembly -----

    @classmethod
    def from_sense_donors(
        cls,
        paths: Sequence[Union[str, Path]],
        *,
        labels: Optional[Sequence[str]] = None,
    ) -> "SensoryOrganism":
        org = cls()
        labels_seq = list(labels) if labels is not None else [None] * len(paths)
        if len(labels_seq) != len(paths):
            raise ValueError("labels length must match paths length")
        for path, lab in zip(paths, labels_seq):
            org.add_branch(SenseBranch.from_sense_donor(path, label=lab))
        return org

    def add_branch(self, branch: SenseBranch) -> None:
        """Append a SenseBranch. Class-namespace overlap is permitted
        (multi-modal pattern); the union-class list grows monotonically
        and existing class indices are preserved."""
        self._branches.append(branch)
        self.branch_nets.append(branch.net)
        for c in branch.classes_covered:
            if c not in self._class_to_union:
                self._class_to_union[c] = len(self._union_classes)
                self._union_classes.append(c)

    # ----- introspection -----

    @property
    def branches(self) -> List[SenseBranch]:
        return list(self._branches)

    @property
    def union_classes(self) -> List[int]:
        return list(self._union_classes)

    # ----- routing -----

    def per_branch_z(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Run each branch's sense pipeline; return a list of (B, l0_dim)
        z tensors, one per branch."""
        return [b.compute_z(images) for b in self._branches]

    def gate_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Per-input, per-branch archive log-likelihood. Each branch
        scores its OWN z under its own per-class manifold archive.
        Shape (B, N_branches)."""
        zs = self.per_branch_z(images)
        cols = [b.archive_log_likelihood(z) for b, z in zip(self._branches, zs)]
        return torch.stack(cols, dim=-1)

    def gates(
        self,
        images: torch.Tensor,
        *,
        mode: str = "soft",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Branch routing weights, shape (B, N_branches), rows sum to 1.

        Modes:
          * ``soft``    — softmax(log_lik / T) across branches
          * ``hard``    — one-hot argmax_b log_lik
          * ``uniform`` — 1/N (parity ablation; matches the old
                          SensoryConductor's mean_logit fusion)
        """
        n = len(self._branches)
        if n == 0:
            raise RuntimeError("SensoryOrganism has no branches.")
        if mode == "uniform":
            return images.new_full((images.shape[0], n), 1.0 / n)
        log_lik = self.gate_logits(images)
        if mode == "hard":
            idx = log_lik.argmax(dim=-1)
            g = torch.zeros_like(log_lik)
            g.scatter_(1, idx.unsqueeze(1), 1.0)
            return g
        if mode == "soft":
            return F.softmax(log_lik / max(temperature, 1e-6), dim=-1)
        raise ValueError(f"Unknown routing mode: {mode}")

    # ----- forward -----

    def forward(
        self,
        images: torch.Tensor,
        *,
        routing: str = "soft",
        temperature: float = 1.0,
        normalize_per_branch: bool = False,
        bias_offset: Optional[torch.Tensor] = None,
        return_extras: bool = False,
    ):
        """Run images through the organism. Returns logits over
        ``union_classes`` in the order they appear in that list.

        normalize_per_branch=False (default) — combine raw head logits
            weighted by gates: combined[c] = Σ_b g_b · pad(logits_b)[c].

        normalize_per_branch=True — each branch's logits pass through
            log_softmax restricted to its OWN covered classes; gates
            enter in log-space:
                combined[c] = log Σ_b g_b · P_b(c | x)   (over branches
                covering c). Implemented in log-space via logsumexp.

        bias_offset (n_union,) is added to combined logits if supplied
            (calibrator parameter; matches MultiBranchOrganism).

        return_extras=True also returns a dict with ``z_per_branch``
            (list of length N_branches), ``gates`` (B, N_branches),
            ``branch_logits_padded`` (B, N_branches, n_union).
        """
        n = len(self._branches)
        if n == 0:
            raise RuntimeError("SensoryOrganism has no branches.")
        device = images.device
        n_union = len(self._union_classes)
        B = images.shape[0]

        zs = self.per_branch_z(images)
        if routing == "uniform":
            gates = images.new_full((B, n), 1.0 / n)
        else:
            log_lik = torch.stack(
                [b.archive_log_likelihood(z) for b, z in zip(self._branches, zs)],
                dim=-1,
            )
            if routing == "hard":
                idx = log_lik.argmax(dim=-1)
                gates = torch.zeros_like(log_lik)
                gates.scatter_(1, idx.unsqueeze(1), 1.0)
            elif routing == "soft":
                gates = F.softmax(log_lik / max(temperature, 1e-6), dim=-1)
            else:
                raise ValueError(f"Unknown routing mode: {routing}")

        # Per-branch padded logits over the union (B, N, n_union).
        branch_padded = images.new_full(
            (B, n, n_union),
            0.0 if not normalize_per_branch else float("-inf"),
        )
        for bi, (b, z) in enumerate(zip(self._branches, zs)):
            head_logits = b.forward_from_l0(z)         # (B, head_size_b)
            cov = b.classes_covered
            cols = head_logits[:, cov]                  # (B, |cov|)
            if normalize_per_branch:
                cols = F.log_softmax(cols, dim=-1)
            for j, c in enumerate(cov):
                ui = self._class_to_union[c]
                branch_padded[:, bi, ui] = cols[:, j]

        if normalize_per_branch:
            log_g = torch.log(gates.clamp_min(1e-30)).unsqueeze(-1)  # (B, N, 1)
            combined = torch.logsumexp(branch_padded + log_g, dim=1)
        else:
            combined = (branch_padded * gates.unsqueeze(-1)).sum(dim=1)

        if bias_offset is not None:
            combined = combined + bias_offset

        if return_extras:
            return combined, {
                "z_per_branch": zs,
                "gates": gates,
                "branch_logits_padded": branch_padded,
            }
        return combined

    # ----- per-branch confidence features (for Stage B calibrator) -----

    def branch_features(
        self, images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Per-input, per-branch confidence signals — feeds the
        Stage-B router/ambiguity heads.

        Returns a dict with these fields, all shape (B, N_branches):
          * archive_loglik     — mixture log-pdf under archive
          * archive_top1_logpdf — max per-class log-pdf
          * archive_margin     — top1 - top2 of per-class log-pdfs
          * head_softmax_max   — max softmax over branch's head columns
          * head_logit_margin  — top1 - top2 of branch's head logits

        Plus ``branch_logits_padded`` (B, N_branches, n_union) so a
        downstream router can fuse with arbitrary gates without
        re-running forward.
        """
        B = images.shape[0]
        n = len(self._branches)
        n_union = len(self._union_classes)
        zs = self.per_branch_z(images)

        archive_ll  = images.new_zeros(B, n)
        archive_top = images.new_zeros(B, n)
        archive_mrg = images.new_zeros(B, n)
        head_smax   = images.new_zeros(B, n)
        head_mrg    = images.new_zeros(B, n)
        padded      = images.new_zeros(B, n, n_union)

        for bi, (b, z) in enumerate(zip(self._branches, zs)):
            # Archive features (per-class diag-Gaussian log-pdf).
            per_class_logpdf = b.per_class_log_likelihood(z)        # (B, C_b)
            archive_ll[:, bi] = b.archive_log_likelihood(z)
            top2 = torch.topk(per_class_logpdf, k=min(2, per_class_logpdf.shape[-1]),
                               dim=-1).values                         # (B, 2)
            archive_top[:, bi] = top2[:, 0]
            if top2.shape[-1] > 1:
                archive_mrg[:, bi] = top2[:, 0] - top2[:, 1]

            # Head features.
            head_logits = b.forward_from_l0(z)                       # (B, head_size)
            cov = b.classes_covered
            cols = head_logits[:, cov]                                # (B, |cov|)
            sm = F.softmax(cols, dim=-1)
            head_smax[:, bi] = sm.max(dim=-1).values
            top2h = torch.topk(cols, k=min(2, cols.shape[-1]),
                               dim=-1).values
            if top2h.shape[-1] > 1:
                head_mrg[:, bi] = top2h[:, 0] - top2h[:, 1]
            else:
                head_mrg[:, bi] = top2h[:, 0]
            for j, c in enumerate(cov):
                ui = self._class_to_union[c]
                padded[:, bi, ui] = cols[:, j]

        return {
            "archive_loglik": archive_ll,
            "archive_top1_logpdf": archive_top,
            "archive_margin": archive_mrg,
            "head_softmax_max": head_smax,
            "head_logit_margin": head_mrg,
            "branch_logits_padded": padded,
        }

    # ----- diagnostics -----

    def storage_bytes(self) -> Dict[str, int]:
        """Per-branch and total byte breakdown. Unlike
        MultiBranchOrganism there's no shared L0 line — each branch
        owns its own L0 since input dims differ across senses."""
        per_branch_l0 = 0
        substrate = 0
        archive = 0
        std = 0
        for b in self._branches:
            l0_layer = b.net.layers[0]
            per_branch_l0 += (
                l0_layer.W.numel() * l0_layer.W.element_size()
                + l0_layer.b.numel() * l0_layer.b.element_size()
            )
            for layer in b.net.layers[1:]:
                substrate += layer.W.numel() * layer.W.element_size()
                substrate += layer.b.numel() * layer.b.element_size()
            for (mu, sg) in b.manifold_stats.values():
                archive += (mu.numel() + sg.numel()) * mu.element_size()
            if b.standardizer is not None:
                std += (
                    b.standardizer.mean.numel() * b.standardizer.mean.element_size()
                    + b.standardizer.std.numel() * b.standardizer.std.element_size()
                )
        return {
            "per_branch_l0_bytes": per_branch_l0,
            "branch_substrate_bytes": substrate,
            "archive_bytes": archive,
            "standardizer_bytes": std,
            "total_bytes": per_branch_l0 + substrate + archive + std,
        }


__all__ = ["SenseBranch", "SensoryOrganism"]
