"""H-space manifold routing — a learned task selector over interior codes.

Promoted from ``archive/experiments/bench_chained_15_v2.py`` (commits 62aa57e,
7e561e4), where a SECOND ManifoldArchive fitted on interior-cell activations
("z2", H-space) routed queries to tasks/classes by Gaussian log-likelihood:
chained-15 full-softmax 0.55 → 0.68 (diagonal) → 0.76 (full-covariance
Mahalanobis).  Transfer to the embodied world validated in
``archive/experiments/world/world_routing.py``.

Why H-space: the shared output head drifts toward the last-trained task
(the forgetting story, manual §7), but the interior code is comparatively
stable — so task identity is inferred from where a query lands in H-space,
not from head logits.  Two routing modes:

- **class** — pure QDA: argmax per-class log-likelihood, bypasses the head.
- **group** — the manifold picks the task/context (robust), then the head's
  logits pick the class within it (accurate).

The archive stores per-class (μ, σ[, Σ]); the router adds only argmax
orchestration — no trainable parameters, no stored data.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from .manifold import ManifoldArchive, get_code_boundary, get_interior_ids

NEG_LL = -1e12  # score for classes with no manifold entry


class ManifoldRouter:
    """Generative task/class selector over a ManifoldArchive of codes.

    ``archive`` is typically fitted on interior (H-space) activations via
    :func:`build_h_archive_from_manifold` / :func:`build_h_archive_from_data`,
    but any code space with per-class astrocytes works (e.g. perception space).
    """

    def __init__(self, archive: ManifoldArchive, full_cov: bool | None = None) -> None:
        self.archive = archive
        self.full_cov = archive.full_cov if full_cov is None else full_cov

    def _ll(self, astro, codes: torch.Tensor) -> torch.Tensor:
        if self.full_cov:
            return astro.log_likelihood_full(codes)
        return astro.log_likelihood(codes)

    def class_log_likelihood(
        self,
        codes: torch.Tensor,
        class_ids: Sequence[int] | None = None,
        n_classes: int | None = None,
    ) -> torch.Tensor:
        """Per-class log-likelihood matrix ``[N, n_classes]``.

        Classes without an archive entry score ``NEG_LL`` so they never win
        the argmax.  ``class_ids`` restricts scoring to a subset (default:
        every class the archive has seen).
        """
        ids = list(class_ids) if class_ids is not None else self.archive.class_ids
        width = n_classes if n_classes is not None else (max(ids) + 1 if ids else 0)
        ll = torch.full((codes.shape[0], width), NEG_LL, device=codes.device)
        for cid in ids:
            astro = self.archive.get(cid)
            if astro is not None:
                ll[:, cid] = self._ll(astro, codes)
        return ll

    def route_class(
        self,
        codes: torch.Tensor,
        class_ids: Sequence[int] | None = None,
        n_classes: int | None = None,
    ) -> torch.Tensor:
        """Pure QDA routing: argmax class per query, bypassing the head."""
        return self.class_log_likelihood(codes, class_ids, n_classes).argmax(dim=1)

    def group_log_likelihood(
        self, codes: torch.Tensor, groups: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Per-group score ``[N, n_groups]`` = max over member-class likelihoods."""
        scores = torch.full((codes.shape[0], len(groups)), NEG_LL, device=codes.device)
        for g_idx, members in enumerate(groups):
            ll_max = torch.full((codes.shape[0],), NEG_LL, device=codes.device)
            for cid in members:
                astro = self.archive.get(cid)
                if astro is not None:
                    ll_max = torch.maximum(ll_max, self._ll(astro, codes))
            scores[:, g_idx] = ll_max
        return scores

    def route_group(
        self, codes: torch.Tensor, groups: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Pick the task/context group per query."""
        return self.group_log_likelihood(codes, groups).argmax(dim=1)

    def route_prediction(
        self,
        codes: torch.Tensor,
        logits: torch.Tensor,
        groups: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Group routing + head classification: the chained-15 "task" mode.

        The manifold picks the group; the head's logits, restricted to that
        group's classes, pick the class.  Returns global class ids ``[N]``.
        """
        best_group = self.route_group(codes, groups)
        pred = torch.zeros(codes.shape[0], dtype=torch.long, device=codes.device)
        for g_idx, members in enumerate(groups):
            mask = best_group == g_idx
            if not mask.any():
                continue
            members_t = torch.tensor(list(members), dtype=torch.long, device=logits.device)
            local = logits[mask][:, members_t].argmax(dim=1)
            pred[mask] = members_t[local]
        return pred


# ── H-space archive builders ─────────────────────────────────────


@torch.no_grad()
def build_h_archive_from_data(
    sub,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    h_interior_ids: torch.Tensor | None = None,
    full_cov: bool = False,
) -> ManifoldArchive:
    """Fit an H-space archive by forwarding real ``(x, y)`` batches.

    The oracle path — needs access to data, so it is an upper bound / test
    harness, not the continual-learning path.
    """
    if h_interior_ids is None:
        h_interior_ids = get_interior_ids(sub.arena).long()
    fresh = ManifoldArchive(sub.arena, full_cov=full_cov)
    for x, y in batches:
        _ = sub(x)
        h_act = sub.last_activations[:, h_interior_ids]
        for cid in y.unique().tolist():
            mask = y == cid
            fresh.update_class(int(cid), h_act[mask])
    fresh.finalize_all()
    return fresh


@torch.no_grad()
def build_h_archive_from_manifold(
    sub,
    perc_archive: ManifoldArchive,
    h_interior_ids: torch.Tensor | None = None,
    n_perc: int | None = None,
    samples_per_class: int = 400,
    full_cov: bool = False,
    full_sample: bool = False,
    sample_rank: int | None = None,
    jitter: float = 0.0,
) -> ManifoldArchive:
    """Fit an H-space archive from synthetic perception-manifold samples.

    The storage-free continual-learning path: sample pseudo-inputs from the
    perception-space archive, forward them through the CURRENT substrate, and
    collect interior activations.  Rebuilding against the current substrate
    fixes stale statistics — H-cell activations drift as later tasks train.

    Note: each build allocates fresh astrocyte cells in the arena; rebuild at
    consolidation boundaries, not per batch.
    """
    if h_interior_ids is None:
        h_interior_ids = get_interior_ids(sub.arena).long()
    if n_perc is None:
        n_perc = int(get_code_boundary(sub.arena).numel())
    fresh = ManifoldArchive(sub.arena, full_cov=full_cov)
    for cid in perc_archive.class_ids:
        astro = perc_archive.get(cid)
        if astro is None:
            continue
        syn = None
        if perc_archive.mixture_k > 0:
            syn = perc_archive.sample_mixture(cid, samples_per_class)
        if syn is None:
            if full_sample:
                syn = astro.sample_full(samples_per_class, rank=sample_rank)
            else:
                syn = astro.sample(samples_per_class)
        if jitter > 0:
            syn = syn + jitter * torch.randn_like(syn)
        x_in = torch.zeros(syn.shape[0], n_perc, device=sub.arena.device)
        w = min(syn.shape[1], n_perc)
        x_in[:, :w] = syn[:, :w]
        _ = sub(x_in)
        h_act = sub.last_activations[:, h_interior_ids]
        fresh.update_class(cid, h_act)
    fresh.finalize_all()
    return fresh
