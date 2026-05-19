"""Trioron 2.0 — localized growth-direction primitives.

When the substrate is asked to grow (a node, an edge, an inserted
layer), it needs to choose an initialization that captures the
*direction of current unfit signal* at the growth point — what the
substrate is currently failing to express. Trioron 1.0 had one
mechanism for this: residual-SVD over a contrastive pair (top-1 right
singular vector of `f_a - f_b`), used by bench_packnet / bench_step8 /
bench_harder / bench_50task. Many callsites (classification heads,
several probes) bypass it and use random Kaiming init.

This module consolidates the contrastive primitive and generalizes it
to non-contrastive settings, addressing trioron_2_0.md §4 (growth
signal reuse) and unblocking insert_layer's growth_direction init
mode.

Three primitives:

  from_contrastive_pair(net, a, b, dest_layer_idx, k=1)
      The trioron 1.0 mechanism: features f_a and f_b at the input of
      the destination layer, top-K right singular vectors of (f_a - f_b).

  from_per_class_scatter(features, labels, k=1)
      The label-aware non-contrastive generalization. Top-K
      eigenvectors of the between-class scatter matrix S_B = Σ_c n_c
      (μ_c - μ)(μ_c - μ)^T. Equals the contrastive primitive's top-1
      direction (up to sign) when there are exactly two classes with
      equal counts.

  features_at_growth_point(net, x, dest_layer_idx)
      Shared helper: run the forward pass up to (but not including)
      layer `dest_layer_idx` and return the features there. The
      features are the inputs that a new node / edge / inserted layer
      at `dest_layer_idx` would read from.

All three return unit-norm vectors (or row-unit matrices for k > 1).
Caller decides scale (typically via the new node's gain / EWC pull).
"""

from __future__ import annotations
from typing import Optional

import torch


def features_at_growth_point(
    net,
    x: torch.Tensor,
    dest_layer_idx: int,
) -> torch.Tensor:
    """Run the network forward up to (but not including) layer
    `dest_layer_idx` and return the features there. These are the
    inputs that a new node / edge / inserted layer at
    `dest_layer_idx` would read.

    dest_layer_idx == 0 means "at the network input": returns x
    unchanged (a new layer-0 row would read raw x).

    For insert_layer(between=(i, i+1)): the new layer's fan_in matches
    the output of layer i, i.e. the features at dest_layer_idx == i+1.
    For grow_node at layer L: the new node's fan_in matches features
    at dest_layer_idx == L.

    Returns shape (batch, feat_dim) where feat_dim equals the new
    node/edge/layer's fan_in.
    """
    if dest_layer_idx < 0 or dest_layer_idx > len(net.layers):
        raise IndexError(
            f"dest_layer_idx {dest_layer_idx} out of range "
            f"[0, {len(net.layers)}]"
        )
    h = x
    with torch.no_grad():
        for k in range(dest_layer_idx):
            h = net.layers[k](h)
    return h


def from_contrastive_pair(
    net,
    a: torch.Tensor,
    b: torch.Tensor,
    dest_layer_idx: int,
    k: int = 1,
) -> torch.Tensor:
    """Trioron 1.0's residual-SVD growth direction, lifted to a
    canonical module location.

    a, b: matched batches of contrastive inputs (e.g., two opposite
        concept clusters). Each has shape (batch, x_dim).
    dest_layer_idx: the layer whose input we read from. The returned
        vectors live in that input space.
    k: number of directions. k=1 reproduces 1.0 behavior.

    Returns shape (k, feat_dim) — top-K right singular vectors of
    (f_a - f_b) where f_a, f_b are features at the growth point.
    Each row is unit-norm.

    Equivalent to bench_packnet.compute_growth_direction when k=1.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"contrastive pair shape mismatch: a={tuple(a.shape)}, "
            f"b={tuple(b.shape)} — batch dims must match"
        )
    f_a = features_at_growth_point(net, a, dest_layer_idx)
    f_b = features_at_growth_point(net, b, dest_layer_idx)
    D = (f_a - f_b).to(torch.float32)
    _, _, Vh = torch.linalg.svd(D, full_matrices=False)
    # Vh: (min(batch, feat_dim), feat_dim). Top-K rows.
    if k > Vh.shape[0]:
        raise ValueError(
            f"requested k={k} but residual SVD yields only "
            f"{Vh.shape[0]} singular vectors at this growth point"
        )
    vecs = Vh[:k]
    # Normalize defensively (SVD already returns unit-norm rows).
    norms = vecs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return vecs / norms


def from_per_class_scatter(
    features: torch.Tensor,
    labels: torch.Tensor,
    k: int = 1,
) -> torch.Tensor:
    """Non-contrastive growth direction: top-K eigenvectors of the
    between-class scatter matrix at the growth point.

    Per-class means μ_c, global mean μ. Scatter:
        S_B = Σ_c n_c (μ_c - μ)(μ_c - μ)^T
    Top-K eigenvectors of S_B = directions in feature space that best
    separate classes (= LDA's between-class projection).

    Equivalent to from_contrastive_pair's top-1 direction (up to sign)
    when there are exactly two classes with equal counts.

    features: (batch, feat_dim). Caller is responsible for computing
        these via features_at_growth_point() or equivalent.
    labels: (batch,) integer class IDs.
    k: number of directions.

    Returns shape (k, feat_dim) — top-K unit eigenvectors, descending
    by eigenvalue.

    Raises if fewer than 2 classes are present (between-class scatter
    is undefined) or if k > feat_dim.
    """
    if features.ndim != 2:
        raise ValueError(
            f"features must be 2D (batch, feat_dim); got {tuple(features.shape)}"
        )
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} incompatible with features "
            f"shape {tuple(features.shape)}"
        )
    feat_dim = features.shape[1]
    if k > feat_dim:
        raise ValueError(
            f"requested k={k} > feat_dim={feat_dim}"
        )

    # Cast to FP32 for the eigendecomposition; the substrate may be
    # running BF16/FP16 but eigh requires real-typed FP32+ on CPU.
    f = features.detach().to(torch.float32)
    classes = torch.unique(labels)
    if classes.numel() < 2:
        raise ValueError(
            f"per-class scatter needs >= 2 classes; got {classes.numel()}"
        )

    mu_global = f.mean(dim=0)
    S_B = torch.zeros(feat_dim, feat_dim, dtype=torch.float32, device=f.device)
    for c in classes.tolist():
        mask = labels == c
        n_c = int(mask.sum().item())
        if n_c == 0:
            continue
        mu_c = f[mask].mean(dim=0)
        diff = (mu_c - mu_global).unsqueeze(1)  # (feat_dim, 1)
        S_B += float(n_c) * (diff @ diff.T)

    # eigh on a symmetric PSD matrix returns ascending eigenvalues with
    # orthonormal eigenvectors as columns of `vecs`. Take the last k.
    _, vecs = torch.linalg.eigh(S_B)
    top_k = vecs[:, -k:].T  # (k, feat_dim), ascending → flip to descending
    top_k = torch.flip(top_k, dims=[0])
    norms = top_k.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return top_k / norms
