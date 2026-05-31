"""Substrate Organization Metrics (SOM) — a standardized suite for the
emergent functional structure of a GROWN trioron substrate.

Measuring how a self-organizing substrate organizes itself is largely
uncharted; this suite makes it reproducible by reusing established measures
and reporting each against its chance baseline:

  - sparseness         population lifetime sparseness (Treves-Rolls / Vinje &
                       Gallant 2000). Per cell over class-conditioned mean
                       activations, normalized to [0,1]. 0 = dense/uniform,
                       1 = maximally selective. "Do cells specialize?"
  - assortativity      Newman (2003) categorical assortativity coefficient of
                       the interior↔interior edge graph by each cell's
                       preferred class. r in [-1,1]; 0 = random wiring,
                       >0 = like-function cells wire together (cortical-column
                       analogue). "Do functional modules wire together?"
  - abstraction_grad   Spearman corr between a cell's topological rank and its
                       label-discriminability (eta²). >0 = deeper cells encode
                       the task variable more sharply — the cortical
                       abstraction-rises-with-depth hallmark. "Is depth a
                       hierarchy, or just more layers?"
  - module_silhouette  silhouette score of cells in functional-profile space
                       (best k by silhouette). [-1,1]; higher = better
                       separated functional clusters. "How distinct are the
                       modules?"
  - n_modules          number of well-populated preferred-class groups.
  - effective_depth    number of occupied interior ranks (1 = bipartite MLP).

The novelty is the *application* to a grown/self-arranging substrate, not the
individual statistics — each is standard, which keeps claims modest and
comparable across runs, architectures, and tasks.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score

from trioron.core.epigenome import (
    PERCEPTION, OUTPUT, DENDRITE, RECURRENT, has_gene,
)
from trioron.core.state import CellState


def morphometrics(sub):
    """Anatomical / internal-state census of the substrate (Rocky's ask):
    cell counts by type, dendrite/recurrent counts, lineage clusters,
    average axon (fan-out) and dendritic (fan-in) size, total edges/params.
    Pairs with organization_metrics() to form the full substrate report."""
    a = sub.arena
    alive = a.alive_ids().tolist()
    alive_set = set(alive)

    n_perc = n_out = n_dend = n_rec = n_dormant = n_interior = 0
    for c in alive:
        epi = int(a.epigenome[c].item())
        is_p = has_gene(epi, PERCEPTION)
        is_o = has_gene(epi, OUTPUT)
        if is_p:
            n_perc += 1
        if is_o:
            n_out += 1
        if has_gene(epi, DENDRITE):
            n_dend += 1
        if has_gene(epi, RECURRENT):
            n_rec += 1
        if a.state[c] == CellState.DORMANT:
            n_dormant += 1
        if not is_p and not is_o:
            n_interior += 1

    # lineage clusters: distinct founding ancestors among alive cells
    roots = [int(a.lineage_root[c].item()) for c in alive]
    from collections import Counter
    root_sizes = Counter(roots)
    n_lineage_clusters = len(root_sizes)
    largest_lineage = max(root_sizes.values()) if root_sizes else 0

    # axon (out-degree / fan-out) and dendritic (in-degree / fan-in) sizes
    n = a.edge_cursor
    src = a.edge_src[:n].tolist()
    dst = a.edge_dst[:n].tolist()
    out_deg = Counter(); in_deg = Counter(); n_edges = 0
    for s, d in zip(src, dst):
        if s == d or s not in alive_set or d not in alive_set:
            continue
        out_deg[s] += 1
        in_deg[d] += 1
        n_edges += 1
    # average over cells that actually project / receive
    avg_axon = (sum(out_deg.values()) / len(out_deg)) if out_deg else 0.0
    avg_fanin = (sum(in_deg.values()) / len(in_deg)) if in_deg else 0.0

    n_params = int(sum(p.numel() for p in sub.trainable_tensors()))
    return {
        "n_cells": len(alive),
        "n_perception": n_perc,
        "n_interior": n_interior,
        "n_output": n_out,
        "n_dormant": n_dormant,
        "n_dendrite_cells": n_dend,
        "n_recurrent_cells": n_rec,
        "n_lineage_clusters": n_lineage_clusters,
        "largest_lineage": largest_lineage,
        "n_edges": n_edges,
        "avg_axon_fanout": round(avg_axon, 2),
        "avg_dendritic_fanin": round(avg_fanin, 2),
        "n_params": n_params,
    }


@torch.no_grad()
def _functional_profiles(sub, x, y, iids, n_classes, acts=None):
    """profile[cell, class] = mean |activation| of the cell on class-c inputs.
    Also returns the raw per-sample activations for discriminability.
    `acts` may be precomputed (e.g. decision-step activations of a recurrent
    sequence); otherwise a plain forward on x is used."""
    if acts is None:
        _ = sub(x)
        acts = sub.last_activations[:, iids]
    acts = acts.abs()                                    # [N, H]
    prof = torch.zeros(iids.numel(), n_classes)
    for c in range(n_classes):
        m = (y == c)
        if m.any():
            prof[:, c] = acts[m].mean(dim=0)
    return prof, acts


def _sparseness(prof):
    """Normalized lifetime sparseness per cell, averaged. Treves-Rolls:
    a = (Σ r_c / C)² / (Σ r_c² / C); sparseness = (1-a)/(1-1/C) in [0,1]."""
    C = prof.shape[1]
    r = prof + 1e-8
    a = (r.mean(dim=1) ** 2) / (r.pow(2).mean(dim=1))     # [H], in (0,1]
    sp = (1.0 - a) / (1.0 - 1.0 / C)
    return float(sp.mean().item())


def _newman_assortativity(edges, label_of, n_labels):
    """Newman (2003) categorical assortativity over edges (undirected count).
    r = (Tr e - ||e²||) / (1 - ||e²||), e = normalized mixing matrix."""
    if not edges:
        return float("nan")
    e = np.zeros((n_labels, n_labels))
    for s, d in edges:
        ls, ld = label_of.get(s), label_of.get(d)
        if ls is None or ld is None:
            continue
        e[ls, ld] += 1.0
        e[ld, ls] += 1.0                                 # symmetrize
    tot = e.sum()
    if tot == 0:
        return float("nan")
    e /= tot
    a = e.sum(axis=1)
    tr = np.trace(e)
    denom = 1.0 - (a * a).sum()
    if abs(denom) < 1e-12:
        return float("nan")
    return float((tr - (a * a).sum()) / denom)


def _abstraction_gradient(acts, y, ranks_vec, n_classes):
    """eta² (between-class / total variance) per cell vs rank, Spearman corr.
    >0 ⇒ deeper cells discriminate the task variable more (abstraction rises)."""
    N, H = acts.shape
    grand = acts.mean(dim=0)                             # [H]
    ss_tot = ((acts - grand) ** 2).sum(dim=0) + 1e-8     # [H]
    ss_between = torch.zeros(H)
    for c in range(n_classes):
        m = (y == c)
        if m.any():
            nc = int(m.sum())
            mc = acts[m].mean(dim=0)
            ss_between += nc * (mc - grand) ** 2
    eta2 = (ss_between / ss_tot).cpu().numpy()           # [H], in [0,1]
    if len(set(ranks_vec.tolist())) < 2:
        return float("nan"), float(eta2.mean())          # no rank spread
    rho, _ = spearmanr(ranks_vec.cpu().numpy(), eta2)
    return float(rho), float(eta2.mean())


def organization_metrics(sub, x, y, iids, ranks, n_classes, edges_ii, acts=None):
    """Compute the full SOM suite. `ranks` = dict cell_id->rank; `edges_ii` =
    list of (src,dst) interior↔interior edges. `acts` optionally supplies
    precomputed per-sample interior activations (for recurrent/sequence tasks).
    Returns a dict."""
    prof, acts = _functional_profiles(sub, x, y, iids, n_classes, acts=acts)
    norm = prof / (prof.sum(dim=1, keepdim=True) + 1e-8)
    preferred = norm.argmax(dim=1)
    ranks_vec = torch.tensor([ranks[int(c)] for c in iids.tolist()])

    sparseness = _sparseness(prof)

    label_of = {int(c): int(preferred[i].item()) for i, c in enumerate(iids.tolist())}
    assort = _newman_assortativity(edges_ii, label_of, n_classes)

    abstraction_rho, eta2_mean = _abstraction_gradient(acts, y, ranks_vec, n_classes)

    # module separation: silhouette in functional-profile space (need ≥2
    # populated clusters and > n_clusters samples)
    feats = norm.cpu().numpy()
    labels = preferred.cpu().numpy()
    uniq = np.unique(labels)
    if len(uniq) >= 2 and len(labels) > len(uniq):
        try:
            sil = float(silhouette_score(feats, labels))
        except Exception:
            sil = float("nan")
    else:
        sil = float("nan")
    from collections import Counter
    pop = Counter(labels.tolist())
    n_modules = sum(1 for v in pop.values() if v >= max(2, len(labels) // 20))

    occupied = sorted(set(ranks_vec.tolist()))
    return {
        "n_interior": int(iids.numel()),
        "effective_depth": len(occupied),
        "max_rank": int(max(occupied)),
        "ii_edges": len(edges_ii),
        "sparseness": sparseness,
        "assortativity": assort,            # chance 0
        "abstraction_grad": abstraction_rho,  # chance 0 (>0 = cortex-like)
        "eta2_mean": eta2_mean,
        "module_silhouette": sil,
        "n_modules": n_modules,
        "pref_hist": dict(sorted(pop.items())),
    }
