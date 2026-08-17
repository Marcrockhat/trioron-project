"""Grafting — donor-to-recipient transplant.  See spec §5.3."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from trioron.core.arena import Arena
from trioron.core.construct import Substrate
from trioron.core.epigenome import (
    PERCEPTION, OUTPUT, has_gene,
)
from trioron.core.state import CellState


@dataclass
class GraftResult:
    donor_ids: list[int]
    recipient_ids: list[int]
    n_edges_added: int
    id_map: dict[int, int]


def graft(
    recipient: Substrate,
    donor: Substrate,
    *,
    freeze: bool = True,
    wiring: str = "dense",
    merge_output: bool = False,
) -> GraftResult:
    """Transplant donor cells into recipient arena.

    Donor perception cells are NOT copied — donor edges that sourced
    from perception are rewired to the recipient's perception cells
    (matched by rank-order position). This avoids doubling the input
    dimension.

    Per-cell state carried across: bias, λ (node_lambda), engagement,
    position, epigenome/phenotype, rank, age, output_dim, k_unroll,
    division_mode, and the dendritic compartment state (n_branches,
    branch_alpha, per-edge edge_branch) — a quad-dendrite donor stays a
    quad-dendrite after the graft (spec §3.6; without this the transplant
    silently linearises the donor).

    Args:
        recipient: target substrate that receives the graft.
        donor: source substrate whose cells are copied.
        freeze: if True (Protocol A), donor cells are dormant with frozen weights.
                if False (Protocol B), donor cells are active and trainable.
        wiring: edge policy between donor and recipient interior cells.
            ``"dense"`` — every recipient output-rank cell gets edges from
                          every donor interior cell.
            ``"sparse"`` — random subset bounded by fan-in cap.
            ``"none"`` — no new cross-edges (pure transplant; use with
                          ``merge_output`` for absorption).
        merge_output: head-merged absorption (the v2 analog of v1.1
            pool-matched absorb, spec §5.3). Donor OUTPUT cells are NOT
            copied; donor edges into donor output cell *j* are rewired onto
            the recipient's output cell *j* (rank-order match, weights kept)
            and the donor output bias is added to the recipient's. Requires
            equal output widths. Result: ``merged(x) == recipient(x) +
            donor(x)`` exactly (Q-heads sum) — donor and recipient must
            share input AND output spaces.

    Returns:
        GraftResult with mapping from donor to recipient cell ids.
    """
    r_arena = recipient.arena
    d_arena = donor.arena

    d_alive = d_arena.alive_ids()

    d_perc_ids = set()
    d_out_ids: list[int] = []
    d_non_perc = []
    for d_id in d_alive.tolist():
        epi = int(d_arena.epigenome[d_id].item())
        if has_gene(epi, PERCEPTION):
            d_perc_ids.add(d_id)
        elif merge_output and has_gene(epi, OUTPUT):
            d_out_ids.append(d_id)
        else:
            d_non_perc.append(d_id)

    if not d_non_perc:
        return GraftResult([], [], 0, {})

    r_perc_mask = r_arena.alive & has_gene(r_arena.epigenome, PERCEPTION).bool()
    r_perc_ids = r_perc_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

    d_perc_sorted = sorted(d_perc_ids)
    perc_remap: dict[int, int] = {}
    for i, d_pid in enumerate(d_perc_sorted):
        if i < len(r_perc_ids):
            perc_remap[d_pid] = r_perc_ids[i]

    out_remap: dict[int, int] = {}
    if merge_output:
        r_out_mask = r_arena.alive & has_gene(r_arena.epigenome, OUTPUT).bool()
        r_out_sorted = r_out_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        d_out_sorted = sorted(d_out_ids)
        if len(d_out_sorted) != len(r_out_sorted):
            raise ValueError(
                f"graft(merge_output=True): output widths differ "
                f"(donor {len(d_out_sorted)}, recipient {len(r_out_sorted)})")
        out_remap = dict(zip(d_out_sorted, r_out_sorted))
        with torch.no_grad():
            for d_oid, r_oid in out_remap.items():
                r_arena.bias[r_oid] += d_arena.bias[d_oid].detach()

    new_ids = r_arena.alloc(len(d_non_perc))
    id_map: dict[int, int] = {}

    # A graft is a STRUCTURAL transplant, never a differentiable op. The donor's bias
    # (and weights) carry ``requires_grad`` if the donor was trained; copying them into
    # the recipient via in-place index-assign WITHOUT no_grad makes the recipient's bias
    # a non-leaf tensor, which the optimizer then refuses ("can't optimize a non-leaf
    # Tensor"). Detach under no_grad so recipient parameters stay optimizable leaves.
    with torch.no_grad():
        for i, d_id in enumerate(d_non_perc):
            r_id = int(new_ids[i].item())
            id_map[d_id] = r_id

            r_arena.bias[r_id] = d_arena.bias[d_id].detach().clone()
            r_arena.engagement[r_id] = d_arena.engagement[d_id].clone()
            r_arena.utility[r_id] = 0.0
            r_arena.position[r_id] = d_arena.position[d_id].clone()
            r_arena.epigenome[r_id] = d_arena.epigenome[d_id].clone()
            r_arena.phenotype_cache[r_id] = d_arena.phenotype_cache[d_id].clone()
            r_arena.rank[r_id] = d_arena.rank[d_id].clone()
            r_arena.age[r_id] = d_arena.age[d_id].clone()
            r_arena.output_dim[r_id] = d_arena.output_dim[d_id].clone()
            r_arena.forward_inclusion[r_id] = True
            # λ + recurrent/division + dendritic compartments (spec §3.6)
            r_arena.node_lambda[r_id] = d_arena.node_lambda[d_id].clone()
            r_arena.k_unroll[r_id] = d_arena.k_unroll[d_id].clone()
            r_arena.division_mode[r_id] = d_arena.division_mode[d_id].clone()
            r_arena.n_branches[r_id] = d_arena.n_branches[d_id].clone()
            nb = min(r_arena.branch_cap, d_arena.branch_cap)
            r_arena.branch_alpha[r_id, :nb] = d_arena.branch_alpha[d_id, :nb].detach()

            if freeze:
                r_arena.state[r_id] = CellState.DORMANT
            else:
                r_arena.state[r_id] = CellState.ACTIVE

            r_arena.lineage_root[r_id] = r_id
            r_arena.parent[r_id] = -1

    full_remap = {**perc_remap, **out_remap, **id_map}

    d_src = d_arena.edge_src[:d_arena.edge_cursor]
    d_dst = d_arena.edge_dst[:d_arena.edge_cursor]
    d_wt = d_arena.edge_weight[:d_arena.edge_cursor]
    d_br = d_arena.edge_branch[:d_arena.edge_cursor]

    intra_src = []
    intra_dst = []
    intra_wt = []
    intra_br = []

    for ei in range(d_arena.edge_cursor):
        s = int(d_src[ei].item())
        d = int(d_dst[ei].item())
        if s in full_remap and d in full_remap:
            intra_src.append(full_remap[s])
            intra_dst.append(full_remap[d])
            intra_wt.append(d_wt[ei].item())
            intra_br.append(int(d_br[ei].item()))

    if intra_src:
        e0 = r_arena.edge_cursor
        r_arena.add_edges(
            torch.tensor(intra_src, dtype=torch.int32, device=r_arena.device),
            torch.tensor(intra_dst, dtype=torch.int32, device=r_arena.device),
            torch.tensor(intra_wt, device=r_arena.device),
        )
        r_arena.edge_branch[e0:e0 + len(intra_br)] = torch.tensor(
            intra_br, dtype=torch.int8, device=r_arena.device)

    n_cross = 0
    if wiring == "dense":
        n_cross = _wire_dense(r_arena, id_map, d_arena)
    elif wiring == "sparse":
        n_cross = _wire_sparse(r_arena, id_map, d_arena, max_per_cell=4)

    recipient.compile()

    return GraftResult(
        donor_ids=list(id_map.keys()),
        recipient_ids=list(id_map.values()),
        n_edges_added=len(intra_src) + n_cross,
        id_map=id_map,
    )


def _wire_dense(
    r_arena: Arena,
    id_map: dict[int, int],
    d_arena: Arena,
) -> int:
    """Wire every recipient output-rank cell to read from donor interior cells."""
    donor_interior = []
    for d_id, r_id in id_map.items():
        epi = int(d_arena.epigenome[d_id].item())
        if not has_gene(epi, PERCEPTION) and not has_gene(epi, OUTPUT):
            donor_interior.append(r_id)

    if not donor_interior:
        return 0

    r_output_mask = (
        r_arena.alive
        & has_gene(r_arena.epigenome, OUTPUT).bool()
    )
    r_out_ids = r_output_mask.nonzero(as_tuple=False).squeeze(-1)

    existing_donor_set = set(id_map.values())
    r_out_ids = torch.tensor(
        [oid for oid in r_out_ids.tolist() if oid not in existing_donor_set],
        dtype=torch.int32,
        device=r_arena.device,
    )

    if r_out_ids.numel() == 0:
        return 0

    src_list = []
    dst_list = []
    for oid in r_out_ids.tolist():
        for did in donor_interior:
            src_list.append(did)
            dst_list.append(oid)

    if src_list:
        r_arena.add_edges(
            torch.tensor(src_list, dtype=torch.int32, device=r_arena.device),
            torch.tensor(dst_list, dtype=torch.int32, device=r_arena.device),
        )

    return len(src_list)


def _wire_sparse(
    r_arena: Arena,
    id_map: dict[int, int],
    d_arena: Arena,
    max_per_cell: int = 4,
) -> int:
    """Wire a random subset of cross-edges, bounded by fan-in cap."""
    donor_interior = []
    for d_id, r_id in id_map.items():
        epi = int(d_arena.epigenome[d_id].item())
        if not has_gene(epi, PERCEPTION) and not has_gene(epi, OUTPUT):
            donor_interior.append(r_id)

    if not donor_interior:
        return 0

    r_output_mask = (
        r_arena.alive
        & has_gene(r_arena.epigenome, OUTPUT).bool()
    )
    r_out_ids = r_output_mask.nonzero(as_tuple=False).squeeze(-1)

    existing_donor_set = set(id_map.values())
    r_out_ids = torch.tensor(
        [oid for oid in r_out_ids.tolist() if oid not in existing_donor_set],
        dtype=torch.int32,
        device=r_arena.device,
    )

    if r_out_ids.numel() == 0:
        return 0

    src_list = []
    dst_list = []

    for oid in r_out_ids.tolist():
        k = min(max_per_cell, len(donor_interior))
        perm = torch.randperm(len(donor_interior), device=r_arena.device)[:k]
        for idx in perm.tolist():
            src_list.append(donor_interior[idx])
            dst_list.append(oid)

    if src_list:
        r_arena.add_edges(
            torch.tensor(src_list, dtype=torch.int32, device=r_arena.device),
            torch.tensor(dst_list, dtype=torch.int32, device=r_arena.device),
        )

    return len(src_list)


def compose_heads(
    donor_manifold_log_likelihood: torch.Tensor,
    donor_logits: torch.Tensor,
    recipient_logits: torch.Tensor,
) -> torch.Tensor:
    """Soft routing at the head when donor/recipient cover different class spaces.

    logit_c = log sum_b g_b * exp(branch_b logit for c)

    Args:
        donor_manifold_log_likelihood: [batch] per-input donor log-likelihood.
        donor_logits: [batch, n_donor_classes] from donor branch.
        recipient_logits: [batch, n_recipient_classes] from recipient.

    Returns:
        Composed logits [batch, max(n_donor, n_recipient)] with log-sum-exp routing.
    """
    batch = donor_logits.shape[0]
    n_d = donor_logits.shape[1]
    n_r = recipient_logits.shape[1]
    n_out = max(n_d, n_r)

    gate = torch.sigmoid(donor_manifold_log_likelihood).unsqueeze(-1)

    d_pad = torch.full((batch, n_out), float("-inf"), device=donor_logits.device)
    d_pad[:, :n_d] = donor_logits

    r_pad = torch.full((batch, n_out), float("-inf"), device=recipient_logits.device)
    r_pad[:, :n_r] = recipient_logits

    composed = torch.logsumexp(
        torch.stack([
            d_pad + gate.log(),
            r_pad + (1 - gate).log(),
        ], dim=0),
        dim=0,
    )

    return composed
