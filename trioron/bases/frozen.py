"""Frozen base — load a checkpoint as an immutable base.  See spec §2.10."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from trioron.core.epigenome import PERCEPTION, has_gene
from trioron.core.state import CellState

if TYPE_CHECKING:
    from trioron.core.construct import Substrate


class Frozen:
    """Load a shipped checkpoint as a frozen (dormant) base population.

    Non-perception cells are loaded dormant with forward_inclusion=True.
    Perception cells are NOT copied — their edges are rewired to read
    from whatever perception cells exist in the arena at load time
    (typically from a subsequent base like ``seeded``). Use with
    ``compose(frozen(...), seeded(...))`` so seeded provides perception.

    If ``standalone=True``, perception cells ARE included (for cases
    where no other base will supply them).

    Args:
        path: checkpoint file path (from :func:`lifecycle.ship.ship`).
        freeze: if True (default), all loaded cells are dormant.
        standalone: if True, include perception cells.
    """

    def __init__(
        self,
        path: str | Path,
        freeze: bool = True,
        standalone: bool = False,
    ) -> None:
        self.path = Path(path)
        self.freeze = freeze
        self.standalone = standalone
        self._perc_remap_pending: dict[int, int] | None = None
        self._edge_state: dict | None = None

    def __call__(self, substrate: Substrate) -> None:
        state = torch.load(self.path, map_location=substrate.arena.device, weights_only=False)
        arena_state = state["arena"]
        a = substrate.arena

        n = arena_state["cursor"]
        saved_epi = arena_state["epigenome"]

        if self.standalone:
            self._load_all(a, arena_state, n)
        else:
            self._load_skip_perception(a, arena_state, n, saved_epi)

    def _load_all(self, a, arena_state, n):
        new_ids = a.alloc(n)
        start = int(new_ids[0].item())

        for field in [
            "bias", "engagement", "utility", "position", "epigenome",
            "phenotype_cache", "lineage_root", "parent", "rank",
            "age", "output_dim", "forward_inclusion",
        ]:
            src = arena_state[field].to(a.device)
            getattr(a, field)[start:start + n] = src

        if self.freeze:
            a.state[start:start + n] = CellState.DORMANT
        else:
            a.state[start:start + n] = arena_state["state"].to(a.device)

        a.forward_inclusion[start:start + n] = True

        ec = arena_state["edge_cursor"]
        if ec > 0:
            a.add_edges(
                (arena_state["edge_src"].to(a.device) + start).to(torch.int32),
                (arena_state["edge_dst"].to(a.device) + start).to(torch.int32),
                arena_state["edge_weight"].to(a.device),
            )

        a.rank_dirty = True

    def _load_skip_perception(self, a, arena_state, n, saved_epi):
        perc_indices = []
        non_perc_indices = []
        for i in range(n):
            epi = int(saved_epi[i].item())
            if has_gene(epi, PERCEPTION):
                perc_indices.append(i)
            else:
                non_perc_indices.append(i)

        if not non_perc_indices:
            return

        new_ids = a.alloc(len(non_perc_indices))
        id_map: dict[int, int] = {}
        for j, old_idx in enumerate(non_perc_indices):
            r_id = int(new_ids[j].item())
            id_map[old_idx] = r_id

        for old_idx, r_id in id_map.items():
            for field in [
                "bias", "engagement", "utility", "position", "epigenome",
                "phenotype_cache", "rank", "age", "output_dim",
            ]:
                getattr(a, field)[r_id] = arena_state[field][old_idx].to(a.device)

            a.forward_inclusion[r_id] = True
            a.lineage_root[r_id] = r_id
            a.parent[r_id] = -1

            if self.freeze:
                a.state[r_id] = CellState.DORMANT
            else:
                a.state[r_id] = int(arena_state["state"][old_idx].item())

        self._perc_remap_pending = {i: i for i in perc_indices}
        self._edge_state = {
            "src": arena_state["edge_src"],
            "dst": arena_state["edge_dst"],
            "weight": arena_state["edge_weight"],
            "cursor": arena_state["edge_cursor"],
            "id_map": id_map,
            "perc_indices": perc_indices,
            "device": a.device,
        }

        r_perc_mask = a.alive & has_gene(a.epigenome, PERCEPTION).bool()
        r_perc_ids = r_perc_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

        perc_remap = {}
        for i, old_p in enumerate(sorted(perc_indices)):
            if i < len(r_perc_ids):
                perc_remap[old_p] = r_perc_ids[i]

        full_remap = {**perc_remap, **id_map}

        ec = arena_state["edge_cursor"]
        if ec > 0:
            e_src = arena_state["edge_src"][:ec]
            e_dst = arena_state["edge_dst"][:ec]
            e_wt = arena_state["edge_weight"][:ec]

            new_src = []
            new_dst = []
            new_wt = []
            for ei in range(ec):
                s = int(e_src[ei].item())
                d = int(e_dst[ei].item())
                if s in full_remap and d in full_remap:
                    new_src.append(full_remap[s])
                    new_dst.append(full_remap[d])
                    new_wt.append(e_wt[ei].item())

            if new_src:
                a.add_edges(
                    torch.tensor(new_src, dtype=torch.int32, device=a.device),
                    torch.tensor(new_dst, dtype=torch.int32, device=a.device),
                    torch.tensor(new_wt, device=a.device),
                )

        a.rank_dirty = True


def frozen(path: str | Path, freeze: bool = True, standalone: bool = False) -> Frozen:
    """Convenience factory for frozen base."""
    return Frozen(path, freeze=freeze, standalone=standalone)
