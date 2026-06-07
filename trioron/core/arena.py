"""Arena — SoA parameter storage.  See spec §2.4."""
from __future__ import annotations

import torch

from .envelope import Envelope
from .epigenome import LINEAR, DENDRITE, primary_phenotype
from .state import CellState

_DEFAULT_CAPACITY: int = 4096
_EDGE_FAN_FACTOR: int = 64
_DEFAULT_BRANCH_CAP: int = 4  # B_max — per-cell dendritic branch budget (spec §3.6)


class Arena:
    """Structure-of-Arrays storage for all per-cell and per-edge state.

    The Arena pre-allocates flat tensors up to *capacity* slots.  Cells
    are appended via :meth:`alloc`; edges via :meth:`add_edges`.  No
    realloc or reshape happens during growth (commitment 3).  Compaction
    is a separate operation that may remap ids.
    """

    def __init__(
        self,
        envelope: Envelope,
        *,
        device: str | torch.device = "cpu",
        capacity: int | None = None,
        branch_cap: int = _DEFAULT_BRANCH_CAP,
    ) -> None:
        self.envelope = envelope
        self.device = torch.device(device)

        cap = capacity or envelope.max_cells or _DEFAULT_CAPACITY
        self.capacity = cap
        self.cursor: int = 0
        self.branch_cap = branch_cap  # B_max (spec §3.6)

        # ── Per-cell SoA tensors (§2.1 / §2.4) ──────────────────
        self.bias = torch.zeros(cap, device=self.device)
        # Epigenetic lock λ (the namesake third node variable, blueprint §3 / paper §A.1):
        # a PER-NODE plasticity gate (one per cell). Default driver = row-sum of the cell's
        # incoming Fisher, but settable from any signal (reward / environment / attention)
        # via learning/epigenetic_lock.set_lambda. node_lambda gates both incoming edges and
        # the bias; *_anchor hold the consolidated values it pulls toward; *_fisher are the
        # per-param Fisher EMAs that roll up (row-sum) into node_lambda.
        self.node_lambda = torch.zeros(cap, device=self.device)
        self.bias_fisher = torch.zeros(cap, device=self.device)
        self.bias_anchor = torch.zeros(cap, device=self.device)
        self.engagement = torch.zeros(cap, device=self.device)
        self.utility = torch.zeros(cap, device=self.device)
        self.position = torch.zeros(cap, 3, device=self.device)
        self.epigenome = torch.zeros(cap, dtype=torch.int32, device=self.device)
        self.phenotype_cache = torch.zeros(cap, dtype=torch.int32, device=self.device)
        self.lineage_root = torch.full((cap,), -1, dtype=torch.int32, device=self.device)
        self.parent = torch.full((cap,), -1, dtype=torch.int32, device=self.device)
        self.rank = torch.zeros(cap, dtype=torch.int32, device=self.device)
        self.state = torch.zeros(cap, dtype=torch.int32, device=self.device)
        self.age = torch.zeros(cap, dtype=torch.int32, device=self.device)
        self.output_dim = torch.ones(cap, dtype=torch.int32, device=self.device)
        self.forward_inclusion = torch.ones(cap, dtype=torch.bool, device=self.device)

        # Division mode: probability of symmetric (lateral) division [0, 1]
        # 1.0 = always lateral (builds width), 0.0 = always axial (builds depth)
        self.division_mode = torch.full((cap,), 0.8, device=self.device)

        # ── Dendritic compartmentalization (Axis 5 / spec §3.6) ──
        # n_branches: K per cell (1 = point neuron ≡ linear). branch_alpha:
        # learned per-branch pooling weight α_{v,k}; column 0 initialized to 1
        # so a fresh (K=1) cell pools its single branch at unit gain (affine →
        # linear, the back-compat guarantee). edge_branch: which branch each
        # incoming edge feeds (default 0). See grow_branch().
        self.n_branches = torch.ones(cap, dtype=torch.int32, device=self.device)
        self.branch_alpha = torch.zeros(cap, self.branch_cap, device=self.device)
        self.branch_alpha[:, 0] = 1.0

        # Slot liveness (True if slot holds a cell — active or dormant)
        self.alive = torch.zeros(cap, dtype=torch.bool, device=self.device)

        # ── Edge storage (COO, append-only within a task) ────────
        ecap = cap * _EDGE_FAN_FACTOR
        self._edge_capacity = ecap
        self.edge_src = torch.empty(ecap, dtype=torch.int32, device=self.device)
        self.edge_dst = torch.empty(ecap, dtype=torch.int32, device=self.device)
        self.edge_weight = torch.zeros(ecap, device=self.device)
        self.edge_fisher = torch.zeros(ecap, device=self.device)   # per-edge Fisher EMA (rolls up to node_lambda)
        self.edge_anchor = torch.zeros(ecap, device=self.device)   # consolidated w_anchor
        self.edge_cursor: int = 0
        self.edge_protected = torch.zeros(ecap, dtype=torch.bool, device=self.device)
        self.edge_branch = torch.zeros(ecap, dtype=torch.int8, device=self.device)  # branch id per edge (spec §3.6)

        # Rank dirtiness flag (commitment 4: deferred recompute)
        self.rank_dirty: bool = False

    # ── Cell allocation ───────────────────────────────────────────

    def alloc(self, n: int = 1) -> torch.Tensor:
        """Allocate *n* consecutive cell slots.  Returns int32 ids."""
        if self.cursor + n > self.capacity:
            raise RuntimeError(
                f"Arena capacity exceeded: need {self.cursor + n}, "
                f"have {self.capacity}"
            )
        start = self.cursor
        self.cursor += n
        ids = torch.arange(
            start, start + n, dtype=torch.int32, device=self.device
        )
        self.alive[start : start + n] = True
        self.state[start : start + n] = CellState.ACTIVE
        self.epigenome[start : start + n] = 1 << LINEAR
        self.phenotype_cache[start : start + n] = LINEAR
        return ids

    # ── Edge mutation ─────────────────────────────────────────────

    def add_edges(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        """Append directed edges src[i]→dst[i]."""
        n = src.shape[0]
        if self.edge_cursor + n > self._edge_capacity:
            raise RuntimeError("Edge buffer full")
        s = self.edge_cursor
        self.edge_src[s : s + n] = src.to(torch.int32)
        self.edge_dst[s : s + n] = dst.to(torch.int32)
        with torch.no_grad():
            if weights is not None:
                self.edge_weight[s : s + n] = weights
            else:
                self.edge_weight[s : s + n].normal_(std=0.01)
        self.edge_cursor += n
        self.rank_dirty = True

    def inputs_of(self, cell_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (source_ids, weights) for all edges into *cell_id*."""
        active = self.edge_dst[: self.edge_cursor]
        mask = active == cell_id
        return (
            self.edge_src[: self.edge_cursor][mask],
            self.edge_weight[: self.edge_cursor][mask],
        )

    def fan_in(self, cell_id: int) -> int:
        return int((self.edge_dst[: self.edge_cursor] == cell_id).sum().item())

    # ── Dendritic growth (Axis 5 / spec §3.6) ─────────────────────

    def grow_branch(
        self, cell_id: int, source_cells: torch.Tensor | list[int]
    ) -> int:
        """Split a new dendritic branch off *cell_id*'s fan-in.

        The incoming edges from *source_cells* are reassigned to a fresh
        branch; the cell's ``DENDRITE`` gene is expressed (LINEAR cleared so
        it dispatches as a dendrite), ``n_branches`` is incremented and the
        new branch's pooling weight ``α`` is initialized to 1. Returns the
        new branch index. A K=1 cell stays linear-equivalent (spec §3.6); the
        per-branch quad σ(z)=z+z² only engages once K ≥ 2.
        """
        k = int(self.n_branches[cell_id].item())
        if k >= self.branch_cap:
            raise RuntimeError(
                f"branch budget B_max={self.branch_cap} reached for cell {cell_id}"
            )
        new_branch = k
        srcs = torch.as_tensor(source_cells, dtype=torch.int32, device=self.device)
        dst = self.edge_dst[: self.edge_cursor]
        src = self.edge_src[: self.edge_cursor]
        mask = (dst == cell_id) & torch.isin(src, srcs)
        self.edge_branch[: self.edge_cursor][mask] = new_branch
        self.n_branches[cell_id] = k + 1
        with torch.no_grad():
            self.branch_alpha[cell_id, new_branch] = 1.0
        epi = int(self.epigenome[cell_id].item())
        self.epigenome[cell_id] = (epi & ~(1 << LINEAR)) | (1 << DENDRITE)
        self.refresh_phenotype(cell_id)
        return new_branch

    # ── Phenotype cache ───────────────────────────────────────────

    def refresh_phenotype(self, cell_id: int) -> None:
        """Recompute cached primary phenotype from epigenome."""
        epi = int(self.epigenome[cell_id].item())
        self.phenotype_cache[cell_id] = primary_phenotype(epi)

    def refresh_all_phenotypes(self) -> None:
        """Batch-refresh phenotype cache for all alive cells."""
        alive_ids = self.alive_ids()
        for cid in alive_ids.tolist():
            self.refresh_phenotype(cid)

    # ── Queries ───────────────────────────────────────────────────

    def alive_ids(self) -> torch.Tensor:
        """Int32 tensor of all alive cell ids."""
        return self.alive.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)

    def active_ids(self) -> torch.Tensor:
        mask = self.alive & (self.state == CellState.ACTIVE)
        return mask.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)

    def schedulable_ids(self) -> torch.Tensor:
        """Alive, active, forward_inclusion=True (neurons only)."""
        mask = (
            self.alive
            & (self.state[: self.capacity] == CellState.ACTIVE)
            & self.forward_inclusion
        )
        return mask.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)

    # ── Usage accounting ──────────────────────────────────────────

    @property
    def n_cells(self) -> int:
        return int(self.alive.sum().item())

    @property
    def n_edges(self) -> int:
        return self.edge_cursor

    @property
    def param_bytes(self) -> int:
        """Approximate parameter bytes: 4 per edge weight + 4 per bias."""
        return self.edge_cursor * 4 + self.n_cells * 4

    def pressure(self) -> float:
        return self.envelope.pressure(self.n_cells, self.n_edges, self.param_bytes)

    def allows_growth(
        self, *, add_cells: int = 0, add_edges: int = 0, add_bytes: int = 0
    ) -> bool:
        return self.envelope.allows_growth(
            self.n_cells,
            self.n_edges,
            self.param_bytes,
            add_cells=add_cells,
            add_edges=add_edges,
            add_bytes=add_bytes,
        )
