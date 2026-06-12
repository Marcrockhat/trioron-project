"""Composer arm — genome-constrained growth of REAL tissue.  [D14+D16+§4b]

See docs/design/mixed_stream_growth.md §4 + §4b (all approved, Rocky
s032/s033). The composer is the discrimination growth path that earns
the organism its first computing cells, edges, and depth: a candidate
composition over a receptor dim pair is trialed on buffered evidence
and, when it wins, SPAWNED as a real cell — epigenome = the winning
expression gene, 2 edges from its source receptor cells, rank earned
from topology at compile(). The new cell computes its composition in
the forward path each tick and deposits into its own lock-in row like
any receptor, under a FIXED static frame known at spawn (causal
quantization: every genome form is bounded by construction).

The genome candidate set (ATTENTION/CONV/RECURRENT deferred — no
scalar form on receptor dims):

  LINEAR   : w₀c_i + w₁c_j, wirings sum/diff            frame [−1, 1]
  TANH     : tanh(2.5(w₀c_i + w₁c_j)), sum/diff         frame [−1, 1]
  DENDRITE : c_i² + c_j² — two branches, one input each frame [0, ½]

over pocket values c = q/N_QUANTA − ½. The spawn folds each form into
the REAL phenotype forward over phase activations a = 2π(c+½):
LINEAR/TANH fold affinely (edge w/2π, bias −Σw/2); DENDRITE uses the
s033 identity σ(−(c+½)) = c² − ¼ for σ(z) = z+z² — edge weight
w = −1/(2π) cancels the tied linear term EXACTLY, soma bias ½.

Trial machinery (§4b, probe-validated `run_composer_genome.py`):

  statistic   — residual-incoherence ratio (1−null)/(1−carve) >
                NULL_RATIO=2 (R-unit gain is blind on clumpy
                manifold marginals); permutation null ×3 (shuffle col
                j against col i: marginals survive, the relation
                dies); split-half confirm (selection on one half must
                reproduce on the held-out half).
  wiring      — importance-gated [D14]: dims on which ANY
                judgable-size class buffer coheres (R > IMPORTANCE_R);
                noise columns never enter a trial (failure mode 4).
  settlement  — future-deposit [D9/D13]: the spawn settles ONLY on
                members that arrive after it exists, via the M3
                subject/testify machinery; success pays the winning
                gene's council group (gene-targeted step4 transfer);
                PATIENCE failure hard-retires the cell [D16].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import torch

from trioron.core.epigenome import DENDRITE, LINEAR, TANH
from trioron.core.receptor import N_QUANTA
from trioron.core.state import CellState

from .division import MIN_CHILD, circ_2means

TRIGGER_R = 0.55       # buffer mean coherence below this → hold a trial
SPAWN_R = 0.80         # the carved child must cohere this much
NULL_RATIO = 2.0       # (1−null)/(1−carve): noise pairs 1.0-1.2, tanh
                       # relations 2.5-2.9, dendrite on ring pools 13-22
IMPORTANCE_R = 0.9     # any-class buffer coherence a dim needs to be wired
FAMILY_DEGREE = 3      # lineage-pool walk bound (Rocky s033; 8-seed sweep:
                       # plateau at 3-5, near-root pools degrade results)
PATIENCE = 3           # boundaries for a spawn to gather its virgin verdict
FRESH_MIN = 60         # virgin members for the early verdict
MAX_SPAWNED = 6        # envelope guard on live+pending composer cells
GENE_OF = {"linear": LINEAR, "tanh": TANH, "dendrite": DENDRITE}


@dataclass(frozen=True)
class ComposerSpec:
    """One genome candidate over a receptor dim pair (column indices
    into the controller's receptor order)."""
    gene: str                       # 'linear' | 'tanh' | 'dendrite'
    i: int
    j: int
    w: Optional[Tuple[float, float]]   # sum/diff wiring; None = dendrite

    def value(self, ci: torch.Tensor, cj: torch.Tensor) -> torch.Tensor:
        """The composition over pocket values c ∈ [−½, ½] — EXACTLY what
        the spawned phenotype computes in the forward path."""
        if self.gene == "dendrite":
            return ci * ci + cj * cj
        z = self.w[0] * ci + self.w[1] * cj
        return torch.tanh(2.5 * z) if self.gene == "tanh" else z

    def frame(self) -> Tuple[float, float]:
        """Fixed static frame — bounded by construction [D14]."""
        return (0.0, 0.5) if self.gene == "dendrite" else (-1.0, 1.0)

    def pockets(self, q: torch.Tensor) -> torch.Tensor:
        """Composed pockets from receptor pockets [n, F] (canonical
        frame): value → the spec's static frame → pocket ∈ [1, 999]."""
        c = q / N_QUANTA - 0.5
        v = self.value(c[:, self.i], c[:, self.j])
        lo, hi = self.frame()
        return (N_QUANTA * (v - lo) / (hi - lo)).clamp(1, N_QUANTA - 1)

    def quantize(self, v: torch.Tensor) -> torch.Tensor:
        """Composed pockets from forward-path ACTIVATIONS (the real
        tissue computed v already)."""
        lo, hi = self.frame()
        return (N_QUANTA * (v - lo) / (hi - lo)).clamp(1, N_QUANTA - 1)


def candidates(i: int, j: int):
    for gene in ("linear", "tanh"):
        for w in ((1.0, 1.0), (1.0, -1.0)):
            yield ComposerSpec(gene, i, j, w)
    yield ComposerSpec("dendrite", i, j, None)


# ── trial statistics (§4b, probe-validated) ───────────────────────

def carve_r(q: torch.Tensor, spec: ComposerSpec) -> float:
    """Best-child coherence of the composed dim's circular 2-means
    carve over receptor pockets q [n, F]."""
    phi = 2 * math.pi * spec.pockets(q) / N_QUANTA
    side = circ_2means(phi)
    if min(int(side.sum()), int((~side).sum())) < MIN_CHILD:
        return 0.0
    z = torch.exp(1j * phi)
    return max(float(z[~side].mean().abs()), float(z[side].mean().abs()))


def null_r(q: torch.Tensor, spec: ComposerSpec, g: torch.Generator) -> float:
    """Permutation null ×3: shuffle col j against col i — marginals and
    the static frame survive, the RELATION dies."""
    total = 0.0
    for _ in range(3):
        shuf = q.clone()
        shuf[:, spec.j] = q[torch.randperm(len(q), generator=g), spec.j]
        total += carve_r(shuf, spec) / 3
    return total


def trial_ratio(q: torch.Tensor, spec: ComposerSpec,
                g: torch.Generator) -> Tuple[float, float]:
    """(carve, residual-incoherence ratio) — the §4b trial statistic."""
    r = carve_r(q, spec)
    if r < SPAWN_R or r >= 1.0:
        return r, 0.0
    return r, (1.0 - null_r(q, spec, g)) / (1.0 - r)


def best_candidate(q: torch.Tensor, wired: List[int],
                   taken: Set[ComposerSpec], seed: int
                   ) -> Tuple[Optional[ComposerSpec], float, float]:
    """Overproduce candidates over importance-gated pairs; select on one
    half, confirm on the held-out half (failure modes 1-3). Returns
    (spec, carve, ratio) of the winner, or (None, 0, ratio-floor)."""
    best, best_ratio, best_r = None, NULL_RATIO, 0.0
    if len(q) < 4 * MIN_CHILD:        # split-half must seat 2×2 children
        return best, best_r, best_ratio
    g = torch.Generator().manual_seed(seed)
    half = len(q) // 2
    q1, q2 = q[:half], q[half:]
    for i in wired:
        for j in wired:
            if j <= i:
                continue
            for spec in candidates(i, j):
                if spec in taken:
                    continue
                r1, ratio1 = trial_ratio(q1, spec, g)
                if ratio1 <= best_ratio:
                    continue
                _, ratio2 = trial_ratio(q2, spec, g)
                if ratio2 > NULL_RATIO:
                    best, best_ratio, best_r = spec, ratio1, r1
    return best, best_r, best_ratio


# ── spawn / prune — the structural contract [D11] ─────────────────

def spawn(substrate, spec: ComposerSpec,
          src_cells: Tuple[int, int], parent: int = -1) -> int:
    """Commit the winning candidate as REAL tissue: +1 computing cell
    (epigenome = the winning gene), +2 edges from its source receptor
    cells, rank earned from topology at compile(). The phenotype
    forward over phase activations a = 2π(c+½) reproduces spec.value()
    EXACTLY via affine folding (LINEAR/TANH) or the σ(−(c+½)) = c²−¼
    identity (DENDRITE, w = −1/(2π), soma bias ½)."""
    a = substrate.arena
    cid = int(a.alloc(1).item())
    s = 1.0 / (2 * math.pi)
    src = torch.tensor(src_cells, dtype=torch.int32)
    dst = torch.tensor([cid, cid], dtype=torch.int32)
    if spec.gene == "dendrite":
        a.add_edges(src, dst, torch.tensor([-s, -s]))
        with torch.no_grad():
            a.bias[cid] = 0.5
        a.grow_branch(cid, [src_cells[1]])   # K=2; sets DENDRITE gene
    else:
        k = 2.5 if spec.gene == "tanh" else 1.0
        w0, w1 = spec.w
        a.add_edges(src, dst, torch.tensor([k * w0 * s, k * w1 * s]))
        with torch.no_grad():
            a.bias[cid] = -k * (w0 + w1) / 2
        if spec.gene == "tanh":
            a.epigenome[cid] = 1 << TANH
            a.refresh_phenotype(cid)
    if parent >= 0:
        a.parent[cid] = parent               # the progenitor spawned it
    substrate.compile()                      # rank recomputed from topology
    return cid


def prune(substrate, cell_id: int) -> None:
    """PATIENCE failure hard-retires the cell [D16]: out of the forward
    path, edges masked (weights zeroed), state DORMANT."""
    a = substrate.arena
    a.forward_inclusion[cell_id] = False
    a.state[cell_id] = CellState.DORMANT
    ec = a.edge_cursor
    mask = (a.edge_dst[:ec] == cell_id) | (a.edge_src[:ec] == cell_id)
    with torch.no_grad():
        a.edge_weight[:ec][mask] = 0.0
    substrate.compile()


# ── settlement subject (M3 testify machinery) ─────────────────────

@dataclass
class ComposerPending:
    """The growth decision's subject [D13]: carries the GENE (the
    stress router pays the winning gene's council group) and the
    future-deposit store (virgin members of the proposing lineage)."""
    spec: ComposerSpec
    cell_id: int
    dim: int                              # column in the composed space
    age: int = 0
    fresh: Optional[torch.Tensor] = None  # [n, F] receptor pockets
    lineage: Set[str] = field(default_factory=set)   # class names

    @property
    def gene(self) -> int:
        return GENE_OF[self.spec.gene]
