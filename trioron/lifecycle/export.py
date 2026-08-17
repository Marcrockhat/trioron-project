"""Dense export — compile a substrate to a fixed, dependency-free forward.
See spec §5.4 (Ship → Export) and §6.4.

The live substrate forward walks the arena: a ``[batch, capacity]``
activation buffer over ALL allocated cells (dormant capacity included),
gather/scatter over edge lists per (rank, phenotype) bucket. That is the
right shape for a *growing* organism — cells divide, edges append — but
it is pure overhead for a *shipped* one. Measured s050 on the world
router (77-in / 32 interior / 6-out, 113 live cells in a 2048-cell
arena, 1 CPU thread): arena forward 485 µs; this export 47 µs
JIT-frozen — the same latency as a 27 K-param DQN MLP at 1/5 the
parameters. A nest decision (router + one leaf) is ~106 µs.

The export is exact (max |Δ| ~1e-7 fp32) for substrates whose live
buckets are LINEAR, DENDRITE (quad σ(z)=z+z², per-branch α, K=1
suppressed to linear) or TANH — the phenotypes every shipped world
leaf/router uses. Buckets are folded in plan order into a running
activation vector laid out ``[inputs | bucket0 cells | bucket1 cells |
…]``, so each stage is one matmul over everything upstream and a
concatenation — no index writes, traceable by ``torch.jit.trace``.
Receptor cells (PCLL, §10.2) and ATTENTION/CONV/RECURRENT buckets are
NOT folded; ``export_dense`` raises for them (fall back to the live
forward, or extend this module along the same pattern).

**The export does not learn.** It is buffers-only (no grad, no arena):
no cells, no λ, no growth/lock/dream. Learning stays in the arena
substrate (``ship`` its checkpoint); after every wake/extend/dream cycle
re-run ``export_dense`` (milliseconds) and swap the serving artifact.
There is deliberately no export→arena path — the export drops lineage,
epigenome, λ and dormant capacity, everything the organism grows with.

Usage::

    module = export_dense(substrate)          # torch.nn.Module, buffers only
    y = module(x)                             # == substrate(x)
    fast = torch.jit.freeze(torch.jit.trace(module, x[:1]))
    verify_export(substrate, module, x)       # max |Δ|
"""
from __future__ import annotations

import torch

from trioron.core.construct import Substrate
from trioron.core.epigenome import DENDRITE, LINEAR, TANH

_SUPPORTED = (LINEAR, DENDRITE, TANH)


class DenseStage(torch.nn.Module):
    """One bucket: y = bias + Σ_k α_k · σ_k(Σ_j chunk_j @ W_{k,j}), σ per
    phenotype. ``chunks`` are the upstream activation blocks
    (inputs, then each earlier stage's output) — summing per-chunk matmuls
    avoids materialising a concatenated activation vector."""

    def __init__(self, Ws: list[torch.Tensor], alpha: torch.Tensor,
                 is_multi: torch.Tensor, bias: torch.Tensor,
                 phenotype: int) -> None:
        super().__init__()
        self.n_chunks = len(Ws)
        self.k1 = Ws[0].shape[0] == 1                # single branch: plain 2-D matmul
        for j, W in enumerate(Ws):                  # each [K, n_chunk_j, n_cells]
            self.register_buffer(f"W{j}", W[0].contiguous() if self.k1 else W)
        self.register_buffer("alpha", alpha.unsqueeze(1))  # [K, 1, n_cells]
        self.register_buffer("is_multi", is_multi)  # [n_cells]
        self.register_buffer("bias", bias)          # [n_cells]
        self.tanh = phenotype == TANH
        self.quad = bool(is_multi.any())
        # K=1 ⇒ α is the fixed 1.0 (spec §3.6 back-compat), so it is folded away
        self.use_alpha = not self.k1 or not bool((alpha == 1).all())

    def forward(self, chunks: list[torch.Tensor]) -> torch.Tensor:
        z = torch.matmul(chunks[0], self.W0)        # [K, batch, n] or [batch, n]
        for j in range(1, self.n_chunks):
            z = z + torch.matmul(chunks[j], getattr(self, f"W{j}"))
        if self.quad:
            z = z + self.is_multi * z * z           # σ(z)=z+z² where K≥2
        if self.k1:
            y = self.bias + (z * self.alpha[0] if self.use_alpha else z)
        else:
            y = self.bias + (self.alpha * z).sum(0)
        return torch.tanh(y) if self.tanh else y


class DenseExport(torch.nn.Module):
    """Buffers-only module equal to ``substrate.forward`` on live cells."""

    def __init__(self, n_in: int, stages: list[DenseStage],
                 out_idx: torch.Tensor, out_is_last: bool) -> None:
        super().__init__()
        self.n_in = n_in
        self.stages = torch.nn.ModuleList(stages)
        self.register_buffer("out_idx", out_idx)
        self.out_is_last = out_is_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = [x[:, : self.n_in]]
        for st in self.stages:
            chunks.append(st(chunks))
        if self.out_is_last:
            return chunks[-1]
        return torch.cat(chunks, dim=1)[:, self.out_idx]

    @property
    def n_params(self) -> int:
        return sum(b.numel() for b in self.buffers()) - self.out_idx.numel()


@torch.no_grad()
def export_dense(sub: Substrate) -> DenseExport:
    """Fold the compiled dispatch plan into a ``DenseExport``.

    Requires ``sub.compile()`` to have run (it has if the substrate was
    trained or ``prepare_training()``'d). Raises ``NotImplementedError``
    for receptor cells or unsupported phenotypes."""
    plan, a = sub.scheduler._plan, sub.arena
    if plan.receptor_ids.numel() > 0:
        raise NotImplementedError("dense export: receptor cells (PCLL) are not folded")
    col = plan.column_ids.long()
    n_in = col.numel()
    idx = torch.full((a.capacity,), -1, dtype=torch.long)     # compact index
    chunk_of = torch.full((a.capacity,), -1, dtype=torch.long)  # which chunk
    idx[col] = torch.arange(n_in)
    chunk_of[col] = 0
    chunk_sizes = [n_in]
    stages: list[DenseStage] = []
    for b in plan.buckets:
        if b.phenotype not in _SUPPORTED:
            raise NotImplementedError(f"dense export: phenotype {b.phenotype} not folded")
        cells = b.cell_ids.long()
        n = cells.numel()
        src_c = chunk_of[b.edge_src.long()]
        src_i = idx[b.edge_src.long()]
        if (src_c < 0).any():
            raise RuntimeError("dense export: edge from a cell outside the plan")
        dst = b.edge_dst_local.long()
        w = a.edge_weight[b.edge_indices].detach()
        if b.phenotype == DENDRITE and b.edge_indices.numel() > 0:
            eb = a.edge_branch[b.edge_indices].long()
            K = int(eb.max()) + 1
            alpha = a.branch_alpha[cells][:, :K].detach().t().contiguous()
            is_multi = (a.n_branches[cells] >= 2).float()
        else:
            eb = torch.zeros_like(dst)
            K = 1
            alpha = torch.ones(1, n)
            is_multi = torch.zeros(n)
        Ws = []
        for j, sz in enumerate(chunk_sizes):
            m = src_c == j
            W = torch.zeros(K, sz, n)
            if bool(m.any()):
                W.index_put_((eb[m], src_i[m], dst[m]), w[m], accumulate=True)
            Ws.append(W)
        stages.append(DenseStage(Ws, alpha, is_multi,
                                 a.bias[cells].detach().clone(), b.phenotype))
        idx[cells] = torch.arange(n)
        chunk_of[cells] = len(chunk_sizes)
        chunk_sizes.append(n)
    outs = plan.output_ids.long()
    if (chunk_of[outs] < 0).any():
        raise RuntimeError("dense export: output cell not reached by any bucket")
    offsets = torch.tensor([0] + chunk_sizes[:-1]).cumsum(0)
    out_idx = offsets[chunk_of[outs]] + idx[outs]
    last = len(chunk_sizes) - 1
    out_is_last = bool((chunk_of[outs] == last).all()) and \
        bool((idx[outs] == torch.arange(outs.numel())).all()) and \
        outs.numel() == chunk_sizes[-1]
    return DenseExport(n_in, stages, out_idx, out_is_last).eval()


@torch.no_grad()
def verify_export(sub: Substrate, module: torch.nn.Module,
                  x: torch.Tensor) -> float:
    """max |module(x) − substrate(x)| — should be ~1e-7 in fp32."""
    return float((module(x[:, : module.n_in]) - sub(x)).abs().max())
