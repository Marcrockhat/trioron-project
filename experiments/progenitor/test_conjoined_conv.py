"""Conjoined-twin CONV feasibility (s040) — design doc §8.

Hand-wires CONV cells against the REAL ``conv.forward_batch`` (the
production weight-tying path) to validate the mechanism before any vote/
proposer is built:

  CHECK 1 (reduction)      a lone CONV cell (own root) == linear y=b+Σw·a.
                           => the CONV gene is inert without a shared root,
                              so a 1-cell spawn is "linear in disguise".
  CHECK 2 (tying)          two twins sharing ONE lineage_root apply the SAME
                           kernel at two patches (translation equivariance),
                           and a backward pass ties their gradients into the
                           one kernel.
  CHECK 3 (benefit)        on a translation task (a local pattern at varying
                           positions, train/test positions DISJOINT), the
                           tied cohort generalizes to unseen positions; an
                           untied (own-root => independent) cohort cannot.

Run:  python3 experiments/progenitor/test_conjoined_conv.py
"""
from __future__ import annotations

import torch

from trioron.core.arena import Arena
from trioron.core.envelope import Envelope
from trioron.core.scheduler import Bucket
from trioron.core.epigenome import CONV, LINEAR
from trioron.phenotype import conv


# ── wiring helpers ──────────────────────────────────────────────────────

def _bucket_for(arena: Arena, cells: list[int]) -> Bucket:
    """Build a CONV bucket over *cells* from the arena's current edges,
    preserving edge insertion order (=> tap ordinal = insertion order
    within each cell's fan-in)."""
    cells_t = torch.tensor(cells, dtype=torch.int32)
    local_of = {c: i for i, c in enumerate(cells)}
    dst = arena.edge_dst[: arena.edge_cursor]
    keep = torch.tensor([int(d) in local_of for d in dst.tolist()])
    e_idx = keep.nonzero(as_tuple=False).squeeze(-1)
    e_dst = arena.edge_dst[e_idx].long()
    e_src = arena.edge_src[e_idx].long()
    e_dst_local = torch.tensor([local_of[int(d)] for d in e_dst.tolist()])
    return Bucket(rank=1, phenotype=CONV, cell_ids=cells_t,
                  edge_indices=e_idx, edge_src=e_src,
                  edge_dst_local=e_dst_local, bias_ids=cells_t)


def _wire_patch(arena: Arena, cell: int, in_cells: list[int],
                kernel: torch.Tensor | None) -> None:
    """Wire *cell*'s fan-in to *in_cells* in offset order. If *kernel*
    given, set those edge weights (the cell carries the kernel); else
    leave default (the cell will read its root's weights)."""
    src = torch.tensor(in_cells, dtype=torch.int32)
    dst = torch.full((len(in_cells),), cell, dtype=torch.int32)
    arena.add_edges(src, dst, kernel)


def _set_conv(arena: Arena, cells: list[int], root: int | None) -> None:
    for c in cells:
        arena.epigenome[c] = 1 << CONV          # CONV only (clear LINEAR)
        arena.phenotype_cache[c] = CONV
    if root is not None:
        for c in cells:
            arena.lineage_root[c] = root        # share one kernel
    # root == None => leave lineage_root = -1 (own root => independent)


# ── CHECK 1: lone CONV == linear ────────────────────────────────────────

def check_reduction() -> bool:
    torch.manual_seed(0)
    a = Arena(Envelope(), capacity=64)
    a.alloc(8)                                   # 0..3 inputs, 4 the conv cell
    in_cells = [0, 1, 2, 3]
    cell = 4
    w = torch.randn(4)
    a.bias[cell] = 0.37
    _set_conv(a, [cell], root=None)              # own root
    _wire_patch(a, cell, in_cells, w)
    bucket = _bucket_for(a, [cell])

    x = torch.randn(5, 64)                        # act indexed by global id
    out = conv.forward_batch(x, bucket, a)        # [5, 1]
    ref = a.bias[cell] + (x[:, in_cells] * w).sum(1)
    ok = torch.allclose(out.squeeze(1), ref, atol=1e-6)
    print(f"CHECK 1 reduction (lone CONV == linear): "
          f"{'PASS' if ok else 'FAIL'}  max|Δ|={float((out.squeeze(1)-ref).abs().max()):.2e}")
    return ok


# ── CHECK 2: shared kernel + translation equivariance + gradient tying ──

def check_tying() -> bool:
    torch.manual_seed(0)
    a = Arena(Envelope(), capacity=64)
    a.alloc(16)
    # inputs 0..9 (a length-10 line); two disjoint patches of width 3
    patchA = [1, 2, 3]
    patchB = [6, 7, 8]
    twin0, twin1 = 10, 11                         # twin0 holds the kernel
    k = torch.tensor([0.5, -1.0, 0.5])
    _set_conv(a, [twin0, twin1], root=twin0)      # conjoined: share twin0
    _wire_patch(a, twin0, patchA, k)              # twin0 carries kernel
    _wire_patch(a, twin1, patchB, None)           # twin1 reads twin0's
    a.edge_weight.requires_grad_(True)
    bucket = _bucket_for(a, [twin0, twin1])

    # input where patchA of sample 0 == patchB of sample 1 (a translate)
    x = torch.zeros(2, 64)
    pat = torch.tensor([0.2, 0.9, -0.4])
    x[0, patchA] = pat
    x[1, patchB] = pat
    out = conv.forward_batch(x, bucket, a)        # [2, 2] (cols: twin0, twin1)

    equiv = torch.allclose(out[0, 0], out[1, 1], atol=1e-6)
    print(f"CHECK 2a translation equivariance "
          f"(twinA@pos0 == twinB@pos1): {'PASS' if equiv else 'FAIL'}  "
          f"{float(out[0,0]):.4f} vs {float(out[1,1]):.4f}")

    # gradient tying: a loss touching ONLY twin1's output must produce a
    # gradient on twin0's kernel edges (the shared weight).
    a.edge_weight.grad = None
    out2 = conv.forward_batch(x, bucket, a)
    out2[1, 1].backward()                         # only twin1, sample 1
    # twin0's kernel edges are the first 3 appended (patchA)
    g = a.edge_weight.grad[:3]
    tied = g is not None and float(g.abs().sum()) > 1e-8
    print(f"CHECK 2b gradient tying "
          f"(twin1's loss reaches twin0's kernel): "
          f"{'PASS' if tied else 'FAIL'}  |g_kernel|={float(g.abs().sum()):.4f}")
    a.edge_weight.requires_grad_(False)
    return equiv and tied


# ── CHECK 3: translation task — tied generalizes, untied doesn't ────────

def _make_task(n: int, positions: list[int], L: int, k: int,
               target: torch.Tensor, g: torch.Generator):
    """Class 1: `target` inserted at a random position drawn from
    `positions`. Class 0: pure background noise (no target anywhere) — an
    unambiguous "is the pattern present?" detection task. Returns x, y."""
    x = torch.randn(n, L, generator=g) * 0.3
    y = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        if i % 2 == 0:
            p = positions[int(torch.randint(len(positions), (1,), generator=g))]
            x[i, p:p + k] = target
            y[i] = 1
    return x, y


def _train_eval(tied: bool, seed: int) -> tuple[float, float]:
    g = torch.Generator().manual_seed(seed)
    L, k = 16, 3
    stride = 1
    pos_all = list(range(0, L - k + 1))           # 14 positions
    train_pos = pos_all[::2]                       # even positions
    test_pos = pos_all[1::2]                       # odd positions (DISJOINT)
    target = torch.tensor([1.0, -1.5, 1.0])

    a = Arena(Envelope(), capacity=256)
    n_cells = L + len(pos_all)                     # inputs + one twin per pos
    a.alloc(n_cells)
    in_cells = list(range(L))
    twins = list(range(L, L + len(pos_all)))
    root = twins[0] if tied else None
    _set_conv(a, twins, root=root)
    torch.manual_seed(seed)
    if tied:
        kern = torch.randn(k) * 0.1
        _wire_patch(a, twins[0], in_cells[0:k], kern)
        for t, p in zip(twins[1:], pos_all[1:]):
            _wire_patch(a, t, in_cells[p:p + k], None)
    else:
        for t, p in zip(twins, pos_all):
            _wire_patch(a, t, in_cells[p:p + k], torch.randn(k) * 0.1)
    bucket = _bucket_for(a, twins)

    # readout MUST be translation-invariant too (else a per-position head
    # discards the kernel's invariance — the consumer-must-pool lesson):
    # global max/mean pool over the cohort -> 2 features -> 2 logits.
    head = torch.nn.Linear(1, 2)                     # global-max pool -> 2 logits
    a.edge_weight.requires_grad_(True)
    opt = torch.optim.Adam(list(head.parameters()) + [a.edge_weight], lr=0.05)

    def logits_of(X):
        h = conv.forward_batch(X, bucket, a)        # [N, n_twins]
        feat = h.max(1).values.unsqueeze(1)         # global-max pool [N,1]
        return head(feat)

    xtr, ytr = _make_task(512, train_pos, L, k, target, g)
    Xtr = torch.zeros(len(xtr), a.capacity); Xtr[:, in_cells] = xtr
    for _ in range(800):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_of(Xtr), ytr)
        loss.backward()
        opt.step()

    def acc(positions):
        xe, ye = _make_task(512, positions, L, k, target, g)
        Xe = torch.zeros(len(xe), a.capacity); Xe[:, in_cells] = xe
        with torch.no_grad():
            pred = logits_of(Xe).argmax(1)
        return float((pred == ye).float().mean())

    tr, te = acc(train_pos), acc(test_pos)
    a.edge_weight.requires_grad_(False)
    return tr, te


def check_benefit() -> bool:
    seeds = [0, 1, 2]
    tied = torch.tensor([_train_eval(True, s) for s in seeds])
    untd = torch.tensor([_train_eval(False, s) for s in seeds])
    print(f"CHECK 3 translation task (train/test positions DISJOINT), "
          f"n={len(seeds)} seeds:")
    print(f"  tied   cohort  train={tied[:,0].mean():.3f}  "
          f"TEST(held-out pos)={tied[:,1].mean():.3f}")
    print(f"  untied cohort  train={untd[:,0].mean():.3f}  "
          f"TEST(held-out pos)={untd[:,1].mean():.3f}")
    ok = tied[:, 1].mean() > untd[:, 1].mean() + 0.10
    print(f"  => tied generalizes to unseen positions: "
          f"{'PASS' if ok else 'FAIL'}  "
          f"(Δtest={float(tied[:,1].mean()-untd[:,1].mean()):+.3f})")
    return ok


if __name__ == "__main__":
    r1 = check_reduction()
    print()
    r2 = check_tying()
    print()
    r3 = check_benefit()
    print()
    print(f"SUMMARY: reduction={r1}  tying={r2}  benefit={r3}  "
          f"=> {'ALL PASS' if (r1 and r2 and r3) else 'SEE ABOVE'}")
