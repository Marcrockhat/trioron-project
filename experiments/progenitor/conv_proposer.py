"""CONV proposer + coordinate field (s040) — design doc §8, fix #2.

Automates the conjoined-twin CONV spawn that test_conjoined_conv.py proved
by hand. Two jobs:

  WHERE  a coordinate field (f -> grid position) tiles a conjoined cohort
         of CONV cells (one shared kernel) across the input — `spawn_conv_cohort`.
  WHEN   a vote: train a CONV-cohort trial and a LINEAR (tabular) trial,
         spawn CONV only if its held-out (unseen-position) accuracy beats
         linear by a margin — `propose`.

The falsifiable claim is DATA-TYPE ADAPTIVITY: the SAME proposer must vote
CONV on real 2-D spatial data and REJECT it when locality is destroyed
(columns shuffled => the coord field no longer matches the data, the
cohort wires non-adjacent pixels, the conv advantage must vanish).

Run:  python3 experiments/progenitor/conv_proposer.py
"""
from __future__ import annotations

import torch

from trioron.core.arena import Arena
from trioron.core.envelope import Envelope
from trioron.core.scheduler import Bucket
from trioron.core.epigenome import CONV
from trioron.phenotype import conv


# ── coordinate field + patch tiling ─────────────────────────────────────

def tile_patches(H: int, W: int, k: int, stride: int = 1):
    """k×k patches over an H×W grid in COLUMN-INDEX space (row-major
    f = r*W + c). Returns (patches, centers): patches[i] is the list of
    column indices in the i-th patch (row-major => matched tap order);
    centers[i] is its (row, col). This list IS the coordinate field the
    proposer reads — for our senses it's a known exact lookup, not learned."""
    patches, centers = [], []
    for r in range(0, H - k + 1, stride):
        for c in range(0, W - k + 1, stride):
            patch = [(r + i) * W + (c + j) for i in range(k) for j in range(k)]
            patches.append(patch)
            centers.append((r + k // 2, c + k // 2))
    return patches, centers


def _bucket_for(arena: Arena, cells: list[int]) -> Bucket:
    cells_t = torch.tensor(cells, dtype=torch.int32)
    local_of = {c: i for i, c in enumerate(cells)}
    dst = arena.edge_dst[: arena.edge_cursor]
    keep = torch.tensor([int(d) in local_of for d in dst.tolist()])
    e_idx = keep.nonzero(as_tuple=False).squeeze(-1)
    e_dst = arena.edge_dst[e_idx].long()
    e_dst_local = torch.tensor([local_of[int(d)] for d in e_dst.tolist()])
    return Bucket(rank=1, phenotype=CONV, cell_ids=cells_t,
                  edge_indices=e_idx, edge_src=arena.edge_src[e_idx].long(),
                  edge_dst_local=e_dst_local, bias_ids=cells_t)


def spawn_conv_cohort(arena: Arena, in_cells: list[int], patches, k: int,
                      tied: bool, seed: int):
    """Spawn a conjoined cohort: one CONV cell per patch. tied => all share
    cohort[0]'s lineage_root (ONE kernel of k² weights, tiled). Returns
    (cohort_ids, bucket)."""
    n = len(patches)
    ids = arena.alloc(n).tolist()
    for c in ids:
        arena.epigenome[c] = 1 << CONV
        arena.phenotype_cache[c] = CONV
    if tied:
        for c in ids:
            arena.lineage_root[c] = ids[0]
    g = torch.Generator().manual_seed(seed)
    for i, (cell, patch) in enumerate(zip(ids, patches)):
        src = torch.tensor([in_cells[p] for p in patch], dtype=torch.int32)
        dst = torch.full((len(patch),), cell, dtype=torch.int32)
        if tied:
            w = torch.randn(k * k, generator=g) * 0.1 if i == 0 else None
        else:
            w = torch.randn(k * k, generator=g) * 0.1
        arena.add_edges(src, dst, w)
    return ids, _bucket_for(arena, ids)


# ── 2-D detection task (with optional locality-breaking shuffle) ─────────

def make_task(n, H, W, k, target, positions, perm, g):
    """Class 1: the k×k `target` ARRANGEMENT inserted at a random (r,c).
    Class 0: the SAME 9 values, locally SCRAMBLED, at a random (r,c) —
    identical histogram, only the spatial arrangement differs, so the task
    is unsolvable WITHOUT locality (rules out magnitude cues). `perm`
    applied LAST: None keeps grid locality, a random perm destroys it."""
    x = torch.randn(n, H * W, generator=g) * 0.3
    y = torch.zeros(n, dtype=torch.long)
    tflat = target.flatten()
    for i in range(n):
        r, c = positions[int(torch.randint(len(positions), (1,), generator=g))]
        cols = [(r + a) * W + (c + b) for a in range(k) for b in range(k)]
        if i % 2 == 0:
            x[i, cols] = tflat                       # correct arrangement
            y[i] = 1
        else:
            x[i, cols] = tflat[torch.randperm(k * k, generator=g)]  # scrambled
    if perm is not None:
        x = x[:, perm]
    return x, y


# ── trials ──────────────────────────────────────────────────────────────

def _fit(logits_of, params, xtr, ytr, steps=400, lr=0.05):
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_of(xtr), ytr)
        loss.backward()
        opt.step()


def conv_trial(H, W, k, target, train_pos, test_pos, perm, seed):
    g = torch.Generator().manual_seed(seed)
    a = Arena(Envelope(), capacity=H * W + 256)
    a.alloc(H * W)                                    # input cells 0..H*W-1
    in_cells = list(range(H * W))
    patches, _ = tile_patches(H, W, k)
    _, bucket = spawn_conv_cohort(a, in_cells, patches, k, tied=True, seed=seed)
    a.edge_weight.requires_grad_(True)
    head = torch.nn.Linear(1, 2)

    def logits_of(x):
        X = torch.zeros(len(x), a.capacity); X[:, in_cells] = x
        h = conv.forward_batch(X, bucket, a)            # [N, n_cohort]
        return head(h.max(1).values.unsqueeze(1))       # global-max pool

    xtr, ytr = make_task(512, H, W, k, target, train_pos, perm, g)
    _fit(logits_of, list(head.parameters()) + [a.edge_weight], xtr, ytr)
    xte, yte = make_task(512, H, W, k, target, test_pos, perm, g)
    with torch.no_grad():
        acc = float((logits_of(xte).argmax(1) == yte).float().mean())
    a.edge_weight.requires_grad_(False)
    return acc


def linear_trial(H, W, k, target, train_pos, test_pos, perm, seed):
    """The tabular learner: logistic regression over all columns."""
    g = torch.Generator().manual_seed(seed)
    head = torch.nn.Linear(H * W, 2)
    logits_of = lambda x: head(x)
    xtr, ytr = make_task(512, H, W, k, target, train_pos, perm, g)
    _fit(logits_of, list(head.parameters()), xtr, ytr)
    xte, yte = make_task(512, H, W, k, target, test_pos, perm, g)
    with torch.no_grad():
        acc = float((head(xte).argmax(1) == yte).float().mean())
    return acc


# ── proposer ────────────────────────────────────────────────────────────

def propose(H, W, k, target, perm, seeds, margin=0.10):
    pos, _ = tile_patches(H, W, k)
    all_pos = [(r, c) for r in range(H - k + 1) for c in range(W - k + 1)]
    train_pos = all_pos[::2]
    test_pos = all_pos[1::2]                              # DISJOINT
    cv = torch.tensor([conv_trial(H, W, k, target, train_pos, test_pos, perm, s)
                       for s in seeds])
    lv = torch.tensor([linear_trial(H, W, k, target, train_pos, test_pos, perm, s)
                       for s in seeds])
    adv = float(cv.mean() - lv.mean())
    # spawn only if CONV reaches real competence on UNSEEN positions
    # (well above chance) AND adds value over the tabular learner. A
    # chance-level cohort (locality destroyed) fails the competence bar
    # even when it "beats" a below-chance linear.
    competent = cv.mean().item() > 0.5 + margin
    vote = "SPAWN CONV" if (competent and adv > margin) else "reject (stay tabular)"
    return cv.mean().item(), lv.mean().item(), adv, vote


if __name__ == "__main__":
    H = W = 10
    k = 3
    torch.manual_seed(0)
    target = torch.tensor([[1.0, -1.5, 1.0],
                           [-1.5, 2.0, -1.5],
                           [1.0, -1.5, 1.0]])
    seeds = [0, 1, 2]

    print(f"2-D detection, {H}x{W} grid, {k}x{k} target, train/test "
          f"positions DISJOINT, n={len(seeds)} seeds.\n")

    cv, lv, adv, vote = propose(H, W, k, target, perm=None, seeds=seeds)
    print(f"DATA TYPE A — real image (grid locality intact):")
    print(f"  CONV-cohort test={cv:.3f}   LINEAR(tabular) test={lv:.3f}   "
          f"adv={adv:+.3f}  ->  VOTE: {vote}\n")

    g = torch.Generator().manual_seed(123)
    perm = torch.randperm(H * W, generator=g)             # destroy locality
    cv2, lv2, adv2, vote2 = propose(H, W, k, target, perm=perm, seeds=seeds)
    print(f"DATA TYPE B — same data, columns SHUFFLED (locality destroyed):")
    print(f"  CONV-cohort test={cv2:.3f}   LINEAR(tabular) test={lv2:.3f}   "
          f"adv={adv2:+.3f}  ->  VOTE: {vote2}\n")

    ok = (vote.startswith("SPAWN")) and (vote2.startswith("reject"))
    print(f"DATA-TYPE ADAPTIVITY: {'PASS' if ok else 'FAIL'} — proposer "
          f"{'spawns CONV on image, rejects on shuffled' if ok else 'did not flip'}")
