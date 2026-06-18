"""u (utility) + credit-locking — finishing the triparametric node (s043).

Prior files wired w (learnable kernels) and λ (epigenetic lock) into the
scattering organ. This wires the THIRD node variable — u (utility), the running
gradient magnitude that drives CREDIT-BASED LOCKING (active→dormant at a task
boundary) — and closes the council→soma loop Rocky chose:

  GLOBAL VOTE → ONE SOMA BRANCH (council decision, this session):
    germline sensors (fixed-Gabor scattering, NEVER locked) -> pooled descriptor
    -> the 6-phenotype council votes ONE phenotype for a single soma branch
       (here the applicable palette on a POOLED vector: LINEAR / TANH / DENDRITE;
       CONV/ATTENTION/RECURRENT have no spatial/temporal axis left post-pool and
       are inapplicable — the data would reject them, so we don't trial them)
    -> the winning-phenotype soma (a hidden layer of real arena cells, marked
       CREDIT_ELIGIBLE) is what credit-locking protects; per-task linear heads.

  THE NODE, COMPLETE: w (soma edge weights, trained) + λ (on sensors, prior file)
  + u (CreditTracker.update_utility -> consolidate locks settled soma cells).

Credit-locking is GATED to soma: germline sensors lack CREDIT_ELIGIBLE so
consolidate() skips them (spec §3.1 — germline never frozen). Forgetting of an
early task = soma drift; locking settled soma cells (grads zeroed while dormant)
defends it, the hard-anchor counterpart to λ's soft pull.

ARENA: split-MNIST, 5 binary tasks {0,1},{2,3},…,{8,9}; shared soma, per-task
heads → early-task forgetting is pure soma drift. Arms: locking OFF vs ON.

Run:  python3 experiments/progenitor/council_soma_credit.py
Env:  SEEDS H N_CH EPOCHS PER_CLASS THETA_E G_MIN LOCK_RATE
"""
from __future__ import annotations

import os
import time
import torch
import torch.nn.functional as F

from trioron.core.arena import Arena
from trioron.core.envelope import Envelope
from trioron.core.epigenome import CREDIT_ELIGIBLE, OUTPUT, LINEAR, TANH, DENDRITE
from trioron.core.state import CellState
from trioron.phenotype import conv
from trioron.learning.credit import CreditTracker, CreditConfig
from trioron.legacy.donorkit.datasets import DatasetBundle
from experiments.progenitor.conv_proposer import tile_patches, _bucket_for
from experiments.progenitor.mnist_conv_fixed import spawn_fixed_cohort, conv_map
from experiments.progenitor.mnist_scatter_deep import gabor, pool_to

GRID, K, STRIDE = 28, 9, 2
H1 = (GRID - K) // STRIDE + 1
N_CH = int(os.environ.get("N_CH", "8"))
ORIENTS, FREQS = [0, 45, 90, 135], [0.16, 0.28]
POOL = 5
DESC = N_CH * POOL * POOL
H = int(os.environ.get("H", "32"))             # soma hidden width
EPOCHS = int(os.environ.get("EPOCHS", "5"))
PER_CLASS = int(os.environ.get("PER_CLASS", "400"))
BATCH, LR = 128, 0.01
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

NONLIN = {LINEAR: lambda z: z, TANH: torch.tanh, DENDRITE: lambda z: z + z ** 2}
GNAME = {LINEAR: "LINEAR", TANH: "TANH", DENDRITE: "DENDRITE"}


# ── germline sensors: fixed-Gabor scattering -> pooled descriptor ──────────

def build_sensors():
    patches, _ = tile_patches(GRID, GRID, K, STRIDE)
    a = Arena(Envelope(), capacity=GRID * GRID + 2 * N_CH * len(patches) + 128)
    a.alloc(GRID * GRID); inc = list(range(GRID * GRID))
    import math
    bank = []
    for th, fr in [(t, f) for t in ORIENTS for f in FREQS][:N_CH]:
        bc = spawn_fixed_cohort(a, inc, patches, gabor(K, th, fr, 0.0))
        bs = spawn_fixed_cohort(a, inc, patches, gabor(K, th, fr, -math.pi / 2))
        bank.append((bc, bs))
    return a, inc, bank


@torch.no_grad()
def descriptor(a, inc, bank, X):
    feats = []
    for bc, bs in bank:
        mc = conv_map(a, inc, bc, X)
        ms = conv_map(a, inc, bs, X)
        feats.append(pool_to((mc ** 2 + ms ** 2 + 1e-6).sqrt(), H1, POOL))
    return torch.cat(feats, 1)


# ── council GLOBAL VOTE: pick one soma phenotype, data decides ─────────────

def council_vote(Dtr, ytr, Dva, yva, n_class, seed):
    """Trial one daughter per applicable phenotype; the phenotype whose own
    learning signal (val acc) is best wins (§3.4 trial-vote)."""
    votes = {}
    for gene, nl in NONLIN.items():
        torch.manual_seed(seed)
        W1 = torch.nn.Linear(DESC, H); W2 = torch.nn.Linear(H, n_class)
        opt = torch.optim.Adam(list(W1.parameters()) + list(W2.parameters()), lr=LR)
        for _ in range(EPOCHS):
            opt.zero_grad()
            loss = F.cross_entropy(W2(nl(W1(Dtr))), ytr)
            loss.backward(); opt.step()
        with torch.no_grad():
            votes[gene] = float((W2(nl(W1(Dva))).argmax(1) == yva).float().mean())
    winner = max(votes, key=votes.get)
    return winner, votes


# ── soma: winning-phenotype hidden layer of credit-eligible arena cells ────

def build_soma(gene, seed):
    """D input cells (descriptor holder) + H soma cells (own-root => linear via
    conv.forward_batch), marked CREDIT_ELIGIBLE|OUTPUT|<gene>. Dense desc->soma."""
    a = Arena(Envelope(), capacity=DESC + H + 16)
    a.alloc(DESC); inc = list(range(DESC))
    ids = a.alloc(H).tolist()
    for c in ids:
        a.epigenome[c] = (1 << gene) | (1 << CREDIT_ELIGIBLE) | (1 << OUTPUT)
        a.phenotype_cache[c] = gene
    g = torch.Generator().manual_seed(seed)
    for c in ids:
        src = torch.tensor(inc, dtype=torch.int32)
        dst = torch.full((DESC,), c, dtype=torch.int32)
        a.add_edges(src, dst, torch.randn(DESC, generator=g) * (1.0 / DESC ** 0.5))
    a.edge_weight.requires_grad_(True); a.bias.requires_grad_(True)
    return a, inc, ids, _bucket_for(a, ids)


def soma_act(a, inc, bucket, gene, desc):
    A = torch.zeros(len(desc), a.capacity)
    A[:, inc] = desc
    return NONLIN[gene](conv.forward_batch(A, bucket, a))      # [N, H]


def zero_dormant_grads(a, soma_ids):
    dorm = [c for c in soma_ids if int(a.state[c]) == CellState.DORMANT]
    if not dorm:
        return
    dset = set(dorm)
    if a.edge_weight.grad is not None:
        dst = a.edge_dst[:a.edge_cursor]
        mask = torch.tensor([int(d) in dset for d in dst.tolist()])
        a.edge_weight.grad[:a.edge_cursor][mask] = 0.0
    if a.bias.grad is not None:
        a.bias.grad[torch.tensor(dorm)] = 0.0


# ── one continual run over the 5 tasks ────────────────────────────────────

def run(seed, lock_on):
    b = DatasetBundle(["mnist"])
    Xtr, ytr = b.task_view("mnist", list(range(10)), list(range(10)), split="train").all_examples()
    Xte, yte = b.task_view("mnist", list(range(10)), list(range(10)), split="test").all_examples()
    sa, sinc, bank = build_sensors()

    def task_set(X, y, cls, per, sd):
        g = torch.Generator().manual_seed(sd)
        xs, ys = [], []
        for new, c in enumerate(cls):
            idx = torch.nonzero(y == c, as_tuple=False).squeeze(1)
            idx = idx[torch.randperm(len(idx), generator=g)[:per]]
            xs.append(descriptor(sa, sinc, bank, X[idx]))
            ys.append(torch.full((len(idx),), new))
        D = torch.cat(xs); Y = torch.cat(ys)
        p = torch.randperm(len(D), generator=g)
        return D[p], Y[p]

    Dtr = [task_set(Xtr, ytr, c, PER_CLASS, seed + i) for i, c in enumerate(TASKS)]
    Dte = [task_set(Xte, yte, c, PER_CLASS, seed + 100 + i) for i, c in enumerate(TASKS)]

    # council votes the soma phenotype on task 0
    nv = len(Dtr[0][0]) * 3 // 4
    winner, votes = council_vote(Dtr[0][0][:nv], Dtr[0][1][:nv],
                                 Dtr[0][0][nv:], Dtr[0][1][nv:], 2, seed)
    a, inc, soma_ids, bucket = build_soma(winner, seed)
    cfg = CreditConfig(consecutive_tasks=1,
                       theta_e=float(os.environ.get("THETA_E", "0.30")),
                       g_min=float(os.environ.get("G_MIN", "1e-3")),
                       lock_base_rate=float(os.environ.get("LOCK_RATE", "1.0")),
                       engagement_decay=0.3)
    credit = CreditTracker(a, cfg)
    heads = [torch.nn.Linear(H, 2) for _ in TASKS]
    locked_total = 0
    # mechanism check: snapshot a locked cell's incoming weights at lock time;
    # if dormant-grad-zeroing works, they must not move afterward.
    lock_snap: dict[int, torch.Tensor] = {}

    def incoming(cell):
        m = (a.edge_dst[:a.edge_cursor] == cell)
        return a.edge_weight[:a.edge_cursor][m].detach().clone()

    for t, (Dx, Dy) in enumerate(Dtr):
        head = heads[t]
        opt = torch.optim.Adam(list(head.parameters()) + [a.edge_weight, a.bias], lr=LR)
        gg = torch.Generator().manual_seed(seed + t)
        for _ in range(EPOCHS):
            perm = torch.randperm(len(Dx), generator=gg)
            for i in range(0, len(Dx), BATCH):
                bi = perm[i:i + BATCH]
                opt.zero_grad()
                act = soma_act(a, inc, bucket, winner, Dx[bi])
                loss = F.cross_entropy(head(act), Dy[bi])
                loss.backward()
                credit.update_utility()
                if lock_on:
                    zero_dormant_grads(a, soma_ids)
                opt.step()
                with torch.no_grad():
                    full = torch.zeros(len(bi), a.capacity)
                    full[:, soma_ids] = act.detach()
                    credit.update_engagement(full)
        if lock_on:
            newly = credit.consolidate()
            for c in newly:
                lock_snap[c] = incoming(c)        # freeze-point snapshot
            locked_total += len(newly)

    # mechanism check: post-lock drift of locked cells (should be ~0)
    lock_drift = 0.0
    if lock_snap:
        lock_drift = float(torch.stack([(incoming(c) - w0).abs().max()
                                        for c, w0 in lock_snap.items()]).max())

    # evaluate every task with its own (frozen) head + current soma
    accs = []
    with torch.no_grad():
        for t, (Dx, Dy) in enumerate(Dte):
            act = soma_act(a, inc, bucket, winner, Dx)
            accs.append(float((heads[t](act).argmax(1) == Dy).float().mean()))
    return winner, votes, accs, locked_total, lock_drift


def main():
    t0 = time.time()
    seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
    print("split-MNIST 5-task — council global-vote soma + credit-locking (u)")
    print(f"  germline fixed-Gabor sensors (N_CH={N_CH}, desc={DESC}) -> soma H={H} "
          f"| EPOCHS={EPOCHS} PER_CLASS={PER_CLASS} | n={len(seeds)} seeds\n")
    res = {True: [], False: []}
    win_log = []
    for s in seeds:
        for lock in (False, True):
            w, votes, accs, nlk, drift = run(s, lock)
            res[lock].append(accs)
            if lock:
                win_log.append((w, votes, nlk, drift))
    w0, v0, _, _ = win_log[0]
    print(f"  council vote (seed0): " +
          "  ".join(f"{GNAME[g]} {v0[g]:.3f}" for g in v0) +
          f"  -> WON: {GNAME[w0]}\n")
    print(f"  {'arm':<14} " + " ".join(f"T{i}" for i in range(len(TASKS))) +
          f"  {'mean':>6} {'locked':>7}")
    for lock, tag in [(False, "credit OFF"), (True, "credit ON")]:
        M = torch.tensor(res[lock]).mean(0)
        nlk = sum(x[2] for x in win_log) / len(win_log) if lock else 0
        print(f"  {tag:<14} " + " ".join(f"{m:.2f}" for m in M) +
              f"  {M.mean():>6.3f} {nlk:>7.1f}")
    off = torch.tensor(res[False]).mean(0).mean()
    on = torch.tensor(res[True]).mean(0).mean()
    max_drift = max(x[3] for x in win_log)
    print(f"\n  credit-locking Δ mean-acc = {float(on - off):+.3f} "
          f"(OFF {off:.3f} -> ON {on:.3f})")
    print(f"  MECHANISM: max post-lock weight drift of any locked cell = "
          f"{max_drift:.2e}  (≈0 ⇒ lock truly freezes); germline sensors live in a "
          f"separate arena with 0 CREDIT_ELIGIBLE cells ⇒ never lockable.")
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
