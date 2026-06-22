"""Stage 2+3 of the MIXED cross-domain stream (s044): the fully-upgraded
trioron node on an INTERLEAVED taxonomy+MNIST+CIFAR continual stream.

Perception (Stage 1, `mixed_stream_lenses`): three pure-trioron lenses ->
block-structured 52-class descriptors (tax[22] | mnist[456] | cifar[1368]).
This file adds the LIVING soma and runs continual learning over it.

THE NODE, COMPLETE — all three variables live on the SAME soma cells:
  w  = soma edge weights (trained every task)
  λ  = epigenetic lock (soft EWC anchor; accumulate_saliency -> refresh_lambda
       -> anchor at each boundary; strength·ewc_penalty added to the next loss)
  u  = utility -> CREDIT-LOCKING (hard freeze of settled high-engagement cells)
Germline sensors are fixed-Gabor / phase-code lenses, NEVER locked (spec §3.1).

STREAM: interleaved chained tasks across the three domains. Each domain is split
into N_CHUNK class-chunks; the chunks are interleaved round-robin so the organism
ping-pongs across modalities (tax_a, mnist_a, cifar_a, tax_b, ...). ONE shared
52-way head — FULL-SOFTMAX over the whole union, no task id at test (Rocky's
chosen metric). Forgetting therefore includes shared-head drift; the head is held
identical across arms, so arm deltas isolate what λ / credit do on the soma.

ARMS (soma protection): none | λ (soft EWC) | credit (hard lock) | both |
  replay (manifold pseudo-rehearsal over the fixed-lens block descriptors) |
  all (λ + credit + replay). Replay defends BOTH soma weights and past-class
  head logits by re-presenting old-class descriptors during each new task.
METRICS: final per-domain full-softmax acc; mean acc; mean forgetting
  (acc on a task right after training it − acc at stream end).

Run:  python3 experiments/progenitor/mixed_stream_cl.py
Env:  SEEDS H EPOCHS N_CHUNK STRENGTH LOCK_RATE THETA_E G_MIN ARMS EVAL_PER
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
from trioron.learning import epigenetic_lock as epi
from trioron.learning.manifold import ManifoldArchive, ManifoldConfig
from experiments.progenitor.conv_proposer import _bucket_for
from experiments.progenitor import mixed_stream_lenses as L

H = int(os.environ.get("H", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "6"))
N_CHUNK = int(os.environ.get("N_CHUNK", "2"))      # class-chunks per domain
STRENGTH = float(os.environ.get("STRENGTH", "1e3"))
REPLAY_BS = int(os.environ.get("REPLAY_BS", "32"))     # pseudo-samples per past class per step
REPLAY_STEPS = int(os.environ.get("REPLAY_STEPS", "1"))
REPLAY_W = float(os.environ.get("REPLAY_W", "1.0"))    # weight on the replay CE term
BATCH, LR = 128, 0.01

NONLIN = {LINEAR: lambda z: z, TANH: torch.tanh, DENDRITE: lambda z: z + z ** 2}
GNAME = {LINEAR: "LINEAR", TANH: "TANH", DENDRITE: "DENDRITE"}
DOM_OF = {"taxonomy": (L.TAX_OFF, L.TAX_K), "mnist": (L.MNIST_OFF, L.MNIST_K),
          "cifar": (L.CIFAR_OFF, L.CIFAR_K)}


# ── block matrices: every example as a block vector + GLOBAL label ─────────
def block_matrices(per, split):
    Xs, Ys = [], []
    for name in per:
        X, y = L.to_block(per, name, split)
        Xs.append(X); Ys.append(y)
    return torch.cat(Xs), torch.cat(Ys)


def interleaved_tasks():
    """Round-robin class-chunks across domains -> ordered list of global-class
    lists. e.g. N_CHUNK=2: [tax0..15],[mnist32..36],[cifar42..46],[tax16..31],..."""
    chunks = {}
    for name, (off, k) in DOM_OF.items():
        step = -(-k // N_CHUNK)                       # ceil
        chunks[name] = [list(range(off + i, off + min(i + step, k)))
                        for i in range(0, k, step)]
    tasks = []
    for i in range(N_CHUNK):
        for name in ("taxonomy", "mnist", "cifar"):
            if i < len(chunks[name]):
                tasks.append((name, chunks[name][i]))
    return tasks


# ── organism: soma hidden layer + IN-SUBSTRATE 52-way head, both credit-eligible
# (own-root => linear via forward_batch). λ (ewc over all edges) + credit-locking
# therefore protect the HEAD rows of learned classes too, not just the soma.
def build_net(block_dim, gene, seed):
    a = Arena(Envelope(), capacity=block_dim + H + L.N_CLASS + 16)
    a.alloc(block_dim); inc = list(range(block_dim))
    soma_ids = a.alloc(H).tolist()
    head_ids = a.alloc(L.N_CLASS).tolist()
    g = torch.Generator().manual_seed(seed)
    for c in soma_ids:                                         # input -> soma (winning phenotype)
        a.epigenome[c] = (1 << gene) | (1 << CREDIT_ELIGIBLE)
        a.phenotype_cache[c] = gene
        src = torch.tensor(inc, dtype=torch.int32)
        dst = torch.full((block_dim,), c, dtype=torch.int32)
        a.add_edges(src, dst, torch.randn(block_dim, generator=g) * (1.0 / block_dim ** 0.5))
    for c in head_ids:                                         # soma -> head (linear readout)
        a.epigenome[c] = (1 << LINEAR) | (1 << CREDIT_ELIGIBLE) | (1 << OUTPUT)
        a.phenotype_cache[c] = LINEAR
        src = torch.tensor(soma_ids, dtype=torch.int32)
        dst = torch.full((H,), c, dtype=torch.int32)
        a.add_edges(src, dst, torch.randn(H, generator=g) * (1.0 / H ** 0.5))
    a.edge_weight.requires_grad_(True); a.bias.requires_grad_(True)
    return a, inc, soma_ids, head_ids, _bucket_for(a, soma_ids), _bucket_for(a, head_ids)


def forward(a, inc, soma_ids, gene, soma_bucket, head_bucket, Xblock):
    """Block input -> soma (phenotype nonlin) -> in-substrate linear head.
    Returns (logits [N, N_CLASS], soma_act [N, H]) — soma_act for engagement."""
    A = torch.zeros(len(Xblock), a.capacity)
    A[:, inc] = Xblock
    sact = NONLIN[gene](conv.forward_batch(A, soma_bucket, a))   # [N, H]
    A2 = torch.zeros(len(Xblock), a.capacity)
    A2[:, soma_ids] = sact
    logits = conv.forward_batch(A2, head_bucket, a)             # [N, N_CLASS]
    return logits, sact


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


# ── council vote: pick one soma phenotype on the first task ────────────────
def council_vote(block_dim, Dtr, ytr, Dva, yva, n_class, seed):
    votes = {}
    for gene, nl in NONLIN.items():
        torch.manual_seed(seed)
        W1 = torch.nn.Linear(block_dim, H); W2 = torch.nn.Linear(H, n_class)
        opt = torch.optim.Adam(list(W1.parameters()) + list(W2.parameters()), lr=LR)
        for _ in range(EPOCHS):
            opt.zero_grad()
            F.cross_entropy(W2(nl(W1(Dtr))), ytr).backward(); opt.step()
        with torch.no_grad():
            votes[gene] = float((W2(nl(W1(Dva))).argmax(1) == yva).float().mean())
    return max(votes, key=votes.get), votes


# ── one continual run over the interleaved stream ─────────────────────────
def run(per, block_dim, tasks, seed, arm):
    use_lam = arm in ("lambda", "both", "all")
    use_cred = arm in ("credit", "both", "all")
    use_replay = arm in ("replay", "all")
    Xtr, ytr = block_matrices(per, "train")
    Xte, yte = block_matrices(per, "test")
    eval_per = int(os.environ.get("EVAL_PER", "120"))

    # council votes the soma phenotype on the FIRST task (local labels)
    name0, cls0 = tasks[0]
    m0 = torch.isin(ytr, torch.tensor(cls0))
    Xv, yv = Xtr[m0], ytr[m0] - cls0[0]
    nv = len(Xv) * 3 // 4
    gene, votes = council_vote(block_dim, Xv[:nv], yv[:nv], Xv[nv:], yv[nv:], len(cls0), seed)

    a, inc, soma_ids, head_ids, sbk, hbk = build_net(block_dim, gene, seed)
    elig = soma_ids + head_ids                       # both soma and head are credit-eligible
    cfg = CreditConfig(consecutive_tasks=1,
                       theta_e=float(os.environ.get("THETA_E", "0.30")),
                       g_min=float(os.environ.get("G_MIN", "1e-3")),
                       lock_base_rate=float(os.environ.get("LOCK_RATE", "1.0")),
                       engagement_decay=0.3)
    credit = CreditTracker(a, cfg)
    locked_total = 0

    # ── manifold replay: per-class (μ,σ) over the FIXED-LENS block descriptors.
    # The lenses are germline (never trained), so block descriptors are a
    # permanently-valid generative model; replaying them forward through the
    # current soma+head defends BOTH the soma weights and past-class head logits
    # against the new task's full-CE suppression. Lives in its OWN arena so it
    # never collides with the net's bucket dispatch / capacity.
    archive = None
    if use_replay:
        arch_arena = Arena(Envelope(), capacity=L.N_CLASS + 4)
        archive = ManifoldArchive(arch_arena,
                                  ManifoldConfig(replay_steps_per_class=REPLAY_STEPS),
                                  full_cov=False)

    def replay_ce():
        """CE on pseudo-descriptors for all archived (past, dormant) classes."""
        rxs, rys = [], []
        for samples, cid in archive.replay_batches(REPLAY_BS):
            rxs.append(samples); rys.append(torch.full((len(samples),), cid, dtype=torch.long))
        if not rxs:
            return None
        rX, rY = torch.cat(rxs), torch.cat(rys)
        rlog, _ = forward(a, inc, soma_ids, gene, sbk, hbk, rX)
        return F.cross_entropy(rlog, rY)

    @torch.no_grad()
    def acc_on(classes):
        m = torch.isin(yte, torch.tensor(classes))
        Xc, yc = Xte[m][:eval_per * len(classes)], yte[m][:eval_per * len(classes)]
        logits, _ = forward(a, inc, soma_ids, gene, sbk, hbk, Xc)
        return float((logits.argmax(1) == yc).float().mean())

    acc_after = []                                   # acc_after[t] = acc on task t right after training it
    for t, (name, cls) in enumerate(tasks):
        m = torch.isin(ytr, torch.tensor(cls))
        Dx, Dy = Xtr[m], ytr[m]
        opt = torch.optim.Adam([a.edge_weight, a.bias], lr=LR)
        gg = torch.Generator().manual_seed(seed + t)
        for _ in range(EPOCHS):
            perm = torch.randperm(len(Dx), generator=gg)
            for i in range(0, len(Dx), BATCH):
                bi = perm[i:i + BATCH]
                opt.zero_grad()
                logits, sact = forward(a, inc, soma_ids, gene, sbk, hbk, Dx[bi])
                loss = F.cross_entropy(logits, Dy[bi])
                if use_lam:
                    loss = loss + STRENGTH * epi.ewc_penalty(a)
                if use_replay:
                    rce = replay_ce()
                    if rce is not None:
                        loss = loss + REPLAY_W * rce
                loss.backward()
                credit.update_utility()
                if use_lam:
                    epi.accumulate_saliency(a)
                if use_cred:
                    zero_dormant_grads(a, elig)
                opt.step()
                with torch.no_grad():
                    full = torch.zeros(len(bi), a.capacity)
                    full[:, soma_ids] = sact.detach()
                    full[:, head_ids] = logits.detach()
                    credit.update_engagement(full)
        acc_after.append(acc_on(cls))
        if use_lam:
            epi.refresh_lambda(a); epi.anchor(a)
        if use_cred:
            locked_total += len(credit.consolidate())
        if use_replay:                                 # sketch this task's classes, then freeze them
            for c in cls:
                cm = (Dy == c)
                if cm.any():
                    archive.update_class(c, Dx[cm])
            archive.finalize_all()                     # active -> dormant => replayable next task

    # final per-domain full-softmax accuracy + per-task forgetting
    final_dom = {name: acc_on(list(range(off, off + k))) for name, (off, k) in DOM_OF.items()}
    forget = []
    for t, (name, cls) in enumerate(tasks):
        forget.append(acc_after[t] - acc_on(cls))
    store_kb = (archive.storage_bytes() / 1024.0) if use_replay else 0.0
    return dict(gene=gene, votes=votes, final_dom=final_dom,
                mean=sum(final_dom.values()) / 3, forget=sum(forget) / len(forget),
                acq=sum(acc_after) / len(acc_after), locked=locked_total, store_kb=store_kb)


def main():
    t0 = time.time()
    seeds = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]
    arms = os.environ.get("ARMS", "none,lambda,both,replay,all").split(",")
    tasks = interleaved_tasks()

    print(f"MIXED cross-domain CL — fully-upgraded node (w,λ,u) on {L.N_CLASS}-class union")
    print(f"  stream ({len(tasks)} interleaved tasks): " +
          " -> ".join(f"{n}[{c[0]}..{c[-1]}]" for n, c in tasks))
    print(f"  soma H={H} EPOCHS={EPOCHS} STRENGTH={STRENGTH} LOCK_RATE="
          f"{os.environ.get('LOCK_RATE','1.0')} | n={len(seeds)} seeds\n")

    per_seed = {arm: [] for arm in arms}
    gene_log = None
    for s in seeds:
        per, block_dim, _ = L.build_descriptors(s)
        for arm in arms:
            r = run(per, block_dim, tasks, s, arm)
            per_seed[arm].append(r)
            if gene_log is None:
                gene_log = r["votes"], r["gene"]
    v, g = gene_log
    print("  council vote (seed0): " + "  ".join(f"{GNAME[k]} {v[k]:.3f}" for k in v) +
          f"  -> WON: {GNAME[g]}\n")

    def ms(vals):
        t = torch.tensor(vals)
        return (f"{t.mean():.3f}" + (f"±{t.std(0):.3f}" if len(vals) > 1 else ""))

    print(f"  {'arm':<10} {'taxonomy':>11} {'mnist':>11} {'cifar':>11} "
          f"{'mean':>11} {'acq':>6} {'forget':>8} {'locked':>7} {'store':>8}")
    for arm in arms:
        R = per_seed[arm]
        td = ms([r["final_dom"]["taxonomy"] for r in R])
        md = ms([r["final_dom"]["mnist"] for r in R])
        cd = ms([r["final_dom"]["cifar"] for r in R])
        mn = ms([r["mean"] for r in R])
        aq = sum(r["acq"] for r in R) / len(R)
        fg = sum(r["forget"] for r in R) / len(R)
        lk = sum(r["locked"] for r in R) / len(R)
        sk = sum(r["store_kb"] for r in R) / len(R)
        st = f"{sk:.0f}KB" if sk else "-"
        print(f"  {arm:<10} {td:>11} {md:>11} {cd:>11} {mn:>11} {aq:>6.3f} {fg:>+8.3f} {lk:>7.1f} {st:>8}")
    print(f"\n[{time.time() - t0:.0f}s]  (acq = mean acc right after each task; "
          f"forget = mean drop to stream end; want high acq, low forget)")


if __name__ == "__main__":
    main()
