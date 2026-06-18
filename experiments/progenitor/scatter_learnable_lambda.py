"""LEARNABLE scattering kernels + the epigenetic lock λ (s043, the trioron bridge).

The s042/s043 scattering front-end was FIXED Gabor (gradient-free) — it used the
trioron substrate (CONV gene + lineage_root weight-tie) as a wiring fabric but
left the triparametric node's defining variables dormant. This wires two of the
three back in:

  w  — the conv kernels are now LEARNABLE (Adam flows into the cohort-root edges
       through conv.forward_batch's differentiable gather; the lineage tie ties
       the gradient across all positions, so we learn ONE shared kernel/channel).
  λ  — the epigenetic lock (trioron/learning/epigenetic_lock.py): at the task
       boundary we anchor the learned kernels and accumulate |w·g| saliency into
       node_lambda, then defend them with strength·ewc_penalty while the next
       task trains. λ is PER-CELL, row-summed over each conv cell's kernel edges.

ARENA — split-MNIST: task A = digits {0..4}, task B = {5..9}, 5-way each.
The conv kernels are SHARED across tasks (the perception organ); the readout is a
per-task linear head. After training B, the only thing that can have degraded the
features for A is KERNEL DRIFT — so forgetting on A is a clean readout of whether
λ held the kernels. We compare λ OFF (strength=0) vs λ ON from the SAME post-A
state.

Run:  python3 experiments/progenitor/scatter_learnable_lambda.py
Env:  SEEDS  STRENGTH  EPOCHS  PER_CLASS
"""
from __future__ import annotations

import os
import time
import torch
import torch.nn.functional as F

from trioron.core.arena import Arena
from trioron.core.envelope import Envelope
from trioron.phenotype import conv
from trioron.legacy.donorkit.datasets import DatasetBundle
from trioron.learning import epigenetic_lock as epi
from experiments.progenitor.conv_proposer import tile_patches
from experiments.progenitor.mnist_conv_fixed import spawn_fixed_cohort, conv_map
from experiments.progenitor.mnist_scatter_deep import gabor, pool_to

GRID, K, STRIDE = 28, 9, 2
H1 = (GRID - K) // STRIDE + 1                 # 10
# Defaults = the λ-demonstration regime: capacity is STARVED (N_CH=2) so the two
# shared kernels are load-bearing, and task B trains aggressively (long, high-LR)
# so it WOULD repurpose them — making λ's protection visible. With the roomy
# N_CH=8 front-end the scattering features are too general to forget (drift is
# harmless), which is itself the honest finding (forgetting lives in the readout,
# not a redundant perception organ — manual §7).
N_CH = int(os.environ.get("N_CH", "2"))        # oriented-energy channels (capacity)
ORIENTS, FREQS = [0, 45, 90, 135], [0.16, 0.28]
POOL = 5
DESC = N_CH * POOL * POOL
EPOCHS = int(os.environ.get("EPOCHS", "6"))
EPOCHS_B = int(os.environ.get("EPOCHS_B", "20"))   # B-phase length (drift pressure)
PER_CLASS = int(os.environ.get("PER_CLASS", "300"))
STRENGTH = float(os.environ.get("STRENGTH", "1e3"))
BATCH, LR = 256, float(os.environ.get("LR", "0.02"))
TASK_A, TASK_B = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]


def task_data(Xall, yall, classes, per_class, seed):
    """Balanced subset over `classes`, labels remapped to 0..len(classes)-1."""
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for new, c in enumerate(classes):
        idx = torch.nonzero(yall == c, as_tuple=False).squeeze(1)
        idx = idx[torch.randperm(len(idx), generator=g)[:per_class]]
        xs.append(Xall[idx]); ys.append(torch.full((len(idx),), new))
    X = torch.cat(xs); y = torch.cat(ys)
    p = torch.randperm(len(X), generator=g)
    return X[p], y[p]


def build_kernels(seed):
    """L1 conv bank: N_CH learnable (cos,sin) cohort pairs over the input, init
    from Gabor (warm start). Returns (arena, in_cells, bank)."""
    patches, _ = tile_patches(GRID, GRID, K, STRIDE)
    a = Arena(Envelope(), capacity=GRID * GRID + 2 * N_CH * len(patches) + 128)
    a.alloc(GRID * GRID); inc = list(range(GRID * GRID))
    pairs = [(th, fr) for th in ORIENTS for fr in FREQS][:N_CH]
    bank = []
    for th, fr in pairs:
        bc = spawn_fixed_cohort(a, inc, patches, gabor(K, th, fr, 0.0))
        bs = spawn_fixed_cohort(a, inc, patches, gabor(K, th, fr, -3.14159 / 2))
        bank.append((bc, bs))
    a.edge_weight.requires_grad_(True)
    return a, inc, bank


def descriptor(a, inc, bank, X):
    feats = []
    for bc, bs in bank:
        mc = conv_map(a, inc, bc, X)
        ms = conv_map(a, inc, bs, X)
        # modulus + pool; eps inside sqrt keeps the gradient finite at zero energy
        # (sqrt'(0)=inf would otherwise blow the learnable kernels to NaN)
        feats.append(pool_to((mc ** 2 + ms ** 2 + 1e-6).sqrt(), H1, POOL))
    return torch.cat(feats, 1)


def train_task(a, inc, bank, head, X, y, params, strength, seed, epochs=EPOCHS):
    opt = torch.optim.Adam(params, lr=LR)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), BATCH):
            bi = perm[i:i + BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(head(descriptor(a, inc, bank, X[bi])), y[bi])
            if strength > 0:
                loss = loss + strength * epi.ewc_penalty(a)
            loss.backward()
            opt.step()


def lock_in(a, inc, bank, head, X, y, seed):
    """Boundary: anchor learned kernels + accumulate |w·g| saliency -> λ."""
    epi.anchor(a)
    a.edge_fisher.zero_()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    for i in range(0, len(X), BATCH):
        bi = perm[i:i + BATCH]
        a.edge_weight.grad = None
        loss = F.cross_entropy(head(descriptor(a, inc, bank, X[bi])), y[bi])
        loss.backward()
        epi.accumulate_saliency(a)
    epi.refresh_lambda(a)


@torch.no_grad()
def acc(a, inc, bank, head, X, y):
    return float((head(descriptor(a, inc, bank, X)).argmax(1) == y).float().mean())


def run(seed):
    b = DatasetBundle(["mnist"])
    Xtr, ytr = b.task_view("mnist", list(range(10)), list(range(10)), split="train").all_examples()
    Xte, yte = b.task_view("mnist", list(range(10)), list(range(10)), split="test").all_examples()
    Atr = task_data(Xtr, ytr, TASK_A, PER_CLASS, seed)
    Ate = task_data(Xte, yte, TASK_A, PER_CLASS, seed + 7)
    Btr = task_data(Xtr, ytr, TASK_B, PER_CLASS, seed + 1)
    Bte = task_data(Xte, yte, TASK_B, PER_CLASS, seed + 9)

    # ── phase 1: learn kernels + head_A on task A ──
    a, inc, bank = build_kernels(seed)
    headA = torch.nn.Linear(DESC, 5)
    train_task(a, inc, bank, headA, *Atr, list(headA.parameters()) + [a.edge_weight],
               strength=0.0, seed=seed)
    accA0 = acc(a, inc, bank, headA, *Ate)
    lock_in(a, inc, bank, headA, *Atr, seed=seed)              # anchor + λ
    W_postA = a.edge_weight.detach().clone()                    # restore point
    lam = a.node_lambda.clone()

    results = {}
    for tag, strength in [("λ OFF", 0.0), ("λ ON", STRENGTH)]:
        with torch.no_grad():
            a.edge_weight.copy_(W_postA)                        # same start each arm
        a.node_lambda.copy_(lam)
        a.edge_weight.requires_grad_(True)
        headB = torch.nn.Linear(DESC, 5)
        train_task(a, inc, bank, headB, *Btr,
                   list(headB.parameters()) + [a.edge_weight], strength, seed=seed + 2,
                   epochs=EPOCHS_B)
        accA1 = acc(a, inc, bank, headA, *Ate)                  # A via frozen head_A
        accB = acc(a, inc, bank, headB, *Bte)                   # B learned this arm
        drift = float((a.edge_weight.detach() - W_postA).pow(2).sum().sqrt())
        results[tag] = (accA0, accA1, accA0 - accA1, accB, drift)
    return results


def main():
    t0 = time.time()
    seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
    print(f"split-MNIST learnable-kernel scattering + epigenetic lock λ")
    print(f"  task A {TASK_A} -> task B {TASK_B}, shared learnable kernels, "
          f"per-task heads")
    print(f"  N_CH={N_CH} K={K} desc={DESC} | EPOCHS={EPOCHS} PER_CLASS={PER_CLASS} "
          f"STRENGTH={STRENGTH:g} | n={len(seeds)} seeds\n")
    agg = {"λ OFF": [], "λ ON": []}
    for s in seeds:
        for tag, r in run(s).items():
            agg[tag].append(r)
    print(f"  {'arm':<8} {'accA(pre-B)':>12} {'accA(post-B)':>13} "
          f"{'forgetting':>12} {'accB':>8} {'kernel drift':>13}")
    for tag in ("λ OFF", "λ ON"):
        T = torch.tensor(agg[tag])
        print(f"  {tag:<8} {T[:,0].mean():>12.3f} {T[:,1].mean():>13.3f} "
              f"{T[:,2].mean():>12.3f} {T[:,3].mean():>8.3f} {T[:,4].mean():>13.3f}")
    off = torch.tensor(agg["λ OFF"])[:, 2]
    on = torch.tensor(agg["λ ON"])[:, 2]
    print(f"\n  λ reduces forgetting by {float(off.mean() - on.mean()):+.3f} "
          f"(OFF {off.mean():.3f} -> ON {on.mean():.3f}); "
          f"task B retained accB ON {torch.tensor(agg['λ ON'])[:,3].mean():.3f} "
          f"vs OFF {torch.tensor(agg['λ OFF'])[:,3].mean():.3f}")
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
