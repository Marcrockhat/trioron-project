"""Step 2 — input-layer genesis: the recycle-by-saliency loop.

The progenitor accepts an OVERSIZED, position-tagged aperture (the universal
intake; here a small stand-in for the 1.5 Mi ceiling). It chunks the aperture
into candidate sensor nodes (lowest resolution = one node per slot), each
position-tagged from the seeded field (the spatial S). The sensors feed a proxy
council (the readout). The council judges each sensor by TWO native signals:

  1. variance  — an empty chunk (no variance) carries no information → recycle.
  2. saliency u = |w·g| — a chunk with variance but no *learning* (it doesn't
     help the council reduce loss) → recycle. The third triparametric parameter.

Survivors are the chunks carrying learnable signal; they become the input layer.
For the 1-D weight data this converges to a SINGLE node (the weight), with the
noise + empty chunks recycled.

(The proxy council here is the linear readout; the standing 5×4 council is Step 3.
Amplify-into-branch and the progenitor disconnect are the tail of this thread.)

Run: python3 -m experiments.progenitor.step2_genesis
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from trioron.core import Envelope, construct
from trioron.core.epigenome import PERCEPTION, OUTPUT, set_gene
from trioron.phenotype import default_dispatch_table

from .data import make_data, bayes_accuracy
from .positions import sensor_positions

APERTURE_SEED = 0xC0FFEE      # the shared spatial-S seed (handshake basis)
N_NOISE = 4                   # noise chunks: variance present, NO learning
N_ZERO = 3                    # empty chunks: no variance
VAR_EPS = 1e-4                # variance gate
SAL_FRAC = 0.10               # saliency gate: keep sensors with u >= SAL_FRAC * max(u)


def build_aperture(data, n_noise, n_zero, seed):
    """Embed the 1-D weight in an oversized aperture: [weight | noise | zeros].
    Returns (X_aperture, slot_names)."""
    g = torch.Generator().manual_seed(seed)
    n = data.x.shape[0]
    cols = [data.x]                                            # slot 0: the real feature
    names = ["weight"]
    cols.append(torch.randn(n, n_noise, generator=g))         # noise slots
    names += [f"noise{i}" for i in range(n_noise)]
    cols.append(torch.zeros(n, n_zero))                       # empty slots
    names += [f"zero{i}" for i in range(n_zero)]
    return torch.cat(cols, dim=1), names


def build_candidates(n_sensors, n_out, pos):
    """Hand-wire: n_sensors perception cells (position-tagged) → n_out outputs."""
    def base(sub):
        a = sub.arena
        sens = a.alloc(n_sensors)
        outs = a.alloc(n_out)
        for k, s in enumerate(sens.tolist()):
            a.epigenome[s] = set_gene(int(a.epigenome[s].item()), PERCEPTION)
            a.rank[s] = 0
            a.position[s] = pos[k]
        for o in outs.tolist():
            a.epigenome[o] = set_gene(int(a.epigenome[o].item()), OUTPUT)
            a.rank[o] = 1
        a.refresh_all_phenotypes()
        a.add_edges(sens.repeat(n_out), outs.repeat_interleave(n_sensors))
    return construct(base=base, envelope=Envelope(),
                     dispatch_table=default_dispatch_table(), capacity=128)


def train(sub, X, y, steps=400, lr=0.05, seed=0):
    torch.manual_seed(seed)
    sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(sub(X), y).backward()
        opt.step()


def sensor_saliency(sub, X, y, sensor_ids):
    """Per-sensor u = Σ_out-edges |w·g| (the triparametric saliency)."""
    a = sub.arena
    sub.prepare_training()
    a.edge_weight.grad = None
    F.cross_entropy(sub(X), y).backward()
    g = a.edge_weight.grad
    sal = {}
    for s in sensor_ids:
        m = (a.edge_src[:a.edge_cursor] == s)
        wg = (a.edge_weight[:a.edge_cursor][m] * g[:a.edge_cursor][m]).abs().sum()
        sal[s] = float(wg.item())
    return sal


def main() -> None:
    train_d = make_data(seed=0, n_per_class=256)
    test_d = make_data(seed=1, n_per_class=512)
    Xtr, names = build_aperture(train_d, N_NOISE, N_ZERO, seed=1)
    n_sensors = Xtr.shape[1]
    n_out = len(train_d.names)

    pos = sensor_positions(n_sensors, seed=APERTURE_SEED, dim=1)
    sub = build_candidates(n_sensors, n_out, pos)
    sensor_ids = [c for c in sub.arena.alive_ids().tolist()
                  if int(sub.arena.epigenome[c].item()) >> PERCEPTION & 1]

    print(f"Step 2 — input-layer genesis  (oversized aperture = {n_sensors} chunks)\n")
    print("  candidate sensors (chunk → position x):")
    for k, s in enumerate(sensor_ids):
        print(f"    cell {s:>2}  {names[k]:>7}   pos.x = {float(pos[k,0]):.3f}")

    # variance gate (information present?)
    var = Xtr.var(0)
    # train the proxy council, then read saliency u=|w·g|
    train(sub, Xtr, train_d.y)
    sal = sensor_saliency(sub, Xtr, train_d.y, sensor_ids)
    sal_max = max(sal.values())

    print(f"\n  council judgment  (var_eps={VAR_EPS}, keep u ≥ {SAL_FRAC:.0%} of max u):")
    print(f"    {'chunk':>7} {'variance':>9} {'u=|w·g|':>9}  verdict")
    survivors = []
    for k, s in enumerate(sensor_ids):
        v = float(var[k]); u = sal[s]
        if v < VAR_EPS:
            verdict = "recycle (empty)"
        elif u < SAL_FRAC * sal_max:
            verdict = "recycle (no learning)"
        else:
            verdict = "KEEP"
            survivors.append(k)
        print(f"    {names[k]:>7} {v:>9.4f} {u:>9.4f}  {verdict}")

    print(f"\n  → input layer = {len(survivors)} surviving node(s): "
          f"{[names[k] for k in survivors]}")

    # rebuild the input layer from survivors only, confirm accuracy holds
    Xte, _ = build_aperture(test_d, N_NOISE, N_ZERO, seed=1)
    keep = torch.tensor(survivors)
    pos2 = sensor_positions(len(survivors), seed=APERTURE_SEED, dim=1)
    sub2 = build_candidates(len(survivors), n_out, pos2)
    train(sub2, Xtr[:, keep], train_d.y)
    with torch.no_grad():
        acc = float((sub2(Xte[:, keep]).argmax(1) == test_d.y).float().mean())
    print(f"\n  converged input layer accuracy: {acc:.3f}  "
          f"(Step-1 1-node baseline 0.807, Bayes {bayes_accuracy(test_d):.3f})")
    print("  → the oversized aperture collapsed to the single informative node, "
          "noise + empties recycled.")


if __name__ == "__main__":
    main()
