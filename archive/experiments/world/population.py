"""Mortal population + mirror learning — breaking the immortality plateau.

Rocky's diagnosis (2026-06-01): an *immortal solo* organism plateaus (~18)
because costless death removes the pressure to get good — "immortality grants
nothing besides laziness." Three learning algorithms (Q, world-model aux,
model-based planning) all stalled at ~15-18, confirming the bottleneck is the
PARADIGM, not the algorithm.

The fix, two parts:
  - MORTALITY + SELECTION: N organisms, each mortal; a dead one respawns as a
    *mutated clone of a survivor* (fitness = recent survival). Death has stakes;
    survivors propagate. Evolution restores the pressure immortality removed.
  - MIRROR (social) LEARNING: every organism trains on a SHARED pool of
    transitions — especially other organisms' DEATHS. You learn from others'
    mistakes without dying of them yourself. This is the function of mirror
    cells (cells that fire on observed action); the cell-type form is a refinement.

Arms test what each part buys:
  solo            N=1, own experience only, immortal respawn-retain (the plateau).
  pop-selection   N organisms, selection on death, NO mirror.
  shared-replay      N organisms, selection + shared (mirror) experience.
Metric: mean survival across the population over a long run. Does it beat ~18?
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trioron.core import Envelope, construct
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from experiments.world.tile_world import TileWorld, N_ACTION

PERCEPT_DIM = 77


def make_sub(seed):
    torch.manual_seed(seed)
    sub = construct(base=seeded(PERCEPT_DIM, N_ACTION, interior_cells=32),
                    envelope=Envelope(max_parameter_bytes=400_000),
                    dispatch_table=default_dispatch_table(), capacity=2048, sparsity_k=0)
    sub.compile(); sub.prepare_training()
    return sub


def clone_into(dst, src, mutate, g):
    """Reproduction: copy a survivor's weights into dst with gaussian mutation."""
    with torch.no_grad():
        for pd, ps in zip(dst.trainable_tensors(), src.trainable_tensors()):
            pd.copy_(ps + mutate * torch.randn(ps.shape, generator=g))


class Organism:
    def __init__(self, seed):
        self.sub = make_sub(seed)
        self.opt = torch.optim.Adam(self.sub.trainable_tensors(), lr=3e-3)
        self.buf = deque(maxlen=8000)
        self.world = None
        self.fitness = 15.0          # EMA of survival length
        self.born = 0

    def reset_world(self, wseed, max_steps):
        self.world = TileWorld(seed=wseed, max_steps=max_steps)
        self.p = self.world.percept()


def run(arm, *, pop, gen_steps, seed, mutate=0.05, gamma=0.95, batch=64,
        mirror_frac=0.5, max_steps=300):
    g = torch.Generator().manual_seed(seed + 91)
    N = 1 if arm == "solo" else pop
    orgs = [Organism(seed * 100 + i) for i in range(N)]
    wseed = [seed * 99000 + i for i in range(N)]
    for i, o in enumerate(orgs):
        o.reset_world(wseed[i], max_steps)
    mirror = deque(maxlen=20000)     # shared experience pool (esp. deaths)
    demo = deque(maxlen=5000)        # (state, action) from the current FITTEST — for imitation
    survivals = []
    eps = 0.3                        # fixed modest exploration

    for step in range(gen_steps):
        fittest = max(range(len(orgs)), key=lambda j: orgs[j].fitness)  # the demonstrator
        for i, o in enumerate(orgs):
            # act
            if torch.rand(1, generator=g).item() < eps:
                a = int(torch.randint(0, N_ACTION, (1,), generator=g))
            else:
                with torch.no_grad():
                    a = int(o.sub(o.p.unsqueeze(0))[0].argmax())
            p2, r, done, info = o.world.step(a)
            trans = (o.p, a, r, p2, float(done))
            o.buf.append(trans)
            if arm == "shared-replay":
                mirror.append(trans)         # share with the population
            if arm == "pop-apprentice" and i == fittest:
                demo.append((o.p, a))        # the fittest demonstrates (state, action)
            o.p = p2

            # learn (own experience + mirrored, if enabled)
            src = list(o.buf)
            if arm == "shared-replay" and len(mirror) >= batch:
                k = int(batch * mirror_frac)
                idx = torch.randint(0, len(mirror), (k,), generator=g)
                src = src + [mirror[j] for j in idx]
            if len(src) >= batch:
                bidx = torch.randint(0, len(src), (batch,), generator=g)
                bs = [src[j] for j in bidx]
                bp = torch.stack([b[0] for b in bs]); ba = torch.tensor([b[1] for b in bs])
                br = torch.tensor([b[2] for b in bs]); bp2 = torch.stack([b[3] for b in bs])
                bd = torch.tensor([b[4] for b in bs])
                q = o.sub(bp)[torch.arange(batch), ba]
                with torch.no_grad():
                    qn = o.sub(bp2).max(dim=1).values
                    tgt = br + gamma * qn * (1 - bd)
                loss = torch.nn.functional.mse_loss(q, tgt)
                # MIRROR/imitation: copy the fittest's actions (behavioral
                # cloning toward the demonstrator) — push Q-as-logits to the
                # demonstrated action on the fittest's states.
                if arm == "pop-apprentice" and i != fittest and len(demo) >= batch:
                    didx = torch.randint(0, len(demo), (batch,), generator=g)
                    ds = torch.stack([demo[j][0] for j in didx])
                    da = torch.tensor([demo[j][1] for j in didx])
                    logits = o.sub(ds)
                    loss = loss + 0.5 * torch.nn.functional.cross_entropy(logits, da)
                o.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(o.sub.trainable_tensors(), 1.0)
                o.sub.zero_dormant_grads(); o.opt.step()

            # death → record + respawn
            if done:
                life = info["t"]
                o.fitness = 0.9 * o.fitness + 0.1 * life
                if step > gen_steps // 3:        # record after warmup
                    survivals.append(life)
                if arm != "solo":
                    # selection: clone a fitness-weighted survivor (incl. self)
                    fits = torch.tensor([max(1.0, x.fitness) for x in orgs])
                    parent = orgs[int(torch.multinomial(fits, 1, generator=g))]
                    if parent is not o:
                        clone_into(o.sub, parent.sub, mutate, g)
                        o.opt = torch.optim.Adam(o.sub.trainable_tensors(), lr=3e-3)
                        o.fitness = parent.fitness
                wseed[i] += 100000
                o.reset_world(wseed[i], max_steps)
    return survivals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.steps, args.pop = 1, 6000, 6

    print(f"population: mortal pop + mirror learning. pop={args.pop} "
          f"steps={args.steps} seeds={args.seeds}")
    print("discriminating world: random≈28, reactive≈38, solo-learner≈50. Does pop/mirror/imitate beat solo?")
    for arm in ("solo", "pop-selection", "shared-replay", "pop-apprentice"):
        finals = []
        for seed in range(args.seeds):
            S = run(arm, pop=args.pop, gen_steps=args.steps, seed=seed)
            if S:
                finals.append(statistics.mean(S[-50:]))   # late-population survival
        m = statistics.mean(finals); s = statistics.stdev(finals) if len(finals) > 1 else 0.0
        print(f"  {arm:>14s}: late survival = {m:.1f} ± {s:.1f}  (n_deaths sampled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
