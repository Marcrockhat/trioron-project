"""Grounded sense→valence world — does the substrate learn predictive valence
that *generalizes* to novel percepts?  (sense→logic rung, no LLM)

Design conversation 2026-05-31 (Rocky + Chloe). The curriculum is
sense → logic/association → symbol → language, built bottom-up ("logic before
language" — discover fire before telling stories around it). This bench is the
first rung above perception: ground percepts in *innate drives*, let *emotion*
(= learned anticipation of a drive-consequence) emerge, and test whether that
learned valence COMPOSES to percepts never seen.

State layers (per the design):
  - SENSORY  (input)  — composable percept primitives (vision/taste/touch/…).
  - DRIVES   (innate) — the ONLY hardwired valence; resolves a percept to a
                        grounded consequence (nourishing/hydrating/painful/…).
  - EMOTION  (learned, emergent) — the substrate's predicted consequence from
                        the percept alone. NOT an input. "Fire looks scary"
                        because it learned fire→pain. Biased experience →
                        biased emotion (the racism point): same mechanism.

The headline is the SYSTEMATIC split (compositional generalization), not i.i.d.:
train where a class is cued one way, test where it's cued by a held-out
synonym cue. Did it learn the abstract cue, or memorize the surface feature?

Arms: no-growth / bipartite / self-arrange  (does self-organized depth help
the systematic generalization?). Reuses the Arena substrate + divide() growth.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trioron.lifecycle import divide, GrowthConfig
from experiments.bench_arena_hierarchical import (
    build_substrate, interior_parents, rank_stats, eval_acc,
)


# ----------------------------------------------------------------------
# Sensory schema — index of each percept primitive in the input vector
# ----------------------------------------------------------------------
SENSES = [
    # vision
    "bright", "red", "green", "blue", "big", "moving",
    # touch / texture
    "hard", "rough", "hot", "wet",
    # taste (taste buds)
    "sweet", "sour", "salty", "bitter", "umami",
    # smell
    "foul",
]
SIDX = {s: i for i, s in enumerate(SENSES)}
N_SENSE = len(SENSES)

# Innate consequence classes (the grounded valence the drives assign).
CONSEQUENCES = ["nourishing", "hydrating", "painful", "toxic", "neutral"]
CIDX = {c: i for i, c in enumerate(CONSEQUENCES)}
N_CONSEQ = len(CONSEQUENCES)


def innate_consequence(p: torch.Tensor, *, nourish_cue: str) -> int:
    """Hardwired drive resolution of a percept p (binary feature vector).

    Conjunctive/disjunctive grounded rules. `nourish_cue` selects which taste
    cues "nourishing" — used to create the systematic train/test gap:
      - train items use nourish_cue="sweet"  (nourishing iff sweet & soft)
      - test  items use nourish_cue="umami"  (nourishing iff umami & soft)
    A model that learned the abstract 'edible-taste & soft → nourishing'
    generalizes; one that memorized 'sweet' fails on the umami test.
    """
    def has(s):  # feature present?
        return p[SIDX[s]] > 0.5
    soft = not has("hard")
    # painful: hot + bright (fire-like), regardless of taste
    if has("hot") and has("bright"):
        return CIDX["painful"]
    # toxic: foul or bitter
    if has("foul") or has("bitter"):
        return CIDX["toxic"]
    # nourishing: the (held-out-able) taste cue + soft
    if has(nourish_cue) and soft:
        return CIDX["nourishing"]
    # hydrating: wet + not hot
    if has("wet") and not has("hot"):
        return CIDX["hydrating"]
    return CIDX["neutral"]


def sample_percepts(n: int, *, nourish_cue: str, noise: float, gen):
    """Sample n binary percepts (Bernoulli per feature) + grounded labels.
    Adds Gaussian sensory noise so it's a representation task, not a lookup."""
    p = (torch.rand(n, N_SENSE, generator=gen) < 0.35).float()
    y = torch.tensor([innate_consequence(p[i], nourish_cue=nourish_cue)
                      for i in range(n)], dtype=torch.long)
    x = p + noise * torch.randn(n, N_SENSE, generator=gen)
    return x, y, p


def make_world(seed: int, noise: float):
    """Train uses sweet-cued nourishing; the systematic test uses umami-cued.
    An i.i.d. test (also sweet-cued) is reported for contrast."""
    g = torch.Generator().manual_seed(seed)
    x_tr, y_tr, p_tr = sample_percepts(8000, nourish_cue="sweet", noise=noise, gen=g)
    x_iid, y_iid, _ = sample_percepts(2000, nourish_cue="sweet", noise=noise, gen=g)
    # systematic: nourishing is now cued by umami (never the train cue).
    # Keep only items whose label actually depends on the cue swap or not —
    # simplest: regenerate fresh items under umami rule.
    x_sys, y_sys, _ = sample_percepts(2000, nourish_cue="umami", noise=noise, gen=g)
    return (x_tr, y_tr), (x_iid, y_iid), (x_sys, y_sys)


# ----------------------------------------------------------------------
# Train one arm (mirrors bench_arena_hierarchical; deterministic matched growth)
# ----------------------------------------------------------------------
def train_arm(arm, train, iid, sysset, *, h_init, cap_bytes, seed, epochs,
              batch, lr, growth_budget, grow_every, verbose):
    x_tr, y_tr = train
    sub = build_substrate(N_SENSE, N_CONSEQ, h_init, cap_bytes, seed)
    sub.compile(); sub.prepare_training()
    cfg = GrowthConfig(same_rank_edges=(arm == "self-arrange"))
    do_grow = (arm != "no-growth")
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    torch.manual_seed(seed + 1000)
    n_tr = x_tr.shape[0]; grown = 0; step = 0
    for epoch in range(epochs):
        perm = torch.randperm(n_tr)
        for b in range(max(1, n_tr // batch)):
            idx = perm[b * batch:(b + 1) * batch]
            loss = torch.nn.functional.cross_entropy(sub(x_tr[idx]), y_tr[idx])
            loss.backward(); sub.zero_dormant_grads(); opt.step(); opt.zero_grad()
            if do_grow and grown < growth_budget and step > 0 and step % grow_every == 0:
                ip = interior_parents(sub.arena)
                if ip:
                    parent = ip[torch.randint(0, len(ip), (1,)).item()]
                    if divide(sub.arena, parent, cfg):
                        grown += 1; sub.compile()
                        opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
            step += 1
        if verbose and epoch % 10 == 0:
            print(f"    [{arm}] ep{epoch} loss={loss.item():.3f} "
                  f"train={eval_acc(sub, x_tr, y_tr):.3f} grown={grown}")
    mr, deep, hist = rank_stats(sub.arena)
    return {
        "arm": arm,
        "train_acc": eval_acc(sub, x_tr, y_tr),
        "iid_acc": eval_acc(sub, *iid),
        "sys_acc": eval_acc(sub, *sysset),
        # The real compositional signal: recall on the cue-swapped class only.
        "sys_nourish_recall": class_recall(sub, *sysset, CIDX["nourishing"]),
        "iid_nourish_recall": class_recall(sub, *iid, CIDX["nourishing"]),
        "grown": grown, "max_rank": mr, "deep": deep, "hist": hist,
    }


@torch.no_grad()
def class_recall(sub, x, y, cls):
    """Recall on one class: of items truly of `cls`, how many predicted `cls`.
    On the umami-cued SYS set, nourishing recall is the compositional test —
    a model that memorized 'sweet→nourishing' scores ~0 here."""
    mask = (y == cls)
    if mask.sum() == 0:
        return float("nan")
    preds = sub(x[mask]).argmax(dim=-1)
    return float((preds == cls).float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=6e-3)
    ap.add_argument("--h-init", type=int, default=8)
    ap.add_argument("--growth-budget", type=int, default=40)
    ap.add_argument("--grow-every", type=int, default=20)
    ap.add_argument("--cap-bytes", type=int, default=200_000)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--arms", default="no-growth,bipartite,self-arrange")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = 1, 20

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print(f"bench_grounded_world: seeds={args.seeds} epochs={args.epochs} "
          f"h_init={args.h_init} grow={args.growth_budget}@{args.grow_every} "
          f"noise={args.noise} senses={N_SENSE} conseq={N_CONSEQ} arms={arms}")

    results = {a: [] for a in arms}
    for seed in range(args.seeds):
        train, iid, sysset = make_world(seed, args.noise)
        if seed == 0:
            print(f"  train={train[0].shape[0]} label balance={torch.bincount(train[1], minlength=N_CONSEQ).tolist()} "
                  f"({CONSEQUENCES})")
            print(f"  sys-test (umami-cued) balance={torch.bincount(sysset[1], minlength=N_CONSEQ).tolist()}")
        for arm in arms:
            t0 = time.time()
            r = train_arm(arm, train, iid, sysset, h_init=args.h_init,
                          cap_bytes=args.cap_bytes, seed=seed, epochs=args.epochs,
                          batch=args.batch, lr=args.lr,
                          growth_budget=args.growth_budget, grow_every=args.grow_every,
                          verbose=args.smoke)
            r["t"] = time.time() - t0
            results[arm].append(r)
            print(f"  [s{seed} {arm:>12s}] train={r['train_acc']:.3f} "
                  f"iid={r['iid_acc']:.3f} SYS={r['sys_acc']:.3f} "
                  f"nourish-recall iid={r['iid_nourish_recall']:.3f}→SYS={r['sys_nourish_recall']:.3f} "
                  f"grown={r['grown']} max_rank={r['max_rank']} deep={r['deep']} "
                  f"({r['t']:.0f}s)")

    print("\n=== aggregate (mean ± std) — SYS is the compositional headline ===")
    for arm in arms:
        rs = results[arm]
        def ms(k):
            xs = [r[k] for r in rs]
            return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)
        ta = ms("train_acc"); ii = ms("iid_acc"); sy = ms("sys_acc")
        nr = ms("sys_nourish_recall")
        mr = statistics.mean([r["max_rank"] for r in rs])
        dp = statistics.mean([r["deep"] for r in rs])
        print(f"  {arm:>12s}  train={ta[0]:.3f}  iid={ii[0]:.3f}  "
              f"SYS={sy[0]:.3f}  nourish-recall(SYS)={nr[0]:.3f}±{nr[1]:.3f}  "
              f"max_rank={mr:.1f} deep={dp:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
