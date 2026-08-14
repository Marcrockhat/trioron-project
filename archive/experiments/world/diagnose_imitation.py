"""Why can't the organism learn from the PERFECT master? (2026-08-14)

The perfect-master apprenticeship REGRESSED survival (52.4 → 35.1) while the
master bars 131. The June probe (imitation_ceiling.py) cleared substrate+percept
— but only on single-skill in-band states, with a full-credit supervised arm.
Our failure is the FULL arbiter over ALL states through the MIRROR-GATED channel.

Decompose into four candidate causes, one supervised probe each (predict the
master's action on held-out states; accuracy is the clean signal):

  linear(percept)        H-linear: the substrate forward is nonlinear=False. An
                         arbiter full of threshold sign-flips (seek fire when
                         cold / flee fire when warm) may be linearly inexpressible.
  MLP(percept)           the nonlinear ceiling on what the organism SEES.
  MLP(privileged state)  perception gap (now WITH predator features — the June
                         state_features had none, but the master evades).
  trioron full-credit    substrate as a supervised clone (all params train).
  trioron mirror-gated   the ACTUAL DAgger imitation channel: CE gradient only
                         through 8 mirror cells' incident params.
  trioron nonlinear      full-credit again with nonlinear=True — substrate's own
                         fix if H-linear is the wall.

Accuracy checkpoints during training (learning SPEED, not just asymptote).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.imitation_ceiling import (
    collect, state_features, MLP, _p,
)
from experiments.world.revisit_smoke import perfect_master
from experiments.world.mirror_cells import (
    build_mirror, _solo, keep_only_mirror_grads,
)
from experiments.world.tile_world import N_ACTION, TileWorld

CHECKPOINTS = (10, 30, 100, 300)


def _pred_features(w):
    s = w.size
    dx = ((w.pred[0] - w.px + s // 2) % s) - s // 2
    dy = ((w.pred[1] - w.py + s // 2) % s) - s // 2
    dist = abs(dx) + abs(dy)
    return torch.tensor([dx / s, dy / s, 1.0 / (1.0 + dist)],
                        dtype=torch.float32)


@torch.no_grad()
def collect_all(master_fn, *, seeds, cap, max_steps=300):
    """Like imitation_ceiling.collect(which='all') but privileged state also
    carries predator features (the perfect master evades)."""
    P, S, Y = [], [], []
    for ep in range(seeds):
        w = TileWorld(seed=ep * 31 + 3, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            a = master_fn(w)
            P.append(p)
            S.append(torch.cat([state_features(w), _pred_features(w)]))
            Y.append(a)
            p, _, done, _ = w.step(a)
            if len(P) >= cap:
                break
        if len(P) >= cap:
            break
    return torch.stack(P), torch.stack(S), torch.tensor(Y)


def _fit(model, params, Xtr, Ytr, Xte, Yte, *, lr, fwd=None, post_backward=None,
         batch=128, epochs=300):
    """Minibatch CE training with checkpointed held-out accuracy."""
    opt = torch.optim.Adam(params, lr=lr)
    ce = torch.nn.functional.cross_entropy
    fwd = fwd or (lambda x: model(x))
    curve = {}
    n = Xtr.shape[0]
    g = torch.Generator().manual_seed(7)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            ce(fwd(Xtr[idx]), Ytr[idx]).backward()
            if post_backward is not None:
                post_backward()
            opt.step()
        if ep in CHECKPOINTS:
            with torch.no_grad():
                curve[ep] = (fwd(Xte).argmax(1) == Yte).float().mean().item()
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--cap", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-mirror", type=int, default=8)
    args = ap.parse_args()

    _p("=== DIAGNOSE: why no fast learning from the perfect master? ===")
    P, S, Y = collect_all(perfect_master, seeds=args.seeds, cap=args.cap)
    n = P.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    ntr = int(n * 0.8)
    tr, te = perm[:ntr], perm[ntr:]
    maj = int(torch.bincount(Y[tr], minlength=N_ACTION).argmax())
    chance = (Y[te] == maj).float().mean().item()
    dist = torch.bincount(Y, minlength=N_ACTION).float() / n
    _p(f"n={n} states, action distribution: "
       + " ".join(f"{d:.2f}" for d in dist.tolist())
       + f"   chance(majority)={chance:.3f}\n")

    arms = {}

    torch.manual_seed(0)
    lin = torch.nn.Linear(P.shape[1], N_ACTION)
    arms["linear(percept)"] = _fit(lin, lin.parameters(),
                                   P[tr], Y[tr], P[te], Y[te],
                                   lr=3e-3, epochs=args.epochs)

    torch.manual_seed(0)
    mlp_p = MLP(P.shape[1])
    arms["MLP(percept)"] = _fit(mlp_p, mlp_p.parameters(),
                                P[tr], Y[tr], P[te], Y[te],
                                lr=3e-3, epochs=args.epochs)

    torch.manual_seed(0)
    mlp_s = MLP(S.shape[1])
    arms["MLP(priv state)"] = _fit(mlp_s, mlp_s.parameters(),
                                   S[tr], Y[tr], S[te], Y[te],
                                   lr=3e-3, epochs=args.epochs)

    for label, nl, gated in (("trioron full-credit", False, False),
                             ("trioron MIRROR-GATED", False, True),
                             ("trioron nonlinear", True, False)):
        torch.manual_seed(0)
        sub = build_mirror(0, n_mirror=args.n_mirror, nonlinear=nl)
        arms[label] = _fit(
            sub, sub.trainable_tensors(), P[tr], Y[tr], P[te], Y[te],
            lr=3e-3, epochs=args.epochs,
            fwd=lambda x, s=sub: s(_solo(x)),
            post_backward=(lambda s=sub: (keep_only_mirror_grads(s),
                                          s.zero_dormant_grads())) if gated
            else (lambda s=sub: s.zero_dormant_grads()))

    _p("held-out accuracy at epoch checkpoints "
       f"{CHECKPOINTS} (chance={chance:.3f}):")
    for label, curve in arms.items():
        row = "  ".join(f"{curve[c]:.3f}" for c in CHECKPOINTS)
        _p(f"  {label:>22s}: {row}")

    _p("\n=== VERDICT GUIDE ===")
    _p("  linear low, MLP(percept) high        → LINEAR substrate can't express the arbiter")
    _p("  MLP(percept) low, MLP(state) high    → perception gap (percept hides decision info)")
    _p("  full-credit ≪ MLP(percept)           → substrate/training gap beyond linearity")
    _p("  MIRROR-GATED ≪ full-credit           → the 8-mirror-cell credit gate is the choke")
    _p("  nonlinear ≫ full-credit              → substrate's own nonlinearity is the fix")
    _p("  slow rise to a high asymptote        → speed problem, not capacity problem")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
