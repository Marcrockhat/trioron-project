"""z2 H-space routing — does the chained-15 full-accuracy pump transfer to the world?

Rocky (2026-06-04): the chained-15 full-softmax pump (0.55 -> 0.69 storage-free / 0.76
oracle, commits 62aa57e/7e561e4) came NOT from perception replay but from a SECOND
manifold in H-space (the interior code) that does ROUTING — "z2" — with full-covariance
Mahalanobis the further lever. The world is full-softmax-hard precisely because it has no
task selector; the z2 router IS a learned task selector (infer fire-vs-water context from
the stable interior representation, which doesn't drift like the shared head).

This is the PORT-AND-VALIDATE step (de-risk before architecture): lift the exact dual-
manifold mechanism (ManifoldArchive over interior activations, log_likelihood vs
log_likelihood_full) and ask the minimal question — can the z2 manifold route fire-context
vs water-context from the 32-d interior code at all, and does full-cov beat diagonal the
way it did on chained-15? If yes, building per-skill routing is justified. If the interior
code can't separate the contexts, routing is dead and we learned it cheaply.

No skill training here — just a competent solo organism's interior representation + the
router. Routing accuracy on held-out context states is the whole result.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.fire_taming import fire_oracle
from experiments.world.quest import water_master
from experiments.world.mirror_cells import _solo
from experiments.world.render_organism import train_solo
from experiments.world.consolidate_base import build_battery, _gene_mask
from trioron.core.epigenome import OUTPUT, CREDIT_ELIGIBLE, MIRROR
from trioron.learning.manifold import ManifoldArchive


def _p(*a):
    print(*a, flush=True)


@torch.no_grad()
def interior_codes(sub, percepts, interior_ids):
    """Push percepts through the organism, return the 32-d interior activations [N, H]."""
    sub(_solo(percepts))
    return sub.last_activations[:, interior_ids].clone()


def route_eval(sub, interior_ids, codes_by_ctx, *, full_cov, split=0.7):
    """Fit a per-context H-manifold on a train split, route the held-out split by
    argmax log-likelihood. Returns (accuracy, per-context recall)."""
    archive = ManifoldArchive(sub.arena, full_cov=full_cov)
    test = {}
    for ctx, codes in codes_by_ctx.items():
        n_tr = int(split * codes.shape[0])
        archive.update_class(ctx, codes[:n_tr])
        test[ctx] = codes[n_tr:]
    ctxs = sorted(codes_by_ctx)
    astros = {c: archive.get(c) for c in ctxs}
    correct = 0; total = 0
    recall = {}
    for ctx, tc in test.items():
        if tc.shape[0] == 0:
            recall[ctx] = float("nan"); continue
        lls = []
        for c in ctxs:
            astro = astros[c]
            ll = astro.log_likelihood_full(tc) if full_cov else astro.log_likelihood(tc)
            lls.append(ll)
        pred = torch.stack(lls, dim=1).argmax(dim=1)             # index into ctxs
        pred_ctx = torch.tensor([ctxs[i] for i in pred.tolist()])
        hit = (pred_ctx == ctx).sum().item()
        correct += hit; total += tc.shape[0]
        recall[ctx] = hit / tc.shape[0]
    return correct / max(total, 1), recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--solo-ep", type=int, default=100)
    args = ap.parse_args()

    _p("=== z2 H-ROUTING TRANSFER TEST — can the interior code route fire vs water? ===")
    _p("    diagonal Gaussian vs full-covariance Mahalanobis (the chained-15 pump)\n")

    torch.manual_seed(20260604)
    fire_states = build_battery(fire_oracle, which="cold")[0]      # cold/fire-context
    water_states = build_battery(water_master, which="thirsty")[0] # thirsty/water-context
    _p(f"context states: {fire_states.shape[0]} fire, {water_states.shape[0]} water "
       f"(chance = 50%)\n")

    diag, fcov = [], []
    for seed in range(args.seeds):
        sub = train_solo(seed, args.solo_ep, n_mirror=8)
        interior_ids = torch.nonzero(
            _gene_mask(sub, CREDIT_ELIGIBLE, exclude=(OUTPUT, MIRROR))).flatten()
        codes = {0: interior_codes(sub, fire_states, interior_ids),
                 1: interior_codes(sub, water_states, interior_ids)}
        acc_d, rec_d = route_eval(sub, interior_ids, codes, full_cov=False)
        acc_f, rec_f = route_eval(sub, interior_ids, codes, full_cov=True)
        diag.append(acc_d); fcov.append(acc_f)
        _p(f"  seed{seed} (H={interior_ids.numel()}): "
           f"diagonal {acc_d:.3f} (fire {rec_d[0]:.2f}/water {rec_d[1]:.2f})  |  "
           f"full-cov {acc_f:.3f} (fire {rec_f[0]:.2f}/water {rec_f[1]:.2f})")

    _p(f"\n=== SUMMARY (n={args.seeds}, chance 0.50) ===")
    _p(f"  diagonal Gaussian : {statistics.mean(diag):.3f} +/- "
       f"{statistics.pstdev(diag) if len(diag) > 1 else 0:.3f}")
    _p(f"  full-cov Mahalanobis: {statistics.mean(fcov):.3f} +/- "
       f"{statistics.pstdev(fcov) if len(fcov) > 1 else 0:.3f}")
    _p(f"  full-cov lift over diagonal: "
       f"{statistics.mean(fcov) - statistics.mean(diag):+.3f}")
    _p("\n  routing transfers if accuracy >> 0.50; full-cov validated if it beats diagonal")
    _p("  (chained-15: dual-manifold 0.55->0.68, full-cov 0.68->0.76).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
