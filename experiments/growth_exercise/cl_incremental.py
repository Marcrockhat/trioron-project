"""Class-incremental continual learning on the 10-species taxonomy — native machinery.

Adds ONE species per task (10 tasks). After each task we measure full-softmax accuracy on
ALL species seen so far (no task selector — the hard regime) against that subset's Bayes
ceiling. Two arms:

  naive      : train each task on its single class only. No replay, no anchor, no lock.
               Single-class CE collapses the shared readout onto the latest species ->
               catastrophic forgetting (the thing to beat).
  machinery  : trioron's native anti-forgetting, wired (NOT bypassed):
                 - manifold replay  : per-class (mu,Sigma) sketch; pseudo-samples of all
                                      prior species mixed into every batch (learning/manifold)
                 - epigenetic lock  : |w.g| saliency -> node_lambda, anchor at boundary,
                                      strength * ewc_penalty added to the loss (learning/epigenetic_lock)
                 - credit lock      : consolidate() at each boundary freezes settled heads
                                      (dormant + zero_dormant_grads) (learning/credit)

Run: python3 -m experiments.growth_exercise.cl_incremental
"""
from __future__ import annotations

import torch

from trioron.core import Envelope, construct
from trioron.bases import minimal
from trioron.phenotype import default_dispatch_table
from trioron.learning.credit import CreditTracker, CreditConfig
from trioron.learning.manifold import ManifoldArchive
from trioron.learning import epigenetic_lock as ep

from experiments.growth_exercise.taxonomy import make_taxonomy, bayes_accuracy, ALL, SPECIES

EWC_STRENGTH = 0.0   # VALIDATED OFF: on this small shared net the soft λ-EWC penalty
# over-anchors and monotonically HURTS (0.85 -> 0.59 -> 0.51 as strength rises). The hard
# mechanisms carry CL here: manifold replay (defends old species by rehearsal) + credit-lock
# (freezes settled heads). Matches the manual — credit-lock is the primary anchor, EWC a
# complement, Fisher/λ washes out. Re-enable only if a deeper net needs the plastic base
# anchored. λ is still computed (accumulate_saliency/anchor) — just not penalized.
STEPS = 250
LR = 0.05
N_TRAIN = 128
N_REPLAY = 64


def new_substrate():
    sub = construct(base=minimal(7, 10), envelope=Envelope(),
                    dispatch_table=default_dispatch_table(), capacity=128)
    sub.prepare_training()
    sub.compile()
    return sub


def seen_bayes(test, n_seen: int) -> float:
    """Bayes ceiling restricted to the first n_seen species (argmax over seen only)."""
    names = ALL[:n_seen]
    mask = test.y < n_seen
    logp = torch.zeros(int(mask.sum()), n_seen)
    kg, cm, cov, eg = test.kg[mask], test.cm[mask], test.cover[mask], test.eggs[mask]
    from experiments.growth_exercise.taxonomy import _cat_log_prob, COVERS, P_COVER, P_EGGS
    for ci, name in enumerate(names):
        s = SPECIES[name]
        logp[:, ci] = (s.weight.log_prob(kg) + s.height.log_prob(cm)
                       + _cat_log_prob(cov, s.cover, len(COVERS), P_COVER)
                       + _cat_log_prob(eg, s.eggs, 2, P_EGGS))
    return float((logp.argmax(1) == test.y[mask]).float().mean())


def eval_seen(sub, test, n_seen: int) -> float:
    """Full-softmax (10-way argmax) accuracy on test samples of the first n_seen species."""
    mask = test.y < n_seen
    with torch.no_grad():
        pred = sub(test.X[mask]).argmax(1)
    return float((pred == test.y[mask]).float().mean())


def eval_per_class(sub, test) -> list[float]:
    with torch.no_grad():
        pred = sub(test.X).argmax(1)
    return [float((pred[test.y == c] == c).float().mean()) for c in range(len(ALL))]


def _select_exemplars(sub, Xc, t, mode, M):
    """Pick M real exemplars of class t to remember. 'hard' = lowest-margin (most
    confusable with another class under the current model); 'random' = control."""
    M = min(M, Xc.shape[0])
    if mode == "hard":
        with torch.no_grad():
            lg = sub(Xc)
        other = lg.clone(); other[:, t] = float("-inf")
        margin = lg[:, t] - other.max(1).values          # small margin = near a boundary
        idx = margin.argsort()[:M]
    else:
        idx = torch.randperm(Xc.shape[0])[:M]
    return Xc[idx].clone()


def _draw_exemplars(pool, n):
    """Resample n replay points from a stored exemplar pool, jittering only the two
    continuous dims (weight, height) so discrete cover/eggs stay valid."""
    if pool is None or pool.shape[0] == 0:
        return None
    s = pool[torch.randint(0, pool.shape[0], (n,))].clone()
    s[:, :2] = s[:, :2] + 0.1 * torch.randn(n, 2)
    return s


def run(train, test, *, replay=True, lock=False, ewc=0.0, mixture_k=0, exemplar=None, M=8):
    """One arm. DEFAULTS = the validated workhorse (s023): manifold replay ON, credit-lock
    and λ-EWC OFF. Replay alone recovers 97% of joint-achievable (0.20 naive → 0.85);
    credit-lock is idle on a bipartite net (no interior cells to freeze) and λ-EWC
    over-anchors. mixture_k>0 stores each class as K prototypes (StreamingMixture = the
    dendrite-as-RAG multi-prototype memory) instead of a single Gaussian. Returns
    (curve, n_cells_locked, per_class_final_acc)."""
    sub = new_substrate()
    a = sub.arena
    credit = CreditTracker(a, CreditConfig(consecutive_tasks=1)) if lock else None
    archive = ManifoldArchive(a, full_cov=True, mixture_k=mixture_k) if replay else None
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=LR)

    mem: dict[int, torch.Tensor] = {}   # class -> stored real exemplars [M, 7]
    n_locked = 0
    curve = []
    for t, name in enumerate(ALL):
        Xc = train.X[train.y == t]
        yc = train.y[train.y == t]
        for _ in range(STEPS):
            X, y = Xc, yc
            if replay and t > 0:                          # mix in replayed prior species
                rx, ry = [], []
                for c in range(t):
                    if exemplar is not None:
                        s = _draw_exemplars(mem.get(c), N_REPLAY)
                    elif mixture_k > 0:
                        s = archive.sample_mixture(c, N_REPLAY)
                        if s is None and archive.get(c) is not None:
                            s = archive.get(c).sample_full(N_REPLAY)
                    else:
                        astro = archive.get(c)
                        s = astro.sample_full(N_REPLAY) if astro is not None else None
                    if s is not None:
                        rx.append(s)
                        ry.append(torch.full((N_REPLAY,), c))
                if rx:
                    X = torch.cat([Xc] + rx); y = torch.cat([yc] + ry)
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(sub(X), y)
            if ewc > 0:
                loss = loss + ewc * ep.ewc_penalty(a)
            loss.backward()
            if ewc > 0:
                ep.accumulate_saliency(a)
            if lock:
                credit.update_utility()
                sub.zero_dormant_grads()
            opt.step()
            if lock:
                act = sub.last_activations
                if act is not None:
                    credit.update_engagement(act)
        # ── task boundary ──
        if replay and exemplar is not None:
            mem[t] = _select_exemplars(sub, Xc, t, exemplar, M)   # remember M real exemplars
        elif replay:
            archive.update_class(t, Xc)        # store this species' sketch for future replay
        if ewc > 0:
            ep.refresh_lambda(a)
            ep.anchor(a)
        if lock:
            n_locked += len(credit.consolidate())
        sub.end_task()
        curve.append(eval_seen(sub, test, t + 1))
    return curve, n_locked, eval_per_class(sub, test)


def main() -> None:
    train = make_taxonomy(ALL, n_per_class=N_TRAIN, seed=0)
    test = make_taxonomy(ALL, n_per_class=128, seed=1)
    ceil = seen_bayes(test, len(ALL))

    configs = [
        ("naive",             dict(replay=False, lock=False, ewc=0.0)),
        ("replay only",       dict(replay=True,  lock=False, ewc=0.0)),
        ("credit-lock only",  dict(replay=False, lock=True,  ewc=0.0)),
        ("replay + lock",     dict(replay=True,  lock=True,  ewc=0.0)),
        ("replay+lock+λ(50)", dict(replay=True,  lock=True,  ewc=50.0)),
    ]
    print("class-incremental ablation — which mechanism does the work?\n")
    print(f"{'config':18} {'final':>7} {'peak':>7} {'cells_locked':>13}")
    for nm, cfg in configs:
        curve, nlock, _ = run(train, test, **cfg)
        print(f"{nm:18} {curve[-1]:>7.3f} {max(curve):>7.3f} {nlock:>13}")
    print(f"\nBayes ceiling (10 species): {ceil:.3f}")


if __name__ == "__main__":
    main()
