"""Two brains in one body: a donor's organs nested beside the recipient's, read on the
recipient's fetal gate (s070; Rocky: "I imagine that these are two different brain in an
organism" — the nest handles which one answers).

IN MEMORY ONLY: both saved bodies and logs are loaded; nothing is saved; no home changes.
One organism each, one seed.

Per organ the two leaves stand side by side, nothing merged.  trioron's own nest router
(`ManifoldRouter` over a full-covariance profile per leaf, as `PhasecyteNest.fit_router`)
is fitted from what each leaf lived (its own log, the organ's numbers unblurred, ROUTER_N
moments each) and picks ONE leaf for every moment; that leaf's class is the body's class.
The gate rows are the recipient's (`fetal.gate`, his world, his unseen days), each read
three ways: his brain alone, her brain alone, the nest.  The donor's brain has not lived
his world: it classifies, it does not learn.

    VEE_HOME=outputs/incubator/weiss DONOR=outputs/incubator/vee python3 experiments/incubator/nest_probe.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import vee_body as vb
import fetal
from germline_checks import readable
from world import ACTIONS, HIDDEN
from trioron.core import construct
from trioron.learning.manifold import ManifoldArchive
from trioron.learning.router import ManifoldRouter
from trioron.pcll.progenitor import germline_base

DONOR = os.environ.get("DONOR", "outputs/incubator/vee")
ROUTER_N = int(os.environ.get("ROUTER_N", 4000))


def fit_router(parts: list, seed: int):
    """One profile per leaf from what it lived; constant numbers are left out (no spread to read)."""
    keep = torch.stack([p.std(0) > 1e-6 for p in parts]).all(0)
    archive = ManifoldArchive(construct(germline_base, capacity=64).arena, full_cov=True)
    for g, D in enumerate(parts):
        gen = torch.Generator().manual_seed(7000 + seed + g)
        archive.update_class(g, D[torch.randperm(len(D), generator=gen)[:ROUTER_N]][:, keep])
    archive.finalize_all()
    router = ManifoldRouter(archive, full_cov=True)
    return lambda P: torch.cat([router.route_class(P[i:i + 2000][:, keep], class_ids=[0, 1], n_classes=2)
                                for i in range(0, len(P), 2000)])


def main():
    his = vb.Body.load()
    hers = torch.load(f"{DONOR}/body.pt", weights_only=False)
    log, dlog = torch.load(fetal.LOG, weights_only=False), torch.load(f"{DONOR}/fetal_log.pt", weights_only=False)
    fetal.CORRIDOR = his.extra["world"].corridor
    X, Hd, T, A = (torch.cat(log[k]) for k in ("X", "hidden", "text", "action"))
    Xd = torch.cat(dlog["X"])
    ambient = vb.language_sense()(vb.wiki_states())
    Xp, Hp, Tp, Ap = fetal.probe(ambient, his.seed)
    n = len(X)
    K, Kp, Kd = vb.changes(X), vb.changes(Xp), vb.changes(Xd)
    print(f"# two brains, in memory only: {vb.HOME} ({his.extra['fetal_day']} fetal days, classes {his.classes}) "
          f"with the organs of {DONOR} ({hers.extra['fetal_day']} fetal days, classes {hers.classes}); "
          f"classes in the nest {({k: his.classes[k] + hers.classes[k] for k in his.classes})}")

    route = {"kin": fit_router([K, Kd], his.seed),
             "soma": fit_router([X[:, vb.LANG:], Xd[:, vb.LANG:]], his.seed),
             "ear": fit_router([X[:, :vb.LANG], Xd[:, :vb.LANG]], his.seed)}

    def reader(who: str, organ: str):
        """class of the body per moment, from the organ's numbers P (unblurred)."""
        a = lambda P: his.organs()[organ].mixed.classify(his.seen(organ, P))[0]
        b = lambda P: hers.organs()[organ].mixed.classify(hers.seen(organ, P))[0]
        if who == "his": return a
        if who == "hers": return b
        def nest(P):
            r = route[organ](P)
            return torch.where(r == 0, a(P), b(P) + 100000)
        return nest

    def to_her(organ, P):
        return float((route[organ](P) == 1).float().mean())
    print(f"  moments of HIS world the router gives to her brain: sense of change {to_her('kin', torch.cat([K, Kp])):.3f}, "
          f"body sense {to_her('soma', torch.cat([X, Xp])[:, vb.LANG:]):.3f}, ear {to_her('ear', torch.cat([X, Xp])[:, :vb.LANG]):.3f}; "
          f"of HER lived moments: {to_her('kin', Kd):.3f}, {to_her('soma', Xd[:, vb.LANG:]):.3f}, {to_her('ear', Xd[:, :vb.LANG]):.3f}")

    def events(Hx, Ax):
        up = lambda j: torch.cat([torch.zeros(1, dtype=torch.bool), (Hx[1:, j] == 1) & (Hx[:-1, j] == 0)])
        out = {}
        for name, act, j in (("CPU rise", "work", HIDDEN.index("busy")), ("memory rise", "hold", HIDDEN.index("full"))):
            mine, world = Ax == ACTIONS.index(act), up(j)
            keep = mine ^ world
            out[name] = (keep, mine[keep].long())
        return out

    rows: dict = {}
    for who in ("his", "hers", "nest"):
        kin, soma_r, ear_r = reader(who, "kin"), reader(who, "soma"), reader(who, "ear")
        out = rows.setdefault(who, {})
        cls, clsp = kin(K), kin(Kp)
        out["1 feels the kicks (k-means 1.00)"] = fetal.recall_per_value(fetal.vote(cls, A, clsp, 6), Ap)
        mu, sd = K[:, 8:].mean(0), K[:, 8:].std(0) + 1e-6
        Kh = Kp.clone(); Kh[:, 8:] = 0
        ch, right = kin(Kh), 0
        for an, cols in fetal.MOVES.items():
            hit = []
            for r in (Ap == ACTIONS.index(an)).nonzero().squeeze(1):
                m = cls == ch[r]
                hit.append(bool(m.any()) and int(((K[m][:, 8:].mean(0) - mu) / sd).abs().argmax()) in cols)
            acc = float(np.mean(hit)) if hit else float("nan")
            right += acc > 0.5
            out[f"2 what '{an}' moves"] = acc
        out["2 right, of 5 (>= 4)"] = right
        Ke, Kpe = K.clone(), Kp.clone()
        Ke[:, :8] = 0; Kpe[:, :8] = 0; Ke[:, 41:] = 0; Kpe[:, 41:] = 0
        ce, cpe = kin(Ke), kin(Kpe)
        ev, evp = events(Hd, A), events(Hp, Ap)
        for name in ev:
            (m, y), (mp, yp) = ev[name], evp[name]
            base = max(yp.float().mean().item(), 1 - yp.float().mean().item())
            out[f"3 {name}, echo hidden (base {base:.2f})"] = (fetal.vote(ce[m], y, cpe[mp], 2) == yp).float().mean().item()
        sall, Hall = soma_r(torch.cat([X, Xp])[:, vb.LANG:]), torch.cat([Hd, Hp])
        for j, h in enumerate(HIDDEN):
            if h == "night": continue
            out[f"4 nothing lost: {h} (embryo {fetal.EMBRYO[h]:.2f})"] = readable(sall, Hall[:, j], n)[0]
        out[f"4 ear: text present (embryo {fetal.EAR_EMBRYO:.2f})"] = readable(ear_r(torch.cat([X, Xp])[:, :vb.LANG]), (torch.cat([T, Tp]) >= 0).long(), n)[0]
        print(f"  read: {who}", flush=True)

    print(f"\n  {'row':52} {'his brain':>10} {'her brain':>10} {'the nest':>10}")
    for r in rows["his"]:
        print(f"  {r:52} " + " ".join(f"{rows[w][r]:10.2f}" for w in ("his", "hers", "nest")))


if __name__ == "__main__":
    main()
