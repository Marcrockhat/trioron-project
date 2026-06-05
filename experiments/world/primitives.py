"""Phase 1 of the Mode-E survival recipe — the four clean primitive donors.

The imitation-ceiling diagnosis (session 015) settled the substrate and the
percept: trioron imitates a master at the MLP ceiling on a clean skill. The
recipe that follows: stop holistic imitation of one all-in-one master; instead
build a VOCABULARY of clean single-drive primitives, each learnable on its own,
then compose + route + self-improve.

This module is step 1: train and persist the four primitives, on the v2.0 core
substrate (build_mirror -> construct(seeded(80,6))), and report held-out
imitation fidelity per primitive.

  WARM    temperature   fire_oracle      | band: temp < 0.5   (the cold decisions)
  HYDRATE thirst        water_master     | band: thirst < 0.9
  FORAGE  energy        food_master      | band: energy < 0.85
  EVADE   integrity     evade_master(*)  | band: predator within Chebyshev 2

Collection runs the masters under their NATURAL (stochastic) exploration, then
keeps only states inside each primitive's decision BAND — the band excludes the
satisfied/idle states (where a master idles or random-walks) and leaves the
skill itself (navigate-to-resource + consume / flee). The band does the cleaning
that a deterministic teacher cannot here: a deterministic master is too good and
camps its resource (WATER never depletes), so it never navigates under drive
stress and yields a degenerate always-consume donor. The wander creates the
stress; the band recovers the skill.

(*) EVADE is the missing primitive that the masters' random-explore fallback was
incidentally standing in for (deterministic masters get cornered: survival
85->69). It is made explicit here from the run_reactive flee rule.

A "donor" is a trained mirror substrate saved as (seed, bias, edge_weight) — the
topology is rebuilt deterministically by build_mirror(seed, n_mirror), then the
weights are loaded. Output: runs/primitives/{name}.pt + a fidelity table.

Usage:
    python3 -m experiments.world.primitives --smoke     # fast wiring proof
    python3 -m experiments.world.primitives             # full build (persists donors)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# importing fire_taming pins the tamed-fire physics (WARM_RATE=0.04 etc.) and
# gives us the WARM master + the shared EXPLORE_DETERMINISTIC flag.
import experiments.world.fire_taming as _ft           # noqa: E402
from experiments.world.fire_taming import fire_oracle, evaluate, _near_fire  # noqa: E402
from experiments.world.quest import water_master, food_master    # noqa: E402
from experiments.world.mirror_cells import build_mirror, _solo   # noqa: E402
from experiments.world.tile_world import (              # noqa: E402
    TileWorld, ACTIONS, N_ACTION, FIRE, POISON, _DXY,
)

RUNS = Path(__file__).resolve().parent.parent.parent / "runs"
PRIM_DIR = RUNS / "primitives"

_OPP_ACT = {ACTIONS.index("N"): "S", ACTIONS.index("S"): "N",
            ACTIONS.index("E"): "W", ACTIONS.index("W"): "E"}


# ----------------------------------------------------------------------
# The EVADE master — integrity virtuoso (the new primitive)
# ----------------------------------------------------------------------
def _pred_delta(w):
    """Signed toroidal (dx, dy) from the organism to the predator."""
    s = w.size
    dx = ((w.pred[0] - w.px + s // 2) % s) - s // 2
    dy = ((w.pred[1] - w.py + s // 2) % s) - s // 2
    return dx, dy


def _flee_pred(w):
    """Step along the axis of greatest predator approach, AWAY from it."""
    dx, dy = _pred_delta(w)
    if abs(dx) >= abs(dy):
        # predator to the E (dx>0) -> flee W; to the W -> flee E
        return ACTIONS.index("W" if dx > 0 else "E")
    return ACTIONS.index("N" if dy > 0 else "S")


def evade_master(w):
    """Integrity virtuoso: keep away from the predator and off hazards; naive
    about thirst/energy/temp. Deterministic by construction (a clean donor):
      1. on a hazard tile (FIRE burns, POISON would have to be eaten but standing
         is the tell) -> step off, away from the predator;
      2. predator within Chebyshev 2 -> flee;
      3. otherwise hold (rest) -- no random walk, so the skill is learnable."""
    here = int(w.grid[w.py, w.px])
    if here in (FIRE, POISON):
        return _flee_pred(w)                 # leave the hazard, biased away from pred
    dx, dy = _pred_delta(w)
    if max(abs(dx), abs(dy)) <= 2:           # threat present -> evade
        return _flee_pred(w)
    return ACTIONS.index("rest")             # deterministic default


# ----------------------------------------------------------------------
# Primitive registry: master + decision-band filter
# ----------------------------------------------------------------------
def _band_warm(w):
    # temperature-management CONTEXT, not just "cold": include near-fire states
    # (temp can be >0.5 there) so the donor also learns the critical LEAVE-before-
    # overheat decisions — a temp<0.5-only band omits them and the donor overheats.
    return w.temp < 0.5 or _near_fire(w)


# bands keep the states where the master is exercising its skill (navigating to /
# consuming its resource); off-band is the satisfied/idle/wander behaviour we
# exclude from the clean donor.
def _band_hydrate(w):
    return w.thirst < 0.9


def _band_forage(w):
    return w.energy < 0.85


def _band_evade(w):
    dx, dy = _pred_delta(w)
    return max(abs(dx), abs(dy)) <= 2 or int(w.grid[w.py, w.px]) in (FIRE, POISON)


PRIMITIVES = {
    "WARM":    dict(master=fire_oracle,  band=_band_warm,    drive="temperature"),
    "HYDRATE": dict(master=water_master, band=_band_hydrate, drive="thirst"),
    "FORAGE":  dict(master=food_master,  band=_band_forage,  drive="energy"),
    "EVADE":   dict(master=evade_master, band=_band_evade,   drive="integrity"),
}


# ----------------------------------------------------------------------
# Collect decision-band demos for one primitive
# ----------------------------------------------------------------------
@torch.no_grad()
def collect(master_fn, band_fn, *, seeds=40, cap=900, max_steps=300):
    """Run the master in the full world; keep (percept, action) where this
    primitive's drive is the decision (band_fn True)."""
    P, Y = [], []
    for ep in range(seeds):
        w = TileWorld(seed=ep * 31 + 3, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            keep = band_fn(w)
            a = master_fn(w)
            if keep:
                P.append(p); Y.append(a)
            p, _, done, _ = w.step(a)
            if len(P) >= cap:
                break
        if len(P) >= cap:
            break
    if not P:
        raise ValueError(
            f"collected 0 band states in {seeds} episodes — the band is too rare "
            f"at this budget; raise --collect-seeds")
    return torch.stack(P), torch.tensor(Y)


# ----------------------------------------------------------------------
# Train + evaluate one primitive donor (v2.0 core substrate, supervised CE)
# ----------------------------------------------------------------------
def train_donor(P, Y, *, seed=0, n_mirror=8, epochs=300, lr=3e-3, batch=128):
    """Supervised imitation on the mirror substrate (same arm as
    imitation_ceiling.train_eval_substrate, but the substrate is RETURNED so it
    can be persisted as a donor). 80/20 split -> (substrate, held-out accuracy)."""
    torch.manual_seed(seed)
    n = P.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    ntr = int(n * 0.8)
    tr, te = perm[:ntr], perm[ntr:]
    Ptr, Ytr, Pte, Yte = P[tr], Y[tr], P[te], Y[te]

    sub = build_mirror(seed, n_mirror=n_mirror)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    ce = torch.nn.functional.cross_entropy
    ntr = Ptr.shape[0]
    for _ in range(epochs):
        ep_perm = torch.randperm(ntr)
        for i in range(0, ntr, batch):
            idx = ep_perm[i:i + batch]
            opt.zero_grad()
            ce(sub(_solo(Ptr[idx])), Ytr[idx]).backward()
            sub.zero_dormant_grads()
            opt.step()
    with torch.no_grad():
        acc = (sub(_solo(Pte)).argmax(1) == Yte).float().mean().item()
        maj = int(torch.bincount(Ytr, minlength=N_ACTION).argmax())
        chance = (Yte == maj).float().mean().item()
    return sub, acc, chance, maj


def save_donor(sub, name, *, seed, n_mirror, acc, chance, drive):
    PRIM_DIR.mkdir(parents=True, exist_ok=True)
    path = PRIM_DIR / f"{name}.pt"
    torch.save({
        "name": name, "drive": drive, "seed": seed, "n_mirror": n_mirror,
        "bias": sub.arena.bias.detach().clone(),
        "edge_weight": sub.arena.edge_weight.detach().clone(),
        "fidelity": acc, "chance": chance,
    }, path)
    return path


def load_donor(path):
    """Rebuild the substrate from (seed, n_mirror) and load its trained weights."""
    d = torch.load(path, weights_only=False)
    sub = build_mirror(d["seed"], n_mirror=d["n_mirror"])
    with torch.no_grad():
        sub.arena.bias.copy_(d["bias"])
        sub.arena.edge_weight.copy_(d["edge_weight"])
    return sub, d


def donor_act_fn(sub):
    """Greedy action function in the evaluate() / world interface: (w, p) -> a."""
    @torch.no_grad()
    def act(w, p, m=sub):
        return int(m(_solo(p.unsqueeze(0)))[0].argmax())
    return act


# ----------------------------------------------------------------------
# Build all four
# ----------------------------------------------------------------------
def build_all(*, seed=0, n_mirror=8, epochs=300, collect_seeds=40, cap=900,
              eval_seeds=20, deterministic=False, do_eval=True):
    _ft.EXPLORE_DETERMINISTIC = deterministic
    print(f"=== PHASE 1: four clean primitive donors (v2.0 core) ===")
    print(f"natural-explore collection (deterministic teacher = {deterministic})  "
          f"(seed={seed}, n_mirror={n_mirror}, epochs={epochs}, cap={cap})\n")
    rows = []
    for name, spec in PRIMITIVES.items():
        P, Y = collect(spec["master"], spec["band"],
                       seeds=collect_seeds, cap=cap)
        sub, acc, chance, maj = train_donor(P, Y, seed=seed, n_mirror=n_mirror,
                                            epochs=epochs)
        path = save_donor(sub, name, seed=seed, n_mirror=n_mirror,
                          acc=acc, chance=chance, drive=spec["drive"])
        surv = None
        if do_eval:
            surv, _, _ = evaluate(donor_act_fn(sub), f"{name}-donor",
                                  seeds=eval_seeds)
        rows.append((name, spec["drive"], P.shape[0], chance, acc, surv))
        print(f"  [{name:7s}] drive={spec['drive']:11s} n={P.shape[0]:4d}  "
              f"chance={chance:.3f}  fidelity={acc:.3f}  "
              f"lift={acc - chance:+.3f}  -> {path.name}")

    print("\n=== FIDELITY TABLE ===")
    print(f"  {'primitive':9s} {'drive':12s} {'n':>5s} {'chance':>7s} "
          f"{'fidelity':>9s} {'lift':>7s} {'survival':>9s}")
    for name, drive, n, chance, acc, surv in rows:
        sv = f"{surv:.1f}" if surv is not None else "  -  "
        print(f"  {name:9s} {drive:12s} {n:5d} {chance:7.3f} "
              f"{acc:9.3f} {acc - chance:+7.3f} {sv:>9s}")
    return rows


# ----------------------------------------------------------------------
# Smoke — fast wiring proof (tiny budgets), no verdict
# ----------------------------------------------------------------------
def smoke():
    print("primitives smoke — wiring proof (tiny budgets, not a verdict)\n")
    _ft.EXPLORE_DETERMINISTIC = False        # natural exploration (collection mode)
    # [1] EVADE master is deterministic and flees a near predator
    w = TileWorld(seed=1, max_steps=300)
    w.pred = [(w.px + 1) % w.size, w.py]          # predator one step east
    a1 = evade_master(w); a2 = evade_master(w)
    print(f"[1] EVADE deterministic: a1={ACTIONS[a1]} a2={ACTIONS[a2]} "
          f"(equal & a flee-W expected)")
    assert a1 == a2 == ACTIONS.index("W"), (a1, a2)

    # [2] collect a band for each primitive (enough episodes for the rarer
    # drive-decision bands: a competent master keeps its own drive topped up)
    for name, spec in PRIMITIVES.items():
        P, Y = collect(spec["master"], spec["band"], seeds=25, cap=120)
        assert P.shape[0] > 0, name
        print(f"[2] {name:7s}: collected {P.shape[0]} band states, "
              f"{int(Y.unique().numel())} distinct actions")

    # [3] train one tiny donor end-to-end + reload roundtrip
    P, Y = collect(fire_oracle, _band_warm, seeds=6, cap=200)
    sub, acc, chance, _ = train_donor(P, Y, seed=0, epochs=15)
    path = save_donor(sub, "WARM", seed=0, n_mirror=8, acc=acc, chance=chance,
                      drive="temperature")
    sub2, d = load_donor(path)
    with torch.no_grad():
        same = torch.equal(sub(_solo(P[:8])), sub2(_solo(P[:8])))
    print(f"[3] WARM tiny donor: fidelity={acc:.3f} chance={chance:.3f}; "
          f"reload-identical={same}")
    assert same
    print("\nSMOKE PASS — masters, banded collection, donor train + persist all wire.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast wiring proof")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-mirror", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--collect-seeds", type=int, default=40)
    ap.add_argument("--cap", type=int, default=900)
    ap.add_argument("--eval-seeds", type=int, default=20)
    ap.add_argument("--no-eval", action="store_true", help="skip survival eval")
    ap.add_argument("--deterministic", action="store_true",
                    help="collect under deterministic teachers (default: natural "
                         "stochastic explore — needed for navigation demos)")
    args = ap.parse_args()
    if args.smoke:
        return smoke()
    build_all(seed=args.seed, n_mirror=args.n_mirror, epochs=args.epochs,
              collect_seeds=args.collect_seeds, cap=args.cap,
              eval_seeds=args.eval_seeds, deterministic=args.deterministic,
              do_eval=not args.no_eval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
