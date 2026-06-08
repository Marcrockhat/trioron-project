"""Teach a TEMPORAL primitive — hysteresis as a learned skill, not a constant.

Rocky's paradigm shift (2026-06-05): do not *encode* hysteresis as a hand-tuned
routing constant. *Teach* it — as a primitive the substrate absorbs from
demonstration and carries in its own (recurrent) weights, alongside the four
drive primitives. We trust trioron to adapt/retain/learn, so we teach it instead
of building controllers around it.

The substrate fact that makes this real (not a relabel): a feed-forward donor on
the percept ALONE cannot represent hysteresis. The hysteretic temperature master
below is a two-threshold MODE LATCH (SEEK warmth < lo, FLEE heat > hi, hold the
mode in the band). The same observable state — temp ~0.5 at a given fire-distance
— maps to OPPOSITE actions depending on whether you are mid-approach (SEEK) or
mid-retreat (FLEE). Distance is observable; its derivative / the latched mode is
NOT. A memoryless map percept->action can only emit one action at that point, so
it must overheat or freeze. A recurrent donor that carries a leaky trace across
steps (the temporal/Axis-7 gene, validated in bench_temporal_gate) CAN recover
the mode and act correctly.

This module is the clean test of the paradigm. ONE substrate, ONE training loop
(sequential, BPTT over windows). The only difference between the two arms is
`op.hold`:
  - temporal-OFF (op.hold=False): the trace resets every step -> the honest
    feed-forward equivalent. It should overheat (cannot latch the mode).
  - temporal-ON  (op.hold=True):  intrinsic memory persists -> it should learn
    the latch and stop overheating.

If ON breaks the overheat ceiling and OFF does not, the missing piece was
temporal representational capacity, and "teach hysteresis as a primitive" is the
right move (next: slot the recurrent WARM into the vocabulary organism).

Usage:
    python3 -m experiments.world.temporal_warm --smoke
    python3 -m experiments.world.temporal_warm            # ON vs OFF verdict
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import experiments.world.fire_taming as _ft               # noqa: E402  (pins WARM_RATE=0.04)
from experiments.world.fire_taming import _toward, _away_from_fire  # noqa: E402
from experiments.world.tile_world import TileWorld, ACTIONS, N_ACTION, FIRE  # noqa: E402
from experiments.world.mirror_cells import _solo, INPUT_DIM  # noqa: E402

from trioron.core import Envelope, construct                # noqa: E402
from trioron.core.epigenome import OUTPUT, has_gene         # noqa: E402
from trioron.bases import seeded                            # noqa: E402
from trioron.phenotype import default_dispatch_table        # noqa: E402
from trioron.core.epigenome import RECURRENT                # noqa: E402
from trioron.learning.manifold import get_interior_ids      # noqa: E402
from experiments.satellites_v1 import SatelliteOp, add_satellite  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent.parent / "runs"
PRIM_DIR = RUNS / "primitives"
_REST = ACTIONS.index("rest")


def _p(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------
# The hysteretic temperature master — a genuinely TEMPORAL skill
# ----------------------------------------------------------------------
class HystereticWarm:
    """Two-threshold MODE LATCH. SEEK warmth (toward fire) below `lo`; FLEE heat
    (away from fire) above `hi`; HOLD the current mode between them. The hold band
    is the hysteresis — and it makes the policy memory-dependent: at temp in
    (lo,hi) the action depends on the latched mode, not the observable state."""

    def __init__(self, lo=0.42, hi=0.58):
        self.lo, self.hi = lo, hi
        self.mode = "SEEK"

    def reset(self):
        self.mode = "SEEK"

    def __call__(self, w):
        if w.temp <= self.lo:
            self.mode = "SEEK"
        elif w.temp >= self.hi:
            self.mode = "FLEE"
        if self.mode == "SEEK":
            t = _toward(w, FIRE)
            return t if t is not None else _REST
        return _away_from_fire(w)


# ----------------------------------------------------------------------
# Collect TRAJECTORIES (ordered, per-episode) — the recurrent donor needs the
# temporal continuity, not band-filtered scattered frames.
# ----------------------------------------------------------------------
@torch.no_grad()
def collect_trajectories(master, *, seeds=80, max_steps=300, min_len=24):
    """Run the (stateful) master as sole controller; record each episode's
    ordered (percept[T,77], action[T]). Reset the master's latch per episode."""
    trajs = []
    for ep in range(seeds):
        w = TileWorld(seed=ep * 53 + 11, max_steps=max_steps)
        master.reset()
        p = w.percept(); done = False
        P, Y = [], []
        while not done:
            a = master(w)
            P.append(p); Y.append(a)
            p, _, done, _ = w.step(a)
        if len(P) >= min_len:
            trajs.append((torch.stack(P), torch.tensor(Y)))
    if not trajs:
        raise ValueError("no trajectories of min_len collected; raise seeds")
    return trajs


def make_windows(trajs, window, stride=None):
    """[(P[T,77],Y[T])] -> windows [Nw,window,77],[Nw,window] (strided)."""
    stride = stride or window // 2
    Pw, Yw = [], []
    for P, Y in trajs:
        T = P.shape[0]
        for s in range(0, T - window + 1, stride):
            Pw.append(P[s:s + window]); Yw.append(Y[s:s + window])
    return torch.stack(Pw), torch.stack(Yw)


# ----------------------------------------------------------------------
# Build a recurrent WARM donor — seeded core + recurrent (satellite) memory cells
# wired interior -> recurrent -> output. The trace lives in the SatelliteOp.
# ----------------------------------------------------------------------
def build_recurrent_donor(seed, *, n_rec=12, lam=0.6, interior=32,
                          cap=400_000, capacity=2048):
    torch.manual_seed(seed)
    op = SatelliteOp(lam=lam, w_res=0.0)              # leaky trace, no resource term
    table = default_dispatch_table()
    table[RECURRENT] = op
    sub = construct(
        base=seeded(INPUT_DIM, N_ACTION, interior_cells=interior),
        envelope=Envelope(max_parameter_bytes=cap),
        dispatch_table=table, capacity=capacity, sparsity_k=0,
    )
    sub.compile()
    a = sub.arena
    interior_ids = get_interior_ids(a).long()
    out_ids = torch.tensor(
        [c for c in a.alive_ids().tolist()
         if has_gene(int(a.epigenome[c].item()), OUTPUT)], dtype=torch.long)
    rec_ids = sorted(add_satellite(a, interior_ids, out_ids) for _ in range(n_rec))
    a.rank_dirty = True
    sub.compile()
    sub.prepare_training()
    return sub, op, rec_ids


# ----------------------------------------------------------------------
# Train — sequential BPTT over windows; `hold` toggles the memory (the only knob
# that differs between the two arms).
# ----------------------------------------------------------------------
def train(trajs, *, hold, seed=0, n_rec=12, lam=0.6, epochs=120, lr=3e-3,
          window=24, batch=64, warmup=6):
    sub, op, rec_ids = build_recurrent_donor(seed, n_rec=n_rec, lam=lam)
    op.hold = hold
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    ce = torch.nn.functional.cross_entropy
    Pw, Yw = make_windows(trajs, window)
    n = Pw.shape[0]
    g = torch.Generator().manual_seed(seed + 7)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            B = idx.numel()
            op.reset(B, len(rec_ids))
            loss = 0.0
            for t in range(window):
                logits = sub(_solo(Pw[idx, t]))
                if t >= warmup:                        # let the trace fill first
                    loss = loss + ce(logits, Yw[idx, t])
            loss = loss / (window - warmup)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
            sub.zero_dormant_grads(); opt.step()
    return sub, op, rec_ids


# ----------------------------------------------------------------------
# Evaluate the donor as the SOLE temperature controller. The metric that matters
# is the death-cause breakdown: a good regulator dies of thirst/energy (it
# ignores them), NOT of overheat/cold. Reset the trace per episode.
# ----------------------------------------------------------------------
def _cause(w):
    if w.alive:
        return "timeout"
    if w.temp <= w.temp_low:
        return "cold"
    if w.temp >= w.temp_high:
        return "overheat"
    if w.thirst <= 0:
        return "thirst"
    if w.energy <= 0:
        return "energy"
    return "integrity"


@torch.no_grad()
def eval_controller(sub, op, n_rec, *, seeds=40, max_steps=300):
    surv, causes, temps = [], Counter(), []
    for ep in range(seeds):
        w = TileWorld(seed=ep * 7000 + 7, max_steps=max_steps)
        if op is not None:
            op.reset(1, n_rec)
        p = w.percept(); done = False
        while not done:
            a = int(sub(_solo(p.unsqueeze(0)))[0].argmax())
            p, _, done, info = w.step(a)
            temps.append(float(w.temp))
        surv.append(info["t"])
        causes[_cause(w)] += 1
    return statistics.mean(surv), causes, statistics.pstdev(temps)


# ----------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------
def run(*, seeds=3, eval_seeds=40, collect_seeds=80, epochs=120, n_rec=12,
        lam=0.6, window=24):
    _ft.EXPLORE_DETERMINISTIC = False
    master = HystereticWarm()
    trajs = collect_trajectories(master, seeds=collect_seeds)
    nwin = make_windows(trajs, window)[0].shape[0]
    _p(f"collected {len(trajs)} trajectories -> {nwin} windows (len {window})\n")

    # sanity: the master itself
    m_surv, m_causes, _ = eval_controller_master(master, seeds=eval_seeds)
    _p(f"[master HystereticWarm] survival {m_surv:.1f}   "
       f"overheat {m_causes['overheat']}/{eval_seeds}   deaths {dict(m_causes.most_common())}\n")

    rows = {}
    for hold, name in [(False, "temporal-OFF (memoryless = feed-forward)"),
                       (True,  "temporal-ON  (recurrent hysteresis)")]:
        survs, oh, cold = [], [], []
        last_causes = None
        for s in range(seeds):
            sub, op, rec_ids = train(trajs, hold=hold, seed=s, n_rec=n_rec,
                                     lam=lam, epochs=epochs, window=window)
            sv, c, _ = eval_controller(sub, op, len(rec_ids), seeds=eval_seeds)
            survs.append(sv); oh.append(c["overheat"]); cold.append(c["cold"])
            last_causes = c
        rows[hold] = (survs, oh, cold)
        mo = statistics.mean(oh)
        _p(f"[{name}]")
        _p(f"   survival {statistics.mean(survs):.1f} (n={seeds})   "
           f"overheat {mo:.1f}/{eval_seeds}   cold {statistics.mean(cold):.1f}/{eval_seeds}")
        _p(f"   last-seed deaths {dict(last_causes.most_common())}\n")

    # hysteresis = staying inside the temp band; failing it is cold OR overheat.
    off_th = statistics.mean(o + c for o, c in zip(rows[False][1], rows[False][2]))
    on_th = statistics.mean(o + c for o, c in zip(rows[True][1], rows[True][2]))
    off_oh = statistics.mean(rows[False][1])
    on_oh = statistics.mean(rows[True][1])
    _p("=== VERDICT ===")
    _p(f"    THERMAL death (cold+overheat)  OFF {off_th:.1f}  ->  ON {on_th:.1f}   "
       f"(reduction {off_th - on_th:+.1f}) / {eval_seeds}")
    _p(f"    of which overheat              OFF {off_oh:.1f}  ->  ON {on_oh:.1f}")
    _p(f"    survival                       OFF {statistics.mean(rows[False][0]):.1f}  ->  "
       f"ON {statistics.mean(rows[True][0]):.1f}")
    _p("    PASS = memory (ON) cuts thermal death the feed-forward (OFF) cannot.")
    return rows


@torch.no_grad()
def eval_controller_master(master, *, seeds=40, max_steps=300):
    surv, causes = [], Counter()
    for ep in range(seeds):
        w = TileWorld(seed=ep * 7000 + 7, max_steps=max_steps)
        master.reset()
        p = w.percept(); done = False
        while not done:
            a = master(w)
            p, _, done, info = w.step(a)
        surv.append(info["t"]); causes[_cause(w)] += 1
    return statistics.mean(surv), causes, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval-seeds", type=int, default=40)
    ap.add_argument("--collect-seeds", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--n-rec", type=int, default=12)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--window", type=int, default=24)
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.eval_seeds, args.collect_seeds, args.epochs = 1, 8, 20, 12

    _p("=== TEMPORAL WARM — teach hysteresis as a recurrent primitive ===")
    _p("    ON vs OFF differ ONLY in op.hold (intrinsic memory)\n")
    run(seeds=args.seeds, eval_seeds=args.eval_seeds,
        collect_seeds=args.collect_seeds, epochs=args.epochs,
        n_rec=args.n_rec, lam=args.lam, window=args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
