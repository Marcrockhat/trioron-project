"""I5 — the ARCHITECTURE TEST: the full integrated PCLL stack on the
capacity-hard taxonomy (data_hard: 32 classes × 3 modes, 12 features,
~5 disruptors, Bayes 0.937), against the s029 standalone numbers.

Everything runs through the substrate: germline-only organism →
PerceptionGenesis (tick-1 spawn, census, first sitting, NATAL REPLAY) →
council loop (one meeting per class period; stress router live on the
germline book) → dynamic signatures (per-dim counts, shadows, frame
stamps + read-time translation). Zero gradients anywhere.

Protocol (per data seed): 32 class periods of S=500 (epoch 1, discovery)
+ a second epoch (matches only — structural zero forgetting asserted),
then frozen-world eval on held-out samples:

  ONE-SHOT — single-observation matched filter: per test sample, the
    injection's pockets + per-sample frame; every class translated from
    its stamp into THAT sample's frame; argmax margin → truth via the
    discovery purity map. s029 raw arm: 0.38–0.41.
  DEAD-3   — three random sensors per sample masked (dead = no evidence,
    the raw-arm semantics). s029 raw arm grace: 0.30–0.31.

Gate asserts: 32–36 classes, purity 1.0 (splits allowed, merges not),
epoch-2 matches 32/32 correct (zero forgetting), one-shot ≥ 0.35,
dead-3 ≥ 0.25, votes conserved, all seeds.

Run: python3 -m experiments.progenitor.run_i5_architecture
"""
from __future__ import annotations

import math
from typing import Dict

import torch

from trioron.core import census, construct
from trioron.pcll import PerceptionGenesis, germline_base
from trioron.pcll.signature import translate

from .data_hard import make_spec, sample

SEEDS = 3
S_PERIOD = 500
N_TEST = 100          # held-out samples per class
N_QUANTA = 1000
K_DEAD = 3


def discover(X: torch.Tensor, y: torch.Tensor, n_class: int):
    """Full developmental flow; returns (sub, ctrl, purity map name→truth)."""
    sub = construct(germline_base, capacity=192)
    pg = PerceptionGenesis(sub)
    absorbed: Dict[str, list] = {}

    pg.feed(X[y == 0])
    sitting = sub.end_task()
    assert sitting.event == "birth", "natal period not learned"
    absorbed[sitting.class_name] = [0]
    ctrl = pg.controller

    for c in range(1, n_class):                      # epoch 1 — discovery
        ctrl.observe(X[y == c])
        r = sub.end_task()
        assert r.event != "empty", f"class {c} read empty"
        absorbed.setdefault(r.class_name, []).append(c)

    mapping = {}
    for name, truths in absorbed.items():
        assert len(set(truths)) == 1, f"{name} MERGED truths {truths}"
        mapping[name] = truths[0]

    wrong = 0
    for c in range(n_class):                         # epoch 2 — zero forgetting
        ctrl.observe(X[y == c])
        r = sub.end_task()
        wrong += not (r.event == "match" and mapping.get(r.class_name) == c)
    assert wrong == 0, f"epoch 2: {wrong} periods mis-assigned (forgetting)"

    assert abs(sum(pg.germline.votes.values()) - 28.0) < 1e-9
    assert sub.arena.edge_weight.grad is None and sub.arena.edge_cursor == 0
    return sub, ctrl, mapping


def one_shot_eval(sub, ctrl, mapping, X_te, y_te, dead=None) -> float:
    """Single-observation matched filter through the substrate injection —
    the s029 single_shot protocol exactly: raw evidence E = Re(P·T̄) argmax
    (no σ normalization), dead sensors = a FIXED masked set — plus the
    architecture's per-sample frame translation of each class template."""
    sched = sub.scheduler
    sub.forward(X_te)
    q = sched._last_receptor_q                       # [B, F] pockets
    lo, hi = sched._last_receptor_frame              # [B], [B]
    B, F = q.shape
    m = ((q > 0) & (q < N_QUANTA)).to(torch.float32)
    if dead:
        m[:, dead] = 0.0                             # dead sensor = no evidence
    theta = 2 * math.pi * q / N_QUANTA
    obs = torch.polar(m, theta)                      # masked unit phasors [B,F]

    names, evidence = [], []
    disc = ctrl.learner._discrete
    span = (hi - lo).clamp_min(1e-9)
    for c in ctrl.learner.classes:
        T = torch.where(c.active, c.T, torch.zeros_like(c.T))
        if c.frame is not None:
            lo0, hi0 = c.frame
            a = ((hi0 - lo0) / span).unsqueeze(1)                       # [B,1]
            b = (N_QUANTA * (lo0 - lo) / span).unsqueeze(1)             # [B,1]
            R = T.abs().clamp(max=1 - 1e-9).unsqueeze(0)                # [1,F]
            q_c = ((torch.angle(T) / (2 * math.pi) * N_QUANTA) % N_QUANTA)
            theta_n = 2 * math.pi * (a * q_c.unsqueeze(0) + b) / N_QUANTA
            Tb = (R ** (a * a)) * torch.exp(1j * theta_n)
            Tb = torch.where(disc.unsqueeze(0), T.unsqueeze(0).expand_as(Tb), Tb)
        else:
            Tb = T.unsqueeze(0).expand(B, F)
        evidence.append((obs * Tb.conj()).real.sum(1))
        names.append(c.name)

    pred = torch.stack(evidence, dim=1).argmax(1)
    truth_of = torch.tensor([mapping[n] for n in names])
    return float((truth_of[pred] == y_te).float().mean())


def run_seed(seed: int) -> None:
    spec = make_spec()
    tr, norm = sample(spec, n_per_class=S_PERIOD, seed=seed)
    te, _ = sample(spec, n_per_class=N_TEST, seed=100 + seed, norm=norm)
    n_class = len(tr.names)

    sub, ctrl, mapping = discover(tr.x, tr.y, n_class)
    n_classes = len(ctrl.learner.classes)
    assert n_class <= n_classes <= n_class + 4, f"{n_classes} classes"

    # the s029 scoring population: CLEAN classes only (disruptors, std > 0.3,
    # are excluded — single observations of a near-uniform class carry no
    # identity by construction)
    clean = [c for c in range(n_class) if float(spec.std[c]) <= 0.3]
    keep = torch.isin(te.y, torch.tensor(clean))
    X_te, y_te = te.x[keep], te.y[keep]

    g = torch.Generator().manual_seed(seed)
    dead = torch.randperm(12, generator=g)[:K_DEAD].tolist()

    acc = one_shot_eval(sub, ctrl, mapping, X_te, y_te)
    acc_dead = one_shot_eval(sub, ctrl, mapping, X_te, y_te, dead=dead)
    assert acc >= 0.35, f"one-shot {acc:.3f} below the s029 floor"
    assert acc_dead >= 0.25, f"dead-{K_DEAD} {acc_dead:.3f} below grace floor"

    # M1 structural baseline [D11]: the documented pre-growth anatomy.
    # 12 perception cells fixed at tick 1, zero edges, zero depth — the
    # numbers M2/M4 must move. Tighten/replace when growth lands.
    c = census(sub.arena)
    assert c.computing == 12 and c.edges == 0 and c.depth == 0, str(c)
    assert c.astrocytes == n_classes, str(c)

    print(f"seed={seed}: {n_classes} classes, purity 1.0, epoch-2 32/32 "
          f"(zero forgetting); one-shot {acc:.3f} ({len(clean)} clean classes); "
          f"dead-{K_DEAD} {acc_dead:.3f}; votes conserved; no gradients")
    print(f"  {c}")


def main() -> None:
    for seed in range(SEEDS):
        run_seed(seed)
    print("\nI5 ARCHITECTURE TEST PASS — the integrated stack matches the "
          "s029 standalone (raw arm: 32-34 classes pure, one-shot 0.38-0.41, "
          "dead-3 ~0.30)")


if __name__ == "__main__":
    main()
