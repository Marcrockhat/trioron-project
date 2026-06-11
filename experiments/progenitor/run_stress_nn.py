"""Stress test: does the periodic discovery loop CONVERGE through a deterministic NN?

The capacity-hard taxonomy (data_hard: 32 classes x 3 modes, ~5 wide disruptors,
12 feats, Bayes 0.937) is streamed as class-periods (one period = S draws of one
class, modes resampled per draw — the recurring mixture IS the class signature).
Arms:

  raw — receptor reads the 12 standardized features directly (control)
  nn  — receptor reads H=16 units of a FROZEN random tanh MLP (deterministic,
        never trained, no label leakage; any fixed perception stack behaves the
        same in kind — class-conditional pushforwards are stationary, so the
        signature statistics carry over)

Stages, all label-free except evaluation:

  GENESIS    input layer formed by partitioning to informative units: keep units
             that (a) lock within periods (median margin > K) and (b) MOVE between
             periods (phase dispersion > DISP_FLOOR — a unit that never varies is
             DC, one that never locks is noise). genesis.py's apoptosis->saliency
             pattern, computed purely from period statistics.
  DISCOVERY  ScheduleLearner over R rounds x 32 shuffled class-periods. While
             reading only genesis units, a SHADOW accumulator keeps full-width
             signatures per learned class ("sensed but not read").
  EVAL       convergence (births plateau), classes found, purity, stability,
             disruptor handling (wide => below floor => EMPTY, the frustration-
             gate finding rediscovered), single-shot accuracy (one observation,
             matched filter on learned templates; honest caveat: templates are
             unimodal, classes are 3-mode mixtures).
  DEAD       incomplete input: 3 raw sensors die (read 0). raw arm masks the dead
             channels (no evidence != wrong evidence); the dense nn arm cannot —
             the partitioning argument for genesis, measured.
  RECRUIT    discovery re-run reading only the TOP-6 units, then dropped units
             are added back one at a time, templates extended from shadows —
             self-recovery from ambiguity WITHOUT relearning.

Run: python3 run_stress_nn.py   (~1 min, CPU)
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch

from data_hard import make_spec, sample
from lockin import LockIn, evidence_mask, matched_k
from receptor import N_QUANTA, quantize
from schedule_learn import LearnedClass, ScheduleLearner

SEEDS = 3
S_PERIOD = 500          # observations per period
ROUNDS = 8              # discovery rounds (each = all 32 classes, shuffled)
H_NN = 16
DISP_FLOOR = 0.05       # genesis between-period dispersion gate
N_DEAD = 3              # dead raw sensors in the incomplete-input probe


class FrozenNN:
    """Deterministic frozen 12->24->16 tanh MLP (seeded, never trained)."""

    def __init__(self, d_in: int, seed: int, h1: int = 24, h2: int = H_NN):
        g = torch.Generator().manual_seed(seed)
        s1, s2 = 1.5 / math.sqrt(d_in), 1.5 / math.sqrt(h1)
        self.W1 = torch.randn(d_in, h1, generator=g) * s1
        self.b1 = torch.randn(h1, generator=g) * 0.1
        self.W2 = torch.randn(h1, h2, generator=g) * s2
        self.b2 = torch.randn(h2, generator=g) * 0.1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.tanh(x @ self.W1 + self.b1) @ self.W2 + self.b2)


def draw_class(spec, norm, c: int, n: int, g: torch.Generator,
               mode: Optional[int] = None, dead: Optional[List[int]] = None) -> torch.Tensor:
    """n raw draws of class c (one mode if given), standardized with the shared
    norm. Dead sensors read RAW 0 (standardized to -mu/sigma) before any front-end."""
    K, M, D = spec.means.shape
    mi = torch.full((n,), mode, dtype=torch.long) if mode is not None \
        else torch.randint(0, M, (n,), generator=g)
    x = spec.means[c, mi] + torch.randn(n, D, generator=g) * float(spec.std[c])
    if dead:
        x[:, dead] = 0.0
    return (x - norm[0]) / norm[1]


def to_lockin(feats: torch.Tensor, read: torch.Tensor) -> LockIn:
    q = quantize(feats)
    li = LockIn(feats.shape[1])
    li.absorb(2 * math.pi * q / N_QUANTA, mask=evidence_mask(q) & read)
    return li


def signature(li: LockIn) -> torch.Tensor:
    A = torch.complex(li.re, li.im)
    return torch.where(li.n > 0, A / li.n.clamp_min(1.0), torch.zeros_like(A))


# ---------------------------------------------------------------- genesis

def genesis(spec, norm, front, g: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """One unlabeled period per class -> (read mask, unit scores)."""
    H = front(torch.zeros(1, spec.means.shape[2])).shape[1]
    margins, phases = [], []
    order = torch.randperm(spec.means.shape[0], generator=g).tolist()
    for c in order:
        li = to_lockin(front(draw_class(spec, norm, c, S_PERIOD, g)),
                       torch.ones(H, dtype=torch.bool))
        margins.append(li.margin())
        S = signature(li)
        phases.append(torch.where(S.abs() > 0, S / S.abs().clamp_min(1e-9),
                                  torch.zeros_like(S)))
    M = torch.stack(margins)                      # (P, H)
    P = torch.stack(phases)                       # (P, H) unit phasors
    coherent = M.median(0).values > matched_k()
    locked = M > matched_k()                      # use only periods where the unit locked
    disp = torch.ones(H)
    for f in range(H):
        ps = P[locked[:, f], f]
        if len(ps) >= 3:
            disp[f] = 1.0 - ps.mean(0).abs()
    score = M.median(0).values * disp
    read = coherent & (disp > DISP_FLOOR)
    if read.sum() < 4:                            # safety floor
        read = score >= score.topk(6).values[-1]
    return read, score


# ---------------------------------------------------------------- discovery

def discover(spec, norm, front, read: torch.Tensor, g: torch.Generator,
             per_mode: bool = False):
    """Unsupervised discovery over class- (or mode-) periods + shadow signatures."""
    K, M, _ = spec.means.shape
    H = len(read)
    learner = ScheduleLearner(H)
    shadow: Dict[str, List] = {}                  # name -> [sum_sig (H,) complex, count]
    log: List[Tuple[int, str, Optional[str]]] = []
    units = [(c, m) for c in range(K) for m in range(M)] if per_mode \
        else [(c, None) for c in range(K)]

    for _ in range(ROUNDS if not per_mode else ROUNDS // 2):
        order = torch.randperm(len(units), generator=g).tolist()
        for i in order:
            c, mode = units[i]
            feats = front(draw_class(spec, norm, c, S_PERIOD, g, mode=mode))
            li = to_lockin(feats, read)
            event, name = learner.observe(li)
            log.append((c, event, name))
            if name is not None:                  # shadow: full-width signature
                li_full = to_lockin(feats, torch.ones(H, dtype=torch.bool))
                s = shadow.setdefault(name, [torch.zeros(H, dtype=torch.complex64), 0])
                s[0] += signature(li_full)
                s[1] += 1
    return learner, shadow, log


def discovery_metrics(log, spec, n_periods_per_round: int):
    disrupt = set((spec.std > 0.3).nonzero().flatten().tolist())
    absorbed: Dict[str, Counter] = {}
    last_births = 0
    for i, (c, event, name) in enumerate(log):
        if event == "birth" and i >= len(log) // 2:
            last_births += 1
        if name is not None:
            absorbed.setdefault(name, Counter())[c] += 1
    purity_n = purity_d = 0
    mapping: Dict[str, int] = {}
    for name, cnt in absorbed.items():
        top, n = cnt.most_common(1)[0]
        mapping[name] = top
        purity_n += n
        purity_d += sum(cnt.values())
    dis_periods = [(c, e) for c, e, _ in log if c in disrupt]
    dis_empty = sum(e == "empty" for _, e in dis_periods)
    coverage = len(set(mapping.values()) - disrupt)
    return dict(n_classes=len(absorbed), coverage=coverage,
                purity=purity_n / max(purity_d, 1),
                converged=last_births == 0, late_births=last_births,
                dis_empty=dis_empty / max(len(dis_periods), 1),
                mapping=mapping)


# ---------------------------------------------------------------- evaluation

def single_shot(learner: ScheduleLearner, mapping: Dict[str, int], spec, norm, front,
                read: torch.Tensor, seed: int, dead: Optional[List[int]] = None,
                mask_dead_raw: bool = False) -> float:
    """One-observation matched-filter accuracy on held-out singles (clean classes
    with a learned counterpart). Labels used ONLY here, for scoring."""
    g = torch.Generator().manual_seed(seed)
    K = spec.means.shape[0]
    clean = [c for c in range(K) if float(spec.std[c]) <= 0.3 and c in mapping.values()]
    xs, ys = [], []
    for c in clean:
        xs.append(draw_class(spec, norm, c, 64, g, dead=dead))
        ys.append(torch.full((64,), c))
    feats = front(torch.cat(xs))
    y = torch.cat(ys)
    q = quantize(feats)
    m = evidence_mask(q) & read
    if mask_dead_raw and dead:                    # raw arm: dead sensor = no evidence
        m[:, dead] = False
    P = m.to(torch.complex64) * torch.exp(1j * 2 * math.pi * q / N_QUANTA)
    names = [c.name for c in learner.classes]
    T = torch.stack([c.T for c in learner.classes])          # (C, H)
    E = (P @ T.conj().t()).real
    pred = torch.tensor([mapping[names[i]] for i in E.argmax(1).tolist()])
    return (pred == y).float().mean().item()


def recruit_curve(learner, shadow, mapping, spec, norm, front, read_small,
                  score, seed: int) -> List[Tuple[int, float]]:
    """Add dropped units back (best score first), extending templates from the
    shadows — no relearning — and re-measure single-shot accuracy."""
    read = read_small.clone()
    curve = [(int(read.sum()), single_shot(learner, mapping, spec, norm, front, read, seed))]
    dropped = sorted((~read).nonzero().flatten().tolist(),
                     key=lambda u: -score[u].item())
    for u in dropped:
        read[u] = True
        for c in learner.classes:
            sig_sum, cnt = shadow[c.name]
            c.T[u] = sig_sum[u] / max(cnt, 1)
            c.active[u] = True
        curve.append((int(read.sum()), single_shot(learner, mapping, spec, norm, front, read, seed)))
    return curve


# ---------------------------------------------------------------- main

def run_arm(tag: str, spec, norm, front, seed: int, per_mode: bool = False) -> None:
    g = torch.Generator().manual_seed(10 + seed)
    read, score = genesis(spec, norm, front, g)
    learner, shadow, log = discover(spec, norm, front, read, g, per_mode=per_mode)
    m = discovery_metrics(log, spec, 32)
    acc = single_shot(learner, m["mapping"], spec, norm, front, read, 900 + seed)
    dead = random.Random(seed).sample(range(spec.means.shape[2]), N_DEAD)
    acc_dead = single_shot(learner, m["mapping"], spec, norm, front, read, 900 + seed,
                           dead=dead, mask_dead_raw=(tag == "raw"))
    conv = "CONVERGED" if m["converged"] else f"{m['late_births']} late births"
    print(f"{tag:8s} seed={seed}: read {int(read.sum())}/{len(read)} units | "
          f"{conv} | "
          f"classes {m['n_classes']} (clean coverage {m['coverage']}/27) "
          f"purity {m['purity']:.3f} | disruptor-empty {m['dis_empty']:.2f} | "
          f"1-shot {acc:.3f} -> dead-{N_DEAD}-sensors {acc_dead:.3f}")


def main() -> None:
    spec = make_spec()                            # canonical: 32c x 3m, Bayes 0.937
    _, norm = sample(spec, 128, seed=0)
    D = spec.means.shape[2]
    identity = lambda x: x

    print(f"stress test on capacity-hard taxonomy (32cx3m, ~5 disruptors, "
          f"S={S_PERIOD}, {ROUNDS} rounds)")
    for seed in range(SEEDS):
        run_arm("raw", spec, norm, identity, seed)
    for seed in range(SEEDS):
        run_arm("nn", spec, norm, FrozenNN(D, 1000 + seed), seed)

    print("\nmode-periods (one period = one mode; the 96-modes reality, unsupervised):")
    for seed in range(2):
        run_arm("nn-mode", spec, norm, FrozenNN(D, 1000 + seed), seed, per_mode=True)

    print("\nrecruit-on-ambiguity (nn, read only top-6 units, then add back from shadows):")
    seed = 0
    g = torch.Generator().manual_seed(10 + seed)
    front = FrozenNN(D, 1000 + seed)
    read_full, score = genesis(spec, norm, front, g)
    read6 = torch.zeros_like(read_full)
    read6[score.topk(6).indices] = True
    learner, shadow, log = discover(spec, norm, front, read6, g)
    m = discovery_metrics(log, spec, 32)
    curve = recruit_curve(learner, shadow, m["mapping"], spec, norm, front,
                          read6, score, 900 + seed)
    print(f"  discovery on 6 units: classes {m['n_classes']} purity {m['purity']:.3f}")
    print("  units -> 1-shot acc: " +
          "  ".join(f"{n}:{a:.3f}" for n, a in curve))


if __name__ == "__main__":
    main()
