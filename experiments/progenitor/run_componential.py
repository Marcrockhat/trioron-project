"""Componential-semantics readout — the s029 backlog item Rocky named
(design receptor_period_frustration.md §10 "UNBUILT"), built s034.

A discovered class is described RELATIONALLY against its neighbour
species (Saussurean valeur). KEY FACT making this well-posed: buffer
templates are PHASOR MEANS, and the mean of a mixture is LINEAR in its
constituents — a 50/50 chicken-duck buffer's template sits exactly at
the midpoint of the pure chicken and duck templates. So the readout is
a MIXTURE DECOMPOSITION over species prototypes:

    T_k  ~=  sum_c beta_c . S_c     (beta >= 0, sum beta = 1)

beta_A ~ 1 -> a pure A; beta_B >= BETA_TH -> a relational class
("dog-goat", "duck-ish chicken"). The first cut used the signed
projection alpha_B = Re<T_k - S_A, S_B - S_A>/||S_B - S_A||^2 and
FAILED (precision 0.00, lift = random control): prototypes built from
ALL classes of a species are contaminated by the very blend being
described (a blend that is its species' only class IS the prototype —
the projection vanishes identically). v2 therefore uses:

  - PURE prototypes: S_c from classes with purity >= PURE_TH only,
    weighted by their majority-member count;
  - LEAVE-ONE-OUT: class k is removed from any prototype it feeds
    before describing k;
  - classes whose species has no OTHER pure class are honestly
    UNNAMEABLE (the vocabulary lacks the constituent — you cannot say
    "duck-ish" if duck was never learned as a distinct concept).

Two questions (Rocky, s034):

 1. ACCURACY IF WE ALLOW SUCH CLASSES — relational scoring credits a
    prediction whose label set {A} + modifiers contains the truth.
    Controls: (i) random-modifier sets of IDENTICAL sizes (multi-label
    scoring inflates mechanically — the lift must beat that), (ii) the
    membership-composition oracle ({c : >= 20% of the buffer}) — the
    ceiling relational labels could reach if they perfectly reflected
    buffer composition.
 2. DOES IT MAKE SENSE — (i) do beta-predicted blend partners match
    the class's true member composition (precision/recall vs the
    oracle's extra labels), (ii) named descriptors on the 10-animal
    taxonomy read correctly ("goat-ish chicken, not pig").

MEASURED VERDICT (s034, 3 seeds data_hard + 3-epoch taxonomy):

 * THE CLASSES EXIST, UNSUPERVISED — Rocky's predicted modifier
   classes appear and are stable: taxonomy u-classes at
   chicken:0.49/duck:0.48, goat:0.61/dog:0.39, cat:0.81/dog:0.19
   (unchanged across 3 epochs); data_hard 14-16 true blends per seed
   among ~77 classes.
 * ALLOWING THEM IS WORTH +0.13/+0.16 — scoring a prediction correct
   when the sample's species genuinely belongs to the predicted
   class's composition (>= 20%): data_hard 0.829 -> 0.957, taxonomy
   0.785 -> 0.943. The blend classes sit where the world truly
   overlaps; most of their "errors" are an artifact of forcing one
   species name per class.
 * NO LABEL-FREE READOUT RECOVERS THE NAMES — three falsifications:
   alpha projection precision 0.00 (prototype self-contamination);
   beta mixture precision 0.00 (incoherent disruptor pollution
   averages to ZERO phasor — invisible in template space; prototypes
   mode-smeared on multimodal worlds); buffer crossmatch precision
   0.02 (consolidation already moved every member the matched filter
   would place elsewhere — the residual pollution is exactly what the
   organism's own geometry cannot see). All relational lifts <= the
   random-modifier control. This is the s033 self-labeling law in
   semantic form: A BUFFER'S BLEND IDENTITY IS LABEL INFORMATION.
 * THE VOCABULARY GAP IS STRUCTURAL — "chicken-duck" needs duck as a
   separate concept; division never separates them (3 epochs) because
   the percept genuinely doesn't: the organism's single class IS the
   correct perceptual taxon, and the two names are human nomenclature
   finer than the features support (the perceptual-taxonomy ruling).

Run: python3 -m experiments.progenitor.run_componential
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import torch

from trioron.core import construct
from trioron.core.receptor import N_QUANTA
from trioron.pcll import MixedStreamController, PerceptionGenesis, germline_base

from .data_taxonomy import make_taxonomy
from .run_m2_mixed import SEEDS, WINDOW
from .run_m5_manifold import grow

BETA_TH = 0.20      # mixture weight to count as a blend partner "+B"
PURE_TH = 0.60      # min purity for a class to feed a species prototype
MAX_MODS = 2        # cap modifiers (a class blends 2-3 things, not 31)
ORACLE_FRAC = 0.20  # oracle blend label: species holding >= 20% of the buffer
R_CONTROL = 50      # random-modifier draws for the inflation control
NEG_TH = 0.02       # beta below this for the most-similar OTHER species -> "not X"
CROSS_TH = 0.10     # buffer-member foreign fraction to count as "+B"
TAX_EPOCHS = 3      # taxonomy stream epochs (division pressure on blends)


def phasors(mixed, X: torch.Tensor) -> torch.Tensor:
    return torch.exp(1j * 2 * math.pi * mixed.pockets_of(X) / N_QUANTA)


def composition(mixed, Xs, ys, n_truth: int):
    """Templates + per-class member counts over truths (the eval-side
    self-labeling every gate uses, kept byte-identical to mapping_of:
    raw-E argmax membership)."""
    T = mixed.templates()
    Z = phasors(mixed, Xs)
    member = (Z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    counts = torch.zeros(len(T), n_truth)
    for k in range(len(T)):
        t = ys[member == k]
        if len(t):
            counts[k] = torch.bincount(t, minlength=n_truth).float()
    return T, counts


def pure_prototype_feeds(counts: torch.Tensor) -> Dict[int, List[Tuple[int, float]]]:
    """species c -> [(class k, weight)] over classes pure enough to
    define c (purity >= PURE_TH); weight = majority-member count."""
    tot = counts.sum(1).clamp_min(1)
    truth_of = counts.argmax(1)
    feeds: Dict[int, List[Tuple[int, float]]] = {}
    for k in range(len(counts)):
        if tot[k] < 5:
            continue
        c = int(truth_of[k])
        pur = float(counts[k, c] / tot[k])
        if pur >= PURE_TH:
            feeds.setdefault(c, []).append((k, float(counts[k, c])))
    return feeds


def loo_prototypes(T: torch.Tensor, feeds, k_exclude: int):
    """Species prototypes with class k removed from whatever it feeds.
    Returns (S dict: c -> prototype)."""
    S: Dict[int, torch.Tensor] = {}
    for c, lst in feeds.items():
        kept = [(k, w) for k, w in lst if k != k_exclude]
        if not kept:
            continue
        w = torch.tensor([wi for _, wi in kept])
        w = (w / w.sum()).to(T.dtype)
        S[c] = (T[torch.tensor([k for k, _ in kept])] * w.unsqueeze(1)).sum(0)
    return S


def beta_mixture(Tk: torch.Tensor, S: Dict[int, torch.Tensor],
                 n_cand: int = 6) -> Dict[int, float]:
    """Nonnegative mixture decomposition T_k ~= sum beta_c S_c over the
    n_cand most similar prototypes (complex LS as stacked real LS;
    active-set: drop negative supports, re-solve)."""
    if not S:
        return {}
    cs = list(S.keys())
    sims = torch.tensor([float((Tk * S[c].conj()).real.sum()) for c in cs])
    cs = [cs[i] for i in sims.argsort(descending=True)[:n_cand].tolist()]
    y = torch.cat([Tk.real, Tk.imag])
    active = list(cs)
    beta = {}
    for _ in range(len(cs)):
        A = torch.stack([torch.cat([S[c].real, S[c].imag])
                         for c in active], dim=1)
        sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
        if bool((sol >= -1e-6).all()):
            beta = {c: max(0.0, float(b)) for c, b in zip(active, sol)}
            break
        active = [c for c, b in zip(active, sol) if float(b) > 0]
        if not active:
            return {}
    tot = sum(beta.values())
    return {c: b / tot for c, b in beta.items()} if tot > 1e-6 else {}


def componential(T, counts, feeds, th: float):
    """Per class: (label set, modifiers desc-sorted by beta, the
    'not X' species or None, nameable?). A class is UNNAMEABLE when
    its own species has no other pure class (no LOO prototype)."""
    truth_of = torch.where(counts.sum(1) > 0,
                           counts.argmax(1), torch.full((len(T),), -1))
    sets: List[Set[int]] = []
    mods: List[List[Tuple[float, int]]] = []
    negs: List[Optional[int]] = []
    nameable: List[bool] = []
    for k in range(len(T)):
        A = int(truth_of[k])
        if A < 0:
            sets.append(set()); mods.append([]); negs.append(None)
            nameable.append(False)
            continue
        S = loo_prototypes(T, feeds, k)
        if A not in S:
            sets.append({A}); mods.append([]); negs.append(None)
            nameable.append(False)
            continue
        beta = beta_mixture(T[k], S)
        pos = sorted(((b, c) for c, b in beta.items()
                      if c != A and b >= th), reverse=True)[:MAX_MODS]
        # "not X": the species most SIMILAR to this class among those
        # the decomposition assigns ~zero weight — the informative
        # exclusion ("chicken-duck but NOT pig", where pig is nearby)
        neg = None
        near = sorted(((float((T[k] * S[c].conj()).real.sum()), c)
                       for c in S if c != A and beta.get(c, 0.0) < NEG_TH),
                      reverse=True)
        if near:
            neg = near[0][1]
        sets.append({A} | {c for _, c in pos})
        mods.append(pos)
        negs.append(neg)
        nameable.append(True)
    return truth_of, sets, mods, negs, nameable


def crossmatch(mixed, truth_of: torch.Tensor, n_species: int,
               th: float):
    """v3 — label-free composition from the organism's OWN buffers:
    re-score each class's buffered members against every template; a
    member is FOREIGN when some other class matches it better (argmax
    != home) — its species (via the self-label map) is a blend
    partner. Template arithmetic cannot see incoherent pollution (it
    averages to zero phasor); the members still carry it. Returns
    (label sets, weights list)."""
    T = mixed.templates()
    sets: List[Set[int]] = []
    wts: List[Dict[int, float]] = []
    for k, b in enumerate(mixed.bufs):
        A = int(truth_of[k]) if k < len(truth_of) else -1
        if A < 0 or not len(b):
            sets.append({A} if A >= 0 else set()); wts.append({})
            continue
        E = (b.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1)
        j = E.argmax(1)
        foreign = j != k
        w: Dict[int, float] = {}
        if bool(foreign.any()):
            sp = truth_of[j[foreign]]
            for c in sp[sp >= 0].unique().tolist():
                w[int(c)] = float((sp == c).float().sum() / len(b))
        mods = sorted(((wc, c) for c, wc in w.items()
                       if c != A and wc >= th), reverse=True)[:MAX_MODS]
        sets.append({A} | {c for _, c in mods})
        wts.append(w)
    return sets, wts


def rel_acc(pred: torch.Tensor, y: torch.Tensor,
            sets: List[Set[int]]) -> float:
    return float(sum(int(y[i]) in sets[int(pred[i])]
                     for i in range(len(y))) / len(y))


def control_acc(pred, y, truth_of, sets, n_species, seed: int) -> float:
    """The inflation control: same set SIZES, modifiers drawn at random."""
    g = torch.Generator().manual_seed(7000 + seed)
    accs = []
    for _ in range(R_CONTROL):
        rs = []
        for k in range(len(sets)):
            A = int(truth_of[k])
            n_extra = max(0, len(sets[k]) - 1)
            s = {A} if A >= 0 else set()
            if n_extra:
                cand = torch.tensor([c for c in range(n_species) if c != A])
                pick = cand[torch.randperm(len(cand), generator=g)[:n_extra]]
                s |= {int(b) for b in pick}
            rs.append(s)
        accs.append(rel_acc(pred, y, rs))
    return sum(accs) / len(accs)


def oracle_sets(counts: torch.Tensor) -> List[Set[int]]:
    tot = counts.sum(1, keepdim=True).clamp_min(1)
    truth_of = counts.argmax(1)
    return [set((counts[k] / tot[k] >= ORACLE_FRAC).nonzero()
                .squeeze(1).tolist()) |
            ({int(truth_of[k])} if counts[k].sum() > 0 else set())
            for k in range(len(counts))]


# ── Part 1: data_hard, the M2 default stack (0.829-comparable) ─────

def part1_seed(seed: int):
    sub, mixed, (Xs, ys), (Xt, yt), spec = grow(seed, manifold=False)
    C = len(spec.names)
    T, counts = composition(mixed, Xs, ys, C)
    feeds = pure_prototype_feeds(counts)
    truth_of, sets, mods, negs, nameable = componential(T, counts, feeds,
                                                        BETA_TH)

    Z = phasors(mixed, Xt)
    pred = (Z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    strict = float((truth_of[pred] == yt).float().mean())
    rel = rel_acc(pred, yt, sets)
    ctrl = control_acc(pred, yt, truth_of, sets, C, seed)
    osets = oracle_sets(counts)
    oracle = rel_acc(pred, yt, osets)
    xsets, xwts = crossmatch(mixed, truth_of, C, CROSS_TH)
    xrel = rel_acc(pred, yt, xsets)
    foreign = sum(sum(w.values()) for w in xwts) / max(1, len(xwts))

    def pr(label_sets):
        tp = fp = fn = 0
        for k in range(len(T)):
            comp_extra = label_sets[k] - {int(truth_of[k])}
            orac_extra = osets[k] - {int(truth_of[k])}
            tp += len(comp_extra & orac_extra)
            fp += len(comp_extra - orac_extra)
            fn += len(orac_extra - comp_extra)
        return tp, fp, fn

    n_rel = sum(1 for s in sets if len(s) > 1)
    n_blend = sum(1 for s in osets if len(s) > 1)
    n_unname = sum(1 for nm in nameable if not nm)
    return dict(strict=strict, rel=rel, ctrl=ctrl, oracle=oracle,
                xrel=xrel, foreign=foreign, beta_pr=pr(sets),
                cross_pr=pr(xsets), n_classes=len(T), n_rel=n_rel,
                n_xrel=sum(1 for s in xsets if len(s) > 1),
                n_blend=n_blend, n_unname=n_unname)


def part1() -> None:
    print("── Part 1: data_hard (M2 default stack), relational scoring ──")
    rows = []
    for seed in range(SEEDS):
        r = part1_seed(seed)
        rows.append(r)
        print(f"  seed={seed}: {r['n_classes']} classes "
              f"({r['n_rel']} rel-by-beta, {r['n_xrel']} rel-by-crossmatch, "
              f"{r['n_blend']} true blends, {r['n_unname']} unnameable); "
              f"strict {r['strict']:.3f} -> beta {r['rel']:.3f} / "
              f"crossmatch {r['xrel']:.3f} "
              f"(ctrl {r['ctrl']:.3f}, oracle {r['oracle']:.3f}; "
              f"foreign frac {r['foreign']:.3f})")
    m = {k: sum(r[k] for r in rows) / len(rows)
         for k in ("strict", "rel", "ctrl", "oracle", "xrel")}
    print(f"\n  mean: strict {m['strict']:.3f} | beta {m['rel']:.3f} "
          f"(+{m['rel'] - m['strict']:.3f}) | crossmatch {m['xrel']:.3f} "
          f"(+{m['xrel'] - m['strict']:.3f}) | random-ctrl {m['ctrl']:.3f} "
          f"(+{m['ctrl'] - m['strict']:.3f}) | oracle {m['oracle']:.3f} "
          f"(+{m['oracle'] - m['strict']:.3f})")
    for name, key in (("beta", "beta_pr"), ("crossmatch", "cross_pr")):
        tp = sum(r[key][0] for r in rows)
        fp = sum(r[key][1] for r in rows)
        fn = sum(r[key][2] for r in rows)
        print(f"  {name:>10s} modifier sense: precision "
              f"{tp / max(1, tp + fp):.2f} recall {tp / max(1, tp + fn):.2f} "
              f"(vs true buffer composition, tp={tp} fp={fp} fn={fn})")


# ── Part 2: the 10-animal taxonomy, named descriptors ──────────────

def describe(names, A: int, mods, neg) -> str:
    lead = names[A]
    if mods and mods[0][0] >= 0.40:
        lead = "-".join([names[A]] + [names[c] for _, c in mods])
    elif mods:
        lead = f"{names[mods[0][1]]}-ish {lead}"
    return lead + (f", not {names[neg]}" if neg is not None else "")


def part2(seed: int = 0) -> None:
    print("\n── Part 2: 10-animal taxonomy — do the descriptors read? ──")
    torch.manual_seed(seed)
    tr = make_taxonomy(n_per_class=500, seed=seed)
    te = make_taxonomy(n_per_class=200, seed=100 + seed)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(tr.y), generator=g)
    Xs, ys = tr.x[order], tr.y[order]

    sub = construct(germline_base, capacity=512)
    pg = PerceptionGenesis(sub)
    pg.feed(Xs[:WINDOW]); sub.end_task()
    mixed = MixedStreamController(sub, stress=pg.router, adopt=pg.controller)
    mixed.observe(Xs[:WINDOW]); sub.end_task()
    for w0 in range(WINDOW, len(Xs), WINDOW):
        mixed.observe(Xs[w0:w0 + WINDOW]); sub.end_task()
    for ep in range(1, TAX_EPOCHS):       # fresh draws, same species spec:
        epd = make_taxonomy(n_per_class=500, seed=1000 * ep + seed)
        ge = torch.Generator().manual_seed(1000 * ep + seed)
        Xe = epd.x[torch.randperm(len(epd.y), generator=ge)]
        for w0 in range(0, len(Xe), WINDOW):
            mixed.observe(Xe[w0:w0 + WINDOW]); sub.end_task()

    names = tr.names
    C = len(names)
    T, counts = composition(mixed, Xs, ys, C)
    feeds = pure_prototype_feeds(counts)
    truth_of, sets, mods, negs, nameable = componential(T, counts, feeds,
                                                        BETA_TH)
    xsets, xwts = crossmatch(mixed, truth_of, C, CROSS_TH)

    Z = phasors(mixed, te.x)
    pred = (Z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    strict = float((truth_of[pred] == te.y).float().mean())
    rel = rel_acc(pred, te.y, sets)
    xrel = rel_acc(pred, te.y, xsets)
    osets = oracle_sets(counts)
    oracle = rel_acc(pred, te.y, osets)
    print(f"  {len(T)} discovered classes over {C} species "
          f"({TAX_EPOCHS} epochs); strict {strict:.3f} -> beta {rel:.3f} / "
          f"crossmatch {xrel:.3f} (oracle composition-sets {oracle:.3f})")
    print(f"  {'class':>5s} {'n':>5s} {'purity':>6s}  descriptor")
    tot = counts.sum(1)
    for k in torch.argsort(tot, descending=True).tolist():
        if truth_of[k] < 0 or tot[k] < 5:
            continue
        A = int(truth_of[k])
        pur = float(counts[k, A] / tot[k])
        frac = {names[c]: float(counts[k, c] / tot[k])
                for c in range(C) if counts[k, c] / tot[k] >= 0.10}
        comp_s = " ".join(f"{n}:{f:.2f}" for n, f in
                          sorted(frac.items(), key=lambda t: -t[1]))
        b_s = " ".join(f"+{names[c]}({b:.2f})" for b, c in mods[k])
        x_s = " ".join(f"+{names[c]}({w:.2f})" for c, w in
                       sorted(xwts[k].items(), key=lambda t: -t[1])
                       if c != A and w >= CROSS_TH)
        n_s = f" not-{names[negs[k]]}" if negs[k] is not None else ""
        tag = ("\"" + describe(names, A, mods[k], negs[k]) + "\""
               if nameable[k] else f"[{names[A]}] (unnameable: no other "
               f"pure {names[A]} class in the vocabulary)")
        print(f"  {mixed.classes[k].name:>5s} {int(tot[k]):>5d} {pur:>6.2f}  "
              f"{tag}  [beta: {b_s}{n_s} | cross: {x_s}]  buffer: {comp_s}")


def main() -> None:
    part1()
    part2()


if __name__ == "__main__":
    main()
