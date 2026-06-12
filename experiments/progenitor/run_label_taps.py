"""Label tap bank gate — labels as reference carriers (s034 first cut,
Rocky's design; trioron/pcll/labels.py).

DATA SEPARATION (Rocky, s034): TRAIN and TEST are separate draws from
the one fixed generative spec (data_hard semantics: shared spec +
train-side norm). Labels are supplied ONLY for training-stream rows,
at a given coverage; the TEST split is never streamed, never labeled —
every accuracy below reads on it.

Gates:

  A. NON-DISTURBANCE — the same seed grown twice, labels OFF vs 100%
     coverage: class count, templates, and clean test accuracy must be
     BIT-IDENTICAL. True by construction (the learning channel cannot
     tell labeled from unlabeled rows; the bank is write-only from
     learning's side) — asserted anyway.
  B. NAMING — coverage sweep {1%, 5%, 20%, 100%} x 3 seeds on
     data_hard. Two readouts per class: TEMPLATE-beta (decompose the
     class template over tap prototypes — exact for coherent blends,
     mode-smearing expected on the multimodal world) and MEMBER-MIX
     (match each buffered member against the prototypes — per-member,
     sees what the template mean hides). Metrics vs the s034 probe:
     modifier precision/recall against true buffer composition (s034
     label-free best: 0.02), relational accuracy vs the 0.957 oracle,
     tap-primary naming accuracy vs the eval-side majority map.
  C. DESCRIPTORS — the 10-animal taxonomy at 5% and 100% coverage:
     u-classes should now read "chicken-duck", "goat-dog" (the s034
     vocabulary gap: closed — duck exists as a prototype without a
     duck class).

MEASURED VERDICT (s034 first cut):

 * A: PASS, exact — templates bit-identical with 16k labeled deposits.
 * C (taxonomy, unimodal labels): THE HEADROOM IS RECOVERED. Member-
   mix relational accuracy 0.789 -> 0.948 = the composition oracle
   (0.945), ALREADY AT 5% COVERAGE (271 labels). At 100%, beta names
   the blends outright: "chicken-duck" (+duck 0.30; member mix
   duck 0.51/chicken 0.49 vs truth 0.48/0.49), "goat-dog". The s029
   ask — "goat-ish chicken, not pig" — is now produced by the
   organism. member_mix must rank by margin E/sigma, not raw E (the
   D20 ruling recurs: raw E let the stronger tap of a 0.996-collinear
   pair absorb every member: chicken:1.00). beta needs high coverage
   for collinear pairs (5%: prototype noise swamps the 0.4%
   orthogonal component); member-mix is the workhorse.
 * B (data_hard, multimodal labels): MODE-SMEARING LIMIT, recorded.
   A species = 3 scattered modes; its tap prototype is the phasor
   mean over them — washed out (the disruptor-invisibility physics at
   the species scale). Tap-primary naming 0.53 vs majority-map 0.829;
   relational sets LOSE to strict; P/R <= 0.09/0.30 at any coverage.
   Species-grain carriers cannot name mode-grain classes.
 * D ((class x label) counts, Rocky-approved increment): naming at
   the grain the organism discovered — beats mode-smearing where the
   carriers could not. data_hard relational 0.909 at 100% coverage
   (0.894 at 20%) vs strict 0.829 / oracle 0.957; count-majority
   primary 0.826 ~= the eval-side majority map 0.829 — THE SELF-LABEL
   MAP IS NOW ORGANISM-INTERNAL (deployment names need no eval
   harness). Two measured corrections: (1) restart-at-zero left
   tail-born division children permanently unnamed (taxonomy:
   4 final-boundary children, rel-cnt 0.583) -> children INHERIT
   parent counts by buffer-side fraction (taxonomy 0.911; data_hard
   trades 0.926 -> 0.909 — the estimator smears labels across
   mode-separating splits; exact repair = per-member label tags
   moving with buffers, deferred). (2) The residual oracle gap is
   DEPOSIT-TIME STALENESS: counts annotate one-pass membership as it
   happened; the oracle reads final-template membership.
 * BAYES CONTEXT (Rocky's question, s034): not the limit. Each metric
   vs its own ceiling — single-label: 0.829 vs clean Bayes 0.993 (the
   law-1 self-labeling gap; annotation is write-only by design and
   cannot move it — label-supervised consolidation is the deferred
   lever); set-metric: 0.909 vs its own oracle 0.957. On the taxonomy
   the set metric EXCEEDS single-label Bayes (0.948 > 0.917):
   allowing the discovered blend class + set credit sidesteps the
   chicken/duck confusion every single-label namer must eat — Rocky's
   "allow such new class" insight, quantified.
 * E (label-supervised consolidation, s034c, Rocky-approved DEFAULT):
   the law-1 lever, engaged. Per-member label tags ride the buffers
   (division/merge/consolidation/annealing); a labeled row EVACUATES
   a class when its label holds < SUPERVISE_FRAC of the class's
   tagged members, moving to its label's best mature home — at
   ingress and inside the EM round. Single-label strict on data_hard:
   0.829 -> 0.957 at 100% coverage (= the former set-metric oracle,
   now honest single-label; Bayes 0.993), 0.873 at 20%, baseline
   +-0.007 at 1-5%; purity 0.740 -> 0.813; classes 91 <= the 96-mode
   bound; internal naming = strict (0.957). THREE FALSIFIED RULES on
   the way (all measured): move-from-immature classes lets one
   early-matured class vacuum its label across all modes (naming
   0.752 -> 0.299); rescuing REFUSED rows force-feeds spray past the
   D20 gate (strict -0.023 at 5%); ledger-based majorities are
   staleness-capped at ~0.45-0.59 fractions (zero mature classes) —
   routing majorities must come from buffer TAGS. Genuine blends
   (taxonomy chicken-duck at 0.48/0.49) are correctly LEFT INTACT:
   their members are nobody's minority.

Run: python3 -m experiments.progenitor.run_label_taps
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch

from trioron.core import construct
from trioron.pcll import MixedStreamController, PerceptionGenesis, germline_base

from .data_hard import make_spec, sample
from .data_taxonomy import make_taxonomy
from .run_componential import (composition, oracle_sets, phasors, rel_acc)
from .run_m2_mixed import N_TEST, S_PERIOD, SEEDS, WINDOW

COVERAGES = (0.01, 0.05, 0.20, 1.00)
MIX_TH = 0.20      # composition fraction to count as a blend partner
BETA_TH = 0.20


def grow_labeled(seed: int, coverage: Optional[float],
                 supervise: bool = False):
    """The M2 default-stack protocol with the annotation carrier.
    coverage=None -> labels argument omitted entirely (gate A arm 1).
    supervise -> label_supervise (s034b; the controller default is ON,
    arms pass it explicitly so each gate states its configuration)."""
    torch.manual_seed(seed)
    spec = make_spec()
    tr, norm = sample(spec, n_per_class=S_PERIOD, seed=seed)
    te, _ = sample(spec, n_per_class=N_TEST, seed=100 + seed, norm=norm)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(tr.y), generator=g)
    Xs, ys = tr.x[order], tr.y[order]

    labs: Optional[List[Optional[str]]] = None
    if coverage is not None:
        gm = torch.Generator().manual_seed(40 + seed)
        mask = torch.rand(len(ys), generator=gm) < coverage
        labs = [tr.names[int(y)] if bool(m) else None
                for y, m in zip(ys, mask)]

    sub = construct(germline_base, capacity=512)
    pg = PerceptionGenesis(sub)
    pg.feed(Xs[:WINDOW]); sub.end_task()
    mixed = MixedStreamController(sub, stress=pg.router, adopt=pg.controller,
                                  label_supervise=supervise)

    def lab_slice(a, b):
        return None if labs is None else labs[a:b]

    mixed.observe(Xs[:WINDOW], labels=lab_slice(0, WINDOW)); sub.end_task()
    for w0 in range(WINDOW, len(Xs), WINDOW):
        mixed.observe(Xs[w0:w0 + WINDOW],
                      labels=lab_slice(w0, w0 + WINDOW))
        sub.end_task()

    clean = [c for c in range(len(tr.names)) if float(spec.std[c]) <= 0.3]
    keep = torch.isin(te.y, torch.tensor(clean))
    return mixed, (Xs, ys), (te.x[keep], te.y[keep]), tr.names


def predict(mixed, T, X):
    Z = phasors(mixed, X)
    return (Z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)


def name_sets(mixed, T, names):
    """Per class, the two label-grounded readouts -> (index sets,
    primary indices, raw namings) for template-beta and member-mix."""
    idx = {n: i for i, n in enumerate(names)}
    beta_sets, mix_sets, primaries, raw = [], [], [], []
    for k in range(len(T)):
        prim, mods = (mixed.label_taps.name(T[k], th=BETA_TH)
                      if mixed.label_taps else (None, []))
        bset = ({idx[prim]} | {idx[l] for _, l in mods}) \
            if prim is not None else set()
        mix = (mixed.label_taps.member_mix(
            torch.exp(1j * 2 * math.pi
                      * mixed._recover_q(mixed.bufs[k]) / 1000))
            if mixed.label_taps else {})
        mset = {idx[l] for l, f in mix.items()
                if l != "?" and f >= MIX_TH}
        beta_sets.append(bset)
        mix_sets.append(mset)
        primaries.append(idx.get(prim, -1) if prim is not None else -1)
        raw.append((prim, mods, mix))
    return beta_sets, mix_sets, torch.tensor(primaries), raw


def pr_vs_oracle(label_sets, osets, truth_of):
    """Modifier precision/recall (extras relative to the majority
    truth — comparable to the s034 numbers)."""
    tp = fp = fn = 0
    for k in range(len(label_sets)):
        A = int(truth_of[k])
        pred_extra = label_sets[k] - {A}
        orac_extra = osets[k] - {A}
        tp += len(pred_extra & orac_extra)
        fp += len(pred_extra - orac_extra)
        fn += len(orac_extra - pred_extra)
    return tp, fp, fn


def gate_a() -> None:
    """Non-disturbance, redefined for s034b: ANNOTATION (supervise off)
    must be bit-identical to labels-off; SUPERVISION (default ON) is
    the deliberate, gated touch and must not regress strict accuracy."""
    print("── A. non-disturbance: OFF vs annotate-only vs supervised, "
          "seed 0, 100% coverage ──")
    m0, _, (Xt, yt), _ = grow_labeled(0, None)
    m1, (Xs, ys), _, names = grow_labeled(0, 1.0, supervise=False)
    T0, T1 = m0.templates(), m1.templates()
    assert len(m0.classes) == len(m1.classes), "class count diverged"
    assert T0.shape == T1.shape and bool(torch.equal(T0, T1)), \
        "templates diverged — the bank leaked into learning"
    C = len(names)
    truth0 = composition(m0, Xs, ys, C)[1].argmax(1)
    a0 = float((truth0[predict(m0, T0, Xt)] == yt).float().mean())
    a1 = float((truth0[predict(m1, T1, Xt)] == yt).float().mean())
    assert a0 == a1, (a0, a1)
    n_lab = int(m1.label_taps.counts.sum())
    print(f"  annotate-only: {len(m0.classes)} classes, templates "
          f"BIT-IDENTICAL, clean acc {a0:.3f} == {a1:.3f} "
          f"({n_lab} labeled deposits absorbed)")
    ms, (Xs2, ys2), (Xt2, yt2), names2 = grow_labeled(0, 1.0,
                                                      supervise=True)
    Ts = ms.templates()
    truth_s = composition(ms, Xs2, ys2, C)[1].argmax(1)
    a2 = float((truth_s[predict(ms, Ts, Xt2)] == yt2).float().mean())
    print(f"  supervised:    {len(ms.classes)} classes, clean acc "
          f"{a2:.3f} (vs {a0:.3f} unsupervised)")
    assert a2 >= a0 - 0.01, f"supervision regressed accuracy {a0}->{a2}"
    print("  GATE A PASS")


def gate_b() -> None:
    print("\n── B. naming under the coverage sweep (data_hard, "
          f"{SEEDS} seeds) ──")
    print(f"  {'cov':>5s} {'strict':>6s} {'tap-1ry':>7s} "
          f"{'rel-beta':>8s} {'rel-mix':>7s} {'oracle':>6s}   "
          f"beta P/R    mix P/R")
    for cov in COVERAGES:
        accs = dict(strict=[], prim=[], beta=[], mix=[], orac=[])
        bpr = [0, 0, 0]
        mpr = [0, 0, 0]
        for seed in range(SEEDS):
            mixed, (Xs, ys), (Xt, yt), names = grow_labeled(seed, cov)
            C = len(names)
            T, counts = composition(mixed, Xs, ys, C)
            truth_of = counts.argmax(1)
            pred = predict(mixed, T, Xt)
            osets = oracle_sets(counts)
            bsets, msets, prim, _ = name_sets(mixed, T, names)
            accs["strict"].append(
                float((truth_of[pred] == yt).float().mean()))
            accs["prim"].append(float((prim[pred] == yt).float().mean()))
            accs["beta"].append(rel_acc(pred, yt, bsets))
            accs["mix"].append(rel_acc(pred, yt, msets))
            accs["orac"].append(rel_acc(pred, yt, osets))
            for tot, sets in ((bpr, bsets), (mpr, msets)):
                d = pr_vs_oracle(sets, osets, truth_of)
                for i in range(3):
                    tot[i] += d[i]
        m = {k: sum(v) / len(v) for k, v in accs.items()}
        bp = bpr[0] / max(1, bpr[0] + bpr[1])
        br = bpr[0] / max(1, bpr[0] + bpr[2])
        mp = mpr[0] / max(1, mpr[0] + mpr[1])
        mr = mpr[0] / max(1, mpr[0] + mpr[2])
        print(f"  {cov:>5.0%} {m['strict']:>6.3f} {m['prim']:>7.3f} "
              f"{m['beta']:>8.3f} {m['mix']:>7.3f} {m['orac']:>6.3f}   "
              f"{bp:.2f}/{br:.2f}   {mp:.2f}/{mr:.2f}")


def count_sets(mixed, names):
    """Label sets + primary from the (class x label) counts."""
    idx = {n: i for i, n in enumerate(names)}
    sets, prim = [], []
    for k in range(len(mixed.classes)):
        comp = (mixed.label_taps.composition_of(mixed.classes[k].name)
                if mixed.label_taps else {})
        s = {idx[l] for l, f in comp.items() if f >= MIX_TH}
        p = idx[max(comp, key=comp.get)] if comp else -1
        if p >= 0:
            s |= {p}
        sets.append(s)
        prim.append(p)
    return sets, torch.tensor(prim)


def gate_d() -> None:
    print("\n── D. (class x label) counts under the coverage sweep "
          f"(data_hard, {SEEDS} seeds) ──")
    # the clean-Bayes ceiling this eval is bounded by (single-label,
    # true generative mixture, clean species only)
    from .data_hard import HardTaxon, bayes_accuracy
    spec = make_spec()
    tr, norm = sample(spec, n_per_class=S_PERIOD, seed=0)
    te, _ = sample(spec, n_per_class=N_TEST, seed=100, norm=norm)
    clean = [c for c in range(len(te.names)) if float(spec.std[c]) <= 0.3]
    keep = torch.isin(te.y, torch.tensor(clean))
    bayes = bayes_accuracy(
        HardTaxon(te.x[keep], te.x_raw[keep], te.y[keep], spec, te.names))
    print(f"  clean Bayes ceiling (single-label) {bayes:.3f}")
    print(f"  {'cov':>5s} {'strict':>6s} {'cnt-1ry':>7s} "
          f"{'rel-cnt':>7s} {'oracle':>6s}   cnt P/R")
    last = {}
    for cov in COVERAGES:
        accs = dict(strict=[], prim=[], cnt=[], orac=[])
        cpr = [0, 0, 0]
        for seed in range(SEEDS):
            mixed, (Xs, ys), (Xt, yt), names = grow_labeled(seed, cov)
            C = len(names)
            T, counts = composition(mixed, Xs, ys, C)
            truth_of = counts.argmax(1)
            pred = predict(mixed, T, Xt)
            osets = oracle_sets(counts)
            csets, cprim = count_sets(mixed, names)
            accs["strict"].append(
                float((truth_of[pred] == yt).float().mean()))
            accs["prim"].append(float((cprim[pred] == yt).float().mean()))
            accs["cnt"].append(rel_acc(pred, yt, csets))
            accs["orac"].append(rel_acc(pred, yt, osets))
            d = pr_vs_oracle(csets, osets, truth_of)
            for i in range(3):
                cpr[i] += d[i]
        m = {k: sum(v) / len(v) for k, v in accs.items()}
        cp = cpr[0] / max(1, cpr[0] + cpr[1])
        cr = cpr[0] / max(1, cpr[0] + cpr[2])
        print(f"  {cov:>5.0%} {m['strict']:>6.3f} {m['prim']:>7.3f} "
              f"{m['cnt']:>7.3f} {m['orac']:>6.3f}   {cp:.2f}/{cr:.2f}")
        last = m
    # the gate: at full coverage the counts must recover the bulk of
    # the oracle's composition headroom and the count-majority must
    # match the eval-side majority map (the map becomes
    # ORGANISM-INTERNAL). The residual oracle gap (~0.03, measured) is
    # DEPOSIT-TIME STALENESS: counts annotate one-pass membership as
    # it happened (immature classes; division children restart empty),
    # while the oracle reads final-template membership over the whole
    # stream. Closing it needs per-member label tags moving with
    # buffers through division/merge/consolidation — a heavier
    # integration, deferred.
    assert last["cnt"] >= last["orac"] - 0.05, (
        f"count sets {last['cnt']:.3f} miss the oracle {last['orac']:.3f}")
    assert last["prim"] >= last["strict"] - 0.02, (
        f"count-majority naming {last['prim']:.3f} below the "
        f"eval-side map {last['strict']:.3f}")
    print("  GATE D PASS")


def gate_e() -> None:
    """Label-supervised consolidation [s034b]: the single-label
    (strict) accuracy is THE metric — the law-1 gap supervision was
    built to attack (annotation alone cannot move it by design)."""
    print("\n── E. label-supervised consolidation (data_hard, "
          f"{SEEDS} seeds; clean Bayes 0.993) ──")
    print(f"  {'cov':>5s} {'strict':>6s} {'cnt-1ry':>7s} "
          f"{'rel-cnt':>7s} {'oracle':>6s} {'purity':>6s} {'cls':>4s}")
    base_strict = None
    rows = {}
    for cov in (0.0,) + COVERAGES:
        accs = dict(strict=[], prim=[], cnt=[], orac=[], pur=[], ncl=[])
        for seed in range(SEEDS):
            mixed, (Xs, ys), (Xt, yt), names = grow_labeled(
                seed, cov, supervise=True)
            C = len(names)
            T, counts = composition(mixed, Xs, ys, C)
            truth_of = counts.argmax(1)
            pred = predict(mixed, T, Xt)
            csets, cprim = count_sets(mixed, names)
            accs["strict"].append(
                float((truth_of[pred] == yt).float().mean()))
            accs["prim"].append(float((cprim[pred] == yt).float().mean()))
            accs["cnt"].append(rel_acc(pred, yt, csets))
            accs["orac"].append(rel_acc(pred, yt, oracle_sets(counts)))
            tot = counts.sum(1).clamp_min(1)
            pur = (counts.max(1).values / tot)[counts.sum(1) > 0]
            accs["pur"].append(float(pur.mean()))
            accs["ncl"].append(len(mixed.classes))
        m = {k: sum(v) / len(v) for k, v in accs.items()}
        rows[cov] = m
        if cov == 0.0:
            base_strict = m["strict"]
        print(f"  {cov:>5.0%} {m['strict']:>6.3f} {m['prim']:>7.3f} "
              f"{m['cnt']:>7.3f} {m['orac']:>6.3f} {m['pur']:>6.3f} "
              f"{m['ncl']:>4.0f}")
    # the gate: 0% coverage reproduces the unsupervised baseline; full
    # coverage must lift the single-label metric the annotation arms
    # could not touch
    assert abs(base_strict - 0.829) < 0.02, (
        f"0% arm drifted from the M2 baseline: {base_strict:.3f}")
    assert rows[1.0]["strict"] > base_strict, (
        f"supervision does not lift strict accuracy "
        f"({rows[1.0]['strict']:.3f} vs {base_strict:.3f})")
    print("  GATE E PASS")


def gate_c(coverage: float = 0.05, seed: int = 0) -> None:
    print(f"\n── C. taxonomy descriptors at {coverage:.0%} coverage ──")
    torch.manual_seed(seed)
    tr = make_taxonomy(n_per_class=500, seed=seed)
    te = make_taxonomy(n_per_class=200, seed=100 + seed)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(tr.y), generator=g)
    Xs, ys = tr.x[order], tr.y[order]
    gm = torch.Generator().manual_seed(40 + seed)
    mask = torch.rand(len(ys), generator=gm) < coverage
    labs = [tr.names[int(y)] if bool(m) else None
            for y, m in zip(ys, mask)]

    sub = construct(germline_base, capacity=512)
    pg = PerceptionGenesis(sub)
    pg.feed(Xs[:WINDOW]); sub.end_task()
    mixed = MixedStreamController(sub, stress=pg.router, adopt=pg.controller)
    mixed.observe(Xs[:WINDOW], labels=labs[:WINDOW]); sub.end_task()
    for w0 in range(WINDOW, len(Xs), WINDOW):
        mixed.observe(Xs[w0:w0 + WINDOW], labels=labs[w0:w0 + WINDOW])
        sub.end_task()

    names = tr.names
    C = len(names)
    T, counts = composition(mixed, Xs, ys, C)
    truth_of = counts.argmax(1)
    pred = predict(mixed, T, te.x)
    strict = float((truth_of[pred] == te.y).float().mean())
    bsets, msets, prim, raw = name_sets(mixed, T, names)
    rel_b = rel_acc(pred, te.y, bsets)
    rel_m = rel_acc(pred, te.y, msets)
    csets, _ = count_sets(mixed, names)
    rel_c = rel_acc(pred, te.y, csets)
    oracle = rel_acc(pred, te.y, oracle_sets(counts))
    from .data_taxonomy import bayes_accuracy
    print(f"  {len(T)} classes; strict {strict:.3f} -> "
          f"rel-beta {rel_b:.3f} / rel-mix {rel_m:.3f} / "
          f"rel-cnt {rel_c:.3f} (oracle {oracle:.3f}, "
          f"Bayes single-label {bayes_accuracy(te):.3f}); "
          f"{int(mask.sum())} labeled rows of {len(ys)}")
    # the gate: member-mix recovers the composition headroom, and at
    # full coverage the blends are NAMED (beta modifiers fire)
    assert rel_m >= oracle - 0.02, (
        f"member-mix {rel_m:.3f} below the composition oracle {oracle:.3f}")
    if coverage >= 1.0:
        assert sum(1 for s in bsets if len(s) > 1) >= 2, (
            "no blend names at full coverage")
        print("  GATE C PASS")
    print(f"  {'class':>5s} {'n':>5s}  tap descriptor"
          f"{'':24s}member mix{'':16s}buffer truth")
    tot = counts.sum(1)
    for k in torch.argsort(tot, descending=True).tolist():
        if tot[k] < 5:
            continue
        prim_l, mods, mix = raw[k]
        if prim_l is None:
            continue
        name_s = prim_l + "".join(f"-{l}" for b, l in mods)
        mods_s = " ".join(f"+{l}({b:.2f})" for b, l in mods)
        mix_s = " ".join(f"{l}:{f:.2f}" for l, f in
                         sorted(mix.items(), key=lambda t: -t[1])
                         if f >= 0.10)
        truth_s = " ".join(
            f"{names[c]}:{counts[k, c] / tot[k]:.2f}"
            for c in counts[k].argsort(descending=True).tolist()
            if counts[k, c] / tot[k] >= 0.10)
        print(f"  {mixed.classes[k].name:>5s} {int(tot[k]):>5d}  "
              f"\"{name_s}\" {mods_s:30s} {mix_s:25s} {truth_s}")


def main() -> None:
    gate_a()
    gate_b()
    gate_d()
    gate_e()
    gate_c(0.05)
    gate_c(1.00)


if __name__ == "__main__":
    main()
