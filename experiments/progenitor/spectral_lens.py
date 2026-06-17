"""Multi-carrier wavelength sweep for the scattering lens (s042 exploration).

The single scattering lens (fingerprint_lens.lens_descriptor) reads ONE carrier:
per patch it takes the mean unit phasor exp(i*theta), theta=2*pi*q/1000. That
captures only the first Fourier mode of the patch's phase distribution and
throws the rest away. The "wavelength sweep" generalizes it to a FILTERBANK:
probe each patch at a set of carrier frequencies w and record the response

    R_p(w) = (1/n) * sum_{j in patch} exp(i * w * theta_j)         (complex)

w=1 reproduces the old lens. Sweeping w=1,2,4,... samples the empirical
characteristic function of the patch's value distribution -> a per-patch
SPECTRUM (the within-patch texture the single carrier discarded). Concatenate
(re,im) over carriers and patches -> a richer descriptor; back-end stays the
per-class centroid (= ManifoldArchive).

This script measures, on held-out test data, whether the sweep lifts
nearest-centroid accuracy over the single carrier, and disentangles the sweep
from plain per-dim whitening (z-score). Two datasets, WHAT channel isolated
(perfectly registered patches, so localization error does not mask the lens):

  toy3    : the 3x5 1/2/3 stream (stream_sim_3class)
  digit10 : the 7x5 dot-matrix 0-9 font, centered (digit_bench_2d.FONT/variant)

Run:  python3 experiments/progenitor/spectral_lens.py
"""
from __future__ import annotations

import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trioron.core.receptor import quantize
from experiments.progenitor.fingerprint_lens import adaptive_patches
import experiments.progenitor.stream_sim_3class as S
from experiments.progenitor import digit_bench_2d as DB


# ── the multi-carrier (wavelength-sweep) descriptor ─────────────────────

def spectral_descriptor(q, patches, carriers):
    """Per patch, per carrier w: mean unit phasor at frequency w -> (re,im).
    carriers=[1.0] reproduces fingerprint_lens.lens_descriptor exactly."""
    theta = 2 * math.pi * q.float() / 1000.0
    feats = []
    for p in patches:
        tp = theta[p]
        for w in carriers:
            z = torch.exp(1j * (w * tp)).mean()
            feats += [z.real.item(), z.imag.item()]
    return np.array(feats)


# ── held-out nearest-centroid eval (raw + z-scored) ─────────────────────

def evaluate(Dtr, ytr, Dte, yte, classes):
    """Returns (acc_raw, acc_zscore). z-score = per-dim whitening on train
    stats, then nearest centroid -> isolates 'richer descriptor' from 'better
    scaling'."""
    def nn_centroid(Dtr, Dte):
        cent = {c: Dtr[ytr == c].mean(0) for c in classes}
        pred = np.array([min(classes, key=lambda c: np.linalg.norm(Dte[i] - cent[c]))
                         for i in range(len(Dte))])
        return float((pred == yte).mean())
    acc_raw = nn_centroid(Dtr, Dte)
    mu, sd = Dtr.mean(0), Dtr.std(0) + 1e-8
    acc_z = nn_centroid((Dtr - mu) / sd, (Dte - mu) / sd)
    return acc_raw, acc_z


# ── datasets: descriptors for a given carrier set ───────────────────────

def toy3_data():
    """3x5 1/2/3 stream, per-sample contrast-norm q (NOT the canonical frame),
    train/test split. Returns (q_list_tr, ytr, q_list_te, yte, patches)."""
    X, y = S.build_X(n_per_class=120, seed=7)
    labs = np.array([int(l) for l in y])
    patches = adaptive_patches((5, 3), k=2)
    qs = [quantize(X[i]).to(torch.int64) for i in range(len(X))]
    # split by class: first 70 train, rest test
    tr, te = [], []
    seen = {1: 0, 2: 0, 3: 0}
    for i, c in enumerate(labs):
        (tr if seen[int(c)] < 70 else te).append(i)
        seen[int(c)] += 1
    return ([qs[i] for i in tr], labs[tr], [qs[i] for i in te], labs[te],
            patches, (1, 2, 3))


def digit10_data():
    """7x5 0-9 font, centered (WHAT isolated), train/test split."""
    digits = list(range(10))
    patches = adaptive_patches((DB.RH, DB.RW), k=2)
    gtr = torch.Generator().manual_seed(3)
    gte = torch.Generator().manual_seed(99)
    qtr, ytr, qte, yte = [], [], [], []
    for ch in digits:
        for _ in range(80):
            qtr.append(quantize(DB.variant(str(ch), gtr).reshape(-1)).to(torch.int64))
            ytr.append(ch)
        for _ in range(60):
            qte.append(quantize(DB.variant(str(ch), gte).reshape(-1)).to(torch.int64))
            yte.append(ch)
    return (qtr, np.array(ytr), qte, np.array(yte), patches, tuple(digits))


def descriptors(qlist, patches, carriers):
    return np.stack([spectral_descriptor(q, patches, carriers) for q in qlist])


# ── carrier-count sweep ─────────────────────────────────────────────────

CARRIER_SETS = [
    ("w=1 (baseline)",       [1.0]),
    ("w<=2",                 [1.0, 2.0]),
    ("w<=4",                 [1.0, 2.0, 3.0, 4.0]),
    ("w<=8",                 [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]),
    ("w<=16",                [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]),
    ("w<=32",                [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]),
]


def run(name, loader):
    qtr, ytr, qte, yte, patches, classes = loader()
    print(f"\n=== {name}: {len(qtr)} train / {len(qte)} test, "
          f"{len(classes)} classes, {len(patches)} patches ===")
    print(f"  {'carriers':<18} {'descr-dim':>9} {'acc(raw)':>9} {'acc(zscore)':>12}")
    rows = []
    for label, carriers in CARRIER_SETS:
        Dtr = descriptors(qtr, patches, carriers)
        Dte = descriptors(qte, patches, carriers)
        ar, az = evaluate(Dtr, ytr, Dte, yte, classes)
        rows.append((label, len(carriers), Dtr.shape[1], ar, az))
        print(f"  {label:<18} {Dtr.shape[1]:>9} {ar:>9.3f} {az:>12.3f}")
    return rows, (qtr, ytr, qte, yte, patches, classes)


def main():
    toy_rows, toy = run("toy3 (3x5, classes 1/2/3)", toy3_data)
    dig_rows, dig = run("digit10 (7x5, 0-9, WHAT isolated)", digit10_data)

    # ── figure: acc vs #carriers, both datasets, raw + zscore ───────────
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, (rows, ttl) in zip(ax[:2], [(toy_rows, "toy3"), (dig_rows, "digit10")]):
        nc = [r[1] for r in rows]
        a.plot(nc, [r[3] for r in rows], "o-", color="navy", label="raw centroid")
        a.plot(nc, [r[4] for r in rows], "s-", color="crimson", label="z-scored")
        a.axhline(rows[0][3], color="navy", ls=":", lw=0.8, alpha=0.6)
        a.set_xscale("log", base=2)
        a.set_title(f"{ttl}: nearest-centroid acc vs carrier count", fontsize=9)
        a.set_xlabel("# carriers (log2)"); a.set_ylabel("held-out accuracy")
        a.set_xticks(nc); a.set_xticklabels([str(n) for n in nc])
        a.grid(alpha=0.3); a.legend(fontsize=8)

    # ── per-class absorption spectrum (digit10): mean |R(w)| over patches ─
    qtr, ytr, qte, yte, patches, classes = dig
    wsweep = list(range(1, 25))
    spec = {}
    for c in classes:
        idx = np.where(ytr == c)[0]
        mags = []
        for w in wsweep:
            mw = []
            for i in idx[:40]:
                theta = 2 * math.pi * qtr[i].float() / 1000.0
                m = np.mean([abs(torch.exp(1j * (w * theta[p])).mean().item())
                             for p in patches])
                mw.append(m)
            mags.append(np.mean(mw))
        spec[c] = mags
    for c in classes:
        ax[2].plot(wsweep, spec[c], lw=1.3, label=str(c))
    ax[2].set_title("digit10 per-class absorption spectrum  (mean |R(w)| over patches)",
                    fontsize=9)
    ax[2].set_xlabel("carrier frequency w"); ax[2].set_ylabel("mean response |R(w)|")
    ax[2].legend(fontsize=7, ncol=2, title="digit")
    ax[2].grid(alpha=0.3)

    fig.suptitle("Wavelength sweep: multi-carrier filterbank vs single-carrier lens",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "outputs/spectral_lens.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
