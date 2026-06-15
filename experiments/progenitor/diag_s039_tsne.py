"""s039 t-SNE — is 2D spatial position represented in the quantum bands?
(Rocky's question after the cs/tree smoke.)

Two visualisations, one figure (outputs/diag_s039_tsne.png):

TOP ROW — DIMENSION embedding. t-SNE the 784 pixel dimensions using the
co-quantum-agreement distance (the same metric the NJ tree used), coloured
by grid ROW and grid COL. If 2D position survives in the quantum bands,
spatially-adjacent pixels embed together -> the colour forms a smooth
gradient. If position is lost, the colours are scrambled noise.

BOTTOM ROW — SAMPLE embedding. t-SNE the images (quanta vectors) coloured
by class, for RAW intensity quanta vs |center-surround| quanta. Shows WHY
cs nearly doubles full accuracy vs raw (0.385 -> 0.66): cleaner clusters.

Run:  python3 -m experiments.progenitor.diag_s039_tsne
Env:  S039_DIM_PERCLASS (default 300 for the distance),
      S039_SMP_PERCLASS (default 60 for the sample embed).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from trioron.core.receptor import quantize
from trioron.legacy.donorkit.datasets import (DatasetBundle,
                                              chained_15_specs,
                                              build_task_views)
from experiments.progenitor.diag_s039_patchtree import (
    center_surround, coquantum_distance, balanced_slice)

OUT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))),
    "outputs", "diag_s039_tsne.png")
DIM_PC = int(os.environ.get("S039_DIM_PERCLASS", "300"))
SMP_PC = int(os.environ.get("S039_SMP_PERCLASS", "60"))
N_CLASSES = 30


def main() -> None:
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"],
                           root=DATA_ROOT_FIX(), n_holdout_per_dataset=0)
    specs = chained_15_specs()
    views = build_task_views(bundle, specs, split="train")

    # ---- dimension embedding (co-quantum distance) ----
    Xd, _ = balanced_slice(views, DIM_PC, seed=0)
    D, active = coquantum_distance(Xd)                    # raw-pixel quanta
    print(f"dim embed: {len(active)} active dims", flush=True)
    emb_d = TSNE(n_components=2, metric="precomputed", init="random",
                 perplexity=30, random_state=0).fit_transform(D)
    rows = active // 28
    cols = active % 28

    # ---- sample embedding (raw vs cs quanta) ----
    Xs, ys = balanced_slice(views, SMP_PC, seed=1)
    q_raw = quantize(Xs).numpy()
    q_cs = quantize(center_surround(Xs)).numpy()
    print("sample embed: raw ...", flush=True)
    emb_raw = TSNE(n_components=2, metric="cosine", init="random",
                   perplexity=30, random_state=0).fit_transform(q_raw)
    print("sample embed: cs ...", flush=True)
    emb_cs = TSNE(n_components=2, metric="cosine", init="random",
                  perplexity=30, random_state=0).fit_transform(q_cs)
    yn = ys.numpy()

    fig, ax = plt.subplots(2, 2, figsize=(13, 12))
    for a, color, name in ((ax[0, 0], rows, "grid ROW"),
                           (ax[0, 1], cols, "grid COL")):
        sc = a.scatter(emb_d[:, 0], emb_d[:, 1], c=color, cmap="viridis", s=14)
        a.set_title(f"DIMENSIONS by co-quantum distance — colour = {name}\n"
                    "smooth gradient = position recovered; noise = lost")
        fig.colorbar(sc, ax=a, shrink=0.8)
    for a, emb, name in ((ax[1, 0], emb_raw, "RAW quanta"),
                         (ax[1, 1], emb_cs, "|center-surround| quanta")):
        sc = a.scatter(emb[:, 0], emb[:, 1], c=yn, cmap="gist_ncar", s=10)
        a.set_title(f"SAMPLES — {name} — colour = class (30)")
        fig.colorbar(sc, ax=a, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"saved {OUT}")


def DATA_ROOT_FIX():
    return os.environ.get(
        "AXIS6_DATA_ROOT",
        os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "outputs", "data"))


if __name__ == "__main__":
    main()
