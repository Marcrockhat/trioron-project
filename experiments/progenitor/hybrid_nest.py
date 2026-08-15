"""Hybrid nest — phasecyte acquisition + dream-distilled trioron leaves. s049.

Rocky's hybrid: nest BOTH learners. Per domain, the phasecyte leaf
absorbs its 5 tasks in one gradient-free pass (wake); then a gradient
trioron leaf trains offline on pseudo-pockets sampled from the
phasecyte's own per-class manifold sketches (dream) — no stored data.
Key structural point: the sketches cover all 10 domain classes at dream
time, so the gradient leaf trains on a STATIONARY joint distribution —
the nest converts continual learning into per-leaf joint learning, and
the gradient leaf needs no CL machinery at all.

The untested bridge this measures (flagged in the s049 design turn):
manifold replay was validated on gradient-organism perception space —
here the sketches live in phasecyte POCKET space. Question 1: does the
dream-distilled trioron leaf beat the phasecyte leaf it was distilled
from ("morning after" test)? Question 2: does the routed hybrid
(per-domain best) beat both pure nests?

Arms (routing = same full-cov descriptor router as pcll_nested):
  phasecyte  — routed nest of phasecyte leaves (= pcll_nested numbers)
  trioron    — routed nest of dream-distilled gradient leaves
  hybrid     — routed, per-domain ORACLE pick of the better leaf
               (learned arbitration is the next iteration)

Trioron leaf recipe per s048 diagnosis: nonlinear=True (quad dendrite),
input = phasecyte pockets/1000 (the shared receptor field is the
common currency; logits->quanta is the reverse direction, later).

Run:  PCLL15_SENSE=gabor python3 -m experiments.progenitor.hybrid_nest
Env:  PCLL15_* as usual; HYB_PSEUDO (pseudo-samples/class, def 1500),
      HYB_EPOCHS (def 8), HYB_HIDDEN (def 64), HYB_SEED (def 0).
"""
from __future__ import annotations

import os
import time

import torch

from trioron.pcll import dream_distill as pkg_dream_distill
from trioron.pcll import dreamed_predict

from experiments.progenitor import run_pcll_chained15 as base
from experiments.progenitor.pcll_nested import run_seed, DOMAINS

PSEUDO = int(os.environ.get("HYB_PSEUDO", "1500"))
EPOCHS = int(os.environ.get("HYB_EPOCHS", "8"))
HIDDEN = int(os.environ.get("HYB_HIDDEN", "64"))
SEED = int(os.environ.get("HYB_SEED", "0"))
MIN_SKETCH_N = 8          # skip sketches with fewer real deposits


def dream_distill(leaf, names: torch.Tensor, dom: int, seed: int):
    """Promoted to trioron.pcll.nest (s049) — thin shim keeping the
    experiment's RNG streams (5000/6000 bases) and per-domain labels."""
    dom_labels = sorted(range(10 * dom, 10 * dom + 10))
    return pkg_dream_distill(leaf, names, dom_labels, pseudo=PSEUDO,
                             epochs=EPOCHS, hidden=HIDDEN, seed=seed,
                             min_sketch=MIN_SKETCH_N)


def main() -> None:
    print(f"HYBRID NEST (s049) — phasecyte wake + dream-distilled trioron "
          f"leaves; sense={base.SENSE}, pseudo={PSEUDO}/class, "
          f"epochs={EPOCHS}, hidden={HIDDEN}, seed={SEED}")
    t0 = time.time()
    ta, full, leaves, router = run_seed(SEED, return_state=True)
    sense = base.make_sense(SEED)

    from trioron.legacy.donorkit.datasets import (DatasetBundle,
                                                  chained_15_specs,
                                                  build_task_views)
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"],
                           root=base.DATA_ROOT, n_holdout_per_dataset=0)
    specs = chained_15_specs()[:base.N_TASKS]
    eval_views = build_task_views(bundle, specs, split="test")
    Xt = torch.cat([v.images for v in eval_views])
    yt = torch.cat([v.labels_global for v in eval_views])
    dom_true = torch.cat([torch.full((len(v.labels_global),), ti // 5)
                          for ti, v in enumerate(eval_views)])
    dom_pred = []
    for i in range(0, len(Xt), 2000):
        dom_pred.append(router.route_class(
            sense(Xt[i:i + 2000]), class_ids=list(range(len(leaves))),
            n_classes=len(leaves)))
    dom_pred = torch.cat(dom_pred)

    # phasecyte per-sample predictions (centered readout) per leaf
    pc_pred, gr_pred = [], []
    pc_dom_acc, gr_dom_acc = [], []
    for dom, leaf in enumerate(leaves):
        names = base.names_of(leaf.mixed)
        _, Ec = base.evidence_both(leaf.mixed, sense, Xt)
        pc = names[Ec.argmax(1)]
        sub, dom_labels = dream_distill(leaf, names, dom, SEED)
        gr = dreamed_predict(sub, leaf, sense, Xt, dom_labels)
        pc_pred.append(pc)
        gr_pred.append(gr)
        m = dom_true == dom
        pc_dom_acc.append(float((pc[m] == yt[m]).float().mean()))
        gr_dom_acc.append(float((gr[m] == yt[m]).float().mean()))
        print(f"  [{DOMAINS[dom]}] own-domain acc: phasecyte "
              f"{pc_dom_acc[-1]:.3f}  trioron(dreamed) "
              f"{gr_dom_acc[-1]:.3f}", flush=True)

    def routed(preds, pick) -> float:
        out = torch.empty(len(Xt), dtype=torch.long)
        for dom in range(len(leaves)):
            m = dom_pred == dom
            if m.any():
                out[m] = preds[pick(dom)][dom][m] \
                    if isinstance(preds, dict) else preds[dom][m]
        return float((out == yt).float().mean())

    full_pc = routed(pc_pred, None)
    full_gr = routed(gr_pred, None)
    both = {"pc": pc_pred, "gr": gr_pred}
    best = ["pc" if pc_dom_acc[d] >= gr_dom_acc[d] else "gr"
            for d in range(len(leaves))]
    full_hy = routed(both, lambda d: best[d])
    print(f"\n[seed {SEED}] routed full: phasecyte={full_pc:.3f}  "
          f"trioron(dreamed)={full_gr:.3f}  hybrid(oracle-pick "
          f"{'/'.join(best)})={full_hy:.3f}")
    print(f"[seed {SEED}] reference: nested phasecyte n=3 "
          f"0.474±0.031; monolithic s039 0.446  "
          f"elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
