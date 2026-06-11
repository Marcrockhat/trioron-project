"""GCU-instability diagnosis (s028): WHY does long training crash the deep columns?

Hypothesis: σ(z)=z·cos z is near-linear only for small z; once |z| grows past ~π/2,
cos z flips and z·cos z oscillates with unbounded amplitude (the z² envelope). In a
stack of GCU cells the magnitude compounds with depth, so long training drives the deep
columns' activations up until they blow up — the step-3000 accuracy crash.

This grows + freezes the organism, then long-trains while recording, per checkpoint:
  • test accuracy,
  • max |activation| over the whole net,
  • the deepest column's |activation| BY LAYER (does it amplify with depth?),
  • the largest GCU edge weight and the total gradient norm.
If the crash coincides with the deep-layer activations exploding, the mechanism is
confirmed and the fix is bounding z (clip / squash / depth cap), not the signal.

Run: python3 -m experiments.progenitor.run_gcu_diag
"""
from __future__ import annotations

import io
import contextlib

import torch
import torch.nn.functional as F

from .data_hard import make_split, bayes_accuracy
from .step5_branch import build_base, grow_class_driven
from .step3c_council_decides import measure, LR
from trioron.core.epigenome import DENDRITE


def main() -> None:
    tr, te = make_split()
    torch.manual_seed(0)
    sub = build_base(n_in=tr.x.shape[1], n_out=len(te.names))
    with contextlib.redirect_stdout(io.StringIO()):
        grow_class_driven(sub, tr, te, te.names, bayes_accuracy(te), max_events=80)

    a = sub.arena
    # deepest column and its cells in layer order
    deepest = max(sub.branch_cells, key=lambda b: len(sub.branch_cells[b]))
    col = sub.branch_cells[deepest]                       # [L1, L2, ... ] cell ids in order
    layers = [sub.cell_layer[c] for c in col]
    is_gcu = [bool(int(a.epigenome[c].item()) & (1 << DENDRITE)) for c in col]
    print(f"  deepest column = branch {deepest}, depth {len(col)}")
    print(f"    cells {col}")
    print(f"    layers {layers}   GCU? {is_gcu}\n")

    sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=LR)

    def snapshot(step):
        acc = measure(sub, te, te.names)[2]
        with torch.no_grad():
            sub(tr.x)
        act = sub.last_activations                       # [batch, capacity]
        amax = float(act[:, :a.cursor].abs().max())
        per_layer = [float(act[:, c].abs().max()) for c in col]
        # largest GCU edge weight (edges into any dendrite cell in this column)
        gcu_cells = [c for c, g in zip(col, is_gcu) if g]
        ec = a.edge_cursor
        dstmask = torch.isin(a.edge_dst[:ec], torch.tensor(gcu_cells, dtype=a.edge_dst.dtype))
        wmax = float(a.edge_weight[:ec][dstmask].abs().max()) if dstmask.any() else 0.0
        layerstr = " ".join(f"{l}:{v:7.1f}" for l, v in zip(layers, per_layer))
        print(f"    step {step:>4}  acc {acc:.3f}  max|act| {amax:9.1f}  GCUwmax {wmax:6.2f}  "
              f"[by layer {layerstr}]")

    snapshot(0)
    for k in range(1, 6001):
        opt.zero_grad()
        loss = F.cross_entropy(sub(tr.x), tr.y)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1e9))  # measure, don't clip
        opt.step()
        if k % 250 == 0:
            snapshot(k)
            print(f"        (grad norm {gnorm:.1f}, loss {float(loss):.3f})")


if __name__ == "__main__":
    main()
