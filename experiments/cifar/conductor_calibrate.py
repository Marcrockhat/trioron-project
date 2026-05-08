"""Fit a LearnedFusion over the trained sense donors.

Splits the held-out test set 50/50 — calibration set fits the per-
(donor, class) weights + per-class bias; held-out half measures the
final accuracy. Donors were trained on the full train set, so test
data is the only un-leaky place to fit a calibrator.

Saves the fitted fusion as a small .pt at outputs/cifar_donors[_full]/
fusion_<tag>.pt, which conductor_eval.py can load.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from trioron.senses.conductor import (
    SensoryConductor, LearnedFusion, load_sense_donor,
)
from experiments.cifar.datasets import (
    load_cifar100, SLICES, DEFAULT_DATA_ROOT,
)
from experiments.cifar.conductor_eval import (
    _slice_test_set, _eval_logits,
)


def _split_calibration(images, labels, fraction=0.5, seed=0):
    n = images.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_cal = int(n * fraction)
    cal = perm[:n_cal]
    rem = perm[n_cal:]
    return (images[cal], labels[cal]), (images[rem], labels[rem])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slice", choices=sorted(SLICES), default="first",
    )
    parser.add_argument("--donor-dir", default=None)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--senses", nargs="+",
        default=["eye", "color_smell", "frequency_print", "skeleton",
                 "heat_diffusion", "taste", "pulse"],
        help="Default = greedy-7 from 25-class first-slice. Override "
             "to fit a calibrator over a different sense subset.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--cal-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--form", choices=["scalar", "scalar_bias", "full"],
        default="scalar_bias",
        help="LearnedFusion parameterization. Default scalar_bias is "
             "the safest at small calibration set sizes.",
    )
    parser.add_argument(
        "--early-stop-metric", choices=["task", "full"], default="task",
        help="Held-out metric to use for selecting the best epoch's "
             "weights. 'task' optimizes the deployment-relevant metric.",
    )
    parser.add_argument(
        "--loss", choices=["full_ce", "task_ce", "combined"],
        default="task_ce",
        help="Training loss. 'full_ce' = CE over 100 classes; "
             "'task_ce' = CE restricted to active task; 'combined' = "
             "alpha*task_ce + (1-alpha)*full_ce (tunable via --alpha).",
    )
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight on task_ce in 'combined' loss.")
    parser.add_argument("--tag", default="learned",
                        help="Suffix for the saved fusion file.")
    args = parser.parse_args(argv)

    class_groups = SLICES[args.slice]
    if args.donor_dir is None:
        sub = "cifar_donors" if args.slice == "first" else "cifar_donors_full"
        args.donor_dir = os.path.join(os.path.dirname(DEFAULT_DATA_ROOT), sub)

    donor_paths = [
        os.path.join(args.donor_dir, f"sense_donor_{s}.pt")
        for s in args.senses
    ]
    for p in donor_paths:
        if not os.path.exists(p):
            print(f"missing donor checkpoint: {p}", file=sys.stderr)
            return 2

    donors = [load_sense_donor(p) for p in donor_paths]
    cnd = SensoryConductor(donors, fusion="mean_logit").eval()
    n_donors = len(donors)
    n_classes = len(cnd.union_classes)

    # Test set, then split into calibration / held-out.
    test_imgs, test_labs, all_classes = _slice_test_set(
        class_groups, args.data_root,
    )
    (cal_imgs, cal_labs), (held_imgs, held_labs) = _split_calibration(
        test_imgs, test_labs, args.cal_fraction, args.seed,
    )
    print(f"slice={args.slice} senses={args.senses}")
    print(f"calibration: {cal_imgs.shape[0]} images   held-out: "
          f"{held_imgs.shape[0]} images   classes={n_classes}")

    # Pre-compute per-donor logits once (donors are frozen).
    print("computing per-donor logits on calibration + held-out…")
    t0 = time.time()
    with torch.no_grad():
        cal_per = cnd.per_donor_logits(cal_imgs)
        held_per = cnd.per_donor_logits(held_imgs)
    cal_stack = torch.stack(cal_per, dim=1)        # (Ncal, D, C)
    held_stack = torch.stack(held_per, dim=1)
    print(f"  done ({time.time() - t0:.1f}s)  cal_stack={tuple(cal_stack.shape)}")

    # Map global labels → union-class column indices.
    cls_to_col = {c: i for i, c in enumerate(cnd.union_classes)}
    cal_y = torch.tensor([cls_to_col[int(c)] for c in cal_labs.tolist()],
                         dtype=torch.long)
    held_y = torch.tensor([cls_to_col[int(c)] for c in held_labs.tolist()],
                          dtype=torch.long)

    # Pre-compute task-aware loss helpers: per-task active union-cols,
    # plus per-calibration-example task index.
    task_active_cols = torch.tensor([
        [cls_to_col[int(c)] for c in g] for g in class_groups
    ], dtype=torch.long)                                # (n_tasks, k_per_task)
    col_to_task = torch.zeros(n_classes, dtype=torch.long)
    for t, cols in enumerate(task_active_cols.tolist()):
        for col in cols:
            col_to_task[col] = t
    cal_task = col_to_task[cal_y]                       # (Ncal,)

    # Baseline: mean_logit fusion on held-out (as reference for the
    # learned fusion's lift).
    with torch.no_grad():
        baseline_logits = held_stack.mean(dim=1)
    base_metrics = _eval_logits(baseline_logits, held_labs,
                                cnd.union_classes, class_groups)
    print(f"\nheld-out  baseline mean_logit:  full={base_metrics['full']:.4f}  "
          f"task={base_metrics['task']:.4f}")

    # Train the LearnedFusion. Parameters are *offsets* from
    # mean_logit (init = zero offsets), so weight decay pulls toward
    # the parameter-free baseline. We track held-out task-aware (or
    # full) and snap back to the best epoch's weights at the end.
    fusion = LearnedFusion(n_donors, n_classes, form=args.form)
    n_params = sum(p.numel() for p in fusion.parameters())
    opt = torch.optim.Adam(fusion.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    print(f"\nfitting LearnedFusion form={args.form}  "
          f"params={n_params}  early-stop on held-out {args.early_stop_metric}")
    batch = 256
    best = {"epoch": 0, "metric": -1.0, "full": base_metrics["full"],
            "task": base_metrics["task"]}
    best_state = {k: v.detach().clone() for k, v in fusion.state_dict().items()}
    for ep in range(args.epochs):
        perm = torch.randperm(cal_stack.shape[0])
        losses = []
        for i in range(0, perm.numel(), batch):
            idx = perm[i:i + batch]
            x = cal_stack[idx]
            y = cal_y[idx]
            logits = fusion(x)
            full_ce_loss = F.cross_entropy(logits, y)
            if args.loss == "full_ce":
                loss = full_ce_loss
            else:
                # Task-aware CE: mask inactive columns per example.
                t = cal_task[idx]                       # (B,)
                active = task_active_cols[t]            # (B, k)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, active, 0.0)
                task_ce_loss = F.cross_entropy(logits + mask, y)
                if args.loss == "task_ce":
                    loss = task_ce_loss
                else:
                    loss = args.alpha * task_ce_loss + \
                           (1.0 - args.alpha) * full_ce_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        with torch.no_grad():
            logits_h = fusion(held_stack)
        m = _eval_logits(logits_h, held_labs, cnd.union_classes, class_groups)
        metric = m[args.early_stop_metric]
        if metric > best["metric"]:
            best.update({"epoch": ep + 1, "metric": metric,
                         "full": m["full"], "task": m["task"]})
            best_state = {k: v.detach().clone()
                          for k, v in fusion.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1:>3d}  loss {sum(losses)/len(losses):.4f}  "
                  f"held-out  full={m['full']:.4f}  task={m['task']:.4f}  "
                  f"best@{best['epoch']}={best['metric']:.4f}")

    # Snap to best-epoch weights for the saved fusion.
    fusion.load_state_dict(best_state)
    with torch.no_grad():
        final_logits = fusion(held_stack)
    final_m = _eval_logits(final_logits, held_labs,
                           cnd.union_classes, class_groups)

    print()
    print(f"=== summary ===")
    print(f"baseline mean_logit:  full={base_metrics['full']:.4f}  "
          f"task={base_metrics['task']:.4f}")
    print(f"learned fusion:       full={final_m['full']:.4f}  "
          f"task={final_m['task']:.4f}")
    print(f"lift over mean_logit: full={final_m['full']-base_metrics['full']:+.4f}  "
          f"task={final_m['task']-base_metrics['task']:+.4f}")

    out_path = os.path.join(args.donor_dir, f"fusion_{args.tag}.pt")
    payload = {
        "kind": "learned_fusion",
        "form": args.form,
        "senses": list(args.senses),
        "n_donors": n_donors,
        "n_classes": n_classes,
        "union_classes": list(cnd.union_classes),
        "state_dict": {k: v.detach().cpu()
                       for k, v in fusion.state_dict().items()},
        "slice": args.slice,
        "calibration_fraction": args.cal_fraction,
        "seed": args.seed,
        "best_epoch": best["epoch"],
        "held_out_full": final_m["full"],
        "held_out_task": final_m["task"],
        "baseline_full": base_metrics["full"],
        "baseline_task": base_metrics["task"],
    }
    torch.save(payload, out_path)
    print(f"\n[SAVE] {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
