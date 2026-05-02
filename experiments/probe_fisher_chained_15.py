"""Probe: Fisher / lambda distribution across the chained-15 curriculum.

Question this answers (from Rocky's epigenetic-lock hypothesis,
2026-05-02 late):

    Are the head columns sitting AT the LAMBDA_FLOOR (=0.1) — i.e. the
    floor IS the only thing anchoring them — or well above it (Fisher
    mass anchors them; the floor is moot)?

    If at floor, lowering it (Rocky's proposed 0.001 or dynamic ε)
    frees old-class head rows to drift more under each new task →
    likely WORSENS head bias, opposite of intent.

    If above, the change only affects unused parameters and the
    framing ("epigenetic baseline only, real Fisher does the work")
    holds cleanly.

Mechanism we're inspecting:

    Each consolidate_task: estimate_fisher resets Fisher, re-estimates
    on current task's data only, then update_lambda_all collapses
    fisher_W to per-node lam = fisher_W.mean(dim=1), then clamp to
    LAMBDA_FLOOR. Old-class head rows have NEAR-ZERO fisher (current
    task's data doesn't activate them) → lam clamps to floor.

How we run:

    Single seed of grown_capped_dream, smoke epoch count (4 epochs/task)
    so we see the full 15-task chained curriculum in ~90s. We monkey-
    patch consolidate_task to log per-layer lambda stats and per-class
    head drift after every consolidation.

Output:

    Per-task table (per consolidation):
      - head: lam_min / lam_p50 / lam_max / frac_at_floor / max_drift
      - L1:   lam_min / lam_p50 / lam_p90 / lam_max / frac_at_floor
    Per-class head lam vector at end of curriculum (one row per class).

Run:

    python3 -m experiments.probe_fisher_chained_15 \
        > outputs/probe_fisher_chained_15.log 2>&1
"""
from __future__ import annotations
import os
import sys
from typing import List, Sequence

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.bench_chained_15task as bench
from experiments.datasets import (
    DEFAULT_DATA_ROOT,
    DatasetBundle,
    build_task_views,
    chained_15_specs,
)


# --- logger state ---------------------------------------------------------


_probe_state = {
    "consolidations": [],   # one dict per consolidation call
}


def _summarize_lam(lam: torch.Tensor, floor: float) -> dict:
    """Return min / p10 / p50 / p90 / max, plus fraction at floor and mean.
    `floor` is the LAMBDA_FLOOR value used by the bench (clamp_min target).
    "At floor" means within 1e-7 of `floor` (the clamp lands exactly).
    """
    if lam.numel() == 0:
        return {"n": 0}
    finite = lam.detach().float().cpu()
    qs = torch.tensor([0.0, 0.10, 0.50, 0.90, 1.0])
    quants = torch.quantile(finite, qs).tolist()
    at_floor = (finite <= floor + 1e-7).float().mean().item()
    above_10x = (finite > floor * 10).float().mean().item()
    return {
        "n": int(finite.numel()),
        "min": quants[0], "p10": quants[1], "p50": quants[2],
        "p90": quants[3], "max": quants[4],
        "mean": float(finite.mean()),
        "frac_at_floor": at_floor,
        "frac_above_10x_floor": above_10x,
    }


def _row_drift(layer) -> torch.Tensor:
    """Per-output-row L2 drift ||W[c] - W_anchor[c]||_2. Returns (n_nodes,)."""
    return (layer.W.detach() - layer.W_anchor.detach()).pow(2).sum(dim=1).sqrt()


def _per_class_head_lam(net) -> List[float]:
    """Return the head's lam vector as a Python list (one per output class)."""
    head = net.layers[-1]
    return head.lam.detach().float().cpu().tolist()


def _summarize_fisher(fisher_W: torch.Tensor) -> dict:
    """Per-weight Fisher distribution (NOT collapsed to per-node mean).
    Tells us whether the Fisher signal would survive a lower lambda floor."""
    if fisher_W.numel() == 0:
        return {"n": 0}
    finite = fisher_W.detach().float().abs().cpu().flatten()
    qs = torch.tensor([0.0, 0.50, 0.90, 0.99, 1.0])
    quants = torch.quantile(finite, qs).tolist()
    return {
        "n": int(finite.numel()),
        "min": quants[0], "p50": quants[1], "p90": quants[2],
        "p99": quants[3], "max": quants[4],
        "mean": float(finite.mean()),
        "frac_gt_1e-4": float((finite > 1e-4).float().mean()),
        "frac_gt_1e-3": float((finite > 1e-3).float().mean()),
        "frac_gt_1e-2": float((finite > 1e-2).float().mean()),
    }


def _log_consolidation(
    net, task_idx: int, task_name: str, active_classes: Sequence[int],
):
    floor = bench.LAMBDA_FLOOR
    # The growth-target layer (L1) and the head are what we care about.
    # L0 is frozen so its Fisher / lam are immaterial.
    head = net.layers[-1]
    l1 = net.layers[bench.GROWTH_TARGET_LAYER_IDX]
    head_summary = _summarize_lam(head.lam, floor)
    l1_summary = _summarize_lam(l1.lam, floor)
    head_fisher = _summarize_fisher(head.fisher_W)
    l1_fisher = _summarize_fisher(l1.fisher_W)
    head_drift = _row_drift(head)
    l1_drift = _row_drift(l1)
    head_lam = head.lam.detach().float().cpu().tolist()

    record = {
        "task_idx": task_idx,
        "task_name": task_name,
        "active_classes": list(active_classes),
        "head_n_classes": int(head.n_nodes),
        "head_lam_min": head_summary["min"],
        "head_lam_p50": head_summary["p50"],
        "head_lam_max": head_summary["max"],
        "head_frac_at_floor": head_summary["frac_at_floor"],
        "head_max_drift": float(head_drift.max()) if head_drift.numel() else 0.0,
        "head_p50_drift": float(head_drift.median()) if head_drift.numel() else 0.0,
        "l1_n_nodes": int(l1.n_nodes),
        "l1_lam_min": l1_summary["min"],
        "l1_lam_p50": l1_summary["p50"],
        "l1_lam_p90": l1_summary["p90"],
        "l1_lam_max": l1_summary["max"],
        "l1_frac_at_floor": l1_summary["frac_at_floor"],
        "l1_frac_above_10x_floor": l1_summary["frac_above_10x_floor"],
        "head_lam_full": head_lam,
        "head_fisher": head_fisher,
        "l1_fisher": l1_fisher,
    }
    _probe_state["consolidations"].append(record)

    print(
        f"  [PROBE consol task {task_idx+1:>2d} {task_name:<22s} "
        f"active={list(active_classes)}]"
    )
    print(
        f"    head ({head_summary['n']} cls): "
        f"lam min={head_summary['min']:.4g}  p50={head_summary['p50']:.4g}  "
        f"max={head_summary['max']:.4g}  "
        f"@floor={head_summary['frac_at_floor']*100:.0f}%  "
        f"max_drift={float(head_drift.max()):.4f}"
    )
    print(
        f"    L1   ({l1_summary['n']} nodes): "
        f"lam min={l1_summary['min']:.4g}  p50={l1_summary['p50']:.4g}  "
        f"p90={l1_summary['p90']:.4g}  max={l1_summary['max']:.4g}  "
        f"@floor={l1_summary['frac_at_floor']*100:.0f}%  "
        f">10×floor={l1_summary['frac_above_10x_floor']*100:.0f}%  "
        f"max_drift={float(l1_drift.max()):.4f}"
    )
    print(
        f"    head fisher_W (per-weight): "
        f"p50={head_fisher['p50']:.3g}  p90={head_fisher['p90']:.3g}  "
        f"p99={head_fisher['p99']:.3g}  max={head_fisher['max']:.3g}  "
        f">1e-4={head_fisher['frac_gt_1e-4']*100:.0f}%  "
        f">1e-3={head_fisher['frac_gt_1e-3']*100:.0f}%  "
        f">1e-2={head_fisher['frac_gt_1e-2']*100:.0f}%"
    )
    print(
        f"    L1   fisher_W (per-weight): "
        f"p50={l1_fisher['p50']:.3g}  p90={l1_fisher['p90']:.3g}  "
        f"p99={l1_fisher['p99']:.3g}  max={l1_fisher['max']:.3g}  "
        f">1e-4={l1_fisher['frac_gt_1e-4']*100:.0f}%  "
        f">1e-3={l1_fisher['frac_gt_1e-3']*100:.0f}%  "
        f">1e-2={l1_fisher['frac_gt_1e-2']*100:.0f}%"
    )


# --- monkey-patch the bench's consolidate_task --------------------------


_orig_consolidate = bench.consolidate_task
_consolidation_counter = {"n": 0, "current_task_name": "?",
                          "current_active": []}


def consolidate_task_logged(net, train_view, active_classes):
    _orig_consolidate(net, train_view, active_classes)
    n = _consolidation_counter["n"]
    _log_consolidation(
        net, task_idx=n,
        task_name=getattr(train_view, "name", "?"),
        active_classes=active_classes,
    )
    _consolidation_counter["n"] += 1


bench.consolidate_task = consolidate_task_logged


# --- run --------------------------------------------------------------


def run_probe(seed: int = 0):
    print("=" * 78)
    print("Fisher / λ probe — chained-15 grown_capped_dream, single seed")
    print("=" * 78)
    print(f"seed: {seed}   epochs/task: {bench.N_EPOCHS_PER_TASK_SMOKE} (smoke)")
    print(f"LAMBDA_FLOOR (current): {bench.LAMBDA_FLOOR}")
    print(f"EWC_INTERTASK: {bench.EWC_INTERTASK}")
    print()

    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=DEFAULT_DATA_ROOT,
        n_holdout_per_dataset=bench.N_INFANCY_PER_DATASET,
    )
    specs = chained_15_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [s.global_classes for s in specs]

    r = bench.run_arm(
        "grown_capped_dream",
        seed=seed + (hash("grown_capped_dream") % 7919),
        n_epochs_per_task=bench.N_EPOCHS_PER_TASK_SMOKE,
        train_views=train_views, eval_views=eval_views,
        task_class_lists=task_class_lists,
        infancy_view=None,
        n_passes=1,
    )

    # End-of-curriculum dump
    print()
    print("=" * 78)
    print(f"End-of-curriculum head λ per class (n_classes={len(specs)*2}):")
    print("=" * 78)
    if _probe_state["consolidations"]:
        last = _probe_state["consolidations"][-1]
        for c, lam_c in enumerate(last["head_lam_full"]):
            domain = "MNIST" if c < 10 else "Fashion" if c < 20 else "EMNIST"
            tag = "  ← AT FLOOR" if lam_c <= bench.LAMBDA_FLOOR + 1e-7 else ""
            print(f"  class {c:>2d} ({domain:<8s}): lam = {lam_c:.6g}{tag}")

    # Per-task evolution table — head lam summary
    print()
    print("=" * 78)
    print("Per-task evolution: head λ summary")
    print("=" * 78)
    print(f"  {'tk':>2s}  {'task_name':<20s}  {'n_cls':>5s}  "
          f"{'min':>10s}  {'p50':>10s}  {'max':>10s}  {'@floor':>6s}  "
          f"{'max_drift':>9s}")
    for rec in _probe_state["consolidations"]:
        print(
            f"  {rec['task_idx']+1:>2d}  {rec['task_name']:<20s}  "
            f"{rec['head_n_classes']:>5d}  "
            f"{rec['head_lam_min']:>10.4g}  "
            f"{rec['head_lam_p50']:>10.4g}  "
            f"{rec['head_lam_max']:>10.4g}  "
            f"{rec['head_frac_at_floor']*100:>5.0f}%  "
            f"{rec['head_max_drift']:>9.4f}"
        )

    print()
    print("=" * 78)
    print("Per-task evolution: L1 (growth-target hidden) λ summary")
    print("=" * 78)
    print(f"  {'tk':>2s}  {'task_name':<20s}  {'n_h':>4s}  "
          f"{'min':>10s}  {'p50':>10s}  {'p90':>10s}  {'max':>10s}  "
          f"{'@floor':>6s}  {'>10x':>5s}")
    for rec in _probe_state["consolidations"]:
        print(
            f"  {rec['task_idx']+1:>2d}  {rec['task_name']:<20s}  "
            f"{rec['l1_n_nodes']:>4d}  "
            f"{rec['l1_lam_min']:>10.4g}  "
            f"{rec['l1_lam_p50']:>10.4g}  "
            f"{rec['l1_lam_p90']:>10.4g}  "
            f"{rec['l1_lam_max']:>10.4g}  "
            f"{rec['l1_frac_at_floor']*100:>5.0f}%  "
            f"{rec['l1_frac_above_10x_floor']*100:>4.0f}%"
        )

    print()
    print("=" * 78)
    print("Per-task evolution: raw fisher_W (per-weight, BEFORE mean-collapse)")
    print("=" * 78)
    print(f"  {'tk':>2s}  {'task_name':<20s}  layer  "
          f"{'p50':>9s}  {'p90':>9s}  {'p99':>9s}  {'max':>9s}  "
          f"{'>1e-4':>5s}  {'>1e-3':>5s}  {'>1e-2':>5s}")
    for rec in _probe_state["consolidations"]:
        for layer_label, key in [("head", "head_fisher"), ("L1  ", "l1_fisher")]:
            f = rec[key]
            print(
                f"  {rec['task_idx']+1:>2d}  {rec['task_name']:<20s}  "
                f"{layer_label}  "
                f"{f['p50']:>9.3g}  {f['p90']:>9.3g}  {f['p99']:>9.3g}  "
                f"{f['max']:>9.3g}  "
                f"{f['frac_gt_1e-4']*100:>4.0f}%  "
                f"{f['frac_gt_1e-3']*100:>4.0f}%  "
                f"{f['frac_gt_1e-2']*100:>4.0f}%"
            )

    print()
    print("Headline result on this seed (sanity-check vs n=12 means):")
    print(f"  full   {r['final_accuracy']:.4f}")
    print(f"  domain {r['final_accuracy_domain']:.4f}")
    print(f"  task   {r['final_accuracy_aware']:.4f}")
    return r


if __name__ == "__main__":
    run_probe()
