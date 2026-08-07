"""H-codes shift-detection probe — does the ShiftDetector inherit the H-space win?

s047 real-stream finding: in raw pixel space the detector catches DATASET-level
shifts (MNIST->Fashion) but misses within-dataset class-pair boundaries —
per-sample pixel variance drowns the pair-level mean shift.  Hypothesis (the
routing lesson): the stable interior code makes those boundaries visible.

Method: train a chained-15 substrate via the archived bench (same shim as
experiments/validate_router_promotion.py), then at the final pure-H eval stream
held-out test samples task-by-task through the FROZEN substrate and run TWO
detectors on the identical stream order: one on H-codes, one on raw pixels
(control).  Report boundaries detected (within a 20-sample window) and false
fires for each.

Gate: H-codes must detect MORE within-dataset boundaries than pixels at no
false-fire cost.  This is the de-risk step before wiring the detector into a
live training loop.

Usage: python3 experiments/shift_h_probe.py [--smoke --seed 42 ...]
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments  # noqa: E402
import trioron.legacy.donorkit.datasets as _ds  # noqa: E402
sys.modules["experiments.datasets"] = _ds
experiments.datasets = _ds

import torch  # noqa: E402

from trioron.learning import ShiftDetector  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "bench_v2", ROOT / "archive" / "experiments" / "bench_chained_15_v2.py")
bench = importlib.util.module_from_spec(_spec)
sys.modules["bench_v2"] = bench
_spec.loader.exec_module(bench)

_orig_eval = bench.evaluate_all_tasks

SAMPLES_PER_TASK = 200
DETECT_WINDOW = 20


def _stream_probe(sub, bundle, specs, h_interior_ids):
    """Stream test samples task-by-task; run H-code and pixel detectors."""
    torch.manual_seed(0)
    h_rows, px_rows, boundaries = [], [], []
    for t_idx, spec in enumerate(specs):
        view = bundle.task_view(spec.dataset_name, spec.local_classes,
                                spec.global_classes, split="test",
                                task_name=spec.name)
        x, _y = view.all_examples()
        x = x[torch.randperm(x.shape[0])[:SAMPLES_PER_TASK]]
        if t_idx > 0:
            boundaries.append(t_idx * SAMPLES_PER_TASK)
        with torch.no_grad():
            _ = sub(x)
            h_rows.append(sub.last_activations[:, h_interior_ids].clone())
        px_rows.append(x[:, :784].clone())

    results = {}
    for name, rows in (("H-codes", h_rows), ("pixels", px_rows)):
        det = ShiftDetector(dim=rows[0].shape[1])
        for block in rows:
            for row in block:
                det.update(row)
        hits = [e.step for e in det.events]
        tp = sum(1 for b in boundaries
                 if any(0 <= h - b <= DETECT_WINDOW for h in hits))
        fp = sum(1 for h in hits
                 if not any(0 <= h - b <= DETECT_WINDOW for b in boundaries))
        results[name] = (tp, fp, hits)
        print(f"  [{name:8s}] boundaries={len(boundaries)}  detected={tp}  "
              f"false={fp}  events={hits}")
    print(f"  true boundaries: {boundaries}")
    h_tp, px_tp = results["H-codes"][0], results["pixels"][0]
    print(f"  [PROBE] H-codes {h_tp}/{len(boundaries)} vs pixels "
          f"{px_tp}/{len(boundaries)} -> "
          f"{'H-SPACE WIN' if h_tp > px_tp else 'NO WIN'}")


def evaluate_all_tasks(sub, bundle, specs, tasks_seen, archive=None, detectors=None,
                       h_archive=None, h_interior_ids=None, h_route_mode="task",
                       h_full_cov=False):
    ev = _orig_eval(sub, bundle, specs, tasks_seen, archive, detectors,
                    h_archive=h_archive, h_interior_ids=h_interior_ids,
                    h_route_mode=h_route_mode, h_full_cov=h_full_cov)
    is_final_pure_h = (h_archive is not None and archive is None
                      and detectors is None and tasks_seen == len(specs))
    if is_final_pure_h:
        print("\n--- H-codes shift-detection probe (frozen substrate) ---")
        _stream_probe(sub, bundle, specs, h_interior_ids)
    return ev


bench.evaluate_all_tasks = evaluate_all_tasks

if __name__ == "__main__":
    argv = sys.argv[1:]
    for required in ("--h-routing", "--refresh-h", "--full-cov"):
        if required not in argv:
            argv.append(required)
    sys.argv = ["bench_chained_15_v2.py"] + argv
    raise SystemExit(bench.main())
