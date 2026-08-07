"""Validate the promoted ManifoldRouter against the archived chained-15 v2 bench.

Runs archive/experiments/bench_chained_15_v2.py (the source of the H-routing
numbers: full 0.55 -> 0.68 diag -> 0.76 full-cov oracle / ~0.69 storage-free)
behind an import shim, and monkeypatches ``evaluate_all_tasks`` so the final
post-refresh eval ALSO routes with ``trioron.learning.router.ManifoldRouter``
on the same substrate + H-archive.  The promoted router must reproduce the
bench's full accuracy exactly (same argmax math, deterministic eval) — any
disagreement is a porting bug.

Usage:
    python3 experiments/validate_router_promotion.py --refresh-source real \
        [--seed 42 --epochs 4 --h-route-mode task] [passthrough bench flags]
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Shim: the archive predates the donorkit migration
import experiments  # noqa: E402
import trioron.legacy.donorkit.datasets as _ds  # noqa: E402
sys.modules["experiments.datasets"] = _ds
experiments.datasets = _ds

import torch  # noqa: E402

from trioron.learning.router import ManifoldRouter  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "bench_v2", ROOT / "archive" / "experiments" / "bench_chained_15_v2.py")
bench = importlib.util.module_from_spec(_spec)
sys.modules["bench_v2"] = bench
_spec.loader.exec_module(bench)

_orig_eval = bench.evaluate_all_tasks


def _promoted_full_acc(sub, bundle, specs, tasks_seen, h_archive, h_interior_ids,
                       h_route_mode, h_full_cov):
    """Re-evaluate mean full accuracy using the promoted router."""
    router = ManifoldRouter(h_archive, full_cov=h_full_cov)
    groups = [list(specs[t].global_classes) for t in range(tasks_seen)]
    accs = []
    for t_idx in range(tasks_seen):
        spec = specs[t_idx]
        view = bundle.task_view(spec.dataset_name, spec.local_classes,
                                spec.global_classes, split="test",
                                task_name=spec.name)
        x, y = view.all_examples()
        with torch.no_grad():
            logits = sub(x)
            codes = sub.last_activations[:, h_interior_ids]
            if h_route_mode == "class":
                pred = router.route_class(codes, n_classes=bench.N_GLOBAL_CLASSES)
            else:
                pred = router.route_prediction(codes, logits, groups)
        accs.append((pred == y).float().mean().item())
    return sum(accs) / len(accs)


def evaluate_all_tasks(sub, bundle, specs, tasks_seen, archive=None, detectors=None,
                       h_archive=None, h_interior_ids=None, h_route_mode="task",
                       h_full_cov=False):
    ev = _orig_eval(sub, bundle, specs, tasks_seen, archive, detectors,
                    h_archive=h_archive, h_interior_ids=h_interior_ids,
                    h_route_mode=h_route_mode, h_full_cov=h_full_cov)
    is_final_pure_h = (h_archive is not None and archive is None
                      and detectors is None and tasks_seen == len(specs))
    if is_final_pure_h:
        promoted = _promoted_full_acc(sub, bundle, specs, tasks_seen, h_archive,
                                      h_interior_ids, h_route_mode, h_full_cov)
        match = abs(promoted - ev.mean_full) < 1e-9
        print(f"\n  [VALIDATE] bench full={ev.mean_full:.6f}  "
              f"promoted ManifoldRouter full={promoted:.6f}  "
              f"-> {'MATCH' if match else 'MISMATCH'}")
        if not match:
            print("  [VALIDATE] PORTING BUG — promoted router disagrees with bench")
    return ev


bench.evaluate_all_tasks = evaluate_all_tasks

if __name__ == "__main__":
    # Force the pure-H post-refresh path the headline numbers came from
    argv = sys.argv[1:]
    for required in ("--h-routing", "--refresh-h", "--full-cov"):
        if required not in argv:
            argv.append(required)
    sys.argv = ["bench_chained_15_v2.py"] + argv
    raise SystemExit(bench.main())
