"""Shim runner for the archived chained-15 bench (s034).

The archive is frozen history; its imports predate the v1->legacy move
(trioron.network) and the donorkit migration (experiments.datasets,
experiments.bench_chained_15task). Alias the moved modules in
sys.modules, then execute the bench unmodified.
"""
import runpy
import sys

ROOT = "/home/marcrockhat/trioron-project"
sys.path.insert(0, ROOT)

import trioron.legacy.network as _net               # noqa: E402
sys.modules["trioron.network"] = _net
import trioron.legacy.manifold as _man              # noqa: E402
sys.modules["trioron.manifold"] = _man

import experiments                                   # noqa: E402
import trioron.legacy.donorkit.datasets as _ds      # noqa: E402
import trioron.legacy.donorkit.bench_chained_15task as _b15  # noqa: E402
sys.modules["experiments.datasets"] = _ds
sys.modules["experiments.bench_chained_15task"] = _b15
experiments.datasets = _ds
experiments.bench_chained_15task = _b15

runpy.run_path(ROOT + "/archive/experiments/axis6_credit_chained15.py",
               run_name="__main__")
