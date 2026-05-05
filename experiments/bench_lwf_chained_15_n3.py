"""LwF n=3 multi-seed bench on chained-15.

Li & Hoiem 2017 — knowledge distillation from prior model. The current
network's logits on old-class columns are pulled toward the anchored
network's logits via KL divergence. Already wired in
bench_chained_15task.py via LWF_ENABLED + LWF_LOSS_WEIGHT.

Same shape as fixed_ewc_small (matched-trainable, frozen L0); EWC ON
plus LwF distillation. This is the canonical LwF baseline (LwF papers
typically combine with weight regularization).

Run:
  python3 -m experiments.bench_lwf_chained_15_n3 \
      > outputs/bench_chained_15task_n3_LWF.log 2>&1
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import bench_chained_15task as bench

bench.MANIFOLD_REPLAY_ENABLED = False
bench.HIPPOCAMPAL_ENABLED = False
bench.HIPPOCAMPAL_SYNTHETIC = False
bench.REHEARSAL_ENABLED = False
bench.LWF_ENABLED = True   # the only flag flip vs fixed_ewc_small
bench.BRAINSTEM_ENABLED = False
bench.ENGRAM_ENABLED = False
bench.DIFFERENTIAL_ENABLED = False


def main() -> int:
    argv = [
        "--seeds", "0,1,2",
        "--arms", "fixed_ewc_small",
        "--csv", "bench_chained_15task_n3_LWF.csv",
    ]
    print("LwF n=3 multi-seed (chained-15) — fixed_ewc_small + LWF distillation")
    print(f"  LWF_LOSS_WEIGHT = {bench.LWF_LOSS_WEIGHT}")
    print(f"  LWF_TEMPERATURE = {bench.LWF_TEMPERATURE}")
    return bench.main(argv)


if __name__ == "__main__":
    sys.exit(main())
