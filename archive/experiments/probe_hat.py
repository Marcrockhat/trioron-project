"""Single-seed sanity probe for the HAT port to chained-15.

Runs both hat_matched and hat_standard at seed 0 to verify the wiring
(begin_task, set_temperature, sparsity_loss, scale_grads, clip_embeddings,
end_task, apply_inference_mask) runs end-to-end without errors.

Run:
  python3 -m experiments.probe_hat \
      > outputs/probe_hat.log 2>&1
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
bench.LWF_ENABLED = False
bench.BRAINSTEM_ENABLED = False
bench.ENGRAM_ENABLED = False
bench.DIFFERENTIAL_ENABLED = False


def main() -> int:
    argv = [
        "--seed", "0",
        "--arms", "hat_matched,hat_standard",
        "--csv", "probe_hat.csv",
    ]
    print("HAT single-seed sanity probe (chained-15)")
    return bench.main(argv)


if __name__ == "__main__":
    sys.exit(main())
