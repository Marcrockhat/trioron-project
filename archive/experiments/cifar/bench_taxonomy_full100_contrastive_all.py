"""Apply contrastive δ-replay refinement to all L2 experts trained for
the full-100 pipeline. Skips singletons (no expert) and 2-way clusters
where the contrastive lift is typically tiny but it's still done for
consistency.
"""
from __future__ import annotations
import argparse
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-file",
                        default="outputs/cifar_taxonomy/cluster_assignment_full100_k20.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    args = parser.parse_args(argv)

    ca = torch.load(args.cluster_file, map_location="cpu", weights_only=False)
    print(f"[full100-c-all] k={ca['k']}  seed={args.seed}")

    refined = 0
    skipped = 0
    here = os.path.dirname(os.path.abspath(__file__))
    contrastive_script = os.path.join(here, "bench_taxonomy_l2_contrastive.py")
    for cid, names in enumerate(ca["clusters"]):
        K = len(names)
        if K < 2:
            print(f"  c{cid:02d}: singleton ({names}) — skip")
            skipped += 1
            continue
        in_path = (f"outputs/cifar_taxonomy/donor_full100_l2_c{cid:02d}_"
                   f"{K}way_seed{args.seed}.pt")
        out_path = (f"outputs/cifar_taxonomy/donor_full100_l2_c{cid:02d}_"
                    f"{K}way_seed{args.seed}_contrastive.pt")
        if not os.path.exists(in_path):
            print(f"  c{cid:02d}: missing input {in_path} — skip")
            skipped += 1
            continue
        cmd = [
            "python3", contrastive_script,
            "--donor-path", in_path,
            "--out-path", out_path,
            "--steps", str(args.steps),
            "--margin", str(args.margin),
            "--lr", str(args.lr),
            "--ce-weight", str(args.ce_weight),
            "--margin-weight", str(args.margin_weight),
            "--align-pairs",
            "--seed", str(args.seed),
        ]
        print(f"  c{cid:02d} ({K}-way): {names[:3]}{'…' if K > 3 else ''}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    FAILED: {result.stderr[-500:]}")
            skipped += 1
            continue
        # Pull out the baseline → contrastive line.
        for line in result.stdout.splitlines():
            if "baseline acc:" in line or "contrastive acc:" in line or "Δ:" in line:
                print(f"    {line.strip()}")
        refined += 1
    print(f"\n[full100-c-all] refined {refined} experts; skipped {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
