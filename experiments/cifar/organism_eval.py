"""Parity bench: SensoryOrganism vs SensoryConductor on full-100 CIFAR-100.

Same 7 sense donors (greedy-selected set: eye, color_smell,
frequency_print, heat_diffusion, skeleton, taste, pulse), same test
set, same metrics. Reports:
  * conductor mean_logit / confidence_weighted (reference numbers)
  * organism uniform routing  (parity check — should match mean_logit
                                bit-for-bit)
  * organism soft archive routing (the new gain — per-input gates
                                    softmax-normalized over per-branch
                                    archive log-likelihoods)
  * organism hard archive routing (argmax ablation)
  * normalize_per_branch=True for both modes.
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from trioron.senses.organism import SensoryOrganism
from trioron.senses.conductor import SensoryConductor, load_sense_donor
from experiments.cifar.datasets import (
    SLICES, DEFAULT_DATA_ROOT,
)
from experiments.cifar.conductor_eval import (
    _slice_test_set, _eval_logits,
)


GREEDY_7 = ["eye", "color_smell", "frequency_print", "heat_diffusion",
            "skeleton", "taste", "pulse"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slice", choices=sorted(SLICES), default="full",
        help="Class slice to evaluate against — must match the donors'.",
    )
    parser.add_argument(
        "--donor-dir", default=None,
        help="Default scales with --slice: outputs/cifar_donors[_full].",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--senses", nargs="+", default=GREEDY_7,
        help=f"Senses to fuse; default = greedy-7 selection {GREEDY_7}.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Soft-routing temperature for the organism (T=1.0 default).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Minibatch size for forward passes (test set is ~10K images).",
    )
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

    print(f"slice={args.slice}  donor_dir={args.donor_dir}")
    print(f"senses ({len(args.senses)}): {args.senses}")

    test_imgs, test_labs, all_classes = _slice_test_set(
        class_groups, args.data_root,
    )
    print(f"test set: {test_imgs.shape[0]} images, "
          f"{len(all_classes)} classes")

    n_classes = len(all_classes)
    chance_full = 1.0 / n_classes
    chance_task = 1.0 / max(len(g) for g in class_groups)
    print(f"chance:  full {chance_full:.4f}  task {chance_task:.4f}")

    # ---- conductor reference ----
    donors = [load_sense_donor(p) for p in donor_paths]
    print("\n=== conductor (reference) ===")
    print(f"{'fusion':<26s} {'full':>8s} {'task':>8s}")
    cnd_results: Dict[str, Dict[str, float]] = {}
    for fusion in ("mean_logit", "confidence_weighted"):
        cnd = SensoryConductor(donors, fusion=fusion).eval()
        with torch.no_grad():
            logits = _batched_forward(cnd, test_imgs, args.batch_size)
        m = _eval_logits(logits, test_labs, cnd.union_classes, class_groups)
        cnd_results[fusion] = m
        print(f"{fusion:<26s} {m['full']:>8.4f} {m['task']:>8.4f}")

    # ---- organism ----
    org = SensoryOrganism.from_sense_donors(donor_paths).eval()
    print(f"\n=== organism (storage_bytes) ===\n  {org.storage_bytes()}")
    print("\n=== organism routing modes ===")
    print(f"{'mode':<32s} {'full':>8s} {'task':>8s}  "
          f"{'Δfull':>8s} {'Δtask':>8s}  (Δ = vs conductor mean_logit)")
    base_full = cnd_results["mean_logit"]["full"]
    base_task = cnd_results["mean_logit"]["task"]
    for routing, normalize in [
        ("uniform", False),
        ("uniform", True),
        ("soft",    False),
        ("soft",    True),
        ("hard",    False),
        ("hard",    True),
    ]:
        with torch.no_grad():
            logits = _batched_forward(
                org, test_imgs, args.batch_size,
                routing=routing,
                temperature=args.temperature,
                normalize_per_branch=normalize,
            )
        m = _eval_logits(logits, test_labs, org.union_classes, class_groups)
        d_full = m["full"] - base_full
        d_task = m["task"] - base_task
        tag = f"{routing} norm={'T' if normalize else 'F'}"
        if routing == "soft":
            tag += f" (T={args.temperature:g})"
        print(f"{tag:<32s} {m['full']:>8.4f} {m['task']:>8.4f}  "
              f"{d_full:>+8.4f} {d_task:>+8.4f}")

    return 0


def _batched_forward(
    model, imgs: torch.Tensor, bs: int, **kwargs,
) -> torch.Tensor:
    out_chunks: List[torch.Tensor] = []
    for i in range(0, imgs.shape[0], bs):
        out_chunks.append(model(imgs[i:i + bs], **kwargs))
    return torch.cat(out_chunks, dim=0)


if __name__ == "__main__":
    sys.exit(main())
