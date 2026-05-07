"""Absorb the 8 primitive donors into one vocabulary organism.

Reads donors from outputs/atari_primitive_donors/{group}.pt and writes
the absorbed organism to outputs/atari_primitive_donors/vocabulary.pt.

The vocabulary organism is the substrate the Pong-extension layer
(task #10) builds on top of. After absorb, the multi-branch organism
fires every primitive in parallel on a single state vector — the
action head reads the per-class log-likelihoods.

Sanity test: per-donor held-out evaluation through the absorbed
organism. The eligible-class mask scopes routing to the donor's own
classes, so each branch should re-produce its standalone task-aware
accuracy (within absorb-noise tolerance).

Usage:
    python3 -m experiments.atari_trioron.primitives.absorb_vocabulary
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

PROJ = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from trioron.api import absorb, load_organism   # noqa: E402
from experiments.atari_trioron.primitives.synthetic_env import (   # noqa: E402
    generate_dataset, CLASS_NAMES,
)
from experiments.atari_trioron.primitives.train_donors import (   # noqa: E402
    DONOR_GROUPS, OUT_ROOT,
)


VOCAB_PATH = OUT_ROOT / "vocabulary.pt"


def absorb_all() -> Path:
    donor_paths = [OUT_ROOT / f"{g}.pt" for g in DONOR_GROUPS.keys()]
    missing = [str(p) for p in donor_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing donor files: {missing}; run train_donors.py first"
        )
    print(f"Absorbing {len(donor_paths)} donors:")
    for p in donor_paths:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:32s}  ({size_kb:.1f} KB)")
    t0 = time.time()
    out = absorb(donor_paths=donor_paths, out_path=VOCAB_PATH)
    elapsed = time.time() - t0
    out_kb = out.stat().st_size / 1024
    print(f"absorbed in {elapsed:.1f}s -> {out} ({out_kb:.1f} KB)")
    return out


@torch.no_grad()
def evaluate_branches(
    organism_path: Path,
    *,
    n_per_class: int = 500,
    seed: int = 9999,
) -> Dict[str, float]:
    """Per-branch held-out eval through the absorbed organism.
    Each branch should re-produce its standalone donor accuracy."""
    organism = load_organism(organism_path)
    summary: Dict[str, float] = {}
    for group_name, class_ids in DONOR_GROUPS.items():
        eligible = set(class_ids)
        total_correct = 0
        total = 0
        per_class: Dict[int, float] = {}
        for i, cid in enumerate(class_ids):
            X = generate_dataset(cid, n_per_class, seed=seed + 100 * i)
            logits = organism(X, routing="soft")
            if isinstance(logits, tuple):
                logits = logits[0]
            union = list(organism.union_classes)
            masked = torch.full_like(logits, float("-inf"))
            for j, c in enumerate(union):
                if int(c) in eligible:
                    masked[:, j] = logits[:, j]
            pred_idx = masked.argmax(dim=-1)
            pred_class = torch.tensor([int(union[int(k)]) for k in pred_idx])
            correct = int((pred_class == cid).sum())
            total_correct += correct
            total += n_per_class
            per_class[cid] = correct / n_per_class
        overall = total_correct / total
        summary[group_name] = overall
        print(f"  {group_name:25s}  task-aware {overall:.3f}")
        for cid in class_ids:
            print(f"      {CLASS_NAMES[cid]:>26s}: {per_class[cid]:.3f}")
    return summary


def main():
    out = absorb_all()
    print(f"\n=== branch eval on absorbed organism ===")
    summary = evaluate_branches(out)
    print(f"\n=== summary ===")
    for g, acc in summary.items():
        verdict = ("OK" if acc >= 0.95
                   else ("YELLOW" if acc >= 0.85 else "RED"))
        print(f"  {g:25s}  {acc:.3f}  [{verdict}]")


if __name__ == "__main__":
    main()
