"""Inspect the cells the frustration loop actually spawned (s027, read-only).

Rebuilds the deterministic run_hard organism (seed 0, GCU dendrite + frustration
loop) and walks the arena: which cells existed before growth (perception + the 20
germline council + outputs) vs which were SPAWNED, and for every spawned soma its
phenotype, rank, fan-in/out, branch count, and utility |w·g|.

Run: python3 -m experiments.progenitor.inspect_spawned
"""
from __future__ import annotations

import torch

from trioron.core.epigenome import (
    LINEAR, ATTENTION, CONV, RECURRENT, DENDRITE, PERCEPTION, OUTPUT, has_gene,
)
from .data_hard import make_split, bayes_accuracy
from .step3_council import build_council
from .step4_grow import grow_council_frustration

_GENES = [(PERCEPTION, "perception"), (OUTPUT, "output"), (DENDRITE, "dendrite"),
          (ATTENTION, "attention"), (CONV, "conv"), (RECURRENT, "recurrent"),
          (LINEAR, "linear")]


def _phenotype(epi: int) -> str:
    """First non-linear gene set wins; linear is the fallthrough (it's the default bit)."""
    for gene, name in _GENES:
        if int(has_gene(epi, gene)):
            return name
    return "linear"


def _fan_out(a, cid: int) -> int:
    ec = a.edge_cursor
    return int((a.edge_src[:ec] == cid).sum())


def main() -> None:
    tr, te = make_split()
    torch.manual_seed(0)
    sub = build_council(n_in=tr.x.shape[1], n_out=len(te.names))

    # cells that exist BEFORE any growth
    initial = set(sub.perc) | {c for ids in sub.council.values() for c in ids} | set(sub.outs)

    print("=== rebuilding the deterministic frustration run (GCU dendrite) ===\n")
    grow_council_frustration(sub, tr, te, te.names, bayes_accuracy(te))

    a = sub.arena
    spawned = [c for c in range(a.cursor) if c not in initial]

    print(f"\n=== organism after growth ===")
    print(f"  total cells {a.cursor}  (initial {len(initial)} = "
          f"{len(sub.perc)} perc + 20 council + {len(sub.outs)} out; spawned {len(spawned)})")
    print(f"  total edges {a.edge_cursor}\n")

    # tally spawned by phenotype
    by_ph: dict[str, list[int]] = {}
    for c in spawned:
        by_ph.setdefault(_phenotype(int(a.epigenome[c].item())), []).append(c)
    print("  spawned soma by phenotype:")
    for ph, cells in sorted(by_ph.items(), key=lambda kv: -len(kv[1])):
        print(f"    {ph:>9}: {len(cells)}  cells {cells}")

    print("\n  per-cell detail (spawned only):")
    print(f"    {'id':>4} {'phenotype':>10} {'rank':>4} {'K':>2} "
          f"{'fan_in':>6} {'fan_out':>7} {'utility|w·g|':>12}  note")
    for c in spawned:
        epi = int(a.epigenome[c].item())
        ph = _phenotype(epi)
        fi, fo = a.fan_in(c), _fan_out(a, c)
        k = int(a.n_branches[c].item())
        u = float(a.utility[c]) if a.utility[c] is not None else float("nan")
        note = ""
        if fo == 0:
            note = "DEAD-END (no fan-out — contributes nothing)"
        elif fi == 0:
            note = "no fan-in"
        elif ph == "dendrite" and k < 2:
            note = "K=1 → GCU suppressed (linear-equivalent)"
        print(f"    {c:>4} {ph:>10} {int(a.rank[c].item()):>4} {k:>2} "
              f"{fi:>6} {fo:>7} {u:>12.4f}  {note}")


if __name__ == "__main__":
    main()
