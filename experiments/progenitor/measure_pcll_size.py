"""Exact trained-state byte audit of the nested PCLL organism — s049.

Rocky's question: after data accumulation passes some point, does PCLL
save space vs the gradient stack?  PCLL state is sufficient statistics
(phasor sums), constant in samples seen — so the comparison is state
bytes vs state bytes, and vs anything that stores exemplars.

Trains nest seed 0 (same protocol as pcll_nested), then walks every
torch tensor reachable from each leaf's controller + arena and the
router archive, grouping bytes by component.  Also reports the
INFERENCE-MINIMAL subset (what a deployed nest must carry: class
templates + names + quantizer state + router mu/Sigma).

Reference points printed alongside (recorded, not rerun here):
  * v0.2.2 gradient chained-15 substrate: 371,316 param bytes
    (1019 cells / 91,810 edges, outputs/validate_router_real.log:119)
    + 30 KB manifold replay archive;
  * hippo K=20 exemplar rehearsal at 30 classes: 20*784*4*30 bytes;
  * raw stream retained: n_samples * 784 * 4 bytes.

Run: PCLL15_SENSE=gabor python3 -m experiments.progenitor.measure_pcll_size
"""
from __future__ import annotations

import torch

from experiments.progenitor.pcll_nested import run_seed, DOMAINS


def walk_bytes(obj, seen: set, depth: int = 0) -> int:
    if depth > 6 or id(obj) in seen:
        return 0
    seen.add(id(obj))
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    total = 0
    if isinstance(obj, dict):
        for v in obj.values():
            total += walk_bytes(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            total += walk_bytes(v, seen, depth + 1)
    elif hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            total += walk_bytes(v, seen, depth + 1)
    return total


def component_bytes(obj) -> dict:
    out = {}
    seen: set = set()
    for name, v in vars(obj).items():
        b = walk_bytes(v, seen)
        if b:
            out[name] = b
    return out


def fmt(b: int) -> str:
    return f"{b:>12,d} B ({b / 1024:8.1f} KiB)"


def main() -> None:
    ta, full, leaves, router = run_seed(0, return_state=True)
    print("\n=== EXACT STATE AUDIT (trained nest, seed 0) ===")
    grand = 0
    for dom, leaf in enumerate(leaves):
        print(f"\n-- leaf {DOMAINS[dom]} --")
        comp = component_bytes(leaf.mixed)
        arena_b = component_bytes(leaf.sub.arena)
        ctrl = sum(comp.values())
        aren = sum(arena_b.values())
        for k, v in sorted(comp.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  mixed.{k:<18s} {fmt(v)}")
        for k, v in sorted(arena_b.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  arena.{k:<18s} {fmt(v)}")
        print(f"  controller total       {fmt(ctrl)}")
        print(f"  arena total            {fmt(aren)}")
        grand += ctrl + aren
    seen: set = set()
    router_b = walk_bytes(router.archive, seen)
    print(f"\n-- router archive (full-cov, {len(leaves)} domains) --")
    print(f"  archive total          {fmt(router_b)}")
    grand += router_b
    print(f"\nTOTAL training state     {fmt(grand)}")

    print("\n=== INFERENCE-MINIMAL DEPLOY SET ===")
    mini = 0
    for dom, leaf in enumerate(leaves):
        T = leaf.mixed.templates()
        tb = T.numel() * T.element_size()
        names_b = len(leaf.mixed.classes) * 8
        mini += tb + names_b
        print(f"  {DOMAINS[dom]:<8s} templates {tuple(T.shape)} "
              f"{fmt(tb)}  + names {fmt(names_b)}")
    n_dim = None
    router_min = 0
    for cid in router.archive.class_ids:
        astro = router.archive.get(cid)
        n_dim = astro.mu.numel()
        router_min += astro.mu.numel() * 4 + n_dim * n_dim * 4
    print(f"  router mu+Sigma x{len(router.archive.class_ids)} "
          f"({n_dim}-d)      {fmt(router_min)}")
    mini += router_min
    print(f"  DEPLOY total           {fmt(mini)}")

    print("\n=== REFERENCE POINTS (recorded) ===")
    grad = 371_316
    print(f"  v0.2.2 gradient substrate       {fmt(grad)}"
          f"   (+30 KiB manifold archive)")
    print(f"  hippo K=20 exemplars, 30 cls    {fmt(20 * 784 * 4 * 30)}")
    for n in (10_000, 100_000, 1_000_000):
        print(f"  raw stream retained, n={n:>9,d} {fmt(n * 784 * 4)}")
    print(f"\n(nest accuracy this run: task_aware={ta:.3f} full={full:.3f})")


if __name__ == "__main__":
    main()
