"""Visualize the developmental process: stem spawn → differentiate → wire → train.

Outputs a self-contained HTML viewer at runs/dev_viz/viewer.html.

Usage:
    python3 -m experiments.viz_developmental
"""
import torch
from trioron.core import Envelope, construct
from trioron.core.arena import Arena
from trioron.core.epigenome import PERCEPTION, OUTPUT, has_gene
from trioron.core.state import CellState
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from trioron.lifecycle.developmental import (
    spawn_stem, axon_guidance, differentiate_all_stems,
    MorphogenField, GuidanceConfig,
)
from trioron.viz import Recorder, export_html
from trioron.viz.snapshot import capture_full, Event

from experiments.datasets import DatasetBundle, chained_15_specs, IMAGE_DIM
from trioron.learning import (
    CreditTracker, FrustrationDetector, ManifoldArchive,
    dream_cycle, DreamConfig, interleaved_replay_batch,
)
from trioron.lifecycle import divide, GrowthConfig


def main():
    torch.manual_seed(42)

    rec = Recorder("runs/dev_viz/snapshots/", sample_growth_every_n=1)

    # ── Stage 1: Empty arena with perception + output ────────────
    envelope = Envelope(max_parameter_bytes=200_000)
    arena = Arena(envelope, capacity=2048)

    # Perception
    input_dim = 784
    n_classes = 30
    perc_ids = arena.alloc(input_dim)
    from trioron.core.epigenome import set_gene, CREDIT_ELIGIBLE, LINEAR
    for i, pid in enumerate(perc_ids.tolist()):
        epi = set_gene(int(arena.epigenome[pid].item()), PERCEPTION)
        arena.epigenome[pid] = epi
        arena.position[pid] = torch.tensor([0.5, i / max(input_dim - 1, 1), 0.0])
        arena.rank[pid] = 0

    # Output
    out_ids = arena.alloc(n_classes)
    for i, oid in enumerate(out_ids.tolist()):
        epi = set_gene(int(arena.epigenome[oid].item()), OUTPUT)
        epi = set_gene(epi, CREDIT_ELIGIBLE)
        arena.epigenome[oid] = epi
        arena.position[oid] = torch.tensor([0.5, i / max(n_classes - 1, 1), 1.0])
        arena.rank[oid] = 2
    arena.refresh_all_phenotypes()

    snap = capture_full(arena, task_index=0)
    snap.events_since_last = [Event("init", 0, 0, {"stage": "perception + output only"})]
    rec._write(snap, "stage0_init")
    print(f"Stage 0: {arena.n_cells} cells, {arena.n_edges} edges (perception + output)")

    # ── Stage 2: Spawn stem cells ────────────────────────────────
    stem_ids = spawn_stem(arena, 64)
    snap = capture_full(arena, task_index=0)
    snap.events_since_last = [Event("spawn_stem", 0, 0, {"n_stems": 64})]
    rec._write(snap, "stage1_stems")
    print(f"Stage 1: {arena.n_cells} cells (64 stems spawned)")

    # ── Stage 3: Axon guidance ───────────────────────────────────
    n_edges = axon_guidance(arena, cfg=GuidanceConfig())
    snap = capture_full(arena, task_index=0)
    snap.events_since_last = [Event("axon_guidance", 0, 0, {"edges_added": n_edges})]
    rec._write(snap, "stage2_wired")
    print(f"Stage 2: {arena.n_edges} edges (axon guidance)")

    # ── Stage 4: Differentiation ─────────────────────────────────
    morphogen = MorphogenField(device=arena.device)
    results = differentiate_all_stems(arena, morphogen)
    snap = capture_full(arena, task_index=0)
    pheno_names = {0: "LINEAR", 1: "ATTENTION", 2: "CONV"}
    counts = {}
    for _, p in results:
        name = pheno_names.get(p, f"bit{p}")
        counts[name] = counts.get(name, 0) + 1
    snap.events_since_last = [Event("differentiate", 0, 0, {"phenotypes": counts})]
    rec._write(snap, "stage3_differentiated")
    print(f"Stage 3: differentiated — {counts}")

    # ── Stage 5: Build substrate and train 2 tasks ───────────────
    from trioron.core.construct import Substrate
    from trioron.core.graph import CellGraph
    dispatch = default_dispatch_table()
    sub = Substrate(arena, dispatch)
    sub.morphogen = morphogen
    sub.compile()

    snap = capture_full(arena, task_index=0)
    snap.events_since_last = [Event("compiled", 0, 0)]
    rec._write(snap, "stage4_compiled")
    print(f"Stage 4: compiled, {sub.n_cells} cells, {sub.n_edges} edges")

    # Train on 2 tasks
    specs = chained_15_specs()
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"])

    credit = CreditTracker(arena)
    archive = ManifoldArchive(arena)
    growth_cfg = GrowthConfig(new_edges=6, inherit_frac=0.3)

    for task_idx in range(3):
        spec = specs[task_idx]
        frust = FrustrationDetector()
        train_view = bundle.task_view(spec.dataset_name, spec.local_classes, spec.global_classes,
            split="train", task_name=spec.name)

        sub.prepare_training()
        opt = torch.optim.Adam(sub.trainable_tensors(), lr=5e-3)
        code_boundary = [cid for cid in arena.alive_ids().tolist()
                         if has_gene(int(arena.epigenome[cid].item()), PERCEPTION)]

        rec.on_task_start(arena, task_idx)

        growth_budget = 4
        frust_steps = 0
        for epoch in range(2):
            for x_batch, y_batch in train_view.iter_epoch(64):
                logits = sub(x_batch)
                loss = torch.nn.functional.cross_entropy(logits, y_batch)
                m = frust.step(loss.item())
                if frust.is_frustrated: frust_steps += 1
                (loss * m).backward()
                sub.zero_dormant_grads()
                credit.update_utility()
                if sub.last_activations is not None:
                    credit.update_engagement(sub.last_activations)
                opt.step(); opt.zero_grad()

                code = x_batch[:, :len(code_boundary)]
                replay = interleaved_replay_batch(archive, spec.global_classes,
                    batch_size=64, n_perc=len(code_boundary), code_batch=code)
                if replay is not None:
                    rx, ry = replay
                    rl = sub(rx)
                    rl_loss = torch.nn.functional.cross_entropy(rl, ry)
                    rl_loss.backward(); sub.zero_dormant_grads(); opt.step(); opt.zero_grad()

                for gc in spec.global_classes:
                    mask = y_batch == gc
                    if mask.any(): archive.update_class(gc, code[mask])

                if growth_budget > 0 and frust.is_frustrated and frust_steps >= 25:
                    interior = [cid for cid in arena.alive_ids().tolist()
                        if not has_gene(int(arena.epigenome[cid].item()), PERCEPTION)
                        and not has_gene(int(arena.epigenome[cid].item()), OUTPUT)
                        and arena.state[cid] == CellState.ACTIVE]
                    if interior:
                        parent = interior[torch.randint(0, len(interior), (1,)).item()]
                        event = divide(arena, parent, growth_cfg)
                        if event:
                            growth_budget -= 1
                            rec.on_growth(arena, task_idx, event.child_id)
                            sub.compile()
                            opt = torch.optim.Adam(sub.trainable_tensors(), lr=5e-3)
                            frust_steps = 0

        archive.finalize_all()
        dr = dream_cycle(sub, credit, archive, current_classes=spec.global_classes)
        rec.on_dream(arena, task_idx)
        sub.end_task(); sub.compile()

        for gc in spec.global_classes:
            astro = archive.get(gc)
            if astro and arena.state[astro.cell_id] == CellState.DORMANT:
                arena.state[astro.cell_id] = CellState.ACTIVE

        rec.on_task_end(arena, task_idx)

        # Eval
        full_accs = []
        for t in range(task_idx + 1):
            s = specs[t]
            tv = bundle.task_view(s.dataset_name, s.local_classes, s.global_classes, split="test", task_name=s.name)
            x, y = tv.all_examples()
            with torch.no_grad(): lo = sub(x)
            full_accs.append((lo.argmax(1) == y).float().mean().item())
        mf = sum(full_accs) / len(full_accs)
        print(f"Task {task_idx} ({spec.name}): full={mf:.4f}, cells={sub.n_cells}, edges={sub.n_edges}")

    # ── Export viewer ────────────────────────────────────────────
    path = export_html("runs/dev_viz/snapshots/", "runs/dev_viz/viewer.html")
    print(f"\nViewer exported: {path}")
    print(f"Snapshots: {rec.snapshot_count}")
    print("Open runs/dev_viz/viewer.html in a browser to see the developmental process.")


if __name__ == "__main__":
    main()
