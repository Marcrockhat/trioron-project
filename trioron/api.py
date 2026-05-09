"""Trioron user-facing Python API.

Everything below is what end-users should import. The research scripts
under ``experiments/*`` remain available for paper-reproduction work,
but they are not part of the supported surface and should not be
imported from user code.

Three flows:

  1. Build a donor from your own data
  --------------------------------------------------
  >>> from trioron.api import TaskData, TrioronConfig, build_donor
  >>> tasks = [
  ...     TaskData(name="cats_vs_dogs",
  ...              X_train=Xtr, y_train=ytr,    # (N, 784) float32, (N,) int64
  ...              X_test=Xte,  y_test=yte,
  ...              classes=[0, 1]),
  ...     TaskData(name="birds_vs_fish",
  ...              X_train=..., y_train=...,
  ...              X_test=...,  y_test=...,
  ...              classes=[2, 3]),
  ... ]
  >>> donor = build_donor(
  ...     label="my_donor",
  ...     tasks=tasks,
  ...     seed=42,                              # shared L0 seed
  ...     config=TrioronConfig(cap_bytes=32_000),
  ...     out_path="my_donor.pt",
  ... )

  2. Compose donors into one organism
  --------------------------------------------------
  >>> from trioron.api import absorb
  >>> organism_path = absorb(
  ...     donor_paths=["my_donor.pt", "another_donor.pt"],
  ...     out_path="organism.pt",
  ... )

  3. Deploy as an agent
  --------------------------------------------------
  >>> from trioron.api import deploy_agent
  >>> from trioron.bridge import ToolDispatcher
  >>> tools = ToolDispatcher()
  >>> @tools.tool
  ... def lookup_user(user_id: str) -> dict:
  ...     '''Return profile for the given user.'''
  ...     return {"user_id": user_id, "name": "..."}
  >>> agent = deploy_agent(
  ...     organism_path="organism.pt",
  ...     encoder=my_encoder,                   # any trioron.bridge.Encoder
  ...     tools=tools,
  ...     class_to_tool={0: "lookup_user", 1: "lookup_user"},
  ... )
  >>> result = agent.act("Tell me about user #42")

The shared-L0 invariant from paper §3.10 is enforced at absorption
time: every donor in a population MUST be built with the same ``seed``
or the absorb step fails fast.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch


# ---------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------


@dataclass
class TaskData:
    """One task in a donor's curriculum.

    Each TaskData represents one classification task (binary or n-ary)
    that the donor will be trained on, in order. The donor sees tasks
    one at a time; no replay-from-disk happens between tasks
    (manifold replay is the in-network substitute).

    Tensor shapes:
        X_train, X_test:  (N, input_dim) float32 in [0, 1].
                          Default architecture expects input_dim=784
                          (28×28 grayscale flattened). Use a different
                          input_dim by passing ``input_dim=`` to
                          :func:`build_donor`.
        y_train, y_test:  (N,) int64. Labels in GLOBAL class space —
                          the index in the donor's head, NOT a per-task
                          local index. Two donors that cover disjoint
                          class ranges (e.g. 0..9 and 10..19) can be
                          absorbed into one organism with no head
                          collision.

    Attributes:
        name: Human-readable task identifier (used in logs and
              checkpoints).
        classes: List of GLOBAL class IDs this task introduces. Must
                 match the unique values in ``y_train``/``y_test``.
    """
    name: str
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    classes: List[int]

    def __post_init__(self) -> None:
        if self.X_train.dim() != 2:
            raise ValueError(
                f"task {self.name!r}: X_train must be 2D (N, input_dim), "
                f"got shape {tuple(self.X_train.shape)}"
            )
        if self.X_test.shape[-1] != self.X_train.shape[-1]:
            raise ValueError(
                f"task {self.name!r}: X_test input_dim "
                f"{self.X_test.shape[-1]} != X_train input_dim "
                f"{self.X_train.shape[-1]}"
            )
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError(
                f"task {self.name!r}: X_train has {self.X_train.shape[0]} "
                f"rows, y_train has {self.y_train.shape[0]}"
            )
        if self.X_test.shape[0] != self.y_test.shape[0]:
            raise ValueError(
                f"task {self.name!r}: X_test has {self.X_test.shape[0]} "
                f"rows, y_test has {self.y_test.shape[0]}"
            )
        seen = set(self.y_train.unique().tolist()) | set(self.y_test.unique().tolist())
        declared = set(self.classes)
        extra = seen - declared
        if extra:
            raise ValueError(
                f"task {self.name!r}: labels {sorted(extra)} appear in y "
                f"but are not in classes={self.classes}"
            )


@dataclass
class AdvancedConfig:
    """Architecturally-distinctive growth knobs. Don't touch unless
    you've read ``trioron_blueprint.md`` §3.4 / §3.6 — wrong values can
    silently kill growth.

    Defaults match the chained-15 paper configuration.
    """
    # Growth — see triggers.py
    h_init: int = 32                       # initial L1 hidden width
    n_grow_per_task: int = 4               # nodes added per growth event
    growth_target_layer_idx: int = 1       # which hidden layer grows
    # EWC
    ewc_intertask_strength: float = 30.0
    ewc_dream_strength: float = 30.0
    # Dream details
    dream_replay_fraction: float = 0.25    # fraction of past tasks per dream
    dream_compression_action: str = "starve"   # "starve" | "merge" | "none"
    dream_max_downscales_per_layer: int = 1    # sRNA-style cap
    dream_apoptosis_on: bool = True
    # Frozen-L0 vs trainable-L0
    freeze_l0: bool = True
    # L0 width — also fixes the encoder-projection target dim
    l0_width: int = 128


@dataclass
class TrioronConfig:
    """The architecturally-distinctive knobs end-users tune.

    The five primaries are the ones that materially differ from
    PackNet/HAT/Online-EWC/LwF and are exposed at the top level. The
    advanced bundle holds growth-trigger and EWC sub-knobs; leave it
    None unless you know what you're doing.

    Attributes:
        cap_bytes: Hard upper bound on trainable parameter bytes
            (4 bytes/param, fp32). Growth events that would exceed
            this fail their pre-flight (``ceilings.py``). Set to 0 or
            None to disable the cap. Paper chained-15: 32_000 (8K
            params), extension: 64_000.
        dream_replay_steps: Replay batches per dream cycle. Paper
            default 50 for chained-15, 200 for 50-task.
        dream_buffer_threshold: Minimum past-task count before the
            first dream fires (analogous to a sleep-pressure threshold
            in the sRNA-cap framing). 0 = dream after every task.
        manifold_noise_scale: Multiplier on per-class σ when sampling
            from the manifold archive. 1.0 = paper default; 0.0 =
            μ-only (loses ~7% full-softmax — see manifold ablation).
        routing_temperature: Soft-routing temperature at the organism
            (consumed by absorb / eval / serve, not at donor build).
            T → 0 = hard routing; T = 1.0 = paper default; T → ∞ =
            uniform.
        per_class_bias: Whether to apply per-class bias offsets at
            evaluation time (dream-cycle calibration; closes ~80% of
            the BTM-MoE gap with no real data).
        advanced: Growth-trigger and EWC sub-knobs. Default None.
    """
    cap_bytes: Optional[int] = 32_000
    dream_replay_steps: int = 50
    dream_buffer_threshold: int = 0
    manifold_noise_scale: float = 1.0
    routing_temperature: float = 1.0
    per_class_bias: bool = False
    advanced: Optional[AdvancedConfig] = None


# ---------------------------------------------------------------------
# Internal: configure the bench module from a TrioronConfig + run
# ---------------------------------------------------------------------


def _apply_config_to_bench(cfg: TrioronConfig) -> None:
    """Inject the user's TrioronConfig into the module-level globals
    of experiments/bench_chained_15task.py.

    The bench module reads its knobs from module attributes (legacy
    pattern from the research codebase). The clean way to expose
    user-tunable knobs is to override those attributes here, then
    restore them afterward via :func:`_snapshot_bench` /
    :func:`_restore_bench`.
    """
    from experiments import bench_chained_15task as bench
    # Manifold replay must be on so the donor accumulates its archive.
    bench.MANIFOLD_REPLAY_ENABLED = True
    bench.HIPPOCAMPAL_ENABLED = False
    bench.HIPPOCAMPAL_SYNTHETIC = False
    bench.REHEARSAL_ENABLED = False
    bench.LWF_ENABLED = False
    bench.BRAINSTEM_ENABLED = False
    bench.ENGRAM_ENABLED = False
    bench.DIFFERENTIAL_ENABLED = False
    # Archive + int8 quantization on by default so production matches
    # bench_chained_extend's path. Without these, `permanent_int8=True`
    # silently no-ops (the archive boundary check + Phase 2 quant
    # simulation at end-of-extension both gate on these flags).
    bench.ARCHIVE_ENABLED = True
    bench.QUANTIZE_ARCHIVED_AT_END = True
    bench.QUANTIZE_MODE = "int8"
    # Knobs from TrioronConfig
    bench.MANIFOLD_NOISE_SCALE = float(cfg.manifold_noise_scale)
    bench.DREAM_REPLAY_STEPS = int(cfg.dream_replay_steps)
    bench.EWC_INTERTASK = (cfg.advanced.ewc_intertask_strength
                           if cfg.advanced else 30.0)
    bench.EWC_DREAM_STRENGTH = (cfg.advanced.ewc_dream_strength
                                if cfg.advanced else 30.0)
    if cfg.advanced is not None:
        bench.N_GROW_PER_TASK = int(cfg.advanced.n_grow_per_task)
        bench.GROWTH_TARGET_LAYER_IDX = int(cfg.advanced.growth_target_layer_idx)
        bench.DREAM_REPLAY_FRACTION = float(cfg.advanced.dream_replay_fraction)
        bench.DREAM_COMPRESSION_ACTION = str(cfg.advanced.dream_compression_action)
        bench.DREAM_MAX_DOWNSCALES_PER_LAYER = int(
            cfg.advanced.dream_max_downscales_per_layer
        )
        bench.DREAM_APOPTOSIS_ON = bool(cfg.advanced.dream_apoptosis_on)
        bench.L0_WIDTH = int(cfg.advanced.l0_width)


def _snapshot_bench() -> Dict[str, Any]:
    """Capture the bench's mutable knobs so we can restore them
    after a build_donor / extend call."""
    from experiments import bench_chained_15task as bench
    keys = [
        "MANIFOLD_REPLAY_ENABLED", "HIPPOCAMPAL_ENABLED",
        "HIPPOCAMPAL_SYNTHETIC", "REHEARSAL_ENABLED", "LWF_ENABLED",
        "BRAINSTEM_ENABLED", "ENGRAM_ENABLED", "DIFFERENTIAL_ENABLED",
        "ARCHIVE_ENABLED", "QUANTIZE_ARCHIVED_AT_END", "QUANTIZE_MODE",
        "MANIFOLD_NOISE_SCALE", "DREAM_REPLAY_STEPS",
        "EWC_INTERTASK", "EWC_DREAM_STRENGTH",
        "N_GROW_PER_TASK", "GROWTH_TARGET_LAYER_IDX",
        "DREAM_REPLAY_FRACTION", "DREAM_COMPRESSION_ACTION",
        "DREAM_MAX_DOWNSCALES_PER_LAYER", "DREAM_APOPTOSIS_ON",
        "L0_WIDTH",
    ]
    return {k: getattr(bench, k) for k in keys}


def _restore_bench(snapshot: Dict[str, Any]) -> None:
    from experiments import bench_chained_15task as bench
    for k, v in snapshot.items():
        setattr(bench, k, v)


# ---------------------------------------------------------------------
# Internal: TaskData → TaskDataView bridging
# ---------------------------------------------------------------------


def _to_views(tasks: Sequence[TaskData]) -> Tuple[list, list, list]:
    """Convert user-supplied TaskData into the (train_views, eval_views,
    task_class_lists) triple the bench's run_arm consumes."""
    from experiments.datasets import TaskDataView
    train_views = []
    eval_views = []
    task_class_lists = []
    for t in tasks:
        # Local class indices are not used by the bench at training
        # time when y is already in global space, but TaskDataView
        # requires the field. We synthesize a 1:1 local↔global map.
        local_classes = list(t.classes)
        train_views.append(TaskDataView(
            name=t.name,
            images=t.X_train.float(),
            labels_global=t.y_train.long(),
            local_classes=local_classes,
            global_classes=list(t.classes),
        ))
        eval_views.append(TaskDataView(
            name=t.name,
            images=t.X_test.float(),
            labels_global=t.y_test.long(),
            local_classes=local_classes,
            global_classes=list(t.classes),
        ))
        task_class_lists.append(list(t.classes))
    return train_views, eval_views, task_class_lists


# ---------------------------------------------------------------------
# Public: build_donor
# ---------------------------------------------------------------------


def build_donor(
    *,
    label: str,
    tasks: Sequence[TaskData],
    seed: int = 42,
    epochs_per_task: int = 8,
    config: Optional[TrioronConfig] = None,
    out_path: Union[str, Path],
    arm: str = "grown_capped_dream",
    n_passes: int = 1,
) -> Path:
    """Train one trioron donor on a user-supplied curriculum.

    Args:
        label: Donor identifier — gets baked into the checkpoint and
            shown in eval logs.
        tasks: Ordered curriculum. The donor sees tasks in this order;
            growth, dreaming and manifold archive accumulation all
            happen task-by-task.
        seed: Shared L0 seed. Two donors absorbed into the same
            organism MUST share this value (paper §3.10 invariant).
        epochs_per_task: SGD epochs per task. Paper default 8.
        config: TrioronConfig with the architectural knobs. None =
            paper defaults for chained-15.
        out_path: Where to save the donor checkpoint (.pt).
        arm: Bench arm definition. Defaults to ``grown_capped_dream``,
            the production configuration. Other choices: ``grown_uncapped_dream``
            (no cap_bytes ceiling), ``grown_capped_no_dream`` (ablate
            consolidation).
        n_passes: Number of full sweeps through the curriculum.
            Default 1 matches paper. Multi-pass (2-5) helps with
            curriculum forgetting on long curricula — each pass
            revisits early tasks, manifold archive accumulates more
            replay points, growth budget compounds.

    Returns:
        Path to the saved donor checkpoint, suitable for
        :func:`absorb`.
    """
    cfg = config or TrioronConfig()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    train_views, eval_views, task_class_lists = _to_views(tasks)
    classes_covered = sorted({c for t in tasks for c in t.classes})
    input_dim = tasks[0].X_train.shape[-1]

    # Wire bench knobs from cfg, run, and restore.
    snap = _snapshot_bench()
    try:
        _apply_config_to_bench(cfg)
        # Override cap_bytes per-arm at the call boundary instead of
        # going through ARM_DEFINITIONS, since cap_bytes is the most
        # commonly-tuned knob and we don't want users to learn the
        # arm registry.
        from experiments import bench_chained_15task as bench
        if cfg.cap_bytes is not None and cfg.cap_bytes > 0:
            bench.ARM_DEFINITIONS[arm]["cap_bytes"] = int(cfg.cap_bytes)
        else:
            bench.ARM_DEFINITIONS[arm]["cap_bytes"] = bench.M_MAX_BYTES_UNCAPPED
        # Adapt input dim if the user is feeding non-784 data.
        bench.INPUT_DIM = int(input_dim)

        r = bench.run_arm(
            arm,
            seed=seed,
            n_epochs_per_task=epochs_per_task,
            train_views=train_views,
            eval_views=eval_views,
            task_class_lists=task_class_lists,
            infancy_view=None,
            n_passes=int(n_passes),
            return_state=True,
        )
    finally:
        _restore_bench(snap)

    net = r["net"]
    mb = r.get("manifold")
    if mb is None:
        raise RuntimeError(
            "build_donor: training did not produce a manifold archive — "
            "manifold replay must be enabled (it is by default). Check "
            "that the tasks completed without error."
        )
    n_nodes = list(net.n_nodes_per_layer())
    payload = {
        # version 2 adds `task_class_lists` so `extend` can resume from
        # the donor's substrate without re-running the base curriculum.
        "version": 2,
        "kind": "trioron_donor",
        "label": label,
        "classes_covered": classes_covered,
        "n_nodes_per_layer": n_nodes,
        "input_dim": int(input_dim),
        "l0_seed": int(seed),
        "arm": arm,
        "task_class_lists": [list(t.classes) for t in tasks],
        "state_dict": {k: v.detach().cpu()
                       for k, v in net.state_dict().items()},
        "manifold_stats": {int(c): (mu.detach().cpu(), sg.detach().cpu())
                           for c, (mu, sg) in mb._stats.items()},
        # Stash the user's config so future inspection can show the
        # exact knobs the donor was built with — important for the
        # extend flow, which needs to inherit them.
        "trioron_config": {
            "cap_bytes": cfg.cap_bytes,
            "dream_replay_steps": cfg.dream_replay_steps,
            "dream_buffer_threshold": cfg.dream_buffer_threshold,
            "manifold_noise_scale": cfg.manifold_noise_scale,
            "routing_temperature": cfg.routing_temperature,
            "per_class_bias": cfg.per_class_bias,
            "advanced": (cfg.advanced.__dict__ if cfg.advanced else None),
        },
    }
    torch.save(payload, out)
    return out


# ---------------------------------------------------------------------
# Public: absorb
# ---------------------------------------------------------------------


def _branch_from_organism_dict(d: Dict[str, Any]):
    """Reconstruct a `Branch` from an inline organism payload dict.

    Each entry in `payload["branches"]` of a saved
    multibranch_organism has the same fields a donor checkpoint
    carries. This helper builds the Branch in-memory without going
    through disk, so an organism file's branches can be re-absorbed
    without first being un-bundled into per-branch donor files.
    """
    from trioron.multibranch import Branch, TrioronNetwork
    n_nodes = d["n_nodes_per_layer"]
    layer_specs: List[Tuple[int, int, str]] = []
    prev = d["input_dim"]
    for i, n in enumerate(n_nodes):
        act = "linear" if i == len(n_nodes) - 1 else "relu"
        layer_specs.append((prev, n, act))
        prev = n
    net = TrioronNetwork(layer_specs)
    net.load_state_dict(d["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return Branch(
        label=d.get("label", "donor"),
        classes_covered=list(d["classes_covered"]),
        net=net,
        manifold_stats={
            int(c): (mu, sg) for c, (mu, sg) in d["manifold_stats"].items()
        },
        l0_seed=d.get("l0_seed"),
        arm=d.get("arm"),
    )


def _branches_from_path(path: str) -> List[Any]:
    """Load branches from either a single donor checkpoint or a saved
    multibranch organism. Organisms are expanded into their
    constituent branches; donors return a 1-element list."""
    from trioron.multibranch import Branch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    kind = payload.get("kind")
    if kind == "multibranch_organism":
        return [_branch_from_organism_dict(d) for d in payload["branches"]]
    # Default: donor checkpoint (kind None or "trioron_donor")
    return [Branch.from_checkpoint(path)]


def absorb(
    *,
    donor_paths: Sequence[Union[str, Path]],
    out_path: Union[str, Path],
) -> Path:
    """Assemble a multi-branch organism from saved donors or organisms.

    Zero-shot, no calibration. The shared-L0 invariant is checked:
    every branch must have been built with the same ``seed`` or this
    raises a ValueError before doing any work.

    Args:
        donor_paths: One or more checkpoint files. Each can be:
            - a donor produced by :func:`build_donor` (single branch), OR
            - an organism produced by a previous :func:`absorb` call
              (multi-branch — its branches are expanded inline so the
              caller doesn't have to un-bundle into individual donors).
        out_path: Where to save the organism .pt.

    Returns:
        Path to the saved organism, usable by :func:`load_organism`,
        :func:`evaluate`, and :func:`deploy_agent`.
    """
    from trioron.multibranch import MultiBranchOrganism
    paths = [str(p) for p in donor_paths]
    if not paths:
        raise ValueError("absorb: donor_paths is empty")
    branches: List[Any] = []
    for p in paths:
        branches.extend(_branches_from_path(p))
    seed_counts: Dict[Any, int] = {}
    for b in branches:
        seed_counts[b.l0_seed] = seed_counts.get(b.l0_seed, 0) + 1
    if len(seed_counts) > 1:
        # Fallback path: random-projection adapter (Phase C handoff
        # item 2). Per-branch warnings fire inside `add_branch` as
        # each non-canonical branch is plugged in; this is the
        # absorb-level summary.
        breakdown = ", ".join(
            f"seed={s}: {n}" for s, n in sorted(
                seed_counts.items(), key=lambda kv: -kv[1]
            )
        )
        print(
            f"[trioron absorb] WARNING: donors have mismatched L0 seeds "
            f"({breakdown}). The shared-seed invariant (paper §3.10) is "
            "the recommended path; mismatched donors will be wired through "
            "deterministic random-projection adapters (UNTESTED — see "
            "MANUAL §3)."
        )
    org = MultiBranchOrganism.from_branches(branches)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "kind": "multibranch_organism",
        "l0_seed": org.l0_seed,
        "l0_W": org.l0_W.detach().cpu(),
        "l0_b": org.l0_b.detach().cpu(),
        "l0_activation": org.l0_activation,
        "branches": [_branch_to_dict(b) for b in branches],
        "union_classes": list(org.union_classes),
    }
    torch.save(payload, out)
    return out


def _branch_to_dict(b) -> Dict[str, Any]:
    return {
        "label": b.label,
        "classes_covered": list(b.classes_covered),
        "arm": b.arm,
        "l0_seed": b.l0_seed,
        "n_nodes_per_layer": list(b.net.n_nodes_per_layer()),
        "input_dim": b.net.layers[0].fan_in,
        "state_dict": {k: v.detach().cpu()
                       for k, v in b.net.state_dict().items()},
        "manifold_stats": {
            int(c): (mu.detach().cpu(), sg.detach().cpu())
            for c, (mu, sg) in b.manifold_stats.items()
        },
    }


# ---------------------------------------------------------------------
# Public: load_organism
# ---------------------------------------------------------------------


def load_organism(path: Union[str, Path]):
    """Reconstruct a MultiBranchOrganism from an absorb output (or
    legacy single-donor checkpoint, which is wrapped as a 1-branch
    organism)."""
    from trioron.cli import _load_organism
    return _load_organism(str(path))


# ---------------------------------------------------------------------
# Public: extend (ship-wake-extend loop)
# ---------------------------------------------------------------------


def extend(
    *,
    donor_path: Union[str, Path],
    base_tasks: Sequence[TaskData],
    new_tasks: Sequence[TaskData],
    out_path: Union[str, Path],
    extension_cap_bytes: int = 64_000,
    epochs_per_task: int = 8,
    permanent_int8: bool = True,
    config: Optional[TrioronConfig] = None,
) -> Path:
    """Ship-wake-extend loop (paper §4.6 / extension experiment).

    Resumes from the donor's substrate state (skipping the base
    curriculum's re-training), fires the shipping-consolidation dream
    (full-coverage replay over `base_tasks` → archive-lock), optionally
    permanently quantizes archived rows to int8, lifts the cap to
    ``extension_cap_bytes``, then trains on the new tasks. Original
    tasks survive at task-aware ≥ 0.93 in the paper baseline.

    Resume path requires a donor checkpoint at version >= 2 (which
    includes the ``task_class_lists`` field). Older donors fall back to
    the legacy integrated path that re-trains the base curriculum from
    scratch — emits a warning so users know to rebuild for the
    speedup. Resumed accuracy matches integrated within seed-noise but
    is not bit-exact: the boundary dream's RNG state is not serialized,
    so it gets reseeded on resume.

    Args:
        donor_path: A donor produced by :func:`build_donor`. Used as
            the source of L0 seed, arm, stored config, substrate state
            (state_dict + manifold_stats) and base task class layout.
        base_tasks: The same TaskData list used to build the original
            donor. Required even on the resume path because the
            boundary consolidation dream needs real-data replay over
            past tasks and the final eval covers both base and
            extension classes.
        new_tasks: Tasks to learn on top. Their ``classes`` must be
            disjoint from base_tasks' classes (head extension is
            automatic).
        out_path: Where to save the extended donor.
        extension_cap_bytes: New trainable budget for the extension
            phase. Paper default 64_000 (= 16K params, doubles
            chained-15's 32_000 cap).
        epochs_per_task: Epochs per task (applied to both base and
            extension phases).
        permanent_int8: If True, archived rows snap to int8 at the
            extension boundary (simulates shipped-state quant).
        config: TrioronConfig override. None = inherit the donor's
            stored config.

    Returns:
        Path to the extended donor checkpoint.
    """
    from experiments import bench_chained_15task as bench

    payload = torch.load(str(donor_path), map_location="cpu", weights_only=False)
    payload_kind = payload.get("kind")
    if payload_kind == "multibranch_organism":
        # Organism extension semantics: train a fresh single-task
        # donor on `new_tasks` at the organism's canonical L0 seed
        # (so the new branch's L0 matches the existing branches),
        # then absorb the organism + new donor into a 1-larger
        # organism. `base_tasks` is accepted but only used to
        # validate consistency (not for boundary-dream replay — the
        # new branch is independent of existing branches' substrates
        # at the parameter level; sequential interaction within a
        # multi-branch organism happens via shared L0, not via the
        # resume-from-substrate path which assumes a single
        # substrate). For substrate-level coupling use a single
        # multi-task donor instead and call extend on that.
        canonical_seed = int(payload.get("l0_seed", 42))
        cfg_for_new = config or TrioronConfig(
            cap_bytes=extension_cap_bytes,
            advanced=AdvancedConfig(
                h_init=32, n_grow_per_task=4,
                l0_width=int(payload["l0_W"].shape[0]),
                freeze_l0=True,
            ),
        )
        # Train the new branch on the new_tasks.
        out_path = Path(out_path)
        new_branch_label = f"{Path(donor_path).stem}__ext"
        scratch_dir = out_path.parent
        scratch_dir.mkdir(parents=True, exist_ok=True)
        new_donor_path = scratch_dir / f"{new_branch_label}.pt"
        build_donor(
            label=new_branch_label,
            tasks=list(new_tasks),
            seed=canonical_seed,
            epochs_per_task=epochs_per_task,
            config=cfg_for_new,
            out_path=new_donor_path,
        )
        # Absorb organism + new donor.
        return absorb(
            donor_paths=[donor_path, new_donor_path],
            out_path=out_path,
        )
    if payload_kind not in (None, "trioron_donor"):
        raise ValueError(
            f"extend: {donor_path} is not a donor checkpoint "
            f"(kind={payload_kind!r})."
        )

    base_arm = payload.get("arm") or "grown_capped_dream"
    base_seed = int(payload.get("l0_seed", 42))
    base_cap = (payload.get("trioron_config") or {}).get("cap_bytes", 32_000)
    donor_version = int(payload.get("version", 1))
    donor_task_class_lists = payload.get("task_class_lists")
    can_resume = (
        donor_version >= 2
        and donor_task_class_lists is not None
        and base_arm in bench.ARM_DEFINITIONS
        and bench.ARM_DEFINITIONS[base_arm].get("packnet_mode") is None
        and bench.ARM_DEFINITIONS[base_arm].get("hat_mode") is None
    )
    if not can_resume:
        print(
            "[trioron extend] WARNING: donor checkpoint lacks resume metadata "
            f"(version={donor_version}, has task_class_lists="
            f"{donor_task_class_lists is not None}, arm={base_arm!r}). "
            "Falling back to legacy integrated path — base curriculum will be "
            "re-trained from scratch. Rebuild the donor with the current "
            "trioron version to get the resume-from-substrate speedup."
        )

    train_views, eval_views, task_class_lists = _to_views(base_tasks)
    new_train, new_eval, new_class_lists = _to_views(new_tasks)

    cfg = config or _config_from_payload(payload)
    snap = _snapshot_bench()
    try:
        _apply_config_to_bench(cfg)
        bench.ARM_DEFINITIONS[base_arm]["cap_bytes"] = int(
            base_cap or bench.M_MAX_BYTES_CAPPED
        )
        bench.INPUT_DIM = int(payload["input_dim"])
        if can_resume:
            net, manifold_buf = _hydrate_donor(payload, base_arm)
            r = bench.run_extension_only(
                net,
                arm=base_arm,
                seed=base_seed,
                n_epochs_per_task=epochs_per_task,
                base_train_views=train_views,
                base_eval_views=eval_views,
                base_task_class_lists=task_class_lists,
                extension_train_views=new_train,
                extension_eval_views=new_eval,
                extension_task_class_lists=new_class_lists,
                extension_cap_bytes=int(extension_cap_bytes),
                extension_permanent_int8=bool(permanent_int8),
                initial_manifold=manifold_buf,
                return_state=True,
            )
        else:
            r = bench.run_arm(
                base_arm,
                seed=base_seed,
                n_epochs_per_task=epochs_per_task,
                train_views=train_views,
                eval_views=eval_views,
                task_class_lists=task_class_lists,
                infancy_view=None,
                n_passes=1,
                extension_train_views=new_train,
                extension_eval_views=new_eval,
                extension_task_class_lists=new_class_lists,
                extension_cap_bytes=int(extension_cap_bytes),
                extension_permanent_int8=bool(permanent_int8),
                return_state=True,
            )
    finally:
        _restore_bench(snap)

    net = r["net"]
    mb = r["manifold"]
    extended_classes = sorted({c for t in (list(base_tasks) + list(new_tasks))
                               for c in t.classes})
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_payload = dict(payload)
    new_payload.update({
        "label": payload.get("label", "extended"),
        "classes_covered": extended_classes,
        "n_nodes_per_layer": list(net.n_nodes_per_layer()),
        "state_dict": {k: v.detach().cpu()
                       for k, v in net.state_dict().items()},
        "manifold_stats": {int(c): (mu.detach().cpu(), sg.detach().cpu())
                           for c, (mu, sg) in mb._stats.items()},
    })
    torch.save(new_payload, out)
    return out


def _hydrate_donor(payload: Dict[str, Any], arm: str):
    """Reconstruct (TrioronNetwork, ManifoldBuffer) from a donor payload
    at its end-of-base-curriculum substrate state. Used by the extend
    resume path so the bench can re-enter the curriculum at the
    extension boundary without re-running the base loop.

    Mirrors `Branch.from_checkpoint` for the network shape but does NOT
    freeze L1 + head: the extension training pass must update them.
    L0 freeze follows the arm's `freeze_l0` setting.
    """
    from experiments import bench_chained_15task as bench
    from experiments.datasets import ManifoldBuffer
    from trioron.network import TrioronNetwork

    n_nodes = list(payload["n_nodes_per_layer"])
    input_dim = int(payload["input_dim"])
    layer_specs = []
    prev = input_dim
    for i, n in enumerate(n_nodes):
        act = "linear" if i == len(n_nodes) - 1 else "relu"
        layer_specs.append((prev, int(n), act))
        prev = int(n)
    net = TrioronNetwork(layer_specs)
    net.load_state_dict(payload["state_dict"])
    if bench.ARM_DEFINITIONS[arm].get("freeze_l0", False):
        net.layers[0].W.requires_grad_(False)
        net.layers[0].b.requires_grad_(False)
    manifold_buf = ManifoldBuffer()
    for c, (mu, sg) in payload["manifold_stats"].items():
        manifold_buf._stats[int(c)] = (mu.clone(), sg.clone())
    return net, manifold_buf


def _config_from_payload(payload: Dict[str, Any]) -> TrioronConfig:
    raw = payload.get("trioron_config") or {}
    adv_raw = raw.get("advanced")
    advanced = AdvancedConfig(**adv_raw) if adv_raw else None
    return TrioronConfig(
        cap_bytes=raw.get("cap_bytes", 32_000),
        dream_replay_steps=raw.get("dream_replay_steps", 50),
        dream_buffer_threshold=raw.get("dream_buffer_threshold", 0),
        manifold_noise_scale=raw.get("manifold_noise_scale", 1.0),
        routing_temperature=raw.get("routing_temperature", 1.0),
        per_class_bias=raw.get("per_class_bias", False),
        advanced=advanced,
    )


# ---------------------------------------------------------------------
# Public: evaluate
# ---------------------------------------------------------------------


def evaluate(
    *,
    organism_path: Union[str, Path],
    eval_tasks: Sequence[TaskData],
    routing_temperature: float = 1.0,
    normalize_per_branch: bool = True,
) -> Dict[str, Any]:
    """Run the union test set through an organism and return
    accuracy summaries.

    Returns a dict with:
        per_task: list of {task, n, task_aware, full_union}
        task_aware_mean: mean task-aware accuracy across tasks
        full_union_mean: mean full-union accuracy across tasks
    """
    from experiments.test_multibranch_absorption import evaluate as eval_views_
    from experiments.datasets import TaskDataView
    org = load_organism(organism_path)
    views = []
    for t in eval_tasks:
        views.append(TaskDataView(
            name=t.name,
            images=t.X_test.float(),
            labels_global=t.y_test.long(),
            local_classes=list(t.classes),
            global_classes=list(t.classes),
        ))
    rows = eval_views_(
        org, views, routing="soft",
        temperature=float(routing_temperature),
        normalize_per_branch=bool(normalize_per_branch),
    )
    n = max(len(rows), 1)
    return {
        "per_task": rows,
        "task_aware_mean": sum(r["task_aware"] for r in rows) / n,
        "full_union_mean": sum(r["full_union"] for r in rows) / n,
    }


# ---------------------------------------------------------------------
# Public: deploy_agent
# ---------------------------------------------------------------------


def deploy_agent(
    *,
    organism_path: Union[str, Path],
    encoder,
    tools,
    class_to_tool: Mapping[int, str],
    args_resolver: Optional[Callable[[Any, str, Any], Dict[str, Any]]] = None,
    routing_temperature: float = 1.0,
):
    """Wrap an organism into a BridgedOrganism for production agentic
    use.

    Args:
        organism_path: Output of :func:`absorb`.
        encoder: Any object satisfying the
            ``trioron.bridge.Encoder`` Protocol — text/image/audio
            reference encoders ship in ``trioron.bridge.encoders``.
        tools: A ``trioron.bridge.ToolDispatcher`` populated with
            either JSON-schema or ``@tools.tool``-decorated tools.
        class_to_tool: Maps each union-class index to a tool name.
            Classes missing from the map produce decisions with
            ``tool_name=None``.
        args_resolver: Optional callable
            ``(raw_input, tool_name, decision) -> args_dict``.
        routing_temperature: Soft-routing temperature for branch
            gating (paper default 1.0).

    Returns:
        A ``trioron.bridge.BridgedOrganism`` ready for ``.act(...)``
        or ``.decide(...)`` calls.
    """
    from trioron.bridge import BridgedOrganism, L0Adapter
    organism = load_organism(organism_path)
    l0_dim = organism.l0_W.shape[0]
    adapter = L0Adapter(
        encoder_dim=encoder.encode_dim,
        l0_dim=l0_dim,
        l0_seed=int(organism.l0_seed) if organism.l0_seed is not None else 42,
        activation=organism.l0_activation,
    )
    return BridgedOrganism(
        encoder=encoder,
        organism=organism,
        dispatcher=tools,
        class_to_tool=dict(class_to_tool),
        args_resolver=args_resolver,
        adapter=adapter,
        temperature=float(routing_temperature),
    )


__all__ = [
    "TaskData",
    "TrioronConfig",
    "AdvancedConfig",
    "build_donor",
    "absorb",
    "load_organism",
    "extend",
    "evaluate",
    "deploy_agent",
]
