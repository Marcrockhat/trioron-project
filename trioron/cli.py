"""Trioron command-line entry point.

Wraps the production path of trioron — train a donor, absorb donors
into a multi-branch organism, run inference / eval — into four
subcommands. No experimental knobs are exposed here; the underlying
research scripts (`experiments/*.py`) remain available for deeper
reproduction. Reviewers should be able to reproduce the lossless
absorption result on commodity CPU in <5 minutes via:

    pip install -e .                  # or pip install git+https://...
    trioron train  --donor digits   --out donor_digits.pt
    trioron train  --donor fashion  --out donor_fashion.pt
    trioron absorb --donors donor_digits.pt,donor_fashion.pt --out organism.pt
    trioron eval   --organism organism.pt
    trioron infer  --organism organism.pt --image path/to/image.png

All four subcommands are deterministic given the L0 seed (default 42)
and produce the same accuracy numbers reported in the paper at the
2-donor configuration.
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import List, Optional, Sequence

import torch


# ---------------------------------------------------------------------
# train
# ---------------------------------------------------------------------


def _resolve_py_entry(spec: str):
    """Parse 'path/to/file.py:fn_name' (or 'pkg.mod:fn_name') and
    return the resolved attribute. Used by --from-py and --tools to
    let users point the CLI at their own Python entry points."""
    import importlib
    import importlib.util
    if ":" not in spec:
        raise ValueError(
            f"--from-py value must be 'path:fn' (got {spec!r}). "
            "Example: my_loader.py:make_tasks  or  my.pkg:make_tasks"
        )
    target, attr = spec.rsplit(":", 1)
    if target.endswith(".py") or os.sep in target:
        path = os.path.abspath(target)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Python entry file not found: {path}")
        mod_name = "_trioron_user_" + os.path.splitext(os.path.basename(path))[0]
        spec_obj = importlib.util.spec_from_file_location(mod_name, path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"could not load module from {path}")
        mod = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    if not hasattr(mod, attr):
        raise AttributeError(
            f"module from {target!r} has no attribute {attr!r}"
        )
    return getattr(mod, attr)


def _config_from_args(args: argparse.Namespace):
    """Build a TrioronConfig from CLI flags, applying defaults for
    anything not specified (unset CLI flags are None and fall through
    to TrioronConfig's own defaults)."""
    from trioron.api import TrioronConfig, AdvancedConfig

    def _v(attr, default):
        v = getattr(args, attr, None)
        return default if v is None else v

    advanced = None
    if getattr(args, "advanced", False):
        advanced = AdvancedConfig(
            h_init=_v("h_init", 32),
            n_grow_per_task=_v("n_grow_per_task", 4),
            ewc_intertask_strength=_v("ewc_intertask", 30.0),
            ewc_dream_strength=_v("ewc_dream", 30.0),
            dream_replay_fraction=_v("dream_replay_fraction", 0.25),
            dream_compression_action=_v("dream_compression", "starve"),
            dream_max_downscales_per_layer=_v("dream_max_downscales", 1),
            dream_apoptosis_on=_v("dream_apoptosis", True),
            freeze_l0=not getattr(args, "no_freeze_l0", False),
            l0_width=_v("l0_width", 128),
        )
    return TrioronConfig(
        cap_bytes=_v("cap_bytes", 32_000),
        dream_replay_steps=_v("dream_replay_steps", 50),
        dream_buffer_threshold=_v("dream_buffer_threshold", 0),
        manifold_noise_scale=_v("manifold_noise_scale", 1.0),
        routing_temperature=_v("routing_temperature", 1.0),
        per_class_bias=getattr(args, "per_class_bias", False),
        advanced=advanced,
    )


def _add_tune_args(p: argparse.ArgumentParser) -> None:
    """Attach the distinctive trioron knobs to a subcommand parser
    (shared between train and extend)."""
    p.add_argument("--cap-bytes", type=int, default=None,
                   help="trainable byte budget (default 32_000)")
    p.add_argument("--dream-replay-steps", type=int, default=None,
                   help="replay batches per dream cycle (default 50)")
    p.add_argument("--dream-buffer-threshold", type=int, default=None,
                   help="min past tasks before first dream (default 0)")
    p.add_argument("--manifold-noise-scale", type=float, default=None,
                   help="σ multiplier for manifold sampling (default 1.0)")
    p.add_argument("--routing-temperature", type=float, default=None,
                   help="organism-level soft-routing T (default 1.0)")
    p.add_argument("--per-class-bias", action="store_true",
                   help="enable per-class bias offsets at eval "
                        "(dream-cycle calibration)")
    # Advanced — only consulted when --advanced is also set.
    p.add_argument("--advanced", action="store_true",
                   help="also apply the --h-init / --n-grow-per-task "
                        "/ EWC / dream advanced overrides")
    p.add_argument("--h-init", type=int, default=None,
                   help="(advanced) initial L1 hidden width")
    p.add_argument("--n-grow-per-task", type=int, default=None,
                   help="(advanced) nodes added per growth event")
    p.add_argument("--ewc-intertask", type=float, default=None,
                   help="(advanced) EWC strength between tasks")
    p.add_argument("--ewc-dream", type=float, default=None,
                   help="(advanced) EWC strength inside dream replay")
    p.add_argument("--dream-replay-fraction", type=float, default=None,
                   help="(advanced) fraction of past tasks per dream")
    p.add_argument("--dream-compression", default=None,
                   choices=["starve", "merge", "none"],
                   help="(advanced) dream compression action "
                        "(default starve)")
    p.add_argument("--dream-max-downscales", type=int, default=None,
                   help="(advanced) per-layer downscale cap (sRNA)")
    p.add_argument("--dream-apoptosis", action="store_true",
                   help="(advanced) enable apoptosis spike-decay")
    p.add_argument("--no-freeze-l0", action="store_true",
                   help="(advanced) train L0 instead of freezing")
    p.add_argument("--l0-width", type=int, default=None,
                   help="(advanced) L0 random-projection width")


def cmd_train(args: argparse.Namespace) -> int:
    """Train one trioron donor — either from a built-in chained-15
    split (``--donor``) or from a user-supplied loader
    (``--from-py path:fn``)."""
    if args.from_py:
        from trioron.api import build_donor
        try:
            loader = _resolve_py_entry(args.from_py)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        try:
            tasks = loader()
        except Exception as e:
            print(f"error: loader {args.from_py} raised: {e}",
                  file=sys.stderr)
            return 2
        if not isinstance(tasks, (list, tuple)) or not tasks:
            print("error: loader must return a non-empty list of "
                  "trioron.api.TaskData objects", file=sys.stderr)
            return 2
        cfg = _config_from_args(args)
        label = args.label or os.path.splitext(os.path.basename(args.out))[0]
        out = build_donor(
            label=label, tasks=tasks, seed=args.seed,
            epochs_per_task=args.epochs,
            config=cfg, out_path=args.out,
        )
        print(f"\n[trioron train] saved donor → {out}")
        return 0

    # Fall through to the built-in chained-15 sub-block path.
    if not args.donor:
        print("error: provide either --donor <split> or "
              "--from-py path:fn", file=sys.stderr)
        return 2
    from experiments import train_donor as td
    if args.donor not in td.SPLIT_BLOCKS:
        print(f"error: unknown donor split '{args.donor}'. "
              f"choices: {sorted(td.SPLIT_BLOCKS)}",
              file=sys.stderr)
        return 2
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    sub_argv: List[str] = [
        "--label", args.donor,
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--out-dir", out_dir,
    ]
    if args.data_root:
        sub_argv += ["--data-root", args.data_root]
    rc = td.main(sub_argv)
    if rc != 0:
        return rc
    src = os.path.join(out_dir, f"poc_donor_{args.donor}.pt")
    if os.path.abspath(src) != os.path.abspath(args.out):
        os.replace(src, args.out)
    print(f"\n[trioron train] saved donor → {args.out}")
    return 0


# ---------------------------------------------------------------------
# absorb
# ---------------------------------------------------------------------


def cmd_absorb(args: argparse.Namespace) -> int:
    """Assemble a multi-branch organism from saved donors."""
    from trioron.multibranch import Branch, MultiBranchOrganism
    paths = [p.strip() for p in args.donors.split(",") if p.strip()]
    if len(paths) < 1:
        print("error: --donors is empty", file=sys.stderr)
        return 2
    branches = []
    for p in paths:
        if not os.path.exists(p):
            print(f"error: donor checkpoint not found: {p}", file=sys.stderr)
            return 2
        b = Branch.from_checkpoint(p)
        branches.append(b)
        print(f"  loaded {b.label:<12} arch={list(b.net.n_nodes_per_layer())}  "
              f"classes={b.classes_covered}  "
              f"l0_seed={b.l0_seed}")
    seeds = {b.l0_seed for b in branches}
    if len(seeds) > 1:
        print(f"error: donors have mismatched L0 seeds {seeds} — "
              "shared-seed invariant is required for paste-and-go absorption.",
              file=sys.stderr)
        return 2
    org = MultiBranchOrganism.from_branches(branches)
    payload = {
        "version": 1,
        "kind": "multibranch_organism",
        "l0_seed": next(iter(seeds)),
        "l0_W": org.l0_W.detach().cpu(),
        "l0_b": org.l0_b.detach().cpu(),
        "l0_activation": org.l0_activation,
        "branches": [
            {
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
            for b in branches
        ],
        "union_classes": list(org.union_classes),
    }
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, args.out)
    sb = org.storage_bytes()
    print(f"\n[trioron absorb] organism with {len(branches)} branch(es) → {args.out}")
    print(f"  union_classes = {org.union_classes}")
    print(f"  storage: {sb['total_bytes'] / 1024:.1f} KB total "
          f"(L0 {sb['l0_bytes']/1024:.0f} KB shared, "
          f"branch substrate {sb['branch_substrate_bytes']/1024:.0f} KB, "
          f"archive {sb['archive_bytes']/1024:.0f} KB)")
    return 0


# ---------------------------------------------------------------------
# Helper: rebuild an organism from a saved organism payload
# ---------------------------------------------------------------------


def _load_organism(path: str):
    """Reconstruct a MultiBranchOrganism from a `trioron absorb`
    checkpoint OR (for convenience) from a legacy single-donor
    poc_donor_*.pt — that becomes a 1-branch organism."""
    from trioron.multibranch import Branch, MultiBranchOrganism
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") == "multibranch_organism":
        # Rebuild branches inline (skips the per-branch checkpoint files).
        from trioron.network import TrioronNetwork
        branches = []
        for d in payload["branches"]:
            n_nodes = d["n_nodes_per_layer"]
            specs = []
            prev = d["input_dim"]
            for i, n in enumerate(n_nodes):
                act = "linear" if i == len(n_nodes) - 1 else "relu"
                specs.append((prev, n, act))
                prev = n
            net = TrioronNetwork(specs)
            net.load_state_dict(d["state_dict"])
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            branches.append(Branch(
                label=d["label"], classes_covered=d["classes_covered"],
                net=net, manifold_stats=d["manifold_stats"],
                l0_seed=d.get("l0_seed"), arm=d.get("arm"),
            ))
        return MultiBranchOrganism.from_branches(branches)
    if "manifold_stats" in payload and "state_dict" in payload:
        # Legacy single-donor: wrap as a 1-branch organism.
        b = Branch.from_checkpoint(path)
        return MultiBranchOrganism.from_branches([b])
    raise ValueError(
        f"file at {path} is not a recognized trioron checkpoint "
        "(missing 'kind' or 'manifold_stats')"
    )


# ---------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate an organism on the union test set of its donors' tasks.

    Two data sources:
      - default: built-in chained-15 splits inferred from each branch's
        ``label`` (matches the QUICKSTART reproduction path).
      - ``--from-py path:fn``: user loader returning a list of
        ``trioron.api.TaskData`` whose ``X_test``/``y_test`` fields are
        used as the held-out evaluation set.
    """
    from experiments.test_multibranch_absorption import (
        evaluate as eval_views_,
    )
    org = _load_organism(args.organism)
    print(f"[trioron eval] loaded organism with {len(org.branches)} branch(es)")
    print(f"  branches      = {[b.label for b in org.branches]}")
    print(f"  union_classes = {org.union_classes}")

    if args.from_py:
        try:
            loader = _resolve_py_entry(args.from_py)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        try:
            tasks = loader()
        except Exception as e:
            print(f"error: loader {args.from_py} raised: {e}",
                  file=sys.stderr)
            return 2
        if not isinstance(tasks, (list, tuple)) or not tasks:
            print("error: loader must return a non-empty list of "
                  "trioron.api.TaskData objects", file=sys.stderr)
            return 2
        from experiments.datasets import TaskDataView
        eval_views = []
        for t in tasks:
            eval_views.append(TaskDataView(
                name=t.name,
                images=t.X_test.float(),
                labels_global=t.y_test.long(),
                local_classes=list(t.classes),
                global_classes=list(t.classes),
            ))
    else:
        from experiments.datasets import (
            DatasetBundle, build_task_views, DEFAULT_DATA_ROOT,
        )
        from experiments.train_donor import SPLIT_BLOCKS
        bundle_dataset_names = []
        union_specs = []
        for b in org.branches:
            if b.label not in SPLIT_BLOCKS:
                print(f"error: branch label '{b.label}' is not in the trained "
                      f"split registry; eval needs the matching test split. "
                      f"For custom donors, pass --from-py path:fn returning a "
                      f"list of trioron.api.TaskData.",
                      file=sys.stderr)
                return 2
            specs_fn, ds_name = SPLIT_BLOCKS[b.label]
            if ds_name not in bundle_dataset_names:
                bundle_dataset_names.append(ds_name)
            union_specs.extend(specs_fn())
        bundle = DatasetBundle(
            bundle_dataset_names,
            root=args.data_root or DEFAULT_DATA_ROOT,
            n_holdout_per_dataset=0,
        )
        eval_views = build_task_views(bundle, union_specs, split="test")

    rows_norm = eval_views_(
        org, eval_views, routing="soft",
        temperature=args.temperature, normalize_per_branch=True,
    )
    rows_raw = eval_views_(
        org, eval_views, routing="soft",
        temperature=args.temperature, normalize_per_branch=False,
    )
    n = len(rows_norm)
    ta_norm = sum(r["task_aware"] for r in rows_norm) / n
    fu_norm = sum(r["full_union"] for r in rows_norm) / n
    ta_raw = sum(r["task_aware"] for r in rows_raw) / n
    fu_raw = sum(r["full_union"] for r in rows_raw) / n

    print()
    print("Per-task accuracy (soft routing, per-branch log-softmax):")
    print(f"  {'task':<24}{'n':>6}  {'active':<14}"
          f"{'task-aware':>12}{'full-union':>12}")
    print("  " + "-" * 64)
    for r in rows_norm:
        print(f"  {r['task']:<24}{r['n']:>6}  {str(r['active']):<14}"
              f"{r['task_aware']:>12.4f}{r['full_union']:>12.4f}")
    print()
    print("Headline (mean across union):")
    print(f"  task-aware (production) = {ta_norm:.4f}  "
          f"full-union = {fu_norm:.4f}  "
          "← soft + per-branch log-softmax")
    print(f"  task-aware (raw)        = {ta_raw:.4f}  "
          f"full-union = {fu_raw:.4f}  "
          "← soft, no normalization")
    return 0


# ---------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------


def _load_image_as_tensor(path: str) -> torch.Tensor:
    """Load an image and convert to the 28x28 grayscale flattened
    tensor the chained-15 organism expects. Greyscale-MNIST shape =
    (1, 784) float in [0, 1]."""
    from PIL import Image
    from torchvision import transforms
    img = Image.open(path).convert("L")
    pre = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),  # (1, 28, 28) in [0, 1]
    ])
    x = pre(img)
    return x.view(1, -1)        # (1, 784)


def cmd_infer(args: argparse.Namespace) -> int:
    """Single-image inference. Reports top-k predictions from the union
    softmax (per-branch log-softmax composition)."""
    org = _load_organism(args.organism)
    x = _load_image_as_tensor(args.image)
    with torch.no_grad():
        logits, extras = org(
            x, routing="soft",
            temperature=args.temperature,
            normalize_per_branch=True,
            return_extras=True,
        )
    probs = torch.softmax(logits[0], dim=-1)
    topk = torch.topk(probs, k=min(args.topk, probs.numel()))
    union = org.union_classes
    print(f"[trioron infer] image={args.image}")
    print(f"  branches      = {[b.label for b in org.branches]}")
    print(f"  union_classes = {union}")
    print(f"  routing gates = {extras['gates'][0].tolist()}")
    print()
    print(f"Top-{topk.values.numel()} predictions:")
    for p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"  class {union[int(idx)]:>3}  prob {p:.4f}")
    return 0


# ---------------------------------------------------------------------
# tune  —  inspect or print the architecturally-distinctive knobs
# ---------------------------------------------------------------------


def cmd_tune(args: argparse.Namespace) -> int:
    """Show / inspect the distinctive knobs.

    With ``--inspect <donor.pt>``: print the config baked into the
    saved donor by ``build_donor``.

    With ``--show``: print all knob defaults + descriptions, as a
    cheatsheet for what to pass to ``trioron train``.
    """
    if args.inspect:
        payload = torch.load(args.inspect, map_location="cpu",
                             weights_only=False)
        cfg = payload.get("trioron_config")
        if not cfg:
            print(f"[trioron tune] {args.inspect}: no trioron_config "
                  "stored — donor was built before knob-snapshotting "
                  "was added (legacy poc_donor_*.pt files have no "
                  "config block).")
            return 0
        print(f"[trioron tune] {args.inspect}")
        print(f"  label             = {payload.get('label')!r}")
        print(f"  l0_seed           = {payload.get('l0_seed')}")
        print(f"  arm               = {payload.get('arm')!r}")
        print(f"  classes_covered   = {payload.get('classes_covered')}")
        print(f"  arch              = {payload.get('n_nodes_per_layer')}")
        print(f"  input_dim         = {payload.get('input_dim')}")
        print()
        print("  --- TrioronConfig ---")
        for k in ("cap_bytes", "dream_replay_steps",
                  "dream_buffer_threshold", "manifold_noise_scale",
                  "routing_temperature", "per_class_bias"):
            print(f"  {k:<25} = {cfg.get(k)}")
        adv = cfg.get("advanced")
        if adv:
            print()
            print("  --- AdvancedConfig (growth) ---")
            for k, v in adv.items():
                print(f"  {k:<25} = {v}")
        return 0

    # Default: --show
    print("Trioron — distinctive tunable knobs (pass to `trioron train`):")
    print()
    print("  Primary (architecturally distinctive vs PackNet / HAT / EWC / LwF):")
    print("    --cap-bytes              hard byte budget for trainable params  "
          "(default 32_000)")
    print("    --dream-replay-steps     replay batches per dream cycle         "
          "(default 50)")
    print("    --dream-buffer-threshold min past tasks before first dream      "
          "(default 0)")
    print("    --manifold-noise-scale   σ multiplier for manifold sampling     "
          "(default 1.0)")
    print("    --routing-temperature    soft-routing T at organism-level       "
          "(default 1.0)")
    print("    --per-class-bias         enable dream-cycle bias calibration    "
          "(default off)")
    print()
    print("  Advanced (gated behind --advanced; wrong values silently kill growth):")
    print("    --h-init                 initial L1 hidden width                 "
          "(default 32)")
    print("    --n-grow-per-task        nodes added per growth event            "
          "(default 4)")
    print("    --ewc-intertask          EWC strength between tasks              "
          "(default 30.0)")
    print("    --ewc-dream              EWC strength inside dream replay        "
          "(default 30.0)")
    print("    --dream-replay-fraction  fraction of past tasks per dream         "
          "(default 0.25)")
    print("    --dream-compression      starve | merge | none                    "
          "(default starve)")
    print("    --dream-max-downscales   per-layer downscale cap (sRNA-style)     "
          "(default 1)")
    print("    --dream-apoptosis        on/off for apoptosis spike-decay         "
          "(default on)")
    print("    --no-freeze-l0           train L0 instead of freezing             "
          "(default frozen)")
    print("    --l0-width               L0 random-projection width               "
          "(default 128)")
    print()
    print("Tip: run `trioron tune --inspect donor.pt` to see what knobs a "
          "saved donor was built with.")
    return 0


# ---------------------------------------------------------------------
# extend  —  ship-wake-extend loop
# ---------------------------------------------------------------------


def cmd_extend(args: argparse.Namespace) -> int:
    """Extend an existing donor with new tasks.

    Wraps the integrated base+extension curriculum from
    ``experiments/bench_chained_extend.py``:
      1. Replay the base curriculum (tasks loaded via --base-py) on
         the donor's L0 seed and arm.
      2. Fire the shipping-consolidation dream (full-coverage replay
         → archive-lock).
      3. Permanently snap archived rows to int8 (unless
         --no-permanent-int8).
      4. Lift the cap to --extension-cap-bytes.
      5. Train on the new tasks loaded via --new-py.
    """
    from trioron.api import extend
    try:
        base_loader = _resolve_py_entry(args.base_py)
        new_loader = _resolve_py_entry(args.new_py)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    base_tasks = base_loader()
    new_tasks = new_loader()
    cfg = _config_from_args(args) if any(
        getattr(args, k, None) is not None
        for k in ("cap_bytes", "dream_replay_steps", "manifold_noise_scale")
    ) else None
    out = extend(
        donor_path=args.donor,
        base_tasks=base_tasks,
        new_tasks=new_tasks,
        out_path=args.out,
        extension_cap_bytes=args.extension_cap_bytes,
        epochs_per_task=args.epochs,
        permanent_int8=not args.no_permanent_int8,
        config=cfg,
    )
    print(f"\n[trioron extend] saved extended donor → {out}")
    return 0


# ---------------------------------------------------------------------
# serve  —  REPL + HTTP for BridgedOrganism
# ---------------------------------------------------------------------


def _build_bridge(args: argparse.Namespace):
    """Shared bridge construction for both REPL and HTTP serve modes."""
    from trioron.api import deploy_agent
    encoder_factory = _resolve_py_entry(args.encoder)
    encoder = encoder_factory() if callable(encoder_factory) else encoder_factory
    tools = _resolve_py_entry(args.tools)
    if callable(tools):
        # Allow a function returning the dispatcher, for parity with
        # encoder factories.
        try:
            from trioron.bridge import ToolDispatcher
            maybe = tools()
            if isinstance(maybe, ToolDispatcher):
                tools = maybe
        except TypeError:
            pass
    class_to_tool = _resolve_py_entry(args.class_map)
    if callable(class_to_tool) and not isinstance(class_to_tool, dict):
        class_to_tool = class_to_tool()
    args_resolver = None
    if args.args_resolver:
        args_resolver = _resolve_py_entry(args.args_resolver)
    return deploy_agent(
        organism_path=args.organism,
        encoder=encoder,
        tools=tools,
        class_to_tool=class_to_tool,
        args_resolver=args_resolver,
        routing_temperature=args.temperature,
    )


def cmd_serve(args: argparse.Namespace) -> int:
    """REPL and/or HTTP wrapper around a BridgedOrganism.

    REPL mode is always available. HTTP mode requires the ``[serve]``
    optional extra (``pip install trioron[serve]``) and is started
    when ``--http <port>`` is given. Both modes can run together; the
    HTTP server runs in a background thread so the REPL stays
    interactive.
    """
    bridge = _build_bridge(args)
    print(f"[trioron serve] organism      = {args.organism}")
    print(f"[trioron serve] union_classes = {bridge.organism.union_classes}")
    print(f"[trioron serve] tools         = {bridge.dispatcher.names()}")
    if args.http is not None:
        try:
            import uvicorn  # noqa: F401
            from fastapi import FastAPI  # noqa: F401
        except ImportError:
            print("error: trioron serve --http requires the [serve] "
                  "extra. Install it via `pip install trioron[serve]`.",
                  file=sys.stderr)
            return 2
        import threading
        from trioron.serve_http import build_app, run_uvicorn
        app = build_app(bridge)
        t = threading.Thread(
            target=run_uvicorn, args=(app, args.http),
            daemon=True,
        )
        t.start()
        print(f"[trioron serve] HTTP listening on http://0.0.0.0:{args.http}")
    if args.no_repl:
        if args.http is None:
            print("error: --no-repl requires --http <port>; nothing to "
                  "serve.", file=sys.stderr)
            return 2
        # Block forever so the daemon HTTP thread keeps running.
        import time
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\n[trioron serve] stopped.")
            return 0
    print("\n[trioron serve] REPL mode. Type a query and press <Enter>. "
          "Type 'quit' or send EOF to exit.")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() in ("quit", "exit"):
            break
        if not line.strip():
            continue
        try:
            result = bridge.act(line)
        except Exception as e:
            print(f"  error: {e}")
            continue
        d = result["decision"]
        print(f"  -> class {d.union_class}  "
              f"tool={d.tool_name!r}  conf={d.confidence:.3f}")
        if result["tool_call"] is not None:
            print(f"  call: {result['tool_call']}")
            print(f"  result: {result['tool_result']}")
    return 0


# ---------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trioron",
        description=(
            "Continual-learning architecture with archive-routed "
            "multi-branch absorption."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    p_train = sub.add_parser(
        "train",
        help="train one donor (built-in chained-15 split OR custom "
             "data via --from-py)",
    )
    src = p_train.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--donor",
        choices=["digits", "fashion", "emnist", "emnist_kt", "emnist_uz"],
        help="which built-in sub-block to train on (paper §4.6)",
    )
    src.add_argument(
        "--from-py", default=None, dest="from_py", metavar="path:fn",
        help="custom loader: 'my_loader.py:make_tasks' returning a "
             "list of trioron.api.TaskData objects",
    )
    p_train.add_argument("--label", default=None,
                         help="donor label (defaults to filename stem "
                              "or --donor value)")
    p_train.add_argument("--seed", type=int, default=42,
                         help="shared L0 seed (default 42, must match "
                              "across all donors)")
    p_train.add_argument("--epochs", type=int, default=8,
                         help="epochs per task (default 8)")
    p_train.add_argument("--data-root", default=None,
                         help="dataset cache directory for built-in "
                              "splits (default: <repo>/outputs/data)")
    _add_tune_args(p_train)
    p_train.add_argument("--out", required=True,
                         help="output donor checkpoint path")
    p_train.set_defaults(func=cmd_train)

    p_tune = sub.add_parser("tune",
                            help="show distinctive knobs or inspect a "
                                 "saved donor's config")
    grp = p_tune.add_mutually_exclusive_group(required=True)
    grp.add_argument("--show", action="store_true",
                     help="print all distinctive knobs + defaults")
    grp.add_argument("--inspect", metavar="DONOR.PT",
                     help="print the config baked into a saved donor")
    p_tune.set_defaults(func=cmd_tune)

    p_extend = sub.add_parser(
        "extend",
        help="ship-wake-extend loop: load donor, consolidate, lift "
             "cap, train on new tasks",
    )
    p_extend.add_argument("--donor", required=True,
                          help="donor checkpoint to extend "
                               "(from `trioron train`)")
    p_extend.add_argument("--base-py", required=True, metavar="path:fn",
                          help="loader for the original base curriculum "
                               "(re-played because the extension module "
                               "fuses base+extension into one run)")
    p_extend.add_argument("--new-py", required=True, metavar="path:fn",
                          help="loader for the new tasks to learn on "
                               "top (must use disjoint global classes)")
    p_extend.add_argument("--extension-cap-bytes", type=int,
                          default=64_000,
                          help="lifted byte budget for the extension "
                               "phase (default 64_000)")
    p_extend.add_argument("--epochs", type=int, default=8,
                          help="epochs per task in both phases "
                               "(default 8)")
    p_extend.add_argument("--no-permanent-int8", action="store_true",
                          help="skip the int8 archive snap at the "
                               "extension boundary (keeps fp32 archive)")
    _add_tune_args(p_extend)
    p_extend.add_argument("--out", required=True,
                          help="output extended donor checkpoint path")
    p_extend.set_defaults(func=cmd_extend)

    p_absorb = sub.add_parser("absorb",
                              help="assemble a multi-branch organism "
                                   "from saved donors (zero-shot)")
    p_absorb.add_argument("--donors", required=True,
                          help="comma-separated donor checkpoint paths")
    p_absorb.add_argument("--out", required=True,
                          help="output organism checkpoint path")
    p_absorb.set_defaults(func=cmd_absorb)

    p_eval = sub.add_parser("eval",
                            help="evaluate an organism on the union "
                                 "test set of its donors' tasks")
    p_eval.add_argument("--organism", required=True,
                        help="organism checkpoint path (or a single donor "
                             ".pt — wraps as a 1-branch organism)")
    p_eval.add_argument("--from-py", default=None, dest="from_py",
                        metavar="path:fn",
                        help="custom test loader: 'my_loader.py:make_eval_tasks' "
                             "returning a list of trioron.api.TaskData; "
                             "X_test/y_test are used as the held-out set. "
                             "Required for organisms whose branches were "
                             "built with `train --from-py`.")
    p_eval.add_argument("--temperature", type=float, default=1.0,
                        help="soft-routing temperature (default 1.0)")
    p_eval.add_argument("--data-root", default=None,
                        help="dataset cache for built-in chained-15 splits "
                             "(ignored when --from-py is set)")
    p_eval.set_defaults(func=cmd_eval)

    p_infer = sub.add_parser("infer",
                             help="run single-image inference through "
                                  "an organism")
    p_infer.add_argument("--organism", required=True)
    p_infer.add_argument("--image", required=True,
                         help="path to a 28x28-resizable grayscale image")
    p_infer.add_argument("--topk", type=int, default=5)
    p_infer.add_argument("--temperature", type=float, default=1.0)
    p_infer.set_defaults(func=cmd_infer)

    p_serve = sub.add_parser(
        "serve",
        help="deploy an organism as an agent (REPL and/or HTTP)",
    )
    p_serve.add_argument("--organism", required=True,
                         help="organism checkpoint to serve")
    p_serve.add_argument("--encoder", required=True, metavar="path:fn",
                         help="encoder factory or instance "
                              "(e.g. my_agent.py:make_encoder)")
    p_serve.add_argument("--tools", required=True, metavar="path:fn",
                         help="ToolDispatcher instance or factory "
                              "(e.g. my_agent.py:tools)")
    p_serve.add_argument("--class-map", required=True, metavar="path:fn",
                         help="dict[int,str] (or factory) mapping union "
                              "class IDs to tool names")
    p_serve.add_argument("--args-resolver", default=None,
                         metavar="path:fn",
                         help="optional callable "
                              "(raw, tool_name, decision) -> args dict")
    p_serve.add_argument("--temperature", type=float, default=1.0,
                         help="soft-routing temperature (default 1.0)")
    p_serve.add_argument("--http", type=int, default=None, metavar="PORT",
                         help="also start FastAPI on the given port "
                              "(requires `pip install trioron[serve]`)")
    p_serve.add_argument("--no-repl", action="store_true",
                         help="skip the REPL (only valid with --http)")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
