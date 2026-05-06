"""Build + push the Trioron TTS Space.

Assembles `hf_space_build/` from the project's `trioron/` and
`experiments/` source plus the demo config in
`examples/hf_space_demo/`, then pushes to a private HF Space.

Two-step usage (so authentication happens explicitly):

    # 1. Authenticate once. Use a token with write scope on Spaces.
    huggingface-cli login

    # 2. Build + push.
    python3 examples/hf_space_demo/build_and_push.py \\
        --space-id YOUR_HF_USERNAME/trioron-tts-demo

The script creates the Space with `private=True` if it does not yet
exist; if it already exists, the visibility setting is left untouched
(use the HF web UI to flip it later when the paper publishes).
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
BUILD_DIR = PROJECT_ROOT / "hf_space_build"

# What gets shipped into the Space root.
DEMO_FILES = ["app.py", "README.md", "requirements.txt"]

# trioron/ is small (~21 modules, ~5K LOC) — ship whole.
SOURCE_DIRS = ["trioron"]

# experiments/ is huge (~50 paper-bench scripts) and only TWO files are
# touched at runtime by the demo: datasets.py (TaskDataView,
# ManifoldBuffer, etc. imported at top of bench_chained_15task) and
# bench_chained_15task.py itself (api.build_donor → bench.run_arm).
# Allowlist precisely instead of copytree-ing the whole directory.
EXPERIMENTS_ALLOWLIST = [
    "__init__.py",
    "datasets.py",
    "bench_chained_15task.py",
]


def _assemble() -> Path:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)
    print(f"[build] assembling Space at {BUILD_DIR}")
    for f in DEMO_FILES:
        src = HERE / f
        dst = BUILD_DIR / f
        shutil.copy2(src, dst)
        print(f"  + {f}")
    for d in SOURCE_DIRS:
        src = PROJECT_ROOT / d
        dst = BUILD_DIR / d
        shutil.copytree(
            src, dst,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.egg-info", "outputs",
            ),
        )
        print(f"  + {d}/")
    exp_dst = BUILD_DIR / "experiments"
    exp_dst.mkdir()
    for f in EXPERIMENTS_ALLOWLIST:
        src = PROJECT_ROOT / "experiments" / f
        if not src.exists():
            raise FileNotFoundError(f"required runtime file missing: {src}")
        shutil.copy2(src, exp_dst / f)
        print(f"  + experiments/{f}")
    # Drop a .gitignore for the Space repo itself.
    (BUILD_DIR / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n*.egg-info/\n.pytest_cache/\n"
        "outputs/\nlogs/\n.DS_Store\n",
        encoding="utf-8",
    )
    return BUILD_DIR


def _push(space_id: str, build_dir: Path, private: bool) -> None:
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError:
        print(
            "[push] huggingface_hub not installed. Run:\n"
            "    pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)
    api = HfApi()
    # Create the Space if it does not exist; ignore if it does.
    try:
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            private=private,
            exist_ok=True,
        )
        print(f"[push] ensured Space {space_id} exists (private={private})")
    except Exception as e:
        print(f"[push] create_repo failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[push] uploading {build_dir} → {space_id}")
    # `delete_patterns` removes remote-only files matching the glob so
    # the Space stays in sync with the local artifact. Without this,
    # files pruned from the build (e.g. experiments/* not in the
    # allowlist) would linger on remote forever. We scope the delete
    # to just experiments/ so unrelated repo-level files (e.g. a
    # README touched via the web UI) survive across pushes.
    upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(build_dir),
        commit_message="trioron-tts demo: sync (allowlist + orphan cleanup)",
        delete_patterns=["experiments/*"],
    )
    print(f"[push] done. Space URL: https://huggingface.co/spaces/{space_id}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--space-id", required=True,
        help="HF Space identifier, e.g. 'your-handle/trioron-tts-demo'",
    )
    p.add_argument(
        "--public", action="store_true",
        help="Create the Space public (default: private)",
    )
    p.add_argument(
        "--build-only", action="store_true",
        help="Assemble hf_space_build/ but skip the push step",
    )
    args = p.parse_args()

    build_dir = _assemble()
    if args.build_only:
        print(f"[build] done (build-only). Inspect {build_dir} before push.")
        return 0
    _push(args.space_id, build_dir, private=not args.public)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
