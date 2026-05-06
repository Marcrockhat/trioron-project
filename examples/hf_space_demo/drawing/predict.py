"""Inference + live-extend helpers for the drawing tab."""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from trioron.api import TaskData, extend, load_organism  # noqa: E402

from .data import load_mnist_subset  # noqa: E402
from .state import DrawingSession, EXTEND_CLASSES, TEACH_THRESHOLD  # noqa: E402


def _ensure_loaded(session: DrawingSession) -> None:
    """Lazy-load the organism from session.live_path on first use, and
    reload after every extend() that overwrites live_path."""
    if session.organism is not None:
        return
    session.organism = load_organism(session.live_path)


@torch.no_grad()
def predict(
    session: DrawingSession, image_tensor: torch.Tensor, top_k: int = 3,
) -> List[Tuple[int, float]]:
    """Manifold-archive classification.

    Returns top-k as a list of (global_class, log_lik) sorted by score.
    The full ranking is what the UI shows — single argmax conceals the
    fact that the trioron's manifold archive is a task-aware classifier
    being used as a full-multi-class one. The runner-up scores carry
    real signal."""
    with session._lock:
        _ensure_loaded(session)
        org = session.organism
    branch = org.branches[0]
    x = image_tensor.float().view(1, -1)
    z = org.project_l0(x)
    log_lik = branch.per_class_log_likelihood(z)[0]   # (n_classes,)
    sorted_ll, sorted_idx = log_lik.sort(descending=True)
    out: List[Tuple[int, float]] = []
    for i in range(min(top_k, len(sorted_ll))):
        cls = int(branch.archive_classes[int(sorted_idx[i])])
        out.append((cls, float(sorted_ll[i])))
    return out


def teach(
    session: DrawingSession, image_tensor: torch.Tensor, label: int,
    n_mnist_mix: int = 200,
) -> Tuple[str, bool]:
    """Append `image_tensor` to the buffer for `label`. Once the buffer
    hits TEACH_THRESHOLD samples, trigger extend() to grow a new
    manifold for `label`. Returns (status_message, did_extend).

    The user's own sketches are mixed with `n_mnist_mix` real MNIST
    samples of the same label — a lot of teach buffers (3 sketches in
    a possibly-quirky style) would otherwise produce a manifold the
    archive learns badly. Mixing real samples in stabilises the new
    manifold without the demo claiming "trioron learned it from your
    3 sketches alone." That would be overselling.
    """
    label = int(label)
    if label in session.pretrain_classes:
        return (f"Digit {label} is already in the pretrained set "
                f"(digits 0-4). Try teaching digits 5-9.", False)
    if label not in EXTEND_CLASSES:
        return (f"Digit {label} is outside the extend range. "
                f"Try {EXTEND_CLASSES}.", False)

    buf = session.buffer_for(label)
    buf.append(image_tensor.detach().clone())
    if len(buf) < TEACH_THRESHOLD:
        return (f"Buffered sketch {len(buf)}/{TEACH_THRESHOLD} for digit "
                f"{label}. Need {TEACH_THRESHOLD - len(buf)} more "
                f"before extend fires.", False)

    # Threshold reached → extend
    return _trigger_extend(session, label, buf, n_mnist_mix), True


def _trigger_extend(
    session: DrawingSession, label: int, sketches: List[torch.Tensor],
    n_mnist_mix: int,
) -> str:
    """Call api.extend() with one new task = (sketches + MNIST samples)
    of `label`. Replaces session.live_path on success and clears the
    buffer for that label."""
    with session._lock:
        # Build base_tasks (the pretrain digits) — required by extend()
        # for the consolidation dream's real-data replay.
        base_subset = load_mnist_subset(
            classes=session.pretrain_classes, n_per_class=80, seed=42,
        )
        base_tasks = [
            TaskData(
                name=f"digit_{c}",
                X_train=base_subset[c]["X_train"],
                y_train=base_subset[c]["y_train"],
                X_test=base_subset[c]["X_test"],
                y_test=base_subset[c]["y_test"],
                classes=[c],
            )
            for c in session.pretrain_classes
        ]

        # Mix user sketches with real MNIST samples of the same digit.
        new_subset = load_mnist_subset(
            classes=[label], n_per_class=n_mnist_mix, seed=42,
        )
        sketch_stack = torch.stack(sketches).float()
        sketch_y = torch.full((len(sketches),), label, dtype=torch.long)
        X_train = torch.cat([sketch_stack, new_subset[label]["X_train"]], dim=0)
        y_train = torch.cat([sketch_y, new_subset[label]["y_train"]], dim=0)
        X_test = new_subset[label]["X_test"]
        y_test = new_subset[label]["y_test"]
        new_tasks = [
            TaskData(
                name=f"digit_{label}",
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                classes=[label],
            )
        ]

        out_path = session.live_path.with_suffix(".extended.pt")
        t0 = time.time()
        extend(
            donor_path=session.live_path,
            base_tasks=base_tasks,
            new_tasks=new_tasks,
            out_path=out_path,
            extension_cap_bytes=64_000,
            epochs_per_task=8,
            permanent_int8=False,
        )
        elapsed = time.time() - t0

        # Promote the extended donor to the live path.
        out_path.replace(session.live_path)
        session.organism = None  # force reload on next predict
        if label not in session.learned_classes:
            session.learned_classes.append(label)
        session.n_extends += 1
        session.buffers.pop(label, None)

    return (f"Extended! Trioron grew a new manifold for digit {label} "
            f"in {elapsed:.1f}s. Known classes now: "
            f"{sorted(session.known_classes())}. "
            f"Total extends this session: {session.n_extends}.")
