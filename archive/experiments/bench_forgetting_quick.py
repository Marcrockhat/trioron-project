"""Fast forgetting-curve probe for the lifetime-deployment question.

Use case: estimate trioron's degradation rate over N tasks fast enough
to iterate on (~2 min for N=50). Sits between test_continual_learning_smoke
(2 tasks, seconds, worst-case orthogonal geometry) and bench_chained_15task
(15 tasks, ~hour). Covers the cheapest two retention mechanisms — EWC +
optional cellular growth — and skips manifold/dream so it stays fast.

Output per task t:
    train task t  →  consolidate  →  measure accuracy on every prior task
    record retention[t, j] for j ≤ t
Then fit a stretched-exponential curve r(t) = c + (1-c)·exp(-(t/τ)^β)
to the mean retention and extrapolate.

Usage:
    python3 experiments/bench_forgetting_quick.py
    python3 experiments/bench_forgetting_quick.py --n_tasks 100 --grow
    python3 experiments/bench_forgetting_quick.py --seeds 3 --extrapolate 1000

Tasks are permuted MNIST: each task is MNIST with a different fixed pixel
permutation. Permuted MNIST is the standard continual-learning probe for
exactly this question (Goodfellow et al. 2013) — orthogonal-enough to
stress forgetting, structured-enough to be meaningful.
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trioron.network import TrioronNetwork
from trioron.node import _EwcZeroWarning

# Task 1 trains with all-zero λ (consolidation cycle hasn't run yet), so
# the silent-zero RuntimeWarning fires once and would dump a wall of text
# into our progress log. We pre-disarm it for this probe — the bench's
# whole point is to drive forgetting and we know the first call is zero.
_EwcZeroWarning._warned = True


# ---- data ----------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "data"


def _load_mnist(train_per_task: int, eval_per_task: int):
    """Load MNIST once; return (X_train, y_train, X_eval, y_eval) flat
    tensors. We subsample to keep the per-task budget tiny.
    """
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(
        DATA_ROOT, train=True, download=False, transform=tfm,
    )
    test_ds = datasets.MNIST(
        DATA_ROOT, train=False, download=False, transform=tfm,
    )
    # Materialize as flat tensors once.
    X_train = train_ds.data.float().div(255.0).reshape(-1, 784)
    y_train = train_ds.targets.long()
    X_eval = test_ds.data.float().div(255.0).reshape(-1, 784)
    y_eval = test_ds.targets.long()
    # Subsample to the per-task budget — we'll re-use the same physical
    # rows across tasks with a different permutation, so a single subset
    # is enough.
    g = torch.Generator().manual_seed(0)
    train_idx = torch.randperm(X_train.size(0), generator=g)[:train_per_task]
    eval_idx = torch.randperm(X_eval.size(0), generator=g)[:eval_per_task]
    return (
        X_train[train_idx], y_train[train_idx],
        X_eval[eval_idx], y_eval[eval_idx],
    )


def _make_permutations(n_tasks: int, seed: int) -> torch.Tensor:
    """Per-task pixel permutation. Shape (n_tasks, 784)."""
    g = torch.Generator().manual_seed(seed)
    perms = torch.stack([torch.randperm(784, generator=g) for _ in range(n_tasks)])
    return perms


# ---- replay buffer (simple ring) -----------------------------------------

class _RingReplayBuffer:
    """Stores K already-permuted samples per task. During new-task
    training, each batch is augmented with a uniformly-sampled replay
    batch drawn across ALL stored tasks.

    Not trioron-native (no L0 frozen layer, no per-class Gaussian) — that
    would need the full ManifoldBuffer + forward_from_layer stack. This
    is a stand-in to prove the replay leg of the retention story; expect
    the resulting asymptote to land between bare-EWC (this script with
    samples_per_task=0) and production manifold replay.
    """

    def __init__(self, samples_per_task: int):
        self.samples_per_task = samples_per_task
        self.Xs: List[torch.Tensor] = []
        self.ys: List[torch.Tensor] = []

    def add_task(self, X_task: torch.Tensor, y_task: torch.Tensor) -> None:
        if self.samples_per_task <= 0:
            return
        k = min(self.samples_per_task, X_task.size(0))
        # Take a random K rather than first K — first K can bias toward
        # whatever ordering the task's data subset happens to have.
        idx = torch.randperm(X_task.size(0))[:k]
        self.Xs.append(X_task[idx].clone())
        self.ys.append(y_task[idx].clone())

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.Xs:
            return None, None  # type: ignore
        X_all = torch.cat(self.Xs)
        y_all = torch.cat(self.ys)
        bi = torch.randperm(X_all.size(0))[:min(n, X_all.size(0))]
        return X_all[bi], y_all[bi]

    def total_stored(self) -> int:
        return sum(int(x.size(0)) for x in self.Xs)


# ---- one run -------------------------------------------------------------

def _eval_acc(net: TrioronNetwork, X: torch.Tensor, y: torch.Tensor) -> float:
    net.eval()
    with torch.no_grad():
        logits = net(X)
        pred = logits.argmax(dim=1)
        return (pred == y).float().mean().item()


def run_one_seed(
    *,
    n_tasks: int,
    train_per_task: int,
    eval_per_task: int,
    epochs_per_task: int,
    batch_size: int,
    ewc_strength: float,
    grow_per_task: int,
    replay_per_task: int,
    seed: int,
    verbose: bool = False,
) -> torch.Tensor:
    """Returns retention matrix R of shape (n_tasks, n_tasks); R[t, j] is
    the accuracy on task j AFTER training through task t (upper-triangular
    entries are NaN since the model hasn't seen task j yet)."""
    torch.manual_seed(seed)

    X_train, y_train, X_eval, y_eval = _load_mnist(train_per_task, eval_per_task)
    perms = _make_permutations(n_tasks, seed)

    # Substrate: 784 → 64 → 10. Cellular division on layer 0 if --grow.
    net = TrioronNetwork([(784, 64, "relu"), (64, 10, "linear")])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    replay = _RingReplayBuffer(samples_per_task=replay_per_task)

    R = torch.full((n_tasks, n_tasks), float("nan"))

    for t in range(n_tasks):
        # Permute training data for task t.
        Xt = X_train[:, perms[t]]
        yt = y_train

        # Train one or more epochs, mixing replay samples into each batch.
        net.train()
        for _ in range(epochs_per_task):
            idx = torch.randperm(Xt.size(0))
            for start in range(0, Xt.size(0), batch_size):
                bi = idx[start:start + batch_size]
                Xb = Xt[bi]
                yb = yt[bi]
                Xr, yr = replay.sample(batch_size)
                if Xr is not None:
                    Xb = torch.cat([Xb, Xr])
                    yb = torch.cat([yb, yr])

                opt.zero_grad()
                logits = net(Xb)
                loss = F.cross_entropy(logits, yb)
                loss = loss + ewc_strength * net.ewc_penalty()
                loss.backward()
                net.update_fisher_all()
                opt.step()

        # End-of-task consolidation: estimate Fisher fresh at converged
        # weights, refresh λ (with mean=1 rescale so ewc_strength stays
        # optimizer-independent), anchor.
        def make_batches():
            for _ in range(8):
                bi = torch.randperm(Xt.size(0))[:batch_size]
                yield Xt[bi], yt[bi]

        net.populate_lambda(
            batches=make_batches(),
            loss_fn=lambda p, y: F.cross_entropy(p, y),
            n_batches=8,
            rescale_mean=True,
        )

        # Optional cellular division: add nodes to layer 0 and rebuild
        # the optimizer (Parameter objects are replaced).
        if grow_per_task > 0:
            for _ in range(grow_per_task):
                net.grow_layer(0)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        # Drop K real samples from this task into the replay buffer so
        # the next task's training can rehearse against them.
        replay.add_task(Xt, yt)

        # Measure retention on every task seen so far.
        for j in range(t + 1):
            Xe = X_eval[:, perms[j]]
            R[t, j] = _eval_acc(net, Xe, y_eval)

        if verbose:
            mean_ret = R[t, :t + 1].mean().item()
            print(
                f"  task {t + 1:3d}/{n_tasks}  current={R[t, t].item():.3f}  "
                f"mean-retained={mean_ret:.3f}  n_nodes={net.layers[0].n_nodes}  "
                f"replay_pool={replay.total_stored()}"
            )

    return R


# ---- summarize -----------------------------------------------------------

def summarize(R: torch.Tensor) -> dict:
    """Per-task-position mean retention (averaged across seeds if a 3-D
    tensor is passed: shape (seeds, n_tasks, n_tasks))."""
    if R.dim() == 2:
        R = R.unsqueeze(0)
    n_seeds, n_tasks, _ = R.shape

    # mean_retention_after[t] = mean over j ≤ t of R[t, j], averaged over seeds
    mean_retention = torch.zeros(n_tasks)
    for t in range(n_tasks):
        vals = R[:, t, :t + 1]  # (seeds, t+1)
        mean_retention[t] = vals.mean().item()

    # last-task forgetting curve: how much does task 0's accuracy decay
    # as a function of how many later tasks have arrived?
    task0_curve = R[:, :, 0].mean(dim=0)  # (n_tasks,)

    return {
        "mean_retention_after_t": mean_retention,
        "task0_curve": task0_curve,
        "final_mean_retention": float(mean_retention[-1]),
        "final_task0_acc": float(task0_curve[-1]),
        "n_tasks": n_tasks,
        "n_seeds": n_seeds,
    }


def fit_stretched_exp(curve: torch.Tensor, extrapolate_to: int = 0):
    """Fit r(t) = c + (1-c)·exp(-(t/τ)^β). Returns (c, tau, beta) plus
    extrapolated curve out to `extrapolate_to` if > 0.

    Closed-form fit isn't available — use a simple grid + LM refinement
    via scipy if present, else fall back to grid search alone (no scipy
    dependency required).
    """
    t = torch.arange(1, curve.numel() + 1, dtype=torch.float64)
    y = curve.to(torch.float64).clamp_min(1e-6)

    # Grid search over (c, log10(tau), beta). Then we just return the best
    # grid point — for the lifetime-extrapolation use case the grid is
    # tight enough.
    cs = torch.linspace(0.0, max(0.05, float(y.min()) - 0.02), 12)
    taus = torch.logspace(0.0, 4.0, 25)
    betas = torch.linspace(0.3, 2.0, 18)
    best = (float("inf"), 0.0, 0.0, 0.0)
    for c in cs:
        ceil = 1.0 - float(c)
        for tau in taus:
            for beta in betas:
                pred = float(c) + ceil * torch.exp(-((t / tau) ** beta))
                err = ((pred - y) ** 2).sum().item()
                if err < best[0]:
                    best = (err, float(c), float(tau), float(beta))
    _, c, tau, beta = best

    out = {"c": c, "tau": tau, "beta": beta, "rss": best[0]}
    if extrapolate_to > 0:
        tx = torch.arange(1, extrapolate_to + 1, dtype=torch.float64)
        out["extrapolated_curve"] = (
            c + (1.0 - c) * torch.exp(-((tx / tau) ** beta))
        ).to(torch.float32)
    return out


# ---- CLI -----------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_tasks", type=int, default=50)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--train_per_task", type=int, default=20000)
    p.add_argument("--eval_per_task", type=int, default=1000)
    p.add_argument("--epochs_per_task", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--ewc_strength", type=float, default=1.0)
    p.add_argument("--grow", action="store_true",
                   help="Cellular division: add nodes to layer 0 after each task")
    p.add_argument("--grow_per_task", type=int, default=2)
    p.add_argument("--replay_per_task", type=int, default=200,
                   help="Real samples per task kept in the ring buffer for replay; "
                        "0 disables replay (bare-EWC floor).")
    p.add_argument("--extrapolate", type=int, default=0,
                   help="Extrapolate fitted retention to this task count (0 = off)")
    p.add_argument("--out_dir", type=str,
                   default=str(Path(__file__).resolve().parent.parent / "outputs"))
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grow_per_task = args.grow_per_task if args.grow else 0

    print(
        f"forgetting probe: n_tasks={args.n_tasks} seeds={args.seeds} "
        f"grow={'on' if grow_per_task else 'off'} "
        f"ewc_strength={args.ewc_strength} "
        f"replay_per_task={args.replay_per_task} "
        f"train_per_task={args.train_per_task}"
    )

    t0 = time.time()
    Rs = []
    for s in range(args.seeds):
        seed = 42 + s
        print(f"\n--- seed {s + 1}/{args.seeds} (torch_seed={seed}) ---")
        R = run_one_seed(
            n_tasks=args.n_tasks,
            train_per_task=args.train_per_task,
            eval_per_task=args.eval_per_task,
            epochs_per_task=args.epochs_per_task,
            batch_size=args.batch_size,
            ewc_strength=args.ewc_strength,
            grow_per_task=grow_per_task,
            replay_per_task=args.replay_per_task,
            seed=seed,
            verbose=True,
        )
        Rs.append(R)
    elapsed = time.time() - t0
    Rs = torch.stack(Rs)  # (seeds, n_tasks, n_tasks)

    s = summarize(Rs)
    print(
        f"\nelapsed: {elapsed:.1f}s ({elapsed / args.n_tasks:.2f}s/task)"
    )
    print(
        f"final mean retention across all tasks seen: {s['final_mean_retention']:.3f}"
    )
    print(
        f"task-0 accuracy after {args.n_tasks} tasks:         {s['final_task0_acc']:.3f}"
    )

    # Write retention matrix + per-position curve to CSV.
    csv_path = out_dir / "forgetting_quick.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_position", "mean_retention", "task0_accuracy"])
        for t in range(args.n_tasks):
            w.writerow([
                t + 1,
                f"{s['mean_retention_after_t'][t].item():.4f}",
                f"{s['task0_curve'][t].item():.4f}",
            ])
    print(f"wrote {csv_path}")

    if args.extrapolate > 0:
        fit = fit_stretched_exp(s["mean_retention_after_t"], args.extrapolate)
        print(
            f"\nfit: r(t) = {fit['c']:.3f} + "
            f"{1 - fit['c']:.3f} * exp(-(t/{fit['tau']:.1f})^{fit['beta']:.2f})  "
            f"(rss={fit['rss']:.4f})"
        )
        ext = fit["extrapolated_curve"]
        # Print extrapolated retention at a few landmark task counts.
        for landmark in [50, 100, 250, 500, 1000, args.extrapolate]:
            if landmark <= args.extrapolate:
                print(
                    f"  extrapolated retention at t={landmark:5d}: "
                    f"{ext[landmark - 1].item():.3f}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
