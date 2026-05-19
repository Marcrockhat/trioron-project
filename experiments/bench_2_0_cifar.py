"""Trioron 2.0 — depth-vs-width on CIFAR-100 classical sensors.

Follow-up to bench_2_0_single_task (synthetic XOR, depth didn't win)
and bench_2_0_regime (continual XOR, no 2.0 arm dominated grow_node).

Hypothesis: CIFAR-100 has compositional visual structure that
genuinely favors depth — synthetic XOR's 4-bit feature interactions
don't capture the kind of compositionality (texture → object part →
object) that real images have. If depth ever wins, it should win
here.

Setup
-----
- Load CIFAR-100 (50000 train / 10000 test)
- Apply `classical` sense (33-d, no cortex/CNN — pure trioron scope
  per the project's feedback_pure_trioron_scope memory)
- Train a TrioronNetwork from scratch (no growth, no CL machinery,
  no consolidation — this is a pure depth-vs-width capacity test)
- Compare arms at matched parameter count

Arms (all at ~8676 params)
-----
  shallow_wide       (33, 64, 100)        ~8676 params
  matched_deep       (33, 47, 47, 100)    ~8654 params (≈shallow)
  deep_narrow_small  (33, 32, 32, 100)    ~5444 params (smaller for ref)
  triple_deep_small  (33, 32, 32, 32, 100) ~6468 params (3 hidden, ref)

Cache: classical-sense features are cached at
       outputs/cifar100_classical_train.pt and ..._test.pt.
       Re-runs reuse them.

3 seeds, ~5 min.
"""

from __future__ import annotations
import csv
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trioron.network import TrioronNetwork
from trioron.node import _EwcZeroWarning
from trioron.senses import apply_sense, Standardizer
from experiments.cifar.datasets import load_cifar100

_EwcZeroWarning._warned = True


N_CLASSES = 100
BATCH = 512
N_STEPS = 5000
LR = 1e-3
SEEDS = (0, 1, 2)
ARMS = (
    ("shallow_wide", (33, 64, N_CLASSES)),
    ("matched_deep", (33, 47, 47, N_CLASSES)),
    ("deep_narrow_small", (33, 32, 32, N_CLASSES)),
    ("triple_deep_small", (33, 32, 32, 32, N_CLASSES)),
)
CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs"


def load_or_cache_features():
    """Load CIFAR-100, apply classical sense, fit a standardizer on
    train, return standardized (x_train, y_train, x_test, y_test).
    Cache to disk so subsequent runs skip the sense pass."""
    train_cache = CACHE_DIR / "cifar100_classical_train.pt"
    test_cache = CACHE_DIR / "cifar100_classical_test.pt"
    if train_cache.exists() and test_cache.exists():
        tr = torch.load(train_cache)
        te = torch.load(test_cache)
        return tr["x"], tr["y"], te["x"], te["y"]

    print("computing classical-sense features (one-time, ~30s) ...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    img_tr, y_tr = load_cifar100(train=True, download=True)
    img_te, y_te = load_cifar100(train=False, download=True)
    x_tr = apply_sense("classical", img_tr)
    x_te = apply_sense("classical", img_te)
    print(f"  sense pass: {time.time() - t0:.1f}s")

    # Fit standardizer on train, apply to both.
    std = Standardizer.fit(x_tr)
    x_tr = std.transform(x_tr)
    x_te = std.transform(x_te)
    torch.save({"x": x_tr, "y": y_tr}, train_cache)
    torch.save({"x": x_te, "y": y_te}, test_cache)
    return x_tr, y_tr, x_te, y_te


def build_net(spec):
    layers = []
    for i in range(len(spec) - 1):
        act = "relu" if i < len(spec) - 2 else "linear"
        layers.append((spec[i], spec[i + 1], act))
    return TrioronNetwork(layers)


def train_one(net, x_tr, y_tr, x_te, y_te):
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = x_tr.shape[0]
    for step in range(N_STEPS):
        idx = torch.randint(0, n, (BATCH,))
        xb = x_tr[idx]
        yb = y_tr[idx]
        opt.zero_grad()
        loss = F.cross_entropy(net(xb), yb)
        loss.backward()
        opt.step()
    with torch.no_grad():
        # Test in chunks to avoid memory issues on the full 10k set.
        correct = 0
        total = x_te.shape[0]
        for i in range(0, total, 2048):
            pred = net(x_te[i:i + 2048]).argmax(dim=1)
            correct += int((pred == y_te[i:i + 2048]).sum().item())
        return correct / total


def run_cell(seed, arm_name, spec, x_tr, y_tr, x_te, y_te):
    torch.manual_seed(seed)
    net = build_net(spec)
    acc = train_one(net, x_tr, y_tr, x_te, y_te)
    return {
        "seed": seed, "arm": arm_name, "arch": spec,
        "test_acc": acc, "n_params": net.n_parameters(),
    }


def main():
    out_path = CACHE_DIR / "bench_2_0_cifar.csv"
    x_tr, y_tr, x_te, y_te = load_or_cache_features()
    print(
        f"CIFAR-100 classical features ready: "
        f"train {tuple(x_tr.shape)}, test {tuple(x_te.shape)}"
    )

    rows = []
    t0 = time.time()
    for arm_name, spec in ARMS:
        for seed in SEEDS:
            r = run_cell(seed, arm_name, spec, x_tr, y_tr, x_te, y_te)
            rows.append(r)
            print(
                f"  {arm_name:>20} seed={seed}  "
                f"acc={r['test_acc']:.4f}  "
                f"params={r['n_params']}  arch={r['arch']}"
            )
    elapsed = time.time() - t0
    print(f"\nelapsed: {elapsed:.1f}s")

    print("\n--- per-arm summary (n={}) ---".format(len(SEEDS)))
    print(f"  {'arm':>20}  {'test_acc':>12}  {'params':>8}  arch")
    for arm_name, spec in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm_name]
        m = statistics.mean(r["test_acc"] for r in arm_rows)
        s = statistics.stdev(r["test_acc"] for r in arm_rows)
        params = arm_rows[0]["n_params"]
        print(
            f"  {arm_name:>20}  {m:.4f}±{s:.4f}  {params:>8}  {spec}"
        )

    print("\n--- depth-vs-width (matched params, ~8676) ---")
    shallow = [r for r in rows if r["arm"] == "shallow_wide"]
    deep = [r for r in rows if r["arm"] == "matched_deep"]
    sm = statistics.mean(r["test_acc"] for r in shallow)
    dm = statistics.mean(r["test_acc"] for r in deep)
    print(f"  shallow_wide  mean acc: {sm:.4f}  ({shallow[0]['n_params']} params)")
    print(f"  matched_deep  mean acc: {dm:.4f}  ({deep[0]['n_params']} params)")
    print(f"  Δ(deep - shallow) = {dm - sm:+.4f}")
    if dm > sm:
        print("  → depth wins on CIFAR-100 classical at matched params")
    else:
        print("  → width still wins; depth doesn't add value at this scale")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "arm", "arch", "test_acc", "n_params"],
        )
        writer.writeheader()
        for r in rows:
            r2 = {**r, "arch": "|".join(str(n) for n in r["arch"])}
            writer.writerow(r2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
