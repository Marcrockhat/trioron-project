"""Step 2 verification — multi-layer continual learning with EWC.

Wires three TrioronLayers into a network, trains on synthetic Task A,
estimates Fisher and anchors, then trains on Task B both WITH and
WITHOUT EWC, and compares retention on Task A.

Where test_node.py's continual_learning_smoke verified single-layer EWC,
this script verifies the machinery survives composition into a real
(small) feedforward graph — the §8 step 2 gate from the blueprint.

Run with:    python3 experiments/continual_2task.py
"""
from __future__ import annotations
import sys
import os

# Make the trioron package importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from trioron.network import TrioronNetwork


# --------------------------------------------------------------------------- #
# Synthetic tasks                                                             #
# --------------------------------------------------------------------------- #

def make_synthetic_task(seed: int, n_samples: int = 256, in_dim: int = 8, out_dim: int = 4):
    """A nonlinear regression task: y = tanh(W2 @ ReLU(W1 @ x + b1) + b2).

    Different seeds → different ground-truth functions of the same input
    distribution. Two tasks with different seeds will have largely
    incompatible weight solutions, which is what we want for a real
    catastrophic-forgetting test.
    """
    g = torch.Generator().manual_seed(seed)
    W1 = torch.randn(16, in_dim, generator=g)
    b1 = torch.randn(16, generator=g) * 0.1
    W2 = torch.randn(out_dim, 16, generator=g)
    b2 = torch.randn(out_dim, generator=g) * 0.1
    X = torch.randn(n_samples, in_dim, generator=g)
    H = torch.relu(X @ W1.T + b1)
    Y = torch.tanh(H @ W2.T + b2)
    return X, Y


def make_batches(X, Y, batch_size: int = 32):
    """Generator yielding (x_batch, y_batch) tuples forever."""
    n = X.shape[0]
    while True:
        idx = torch.randperm(n)[:batch_size]
        yield X[idx], Y[idx]


# --------------------------------------------------------------------------- #
# Training helpers                                                            #
# --------------------------------------------------------------------------- #

def train(net, X, Y, n_steps=400, lr=0.05, batch_size=64, ewc_strength=0.0,
          verbose=False, log_every=100):
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    n = X.shape[0]
    losses = []
    for step in range(n_steps):
        idx = torch.randperm(n)[:batch_size]
        xb, yb = X[idx], Y[idx]
        opt.zero_grad()
        pred = net(xb)
        l_task = (pred - yb).pow(2).mean()
        if ewc_strength > 0:
            l = l_task + ewc_strength * net.ewc_penalty()
        else:
            l = l_task
        l.backward()
        opt.step()
        losses.append(l_task.item())
        if verbose and step % log_every == 0:
            print(f"    step {step:4d}: task_loss {l_task.item():.4f}")
    return losses


def evaluate(net, X, Y):
    with torch.no_grad():
        pred = net(X)
        return (pred - Y).pow(2).mean().item()


# --------------------------------------------------------------------------- #
# Main experiment                                                             #
# --------------------------------------------------------------------------- #

def main():
    torch.manual_seed(0)

    print("=" * 64)
    print("Trioron — Step 2 verification: multi-layer continual learning")
    print("=" * 64)

    # ---- Build network ----
    net = TrioronNetwork(
        [
            (8, 16, "relu"),
            (16, 16, "relu"),
            (16, 4, "linear"),
        ]
    )
    print(f"Network: {net}")
    print(f"Parameters: {net.n_parameters()}")
    print()

    # ---- Generate two tasks ----
    X_a, Y_a = make_synthetic_task(seed=1)
    X_b, Y_b = make_synthetic_task(seed=2)
    print(f"Task A: X {tuple(X_a.shape)} → Y {tuple(Y_a.shape)}")
    print(f"Task B: X {tuple(X_b.shape)} → Y {tuple(Y_b.shape)}")

    # ---- Train task A ----
    print()
    print("--- Training on Task A ---")
    train(net, X_a, Y_a, n_steps=400, verbose=True)
    loss_a_after_a = evaluate(net, X_a, Y_a)
    print(f"Final task-A loss: {loss_a_after_a:.4f}")

    # ---- Estimate Fisher on task A (clean post-convergence pass) ----
    print()
    print("--- Estimating Fisher on Task A (post-convergence) ---")
    net.estimate_fisher(
        make_batches(X_a, Y_a, batch_size=32),
        loss_fn=lambda p, y: (p - y).pow(2).mean(),
        n_batches=30,
    )
    net.update_lambda_all()
    net.anchor_all()
    lambdas = [layer.lam.mean().item() for layer in net.layers]
    print(f"Mean λ per layer: {[f'{x:.6f}' for x in lambdas]}")

    # ---- Snapshot for two parallel B-training runs ----
    snapshot = {k: v.clone() for k, v in net.state_dict().items()}

    # ---- Train task B WITHOUT EWC (control) ----
    print()
    print("--- Training on Task B (control: NO EWC) ---")
    net_no_ewc = TrioronNetwork(
        [
            (8, 16, "relu"),
            (16, 16, "relu"),
            (16, 4, "linear"),
        ]
    )
    net_no_ewc.load_state_dict(snapshot)
    for layer in net_no_ewc.layers:
        layer.lam.zero_()  # disable EWC by zeroing lambda
    train(net_no_ewc, X_b, Y_b, n_steps=400)
    loss_a_no_ewc = evaluate(net_no_ewc, X_a, Y_a)
    loss_b_no_ewc = evaluate(net_no_ewc, X_b, Y_b)

    # ---- Train task B WITH EWC ----
    print("--- Training on Task B (experimental: WITH EWC) ---")
    net.load_state_dict(snapshot)
    EWC_STRENGTH = 2000.0
    train(net, X_b, Y_b, n_steps=400, ewc_strength=EWC_STRENGTH)
    loss_a_with_ewc = evaluate(net, X_a, Y_a)
    loss_b_with_ewc = evaluate(net, X_b, Y_b)

    # ---- Report ----
    print()
    print("=" * 64)
    print("Results")
    print("=" * 64)
    print(f"  Task-A loss after training A only:        {loss_a_after_a:.4f}")
    print(f"  Task-A loss after B (NO EWC):             {loss_a_no_ewc:.4f}    [control]")
    print(f"  Task-A loss after B (WITH EWC={int(EWC_STRENGTH)}):    "
          f"{loss_a_with_ewc:.4f}    [experimental]")
    print()
    print(f"  Task-B loss after B (NO EWC):             {loss_b_no_ewc:.4f}")
    print(f"  Task-B loss after B (WITH EWC):           {loss_b_with_ewc:.4f}")
    print()

    if loss_a_with_ewc < loss_a_no_ewc:
        ratio = loss_a_no_ewc / max(loss_a_with_ewc, 1e-9)
        print(f"  PASS: EWC reduces task-A forgetting by {ratio:.2f}x")
        return 0
    else:
        print(f"  FAIL: EWC did NOT reduce task-A forgetting")
        print(f"        no_ewc={loss_a_no_ewc:.4f} vs with_ewc={loss_a_with_ewc:.4f}")
        print()
        print("  Likely causes:")
        print("    - ewc_strength too low for current Fisher magnitudes")
        print("    - Tasks A and B too similar (low pressure to forget)")
        print("    - Fisher estimation pass too short")
        return 1


if __name__ == "__main__":
    sys.exit(main())
