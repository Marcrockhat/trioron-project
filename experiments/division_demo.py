"""§8-step-5 verification: cellular division on first trigger fire.

Builds on trigger_demo.py — adds the division step plus the §4.1
recipe for new-node initialization:

    1. PCA of residuals: the new node's incoming weight is the principal
       direction of (F_a − F_b) over active contrastive pairs, where F is
       the previous layer's output. This is "the direction the existing
       network is failing to represent" operationalized for our task.
    2. λ_new = 0   (fully plastic; handled by grow_node)
    3. u_new = 0   (neutral; handled by grow_node)
    4. Cross-layer update: not applicable here (we're growing the last
       layer; no downstream layer to extend).
    5. Stabilization: estimate Fisher + anchor BEFORE division, then add
       an EWC penalty term at boosted strength for T_stabilize steps so
       existing nodes resist drift while the new node finds its role.

Falsification target (from the latent-dim sweep calibration):
    Plateau at latent=2 ≈ 0.055.   Plateau at latent=3 ≈ 0.008.
    Expectation: post-division loss should drop substantially toward
    the latent=3 prediction. If it does not, division logic is buggy
    (not architecture).

Outputs:
- outputs/division_demo_log.csv: per-step trace including pre/post
  division and trigger state.
"""
from __future__ import annotations
import csv
import os
import sys
from typing import Optional

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import TrioronNetwork
from trioron.incubator import (
    STATE_DIM,
    ContrastiveCurriculum,
    PAIR_NAMES,
    contrastive_loss,
)
from trioron.triggers import GrowthTrigger, total_gradient_norm
from trioron.ceilings import CeilingsController, REASON_OK


N_STEPS = 12000
BATCH = 32
HIDDEN = 16
LATENT_INIT = 2
MARGIN = 1.0
LR = 3e-3
SEED = 0
LOG_EVERY = 200

TRIGGER_W = 1000
TRIGGER_EPS_LOSS = 0.001
TRIGGER_EPS_RANK = 0.1
TRIGGER_G_MIN = 1e-4
TRIGGER_G_MAX = 10.0

# Stabilization (§4.1.5). The blueprint suggests T_stabilize=200 for
# Orange Pi, 2000 for cloud. We use 400 here — long enough for the new
# node to take on a role, short enough to keep the demo fast.
T_STABILIZE = 400
EWC_STRENGTH_STABILIZE = 50.0   # boosted during stabilization
EWC_STRENGTH_NORMAL = 1.0       # light EWC after stabilization

# Hard ceilings (§4.2 + §7). Orange-Pi-tier values: this demo runs on
# WSL but uses the tighter target so the gate is exercised against the
# constraints the proof-of-concept hardware will enforce. T_div_max=60s
# is generous for a 400-step stabilization on a CPU; we expect ALLOW.
M_MAX_BYTES = 2 * 1024 ** 3      # 2 GB
T_DIV_MAX_SECONDS = 60.0


def make_network(latent: int) -> TrioronNetwork:
    return TrioronNetwork(
        [
            (STATE_DIM, HIDDEN, "relu"),
            (HIDDEN, HIDDEN, "relu"),
            (HIDDEN, latent, "tanh"),
        ]
    )


def combined_contrastive_loss(net, curriculum, batch):
    """Mean of per-pair contrastive losses; also returns the first pair's
    h_a for the trigger's effective-rank computation."""
    total = 0.0
    last_h: Optional[torch.Tensor] = None
    for name in PAIR_NAMES:
        a, b = curriculum.sample_pair(name, batch=batch)
        h_a = net(a)
        h_b = net(b)
        l = contrastive_loss(h_a, h_b, margin=MARGIN)
        total = total + l
        if last_h is None:
            last_h = h_a
    return total / len(PAIR_NAMES), last_h


def compute_growth_direction(net, curriculum, batch=128) -> torch.Tensor:
    """PCA of (F_a − F_b) at the penultimate layer's output, across all 5
    contrastive pairs. Returns a unit vector of shape (HIDDEN,) suitable
    as the new latent node's incoming weight initialization.

    F is the input to the layer being grown — for the last (latent) layer
    that's the second hidden layer's output.
    """
    diffs = []
    with torch.no_grad():
        for name in PAIR_NAMES:
            a, b = curriculum.sample_pair(name, batch=batch)
            f_a, f_b = a, b
            for layer in net.layers[:-1]:
                f_a = layer(f_a)
                f_b = layer(f_b)
            diffs.append(f_a - f_b)
        D = torch.cat(diffs, dim=0)  # (n_pairs * batch, fan_in)
    # Principal right-singular-vector of D = first row of Vh.
    _, _, Vh = torch.linalg.svd(D, full_matrices=False)
    v = Vh[0]
    return v / (v.norm() + 1e-12)


def main() -> int:
    torch.manual_seed(SEED)

    net = make_network(LATENT_INIT)
    cur = ContrastiveCurriculum(seed=SEED)
    opt = optim.Adam(net.parameters(), lr=LR)

    trigger = GrowthTrigger(
        latent_dim=LATENT_INIT,
        window=TRIGGER_W,
        eps_loss=TRIGGER_EPS_LOSS,
        eps_rank=TRIGGER_EPS_RANK,
        g_min=TRIGGER_G_MIN,
        g_max=TRIGGER_G_MAX,
    )

    ceilings = CeilingsController(
        M_max_bytes=M_MAX_BYTES,
        T_div_max_seconds=T_DIV_MAX_SECONDS,
    )

    print("=" * 78)
    print("Trioron — Step 5+7 verification: cellular division with ceilings")
    print("=" * 78)
    print(f"Network:      {net}")
    print(f"Params:       {net.n_parameters()}")
    print(f"Trigger:      {trigger}")
    print(f"Ceilings:     {ceilings}")
    print(f"T_stabilize:  {T_STABILIZE}    EWC boost: {EWC_STRENGTH_STABILIZE}")
    print()

    log_rows: list[list] = []
    division_step: int = -1
    ewc_strength: float = 0.0     # zero pre-division (no anchor yet)
    stabilize_remaining: int = 0
    pre_division_loss_window: list[float] = []
    post_division_loss_window: list[float] = []
    end_of_run_loss_window: list[float] = []

    for step in range(N_STEPS):
        loss, h_a = combined_contrastive_loss(net, cur, BATCH)
        if ewc_strength > 0.0:
            loss = loss + ewc_strength * net.ewc_penalty()

        opt.zero_grad()
        loss.backward()
        gnorm = total_gradient_norm(net.parameters())
        opt.step()

        s = trigger.observe(loss=loss.item(), hidden=h_a.detach(), grad_norm=gnorm)

        # Track loss windows for the report (using the raw contrastive loss
        # without the EWC term, so pre/post comparisons are like-for-like).
        # We can recover it as loss - ewc_strength * ewc_penalty, but the
        # ewc term tends to be small; record loss.item() as a proxy.
        if division_step < 0 and step >= 800:
            pre_division_loss_window.append(loss.item())
        if 0 < division_step <= step < division_step + 1000:
            post_division_loss_window.append(loss.item())
        if step >= N_STEPS - 1000:
            end_of_run_loss_window.append(loss.item())

        # ---- Division on first fire ----
        # The `not ceilings.arrested` guard means once a preflight has
        # denied (memory or time), this whole branch never re-enters —
        # the trigger may keep firing but we do not retry.
        if s.fire and division_step < 0 and not ceilings.arrested:
            target_layer_idx = len(net.layers) - 1
            decision = ceilings.preflight(net, target_layer_idx)
            print()
            print(f"  *** FIRE at step {step}. Preflight: {decision}")
            if not decision.allowed:
                # §4.2 arrest. Don't grow; training continues with
                # plasticity-only updates. Logging falls through normally.
                print(f"      DIVISION DENIED — reason={decision.reason}. "
                      f"Network is mature; continuing with weights-only updates.")
            else:
                print(f"  *** ALLOWED. Initiating cellular division. ***")
                print(f"      pre-fire loss (last 1000 steps): "
                      f"mean={sum(pre_division_loss_window[-1000:]) / max(1, len(pre_division_loss_window[-1000:])):.4f}")
                print(f"      effective rank: {s.effective_rank:.3f} / {trigger.latent_dim}")
                print(f"      grad norm:      {s.grad_norm:.3f}")
                division_step = step

                # 1. Estimate Fisher + anchor BEFORE division (so existing nodes
                #    have non-zero λ and a snapshot to be pulled toward).
                def _calibration_batches(n=20):
                    for _ in range(n):
                        # Use full curriculum loss as the "task" for Fisher.
                        a_list, b_list = [], []
                        for name in PAIR_NAMES:
                            a, b = cur.sample_pair(name, batch=BATCH)
                            a_list.append(a)
                            b_list.append(b)
                        yield torch.cat(a_list, dim=0), torch.cat(b_list, dim=0)

                def _fisher_loss(pred_concat, _y_unused):
                    # pred_concat is the concatenated A-side latents from one
                    # batch; for Fisher estimation we want the magnitude of
                    # gradients under the curriculum loss. Re-run the per-pair
                    # contrastive losses against fresh B samples.
                    total = 0.0
                    offset = 0
                    for name in PAIR_NAMES:
                        h_a_i = pred_concat[offset:offset + BATCH]
                        _, b_i = cur.sample_pair(name, batch=BATCH)
                        h_b_i = net(b_i)
                        total = total + contrastive_loss(h_a_i, h_b_i, margin=MARGIN)
                        offset += BATCH
                    return total / len(PAIR_NAMES)

                net.estimate_fisher(_calibration_batches(20), _fisher_loss, n_batches=20)
                net.update_lambda_all()
                net.anchor_all()
                print(f"      Fisher/λ refreshed; anchor set. "
                      f"mean λ per layer: "
                      f"{[round(layer.lam.mean().item(), 5) for layer in net.layers]}")

                # 2. PCA of residuals → init vec for new node.
                v = compute_growth_direction(net, cur, batch=128)
                print(f"      growth direction ‖v‖={v.norm().item():.4f}  "
                      f"first 5 components: {v[:5].tolist()}")

                # 3. Cellular division.
                new_idx = net.grow_layer(layer_idx=len(net.layers) - 1, init_vec=v)
                print(f"      new node added at index {new_idx}; "
                      f"latent dim now {net.layers[-1].n_nodes}")

                # 4. Trigger reset + bookkeeping.
                trigger.set_latent_dim(net.layers[-1].n_nodes)
                trigger.reset()

                # 5. Optimizer rebuild — required after structural change.
                opt = optim.Adam(net.parameters(), lr=LR)

                # 6. Begin stabilization phase + start the ceilings clock so
                #    a future preflight can veto on T_div_max if we slip.
                ewc_strength = EWC_STRENGTH_STABILIZE
                stabilize_remaining = T_STABILIZE
                ceilings.mark_stabilization_start()
                print(f"      stabilization phase: {T_STABILIZE} steps "
                      f"with ewc_strength={EWC_STRENGTH_STABILIZE}")
                print()

        # Decay stabilization.
        if stabilize_remaining > 0:
            stabilize_remaining -= 1
            if stabilize_remaining == 0:
                ewc_strength = EWC_STRENGTH_NORMAL
                stab_seconds = ceilings.mark_stabilization_end()
                print(f"  step {step}: stabilization phase ended; "
                      f"ewc_strength → {EWC_STRENGTH_NORMAL}; "
                      f"wall-clock {stab_seconds:.2f}s "
                      f"(budget {ceilings.T_div_max_seconds:.0f}s)")

        log_rows.append([
            step, loss.item(), s.effective_rank, s.grad_norm,
            int(s.loss_plateau), int(s.rank_saturated),
            int(s.grad_stable), int(s.fire),
            int(division_step >= 0),
            ewc_strength,
            net.layers[-1].n_nodes,
        ])

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            phase = "pre-fire" if division_step < 0 else (
                "stabilize" if stabilize_remaining > 0 else "post"
            )
            print(
                f"  step {step:5d} [{phase:9s}] loss {loss.item():.4f}  "
                f"rank {s.effective_rank:.3f}/{trigger.latent_dim}  "
                f"L={int(s.loss_plateau)} R={int(s.rank_saturated)} "
                f"G={int(s.grad_stable)} fire={int(s.fire)}"
                + ("  [warmup]" if s.warmup else "")
            )

    # ----- Summary -----
    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    pre_mean = _mean(pre_division_loss_window[-1000:]) if pre_division_loss_window else float("nan")
    post_mean = _mean(post_division_loss_window) if post_division_loss_window else float("nan")
    end_mean = _mean(end_of_run_loss_window)

    print()
    print("=" * 78)
    print("Division summary")
    print("=" * 78)
    print(f"  Division step:                 {division_step}")
    print(f"  Latent dim before/after:       {LATENT_INIT} → {net.layers[-1].n_nodes}")
    print(f"  Pre-division loss (last 1000): {pre_mean:.4f}")
    print(f"  Post-division loss (next 1000): {post_mean:.4f}")
    print(f"  End-of-run loss (last 1000):   {end_mean:.4f}")
    print(f"  Ceilings:                      {ceilings}")
    if ceilings.last_stab_seconds is not None:
        print(f"  Stabilization wall-clock:      "
              f"{ceilings.last_stab_seconds:.2f}s / "
              f"{ceilings.T_div_max_seconds:.0f}s budget")
    print()
    print("  Calibration sweep predictions (from outputs/capacity_sweep.csv):")
    print("    latent=2 plateau ≈ 0.055      latent=3 plateau ≈ 0.008")
    print()

    # Pass criterion: end-of-run loss should be substantially below the
    # latent=2 calibration plateau, heading toward the latent=3 plateau.
    if division_step < 0:
        print("  FAIL: trigger never fired within run — division was not exercised.")
        rc = 1
    elif end_mean < 0.025:    # halfway between the two calibration plateaus
        print(
            f"  PASS: end-of-run loss {end_mean:.4f} is substantially below "
            f"the latent=2 plateau (0.055) — division added real capacity."
        )
        rc = 0
    else:
        print(
            f"  PARTIAL: end-of-run loss {end_mean:.4f} did not drop below "
            "the halfway threshold (0.025). Division mechanism is wired but "
            "the new node may not be finding its role; investigate growth "
            "direction or stabilization parameters."
        )
        rc = 1

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "division_demo_log.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "step", "loss", "effective_rank", "grad_norm",
            "loss_plateau", "rank_saturated", "grad_stable", "fire",
            "post_division", "ewc_strength", "latent_dim",
        ])
        w.writerows(log_rows)
    print(f"  log: {csv_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
