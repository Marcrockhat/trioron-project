"""Mirror cells — apprenticing as a real cell type (2026-06-02).

Until now "apprenticing" (learning from a competent master) was a *function-level
proxy*: a cross-entropy term bolted onto the loss (teacher_student.py). The
roadmap (docs/design/organism_roadmap.md) asks for the real thing — "a new cell
type (like satellites): active in BOTH own-action and observed-action". This
module wires that.

THE MECHANISM (localized-loss design; chosen 2026-06-02):

  observation channel ─┐
   (master's action,   ├─► [MIRROR cells] ─► output (policy/Q)
    one-hot, 6 dims)    │        ▲
  efference copy of ────┘        │
   own action (interior)         imitation CE credit gated HERE ONLY

  1. OBSERVATION CHANNEL — N_ACTION extra perception inputs carrying the master's
     last action one-hot. Zero when the organism acts solo (it sees no one).
  2. MIRROR CELLS — a new cell type (epigenome bit MIRROR). They read from BOTH
     the observation channel (fires on OBSERVED action) AND the interior cells
     (efference copy → fires on OWN action). Same cell, self-or-other: the literal
     mirror-neuron property. They project into the output, so observing reshapes
     the policy. Phenotype is LINEAR — "mirror" is connectivity + credit, not a
     special forward op.
  3. LOCALIZED CREDIT — the apprenticing cross-entropy gradient flows ONLY through
     mirror-incident parameters (edges in/out of mirror cells + mirror biases).
     The self-RL (TD) pathway is trained only by TD. Apprenticed knowledge thus
     LIVES in the mirror cells: countable (count MIRROR genes), lesion-able (zero
     mirror→output), and still USED when acting solo (efference copy keeps mirror
     cells driving the policy even with the observation channel at zero).

Apprenticed knowledge is still Numa/Mima-validated downstream: genuine transfer
persists on re-test (Numa); cargo-cult reverts (Mima).
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trioron.core import Envelope, construct
from trioron.core.epigenome import (
    LINEAR, MIRROR, OUTPUT, PERCEPTION, CREDIT_ELIGIBLE, has_gene, set_gene,
)
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from experiments.world.tile_world import TileWorld, N_ACTION
from experiments.world.organism_v1 import PERCEPT_DIM

# Input layout the mirror organism expects: [percepts(77) | obs_action(6)] = 83.
OBS_DIM = N_ACTION
INPUT_DIM = PERCEPT_DIM + OBS_DIM


# ----------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------
def _ids_with_gene(arena, gene):
    return torch.tensor(
        [c for c in arena.alive_ids().tolist()
         if has_gene(int(arena.epigenome[c].item()), gene)],
        dtype=torch.long,
    )


def add_mirror_cell(arena, src_interior, src_obs, dst_out, seed_w=0.1):
    """Allocate one MIRROR cell: reads own-action (interior) + observed-action
    (obs channel), projects to output. LINEAR phenotype, MIRROR-tagged."""
    cid = int(arena.alloc(1)[0].item())
    arena.parent[cid] = -1
    arena.lineage_root[cid] = cid
    arena.output_dim[cid] = 1
    epi = int(arena.epigenome[cid].item())          # bit LINEAR already set by alloc
    epi = set_gene(epi, MIRROR)
    epi = set_gene(epi, CREDIT_ELIGIBLE)
    arena.epigenome[cid] = epi
    arena.refresh_phenotype(cid)                     # resolves to LINEAR (MIRROR is a marker)
    # incoming: interior (efference copy of own action) + obs channel (observed action)
    src = torch.cat([src_interior, src_obs]).to(torch.int32)
    arena.add_edges(src, torch.full_like(src, cid), seed_w * torch.randn(src.numel()))
    # outgoing: into the policy/output
    d = dst_out.to(torch.int32)
    arena.add_edges(torch.full_like(d, cid), d, seed_w * torch.randn(d.numel()))
    return cid


def build_mirror(seed, *, n_mirror=8, nonlinear=False, capacity=2048):
    """Organism with an observation channel + mirror cells over a seeded core."""
    torch.manual_seed(seed)
    sub = construct(
        base=seeded(INPUT_DIM, N_ACTION, interior_cells=32, nonlinear=nonlinear),
        envelope=Envelope(max_parameter_bytes=400_000),
        dispatch_table=default_dispatch_table(), capacity=capacity, sparsity_k=0,
    )
    sub.compile()
    a = sub.arena
    # The last OBS_DIM perception cells (highest ids) are the observation channel.
    perc = _ids_with_gene(a, PERCEPTION)
    obs_ids = perc[-OBS_DIM:]
    interior = torch.tensor(
        [c for c in a.alive_ids().tolist()
         if has_gene(int(a.epigenome[c].item()), CREDIT_ELIGIBLE)
         and not has_gene(int(a.epigenome[c].item()), OUTPUT)],
        dtype=torch.long,
    )
    out_ids = _ids_with_gene(a, OUTPUT)
    for _ in range(n_mirror):
        add_mirror_cell(a, interior, obs_ids, out_ids)
    a.rank_dirty = True
    sub.compile()
    sub.prepare_training()
    return sub


def mirror_ids(sub):
    return _ids_with_gene(sub.arena, MIRROR)


# ----------------------------------------------------------------------
# Localized credit — keep imitation gradient on mirror-incident params only
# ----------------------------------------------------------------------
def keep_only_mirror_grads(sub):
    """Zero every gradient entry NOT incident to a mirror cell. Mirrors the
    pattern of Scheduler.zero_dormant_grads (per-cell / per-edge masking)."""
    a = sub.arena
    mids = mirror_ids(sub)
    is_mirror = torch.zeros(a.capacity, dtype=torch.bool)
    is_mirror[mids] = True
    if a.bias.grad is not None:
        keep = torch.zeros_like(a.bias.grad, dtype=torch.bool)
        keep[mids] = True
        a.bias.grad[~keep] = 0.0
    if a.edge_weight.grad is not None and a.edge_cursor > 0:
        src = a.edge_src[: a.edge_cursor].long()
        dst = a.edge_dst[: a.edge_cursor].long()
        incident = is_mirror[src] | is_mirror[dst]
        a.edge_weight.grad[: a.edge_cursor][~incident] = 0.0


def lesion_mirror(sub):
    """Zero all mirror→output (and mirror→anything) edge weights in place.
    After this the apprenticed contribution is gone; the TD policy remains."""
    a = sub.arena
    mids = set(mirror_ids(sub).tolist())
    with torch.no_grad():
        for e in range(a.edge_cursor):
            if int(a.edge_src[e].item()) in mids:
                a.edge_weight[e] = 0.0


def obs_onehot(actions, batch):
    """[batch, OBS_DIM] one-hot of observed master actions (int tensor)."""
    oh = torch.zeros(batch, OBS_DIM)
    oh[torch.arange(batch), actions] = 1.0
    return oh


def _solo(p_batch):
    """Pad a [batch, PERCEPT_DIM] percept with a zeroed observation channel."""
    return torch.cat([p_batch, torch.zeros(p_batch.shape[0], OBS_DIM)], dim=1)


# ----------------------------------------------------------------------
# The mirror-cell apprentice — TD self-RL + localized, solo-forward imitation
# ----------------------------------------------------------------------
def train_mirror_student(demos, *, seed, episodes, n_mirror=8, gamma=0.95,
                         lr=3e-3, batch=64, imit_w0=1.0, max_steps=300):
    sub = build_mirror(seed, n_mirror=n_mirror)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    a = sub.arena
    buf = deque(maxlen=20000)
    g = torch.Generator().manual_seed(seed + 31)
    dstates = torch.stack([d[0] for d in demos])
    dacts = torch.tensor([d[1] for d in demos])
    curve = []
    for ep in range(episodes):
        imit_w = imit_w0 * max(0.0, 1 - ep / (0.5 * episodes))   # wean off teacher
        eps = 0.1 + 0.9 * max(0.0, 1 - ep / (0.7 * episodes))
        w = TileWorld(seed=seed * 1000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            if torch.rand(1, generator=g).item() < eps:
                act = int(torch.randint(0, N_ACTION, (1,), generator=g))
            else:
                with torch.no_grad():
                    act = int(sub(_solo(p.unsqueeze(0)))[0].argmax())
            p2, r, done, info = w.step(act)
            buf.append((p, act, r, p2, float(done))); p = p2
            if len(buf) >= batch:
                idx = torch.randint(0, len(buf), (batch,), generator=g)
                bs = [buf[i] for i in idx]
                bp = _solo(torch.stack([b[0] for b in bs]))
                ba = torch.tensor([b[1] for b in bs])
                br = torch.tensor([b[2] for b in bs])
                bp2 = _solo(torch.stack([b[3] for b in bs]))
                bd = torch.tensor([b[4] for b in bs])
                q = sub(bp)[torch.arange(batch), ba]
                with torch.no_grad():
                    tgt = br + gamma * sub(bp2).max(dim=1).values * (1 - bd)
                td_loss = torch.nn.functional.mse_loss(q, tgt)
                opt.zero_grad(); td_loss.backward()
                if imit_w > 0 and len(demos) >= batch:
                    # master-avatar apprenticing (Rocky 2026-06-02). Credit gated
                    # to mirror cells. TWO terms:
                    #   avatar CE   — obs=master_action: keeps the teleoperation
                    #                 channel (obs→mirror→output) competent so the
                    #                 master can actually pilot the body.
                    #   internalize — obs=0: distills the avatar's behaviour into
                    #                 the SOLO policy (interior→mirror→output) so the
                    #                 wisdom persists once the master lets go.
                    td_b = a.bias.grad.clone(); td_w = a.edge_weight.grad.clone()
                    didx = torch.randint(0, len(demos), (batch,), generator=g)
                    da = dacts[didx]
                    avatar = torch.cat([dstates[didx], obs_onehot(da, batch)], dim=1)
                    solo = _solo(dstates[didx])
                    ce = torch.nn.functional.cross_entropy
                    opt.zero_grad()
                    (imit_w * (ce(sub(avatar), da) + ce(sub(solo), da))).backward()
                    keep_only_mirror_grads(sub)
                    a.bias.grad.add_(td_b); a.edge_weight.grad.add_(td_w)
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                sub.zero_dormant_grads(); opt.step()
        curve.append(info["t"])
    return curve, sub


@torch.no_grad()
def eval_mirror_greedy(sub, seed, episodes=30, max_steps=300):
    out = []
    for ep in range(episodes):
        w = TileWorld(seed=seed * 7000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            act = int(sub(_solo(p.unsqueeze(0)))[0].argmax())
            p, _, done, info = w.step(act)
        out.append(info["t"])
    return statistics.mean(out)


@torch.no_grad()
def eval_avatar_greedy(sub, teacher, seed, episodes=30, max_steps=300):
    """Master pilots the body through the observation channel (teleoperation):
    obs = teacher's action on each state. Shows the avatar/hole is a real,
    usable channel — a master can grab the wheel at deployment."""
    out = []
    for ep in range(episodes):
        w = TileWorld(seed=seed * 7000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            m = int(teacher(p.unsqueeze(0))[0].argmax())     # master's intent
            x = torch.cat([p.unsqueeze(0), obs_onehot(torch.tensor([m]), 1)], dim=1)
            act = int(sub(x)[0].argmax()); p, _, done, info = w.step(act)
        out.append(info["t"])
    return statistics.mean(out)


# ----------------------------------------------------------------------
# Verdict — mirror-cells vs loss-only-proxy vs learn-from-scratch
# ----------------------------------------------------------------------
def verdict(seeds, episodes, teacher_episodes, n_mirror=8):
    from experiments.world.organism_v1 import train_organism
    from experiments.world.teacher_student import (
        collect_demos, train_student, eval_greedy, episodes_to,
    )
    print(f"mirror_cells VERDICT: real cell type vs loss-only proxy vs scratch. "
          f"seeds={seeds} student_eps={episodes} n_mirror={n_mirror}")
    print("training teacher...")
    _, teacher = train_organism(99, teacher_episodes)
    t_surv = eval_greedy(teacher, 99)          # teacher_student.eval_greedy returns a mean
    demos = collect_demos(teacher, 40, 99)
    print(f"  teacher greedy survival = {t_surv:.1f}  ({len(demos)} demo steps)")
    print("reactive≈38.5, random≈27.9 | win = apprentice reaches competence FASTER\n")

    def report(name, e2c, earlies, finals, avatars=None):
        clean = [x for x in e2c if x is not None]
        e2c_str = f"{statistics.mean(clean):.0f}" if clean else "never"
        extra = (f"  avatar-piloted={statistics.mean(avatars):.1f}"
                 if avatars else "")
        print(f"  {name:>14s}: early(first30)={statistics.mean(earlies):.1f}  "
              f"eps→beat-reactive={e2c_str} ({len(clean)}/{seeds})  "
              f"final solo={statistics.mean(finals):.1f}{extra}")

    for name in ("scratch", "loss-proxy", "mirror-cells"):
        e2c, earlies, finals, avatars = [], [], [], []
        for seed in range(seeds):
            if name == "mirror-cells":
                curve, sub = train_mirror_student(demos, seed=seed,
                                                  episodes=episodes, n_mirror=n_mirror)
                finals.append(eval_mirror_greedy(sub, seed))
                avatars.append(eval_avatar_greedy(sub, teacher, seed))
            else:
                curve, sub = train_student(name == "loss-proxy", demos,
                                           seed=seed, episodes=episodes)
                finals.append(eval_greedy(sub, seed))
            e2c.append(episodes_to(curve, 38.5))
            earlies.append(statistics.mean(curve[:30]))
        report(name, e2c, earlies, finals, avatars if name == "mirror-cells" else None)
    return 0


# ----------------------------------------------------------------------
# Smoke — prove the wiring before the n=3 verdict run
# ----------------------------------------------------------------------
def smoke():
    print("mirror_cells smoke — wiring proof (not a verdict)\n")
    sub = build_mirror(0, n_mirror=8)
    mids = mirror_ids(sub)
    print(f"[1] cell type exists: {len(mids)} MIRROR cells "
          f"(ids {mids.min().item()}..{mids.max().item()}); "
          f"total cells={sub.n_cells}, edges={sub.n_edges}")
    assert len(mids) == 8

    # [2] mirror cells FIRE ON OBSERVED ACTION: same percepts, different obs
    # channel → mirror activations must change.
    w = TileWorld(seed=1, max_steps=300)
    p = w.percept().unsqueeze(0)                       # [1, 74]
    x_solo = torch.cat([p, torch.zeros(1, OBS_DIM)], dim=1)
    x_obs0 = torch.cat([p, obs_onehot(torch.tensor([0]), 1)], dim=1)
    x_obs3 = torch.cat([p, obs_onehot(torch.tensor([3]), 1)], dim=1)
    with torch.no_grad():
        sub(x_solo); a_solo = sub.last_activations[0, mids].clone()
        sub(x_obs0); a_obs0 = sub.last_activations[0, mids].clone()
        sub(x_obs3); a_obs3 = sub.last_activations[0, mids].clone()
    d0 = (a_obs0 - a_solo).abs().mean().item()
    d3 = (a_obs3 - a_solo).abs().mean().item()
    print(f"[2] fires on observed action: Δact(obs=0)={d0:.4f}  "
          f"Δact(obs=3)={d3:.4f}  (both must be > 0)")
    assert d0 > 0 and d3 > 0

    # [3] LOCALIZED CREDIT: an apprenticing CE step updates ONLY mirror-incident
    # params. Snapshot edge weights, take one gated imitation step, diff.
    a = sub.arena
    mset = set(mids.tolist())
    src = a.edge_src[: a.edge_cursor].long()
    dst = a.edge_dst[: a.edge_cursor].long()
    incident = torch.tensor([(int(s) in mset or int(d) in mset)
                             for s, d in zip(src.tolist(), dst.tolist())])
    opt = torch.optim.SGD(sub.trainable_tensors(), lr=0.1)
    demo_x = torch.cat([w.percept().unsqueeze(0).repeat(16, 1),
                        obs_onehot(torch.randint(0, N_ACTION, (16,)), 16)], dim=1)
    demo_a = torch.randint(0, N_ACTION, (16,))
    before = a.edge_weight[: a.edge_cursor].clone()
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(sub(demo_x), demo_a)
    loss.backward()
    keep_only_mirror_grads(sub)
    opt.step()
    changed = (a.edge_weight[: a.edge_cursor] - before).abs() > 1e-9
    leaked = int((changed & ~incident).sum())
    hit = int((changed & incident).sum())
    print(f"[3] localized credit: {hit} mirror-incident edges moved, "
          f"{leaked} non-mirror edges moved (leak must be 0)")
    assert leaked == 0 and hit > 0

    # [4] LESION: removing mirror output collapses the apprenticed contribution
    # but the forward still runs (TD policy intact).
    with torch.no_grad():
        logits_pre = sub(x_solo).clone()
    lesion_mirror(sub)
    with torch.no_grad():
        logits_post = sub(x_solo)
    print(f"[4] lesion-able: Δlogits after mirror lesion="
          f"{(logits_post - logits_pre).abs().mean().item():.4f} "
          f"(forward still runs, shape {tuple(logits_post.shape)})")

    print("\nSMOKE PASS — mirror cells are a real, firing, credit-localized, "
          "lesion-able cell type.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="wiring proof only")
    ap.add_argument("--verdict", action="store_true", help="n=3 comparison run")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--teacher-episodes", type=int, default=400)
    ap.add_argument("--n-mirror", type=int, default=8)
    args = ap.parse_args()
    if args.verdict:
        raise SystemExit(verdict(args.seeds, args.episodes,
                                 args.teacher_episodes, n_mirror=args.n_mirror))
    raise SystemExit(smoke())
