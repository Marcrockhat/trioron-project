"""Arm D2 — the full-CL-stack synthesis: ONE organism, native machinery.

D1 (arc.py `GatedReflexWisdom`) proved the reflex->wisdom thesis cheaply with a
*composite* reflex (5 separate donors + manifold router) and a *separate* curiosity
net, gated by urgency. D2 collapses that into ONE substrate and exercises the native
continual-learning machinery the world experiments kept bypassing (manual §5/§8):

  1. DISTILL the 5-donor reflex into one substrate. Two routes, run as an ablation:
       graft   — absorb the donors' frozen features into one arena (lifecycle/graft.py)
                 + settle a thin readout head to reproduce org.act. (Rocky's pick; the
                 native route. Foreign seed-mismatched donors can't be zero-shot
                 composed, so the head is settled — the documented absorb+settle recipe.)
       distill — soft-KL behavioural clone of the routed donor's logits into a fresh
                 seeded net. (The comparison arm.)
  2. CONSOLIDATE + LOCK it: credit-based dormant locking (learning/credit.py) + λ
     anchor with |w·g| saliency (learning/epigenetic_lock.py) -> the reflex is innate,
     drift-protected.
  3. GROW curiosity capacity on top (divide under frustration) + TD-learn it on the
     normalized learning-progress reward (arc.py's curiosity driver), defended by the
     λ EWC pull + manifold replay so it does NOT overwrite the locked reflex.
  4. DREAM: manifold-replay of the reflex action-states + consolidate at episode
     boundaries (learning/dream.py).

The urgency GATE (D1-shaped, so survival/density stay comparable): under danger
(max danger >= θ) a FROZEN snapshot of the reflex acts to SURVIVE; in the safe slack
the LIVE (growing) substrate acts to LEARN, learning off-policy from every transition.

Headline beyond D1 — the RETENTION PROBE. Run the whole stack twice, machinery ON
(lock+λ+dream) vs OFF (raw Adam, the recurring world-forgetting bypass), and measure
the LIVE substrate's standalone survival (gate θ=∞) after curiosity training:
  ON  -> reflex intact (the native machinery holds it through curiosity training)
  OFF -> craters (unanchored drift toward the curiosity objective; manual §7).

Usage:
    python3 -m experiments.world.arc_d2 --smoke
    python3 -m experiments.world.arc_d2 --reflex graft   --episodes 200
    python3 -m experiments.world.arc_d2 --reflex distill  --episodes 200
    python3 -m experiments.world.arc_d2 --reflex both --episodes 200   # the ablation
"""
from __future__ import annotations

import argparse
import copy
import statistics
import sys
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trioron.core import Envelope, construct
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from trioron.core.epigenome import (
    OUTPUT, PERCEPTION, has_gene, clear_gene,
)
from trioron.core.state import CellState
from trioron.learning.credit import CreditTracker, CreditConfig
from trioron.learning.manifold import ManifoldArchive
from trioron.learning import epigenetic_lock as epi
from trioron.learning.frustration import FrustrationDetector
from trioron.learning.dream import dream_cycle
from trioron.lifecycle.graft import graft

from experiments.world.tile_world import TileWorld, N_ACTION
from experiments.world.vocabulary import build_vocabulary, danger, PRIM_ORDER
from experiments.world.mirror_cells import _solo
from experiments.world.numa import N_PAIR, contrast_targets
from experiments.world.fire_taming import evaluate, _cause
from experiments.world.arc import build_observer, run_arm, FrozenPolicy, PERCEPT_DIM

# The single-substrate output: N_ACTION action logits (the reflex/Q head) followed by
# N_PAIR world-model logits (the curiosity machinery — its learning-progress signal).
N_OUT = N_ACTION + N_PAIR


# ----------------------------------------------------------------------
# Helpers shared by both reflex routes
# ----------------------------------------------------------------------
def _net_act_fn(sub):
    """A (w, p) -> action greedy policy reading the action slice of a substrate."""
    @torch.no_grad()
    def act(w, p, m=sub):
        return int(m(p.unsqueeze(0))[0, :N_ACTION].argmax())
    return act


@torch.no_grad()
def _collect_states(teacher, *, seeds, max_steps=300, on_policy_frac=0.5):
    """Visited (percept, teacher_action) pairs. Half the episodes are ON-policy (the
    reflex acts, recording the states it actually visits); half are EXPLORATORY (random
    acts, so the clone also learns the teacher's response in the danger states the reflex
    rarely dwells in — DAgger-flavoured coverage where survival is decided)."""
    g = torch.Generator().manual_seed(777)
    P, A = [], []
    for ep in range(seeds):
        w = TileWorld(seed=ep * 53 + 11, max_steps=max_steps)
        p = w.percept(); done = False
        on_policy = (ep % 2 == 0) if on_policy_frac >= 0.5 else False
        while not done:
            a_teacher = teacher.act(w, p)        # label every visited state with org.act
            P.append(p); A.append(a_teacher)
            a = a_teacher if on_policy else int(torch.randint(0, N_ACTION, (1,), generator=g))
            p, _, done, _ = w.step(a)
    return torch.stack(P), torch.tensor(A, dtype=torch.long)


@torch.no_grad()
def _teacher_soft_logits(teacher, states):
    """[N,77] -> [N,N_ACTION] the ROUTED donor's action logits per state (soft target
    for the distill route — carries the donor's relative action preferences)."""
    out = torch.zeros(states.shape[0], N_ACTION)
    for i in range(states.shape[0]):
        p = states[i]
        idx = teacher.router.route(p)
        donor = teacher.donors[PRIM_ORDER[idx]]
        out[i] = donor(_solo(p.unsqueeze(0)))[0]
    return out


# ----------------------------------------------------------------------
# Reflex route 1 — DISTILL (soft-KL behavioural clone, the comparison arm)
# ----------------------------------------------------------------------
def build_reflex_distill(teacher, *, seed, states, actions, epochs=8, lr=3e-3,
                         batch=128, temp=1.0, nonlinear=False):
    soft = _teacher_soft_logits(teacher, states)
    sub = construct(base=seeded(PERCEPT_DIM, N_OUT, interior_cells=32, nonlinear=nonlinear),
                    envelope=Envelope(max_parameter_bytes=500_000),
                    dispatch_table=default_dispatch_table(), capacity=2048, sparsity_k=0)
    sub.compile(); sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    n = states.shape[0]
    tgt = torch.softmax(soft / temp, dim=1)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logp = torch.log_softmax(sub(states[idx])[:, :N_ACTION] / temp, dim=1)
            loss = torch.nn.functional.kl_div(logp, tgt[idx], reduction="batchmean") * (temp ** 2)
            opt.zero_grad(); loss.backward()
            sub.zero_dormant_grads(); opt.step()
    return sub


# ----------------------------------------------------------------------
# Reflex route 2 — GRAFT (absorb the donors' frozen features + settle the head)
# ----------------------------------------------------------------------
def build_reflex_graft(teacher, *, seed, states, actions, settle_epochs=12, lr=3e-3,
                       batch=128, nonlinear=False):
    """Transplant each donor's frozen interior into ONE recipient arena, then settle a
    thin readout head to reproduce org.act. Foreign (seed-mismatched) donors cannot be
    zero-shot logit-composed, so the donor FEATURES are absorbed frozen and only the
    recipient's own cells + head are trained — the documented absorb+settle recipe."""
    recipient = construct(base=seeded(PERCEPT_DIM, N_OUT, interior_cells=8, nonlinear=nonlinear),
                          envelope=Envelope(max_parameter_bytes=2_000_000),
                          dispatch_table=default_dispatch_table(), capacity=4096, sparsity_k=0)
    recipient.compile()
    n_grafted = 0
    # no_grad: graft copies the donor bias (which carries requires_grad from training)
    # into the recipient via in-place index-assign — tracking it would make the recipient
    # bias a non-leaf and break the optimizer. The copies still happen; only autograd
    # tracking is suppressed.
    grafted_ids = []
    for name in PRIM_ORDER:
        # Donors are 83-input (_solo pads a zeroed 6-d obs channel); grafting into a
        # 77-perception recipient drops those dead obs edges by rank-order remap. (Core
        # graft() now detaches the donor bias, so the recipient stays optimizable.)
        res = graft(recipient, teacher.donors[name], freeze=True, wiring="dense")
        grafted_ids += res.recipient_ids
        n_grafted += len(res.recipient_ids)
    # We absorb donor FEATURES, not their heads: strip the OUTPUT gene from grafted
    # cells so they don't expand the head (each donor carried its own 6 action logits).
    # The recipient's own N_OUT head reads the donor interiors (wired dense) and
    # re-derives the readout during settle.
    for cid in grafted_ids:
        e = int(recipient.arena.epigenome[cid].item())
        if has_gene(e, OUTPUT):
            recipient.arena.epigenome[cid] = clear_gene(e, OUTPUT)
            recipient.arena.output_dim[cid] = 0
    recipient.arena.refresh_all_phenotypes()
    recipient.compile()
    recipient.prepare_training()
    # Settle: donor cells are DORMANT (frozen); zero_dormant_grads trains only the
    # recipient's own interior + readout head to map frozen donor features -> org.act.
    opt = torch.optim.Adam(recipient.trainable_tensors(), lr=lr)
    n = states.shape[0]
    ce = torch.nn.functional.cross_entropy
    for _ in range(settle_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = ce(recipient(states[idx])[:, :N_ACTION], actions[idx])
            opt.zero_grad(); loss.backward()
            recipient.zero_dormant_grads(); opt.step()
    return recipient, n_grafted


# ----------------------------------------------------------------------
# Consolidate + lock the reflex (the ON-machinery anchoring; manual §5)
# ----------------------------------------------------------------------
def consolidate_and_lock(sub, states, actions, *, full_cov=False):
    """Credit-lock high-engagement reflex cells (hard anchor) + λ-anchor with |w·g|
    saliency (soft anchor) + build the per-action manifold archive for replay. Returns
    (credit, archive). consecutive_tasks=1 so a SINGLE boundary can lock (manual §5.1)."""
    credit = CreditTracker(sub.arena, CreditConfig(consecutive_tasks=1))
    ce = torch.nn.functional.cross_entropy
    n = states.shape[0]
    g = torch.Generator().manual_seed(9)
    # A few labelled passes to (a) populate engagement/utility for credit and (b) drive
    # the |w·g| saliency EMA for λ.
    for _ in range(6):
        idx = torch.randint(0, n, (128,), generator=g)
        logits = sub(states[idx])[:, :N_ACTION]
        loss = ce(logits, actions[idx])
        sub.arena.edge_weight.grad = None; sub.arena.bias.grad = None
        loss.backward()
        epi.accumulate_saliency(sub.arena)
        credit.update_utility()
        acts = sub.last_activations
        if acts is not None:
            credit.update_engagement(acts)
    epi.refresh_lambda(sub.arena)       # |w·g| row-sum -> node_lambda
    epi.anchor(sub.arena)               # snapshot ŵ the EWC pull defends
    locked = credit.consolidate()       # hard dormant-lock the most-engaged stable cells
    # Per-action manifold archive over the reflex's visited percepts (pseudo-rehearsal).
    archive = ManifoldArchive(sub.arena, full_cov=full_cov)
    for a in range(N_ACTION):
        mask = actions == a
        if int(mask.sum()) >= 2:
            archive.update_class(a, states[mask])
    archive.finalize_all()
    return credit, archive, len(locked)


# ----------------------------------------------------------------------
# The curiosity driver on the LIVE substrate (machinery toggle)
# ----------------------------------------------------------------------
def _active_interior(arena):
    """Active, non-output, non-perception cells — divide candidates for growth. Must NOT
    filter on the LINEAR gene: a nonlinear substrate's interior cells carry DENDRITE (quad)
    instead of LINEAR, so a LINEAR filter would make the candidate set empty and silently
    cap growth at zero on exactly the nonlinear nets the relational skills require."""
    mask = (arena.alive & (arena.state == CellState.ACTIVE)
            & ~has_gene(arena.epigenome, OUTPUT).bool()
            & ~has_gene(arena.epigenome, PERCEPTION).bool())
    return mask.nonzero(as_tuple=False).squeeze(-1)


class D2Driver:
    """Curiosity (normalized learning-progress) TD driver on a SHARED live substrate.

    machinery=True  -> λ EWC penalty + manifold/dream replay + credit consolidate +
                       frustration-gated growth (the native CL stack).
    machinery=False -> raw Adam over trainable_tensors, no anchor/replay/growth (the
                       recurring world-forgetting bypass; the retention-probe control).
    """

    def __init__(self, sub, *, machinery, credit=None, archive=None, episodes,
                 seed=0, lr=3e-3, gamma=0.95, batch=64, eps_start=1.0, eps_end=0.1,
                 ewc_strength=1e3, grow_every=400, max_grow=12, replay_batch=64):
        self.sub = sub
        self.machinery = machinery
        self.credit = credit
        self.archive = archive
        self.opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
        self.lr = lr
        self.buf = deque(maxlen=20000)
        self.g = torch.Generator().manual_seed(seed + 101)
        self.gamma = gamma; self.batch = batch
        self.eps_start = eps_start; self.eps_end = eps_end; self.episodes = episodes
        self._ep = 0; self.eps = eps_start
        self.ri_scale = None; self.rew_sum = 0.0; self.rew_n = 0
        self.ewc_strength = ewc_strength
        self.frust = FrustrationDetector()
        self.grow_every = grow_every; self.max_grow = max_grow
        self.n_grown = 0; self._step = 0; self.replay_batch = replay_batch

    def start_episode(self):
        if self.machinery and self._ep > 0 and self.archive is not None:
            # Dream: manifold-replay all reflex action-states + consolidate (defends the
            # reflex head against the curiosity gradient accumulated this episode).
            dream_cycle(self.sub, self.credit, self.archive, current_classes=[],
                        frustration_pressure=min(1.0, self.frust.multiplier - 1.0))
        self._ep += 1
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * \
            max(0.0, 1 - self._ep / (0.7 * self.episodes))

    def act(self, w, p):
        if torch.rand(1, generator=self.g).item() < self.eps:
            return int(torch.randint(0, N_ACTION, (1,), generator=self.g))
        with torch.no_grad():
            return int(self.sub(p.unsqueeze(0))[0, :N_ACTION].argmax())

    def _wm_err(self, p, pair_tgt):
        with torch.no_grad():
            pred = self.sub(p.unsqueeze(0))[0, N_ACTION:N_OUT]
        return float(((pred - pair_tgt) ** 2).mean())

    def _norm_lp(self, raw):
        if self.ri_scale is None:
            self.ri_scale = raw + 1e-6
        else:
            self.ri_scale = 0.99 * self.ri_scale + 0.01 * raw
        return raw / (self.ri_scale + 1e-8)

    def _maybe_grow(self):
        if not self.machinery or self.n_grown >= self.max_grow:
            return
        if self._step % self.grow_every != 0 or not self.frust.is_frustrated:
            return
        cand = _active_interior(self.sub.arena)
        if cand.numel() == 0:
            return
        pid = int(cand[int(torch.randint(0, cand.numel(), (1,), generator=self.g))])
        from trioron.lifecycle.grow import divide
        if divide(self.sub.arena, pid) is not None:
            self.sub.compile()
            self.opt = torch.optim.Adam(self.sub.trainable_tensors(), lr=self.lr)
            self.n_grown += 1

    def learn(self, p, a, r_world, p2, done, pair_tgt):
        self._step += 1
        err_before = self._wm_err(p, pair_tgt)
        if len(self.buf) >= 2 * self.batch:
            idx = torch.randint(0, len(self.buf), (self.batch,), generator=self.g)
            bs = [self.buf[i] for i in idx]
            bp = torch.stack([b[0] for b in bs]); ba = torch.tensor([b[1] for b in bs])
            brw = torch.tensor([b[2] for b in bs]); bp2 = torch.stack([b[3] for b in bs])
            bd = torch.tensor([b[4] for b in bs]); btgt = torch.stack([b[5] for b in bs])
            out = self.sub(bp)
            q = out[torch.arange(self.batch), ba]
            with torch.no_grad():
                q_next = self.sub(bp2)[:, :N_ACTION].max(dim=1).values
                target = brw + self.gamma * q_next * (1 - bd)
            loss = (torch.nn.functional.mse_loss(q, target)
                    + torch.nn.functional.mse_loss(out[:, N_ACTION:N_OUT], btgt))
            if self.machinery:
                loss = loss + self.ewc_strength * epi.ewc_penalty(self.sub.arena)
            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sub.trainable_tensors(), 1.0)
            self.sub.zero_dormant_grads(); self.opt.step()
            self.frust.step(loss.item())
            self._replay_reflex()
            self._maybe_grow()
        err_after = self._wm_err(p, pair_tgt)
        reward = self._norm_lp(max(0.0, err_before - err_after))
        self.rew_sum += reward; self.rew_n += 1
        self.buf.append((p, a, reward, p2, done, pair_tgt))

    def _replay_reflex(self):
        """Manifold pseudo-rehearsal: sample reflex (percept,action) from the archive and
        re-teach the action head, so curiosity TD does not drift it off the reflex."""
        if not self.machinery or self.archive is None:
            return
        xs, ys = [], []
        per = max(4, self.replay_batch // max(1, N_ACTION))
        for a in range(N_ACTION):
            astro = self.archive.get(a)
            if astro is None:
                continue
            xs.append(astro.sample(per))
            ys.append(torch.full((per,), a, dtype=torch.long))
        if not xs:
            return
        x = torch.cat(xs); y = torch.cat(ys)
        loss = torch.nn.functional.cross_entropy(self.sub(x)[:, :N_ACTION], y)
        if self.machinery:
            loss = loss + self.ewc_strength * epi.ewc_penalty(self.sub.arena)
        self.opt.zero_grad(); loss.backward()
        self.sub.zero_dormant_grads(); self.opt.step()

    def diagnostics(self):
        mean_r = self.rew_sum / max(self.rew_n, 1)
        return (f"r_int mean={mean_r:.3f}  grown={self.n_grown}  "
                f"frust×{self.frust.multiplier:.1f}")


# ----------------------------------------------------------------------
# The D2 agent — urgency gate: frozen reflex under danger, live net in the slack
# ----------------------------------------------------------------------
class D2Agent:
    def __init__(self, frozen_reflex_act, driver, *, theta=0.5):
        self.reflex = frozen_reflex_act
        self.driver = driver
        self.theta = theta
        self.reflex_acts = 0; self.curio_acts = 0

    def start_episode(self):
        self.driver.start_episode()

    def act(self, w, p):
        if max(danger(w).values()) >= self.theta:
            self.reflex_acts += 1
            return self.reflex(w, p)
        self.curio_acts += 1
        return self.driver.act(w, p)

    def learn(self, p, a, r, p2, done, pair_tgt):
        self.driver.learn(p, a, r, p2, done, pair_tgt)   # off-policy: every transition

    def diagnostics(self):
        tot = max(self.reflex_acts + self.curio_acts, 1)
        return (f"gate θ={self.theta} reflex {100*self.reflex_acts/tot:.0f}%/"
                f"curio {100*self.curio_acts/tot:.0f}%  {self.driver.diagnostics()}")


# ----------------------------------------------------------------------
# Retention probe — live-net-only survival (gate θ=∞)
# ----------------------------------------------------------------------
def retention_survival(sub, *, seeds=40):
    surv, _, _ = evaluate(_net_act_fn(sub), "live-net only", seeds=seeds)
    return surv


# ----------------------------------------------------------------------
# Build a reflex (either route) and report its standalone fidelity
# ----------------------------------------------------------------------
def make_reflex(method, teacher, *, seed, states, actions, nonlinear, eval_seeds):
    if method == "graft":
        sub, n_grafted = build_reflex_graft(teacher, seed=seed, states=states,
                                            actions=actions, nonlinear=nonlinear)
        info = f"grafted {n_grafted} donor cells"
    elif method == "distill":
        sub = build_reflex_distill(teacher, seed=seed, states=states, actions=actions,
                                   nonlinear=nonlinear)
        info = "soft-KL behavioural clone"
    else:
        raise ValueError(method)
    fidelity, _, _ = evaluate(_net_act_fn(sub), f"reflex[{method}]", seeds=eval_seeds)
    return sub, fidelity, info


# ----------------------------------------------------------------------
# Run one reflex method through the full D2 stack (ON and OFF machinery)
# ----------------------------------------------------------------------
def run_d2(method, *, teacher, states, actions, episodes, theta, obs_seed, driver_seed,
           nonlinear, eval_seeds, machinery_modes=("on", "off")):
    print(f"\n=== D2 reflex route: {method.upper()} ===", flush=True)
    reflex_sub, fidelity, info = make_reflex(method, teacher, seed=driver_seed,
                                             states=states, actions=actions,
                                             nonlinear=nonlinear, eval_seeds=eval_seeds)
    print(f"  reflex distilled ({info}); standalone survival (fidelity) = {fidelity:.1f}",
          flush=True)
    # Frozen snapshot of the reflex — acts under danger, identical for ON and OFF.
    frozen = copy.deepcopy(reflex_sub); frozen.compile()
    frozen_act = _net_act_fn(frozen)

    results = {}
    for mode in machinery_modes:
        live = copy.deepcopy(reflex_sub); live.compile(); live.prepare_training()
        credit = archive = None; n_locked = 0
        if mode == "on":
            credit, archive, n_locked = consolidate_and_lock(live, states, actions)
        driver = D2Driver(live, machinery=(mode == "on"), credit=credit, archive=archive,
                          episodes=episodes, seed=driver_seed)
        agent = D2Agent(frozen_act, driver, theta=theta)
        print(f"\n  -- machinery {mode.upper()} (locked {n_locked} cells) --", flush=True)
        lengths, _, ledger, density = run_arm(agent, f"D2:{method}/{mode}",
                                              episodes=episodes, obs_seed=obs_seed,
                                              nonlinear=nonlinear)
        retention = retention_survival(live, seeds=eval_seeds)
        results[mode] = dict(last30=statistics.mean(lengths[-30:]), density=density,
                             retention=retention, net=ledger.net())
        print(f"     retention (live-net-only survival) = {retention:.1f}  "
              f"(reflex fidelity was {fidelity:.1f})", flush=True)
    return fidelity, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reflex", default="both", choices=["graft", "distill", "both"])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--router-seeds", type=int, default=120)
    ap.add_argument("--per-prim-cap", type=int, default=1500)
    ap.add_argument("--distill-seeds", type=int, default=60,
                    help="exploratory+on-policy episodes for the distill dataset")
    ap.add_argument("--obs-seed", type=int, default=0)
    ap.add_argument("--driver-seed", type=int, default=0)
    ap.add_argument("--eval-seeds", type=int, default=40)
    ap.add_argument("--nonlinear", action="store_true")
    ap.add_argument("--machinery", default="both", choices=["on", "off", "both"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.router_seeds, args.per_prim_cap = 30, 300
        args.distill_seeds, args.eval_seeds = 8, 6
        if args.episodes == 200:
            args.episodes = 30
    modes = ("on", "off") if args.machinery == "both" else (args.machinery,)
    methods = ["graft", "distill"] if args.reflex == "both" else [args.reflex]

    print("=== ARM D2 — full-CL-stack synthesis (one organism, native machinery) ===")
    print("gate: frozen reflex under danger (θ), live growing net in the slack.")
    print("axis: net Numa DENSITY (per 1k steps); survival = gate; RETENTION = headline.\n")

    teacher, _ = build_vocabulary(full_cov=True, router_seeds=args.router_seeds,
                                  per_prim_cap=args.per_prim_cap)
    teacher_surv, _, _ = evaluate(lambda w, p: teacher.act(w, p), "TEACHER (D1 reflex)",
                                  seeds=args.eval_seeds)
    states, actions = _collect_states(teacher, seeds=args.distill_seeds)
    print(f"  distill set: {states.shape[0]} (percept, org.act) pairs "
          f"from {args.distill_seeds} episodes\n", flush=True)

    summary = {}
    for method in methods:
        fidelity, results = run_d2(method, teacher=teacher, states=states, actions=actions,
                                   episodes=args.episodes, theta=args.theta,
                                   obs_seed=args.obs_seed, driver_seed=args.driver_seed,
                                   nonlinear=args.nonlinear, eval_seeds=args.eval_seeds,
                                   machinery_modes=modes)
        summary[method] = (fidelity, results)

    print("\n=== D2 SUMMARY ===")
    print(f"  teacher (D1 reflex) survival = {teacher_surv:.1f}")
    print(f"  {'route/mode':>16s}  {'last30':>7s}  {'net-Numa/1k':>11s}  "
          f"{'retention':>9s}  {'fidelity':>8s}")
    for method, (fidelity, results) in summary.items():
        for mode, r in results.items():
            print(f"  {method+'/'+mode:>16s}  {r['last30']:>7.1f}  {r['density']:>+11.3f}  "
                  f"{r['retention']:>9.1f}  {fidelity:>8.1f}")
    print("\n  Win: ON retention ≈ fidelity (reflex held) while OFF retention craters,")
    print("  and D2 last30/density ≥ D1 (158.6 / +0.177).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
