"""KINOPSIS (kine- motion + opsis sight) — the single motion-sight organism (s057, NEXT-0/2 of s056; Rocky: "focus on 2(a)"): the s053 organ + the motion-
silhouette leaf + the velocity leaf + the s056 evidence router, built by a CONTINUAL SCHEDULE and
carried as a single saveable object.

Schedule (per seed):
  stage 1  OLD WORLD     : shape_prefeeder.Organ (shape/whole/fill leaves, static shape world)      -> frozen
  stage 2  MOVING WORLD  : (a) optional replay-settle of the organ heads (s054 recipe: class-balanced
                           full-cov archive samples of every old class mixed into each batch);
                           (b) msil leaf acquired on the motion-grouped shape stream (8 ep), frozen;
                           (c) router (motion_router.Router, 57 params) fitted alone on the moving train
                           packets + their FROZEN REPLAY, voters frozen; motion voter hard-gated.
  stage 3  VELOCITY      : vel leaf on motion_full (17-way), paste-and-go (separate head, no interaction).
Scored AFTER EVERY STAGE on moving test (mixed/photo/flat; moving packets) + old world + velocity, so
forgetting is tracked per stage.  Arms (= how stage 2 arbitrates):
  uniform        : equal log-softmax vote (the motion_absorb "+msil voter" arm)            [reference]
  router         : evidence router, no settle                                               [s056 router, now inside the organism]
  settle+router  : replay-settle the organ heads first, then msil + router on the settled voters
  settle+uniform : replay-settle, equal vote                                                [motion_absorb settle+replay arm]
The organism class is the package ``trioron.organisms.Kinopsis`` (s062); this file is the WORLD side
(feature streams, evidence, splits, schedule, scoring). One object: save(path) / load(path); size reported.  Inference API:
  shape(streams, evidence) -> [N,5] routed log-prob;  velocity(Zmotion) -> [N,17] logits.
Env: SEEDS, EPOCHS (leaves), STEPS (router, 300), SETTLE_N (2000), SETTLE_STEPS (200).
"""
import os, sys, time, copy, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import motion as MO, motion_leaf as ML, grouping as GR, shapes as SH
from experiments.progenitor.motion_absorb import moving_streams, _shape_streams, nest_logits, acc
from experiments.progenitor.motion_router import evidence_split, evidence_frozen, evidence_old, conf, Router
from experiments.progenitor.shape_prefeeder import Organ
from trioron.organisms import Kinopsis, MotionOrganism
from trioron.organisms.kinopsis import QDAVoter, vote
import experiments.progenitor.shape_prefeeder as SP
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
SEEDS = int(os.environ.get("SEEDS", "3")); EPOCHS = int(os.environ.get("EPOCHS", "8")); STEPS = int(os.environ.get("STEPS", "300"))
SETTLE_N = int(os.environ.get("SETTLE_N", "2000")); SETTLE_STEPS = int(os.environ.get("SETTLE_STEPS", "200"))
SPLITS = ["train", "test", "test_photo", "test_flat"]; VOTERS = ("shape", "whole", "msil")
def log(*a): print(*a, flush=True)

def build_kinopsis(seed):
    """Stage 1 (OLD world): train the s053 shape organ for this seed and adopt it, plus the
    empty-motion-mask descriptor (what a static image feeds the msil voter)."""
    SP.PRE_SEED = seed; os.environ["PRE_SEED"] = str(seed); organ = Organ()
    X0 = SH.load("test_fresh")[0][:1]; D0, _ = GR.describe(X0, gl=[[]], canon="scale", extras=False)
    empty = torch.cat([D0["silhouette"], D0["frame"], D0["flags"]], 1)[0]
    return Kinopsis.from_organ(organ, empty, seed)

if __name__ == "__main__":
    log(f"MOTION ORGANISM s057: seeds {SEEDS} epochs {EPOCHS} router steps {STEPS} settle_n {SETTLE_N} settle_steps {SETTLE_STEPS}")
    Y = {sp: MO.load(sp)["ys"] for sp in SPLITS}; S = {sp: moving_streams(sp) for sp in SPLITS}; EV = {sp: evidence_split(sp) for sp in SPLITS}; MOV = {sp: Y[sp]["y_vel"] > 0 for sp in SPLITS}
    old = _shape_streams("test_fresh"); y_old = SH.load("test_fresh")[1]["y_shape"]; EV_old = evidence_old(); EV_fr = evidence_frozen("train")
    old_tr = _shape_streams("train"); y_old_tr = SH.load("train")[1]
    Zm = {sp: ML.feats("motion", sp) for sp in SPLITS}
    rows = []
    def score(org, arm, stage, mode, base_old):
        with torch.no_grad():
            r = dict(arm=arm, stage=stage, seed=org.seed, old=acc(org.shape(old, EV_old, mode).argmax(1), y_old)); r["forget"] = base_old - r["old"]
            for sp in SPLITS[1:]: r[sp] = acc(org.shape(S[sp], EV[sp], mode).argmax(1), Y[sp]["y_shape"], MOV[sp])
            r["vel"] = acc(org.velocity(Zm["test"]).argmax(1), Y["test"]["y_vel"]) if org.vel is not None else float("nan")
        rows.append(r); log(f"  s{org.seed} {arm:15s} {stage:9s} | mixed {r['test']:.3f} photo {r['test_photo']:.3f} flat {r['test_flat']:.3f} | old {r['old']:.3f} (forget {r['forget']:+.3f}) | vel {r['vel']:.3f}"); return r
    for seed in range(SEEDS):
        t0 = time.time(); base = build_kinopsis(seed); base_old = acc(base.shape(old, None, "uniform").argmax(1), y_old)
        log(f"  s{seed} stage 1 old world: shape {base_old:.3f} (uniform vote of the organ); zero-shot moving: " + " ".join(f"{sp} {acc(base.shape(S[sp], None, 'uniform').argmax(1), Y[sp]['y_shape'], MOV[sp]):.3f}" for sp in SPLITS[1:]))
        RA = os.environ.get("ROUTER_ARMS")   # e.g. "router,router+H,router+pc,router+H+pc,router+qda,router+qda+H+pc": stage-2 sweep of router variants
        arms = RA.split(",") if RA else ("uniform", "router", "settle+uniform", "settle+router")
        base.acquire_msil(S["train"], Y["train"]["y_shape"], epochs=EPOCHS) if RA else None
        for arm in arms:
            org = copy.deepcopy(base); mode = "router" if "router" in arm else "uniform"
            if arm.startswith("settle"): org.settle_heads(S["train"], Y["train"], old_tr, y_old_tr, n=SETTLE_N, steps=SETTLE_STEPS)
            if org.msil is None: org.acquire_msil(S["train"], Y["train"]["y_shape"], epochs=EPOCHS)
            if mode == "router": org.fit_router(S["train"], EV["train"], EV_fr, Y["train"]["y_shape"], steps=STEPS, use_h="+H" in arm, per_class="+pc" in arm, qda="+qda" in arm, wd=float(os.environ.get("WD", "0")))
            score(org, arm, "stage 2", mode, base_old)
            if RA:
                with torch.no_grad():
                    Ls = org.voters(S["test_photo"]); y = Y["test_photo"]["y_shape"]; mv = MOV["test_photo"]
                    log(f"      photo voters alone: " + " ".join(f"{float((L.argmax(1) == y)[mv].float().mean()):.3f}" for L in Ls) + f" | oracle {float(torch.stack([L.argmax(1) == y for L in Ls]).any(0)[mv].float().mean()):.3f} | router params {sum(p.numel() for p in org.router.parameters())}")
                continue
            org.acquire_vel(Zm["train"], Y["train"]["y_vel"], epochs=EPOCHS); r = score(org, arm, "stage 3", mode, base_old)
            if arm == "router":
                p = os.path.join(MO.OUT, f"motion_organism_s{seed}.pt"); sz = org.save(p); org2 = Kinopsis.load(p)
                with torch.no_grad(): same = torch.equal(org2.shape(S["test"], EV["test"]).argmax(1), org.shape(S["test"], EV["test"]).argmax(1))
                log(f"      saved {p} {sz / 1024:.0f} KiB, {org.n_params()} trainable params; reload identical predictions: {same}; router mean w (shape,whole,msil) photo {org.router(org._feat(org.voters(S['test_photo']), EV['test_photo']))[MOV['test_photo']].mean(0).tolist()}")
        log(f"  seed {seed} done in {time.time() - t0:.0f}s")
    log("\n== summary (mean±std over seeds; moving packets) ==")
    for key in dict.fromkeys((r["arm"], r["stage"]) for r in rows):
        rs = [r for r in rows if (r["arm"], r["stage"]) == key]; m = lambda k: torch.tensor([r[k] for r in rs])
        log(f"  {key[0]:15s} {key[1]:8s} mixed {m('test').mean():.3f}±{m('test').std():.3f} photo {m('test_photo').mean():.3f}±{m('test_photo').std():.3f} flat {m('test_flat').mean():.3f}±{m('test_flat').std():.3f} | old {m('old').mean():.3f} forget {m('forget').mean():+.3f}±{m('forget').std():.3f} | vel {m('vel').mean():.3f}")
