"""Motion arc diagnostics (s055, steps 2-3 validation).  Run after
`python3 experiments/progenitor/motion.py build`.

 A. Velocity read-out: linear probe (multinomial logistic, 300 Adam steps) on
    y_vel (17-way: static / 8 octants x slow,fast) for
      magnitude-only spectra of the mid frame (s052 dense_stereo, 800)  = MOTION-BLIND control
      motion_energy (25)            = where/whether, no direction
      motion_phase  (450)           = the cross-spectrum phasors
      motion_block  (475)
    + direct population decode error |v_hat - v| (no learning) per background kind.
 B. Wagon-wheel: decode error for textured fills split by y_alias (displacement
    along the texture normal > half period) -- the illusion is EXPECTED there.
 C. Common-fate grouping v3 vs colour/Otsu grouping (s053): IoU of the recovered
    body against the exact mask, flat vs photo backgrounds.  On photos colour
    grouping has no bg to threshold against; motion does.
Env: SPLIT (default test), N (cap), PART (A,B,C subset).
"""
import os, sys, math, time, torch, numpy as np
from scipy import ndimage as ndi
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import motion as MO, motion_front as MF, frontend as FE, grouping as GR
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
PART = os.environ.get("PART", "ABC"); NCAP = int(os.environ.get("N", "0"))
def log(*a): print(*a, flush=True)

def probe(Ztr, ytr, Zte, yte, ncls, steps=300):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6; Ztr, Zte = (Ztr - mu) / sd, (Zte - mu) / sd
    torch.manual_seed(0); W = torch.zeros(Ztr.shape[1], ncls, requires_grad=True); b = torch.zeros(ncls, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for _ in range(steps):
        opt.zero_grad(); (F.cross_entropy(Ztr @ W + b, ytr) + 1e-4 * (W ** 2).sum()).backward(); opt.step()
    with torch.no_grad(): return float(((Zte @ W + b).argmax(1) == yte).float().mean())

def load(sp):
    d = MO.load(sp); X, M, ys = d["X"], d["M"], d["ys"]
    if NCAP: X, M, ys = X[:NCAP], M[:NCAP], {k: v[:NCAP] for k, v in ys.items()}
    return X, M, ys

def motion_group(Xf, frame=None, floor_k=4.0, target=3.0, convex=True):
    """common-fate grouping v3 (single dominant body).  Per packet: (1) decode the dominant velocity
    (population read-out); (2) LOCK-IN baseline: pick the frame spacing b so the body moves ~`target`
    px between the two frames (slow motion integrates longer, like a lock-in), centred on `frame`;
    (3) |y_c - y_a| smoothed 3x3, thresholded at max(Otsu, floor_k * median) -- the median is the
    sensor-noise floor, without it a static packet's mask is Otsu-on-noise; (4) the s053 body pipeline:
    closing R=4 + hole fill + largest component + convex hull (edges parallel to the motion produce no
    temporal difference, the hull bridges them; CONVEX=True is the s053 convention too).
    -> bool [N,S,S]; empty when nothing moves."""
    N, T = Xf.shape[:2]; y = FE.Y(Xf.flatten(0, 1)).view(N, T, MO.S, MO.S); t = T // 2 if frame is None else frame
    spd = MF.batched(MF.decode_velocity, Xf).norm(dim=1); out = torch.zeros(N, MO.S, MO.S, dtype=torch.bool)
    for i in range(N):
        b = int(min(T - 1, max(1, math.ceil(target / max(float(spd[i]), 1e-3))))); a = max(0, t - (b + 1) // 2); c = min(T - 1, a + b); a = c - b
        dd = F.avg_pool2d((y[i, c] - y[i, a]).abs()[None, None], 3, 1, 1)[0, 0]
        m = (dd > max(GR._otsu(dd.flatten()), floor_k * float(dd.median()))).numpy()
        if m.sum() < GR.MIN_AREA: continue
        cl = ndi.binary_fill_holes(ndi.binary_closing(m, structure=GR._disk(GR.R_CLOSE), border_value=0) | m)
        lab, k = ndi.label(cl); sz = np.bincount(lab.ravel())[1:]; body = lab == (1 + sz.argmax())
        if convex: body = np.asarray(GR.convex_fill(body), bool)
        out[i] = torch.from_numpy(body)
    return out

def iou(a, b): return ((a & b).flatten(1).sum(1).float() / (a | b).flatten(1).sum(1).clamp(min=1).float())

if __name__ == "__main__":
    sp = os.environ.get("SPLIT", "test"); Xte, Mte, yte = load(sp); Xf = MO.as_float(Xte)
    tru = torch.stack([yte["y_dx"], yte["y_dy"]], 1); mv = yte["y_vel"] > 0; kind = yte["y_bgkind"]
    log(f"split {sp} n={len(Xte)} T={Xte.shape[1]} photo-frac {kind.float().mean():.2f} static-frac {(~mv).float().mean():.2f} alias-frac {yte['y_alias'].float().mean():.2f}")
    if "A" in PART:
        Xtr, Mtr, ytr = load("train"); Xtf = MO.as_float(Xtr); mid = Xtr.shape[1] // 2
        feats = {"static-spectra(mid frame, 800)": lambda Z: FE.batched(FE.dense_stereo, Z[:, mid]),
                 "motion_energy(25)": lambda Z: MF.batched(MF.motion_energy, Z),
                 "motion_phase(450)": lambda Z: MF.batched(MF.motion_phase, Z),
                 "motion_block(475)": lambda Z: MF.batched(MF.motion_block, Z)}
        log("\n== A. linear probe on y_vel (17-way, chance .06; static-vs-moving chance .5) ==")
        for nm, fn in feats.items():
            t0 = time.time(); Ztr, Zte = fn(Xtf), fn(Xf)
            acc = probe(Ztr, ytr["y_vel"], Zte, yte["y_vel"], 17)
            accd = probe(Ztr[ytr["y_vel"] > 0], ytr["y_dir"][ytr["y_vel"] > 0], Zte[mv], yte["y_dir"][mv], 8)
            accs = probe(Ztr, (ytr["y_vel"] > 0).long(), Zte, (yte["y_vel"] > 0).long(), 2)
            accsh = probe(Ztr, ytr["y_shape"], Zte, yte["y_shape"], 3)
            log(f"  {nm:34s} y_vel {acc:.3f} | dir(8, moving only) {accd:.3f} | moving? {accs:.3f} | shape(3) {accsh:.3f}   [{time.time() - t0:.0f}s]")
        vg = MF.batched(MF.decode_velocity, Xf)
        for k in (0, 1):
            s = mv & (kind == k); e = (vg[s] - tru[s]).norm(dim=1)
            log(f"  population decode (no learning) bg={'photo' if k else 'flat '}: median |v̂-v| {e.median():.2f} px  mean {e.mean():.2f}  cos {F.cosine_similarity(vg[s], tru[s]).mean():.2f}  octant-acc {(torch.tensor([MO._octant(*a) for a in vg[s].tolist()]) == yte['y_dir'][s]).float().mean():.3f}  | static |v̂| med {vg[(~mv) & (kind == k)].norm(dim=1).median():.2f}")
    if "B" in PART:
        log("\n== B. wagon-wheel: population decode on textured fills (stripes/dots), by alias tag ==")
        vg = MF.batched(MF.decode_velocity, Xf); tex = mv & ((yte["y_fill"] == 1) | (yte["y_fill"] == 2))
        for al in (0, 1):
            s = tex & (yte["y_alias"] == al)
            if s.sum() == 0: continue
            e = (vg[s] - tru[s]).norm(dim=1); cs = F.cosine_similarity(vg[s], tru[s])
            log(f"  alias={al} n={int(s.sum())}: median err {e.median():.2f} px  cos {cs.mean():.2f}  frac reversed (cos<0) {(cs < 0).float().mean():.3f}  |v| {tru[s].norm(dim=1).mean():.2f}")
        s = mv & (yte["y_fill"] == 0); e = (vg[s] - tru[s]).norm(dim=1); log(f"  solid fills n={int(s.sum())}: median err {e.median():.2f} px  cos {F.cosine_similarity(vg[s], tru[s]).mean():.2f}  frac reversed {(F.cosine_similarity(vg[s], tru[s]) < 0).float().mean():.3f}")
    if "C" in PART:
        log("\n== C. grouping: colour/Otsu (s053) on the mid frame vs common-fate v3 on the packet; IoU vs exact mask ==")
        mid = Xte.shape[1] // 2; Mmid = Mte[:, mid].bool(); Mswept = Mte.bool().any(1)
        n = min(len(Xte), 600); t0 = time.time()
        gl, fg, _ = GR.groups(Xf[:n, mid]); col = torch.stack([g[0]["mask"] if g else torch.zeros(MO.S, MO.S, dtype=torch.bool) for g in gl])
        mot = motion_group(Xf[:n]); raw = motion_group(Xf[:n], convex=False); log(f"  ({time.time() - t0:.0f}s for {n})")
        for k in (0, 1):
            for nm, s in (("moving", mv[:n] & (kind[:n] == k)), ("static", (~mv[:n]) & (kind[:n] == k))):
                if s.sum() == 0: continue
                log(f"  bg={'photo' if k else 'flat '} {nm:6s} n={int(s.sum()):4d}: colour-Otsu IoU(mid) {iou(col[s], Mmid[:n][s]).mean():.3f} | motion-v3 IoU(mid) {iou(mot[s], Mmid[:n][s]).mean():.3f} (no-hull {iou(raw[s], Mmid[:n][s]).mean():.3f}) | mask area {mot[s].float().mean():.3f} vs true {Mmid[:n][s].float().mean():.3f}")
