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

def _chans(Z):   # (Y, 2RG, 2BY) -- chroma up-weighted as in grouping.foreground
    y = 0.299 * Z[..., 0, :, :] + 0.587 * Z[..., 1, :, :] + 0.114 * Z[..., 2, :, :]; rg = Z[..., 0, :, :] - Z[..., 1, :, :]; by = Z[..., 2, :, :] - 0.5 * (Z[..., 0, :, :] + Z[..., 1, :, :])
    return torch.stack([y, 2 * rg, 2 * by], -3)
def motion_gate(Xf, thr=1.4):
    """packet-level 'is anything moving': 99th-pct / median of the colour temporal-difference energy.
    Truly static packets (no translation/rotation/loom) max out at 1.32, movers start at 1.31 (test, n=600)."""
    C = _chans(Xf); e = F.avg_pool2d(torch.sqrt(((C[:, 1:] - C[:, :-1]) ** 2).sum(2)).amax(1)[:, None], 3, 1, 1)[:, 0]
    return (e.flatten(1).quantile(0.99, 1) / e.flatten(1).median(1).values) > thr

def motion_group(Xf, frame=None, convex=True, colour_refine=True):
    """common-fate grouping v4 (single dominant body), s056.  The mid-frame body from a T-frame packet:
    (1) motion_gate: nothing moves -> empty mask (the v3 per-packet noise floor is gone -- it threw away
        weak movers; the gate is a GLOBAL decision, the threshold inside is Otsu only);
    (2) endpoint differences: pixels where the mid frame differs from the FIRST frame form two bands of
        equal thickness 2.5|v| -- the body's leading band (inside) and the revealed background behind it
        (outside); likewise vs the LAST frame (trailing band inside / about-to-be-covered ahead outside).
        A single pixel's time series cannot tell these apart (mirror images); the MOTION DIRECTION can:
        project on v̂ (population decode) and keep the half AHEAD for changed-vs-first, the half BEHIND
        for changed-vs-last; pixels changed vs both ends are body (thin/fast bodies);
    (3) colour refinement: among all changed pixels keep those closer to the body colour than to the
        complementary band's colour (helps flat bg +7 pp, neutral on photos);
    (4) s053 body pipeline: closing R=4 + fill + largest + convex hull bridges the core that never
        uncovers within the packet (a large body moving slowly hides its own interior).
    v3 (temporal-difference edges + lock-in baseline) reached IoU .47/.47; v4 .73 flat / .66 photo.
    -> bool [N,S,S]."""
    N, T = Xf.shape[:2]; t = T // 2 if frame is None else frame; C = _chans(Xf); sm = lambda z: F.avg_pool2d(z[:, None], 3, 1, 1)[:, 0]
    d0 = sm(torch.sqrt(((C[:, t] - C[:, 0]) ** 2).sum(1))); d5 = sm(torch.sqrt(((C[:, t] - C[:, T - 1]) ** 2).sum(1)))
    gate = motion_gate(Xf); vh = MF.batched(MF.decode_velocity, Xf); vh = vh / (vh.norm(dim=1, keepdim=True) + 1e-6)
    yy, xx = MO._yy, MO._xx; out = torch.zeros(N, MO.S, MO.S, dtype=torch.bool)
    for i in range(N):
        if not gate[i]: continue
        dd = torch.maximum(d0[i], d5[i]); th = GR._otsu(dd.flatten()); c0, c5 = d0[i] > th, d5[i] > th
        g1, g2 = c0 & ~c5, c5 & ~c0; p = xx * vh[i, 0] + yy * vh[i, 1]; body = c0 & c5
        if g1.any(): body |= g1 & (p > p[g1].median())
        if g2.any(): body |= g2 & (p < p[g2].median())
        changed = c0 | c5
        if colour_refine and body.sum() > 3 and (changed & ~body).any():
            cm = C[i, t]; bcol = cm[:, body].mean(1); ocol = cm[:, changed & ~body].mean(1)
            body = changed & (((cm - bcol[:, None, None]) ** 2).sum(0) < ((cm - ocol[:, None, None]) ** 2).sum(0))
        m = body.numpy()
        if m.sum() < GR.MIN_AREA: continue
        cl = ndi.binary_fill_holes(ndi.binary_closing(m, structure=GR._disk(GR.R_CLOSE), border_value=0) | m)
        lab, k = ndi.label(cl); sz = np.bincount(lab.ravel())[1:]; b = lab == (1 + sz.argmax())
        out[i] = torch.from_numpy(np.asarray(GR.convex_fill(b), bool) if convex else b)
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
            for nm, s in (("moving", mv[:n] & (kind[:n] == k)), ("static", (~mv[:n]) & (yte["y_dth"][:n] == 0) & (yte["y_dr"][:n] == 0) & (kind[:n] == k))):
                if s.sum() == 0: continue
                log(f"  bg={'photo' if k else 'flat '} {nm:6s} n={int(s.sum()):4d}: colour-Otsu IoU(mid) {iou(col[s], Mmid[:n][s]).mean():.3f} | motion-v4 IoU(mid) {iou(mot[s], Mmid[:n][s]).mean():.3f} (no-hull {iou(raw[s], Mmid[:n][s]).mean():.3f}) | mask area {mot[s].float().mean():.3f} vs true {Mmid[:n][s].float().mean():.3f}")
