"""Moving-shape world (s055, motion arc step 1): the s053 shape generator put IN
TIME.  A packet = T frames (default 6) of 32x32 RGB: one or more exactly-masked
geometric objects translating / rotating / looming over a STATIC background that
is either a flat colour (control) or a real CIFAR-100 photo (scenery: colour
grouping fails there, common fate is the only cue -- the falsification bench).

Per OBJECT (meta[i]["objects"][j]): shape/fill/colour/scale/pose as shapes.py
  plus velocity  dx, dy  px/frame  (speed 0 or in [0.5, SPEED_MAX]),
                 dth     rad/frame (rotation; 0 with prob 1-p_rot),
                 dr      px/frame  (looming/receding; 0 with prob 1-p_loom)
  and the EXACT silhouette mask per frame (`masks` [N,T,S,S] uint8, object 0 only)
Per PACKET labels (`ys`, object 0):
  y_vel   : 0 static | 1..8 direction octant (slow) | 9..16 direction octant (fast)
  y_dir   : direction octant (0..7; -1 when static), y_speed: |v| px/frame
  y_dx, y_dy, y_dth, y_dr : the raw velocities;  y_shape/y_fill/y_hue as before
  y_alias : 1 when a textured fill (stripes/dots, period p) moves > p/2 per frame
            along its own texture normal  -> wagon-wheel regime (kept, not avoided)
  y_bgkind: 0 flat, 1 photo;  y_bgblur: photo background defocus 0 none (flat) / 1 mild sigma .7 / 2 strong 1.5
            (depth of field: the object is in focus, the scenery is not; 50/50 mild/strong)
Frames are stored uint8 (X [N,T,3,S,S]); `as_float(X)` -> [0,1].
build() writes outputs/data/motion/<split>.pt; load(split) reads it.
"""
import json, math, os, pickle, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch.nn.functional as F
from experiments.progenitor import shapes as SH
S = SH.S; T_DEFAULT = 6; SPEED_MAX = 3.0
_xx, _yy = SH._xx, SH._yy
ROOT = SH.ROOT; OUT = os.path.join(ROOT, "outputs", "data", "motion")

def _cifar_bg():
    p = os.path.join(ROOT, "outputs", "data", "cifar-100-python", "train")
    with open(p, "rb") as f: d = pickle.load(f, encoding="latin1")
    return torch.from_numpy(d["data"]).view(-1, 3, S, S)   # uint8 [50000,3,32,32]

def mask_fixed(shape, fill, cx, cy, r, th, shear, flip, thick, tex):
    """shapes._mask with the texture parameters FIXED (tex = (p, ph)) so a fill rides with its body across frames."""
    dx, dy = _xx - cx, _yy - cy
    u = dx * math.cos(th) + dy * math.sin(th); v = -dx * math.sin(th) + dy * math.cos(th)
    u = u + shear * v
    if flip: u = -u
    p, ph = tex
    if shape == 0: d = torch.sqrt(u ** 2 + v ** 2) - r; m = d <= 0
    elif shape == 1: m = (v > -r * 0.6) & (u.abs() < (r * 1.2 - v) * 0.6); d = None
    elif shape == 2: d = torch.maximum(u.abs(), v.abs()) - r; m = d <= 0
    elif shape == 3: return torch.sin(2 * math.pi * u / p) > 0
    else: return (((u % p) - p / 2) ** 2 + ((v % p) - p / 2) ** 2) <= (p * 0.28) ** 2
    if fill == 3:
        if d is None:
            inner = (v > -(r - thick * 1.6) * 0.6) & (u.abs() < ((r - thick * 1.6) * 1.2 - v) * 0.6); return m & ~inner
        return (d <= 0) & (d > -thick)
    if fill == 1: return m & (torch.sin(2 * math.pi * (u * math.cos(ph) + v * math.sin(ph)) / p) > 0)
    if fill == 2: return m & ((((u % p) - p / 2) ** 2 + (((v % p) - p / 2) ** 2)) <= (p * 0.3) ** 2)
    return m

def _octant(dx, dy): return int(((math.atan2(dy, dx) + math.pi / 8) % (2 * math.pi)) // (math.pi / 4))

def sample(n, seed, *, T=T_DEFAULT, maxk=1, bg="photo", p_static=0.15, p_rot=0.3, p_loom=0.2, p_iso=0.1, noise=0.02,
           speed_max=SPEED_MAX, fill_choices=(0, 1, 2, 3), shape_choices=(0, 1, 2), blur=(0.6, 0.4), bgs=None):
    """Returns X [n,T,3,S,S] uint8, masks [n,T,S,S] uint8 (object 0), ys dict, meta list.
    bg: 'flat' | 'photo' | 'mixed' (half/half).  Objects are placed so the trajectory stays >=60 % in frame at both ends."""
    gen = torch.Generator().manual_seed(seed); R = lambda: float(torch.rand(1, generator=gen))
    RI = lambda a, b: int(torch.randint(a, b, (1,), generator=gen))
    if bg != "flat" and bgs is None: bgs = _cifar_bg()
    X = torch.zeros(n, T, 3, S, S, dtype=torch.uint8); M = torch.zeros(n, T, S, S, dtype=torch.uint8); meta = []
    L = lambda: torch.zeros(n, dtype=torch.long); Fz = lambda: torch.zeros(n)
    ys = dict(y_vel=L(), y_dir=L(), y_speed=Fz(), y_dx=Fz(), y_dy=Fz(), y_dth=Fz(), y_dr=Fz(), y_shape=L(), y_fill=L(), y_hue=L(),
              y_iso=L(), y_alias=L(), y_bgkind=L(), y_count=L(), y_scale=Fz(), y_blur=L(), y_bgblur=L())
    for i in range(n):
        kind = 1 if bg == "photo" or (bg == "mixed" and R() < 0.5) else 0
        bgblur = 0
        if kind:   # depth of field (Rocky s056): the eye is focused on the object, the scenery is defocused -- mild or strong, 50/50
            base = bgs[RI(0, len(bgs))].float() / 255; bgblur = 1 if R() < 0.5 else 2; base = SH.gblur(base, SH.BLUR_SIGMA[bgblur]); hb = None; bgc = base.mean((1, 2))
        else: hb, sb, vb = R(), R() * 0.6, R() * 0.6 + 0.3; bgc = SH._hsv(hb, sb, vb); base = bgc.view(3, 1, 1).expand(3, S, S).clone()
        k = RI(1, maxk + 1) if maxk > 1 else 1; bl = 0 if R() < blur[0] else 1
        objs = []
        for j in range(k):
            sh = shape_choices[RI(0, len(shape_choices))]; fl = fill_choices[RI(0, len(fill_choices))]
            r = R() * 8 + 6 if k == 1 else R() * 3 + 4.5   # radius 6..14 (Rocky s056; was 4..13)
            th, shear, flip, thick = R() * 2 * math.pi, R() * 1.0 - 0.5, R() < 0.5, RI(1, 4)
            tex = (R() * 4 + 3, R() * math.pi) if fl == 1 else ((R() * 3 + 4, 0.0) if fl == 2 else (6.0, 0.0))
            static = R() < p_static
            if static: dx = dy = 0.0
            else: sp = R() * (speed_max - 0.5) + 0.5; ang = R() * 2 * math.pi; dx, dy = sp * math.cos(ang), sp * math.sin(ang)
            dth = (R() * 0.5 - 0.25) if (R() < p_rot and sh != 0) else 0.0
            dr = (R() * 1.0 - 0.5) if R() < p_loom else 0.0
            # place so the centre stays within [r*0.4, S - r*0.4] at both ends of the trajectory
            span_x, span_y = dx * (T - 1), dy * (T - 1); lo = 0.4 * r + 1; hi = S - 0.4 * r - 1
            cx0 = R() * max(hi - lo - abs(span_x), 1) + lo + max(-span_x, 0); cy0 = R() * max(hi - lo - abs(span_y), 1) + lo + max(-span_y, 0)
            hf = ((hb if hb is not None else R()) + 0.5 + R() * 0.3 - 0.15) % 1.0; iso = R() < p_iso
            if iso: fg = SH._hsv(hf, 0.9, 1.0); fg = (fg * (SH._lum(bgc) / max(SH._lum(fg), 1e-3))).clamp(0, 1)
            else:
                fg = SH._hsv(hf, R() * 0.8, R())
                if not kind and abs(SH._lum(fg) - SH._lum(bgc)) < 0.25: fg = (1 - bgc).clamp(0, 1)
            # wagon-wheel tag: displacement along the texture normal vs half period
            alias = 0
            if fl in (1, 2) and not static:
                p, ph = tex; nrm = (math.cos(th + ph), math.sin(th + ph)) if fl == 1 else None
                disp = abs(dx * nrm[0] + dy * nrm[1]) if nrm else math.hypot(dx, dy)
                alias = int(disp > p / 2)
            objs.append(dict(shape=sh, name=SH.NAMES[sh], fill=fl, fill_name=SH.FILLS[fl], thick=thick if fl == 3 else 0, hue=round(hf, 3), hue_bin=int(hf * 6) % 6,
                             iso=int(iso), r=round(r, 2), rot=round(th, 3), shear=round(shear, 3), flip=int(flip), cx=round(cx0, 1), cy=round(cy0, 1),
                             dx=round(dx, 3), dy=round(dy, 3), dth=round(dth, 3), dr=round(dr, 3), static=int(static), alias=alias, tex=[round(tex[0], 2), round(tex[1], 3)],
                             _fg=fg, _tex=tex))
        for t in range(T):
            img = base.clone()
            for j, o in enumerate(objs):
                rt = max(o["r"] + o["dr"] * t, 2.0)
                m = mask_fixed(o["shape"], o["fill"], o["cx"] + o["dx"] * t, o["cy"] + o["dy"] * t, rt, o["rot"] + o["dth"] * t, o["shear"], o["flip"], o["thick"] or 1, o["_tex"]).float()
                if j == 0: M[i, t] = (SH._silhouette(o["shape"], o["cx"] + o["dx"] * t, o["cy"] + o["dy"] * t, rt, o["rot"] + o["dth"] * t, o["shear"], o["flip"])).to(torch.uint8)
                lay = torch.cat([m[None], m[None] * o["_fg"].view(3, 1, 1)])
                if bl: lay = SH.gblur(lay, SH.BLUR_SIGMA[1])
                img = img * (1 - lay[0:1]) + lay[1:]
            img = (img + noise * torch.randn(3, S, S, generator=gen)).clamp(0, 1)
            X[i, t] = (img * 255).round().to(torch.uint8)
        o = objs[0]; spd = math.hypot(o["dx"], o["dy"]); d8 = -1 if o["static"] else _octant(o["dx"], o["dy"])
        ys["y_vel"][i] = 0 if o["static"] else 1 + d8 + (8 if spd >= 1.75 else 0)
        ys["y_dir"][i], ys["y_speed"][i], ys["y_dx"][i], ys["y_dy"][i], ys["y_dth"][i], ys["y_dr"][i] = d8, spd, o["dx"], o["dy"], o["dth"], o["dr"]
        ys["y_shape"][i], ys["y_fill"][i], ys["y_hue"][i], ys["y_iso"][i], ys["y_alias"][i], ys["y_bgkind"][i] = o["shape"], o["fill"], o["hue_bin"], o["iso"], o["alias"], kind
        ys["y_count"][i], ys["y_scale"][i], ys["y_blur"][i], ys["y_bgblur"][i] = k, o["r"], bl, bgblur
        for oo in objs: oo.pop("_fg"); oo.pop("_tex")
        meta.append(dict(seed=seed, idx=i, T=T, count=k, bgkind=kind, bgblur=bgblur, blur=bl, objects=objs))
    return X, M, ys, meta

def as_float(X): return X.float() / 255

SPLITS = {
    "train":       dict(n=8000, seed=11, bg="mixed"),
    "test":        dict(n=2000, seed=12, bg="mixed"),
    "test_photo":  dict(n=2000, seed=13, bg="photo"),
    "test_flat":   dict(n=2000, seed=14, bg="flat"),
    "train_multi": dict(n=4000, seed=15, bg="mixed", maxk=2),
    "test_multi":  dict(n=1000, seed=16, bg="mixed", maxk=2),
}
def build(names=None, out=OUT):
    os.makedirs(out, exist_ok=True); bgs = _cifar_bg()
    for nm in names or SPLITS:
        X, M, ys, meta = sample(**SPLITS[nm], bgs=bgs)
        torch.save(dict(X=X, M=M, ys=ys, kwargs=SPLITS[nm]), os.path.join(out, nm + ".pt"))
        with open(os.path.join(out, nm + "_meta.json"), "w") as f: json.dump(meta, f)
        print(nm, tuple(X.shape), "static", float((ys["y_vel"] == 0).float().mean()), "alias", float(ys["y_alias"].float().mean()), flush=True)
def load(nm, out=OUT): return torch.load(os.path.join(out, nm + ".pt"))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build": build(sys.argv[2:] or None)
    else:
        X, M, ys, meta = sample(6, 0, T=4); print(X.shape, M.shape, ys["y_vel"], ys["y_dx"], ys["y_dy"])
