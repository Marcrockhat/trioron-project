"""Shape world (s053): unlimited, fully tagged 32x32 RGB images of simple
geometry with separately controlled nuisance factors, so "never seen" can mean
a held-out factor COMBINATION (compositional test), not just a fresh seed.

Per OBJECT (all recorded in `meta[i]["objects"][j]`):
  shape   : 0 circle, 1 triangle, 2 square, 3 stripes-field, 4 dots-field
            (fields cover the frame; single-object images only)
  fill    : 0 solid, 1 striped, 2 dotted, 3 outline (fields: always 0)
  thick   : outline line thickness 1..3 px (fill=3 only)
  colour  : fg hue h in [0,1), sat, val; hue bin 0..5; iso=1 -> fg has the SAME
            luminance as bg (only chroma carries the object)
  scale   : radius r 3..18 px  (zoom OUT small .. zoom IN overflowing the frame;
            textured fills capped at 13 so a boundary is always visible)
  pose    : rot 0..2pi, shear -0.6..0.6, flip 0/1
  place   : cx, cy;  crop=1 -> centre pushed to a border (r>=6, >=30 % of the
            silhouette stays in frame);  vis = in-frame fraction of the solid
            silhouette (fill-independent, exact)
  blur    : per-object level 0 sharp / 1 mild (sigma .7) / 2 strong (sigma 1.5)
Per IMAGE:
  count k (1..maxk), bg hue/sat/val, noise sigma,
  focus mode: 0 uniform (one blur level for the whole image),
              1 per-object (each object its own level -- depth of field: one
                sharp, another defocused, bg keeps level of object 0),
              2 gradient (sharp at a focal point, blur grows with distance --
                imitating eye focus; level = level at the object's centre)
Image-level label tensors (`ys`): y_shape / y_fill / y_iso / y_hue / y_scale /
  y_rot / y_vis / y_crop / y_blur of object 0; y_count; y_set [N,5] multi-hot;
  y_focus; y_blur_img (max level present).
`sample()` returns (X, ys, meta).  `build()` writes the standard splits to
outputs/data/shapes/*.pt with a manifest.  Held-out (shape,fill) combos:
`exclude=` (train) / `only=` (test).
"""
import json, math, os, torch
import torch.nn.functional as F
S = 32
NAMES = ["circle", "triangle", "square", "stripes", "dots"]
FILLS = ["solid", "striped", "dotted", "outline"]
FOCUS = ["uniform", "per-object", "gradient"]
BLUR_SIGMA = [0.0, 0.7, 1.5]
HELD = {(1, 2), (2, 1), (0, 3)}   # triangle-dotted, square-striped, circle-outline
_yy, _xx = torch.meshgrid(torch.arange(S).float(), torch.arange(S).float(), indexing="ij")

def _hsv(h, s, v):
    i = int(h * 6) % 6; f = h * 6 - int(h * 6); p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return torch.tensor([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i])
def _lum(c): return float(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])

def gblur(img, sigma):   # img [C,S,S] separable gaussian, reflect pad
    if sigma <= 0: return img
    k = int(2 * math.ceil(2 * sigma) + 1); x = torch.arange(k).float() - k // 2; w = torch.exp(-x ** 2 / (2 * sigma ** 2)); w = w / w.sum()
    C = img.shape[0]; p = F.pad(img[None], (k // 2,) * 4, mode="reflect")
    p = F.conv2d(p, w.view(1, 1, 1, k).expand(C, 1, 1, k), groups=C); p = F.conv2d(p, w.view(1, 1, k, 1).expand(C, 1, k, 1), groups=C)
    return p[0]

_BIG = 96
_byy, _bxx = torch.meshgrid(torch.arange(_BIG).float() - 32, torch.arange(_BIG).float() - 32, indexing="ij")
def _silhouette(shape, cx, cy, r, th, shear, flip, big=False):
    """solid mask of a geometric shape; big=True renders on a 96x96 canvas around the frame (unclipped area)."""
    xx, yy = (_bxx, _byy) if big else (_xx, _yy)
    dx, dy = xx - cx, yy - cy
    u = dx * math.cos(th) + dy * math.sin(th); v = -dx * math.sin(th) + dy * math.cos(th)
    u = u + shear * v
    if flip: u = -u
    if shape == 0: return u ** 2 + v ** 2 <= r * r
    if shape == 1: return (v > -r * 0.6) & (u.abs() < (r * 1.2 - v) * 0.6)
    return torch.maximum(u.abs(), v.abs()) <= r
def visible_fraction(shape, cx, cy, r, th, shear, flip):
    return float(_silhouette(shape, cx, cy, r, th, shear, flip).float().sum() / _silhouette(shape, cx, cy, r, th, shear, flip, big=True).float().sum().clamp(min=1))

def _mask(shape, fill, gen, cx, cy, r, th, shear, flip, thick):
    dx, dy = _xx - cx, _yy - cy
    u = dx * math.cos(th) + dy * math.sin(th); v = -dx * math.sin(th) + dy * math.cos(th)
    u = u + shear * v
    if flip: u = -u
    if shape == 0: d = torch.sqrt(u ** 2 + v ** 2) - r; m = d <= 0
    elif shape == 1: m = (v > -r * 0.6) & (u.abs() < (r * 1.2 - v) * 0.6); d = None
    elif shape == 2: d = torch.maximum(u.abs(), v.abs()) - r; m = d <= 0
    elif shape == 3: p = float(torch.rand(1, generator=gen) * 8 + 4); return torch.sin(2 * math.pi * u / p) > 0
    else: p = float(torch.rand(1, generator=gen) * 6 + 5); return (((u % p) - p / 2) ** 2 + ((v % p) - p / 2) ** 2) <= (p * 0.28) ** 2
    if fill == 3:
        if d is None:
            inner = (v > -(r - thick * 1.6) * 0.6) & (u.abs() < ((r - thick * 1.6) * 1.2 - v) * 0.6); return m & ~inner
        return (d <= 0) & (d > -thick)
    if fill == 1:
        p = float(torch.rand(1, generator=gen) * 4 + 3); ph = float(torch.rand(1, generator=gen) * math.pi)
        return m & (torch.sin(2 * math.pi * (u * math.cos(ph) + v * math.sin(ph)) / p) > 0)
    if fill == 2:
        p = float(torch.rand(1, generator=gen) * 3 + 4)
        return m & ((((u % p) - p / 2) ** 2 + (((v % p) - p / 2) ** 2)) <= (p * 0.3) ** 2)
    return m

def sample(n, seed, *, maxk=1, p_iso=0.15, p_crop=0.2, exclude=(), only=None, noise=0.03,
           p_blur=(0.5, 0.3, 0.2), p_focus=(0.6, 0.2, 0.2), fill_choices=(0, 1, 2, 3), shape_choices=(0, 1, 2, 3, 4)):
    """Returns X [n,3,32,32] in [0,1], ys dict of label tensors, meta list of per-image dicts."""
    gen = torch.Generator().manual_seed(seed)
    R = lambda: float(torch.rand(1, generator=gen))
    X = torch.zeros(n, 3, S, S); meta = []
    L = lambda: torch.zeros(n, dtype=torch.long)
    ys = dict(y_shape=L(), y_fill=L(), y_count=L(), y_set=torch.zeros(n, 5), y_iso=L(), y_hue=L(), y_scale=torch.zeros(n),
              y_rot=torch.zeros(n), y_vis=torch.ones(n), y_crop=L(), y_blur=L(), y_focus=L(), y_blur_img=L())
    pairs = [(s, f) for s in shape_choices for f in (fill_choices if s < 3 else (0,))]
    if only is not None: pairs = [p for p in pairs if p in set(only)]
    pairs = [p for p in pairs if p not in set(exclude)]
    pb = torch.tensor(p_blur); pf = torch.tensor(p_focus)
    for i in range(n):
        hb, sb, vb = R(), R() * 0.6, R() * 0.6 + 0.3; bg = _hsv(hb, sb, vb)
        k = int(torch.randint(1, maxk + 1, (1,), generator=gen)) if maxk > 1 else 1
        kp = pairs if k == 1 else [p for p in pairs if p[0] < 3]
        focus = int(torch.multinomial(pf, 1, generator=gen)) if k > 1 or R() < 0.5 else (0 if R() < 0.7 else 2)
        if k == 1 and focus == 1: focus = 0
        base_blur = int(torch.multinomial(pb, 1, generator=gen))
        fx, fy = R() * S, R() * S   # focal point (mode 2)
        cxs = torch.linspace(8, 24, k) if k > 1 else torch.tensor([16.0])
        layers = []; objs = []
        for j in range(k):
            shs = sorted({p[0] for p in kp}); sh = shs[int(torch.randint(len(shs), (1,), generator=gen))]   # shape first (balanced)
            fls = [p[1] for p in kp if p[0] == sh]; fl = fls[int(torch.randint(len(fls), (1,), generator=gen))]
            r = R() * 15 + 3 if k == 1 else R() * 3 + 4
            if fl in (1, 2): r = min(r, 13.0)   # textured fills keep a boundary in frame (else == a field)
            cx = float(cxs[j]) + R() * 4 - 2 + (R() * 8 - 4 if k == 1 else 0); cy = R() * 8 + 12
            th, shear, flip, thick = R() * 2 * math.pi, R() * 1.2 - 0.6, R() < 0.5, int(torch.randint(1, 4, (1,), generator=gen))
            crop = k == 1 and sh < 3 and r >= 6 and R() < p_crop
            vis = 1.0
            if crop:   # centre near a border; pull inward until >= 30 % of the silhouette is in frame
                side = int(torch.randint(4, (1,), generator=gen)); off = R() * 0.6 * r - 0.3 * r
                for _ in range(6):
                    cx2, cy2 = [(-off + 1, cy), (S - 1 + off, cy), (cx, -off + 1), (cx, S - 1 + off)][side]
                    vis = visible_fraction(sh, cx2, cy2, r, th, shear, flip)
                    if vis >= 0.3: break
                    off -= 0.15 * r
                cx, cy = cx2, cy2
            elif sh < 3: vis = visible_fraction(sh, cx, cy, r, th, shear, flip)
            m = _mask(sh, fl, gen, cx, cy, r, th, shear, flip, thick)
            hf = (hb + 0.5 + R() * 0.3 - 0.15) % 1.0; iso = R() < p_iso
            if iso:
                fg = _hsv(hf, 0.9, 1.0); fg = (fg * (_lum(bg) / max(_lum(fg), 1e-3))).clamp(0, 1); sf, vf = 0.9, float(fg.max())
            else:
                sf, vf = R() * 0.8, R(); fg = _hsv(hf, sf, vf)
                if abs(_lum(fg) - _lum(bg)) < 0.25: fg = (1 - bg).clamp(0, 1)
            if focus == 0: bl = base_blur
            elif focus == 1: bl = int(torch.multinomial(pb, 1, generator=gen))
            else: dist = math.hypot(cx - fx, cy - fy); bl = 0 if dist < 8 else (1 if dist < 18 else 2)
            layers.append((m.float(), fg, bl)); objs.append(dict(shape=sh, name=NAMES[sh], fill=fl, fill_name=FILLS[fl], thick=thick if fl == 3 else 0,
                hue=round(hf, 3), hue_bin=int(hf * 6) % 6, sat=round(sf, 3), val=round(vf, 3), iso=int(iso), r=round(r, 2), rot=round(th, 3),
                shear=round(shear, 3), flip=int(flip), cx=round(cx, 1), cy=round(cy, 1), crop=int(crop), vis=round(vis, 3), blur=bl))
        # composite: bg (blur level of object 0 unless gradient) + each object layer blurred at its own level
        img = bg.view(3, 1, 1).expand(3, S, S).clone()
        if focus == 2:   # gradient focus: build sharp composite, then blend three blur levels by distance to focal point
            for m, fg, _ in layers: img = torch.where(m.bool().unsqueeze(0), fg.view(3, 1, 1), img)
            d = torch.sqrt((_xx - fx) ** 2 + (_yy - fy) ** 2); w2 = ((d - 8) / 10).clamp(0, 1); w1 = (d / 8).clamp(0, 1)
            b1, b2 = gblur(img, BLUR_SIGMA[1]), gblur(img, BLUR_SIGMA[2])
            img = img * (1 - w1) + (b1 * (1 - w2) + b2 * w2) * w1
        else:
            img = gblur(img, BLUR_SIGMA[layers[0][2] if focus == 0 else base_blur])
            for m, fg, bl in layers:
                lay = gblur(torch.cat([m[None], m[None] * fg.view(3, 1, 1)]), BLUR_SIGMA[bl]); a = lay[0:1]
                img = img * (1 - a) + lay[1:]
        img = (img + noise * torch.randn(3, S, S, generator=gen)).clamp(0, 1)
        X[i] = img
        o = objs[0]
        ys["y_shape"][i], ys["y_fill"][i], ys["y_iso"][i], ys["y_hue"][i] = o["shape"], o["fill"], o["iso"], o["hue_bin"]
        ys["y_scale"][i], ys["y_rot"][i], ys["y_vis"][i], ys["y_crop"][i], ys["y_blur"][i] = o["r"], o["rot"], o["vis"], o["crop"], o["blur"]
        ys["y_count"][i] = k; ys["y_focus"][i] = focus; ys["y_blur_img"][i] = max(oo["blur"] for oo in objs)
        for oo in objs: ys["y_set"][i, oo["shape"]] = 1
        meta.append(dict(seed=seed, idx=i, count=k, focus=focus, focus_name=FOCUS[focus], bg=dict(hue=round(hb, 3), sat=round(sb, 3), val=round(vb, 3)),
                         focal=(round(fx, 1), round(fy, 1)) if focus == 2 else None, noise=noise, objects=objs))
    return X, ys, meta

SPLITS = {   # name: kwargs  (n and seed fixed so every PC builds the same set)
    "train":        dict(n=20000, seed=1, exclude=HELD),
    "test_fresh":   dict(n=5000, seed=2, exclude=HELD),
    "test_held":    dict(n=3000, seed=3, only=HELD),
    "test_stress":  dict(n=12000, seed=4, exclude=HELD, p_crop=0.3, p_iso=0.3, p_blur=(0.34, 0.33, 0.33)),
    "train_multi":  dict(n=20000, seed=6, maxk=3, exclude=HELD),
    "test_multi":   dict(n=4000, seed=5, maxk=3, exclude=HELD),
}
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "outputs", "data", "shapes")

def build(names=None, out=OUT):
    import time
    os.makedirs(out, exist_ok=True); man = {}
    for nm in (names or SPLITS):
        kw = dict(SPLITS[nm]); n = kw.pop("n"); seed = kw.pop("seed"); t = time.time()
        X, ys, meta = sample(n, seed, **kw)
        torch.save(dict(X=(X * 255).round().to(torch.uint8), ys=ys, meta=meta, kwargs=dict(n=n, seed=seed, **{k: sorted(v) if isinstance(v, set) else v for k, v in kw.items()})), os.path.join(out, nm + ".pt"))
        man[nm] = dict(n=n, seed=seed, shape_hist=torch.bincount(ys["y_shape"], minlength=5).tolist(), fill_hist=torch.bincount(ys["y_fill"], minlength=4).tolist(),
                       blur_hist=torch.bincount(ys["y_blur_img"], minlength=3).tolist(), focus_hist=torch.bincount(ys["y_focus"], minlength=3).tolist(),
                       crop=int(ys["y_crop"].sum()), iso=int(ys["y_iso"].sum()), count_hist=torch.bincount(ys["y_count"], minlength=4).tolist()[1:], secs=round(time.time() - t))
        print(nm, man[nm], flush=True)
    json.dump(dict(names=NAMES, fills=FILLS, focus=FOCUS, blur_sigma=BLUR_SIGMA, held=sorted(HELD), splits=man), open(os.path.join(out, "manifest.json"), "w"), indent=1)

def load(nm, out=OUT):
    d = torch.load(os.path.join(out, nm + ".pt"), weights_only=False); return d["X"].float() / 255, d["ys"], d["meta"]

if __name__ == "__main__":
    import sys, torchvision
    if len(sys.argv) > 1 and sys.argv[1] == "build": build(sys.argv[2:] or None); sys.exit()
    X, ys, meta = sample(64, 0, maxk=3, p_focus=(0.2, 0.4, 0.4))
    torchvision.utils.save_image(X, os.path.join(ROOT, "outputs", "shapes_sample.png"), nrow=8)
    print(json.dumps(meta[1], indent=None)); print("focus", ys["y_focus"][:16].tolist(), "blur", ys["y_blur"][:16].tolist())
