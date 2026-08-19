"""Grouping primitive (s053): figure/ground + object split + boundary/interior split,
BEFORE any descriptor.  No labels, no learned parameters.

  bg colour   : median of the frame's border pixels in (Y, RG, BY)
  distance    : per-pixel weighted colour distance to bg (chroma weighted up so
                iso-luminant objects survive)
  foreground  : Otsu threshold on the distance map (adaptive per image)
  bodies      : morphological CLOSING (radius R_CLOSE) bridges the stripes / dots of a
                textured fill into one body, then connected components; components
                smaller than MIN_AREA are dropped; a body covering > FIELD_FRAC of the
                frame is a FIELD (stripes-/dots-field), else an OBJECT
  per group   : silhouette (closed body mask), boundary ring (silhouette - eroded),
                interior (eroded), colour mean over the raw fg pixels, second-moment
                frame (area, centroid, sqrt-eigen scales, orientation, elongation),
                border-touch flag (crop), count = number of objects
Descriptors are computed on the SEPARATED streams:
  silhouette_desc : boundary-orientation block of the silhouette image (shape only)
  interior_desc   : cepstral texture of luminance * interior mask (fill only)
returned by `describe(X)` as a dict of [N, d] tensors for the LARGEST object.
"""
import math, torch, numpy as np
import torch.nn.functional as F
from scipy import ndimage as ndi
from experiments.progenitor import frontend as FE
S = 32; R_CLOSE = 4; MIN_AREA = 6; FIELD_SPAN = 0.85; OTSU_W = 1.0; SECOND_PASS = True; SECOND_MIN = 1.1

def _otsu(d):   # d [P] flat distances -> threshold
    h = torch.histc(d, 64, float(d.min()), float(d.max()) + 1e-6); e = torch.linspace(float(d.min()), float(d.max()) + 1e-6, 65)
    w0 = h.cumsum(0); w1 = w0[-1] - w0; c = (e[:-1] + e[1:]) / 2; m0 = (h * c).cumsum(0) / w0.clamp(min=1); m1 = ((h * c).sum() - (h * c).cumsum(0)) / w1.clamp(min=1)
    v = w0 * w1 * (m0 - m1) ** 2; return float(c[int(v.argmax())])

def foreground(X):
    """X [N,3,S,S] -> fg mask [N,S,S] bool, distance map [N,S,S]"""
    y = FE.Y(X); rg = X[:, 0] - X[:, 1]; by = X[:, 2] - 0.5 * (X[:, 0] + X[:, 1]); C = torch.stack([y, 2.0 * rg, 2.0 * by], 1)   # chroma up-weighted
    border = torch.cat([C[:, :, 0, :], C[:, :, -1, :], C[:, :, :, 0], C[:, :, :, -1]], 2)   # [N,3,4S]
    bg = border.median(2).values                                                            # [N,3]
    d = torch.sqrt(((C - bg[:, :, None, None]) ** 2).sum(1))                                # [N,S,S]
    d = F.avg_pool2d(d[:, None], 3, 1, 1)[:, 0]                                            # denoise
    bd = torch.cat([d[:, 0, :], d[:, -1, :], d[:, :, 0], d[:, :, -1]], 1).quantile(0.10, dim=1)   # noise floor: bg-side border quantile
    fgs = []
    for i, di in enumerate(d):
        floor = max(0.08, 2.5 * float(bd[i])); t1 = max(OTSU_W * _otsu(di.flatten()), floor); m = di > t1
        if SECOND_PASS:   # a second, lower-contrast layer (e.g. an iso-luminant object next to a high-contrast one)
            rest = di[~F.max_pool2d(m[None, None].float(), 5, 1, 2)[0, 0].bool()]
            if rest.numel() > 50:
                t2 = _otsu(rest)
                if floor * SECOND_MIN <= t2 < t1:
                    m2 = (di > t2) & ~m
                    if 0 < float(m2.float().mean()) < 0.25: m = m | m2
        fgs.append(m)
    return torch.stack(fgs), d

_disk = lambda r: (lambda a: (a[:, None] ** 2 + a[None, :] ** 2 <= r * r))(torch.arange(-r, r + 1).float()).numpy()

def _comp_stats(sil):
    yy, xx = np.nonzero(sil); area = len(yy)
    if area < 3: return area, 1.0, 0.0, 0.5
    cov = np.cov(np.stack([xx - xx.mean(), yy - yy.mean()])) + 1e-6 * np.eye(2); ev, evec = np.linalg.eigh(cov)
    return area, float(math.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6))), math.atan2(evec[1, 1], evec[0, 1]) % math.pi, float(math.sqrt(max(ev[0], 1e-6)))

def bodies_v2(m, r_bridge=R_CLOSE, small=30, elong=3.0, thin=1.5):
    """raw fg mask -> list of body masks. Merge two components within r_bridge only if the smaller
    is a texture element (area <= small) or both are parallel elongated stripes; then close + fill
    holes INSIDE each merged body (per-body bridging: neighbours never merge)."""
    lab, n = ndi.label(m); sizes = np.bincount(lab.ravel())[1:]
    ids = [g for g in range(1, n + 1) if sizes[g - 1] >= 3]
    if not ids: return []
    disk = _disk(r_bridge); comps = {g: lab == g for g in ids}; st = {g: _comp_stats(comps[g]) for g in ids}
    dil = {g: ndi.binary_dilation(comps[g], structure=disk) for g in ids}
    parent = {g: g for g in ids}
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if not (dil[a] & comps[b]).any(): continue
            (aa, ea, ta, wa), (ab, eb, tb, wb) = st[a], st[b]
            texture = min(aa, ab) <= small
            parallel = abs(((ta - tb + math.pi / 2) % math.pi) - math.pi / 2) < 0.35
            stripes = (ea >= elong and eb >= elong and parallel) or (wa <= thin and wb <= thin)
            if texture or stripes: parent[find(a)] = find(b)
    groups_ = {}
    for g in ids: groups_.setdefault(find(g), []).append(g)
    out = []
    for members in groups_.values():
        u = np.zeros_like(m)
        for g in members: u |= comps[g]
        body = ndi.binary_fill_holes(ndi.binary_closing(u, structure=disk, border_value=0) | u)
        out.append(body)
    return out

def groups(X, version=2):
    """-> list (per image) of list of dicts: mask (silhouette bool [S,S]), interior, boundary, area, is_field, touches, colour, frame"""
    fg, d = foreground(X); out = []
    disk = _disk(R_CLOSE); erode_k = _disk(1)
    for i in range(len(X)):
        m = fg[i].numpy()
        # FIELD: repeated texture spread over the frame
        rlab, rn = ndi.label(m); sizes = np.bincount(rlab.ravel())[1:]; big = np.isin(rlab, 1 + np.nonzero(sizes >= 4)[0]); ys_, xs_ = np.nonzero(big)
        if int((sizes >= 4).sum()) >= 3 and len(ys_) and (ys_.max() - ys_.min()) >= FIELD_SPAN * S and (xs_.max() - xs_.min()) >= FIELD_SPAN * S:
            full = np.ones((S, S), bool)
            out.append([dict(mask=torch.from_numpy(full), interior=torch.from_numpy(full), boundary=torch.zeros(S, S, dtype=torch.bool), rawfg=torch.from_numpy(m), area=S * S,
                             is_field=True, touches=True, colour=X[i][:, torch.from_numpy(m)].mean(1), frame=torch.tensor([S / 2, S / 2, S / 3, S / 3, 0.0, 1.0, m.mean()]))]); continue
        if version == 2: bods = bodies_v2(m)
        else:
            closed = ndi.binary_fill_holes(ndi.binary_closing(m, structure=disk, border_value=0) | m); lab, n = ndi.label(closed); bods = [lab == g for g in range(1, n + 1)]
        gs = []
        for sil in bods:
            area = int(sil.sum())
            if area < MIN_AREA: continue
            inter = ndi.binary_erosion(sil, structure=erode_k, border_value=0); bnd = sil & ~inter
            yy, xx = np.nonzero(sil); cy, cx = yy.mean(), xx.mean(); cov = np.cov(np.stack([xx - cx, yy - cy])) if area > 2 else np.eye(2)
            ev, evec = np.linalg.eigh(cov + 1e-6 * np.eye(2)); sc = np.sqrt(np.maximum(ev, 1e-6)); ang = math.atan2(evec[1, 1], evec[0, 1]) % math.pi
            raw = m & sil; col = X[i][:, torch.from_numpy(raw)].mean(1) if raw.any() else X[i][:, torch.from_numpy(sil)].mean(1)
            gs.append(dict(mask=torch.from_numpy(sil), interior=torch.from_numpy(inter), boundary=torch.from_numpy(bnd), rawfg=torch.from_numpy(raw), area=area,
                           is_field=False, touches=bool(sil[0].any() or sil[-1].any() or sil[:, 0].any() or sil[:, -1].any()),
                           colour=col, frame=torch.tensor([cx, cy, sc[1], sc[0], ang, sc[1] / max(sc[0], 1e-3), raw.sum() / max(area, 1)], dtype=torch.float)))
        gs.sort(key=lambda g: -g["area"]); out.append(gs)
    return out, fg, d

def canon_mask(mask, margin=2, size=S):
    """scale canonicalisation: crop the mask's bbox (+margin) and resize to size x size (the fovea's zoom)."""
    yy, xx = torch.nonzero(mask, as_tuple=True)
    if len(yy) == 0: return mask.float()
    y0, y1 = max(int(yy.min()) - margin, 0), min(int(yy.max()) + margin + 1, S); x0, x1 = max(int(xx.min()) - margin, 0), min(int(xx.max()) + margin + 1, S)
    side = max(y1 - y0, x1 - x0); cy, cx = (y0 + y1) // 2, (x0 + x1) // 2   # square crop, aspect preserved
    y0, x0 = max(cy - side // 2, 0), max(cx - side // 2, 0); y1, x1 = min(y0 + side, S), min(x0 + side, S)
    crop = mask[y0:y1, x0:x1].float()[None, None]
    return F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)[0, 0]

def canon_affine(mask, rho=2.6, size=S):
    """rotation/shear/scale canonicalisation: whiten the mask by its second moments (C^-1/2) so
    shear and anisotropic scale vanish (ellipse -> circle, parallelogram -> square, sheared
    triangle -> equilateral); rho std-units map to the frame edge. Residual rotation is left
    to the boundary block's rotation-invariant angular spectrum."""
    yy, xx = torch.nonzero(mask, as_tuple=True)
    if len(yy) < 3: return mask.float()
    pts = torch.stack([xx.float(), yy.float()]); c = pts.mean(1); C = torch.cov(pts) + 0.25 * torch.eye(2)   # +pixel-quantisation cov
    ev, V = torch.linalg.eigh(C); Ch = V @ torch.diag(ev.clamp(min=1e-3).sqrt()) @ V.T                    # C^{1/2}
    theta = torch.cat([Ch * rho / (S / 2), ((c - (S - 1) / 2) / (S / 2))[:, None]], 1)[None]              # out-normalised -> in-normalised
    grid = F.affine_grid(theta, (1, 1, size, size), align_corners=False)
    return F.grid_sample(mask.float()[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]

def _canon(mask, mode):
    if mode in (True, "scale", 1): return canon_mask(mask)
    if mode in ("affine", 2): return canon_affine(mask)
    return mask.float()

def describe(X, gl=None, canon=False):
    """Descriptors of the LARGEST group per image (zeros if none). Returns dict of [N,d] tensors + the group lists."""
    if gl is None: gl, _, _ = groups(X)
    N = len(X); sil_img = torch.zeros(N, 3, S, S); int_img = torch.zeros(N, 3, S, S); frame = torch.zeros(N, 7); col = torch.zeros(N, 3)
    flags = torch.zeros(N, 4)   # n_objects, has_field, touches, fill fraction (rawfg / area)
    for i, gs in enumerate(gl):
        objs = [g for g in gs if not g["is_field"]]; flags[i, 0] = len(objs); flags[i, 1] = float(any(g["is_field"] for g in gs))
        if not gs: continue
        g = gs[0]; sil_img[i] = _canon(g["mask"], canon); int_img[i] = FE.Y(X[i:i + 1])[0] * g["interior"].float() + (~g["interior"]).float() * FE.Y(X[i:i + 1])[0][g["interior"]].mean() if g["interior"].any() else 0
        frame[i] = g["frame"]; col[i] = g["colour"]; flags[i, 2] = float(g["touches"]); flags[i, 3] = g["frame"][6]
    return dict(silhouette=FE.boundary_block(sil_img), interior=FE.dense_pooled(int_img), colour=torch.cat([col, FE.colour_block(X)[:, :0]], 1), frame=frame, flags=flags), gl

def describe_groups(X, gl=None, max_groups=4, canon=False):
    """Per-GROUP descriptors: list (per image) of dict(silhouette [n,92], colour [n,3], frame [n,7], flags [n,4], is_field [n]) for
    the up-to-max_groups largest groups. flags mimic the single-object context ([1, is_field, touches, fillfrac]) so a leaf trained
    on describe() streams can read each group."""
    if gl is None: gl, _, _ = groups(X)
    out = []
    for i, gs in enumerate(gl):
        gs = gs[:max_groups]
        if not gs: out.append(None); continue
        sil = torch.stack([_canon(g["mask"], canon) for g in gs])[:, None].expand(-1, 3, -1, -1)
        out.append(dict(silhouette=FE.boundary_block(sil), colour=torch.stack([g["colour"] for g in gs]), frame=torch.stack([g["frame"] for g in gs]),
                        flags=torch.tensor([[1.0, float(g["is_field"]), float(g["touches"]), float(g["frame"][6])] for g in gs], dtype=torch.float), is_field=torch.tensor([g["is_field"] for g in gs])))
    return out
