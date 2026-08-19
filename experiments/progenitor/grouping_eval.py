"""Grouping sanity bars on ground truth: IoU per fill / stress slice on single-object draws,
count exactness on multi-object draws. python3 experiments/progenitor/grouping_eval.py [version]"""
import os, sys, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import shapes as SH, grouping as G
V = int(sys.argv[1]) if len(sys.argv) > 1 else 2
X, ys, meta = SH.sample(800, 11); t = time.time(); gl, fg, d = G.groups(X, version=V); print(f"v{V}: {(time.time()-t)/800*1000:.1f} ms/img")
geo = [(i, m) for i, m in enumerate(meta) if m["objects"][0]["shape"] < 3]; fields = [(i, m) for i, m in enumerate(meta) if m["objects"][0]["shape"] >= 3]
def iou(o, gs):
    if not gs or gs[0]["is_field"]: return 0.0
    gt = SH._silhouette(o["shape"], o["cx"], o["cy"], o["r"], o["rot"], o["shear"], o["flip"]); pm = gs[0]["mask"]
    return float((pm & gt).sum() / max(int((pm | gt).sum()), 1))
ious = torch.tensor([iou(m["objects"][0], gl[i]) for i, m in geo])
print(f"  single-object geometric n={len(geo)}: IoU {ious.mean():.3f}  IoU>.5 {float((ious>0.5).float().mean()):.3f}  exactly-1-object {np.mean([len([g for g in gl[i] if not g['is_field']])==1 for i,m in geo]):.3f}  wrongly-field {np.mean([bool(gl[i]) and gl[i][0]['is_field'] for i,m in geo]):.3f} | fields flagged {np.mean([bool(gl[i]) and gl[i][0]['is_field'] for i,m in fields]):.3f}")
def sub(pred): idx = [j for j, (i, m) in enumerate(geo) if pred(m["objects"][0])]; return f"{ious[idx].mean():.3f}"
print("  IoU by fill: " + ", ".join(f"{SH.FILLS[fl]} {sub(lambda o, fl=fl: o['fill']==fl)}" for fl in range(4)) + f" | iso {sub(lambda o: o['iso'])} blur2 {sub(lambda o: o['blur']==2)} small {sub(lambda o: o['r']<5)} cropped {sub(lambda o: o['crop'])}")
Xm, ym, mm = SH.sample(800, 12, maxk=3); glm, _, _ = G.groups(Xm, version=V)
n = torch.tensor([len([g for g in gs if not g["is_field"]]) for gs in glm]).clamp(max=3)
print(f"  multi-object count exact {float((n==ym['y_count']).float().mean()):.3f}; per true count " + ", ".join(f"{k}:{float((n[ym['y_count']==k]==k).float().mean()):.2f}" for k in (1, 2, 3)) + f"; mean |err| {float((n-ym['y_count']).abs().float().mean()):.2f}")
# per-object IoU on multi: match each gt object to best group
def obj_gt(o): return SH._silhouette(o["shape"], o["cx"], o["cy"], o["r"], o["rot"], o["shear"], o["flip"])
best = []
for i, m in enumerate(mm):
    for o in m["objects"]:
        gt = obj_gt(o); best.append(max([float((g["mask"] & gt).sum() / max(int((g["mask"] | gt).sum()), 1)) for g in glm[i]] or [0.0]))
print(f"  multi per-object best-match IoU {np.mean(best):.3f}  (>0.5: {np.mean(np.array(best)>0.5):.3f})")
