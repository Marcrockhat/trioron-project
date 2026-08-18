"""Labelled contact sheet of a split: python3 experiments/progenitor/shapes_sheet.py [split] [n]"""
import os, sys, torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import shapes as SH
split = sys.argv[1] if len(sys.argv) > 1 else "train"; n = int(sys.argv[2]) if len(sys.argv) > 2 else 48
X, ys, meta = SH.load(split); X = X[:n]; cols = 6; sc = 3; W, H = 32 * sc, 32 * sc + 34
sheet = Image.new("RGB", (cols * (W + 4), ((n + cols - 1) // cols) * (H + 4)), (20, 20, 20)); d = ImageDraw.Draw(sheet)
for i in range(n):
    im = Image.fromarray((X[i].permute(1, 2, 0).numpy() * 255).astype("uint8")).resize((W, W), Image.NEAREST)
    x0, y0 = (i % cols) * (W + 4), (i // cols) * (H + 4); sheet.paste(im, (x0, y0))
    m = meta[i]; o = m["objects"][0]
    t1 = f"{o['name'][:4]}/{o['fill_name'][:4]} r{o['r']:.0f} sh{o['shear']:+.1f}" + (" ISO" if o["iso"] else "") + (" CROP" if o["crop"] else "")
    t2 = f"blur{o['blur']} {m['focus_name'][:4]} k{m['count']}" + (" +" + ",".join(oo["name"][:3] for oo in m["objects"][1:]) if m["count"] > 1 else "")
    d.text((x0 + 2, y0 + W + 1), t1, fill=(230, 230, 230)); d.text((x0 + 2, y0 + W + 17), t2, fill=(180, 220, 180))
out = os.path.join(SH.ROOT, "outputs", f"shapes_sheet_{split}.png"); sheet.save(out); print(out)
