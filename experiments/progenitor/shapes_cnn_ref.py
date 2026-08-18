"""Dataset-ceiling reference (NOT trioron, OVER CAP): small CNN on raw pixels, 8 epochs.
Tells us how much of the fixed-front-end gap is label noise vs missing primitives."""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import shapes as SH
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6"))); torch.manual_seed(0)
Xtr, ytr, _ = SH.load("train"); Xfr, yfr, _ = SH.load("test_fresh"); Xho, yho, _ = SH.load("test_held"); Xst, yst, _ = SH.load("test_stress")
def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
net = nn.Sequential(blk(3, 32), nn.MaxPool2d(2), blk(32, 64), nn.MaxPool2d(2), blk(64, 128), nn.MaxPool2d(2), blk(128, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 5))
EP = int(os.environ.get("EPOCHS", "30"))
print("params", sum(p.numel() for p in net.parameters())); opt = torch.optim.Adam(net.parameters(), 2e-3); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EP * ((len(Xtr) + 127) // 128))
for ep in range(EP):
    net.train()
    for idx in torch.randperm(len(Xtr)).split(128): opt.zero_grad(); F.cross_entropy(net(Xtr[idx]), ytr["y_shape"][idx]).backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        pf = torch.cat([net(Xfr[i:i+1000]).argmax(1) for i in range(0, 5000, 1000)]); ph = torch.cat([net(Xho[i:i+1000]).argmax(1) for i in range(0, 3000, 1000)])
        ps = torch.cat([net(Xst[i:i+1000]).argmax(1) for i in range(0, 12000, 1000)])
    geo = yfr["y_shape"] < 3; nc = yst["y_crop"] == 0
    print(f"ep{ep} fresh {float((pf==yfr['y_shape']).float().mean()):.3f} geometric-3way(argmax over 5) {float((pf[geo]==yfr['y_shape'][geo]).float().mean()):.3f} held {float((ph==yho['y_shape']).float().mean()):.3f} "
          f"small {float((ps[(yst['y_scale']<5)&nc&(yst['y_shape']<3)]==yst['y_shape'][(yst['y_scale']<5)&nc&(yst['y_shape']<3)]).float().mean()):.3f} cropped {float((ps[yst['y_crop']==1]==yst['y_shape'][yst['y_crop']==1]).float().mean()):.3f} iso {float((ps[(yst['y_iso']==1)&nc]==yst['y_shape'][(yst['y_iso']==1)&nc]).float().mean()):.3f} blur2 {float((ps[(yst['y_blur']==2)&nc]==yst['y_shape'][(yst['y_blur']==2)&nc]).float().mean()):.3f}", flush=True)
