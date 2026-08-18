"""Primitive probe: which coordinate-free / motion primitive carries CIFAR form?
Same supervised 48-cell quad leaf, 25 classes (5 superclasses), 8 epochs."""
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import cifar_eye_nest as B
torch.manual_seed(0)
Xtr,ytr=B.load_cifar100(True); Xte,yte=B.load_cifar100(False); f2c=B.fine_to_coarse()
T=5; cls=sum([(f2c==t).nonzero().squeeze(1).tolist() for t in range(T)],[])
mtr=torch.isin(ytr,torch.tensor(cls)); mte=torch.isin(yte,torch.tensor(cls))
Xtr,ytr,Xte,yte=Xtr[mtr],ytr[mtr],Xte[mte],yte[mte]
def Y(x): return 0.299*x[:,0]+0.587*x[:,1]+0.114*x[:,2]        # [N,32,32]

# (i) eye DoG at centre fixation
eye=B.Eye((16,16),rectify=False)
def eye_feats(X,eye=eye): return torch.cat([eye.sense(X[i:i+2000]) for i in range(0,len(X),2000)])
# (ii) fly: saccade differences — same DoG receptors displaced ±1 px right / down
eyeR=B.Eye((16,17),rectify=False); eyeD=B.Eye((17,16),rectify=False)
def fly_feats(X):
    f0=eye_feats(X); return torch.cat([f0, eye_feats(X,eyeR)-f0, eye_feats(X,eyeD)-f0],1)
# (iii) wave templates: log-polar power spectrum per window (fovea 8x8 / parafovea 16x16 / field 32x32), luminance
def spec_bins(img, nr=6, nt=8):
    # img [N,h,w] -> radial x orientation bins of |FFT|^2 (DC dropped), log
    N,h,w=img.shape
    F=torch.fft.fftshift(torch.fft.fft2(img - img.mean((1,2),keepdim=True)),dim=(-2,-1)).abs()**2
    yy,xx=torch.meshgrid(torch.arange(h)-h//2, torch.arange(w)-w//2, indexing="ij")
    r=torch.sqrt(yy**2+xx**2).float(); th=torch.atan2(yy.float(),xx.float())%math.pi
    rb=torch.clamp((r/(h/2)*nr).long(),max=nr-1); tb=torch.clamp((th/math.pi*nt).long(),max=nt-1)
    idx=(rb*nt+tb).flatten(); mask=(r>0).flatten()
    out=torch.zeros(N,nr*nt).index_add_(1, idx[mask], F.flatten(1)[:,mask])
    return torch.log1p(out)
def wave_feats(X):
    y=Y(X); out=[]
    for i in range(0,len(X),2000):
        yb=y[i:i+2000]
        out.append(torch.cat([spec_bins(yb[:,12:20,12:20],4,8), spec_bins(yb[:,8:24,8:24],6,8), spec_bins(yb,8,8),
                              # chroma spectra coarse
                              spec_bins((X[i:i+2000,0]-X[i:i+2000,1])[:,8:24,8:24],4,4)],1))
    return torch.cat(out)
# (iv) delta-collapse: structure function D(s)=mean (I(x)-I(x+s))^2 over the window, shifts s on a grid, per window
def collapse_feats(X, shifts=range(-6,7,2)):
    y=Y(X); out=[]
    for i in range(0,len(X),2000):
        yb=y[i:i+2000]; feats=[]
        for (r0,r1) in ((8,24),(0,32)):
            w=yb[:,r0:r1,r0:r1]; w=w-w.mean((1,2),keepdim=True)
            for dr in shifts:
                for dc in shifts:
                    if dr==0 and dc==0: continue
                    a=w[:,max(dr,0):w.shape[1]+min(dr,0), max(dc,0):w.shape[2]+min(dc,0)]
                    b=w[:,max(-dr,0):w.shape[1]+min(-dr,0), max(-dc,0):w.shape[2]+min(-dc,0)]
                    feats.append(((a-b)**2).mean((1,2)))
        out.append(torch.stack(feats,1))
    return torch.log1p(torch.cat(out))
import torch.nn.functional as F
def blur(X): return F.interpolate(F.avg_pool2d(X,2), scale_factor=2, mode="bilinear", align_corners=False)
Xte_blur=blur(Xte)
def fit(Xa,ya,Xb,yb,tag,Xb2=None):
    mu,sd=Xa.mean(0),Xa.std(0)+1e-6; Xa=(Xa-mu)/sd; Xb=(Xb-mu)/sd
    sub=B.train_sub(Xa,ya,100,hidden=48,seed=1,epochs=8,tag=tag)
    fa,ta=B.acc_pair(B.logits_of(sub,Xb),yb,f2c)
    fb,tb=B.acc_pair(B.logits_of(sub,(Xb2-mu)/sd),yb,f2c) if Xb2 is not None else (float('nan'),)*2
    print(f"  {tag:>44s} d={Xa.shape[1]:4d}: full={fa:.4f} task={ta:.4f} | 2x-blur full={fb:.4f} task={tb:.4f}",flush=True)
probes={"eye DoG (i)":eye_feats,"fly: DoG + saccade diffs (ii)":fly_feats,"wave: log-polar spectra (iii)":wave_feats,
        "collapse: structure fn (iv)":collapse_feats,
        "eye + wave":lambda X: torch.cat([eye_feats(X),wave_feats(X)],1),
        "fly + wave + collapse":lambda X: torch.cat([fly_feats(X),wave_feats(X),collapse_feats(X)],1),
        "raw pixels (control)":lambda X: X.reshape(len(X),-1)}
for k,fn in probes.items(): fit(fn(Xtr),ytr,fn(Xte),yte,k,fn(Xte_blur))
