"""Shazam/SIFT constellation for form: landmarks = local maxima of oriented
DoG-band energy; features = histogram of landmark PAIRS (band1, band2,
quantised drow, dcol) — relative structure only, no absolute position —
vs the same landmarks with absolute position kept. Same 25-class probe."""
import sys, math, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import cifar_eye_nest as B
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_eye4.py")).read().split("eye=B.Eye")[0])   # shared data + cepstrum + windows
def fit(Xa,ya,Xb,yb,tag,Xb2):
    mu,sd=Xa.mean(0),Xa.std(0)+1e-6; Xa=(Xa-mu)/sd; Xb=(Xb-mu)/sd
    sub=B.train_sub(Xa,ya,100,hidden=48,seed=1,epochs=8,tag=tag)
    fa,ta=B.acc_pair(B.logits_of(sub,Xb),yb,f2c); fb,tb=B.acc_pair(B.logits_of(sub,(Xb2-mu)/sd),yb,f2c)
    print(f"  {tag:>44s} d={Xa.shape[1]:4d}: full={fa:.4f} task={ta:.4f} | 2x-blur full={fb:.4f} task={tb:.4f}",flush=True)
# fixed band bank: 4 orientations x 2 scales of DoG-like oriented energy (no learning; multiplexed linears)
def bank():
    ks=[]
    for s in (1.0,2.0):
        r=int(3*s); xx,yy=torch.meshgrid(torch.arange(-r,r+1).float(),torch.arange(-r,r+1).float(),indexing="ij")
        for th in (0,math.pi/4,math.pi/2,3*math.pi/4):
            u=xx*math.cos(th)+yy*math.sin(th); v=-xx*math.sin(th)+yy*math.cos(th)
            g=torch.exp(-(u**2/(2*(0.6*s)**2)+v**2/(2*(1.5*s)**2)))*torch.cos(2*math.pi*u/(4*s))
            g=g-g.mean(); ks.append(g/g.abs().sum())
    return ks
KS=bank()
def band_energy(y):   # y [N,32,32] -> [N,8,32,32]
    outs=[]
    for k in KS:
        r=k.shape[0]//2; outs.append(F.conv2d(F.pad(y[:,None],(r,r,r,r),mode="reflect"),k[None,None]).abs()[:,0])
    return torch.stack(outs,1)
NB=8; K=12   # landmarks per band-map (top-K local maxima)
def landmarks(E):     # E [N,B,H,W] -> per band top-K local maxima: rows, cols, mask
    N,Bn,H,W=E.shape
    mx=F.max_pool2d(E,5,1,2); loc=(E==mx)&(E>E.flatten(2).mean(2)[:,:,None,None]*1.5)
    sc=torch.where(loc,E,torch.zeros_like(E)).flatten(2)          # [N,B,HW]
    top=sc.topk(K,dim=2)
    rows=(top.indices//W).float(); cols=(top.indices%W).float(); ok=top.values>0
    return rows,cols,ok
DQ=4  # offset quantisation (px)
def constellation(X, absolute=False):
    y=Y(X); E=band_energy(y); rows,cols,ok=landmarks(E)      # [N,B,K]
    N=len(X); B_=NB
    r=rows.reshape(N,-1); c=cols.reshape(N,-1); o=ok.reshape(N,-1); b=torch.arange(B_).repeat_interleave(K).expand(N,-1)
    if absolute:
        # per band, 4x4 grid occupancy histogram (position kept)
        gr=(r//8).long(); gc=(c//8).long(); idx=(b*16+gr*4+gc)
        h=torch.zeros(N,B_*16).scatter_add_(1,idx,o.float()); return h
    # pairs: for all landmark pairs (i<j): hash=(b_i,b_j,q(dr),q(dc)) with dr,dc in [-32,32] -> 5x5 offset bins... use sign+magnitude bins
    dr=(r[:,:,None]-r[:,None,:]); dc=(c[:,:,None]-c[:,None,:])
    qr=torch.clamp(((dr+16)//(32/DQ)).long(),0,DQ-1); qc=torch.clamp(((dc+16)//(32/DQ)).long(),0,DQ-1)
    bi=b[:,:,None].expand(-1,-1,b.shape[1]); bj=b[:,None,:].expand(-1,b.shape[1],-1)
    valid=(o[:,:,None]&o[:,None,:])&(torch.triu(torch.ones(b.shape[1],b.shape[1],dtype=torch.bool),1)[None])
    idx=(((bi*B_+bj)*DQ+qr)*DQ+qc)
    h=torch.zeros(N,B_*B_*DQ*DQ).scatter_add_(1,idx.reshape(N,-1),valid.reshape(N,-1).float())
    return torch.log1p(h)
def batched(fn,X,chunk=1000,**kw): return torch.cat([fn(X[i:i+chunk],**kw) for i in range(0,len(X),chunk)])
probes={"(k) constellation: landmark PAIRS, relative only":lambda X: constellation(X),
        "(l) landmarks with absolute position (4x4 grid)":lambda X: constellation(X,absolute=True),
        "(m) (k) + cepstral spectrogram (b)":lambda X: torch.cat([constellation(X),cep_spectrogram(X)],1)}
for k,fn in probes.items(): fit(batched(fn,Xtr),ytr,batched(fn,Xte),yte,k,batched(fn,Xte_blur))
