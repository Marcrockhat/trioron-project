"""Primitive vocabulary for vision (Rocky, s051): a shape/texture detector
trained ONLY on synthetic 12x12 windows (circle/triangle/square/polkadots/
stripes/blank), applied to the 25 sliding CIFAR windows (positions kept).
Arms: prims alone; prims + cepstral spectrogram (b); (b) trained with blur
augmentation (resolution/frequency-gap point). 25 classes, same leaf."""
import os, sys, math, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import cifar_eye_nest as B
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_eye4.py")).read().split("eye=B.Eye")[0])   # shared data + cepstrum + windows   # reuse data + cepstrum + windows
PRIMS=["circle","triangle","square","polkadots","stripes","blank"]
S=16
def synth(n, gen):
    yy,xx=torch.meshgrid(torch.arange(S).float(),torch.arange(S).float(),indexing="ij")
    X=torch.zeros(n,S,S); y=torch.randint(0,len(PRIMS),(n,),generator=gen)
    for i in range(n):
        k=int(y[i]); cx,cy=(torch.rand(2,generator=gen)*4+6).tolist(); r=float(torch.rand(1,generator=gen)*3.5+3.0)
        th=float(torch.rand(1,generator=gen)*math.pi); dx,dy=xx-cx,yy-cy
        u=dx*math.cos(th)+dy*math.sin(th); v=-dx*math.sin(th)+dy*math.cos(th)
        if k==0: m=(dx**2+dy**2<=r*r)
        elif k==1: m=(v>-r*0.6)&(u.abs()<(r*1.2-v)*0.6)
        elif k==2: m=(u.abs()<=r)&(v.abs()<=r)
        elif k==3:
            p=float(torch.rand(1,generator=gen)*4+4); m=(((u%p)-p/2)**2+((v%p)-p/2)**2)<=(p*0.28)**2
        elif k==4:
            p=float(torch.rand(1,generator=gen)*5+3); m=torch.sin(2*math.pi*u/p)>0
        else: m=torch.zeros(S,S,dtype=torch.bool)
        bg=float(torch.rand(1,generator=gen)); fg=float(torch.rand(1,generator=gen))
        img=torch.where(m,torch.tensor(fg),torch.tensor(bg))
        img=img+0.05*torch.randn(S,S,generator=gen)
        if float(torch.rand(1,generator=gen))<0.5:   # blur half of them
            img=F.avg_pool2d(img[None,None],3,1,1)[0,0]
        X[i]=img
    return X,y
gen=torch.Generator().manual_seed(3)
Xs,ys=synth(24000,gen); Xs2,ys2=synth(4000,gen)
def norm_win(w):
    """detector input = the window's frequency signature (coordinate-free
    within the window): log-polar log-spectrum (5x8) + cepstrum (4x8)."""
    w=w-w.mean((1,2),keepdim=True); w=w/(w.std((1,2),keepdim=True)+1e-3)
    return torch.cat([logpolar(w,6,8).flatten(1), cepstrum(w,6,8,(1,6))],1)
det=B.train_sub(norm_win(Xs),ys,len(PRIMS),hidden=96,seed=2,epochs=15,tag="primitive detector")
pred=B.logits_of(det,norm_win(Xs2)).argmax(1); acc=float((pred==ys2).float().mean())
per={PRIMS[k]: round(float((pred[ys2==k]==k).float().mean()),2) for k in range(len(PRIMS))}
print(f"  primitive detector synthetic acc={acc:.3f} per-class {per}",flush=True)
def prim_feats(X):
    y=Y(X); out=[]
    for w in windows(y,S,4):
        out.append(torch.softmax(B.logits_of(det,norm_win(w)),1))
    return torch.cat(out,1)              # 25 windows x 6
def fit(Xa,ya,Xb,yb,tag,Xb2):
    mu,sd=Xa.mean(0),Xa.std(0)+1e-6; Xa=(Xa-mu)/sd; Xb=(Xb-mu)/sd
    sub=B.train_sub(Xa,ya,100,hidden=48,seed=1,epochs=8,tag=tag)
    fa,ta=B.acc_pair(B.logits_of(sub,Xb),yb,f2c); fb,tb=B.acc_pair(B.logits_of(sub,(Xb2-mu)/sd),yb,f2c)
    print(f"  {tag:>40s} d={Xa.shape[1]:4d}: full={fa:.4f} task={ta:.4f} | 2x-blur full={fb:.4f} task={tb:.4f}",flush=True)
def fitb(Xa,ya,Xb,yb,tag,Xb2,Xaug=None,yaug=None):
    if Xaug is not None: Xa=torch.cat([Xa,Xaug]); ya=torch.cat([ya,yaug])
    fit(Xa,ya,Xb,yb,tag,Xb2)
Ftr_b,Fte_b,Fbl_b=batched(cep_spectrogram,Xtr),batched(cep_spectrogram,Xte),batched(cep_spectrogram,Xte_blur)
Ptr,Pte,Pbl=batched(prim_feats,Xtr),batched(prim_feats,Xte),batched(prim_feats,Xte_blur)
print("prim feats mean response per primitive (train):", Ptr.view(len(Ptr),-1,6).mean((0,1)).tolist())
fitb(Ptr,ytr,Pte,yte,"(g) primitives x25 windows",Pbl)
fitb(torch.cat([Ptr,Ftr_b],1),ytr,torch.cat([Pte,Fte_b],1),yte,"(h) primitives + cepstral spectrogram (b)",torch.cat([Pbl,Fbl_b],1))
Ftr_bl=batched(cep_spectrogram,blur(Xtr))
fitb(Ftr_b,ytr,Fte_b,yte,"(i) (b) trained with blur augmentation",Fbl_b,Ftr_bl,ytr)
fitb(torch.cat([Ptr,Ftr_b],1),ytr,torch.cat([Pte,Fte_b],1),yte,"(j) (h) + blur aug",torch.cat([Pbl,Fbl_b],1),torch.cat([batched(prim_feats,blur(Xtr)),Ftr_bl],1),ytr)
