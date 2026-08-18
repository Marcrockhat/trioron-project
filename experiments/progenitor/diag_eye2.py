import os, sys, torch
os.environ.setdefault("EYE_TASKS","5")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import cifar_eye_nest as B
torch.manual_seed(0)
Xtr,ytr=B.load_cifar100(True); Xte,yte=B.load_cifar100(False); f2c=B.fine_to_coarse()
T=5; cls=sum([(f2c==t).nonzero().squeeze(1).tolist() for t in range(T)],[])
mtr=torch.isin(ytr,torch.tensor(cls)); mte=torch.isin(yte,torch.tensor(cls))
Xtr,ytr,Xte,yte=Xtr[mtr],ytr[mtr],Xte[mte],yte[mte]
eye=B.Eye((16,16),rectify=False)
def feats(X): return torch.cat([eye.sense(X[i:i+2000]) for i in range(0,len(X),2000)])
Ftr,Fte=feats(Xtr),feats(Xte)
from trioron.core.receptor import quantize
Qtr,Qte=quantize(Ftr)/1000,quantize(Fte)/1000
print("plain quantize std", Qtr.std().item())
def fit(Xa,ya,Xb,yb,tag):
    sub=B.train_sub(Xa,ya,100,hidden=48,seed=1,epochs=8,tag=tag)
    fa,ta=B.acc_pair(B.logits_of(sub,Xb),yb,f2c); print(f"  {tag}: full={fa:.4f} task={ta:.4f}",flush=True)
fit(Qtr,ytr,Qte,yte,"supervised on plain quantize(feats)/1000")
mu,sd=Qtr.mean(0),Qtr.std(0)+1e-6
fit((Qtr-mu)/sd,ytr,(Qte-mu)/sd,yte,"supervised on standardized quantize")
# real pockets standardized
pool=Xtr[torch.isin(ytr,torch.tensor(cls[:5]))]
organ=B.FixationOrgan((16,16),pool,0,0,torch.Generator().manual_seed(1)); organ.eye=eye
g=torch.Generator().manual_seed(0)
for t in range(T):
    idx=torch.isin(ytr,torch.tensor(cls[5*t:5*t+5])).nonzero().squeeze(1); idx=idx[torch.randperm(len(idx),generator=g)]
    X=Xtr[idx]; labs=[f"g{int(v):03d}" for v in ytr[idx]]
    for w0 in range(0,len(X),1000): organ.observe(X[w0:w0+1000],labs[w0:w0+1000])
Ptr,Pte=organ.pockets(Xtr),organ.pockets(Xte)
mu,sd=Ptr.mean(0),Ptr.std(0)+1e-6
fit((Ptr-mu)/sd,ytr,(Pte-mu)/sd,yte,"supervised on STANDARDIZED real pockets")
print("pocket vs quantize corr (P part):", torch.corrcoef(torch.stack([Ptr[:,:336].flatten()[:200000], Qtr[:,:336].flatten()[:200000]]))[0,1].item())
