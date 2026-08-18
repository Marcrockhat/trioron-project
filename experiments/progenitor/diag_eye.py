import os, sys, time, torch
os.environ.setdefault("EYE_TASKS","5")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import cifar_eye_nest as B
from trioron.core.receptor import N_QUANTA
torch.manual_seed(0)
Xtr,ytr=B.load_cifar100(True); Xte,yte=B.load_cifar100(False); f2c=B.fine_to_coarse()
T=int(os.environ["EYE_TASKS"]); cls=sum([(f2c==t).nonzero().squeeze(1).tolist() for t in range(T)],[])
mtr=torch.isin(ytr,torch.tensor(cls)); mte=torch.isin(yte,torch.tensor(cls))
Xtr,ytr,Xte,yte=Xtr[mtr],ytr[mtr],Xte[mte],yte[mte]
eye=B.Eye((16,16),rectify=False)
def feats(X): return torch.cat([eye.sense(X[i:i+2000]) for i in range(0,len(X),2000)])
Ftr,Fte=feats(Xtr),feats(Xte)
print("eye feats", Ftr.shape, "std", Ftr.std().item())
def fit(Xa,ya,Xb,yb,tag,hidden=48,epochs=8):
    sub=B.train_sub(Xa,ya,100,hidden=hidden,seed=1,epochs=epochs,tag=tag)
    fa,ta=B.acc_pair(B.logits_of(sub,Xb),yb,f2c); print(f"  {tag}: full={fa:.4f} task={ta:.4f} (chance {1/len(cls):.3f})",flush=True)
fit(Ftr,ytr,Fte,yte,"supervised on raw eye feats")
# now phasecyte pockets (one leaf on P+M concat sense) — real pockets supervised
pool=Xtr[torch.isin(ytr,torch.tensor(cls[:5]))]
organ=B.FixationOrgan((16,16),pool,0,0,torch.Generator().manual_seed(1))
organ.eye.rectify=False; organ.eye=eye  # same signed eye
g=torch.Generator().manual_seed(0)
for t in range(T):
    idx=torch.isin(ytr,torch.tensor(cls[5*t:5*t+5])).nonzero().squeeze(1); idx=idx[torch.randperm(len(idx),generator=g)]
    X=Xtr[idx]; labs=[f"g{int(v):03d}" for v in ytr[idx]]
    for w0 in range(0,len(X),1000): organ.observe(X[w0:w0+1000],labs[w0:w0+1000])
print("leaf classes", {s:len(l.mixed.classes) for s,l in organ.leaves.items()})
Ptr,Pte=organ.pockets(Xtr),organ.pockets(Xte)
print("pockets", Ptr.shape, "std", Ptr.std().item(), "frac zero", (Ptr==0).float().mean().item(), "frac one", (Ptr==1).float().mean().item())
fit(Ptr,ytr,Pte,yte,"supervised on REAL pockets")
Xp,yp,sk=B.dream_pseudo(organ,cls,300,5); print("pseudo", Xp.shape, "skipped", sk, "std", Xp.std().item())
fit(Xp,yp,Pte,yte,"dreamed from sketches (eval on real pockets)")
