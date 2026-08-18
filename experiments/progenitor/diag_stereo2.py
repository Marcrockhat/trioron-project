"""Combine dense 2-D window cepstra with synced stereo (L/R 1-D) spectra. Same probe."""
import os,sys,torch
HERE=os.path.dirname(os.path.abspath(__file__))
__file__=os.path.join(HERE,"diag_stereo.py")
exec(open(__file__).read().split("probes=")[0])
def dense49(X):
    y=Y(X); S=torch.stack([cepstrum(w,4,8,(1,4)) for w in windows(y,8,2)],1)
    r7=torch.tensor([[(r*7)//13*7+(c*7)//13 for c in range(13)] for r in range(13)]).view(-1)
    out=torch.zeros(len(S),49,S.shape[2]); out.index_add_(1,r7,S); cnt=torch.bincount(r7,minlength=49).float().view(1,49,1); return (out/cnt).reshape(len(S),-1)
probes={"dense 2-D pooled 25 + stereo pooled 25":lambda X: torch.cat([dense_pooled(X),sync_pooled(X)],1),
        "dense 2-D pooled 25 + stereo+disp pooled 25":lambda X: torch.cat([dense_pooled(X),sync_pooled_disp(X)],1),
        "dense 2-D pooled 49 (OVER CAP)":dense49,
        "dense 2-D pooled 49 + stereo pooled 49 (OVER CAP)":lambda X: torch.cat([dense49(X),sync_pooled_finer(X)],1)}
for k,fn in probes.items(): fit(batched(fn,Xtr),ytr,batched(fn,Xte),yte,k,batched(fn,Xte_blur))
print("done")
