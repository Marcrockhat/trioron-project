"""Stereo spectra (Rocky, s052): read the image as two frequency STREAMS.
(i) unsynchronised: H stream = 1-D spectrum of each row (time = y), V stream = of each column (time = x).
(ii) synchronised: time = a shared scan path over 8px patches (13x13 raster); at each step channel L =
     horizontal 1-D spectrum of the patch (mean over its rows), channel R = vertical (mean over columns);
     both channels see the same object at the same time. (iii) + disparity L-R.
Same 25-class probe leaf; controls (b), dense-pooled stream, raw pixels."""
import os, sys, math, torch
import torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE,"diag_eye4.py")).read().split("eye=B.Eye")[0].replace('os.path.abspath(__file__)','os.path.abspath(%r)'%os.path.join(HERE,"diag_eye4.py")))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","6")))
def spec1d(x, keep):   # x [..., L] -> log power of bins 1..keep (drop DC), Hann
    L=x.shape[-1]; w=torch.hann_window(L,periodic=False); X=torch.fft.rfft((x-x.mean(-1,keepdim=True))*w,dim=-1)
    return torch.log((X.abs()**2)[...,1:keep+1]+1e-6)
def shape1d(P):   # spectral SHAPE: subtract per-frame mean log power (drop gain)
    return P-P.mean(-1,keepdim=True)
# (i) unsynchronised full-length rows / cols
def H_rows(X): y=Y(X); return shape1d(spec1d(y,16)).reshape(len(X),-1)                     # 32 rows x 16 = 512
def V_cols(X): y=Y(X); return shape1d(spec1d(y.transpose(1,2),16)).reshape(len(X),-1)      # 32 cols x 16 = 512
def HV_unsync(X): return torch.cat([H_rows(X),V_cols(X)],1)                                # 1024 (over cap)
def HV_unsync_cep(X):   # cepstral shape 8 coeffs per frame -> 32x8x2 = 512
    y=Y(X); D=dct_mat(16)[1:9]                                                              # keep c1..c8
    h=torch.einsum("kf,nrf->nrk",D,spec1d(y,16)); v=torch.einsum("kf,nrf->nrk",D,spec1d(y.transpose(1,2),16))
    return torch.cat([h.reshape(len(X),-1),v.reshape(len(X),-1)],1)
# (ii) synchronised along a 13x13 raster of 8px patches
def patches(y,ws=8,st=2): return y.unfold(1,ws,st).unfold(2,ws,st)                          # [N,13,13,8,8]
def LR_sync(X, keep=4):
    P=patches(Y(X)); N=P.shape[0]
    L=spec1d(P,keep).mean(-2)                 # spectrum along x for each row, mean over rows -> [N,13,13,keep]
    R=spec1d(P.transpose(-1,-2),keep).mean(-2)  # along y, mean over cols
    return shape1d(L), shape1d(R)             # each [N,13,13,keep]
rid=torch.tensor([[(r*5)//13*5+(c*5)//13 for c in range(13)] for r in range(13)]).view(-1)
def pool25(Z):   # [N,169,d] -> [N,25*d]
    out=torch.zeros(len(Z),25,Z.shape[2]); out.index_add_(1,rid,Z); cnt=torch.bincount(rid,minlength=25).float().view(1,25,1); return (out/cnt).reshape(len(Z),-1)
def sync_full(X): L,R=LR_sync(X); return torch.cat([L,R],-1).reshape(len(X),-1)             # 169x8 = 1352 (over cap)
def sync_pooled(X): L,R=LR_sync(X); return pool25(torch.cat([L,R],-1).reshape(len(X),169,-1))   # 25x8 = 200
def sync_pooled_disp(X): L,R=LR_sync(X); return pool25(torch.cat([L,R,L-R],-1).reshape(len(X),169,-1))   # 25x12 = 300
def sync_pooled_finer(X):   # 7x7 pooling of the 13x13 (49x8 = 392)
    L,R=LR_sync(X); Z=torch.cat([L,R],-1).reshape(len(X),169,-1)
    r7=torch.tensor([[(r*7)//13*7+(c*7)//13 for c in range(13)] for r in range(13)]).view(-1)
    out=torch.zeros(len(Z),49,Z.shape[2]); out.index_add_(1,r7,Z); cnt=torch.bincount(r7,minlength=49).float().view(1,49,1); return (out/cnt).reshape(len(Z),-1)
def dense_pooled(X):   # s052 dense control: 8px/2 log-polar cepstra pooled to 25 (600)
    y=Y(X); S=torch.stack([cepstrum(w,4,8,(1,4)) for w in windows(y,8,2)],1); return pool25(S)
def fit(Xa,ya,Xb,yb,tag,Xb2):
    mu,sd=Xa.mean(0),Xa.std(0)+1e-6; Xa=(Xa-mu)/sd; Xb=(Xb-mu)/sd
    sub=B.train_sub(Xa,ya,100,hidden=48,seed=1,epochs=8,tag=tag)
    fa,ta=B.acc_pair(B.logits_of(sub,Xb),yb,f2c); fb,tb=B.acc_pair(B.logits_of(sub,(Xb2-mu)/sd),yb,f2c)
    print(f"  {tag:>46s} d={Xa.shape[1]:4d}: full={fa:.4f} task={ta:.4f} | 2x-blur full={fb:.4f} task={tb:.4f}",flush=True)
probes={"(i) H rows stream (unsync)":H_rows,"(i) V cols stream (unsync)":V_cols,"(i) H+V unsync (OVER CAP)":HV_unsync,
        "(i) H+V unsync cepstral 8":HV_unsync_cep,
        "(ii) synced L/R raster, pooled 25":sync_pooled,"(ii) synced L/R, pooled 49":sync_pooled_finer,
        "(iii) synced L/R + disparity, pooled 25":sync_pooled_disp,"(ii) synced L/R full 169 (OVER CAP)":sync_full,
        "dense 8px cepstra pooled 25 (control)":dense_pooled,"(b) 12px/5 (control)":cep_spectrogram}
for k,fn in probes.items(): fit(batched(fn,Xtr),ytr,batched(fn,Xte),yte,k,batched(fn,Xte_blur))
print("raw pixels control (s051): 0.356 / 0.533; done")
