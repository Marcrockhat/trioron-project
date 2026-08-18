"""Proper Shazam probe for form: peaks in the (position x frequency) spectrogram,
anchor->target hashes (f1,f2,drow,dcol) in a target zone, inverted index over the
training set, and OFFSET-CONSISTENT voting where the 2-D window coordinate plays
the role of Shazam's time (translation vote = coordinate-free matching).
Same 25-class probe as diag_eye4/6. Non-parametric: no leaf, no training.
Test sets: clean, 2x-blur, 4-px shift, half-fragment (Shazam's fragment case)."""
import os, sys, math, time, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_eye4.py")).read().split("eye=B.Eye")[0])   # data, Y, blur, cep_spectrogram
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","4")))
NF=32; WS=8; ST=2; NPOS=(32-WS)//ST+1          # 13x13 window grid = "time" plane
P=int(os.environ.get("PEAKS","48")); Z=int(os.environ.get("ZONE","3"))
NZ=(Z+1)*(2*Z+1); NH=NF*NF*NZ                  # hash space
OFF=2*NPOS-1                                   # offset range per axis

def spectrogram(X):
    """[N,3,32,32] -> S [N,NPOS,NPOS,NF] log-polar log power of 8x8 windows (Hann), c0 removed."""
    y=Y(X); N=y.shape[0]
    w=torch.hann_window(WS,periodic=False); w2=w[:,None]*w[None,:]
    pat=y.unfold(1,WS,ST).unfold(2,WS,ST)                # [N,13,13,8,8]
    pat=(pat-pat.mean((-1,-2),keepdim=True))*w2
    Fm=torch.fft.fftshift(torch.fft.fft2(pat),dim=(-2,-1)).abs()**2   # [N,13,13,8,8]
    yy,xx=torch.meshgrid(torch.arange(WS)-WS//2, torch.arange(WS)-WS//2, indexing="ij")
    r=torch.sqrt(yy**2+xx**2).float(); th=torch.atan2(yy.float(),xx.float())%math.pi
    nr,nt=4,8
    rb=torch.clamp((torch.log1p(r)/math.log1p(WS/2)*nr).long(),max=nr-1); tb=torch.clamp((th/math.pi*nt).long(),max=nt-1)
    idx=(rb*nt+tb).flatten(); mask=(r>0).flatten()
    cnt=torch.zeros(nr*nt).index_add_(0, idx[mask], torch.ones(int(mask.sum())))
    S=torch.zeros(N*NPOS*NPOS,nr*nt).index_add_(1, idx[mask], Fm.reshape(-1,WS*WS)[:,mask])/cnt.clamp(min=1)
    S=torch.log(S+1e-6).view(N,NPOS,NPOS,nr*nt)
    return S-S.mean(-1,keepdim=True)                     # spectral shape (drop gain)

def peaks(S):
    """3-D local maxima (row,col,freq), top-P per image. -> r,c,f [N,P] long, ok [N,P] bool"""
    N=S.shape[0]; V=S.permute(0,3,1,2)[:,None]           # [N,1,F,13,13]
    mx=F.max_pool3d(V,3,1,1)[:,0]                        # [N,F,13,13]
    Sp=S.permute(0,3,1,2)
    thr=Sp.flatten(1).mean(1)+0.5*Sp.flatten(1).std(1)
    loc=(Sp==mx)&(Sp>thr[:,None,None,None])
    sc=torch.where(loc,Sp,torch.full_like(Sp,-1e9)).flatten(1)
    top=sc.topk(P,dim=1); ok=top.values>-1e8
    f=top.indices//(NPOS*NPOS); rc=top.indices%(NPOS*NPOS)
    return rc//NPOS, rc%NPOS, f, ok

def hashes(r,c,f,ok):
    """anchor->target pairs in the zone (ahead in raster order, |dr|,|dc|<=Z).
    -> h [N,M] long, ar,ac [N,M] anchor pos, valid [N,M]"""
    dr=r[:,None,:]-r[:,:,None]; dc=c[:,None,:]-c[:,:,None]        # [N,Pa,Pt] target-anchor
    ahead=(dr>0)|((dr==0)&(dc>0))
    zone=ahead&(dr<=Z)&(dc.abs()<=Z)&ok[:,:,None]&ok[:,None,:]
    h=((f[:,:,None]*NF+f[:,None,:])*(Z+1)+dr.clamp(0,Z))*(2*Z+1)+(dc.clamp(-Z,Z)+Z)
    N,Pa,_=h.shape
    return h.reshape(N,-1), r[:,:,None].expand(-1,-1,Pa).reshape(N,-1), c[:,:,None].expand(-1,-1,Pa).reshape(N,-1), zone.reshape(N,-1)

def build_index(X, chunk=1000):
    """inverted index over training exemplars: sorted by hash; entries (img, ar, ac)."""
    H=[];IM=[];AR=[];AC=[]
    for i in range(0,len(X),chunk):
        S=spectrogram(X[i:i+chunk]); r,c,f,ok=peaks(S); h,ar,ac,v=hashes(r,c,f,ok)
        n=v.sum(1); img=torch.arange(i,i+len(h)).unsqueeze(1).expand_as(h)
        H.append(h[v]); IM.append(img[v]); AR.append(ar[v]); AC.append(ac[v])
    H=torch.cat(H); IM=torch.cat(IM); AR=torch.cat(AR); AC=torch.cat(AC)
    o=H.argsort(); H,IM,AR,AC=H[o],IM[o],AR[o],AC[o]
    start=torch.searchsorted(H, torch.arange(NH)); end=torch.searchsorted(H, torch.arange(NH), right=True)
    return dict(start=start,end=end,img=IM,ar=AR,ac=AC,n=len(H),nimg=len(X))

def query(idx, X, ytr, C, chunk=100, kn=(1,5,20)):
    """returns dict of predictions per method (tensor [N])"""
    preds={k:[] for k in ["class_vote","class_bag","ex1"]+[f"ex_k{k}" for k in kn]}
    for i in range(0,len(X),chunk):
        S=spectrogram(X[i:i+chunk]); r,c,f,ok=peaks(S); h,ar,ac,v=hashes(r,c,f,ok)
        for q in range(len(h)):
            hq=h[q][v[q]]; rq=ar[q][v[q]]; cq=ac[q][v[q]]
            s=idx["start"][hq]; e=idx["end"][hq]; cnt=e-s
            if cnt.sum()==0:
                for k in preds: preds[k].append(torch.tensor(0)); continue
            rep=torch.repeat_interleave(torch.arange(len(hq)),cnt)
            pos=torch.repeat_interleave(s,cnt)+ (torch.arange(int(cnt.sum()))-torch.repeat_interleave(torch.cumsum(cnt,0)-cnt,cnt))
            im=idx["img"][pos]; off=(idx["ar"][pos]-rq[rep]+NPOS-1)*OFF+(idx["ac"][pos]-cq[rep]+NPOS-1)
            cl=ytr[im]
            # (A) class as song: offset-consistent vote per class
            vc=torch.bincount(cl*OFF*OFF+off, minlength=C*OFF*OFF).view(C,-1)
            preds["class_vote"].append(vc.amax(1).argmax())
            # ablation: bag-of-hash matches, no offset consistency
            preds["class_bag"].append(torch.bincount(cl,minlength=C).argmax())
            # (B) exemplar as song: best aligned exemplar(s)
            ve=torch.bincount(im*OFF*OFF+off, minlength=idx["nimg"]*OFF*OFF).view(idx["nimg"],-1).amax(1)
            preds["ex1"].append(ytr[ve.argmax()])
            for k in kn:
                top=ve.topk(k).indices; preds[f"ex_k{k}"].append(torch.bincount(ytr[top],minlength=C).argmax())
    return {k:torch.stack(v) for k,v in preds.items()}

def knn(Xa,ya,Xb,C,k=1,chunk=500):
    Xa=F.normalize(Xa-Xa.mean(0),dim=1); Xb=F.normalize(Xb-Xa.mean(0),dim=1); out=[]
    for i in range(0,len(Xb),chunk):
        sim=Xb[i:i+chunk]@Xa.T; top=sim.topk(k,dim=1).indices
        out.append(torch.stack([torch.bincount(ya[t],minlength=C).argmax() for t in top]))
    return torch.cat(out)

def shift(X,d=4): return torch.roll(X,shifts=(d,d),dims=(2,3))
def fragment(X):   # keep a 20x20 crop at a random-ish fixed offset, grey elsewhere
    M=torch.zeros_like(X); M[:,:,6:26,4:24]=1; return X*M+(1-M)*X.mean((2,3),keepdim=True)

if __name__=="__main__":
    # remap labels to 0..24 for compact bincounts
    lut=torch.full((100,),-1,dtype=torch.long); lut[torch.tensor(cls)]=torch.arange(len(cls)); C=len(cls)
    ytr25=lut[ytr]; yte25=lut[yte]; f2c25=f2c[torch.tensor(cls)]
    def acc_pair25(pred,y):
        full=(pred==y).float().mean().item()
        # task-aware: restrict to same superclass -> for a hard prediction, task acc = pred correct within superclass
        return full
    print(f"peaks P={P} zone Z={Z} hash space {NH} pos grid {NPOS}x{NPOS}",flush=True)
    t0=time.time(); idx=build_index(Xtr); print(f"index: {idx['n']} entries over {idx['nimg']} exemplars ({idx['n']/idx['nimg']:.0f}/img), {time.time()-t0:.0f}s",flush=True)
    tests={"clean":Xte,"2x-blur":Xte_blur,"shift4":shift(Xte),"fragment20":fragment(Xte)}
    for name,Xq in tests.items():
        t0=time.time(); pr=query(idx,Xq,ytr25,C)
        line=" ".join(f"{k}={((v==yte25).float().mean().item()):.4f}" for k,v in pr.items())
        print(f"[{name:>10s}] shazam: {line}  ({time.time()-t0:.0f}s)",flush=True)
        # baselines: kNN raw pixels, kNN cepstral spectrogram (b)
        for k in (1,20):
            a=(knn(Xtr.flatten(1),ytr25,Xq.flatten(1),C,k)==yte25).float().mean().item()
            b_=(knn(batched(cep_spectrogram,Xtr),ytr25,batched(cep_spectrogram,Xq),C,k)==yte25).float().mean().item()
            print(f"[{name:>10s}] kNN k={k:2d}: raw={a:.4f} cep(b)={b_:.4f}",flush=True)
    print("chance 0.04")
