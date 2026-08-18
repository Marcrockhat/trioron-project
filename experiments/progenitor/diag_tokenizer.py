"""P_F tokenizer probe (design docs/design/canonical_frame_primitives.md §2b).
Wave stream = per-window cepstral shape (32-d) on the 5x5 window grid of (b).
Base symbols: k-means codebook (K0). BPE-like: merge the most frequent ADJACENT
(right/down) token pair into a new token until V tokens; fragments = merged
extents. Metrics: compression (tokens/img), reuse (per-token class entropy),
mosaic boundary respect, tokens/img on 1- vs 4-tile mosaics (number signal),
and downstream: same 25-class probe leaf over token representations vs (b)."""
import os, sys, math, time, torch
import torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE,"diag_eye4.py")).read().split("eye=B.Eye")[0].replace('os.path.abspath(__file__)','os.path.abspath(%r)'%os.path.join(HERE,"diag_eye4.py")))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","6")))
K0=int(os.environ.get("K0","64")); V=int(os.environ.get("V","256"))
WS=int(os.environ.get("WS","12")); ST=int(os.environ.get("ST","5")); G=(32-WS)//ST+1; NS=G*G
NR=int(os.environ.get("NR","5"))   # radial bins for the per-window cepstrum (5 for 12px; use 4 for 8px)
CD=(NR-1)*8                        # cepstral dims per window
DIRS=((0,1),(1,0))   # right, down

def stream(X):   # [N,NS,CD] per-window cepstral shape ((b) unflattened when WS=12,ST=5,NR=5)
    y=Y(X); return torch.stack([cepstrum(w,NR,8,(1,NR)) for w in windows(y,WS,ST)],1)
def kmeans(Z,k,iters=25,seed=0):
    g=torch.Generator().manual_seed(seed); C=Z[torch.randperm(len(Z),generator=g)[:k]].clone()
    for _ in range(iters):
        a=torch.cdist(Z,C).argmin(1)
        for j in range(k):
            m=a==j
            if m.any(): C[j]=Z[m].mean(0)
    return C
def assign(Z,C): return torch.cdist(Z,C).argmin(1)

class Tok:
    """2-D BPE over a GxG grid. State per image: anchor grid A [N,G,G] (token id at
    fragment anchor slot, -1 elsewhere) and owner map M [N,G,G] (anchor flat index)."""
    def __init__(self,K0): self.K0=K0; self.rules=[]; self.parts={i:(i,) for i in range(K0)}   # token -> base symbols (for embedding)
    def init(self,S):
        N=S.shape[0]; A=S.view(N,G,G).clone(); M=torch.arange(NS).view(1,G,G).expand(N,-1,-1).clone(); return A,M
    def pair_counts(self,A):
        V_=self.K0+len(self.rules); cnt=torch.zeros(V_*V_*2,dtype=torch.long)
        for d,(dr,dc) in enumerate(DIRS):
            a=A[:,:G-dr,:G-dc]; b=A[:,dr:,dc:]; ok=(a>=0)&(b>=0)
            cnt+=torch.bincount((a[ok]*V_+b[ok])*2+d,minlength=V_*V_*2)
        return cnt,V_
    def apply(self,A,M,rule,new):
        a,b,d=rule; dr,dc=DIRS[d]
        m=(A[:,:G-dr,:G-dc]==a)&(A[:,dr:,dc:]==b)
        # resolve overlaps when a==b (a a a): keep leftmost/topmost non-overlapping
        if a==b:
            if d==0:
                for c in range(1,G-dc): m[:,:,c]&=~m[:,:,c-1]
            else:
                for r in range(1,G-dr): m[:,r,:]&=~m[:,r-1,:]
        if not m.any(): return
        idxX=torch.nonzero(m)                        # (n, r, c)
        n,r,c=idxX[:,0],idxX[:,1],idxX[:,2]
        ax=r*G+c; ay=(r+dr)*G+(c+dc)
        A[n,r,c]=new; A[n,r+dr,c+dc]=-1
        N=A.shape[0]; R=torch.arange(NS).view(1,NS).expand(N,-1).clone(); R[n,ay]=ax   # per-image anchor remap (all matches at once)
        Mf=M.view(N,NS); Mf.copy_(torch.gather(R,1,Mf))
    def learn(self,S,V,min_count=50):
        A,M=self.init(S)
        while self.K0+len(self.rules)<V:
            cnt,V_=self.pair_counts(A); j=cnt.argmax(); c=int(cnt[j])
            if c<min_count: print(f"  stop: best pair count {c} < {min_count}"); break
            a=int(j//2//V_); b=int(j//2%V_); d=int(j%2); new=V_
            self.rules.append((a,b,d)); self.parts[new]=self.parts[a]+self.parts[b]
            self.apply(A,M,(a,b,d),new)
        return A,M
    def encode(self,S):
        A,M=self.init(S)
        for i,rule in enumerate(self.rules): self.apply(A,M,rule,self.K0+i)
        return A,M

def token_ids(A,M):   # per-slot token id (broadcast anchor's token to owned slots)
    N=A.shape[0]; Af=A.view(N,NS); Mf=M.view(N,NS); return torch.gather(Af,1,Mf)
def n_tokens(A): return (A>=0).view(A.shape[0],-1).sum(1)

def mosaic(X,k,seed=0):   # k in {1,4}: 4 = 2x2 tiles of 4 different images downscaled 2x
    g=torch.Generator().manual_seed(seed)
    if k==1: return X
    idx=torch.randint(0,len(X),(len(X),4),generator=g); T=F.avg_pool2d(X,2)   # [N,3,16,16]
    out=torch.zeros_like(X); out[:,:,:16,:16]=T[idx[:,0]]; out[:,:,:16,16:]=T[idx[:,1]]; out[:,:,16:,:16]=T[idx[:,2]]; out[:,:,16:,16:]=T[idx[:,3]]
    return out

if __name__=="__main__":
    print(f"K0={K0} V={V} grid {G}x{G}",flush=True)
    Str=batched(stream,Xtr); Ste=batched(stream,Xte); Sbl=batched(stream,Xte_blur)
    mu,sd=Str.reshape(-1,CD).mean(0),Str.reshape(-1,CD).std(0)+1e-6
    Ztr=((Str-mu)/sd); Zte=((Ste-mu)/sd); Zbl=((Sbl-mu)/sd)
    t0=time.time(); C=kmeans(Ztr.reshape(-1,CD)[torch.randperm(len(Ztr)*NS)[:60000]],K0); print(f"codebook {K0} ({time.time()-t0:.0f}s)",flush=True)
    Btr=assign(Ztr.reshape(-1,CD),C).view(-1,NS); Bte=assign(Zte.reshape(-1,CD),C).view(-1,NS); Bbl=assign(Zbl.reshape(-1,CD),C).view(-1,NS)
    tok=Tok(K0); t0=time.time(); Atr,Mtr=tok.learn(Btr,V); print(f"learned {len(tok.rules)} merges -> V={K0+len(tok.rules)} ({time.time()-t0:.0f}s)",flush=True)
    Ate,Mte=tok.encode(Bte); Abl,Mbl=tok.encode(Bbl)
    Vt=K0+len(tok.rules)
    # --- tokenizer metrics
    ntr=n_tokens(Atr).float(); nte=n_tokens(Ate).float()
    print(f"compression: tokens/img train {ntr.mean():.2f}±{ntr.std():.2f}  test {nte.mean():.2f}  (25 slots)",flush=True)
    sizes=torch.tensor([len(tok.parts[t]) for t in range(Vt)]); print(f"token extents: mean slots/token {sizes[K0:].float().mean():.2f}, max {sizes.max()}",flush=True)
    # reuse: per-token class distribution entropy over the 25 classes (bits; max log2 25 = 4.64)
    lut=torch.full((100,),-1,dtype=torch.long); lut[torch.tensor(cls)]=torch.arange(len(cls)); y25=lut[ytr]
    tid=Atr.view(len(Atr),-1); img=torch.arange(len(Atr)).unsqueeze(1).expand_as(tid); ok=tid>=0
    joint=torch.zeros(Vt,25).index_put_((tid[ok],y25[img[ok]]),torch.ones(int(ok.sum())),accumulate=True)
    p=joint/joint.sum(1,keepdim=True).clamp(min=1); H=-(p*torch.log2(p.clamp(min=1e-12))).sum(1); freq=joint.sum(1)
    used=freq>0
    print(f"reuse: tokens used {int(used.sum())}/{Vt}; class-entropy mean {H[used].mean():.2f} bits (max 4.64), freq-weighted {(H*freq).sum()/freq.sum():.2f}; base symbols {H[:K0].mean():.2f}, merged {H[K0:][used[K0:]].mean():.2f}",flush=True)
    # mosaic boundary respect + number signal
    Xm=mosaic(Xte,4); Sm=(batched(stream,Xm)-mu)/sd; Bm=assign(Sm.reshape(-1,CD),C).view(-1,NS); Am,Mm=tok.encode(Bm)
    # slot (r,c) tile = (r>=3? , c>=3?) with 12px windows stride 5: window starts 0,5,10,15,20 -> centre 6,11,16,21,26 -> tile boundary at 16: slots 0,1 top, 3,4 bottom, 2 straddles
    def tl(i):   # window i: start i*ST, end i*ST+WS; tile 0 if entirely <16, 1 if entirely >=16, else 2 (straddle)
        a,b=i*ST,i*ST+WS; return 0 if b<=16 else (1 if a>=16 else 2)
    tile=torch.tensor([[(2 if (tl(r)==2 or tl(c)==2) else tl(r)*3+tl(c)) for c in range(G)] for r in range(G)]).view(-1)   # 2 = straddle
    Mf=Mm.view(-1,NS); own=tile[Mf]; me=tile.view(1,-1).expand_as(Mf)
    clean=(me!=2)&(own!=2); cross=((own!=me)&clean).sum().item()/max(1,clean.sum().item())

    print(f"mosaic(4 tiles): tokens/img {n_tokens(Am).float().mean():.2f} vs single {nte.mean():.2f}; slots owned across a tile boundary {cross:.3f} (0 = boundaries respected; chance ~0.19)",flush=True)
    # --- downstream read: same leaf; per-slot representations are pooled to a 5x5 region grid (25*CD dims; identity when G=5)
    rid=torch.tensor([[(r*5)//G*5+(c*5)//G for c in range(G)] for r in range(G)]).view(-1)
    def pool25(Xs):   # [N,NS,CD] -> [N,25*CD]
        out=torch.zeros(len(Xs),25,Xs.shape[2]); out.index_add_(1,rid,Xs); cnt=torch.bincount(rid,minlength=25).float().view(1,25,1); return (out/cnt).reshape(len(Xs),-1)
    Cb=torch.cat([C, torch.stack([C[list(tok.parts[t])].mean(0) for t in range(K0,Vt)])]) if Vt>K0 else C   # token -> mean base centroid (32-d)
    g=torch.Generator().manual_seed(1); E=torch.randn(Vt,CD,generator=g)/math.sqrt(CD)   # fixed random embedding
    def rep_vq(Bx): return pool25(C[Bx])                          # per-slot base centroid (quantized stream), region-pooled
    def rep_tokcent(A,M): return pool25(Cb[token_ids(A,M)])      # per-slot merged-token centroid, region-pooled
    def rep_tokemb(A,M): return pool25(E[token_ids(A,M)])        # per-slot random token embedding, region-pooled
    def rep_bag(A):
        t=A.view(len(A),-1); ok=t>=0; h=torch.zeros(len(A),Vt); h.scatter_add_(1,t.clamp(min=0),ok.float()); return h   # 256, position-free
    def rep_bag_vq(Bx): return torch.zeros(len(Bx),K0).scatter_add_(1,Bx,torch.ones_like(Bx,dtype=torch.float))
    def fit(Xa,ya,Xb,yb,tag,Xb2):
        m,s=Xa.mean(0),Xa.std(0)+1e-6; Xa=(Xa-m)/s; Xb=(Xb-m)/s
        sub=B.train_sub(Xa,ya,100,hidden=48,seed=1,epochs=8,tag=tag)
        fa,ta=B.acc_pair(B.logits_of(sub,Xb),yb,f2c); fb,tb=B.acc_pair(B.logits_of(sub,(Xb2-m)/s),yb,f2c)
        print(f"  {tag:>44s} d={Xa.shape[1]:4d}: full={fa:.4f} task={ta:.4f} | 2x-blur full={fb:.4f} task={tb:.4f}",flush=True)
    fit(pool25(Ztr),ytr,pool25(Zte),yte,f"stream {G}x{G}x{CD} region-pooled to 25 (control)",pool25(Zbl))
    if G!=5:
        Str5=batched(lambda X: torch.stack([cepstrum(w,5,8,(1,5)) for w in windows(Y(X),12,5)],1),Xtr).reshape(len(Xtr),-1)
        Ste5=batched(lambda X: torch.stack([cepstrum(w,5,8,(1,5)) for w in windows(Y(X),12,5)],1),Xte).reshape(len(Xte),-1)
        Sbl5=batched(lambda X: torch.stack([cepstrum(w,5,8,(1,5)) for w in windows(Y(X),12,5)],1),Xte_blur).reshape(len(Xte),-1)
        fit(Str5,ytr,Ste5,yte,"(b) 12px/5 cepstral spectrogram (control)",Sbl5)
    fit(rep_vq(Btr),ytr,rep_vq(Bte),yte,f"VQ K0={K0} per-slot centroid",rep_vq(Bbl))
    fit(rep_bag_vq(Btr),ytr,rep_bag_vq(Bte),yte,f"VQ bag (position-free) K0={K0}",rep_bag_vq(Bbl))
    fit(rep_tokcent(Atr,Mtr),ytr,rep_tokcent(Ate,Mte),yte,f"tokens V={Vt} per-slot centroid",rep_tokcent(Abl,Mbl))
    fit(rep_tokemb(Atr,Mtr),ytr,rep_tokemb(Ate,Mte),yte,f"tokens V={Vt} per-slot random emb",rep_tokemb(Abl,Mbl))
    fit(rep_bag(Atr),ytr,rep_bag(Ate),yte,f"token bag (position-free) V={Vt}",rep_bag(Abl))
    fit(torch.cat([rep_bag(Atr),rep_bag_vq(Btr)],1),ytr,torch.cat([rep_bag(Ate),rep_bag_vq(Bte)],1),yte,"token bag + VQ bag",torch.cat([rep_bag(Abl),rep_bag_vq(Bbl)],1))
    # --- overlapping tokens (Rocky, s052): n-gram bag — every adjacent pair / triple of base symbols counts,
    #     a slot may sit in several; vocabulary = top-V most frequent n-grams on train
    def ngrams(Bx):
        Bg=Bx.view(len(Bx),G,G); keys=[]
        for d,(dr,dc) in enumerate(DIRS):
            a=Bg[:,:G-dr,:G-dc]; b=Bg[:,dr:,dc:]; keys.append(((a*K0+b)*2+d).reshape(len(Bx),-1))
        pairs=torch.cat(keys,1)
        tri=[]
        for d,(dr,dc) in enumerate(DIRS):
            a=Bg[:,:G-2*dr,:G-2*dc]; b=Bg[:,dr:G-dr if dr else G, dc:G-dc if dc else G]; c=Bg[:,2*dr:,2*dc:]
            tri.append((((a*K0+b)*K0+c)*2+d).reshape(len(Bx),-1))
        return pairs, torch.cat(tri,1)
    Ptr,Ttr=ngrams(Btr); Pte,Tte=ngrams(Bte); Pbl,Tbl=ngrams(Bbl)
    def topv(keys,V_): u,cnt=torch.unique(keys.flatten(),return_counts=True); return u[cnt.topk(min(V_,len(u))).indices]
    def bag_of(keys,vocab):
        pos=torch.searchsorted(vocab,keys.clamp(max=vocab.max())); hit=(vocab[pos.clamp(max=len(vocab)-1)]==keys)
        h=torch.zeros(len(keys),len(vocab)); h.scatter_add_(1,pos.clamp(max=len(vocab)-1),hit.float()); return h
    vp=topv(Ptr,V).sort().values; vt=topv(Ttr,V).sort().values
    print(f"overlap: distinct pairs {len(torch.unique(Ptr))}, triples {len(torch.unique(Ttr))}; top-{V} pairs cover {bag_of(Ptr,vp).sum()/Ptr.numel():.3f} of pair occurrences, triples {bag_of(Ttr,vt).sum()/Ttr.numel():.3f}",flush=True)
    fit(bag_of(Ptr,vp),ytr,bag_of(Pte,vp),yte,f"OVERLAP pair bag top-{V}",bag_of(Pbl,vp))
    fit(torch.cat([bag_of(Ptr,vp),bag_of(Ttr,vt)],1),ytr,torch.cat([bag_of(Pte,vp),bag_of(Tte,vt)],1),yte,f"OVERLAP pair+triple bag 2x{V}",torch.cat([bag_of(Pbl,vp),bag_of(Tbl,vt)],1))
    fit(torch.cat([bag_of(Ptr,vp),rep_bag_vq(Btr)],1),ytr,torch.cat([bag_of(Pte,vp),rep_bag_vq(Bte)],1),yte,f"OVERLAP pair bag + VQ bag",torch.cat([bag_of(Pbl,vp),rep_bag_vq(Bbl)],1))
    fit(torch.cat([bag_of(Ptr,vp),rep_vq(Btr)],1),ytr,torch.cat([bag_of(Pte,vp),rep_vq(Bte)],1),yte,f"OVERLAP pair bag + VQ per-slot (OVER CAP)",torch.cat([bag_of(Pbl,vp),rep_vq(Bbl)],1))
    print("done")
