"""Phasecyte pilot, night replay — where can the night's rewordings come from?
(Rocky s066: day = a fact is heard ONCE; night = it is replayed to the phasecyte.)

pilot_stream.py fed helper-written rewordings.  Here only the SOURCE of the
night material changes; the stream (four days of 10 facts, nothing frozen,
earlier days never replayed), the held-out test wordings (helper-written, the
same splits) and the scoring are those of pilot_stream.py.

Sources of a fact's night material (the original wording is always included):
  matron   helper-written rewordings (the pilot; reference, costs helper tokens)
  qwen     rewordings the reflex wrote itself (pilot_reword.py; free, lower quality)
  noise    the original's read state + wording residuals borrowed from other
           articles' questions (no text is written at all)
  repeat   the original wording only, repeated (floor: no rewording)

Arms per source: phasecyte (told, named from current members) and a running
class mean of the same feed.  Reference: the one-shot key table (no night).

Run from repo root:  python3 experiments/incubator/pilot_night.py
Env: as pilot_stream.py (defaults here WINDOW=50 CAP=0)  NPER=40 (noise/repeat samples per fact)
     SOURCES=matron,qwen,noise,repeat
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("WINDOW", "50"); os.environ.setdefault("CAP", "0"); os.environ.setdefault("ARMS", "told")
from stage2_memory import load
from pilot_stream import E, HOLD, KP, N_WAVES, OUT, SEEDS, WINDOW, phasecyte, pilot_rows, scores
from trioron.organisms.base import train_contrastive
from trioron.senses.hidden_state import HiddenStateSense

NPER, SOURCES = E("NPER", 40), E("SOURCES", "matron,qwen,noise,repeat").split(",")

if __name__ == "__main__":
    H2, group2, block2, _, texts2 = load("both")
    H, fact, wave, orig, orig_texts = pilot_rows(H2, group2, texts2)
    other = ~torch.isin(block2, block2[torch.tensor([t in orig_texts for t in texts2])].unique())
    n_fact, n0 = int(fact.max()) + 1, len(H)
    first = torch.stack([(orig & (fact == f)).nonzero()[0, 0] for f in range(n_fact)])   # the wording heard by day

    qid2f = {json.loads(l)["qid"]: int(f) for l, f in zip(open(f"{OUT}/pilot_keys.jsonl"), fact)}
    Hq, fq = torch.zeros(0, H.shape[1]), torch.zeros(0, dtype=torch.long)
    if "qwen" in SOURCES:
        rows = [json.loads(l) for l in open(f"{OUT}/pilot_qwen_keys.jsonl")]
        z = np.load(f"{OUT}/pilot_qwen_keys.npz")
        Hq = torch.cat([torch.tensor(z[p][:len(rows)].astype(np.float32)) for p in ("h_last", "h_mean")], 1)
        fq = torch.tensor([qid2f[r["qid"]] for r in rows])
    g = torch.Generator().manual_seed(0)
    gm = torch.zeros(int(group2.max()) + 1, H2.shape[1]).index_add_(0, group2, H2) / 3
    resid = (H2 - gm[group2])[other]                                   # wording residuals, other articles
    fn = torch.arange(n_fact).repeat_interleave(NPER)
    Hn = H[first][fn] + resid[torch.randint(len(resid), (len(fn),), generator=g)]

    Hall, fact_all = torch.cat([H, Hq, Hn]), torch.cat([fact, fq, fn])
    src = torch.cat([torch.zeros(n0), torch.ones(len(fq)), torch.full((len(fn),), 2)]).long()
    wave_all = torch.where(fact_all >= 0, fact_all // 10, -1)          # fact ids are ordered by wave
    neg = fact_all < 0
    per = torch.bincount(fq, minlength=n_fact) if len(fq) else torch.zeros(n_fact)
    print(f"# pilot night  facts={n_fact} window={WINDOW} kp={KP} hold={HOLD} seeds={SEEDS} nper={NPER} "
          f"qwen rewordings/fact mean {per.float().mean():.1f} min {int(per.min())}", flush=True)
    S = HiddenStateSense(H2[other], KP, dup=False)(Hall)

    res: dict[tuple, list] = {}
    for seed in range(SEEDS):
        g = torch.Generator().manual_seed(seed)
        held = torch.zeros(len(Hall), dtype=torch.bool)
        for f in range(n_fact):                                        # same held-out wordings as pilot_stream
            idx = ((fact == f) & ~orig).nonzero().squeeze(1)
            held[idx[torch.randperm(len(idx), generator=g)[:HOLD]]] = True

        def emit(arm, w, pred, conf, info=""):
            row = [scores(pred, conf, fact_all, held & (wave_all == v), neg) for v in range(w + 1)]
            res.setdefault((arm, w), []).append(row)
            print(f"  seed {seed} {arm:32} after wave {w}: " +
                  "  ".join(f"w{v} {t:.2f}/{h:.2f}" for v, (t, h) in enumerate(row)) + f"   {info}", flush=True)

        for name in SOURCES:
            order_by_wave = []
            for w in range(N_WAVES):
                day = first[w * 10:(w + 1) * 10]
                if name == "matron": idx = ((src == 0) & (wave_all == w) & ~held).nonzero().squeeze(1)
                elif name == "repeat": idx = day.repeat(NPER)
                else: idx = torch.cat([day, ((src == (1 if name == "qwen" else 2)) & (wave_all == w)).nonzero().squeeze(1)])
                order_by_wave.append(idx[torch.randperm(len(idx), generator=g)])
            try:
                phasecyte(S, fact_all, order_by_wave, True, seed,
                          lambda arm, *a: "tags" in arm and emit(f"{name}: phasecyte", *a))
            except Exception as e:                                     # e.g. zero-variance feed
                print(f"  seed {seed} {name}: phasecyte FAILED {type(e).__name__}: {e}", flush=True)
            mu = torch.zeros(n_fact, S.shape[1])
            for w, order in enumerate(order_by_wave):
                mu.index_add_(0, fact_all[order], S[order])
                conf, pred = (F.normalize(S, dim=1) @ F.normalize(mu[:(w + 1) * 10], dim=1).T).max(1)
                emit(f"{name}: class mean", w, pred, conf)

        torch.manual_seed(seed)
        P = HiddenStateSense(H2[other], 256, dup=False)
        lin = torch.nn.Linear(256, 32)
        train_contrastive(lin, list(lin.parameters()), P(H2[other]), group2[other], block2[other], seed, 30)
        with torch.no_grad(): K = F.normalize(lin(P(Hall)), dim=1)
        for w in range(N_WAVES):
            conf, pred = (K @ K[first[:(w + 1) * 10]].T).max(1)
            emit("one-shot key table (no night)", w, pred, conf)

    print(f"\n# after the last wave, {SEEDS} splits: top1 per wave, mean ± sd over splits of the wave mean; hit at {E('FALSE_RATE', 0.05):.0%} false")
    for (arm, w), runs in res.items():
        if w != N_WAVES - 1: continue
        r = np.array(runs)                                             # [split, wave, (top1, hit)]
        m = r[:, :, 0].mean(1)
        print(f"{arm:32} " + "  ".join(f"w{v} {t:.2f}" for v, t in enumerate(r[:, :, 0].mean(0))) +
              f"   mean {m.mean():.3f} ± {m.std(ddof=1) if len(m) > 1 else 0:.3f}   hit {r[:, :, 1].mean():.3f}")
