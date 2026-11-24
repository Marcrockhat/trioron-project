"""Phasecyte pilot — can a fact memory be GROWN from a stream and keep learning?
(Rocky s066: the organism is never batch-trained; it is a baby that keeps
absorbing.  Option 2: one class per fact by repeated exposure.)

40 facts (20 articles x 2 questions, so every fact has a same-subject sibling),
each heard in ~50 wordings.  The facts arrive in FOUR WAVES of 10; nothing is
frozen between waves and earlier waves are never shown again.  After every
wave, recall is measured on held-out wordings of every wave so far — a falling
number in an old wave's column is forgetting.

Arms (same stream, same held-out wordings):
  phasecyte, told      one PhasecyteLeaf; every sample carries its fact id as a
                       label tap (the matron says which fact this is).  Two
                       readouts of the same organism: "ledger" = the shipped
                       PhasecyteLeaf.names() (lifetime label counts per class),
                       "tags" = majority label of the class's CURRENT buffer
                       members (PhasecyteLeaf.member_names(); exact, no division inheritance)
  phasecyte, untold    same, no labels; classes are named afterwards by the
                       majority fact of the training samples they claim
                       (an eval-side map: measures discovery, not a deployable readout)
  class mean           running mean of the sense vector per fact, cosine
                       (told; cannot forget by construction — the plain reference)
  one-shot key table   s065 learned linear key (batch-trained on OTHER articles),
                       ONE stored wording per fact — the stage-2 memory

Scores: top1 = right fact; hit = right fact AND confidence above the threshold
that lets FALSE_RATE of the untaught questions through.

Run from repo root:  python3 experiments/incubator/pilot_stream.py
Env: WINDOW=100 (samples per boundary)  REPEATS=1 (passes over a wave)  KP=64 (sense dims)
     CAP=43  HOLD=10 (held-out wordings per fact)  SEEDS=3  COMPOSER=1  FALSE_RATE=0.05
     ARMS=told,untold  SYNTH=1 (mechanics smoke on synthetic wordings, no pilot data needed)
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
from stage2_memory import load
from trioron.organisms.base import train_contrastive
from trioron.pcll import PhasecyteLeaf
from trioron.senses.hidden_state import HiddenStateSense

OUT = "outputs/incubator"
E = lambda k, d: type(d)(os.environ.get(k, d))
WINDOW, REPEATS, KP, CAP, HOLD = E("WINDOW", 100), E("REPEATS", 1), E("KP", 64), E("CAP", 43), E("HOLD", 10)
SEEDS, COMPOSER, FALSE_RATE = E("SEEDS", 3), E("COMPOSER", 1), E("FALSE_RATE", 0.05)
ARMS = E("ARMS", "told,untold").split(",")
N_WAVES = 4


def pilot_rows(H2, group2, texts2):
    """H [N, d], fact (0..39, -1 untaught), wave, is_original; fact ids ordered by wave."""
    if E("SYNTH", 0):                                  # 60 stage-2 questions + borrowed wording residuals
        g = torch.Generator().manual_seed(0)
        gm = torch.zeros(int(group2.max()) + 1, H2.shape[1]).index_add_(0, group2, H2) / 3
        resid = H2 - gm[group2]
        pick = torch.randperm(int(group2.max()) + 1, generator=g)[:60]
        H = torch.cat([gm[q] + resid[torch.randint(len(resid), (51 if j < 40 else 6,), generator=g)]
                       for j, q in enumerate(pick)])
        fact = torch.cat([torch.full((51 if j < 40 else 6,), j if j < 40 else -1) for j in range(60)])
        orig = torch.cat([torch.arange(51 if j < 40 else 6) == 0 for j in range(60)])
        return H, fact, torch.where(fact >= 0, fact // 10, -1), orig, set()
    rows = [json.loads(l) for l in open(f"{OUT}/pilot_keys.jsonl")]
    z = np.load(f"{OUT}/pilot_keys.npz")
    H = torch.cat([torch.tensor(z[p][:len(rows)].astype(np.float32)) for p in ("h_last", "h_mean")], 1)
    ids: dict[str, int] = {}
    for r in sorted(rows, key=lambda r: r["wave"] if r["wave"] >= 0 else 99):
        if r["wave"] >= 0: ids.setdefault(r["qid"], len(ids))
    fact = torch.tensor([ids.get(r["qid"], -1) for r in rows])
    wave = torch.tensor([r["wave"] for r in rows])
    orig = torch.tensor([r["i"] < 0 for r in rows])
    return H, fact, wave, orig, {r["text"] for r in rows if r["i"] < 0}


def scores(pred, conf, fact, m_eval, m_neg):
    tau = torch.quantile(conf[m_neg], 1 - FALSE_RATE)
    right = pred[m_eval] == fact[m_eval]
    return right.float().mean().item(), (right & (conf[m_eval] > tau)).float().mean().item()


def phasecyte(S, fact, order_by_wave, told, seed, emit):
    """Feed the stream wave by wave; ``emit(arm, w, pred, conf, info)`` evaluates."""
    first = order_by_wave[0][:WINDOW]
    leaf = PhasecyteLeaf(lambda x: x, S[first], seed, 0, window=len(first), class_cap=CAP,
                         manifold=True, composer=bool(COMPOSER), verbose=False)
    seen = []
    for w, order in enumerate(order_by_wave):
        for _ in range(REPEATS):
            for i in range(0, len(order), WINDOW):
                bi = order[i:i + WINDOW]
                leaf.mixed.observe(S[bi], labels=[f"g{int(f):02d}" for f in fact[bi]] if told else None)
                leaf.sub.end_task()
        seen.append(order)
        best, marg = leaf.mixed.classify(S)
        readouts = {}
        if told:
            readouts["phasecyte, told ledger"] = leaf.names()
            readouts["phasecyte, told tags"] = leaf.member_names()
        else:                                           # eval-side: class -> majority fact of claimed training samples
            tr = torch.cat(seen)
            names = torch.full((len(leaf.mixed.classes),), -1)
            for c in best[tr].unique():
                names[c] = fact[tr][best[tr] == c].mode().values
            readouts["phasecyte, untold"] = names
        for arm, names in readouts.items():
            pred = torch.where(best >= 0, names[best.clamp(min=0)], -1)
            emit(arm, w, pred, marg, f"classes {len(leaf.mixed.classes)} facts covered "
                                     f"{len(set(names[names >= 0].tolist()))} spawned {len(leaf.mixed.specs)}")


if __name__ == "__main__":
    H2, group2, block2, _, texts2 = load("both")
    H, fact, wave, orig, orig_texts = pilot_rows(H2, group2, texts2)
    other = ~torch.isin(block2, block2[torch.tensor([t in orig_texts for t in texts2])].unique())   # not a pilot article
    print(f"# pilot stream  rows={len(H)} facts={int(fact.max()) + 1} untaught rows={int((fact < 0).sum())} "
          f"window={WINDOW} repeats={REPEATS} kp={KP} cap={CAP} hold={HOLD} composer={COMPOSER} seeds={SEEDS}", flush=True)
    res: dict[tuple, list] = {}
    for seed in range(SEEDS):
        g = torch.Generator().manual_seed(seed)
        held = torch.zeros(len(H), dtype=torch.bool)
        for f in range(int(fact.max()) + 1):            # hold out HOLD rewordings per fact (never the original)
            idx = ((fact == f) & ~orig).nonzero().squeeze(1)
            held[idx[torch.randperm(len(idx), generator=g)[:HOLD]]] = True
        neg = fact < 0
        order_by_wave = []
        for w in range(N_WAVES):
            idx = ((wave == w) & ~held).nonzero().squeeze(1)
            order_by_wave.append(idx[torch.randperm(len(idx), generator=g)])

        def emit(arm, w, pred, conf, info=""):
            if True:
                row = [scores(pred, conf, fact, held & (wave == v), neg) for v in range(w + 1)]
                res.setdefault((arm, w), []).append(row)
                print(f"  seed {seed} {arm:24} after wave {w}: " +
                      "  ".join(f"w{v} {t:.2f}/{h:.2f}" for v, (t, h) in enumerate(row)) + f"   {info}", flush=True)

        S = HiddenStateSense(H2[other], KP, dup=False)(H)          # sense fitted on other articles only
        for arm in ARMS:
            phasecyte(S, fact, order_by_wave, arm == "told", seed, emit)

        mu = torch.zeros(int(fact.max()) + 1, S.shape[1])
        for w, order in enumerate(order_by_wave):                  # running mean per fact (sum is enough for cosine)
            mu.index_add_(0, fact[order], S[order])
            conf, pred = (F.normalize(S, dim=1) @ F.normalize(mu[:(w + 1) * 10], dim=1).T).max(1)
            emit("class mean", w, pred, conf, f"numbers per fact {S.shape[1]}")

        torch.manual_seed(seed)
        P = HiddenStateSense(H2[other], 256, dup=False)
        lin = torch.nn.Linear(256, 32)
        train_contrastive(lin, list(lin.parameters()), P(H2[other]), group2[other], block2[other], seed, 30)
        with torch.no_grad(): K = F.normalize(lin(P(H)), dim=1)
        store = torch.stack([(orig & (fact == f)).nonzero()[0, 0] for f in range(int(fact.max()) + 1)])
        for w in range(N_WAVES):
            conf, pred = (K @ K[store[:(w + 1) * 10]].T).max(1)
            emit("one-shot key table", w, pred, conf, "numbers per fact 32")

    print(f"\n# mean over {SEEDS} splits: top1/hit per wave column (rows: after wave)")
    for (arm, w), runs in res.items():
        r = np.array(runs).mean(0)
        print(f"{arm:24} after wave {w}: " + "  ".join(f"w{v} {t:.2f}/{h:.2f}" for v, (t, h) in enumerate(r)))
