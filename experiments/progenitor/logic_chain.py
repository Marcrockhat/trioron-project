"""s059 NEXT-0 — depth gate for the logic chain (no visuals, no Kinopsis).

World: abstract symbol sequences whose REQUIRED depth scales with k.
  parity_k : 12 binary symbols, label = parity of the first k.  With the quad
             cell sigma(z)=z+z^2 a layer doubles polynomial degree, so parity
             of k bits needs >= ceil(log2 k) quad layers (exact lower bound).
  hop_k    : 8 slots, each holding a pointer symbol in 0..7 (a random map f),
             label = f^k(0).  Pointer chasing; depth ~ k hops.
Link 0 here = ORACLE symbol evidence: per position one-hot, zero-padded to
class_cap (= the shape PhasecyteLeaf.evidence() emits); gradient stops there.
Links i>=1 = Seeded(32 + L*class_cap -> 32, interior 48, nonlinear quad),
input concat[h_{i-1}, e]; head = linear readout on the LAST link only.

Arms: shallow (1 link) | mlp3 (torch ReLU 3x48, reference) | grown (link added
on Numa stall: freeze 0..k, spawn k+1, move head) | grown_joint (spawn on stall,
all links stay trainable — lock deferred to settlement) | tied (one link re-applied
R times, BPTT, R grown on stall).  PASS = grown tracks k where shallow breaks.

Run from repo root:  python3 experiments/progenitor/logic_chain.py
Env: SEEDS=3 TASKS=parity,hop KS=.. ARMS=shallow,mlp3,grown,tied EPOCHS=12
"""
from __future__ import annotations
import math, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.getcwd())
from trioron.core import construct
from trioron.bases.seeded import Seeded

CLASS_CAP = 8
H = 32
HID = 48
MAX_LINKS = int(os.environ.get("MAX_LINKS", 5))
STAGE_EP = int(os.environ.get("EPOCHS", 12))
PATIENCE = int(os.environ.get("PATIENCE", 3))
REL_TOL = 0.01
TARGET = 0.97


# ----------------------------------------------------------------- worlds
def make_parity(n, k, g):
    bits = torch.randint(0, 2, (n, 12), generator=g)
    y = bits[:, :k].sum(1) % 2
    return bits, y, 2


def make_hop(n, k, g):
    L = 8
    f = torch.randint(0, L, (n, L), generator=g)
    pos = torch.zeros(n, dtype=torch.long)
    for _ in range(k):
        pos = f[torch.arange(n), pos]
    return f, pos, L


def evidence(sym: torch.Tensor) -> torch.Tensor:
    """Oracle Link-0: [N, L] symbols -> [N, L*CLASS_CAP] zero-padded one-hot."""
    return F.one_hot(sym, CLASS_CAP).float().flatten(1)


WORLDS = {"parity": make_parity, "hop": make_hop}


# ----------------------------------------------------------------- models
def new_link(in_dim, seed, layers=1):
    torch.manual_seed(seed)
    sub = construct(Seeded(in_dim, H, interior_cells=HID, interior_layers=layers,
                           nonlinear=True),
                    capacity=in_dim + HID * layers + H + 8)
    sub.prepare_training()
    return sub


class Chain:
    """links[i]: Seeded(H + E -> H); head: Linear(H -> C) on the last link."""

    def __init__(self, E, C, seed, joint=False):
        self.E, self.C, self.seed, self.joint = E, C, seed, joint
        self.links = [new_link(H + E, seed * 100)]
        torch.manual_seed(seed)
        self.head = nn.Linear(H, C)

    def grow(self):
        self.links.append(new_link(H + self.E, self.seed * 100 + len(self.links)))
        torch.manual_seed(self.seed + len(self.links))
        self.head = nn.Linear(H, self.C)      # head moves to the new link

    def forward(self, e, train_last_only=True):
        train_last_only = train_last_only and not self.joint
        h = torch.zeros(len(e), H)
        for i, link in enumerate(self.links):
            frozen = train_last_only and i < len(self.links) - 1
            with torch.set_grad_enabled(not frozen):
                h = link(torch.cat([h, e], 1))
            if frozen:
                h = h.detach()
        return self.head(h)

    def params(self):
        links = self.links if self.joint else self.links[-1:]
        return sum((l.trainable_tensors() for l in links), []) + list(self.head.parameters())


class Tied:
    def __init__(self, E, C, seed):
        self.link = new_link(H + E, seed * 100)
        torch.manual_seed(seed)
        self.head = nn.Linear(H, C)
        self.R = 1

    def grow(self):
        self.R += 1

    def forward(self, e, **_):
        h = torch.zeros(len(e), H)
        for _ in range(self.R):
            h = self.link(torch.cat([h, e], 1))
        return self.head(h)

    def params(self):
        return self.link.trainable_tensors() + list(self.head.parameters())


class Channels:
    """s060-0a: width by partition. E is split into NCH disjoint windows; each
    window has its own link (no h input); a combiner link eats concat of the
    channel outputs. Joint training. For parity any partition works."""

    def __init__(self, E, C, seed, nch=None):
        nch = nch or int(os.environ.get("NCH", 3))
        self.E, self.C = E, C
        self.bounds = [(E * i // nch, E * (i + 1) // nch) for i in range(nch)]
        cd = int(os.environ.get("CH_DEPTH", 1))     # quad layers per channel / combiner
        self.chan = [new_link(b - a, seed * 100 + i, cd) for i, (a, b) in enumerate(self.bounds)]
        self.comb = new_link(H * nch, seed * 100 + 50, cd)
        torch.manual_seed(seed)
        self.head = nn.Linear(H, C)

    def grow(self): pass

    def forward(self, e, **_):
        hs = [l(e[:, a:b]) for l, (a, b) in zip(self.chan, self.bounds)]
        return self.head(self.comb(torch.cat(hs, 1)))

    def params(self):
        return sum((l.trainable_tensors() for l in self.chan + [self.comb]), []) \
            + list(self.head.parameters())


class TiedWTA(Tied):
    """s060-0b: discrete state between applications — softmax(h/τ) (soft WTA),
    the substrate's scratchpad token."""
    TAU = float(os.environ.get("TAU", 0.3))

    def forward(self, e, **_):
        h = torch.zeros(len(e), H)
        for _ in range(self.R):
            h = torch.softmax(self.link(torch.cat([h, e], 1)) / self.TAU, 1)
        return self.head(h)


class TiedHard(Tied):
    """Crisp state (Rocky): forward uses the true one-hot argmax of h,
    backward is straight-through via softmax. Logic states are discrete;
    softmax only as the gradient path."""
    TAU = float(os.environ.get("TAU", 1.0))

    def forward(self, e, **_):
        h = torch.zeros(len(e), H)
        for _ in range(self.R):
            z = self.link(torch.cat([h, e], 1))
            soft = torch.softmax(z / self.TAU, 1)
            hard = F.one_hot(z.argmax(1), H).float()
            h = hard + soft - soft.detach()
        return self.head(h)


class TiedGated(Tied):
    """Gated ambiguity filter (Rocky): clear states (top-2 margin of the
    softmax > THETA) are snapped to a crisp one-hot (straight-through);
    ambiguous states pass through soft — doubt is carried, not computed
    away. In the full architecture the ambiguous branch is the Phasecyte's
    frustration signal (divide)."""
    THETA = float(os.environ.get("THETA", 0.3))

    def forward(self, e, **_):
        h = torch.zeros(len(e), H)
        for _ in range(self.R):
            z = self.link(torch.cat([h, e], 1))
            soft = torch.softmax(z, 1)
            top2 = soft.topk(2, 1).values
            clear = ((top2[:, 0] - top2[:, 1]) > self.THETA).float().unsqueeze(1)
            hard = F.one_hot(z.argmax(1), H).float()
            crisp = hard + soft - soft.detach()
            h = clear * crisp + (1 - clear) * soft
        return self.head(h)


class MLP3:
    def __init__(self, E, C, seed):
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(E, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(),
                                 nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, C))

    def grow(self): pass
    def forward(self, e, **_): return self.net(e)
    def params(self): return list(self.net.parameters())


# ----------------------------------------------------------------- training
def accuracy(model, e, y):
    with torch.no_grad():
        return (model.forward(e).argmax(1) == y).float().mean().item()


def train_stage(model, e, y, et, yt, epochs, seed, lr=3e-3, bs=128):
    """One growth stage. Returns (stalled, history). Numa stall = train loss
    fails to drop REL_TOL for PATIENCE epochs while test acc < TARGET."""
    opt = torch.optim.Adam(model.params(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    best, bad = float("inf"), 0
    for ep in range(epochs):
        perm = torch.randperm(len(e), generator=g)
        tot = 0.0
        for i in range(0, len(e), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(model.forward(e[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.params(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        tot /= len(e)
        acc = accuracy(model, et, yt)
        if tot < best * (1 - REL_TOL):
            best, bad = tot, 0
        else:
            bad += 1
        if acc >= TARGET:
            return False, acc, tot
        if bad >= PATIENCE:
            return True, acc, tot
    return True, acc, tot


def run(arm, task, k, seed):
    g = torch.Generator().manual_seed(1000 * seed + k)
    s, y, C = WORLDS[task](int(os.environ.get("NTRAIN", 8000)), k, g)
    st, yt, _ = WORLDS[task](4000, k, g)
    e, et = evidence(s), evidence(st)
    E = e.shape[1]
    model = (Chain(E, C, seed, joint=True) if arm == "grown_joint" else
             {"shallow": Chain, "grown": Chain, "tied": Tied, "mlp3": MLP3,
              "channels": Channels, "tied_wta": TiedWTA, "tied_hard": TiedHard, "tied_gated": TiedGated}[arm](E, C, seed))
    grows = MAX_LINKS - 1 if arm in ("grown", "grown_joint", "tied", "tied_wta", "tied_hard", "tied_gated") else 0
    # shallow / mlp3 get the same TOTAL epoch budget as a fully grown chain
    epochs = STAGE_EP if grows else STAGE_EP * MAX_LINKS
    depth = 1
    t0 = time.time()
    while True:
        stalled, acc, loss = train_stage(model, e, y, et, yt, epochs, seed * 7 + depth)
        if not stalled or grows == 0:
            break
        model.grow(); grows -= 1; depth += 1
    return acc, depth, time.time() - t0


if __name__ == "__main__" and not os.environ.get("CURRICULUM") and not os.environ.get("CH_CURRICULUM"):
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    seeds = range(int(os.environ.get("SEEDS", 3)))
    tasks = os.environ.get("TASKS", "parity,hop").split(",")
    arms = os.environ.get("ARMS", "shallow,mlp3,grown,tied").split(",")
    KS = {"parity": [2, 4, 8, 12], "hop": [1, 2, 3, 4]}
    if os.environ.get("KS"):
        KS = {t: [int(x) for x in os.environ["KS"].split(",")] for t in tasks}
    print(f"# depth gate  seeds={list(seeds)} stage_ep={STAGE_EP} max_links={MAX_LINKS}")
    print(f"{'task':7}{'k':>3} {'arm':9} {'acc':>12} {'depth':>8}  secs")
    for task in tasks:
        for k in KS[task]:
            for arm in arms:
                res = [run(arm, task, k, s) for s in seeds]
                a = torch.tensor([r[0] for r in res]); d = [r[1] for r in res]
                print(f"{task:7}{k:>3} {arm:9} {a.mean():.3f}±{a.std():.3f} "
                      f"{str(d):>8}  {sum(r[2] for r in res):.0f}", flush=True)


# ------------------------------------------------------------ channel curriculum (Mode E)
def run_channel_curriculum(k, seed, epochs, nch=3):
    """Stage A: each channel learns the parity of ITS OWN window under its own
    head (own frustration), then locks. Stage B: combiner + head learn
    parity-k over the settled channel states. Primitives first, then compose."""
    g = torch.Generator().manual_seed(1000 * seed + k)
    s, y, C = make_parity(int(os.environ.get("NTRAIN", 8000)), k, g)
    st, yt, _ = make_parity(4000, k, g)
    e, et = evidence(s), evidence(st)
    m = Channels(e.shape[1], C, seed, nch)
    w = 12 // nch                                         # bits per window
    accA = []
    for i, (a, b) in enumerate(m.bounds):                 # stage A
        ya, yta = s[:, i * w:(i + 1) * w].sum(1) % 2, st[:, i * w:(i + 1) * w].sum(1) % 2

        class One:                                        # channel + its own head
            def __init__(self, link):
                self.link = link; torch.manual_seed(seed + i); self.head = nn.Linear(H, 2)
            def forward(self, x, **_): return self.head(self.link(x))
            def params(self): return self.link.trainable_tensors() + list(self.head.parameters())
        one = One(m.chan[i])
        _, acc, _ = train_stage(one, e[:, a:b], ya, et[:, a:b], yta, epochs, seed * 7 + i)
        accA.append(acc)
        for t in m.chan[i].trainable_tensors(): t.requires_grad_(False)   # lock

    class Comb:                                           # stage B
        def forward(self, x, **_):
            with torch.no_grad():
                hs = torch.cat([l(x[:, a:b]) for l, (a, b) in zip(m.chan, m.bounds)], 1)
            return m.head(m.comb(hs))
        def params(self): return m.comb.trainable_tensors() + list(m.head.parameters())
    _, accB, _ = train_stage(Comb(), e, y, et, yt, epochs, seed * 7 + 99)
    return accA, accB


if __name__ == "__main__" and os.environ.get("CH_CURRICULUM"):
    seeds = range(int(os.environ.get("SEEDS", 3)))
    ks = [int(x) for x in os.environ.get("KS", "8,12").split(",")]
    print(f"# channel curriculum  ks={ks} stage_ep={STAGE_EP} nch={os.environ.get('NCH', 3)}")
    for k in ks:
        out = [run_channel_curriculum(k, s, STAGE_EP, int(os.environ.get("NCH", 3))) for s in seeds]
        A = torch.tensor([o[0] for o in out]); B = torch.tensor([o[1] for o in out])
        print(f"parity {k:>3} ch_curric  stageA per-channel {A.mean(0).tolist()}  "
              f"stageB {B.mean():.3f}±{B.std():.3f}", flush=True)


# ------------------------------------------------------------ curriculum arm
def run_curriculum(task, ks, seed, epochs):
    """Tied link grown THROUGH A LIFECYCLE: hop-1 at R=1, then the same
    weights continue on hop-2 at R=2, ... (continual, Mode-E). Before each
    stage we probe zero-shot: R=k on hop-k with weights trained to hop-(k-1)
    — generalisation to depth deeper than trained. Returns rows."""
    rows = []
    model = None
    for k in ks:
        g = torch.Generator().manual_seed(1000 * seed + k)
        s, y, C = WORLDS[task](int(os.environ.get("NTRAIN", 8000)), k, g)
        st, yt, _ = WORLDS[task](4000, k, g)
        e, et = evidence(s), evidence(st)
        if model is None:
            model = Tied(e.shape[1], C, seed)
        else:
            model.grow()                       # R += 1, same link
        zs = accuracy(model, et, yt)           # zero-shot at new depth
        _, acc, _ = train_stage(model, e, y, et, yt, epochs, seed * 7 + k)
        rows.append((k, zs, acc, model.R))
    return rows


if __name__ == "__main__" and os.environ.get("CURRICULUM"):
    seeds = range(int(os.environ.get("SEEDS", 3)))
    task = os.environ.get("TASKS", "hop")
    ks = [int(x) for x in os.environ.get("KS", "1,2,3,4").split(",")]
    print(f"# curriculum tied  task={task} ks={ks} stage_ep={STAGE_EP}")
    out = [run_curriculum(task, ks, s, STAGE_EP) for s in seeds]
    for i, k in enumerate(ks):
        zs = torch.tensor([o[i][1] for o in out]); ac = torch.tensor([o[i][2] for o in out])
        print(f"{task:7}{k:>3} curric   zero-shot {zs.mean():.3f}±{zs.std():.3f}  "
              f"trained {ac.mean():.3f}±{ac.std():.3f}  R={[o[i][3] for o in out]}", flush=True)
