"""Graft one organism's classes into another and see whether the classes reduce (s070;
Rocky: "try to graft Vee to Weiss and check if the class reduced" — "Vee's classes into
Weiss").

IN MEMORY ONLY: both saved bodies are loaded; nothing is saved; no home changes.
One organism each, one seed.

A phasecyte organ's knowledge is its classes' members (phasors in the organ's own pocket
space), so trioron's arena graft (spec §5.3) does not carry it.  What is done here, per organ:

  1  the donor's members are read back to the numbers she sensed (her pockets, through her
     own frame; checked against her lived log), then seen as the recipient sees — through
     his blur and his receptors — which gives each donor class as members in HIS space;
  2  every donor class with at least 5 such members is added to him as a class of its own,
     with the donor's lineage beside his;
  3  his own consolidation round (`_consolidate`: every member to its best class, a class
     left with fewer than 5 retires) is run until nothing moves, at most ROUNDS times.

Beside it, the same rounds on the recipient alone.  Not done: the fold (it needs a trace
of use the grafted classes do not have) and living on after the graft.

    RECIPIENT=outputs/incubator/weiss/body.pt DONOR=outputs/incubator/vee/body.pt \
        python3 experiments/incubator/graft_probe.py
Env: ROUNDS=4; ORGANS=ear,soma,kin
"""
from __future__ import annotations
import copy, math, os, sys, time
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import vee_body as vb
from trioron.pcll.mixed import N_QUANTA

RECIPIENT = os.environ.get("RECIPIENT", "outputs/incubator/weiss/body.pt")
DONOR = os.environ.get("DONOR", "outputs/incubator/vee/body.pt")
ROUNDS = int(os.environ.get("ROUNDS", 4))
ORGANS = os.environ.get("ORGANS", "ear,soma,kin").split(",")
WIDTH = {"ear": vb.LANG, "soma": vb.WIDTH - vb.LANG, "kin": 43}


def sensed(leaf, Zr: torch.Tensor, width: int) -> torch.Tensor:
    """The numbers an organ sensed, back from its members' receptor phasors."""
    m, plan = leaf.mixed, leaf.sub.scheduler._plan
    q = (torch.angle(Zr) / (2 * math.pi) * N_QUANTA) % N_QUANTA
    k = leaf.sub.arena.receptor_levels[plan.receptor_ids.long()].to(q.dtype)
    cont = m._continuous[: q.shape[1]]
    x = torch.where(cont, m._flo + q / N_QUANTA * (m._fhi - m._flo),
                    ((2 * k.clamp(min=1) * q / N_QUANTA - 1) / 2).round())
    P = torch.zeros(len(q), width)
    P[:, plan.receptor_cols] = x
    for c, lv in (getattr(m, "codec_levels", None) or {}).items():      # a discrete column: level index -> the value it stands for
        lv = torch.as_tensor(lv, dtype=P.dtype)
        P[:, c] = lv[P[:, c].long().clamp(0, len(lv) - 1)]
    return P


def as_he_sees(body, organ: str, P: torch.Tensor) -> torch.Tensor:
    mix, out = body.organs()[organ].mixed, []
    for i in range(0, len(P), 4096):
        q = mix.pockets_of(body.seen(organ, P[i:i + 4096]))
        out.append(torch.exp(1j * 2 * math.pi * q / N_QUANTA).to(mix.bufs[0].dtype))
    return torch.cat(out) if out else torch.zeros(0, mix.bufs[0].shape[1], dtype=mix.bufs[0].dtype)


def settle(mix, tag: str):
    for r in range(ROUNDS):
        t0, before = time.time(), len(mix.classes)
        moved = mix._consolidate()
        print(f"     {tag} round {r + 1}: {moved} members moved, classes {before} -> {len(mix.classes)}, {time.time() - t0:.0f} s", flush=True)
        if moved == 0:
            break


def check_inverse(donor, organ: str):
    """Her lived moments -> her pockets -> the numbers back: how exact is the reading-back."""
    p = os.path.join(os.path.dirname(DONOR), "fetal_log.pt")
    if not os.path.exists(p):
        return
    X = torch.cat(torch.load(p, weights_only=False)["X"][-3:])
    leaf = donor.organs()[organ]
    P = donor.seen(organ, donor.part(organ, X))
    q = leaf.mixed.pockets_of(P)[:, : leaf.sub.scheduler._plan.receptor_ids.numel()]
    back = sensed(leaf, torch.exp(1j * 2 * math.pi * q / N_QUANTA), P.shape[1])
    cols = leaf.sub.scheduler._plan.receptor_cols
    err = (back[:, cols] - P[:, cols]).abs()
    print(f"   reading back the donor's senses, on her last 3 lived days: mean error {float(err.mean()):.4f}, "
          f"largest {float(err.max()):.4f} (her numbers span {float(P[:, cols].min()):.2f} .. {float(P[:, cols].max()):.2f})")


def main():
    donor = torch.load(DONOR, weights_only=False)
    print(f"# graft, in memory only: donor {DONOR} ({donor.moments} moments) into recipient {RECIPIENT}; at most {ROUNDS} rounds")
    for organ in ORGANS:
        rec = torch.load(RECIPIENT, weights_only=False)
        mix, dmix = rec.organs()[organ].mixed, donor.organs()[organ].mixed
        dleaf = donor.organs()[organ]
        n_rec = dleaf.sub.scheduler._plan.receptor_ids.numel()
        own, theirs = len(mix.classes), len(dmix.classes)
        print(f"\n== {organ}: recipient {own} classes ({sum(len(b) for b in mix.bufs)} members), donor {theirs} classes ({sum(len(b) for b in dmix.bufs)} members)")
        check_inverse(donor, organ)
        mix.manifold = None; mix.label_taps = None          # sketches and label counts are not needed for the round (in memory only)

        alone = copy.deepcopy(mix)
        settle(alone, "recipient alone,")
        n_alone = len(alone.classes); del alone

        # 1 the donor's classes as he would sense them
        sizes = [len(b) for b in dmix.bufs]
        Z = as_he_sees(rec, organ, sensed(dleaf, torch.cat(dmix.bufs)[:, :n_rec], WIDTH[organ]))
        parts = torch.split(Z, sizes)
        best, _ = mix._margins(Z)                            # before the graft: where his own classes would put her members
        dhome = torch.repeat_interleave(torch.arange(theirs), torch.tensor(sizes))
        spread = [int(best[dhome == k].unique().numel()) for k in range(theirs) if sizes[k]]
        whole = sum(1 for s in spread if s == 1)
        print(f"   before the graft, her members placed in HIS classes: they land in {int(best.unique().numel())} of his {own}; "
              f"{whole} of her {len(spread)} classes land whole in one class of his (median class of hers spreads over {sorted(spread)[len(spread) // 2]})")
        distinct = [int(torch.unique(torch.round(torch.angle(p) * 1000), dim=0).shape[0]) for p in parts]

        # 2 add them as classes of their own, her lineage beside his
        proto, base = mix.classes[0], max(list(mix._parent_node) + [0]) + 1
        added = 0
        for k, c in enumerate(dmix.classes):
            if sizes[k] < 5:
                continue
            new = copy.deepcopy(proto)
            new.name, new.cell_id, new.T = f"graft:{c.name}", -2, parts[k].mean(0)
            mix.classes.append(new); mix.bufs.append(parts[k]); mix.tags.append(torch.full((sizes[k],), -1, dtype=torch.long))
            nid = dmix._node.get(c.name)
            if nid is not None:
                mix._node[new.name] = base + nid
            added += 1
        for nid, par in dmix._parent_node.items():
            mix._parent_node[base + nid] = None if par is None else base + par
        print(f"   grafted {added} of her classes (median {sorted(distinct)[len(distinct) // 2]} distinct patterns a class as he sees them); joined {len(mix.classes)} classes")

        # 3 his round
        settle(mix, "joined,")
        his = sum(1 for c in mix.classes if not c.name.startswith("graft:"))
        hers = len(mix.classes) - his
        in_his = sum(len(b) for c, b in zip(mix.classes, mix.bufs) if not c.name.startswith("graft:"))
        in_hers = sum(len(b) for c, b in zip(mix.classes, mix.bufs) if c.name.startswith("graft:"))
        print(f"   RESULT {organ}: his {own} + hers {added} = {own + added} joined -> {len(mix.classes)} after the rounds "
              f"({his} of his, {hers} of hers; members held in his {in_his}, in hers {in_hers}); "
              f"recipient alone after the same rounds: {n_alone}")
        del rec, mix


if __name__ == "__main__":
    main()
