"""Layered visualization demo — see the linear + dendrite mixture, not a hairball.

The shipped viewer (trioron/viz/export.py) lays the substrate out with a force-directed
'cose' layout (randomize:true), which scrambles depth — you cannot read layers, wiring,
or asymmetry off it. Every snapshot already carries `rank` (the topological layer); this
demo lays cells out BY rank (x = depth, y = spread within a layer) so the structure is
legible.

It builds the structure the disruptor-dog probe implied: five LINEAR class outputs +
one DOG output fed by a 4-branch DENDRITE (each branch = a stored prototype). The
asymmetry is structural and visible — dog sits a layer deeper with a 4-way fan-in; the
clean classes are a single direct edge.

Run:  python3 -m experiments.growth_exercise.viz_demo   (writes runs/viz_demo.html)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from trioron.core import Envelope, construct
from trioron.bases import minimal
from trioron.phenotype import default_dispatch_table
from trioron.core.epigenome import set_gene, DENDRITE, CREDIT_ELIGIBLE
from trioron.viz.snapshot import capture_full

CLASSES = ["chicken", "cat", "dog", "goat", "cow", "elephant"]
DOG = 3                      # output cell id for the disruptor class
N_BRANCH = 4                 # prototypes the K=4 probe recruited for dog


def build_structure():
    sub = construct(base=minimal(1, len(CLASSES)), envelope=Envelope(),
                    dispatch_table=default_dispatch_table(), capacity=64)
    a = sub.arena

    # Promote the dog output one layer deeper and feed it from a dendrite branch bank.
    a.rank[DOG] = 2
    a.position[DOG] = torch.tensor([1.0, 0.4, 0.5], device=a.device)

    branch_ids = a.alloc(N_BRANCH)
    for j, t in enumerate(branch_ids.tolist()):
        cid = int(t)
        epi = set_gene(set_gene(0, DENDRITE), CREDIT_ELIGIBLE)
        a.epigenome[cid] = epi
        a.refresh_phenotype(cid)
        a.rank[cid] = 1
        a.position[cid] = torch.tensor([0.5, 0.30 + 0.06 * j, 0.5], device=a.device)
        a.lineage_root[cid] = DOG
        a.parent[cid] = DOG
        a.forward_inclusion[cid] = True
        a.add_edges(torch.tensor([0], dtype=torch.int32, device=a.device),       # perception → branch
                    torch.tensor([cid], dtype=torch.int32, device=a.device))
        a.add_edges(torch.tensor([cid], dtype=torch.int32, device=a.device),     # branch → dog output
                    torch.tensor([DOG], dtype=torch.int32, device=a.device))

    # Mute the residual direct perception→dog edge so the branch path reads cleanly.
    for i in range(a.edge_cursor):
        if int(a.edge_src[i].item()) == 0 and int(a.edge_dst[i].item()) == DOG:
            a.edge_weight[i] = 0.0
    return sub


# ── Layered renderer (preset layout from rank) ──────────────────────────────────

_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Trioron — layered</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
<style>body{{margin:0;background:#0d1117;color:#c9d1d9;font-family:Segoe UI,monospace}}
#cy{{position:absolute;top:0;left:0;right:260px;bottom:0}}
#p{{position:absolute;top:0;right:0;width:260px;bottom:0;background:#161b22;border-left:1px solid #30363d;padding:14px;font-size:12px}}
h3{{color:#58a6ff;font-size:13px;margin:10px 0 6px}} .li{{display:flex;align-items:center;gap:6px;margin:3px 0}}
.d{{width:11px;height:11px;border-radius:50%}}</style></head><body>
<div id="cy"></div><div id="p"><h3>Trioron — layered by rank</h3>
<div>x = depth (rank), y = within-layer.<br>Edges flow left → right.</div>
<h3>Phenotype</h3>
<div class="li"><div class="d" style="background:#58a6ff"></div>perception</div>
<div class="li"><div class="d" style="background:#f0883e"></div>output</div>
<div class="li"><div class="d" style="background:#8b949e"></div>linear</div>
<div class="li"><div class="d" style="background:#db61a2"></div>dendrite (branch)</div>
<h3>Read</h3><div>5 classes = 1 direct linear edge.<br><b>dog</b> = rank-2 output with a
{nb}-branch dendrite fan-in (the asymmetry).</div></div>
<script>
const CELLS={cells}, EDGES={edges};
function color(c){{const e=c.epigenome; if(e&(1<<5))return '#58a6ff'; if(e&(1<<6))return '#f0883e';
  return c.phenotype==='dendrite'?'#db61a2':'#8b949e';}}
const ranks=[...new Set(CELLS.map(c=>c.rank))].sort((a,b)=>a-b);
const byRank={{}}; CELLS.forEach(c=>{{(byRank[c.rank]=byRank[c.rank]||[]).push(c);}});
const els=[];
CELLS.forEach(c=>{{const col=byRank[c.rank]; const i=col.indexOf(c);
  els.push({{group:'nodes',data:{{id:'n'+c.id,label:c.id+(c.phenotype==='dendrite'?' δ':''),
    color:color(c),pheno:c.phenotype,rank:c.rank}},
    position:{{x:120+ranks.indexOf(c.rank)*320, y:90+i*(620/(col.length))}}}});}});
EDGES.forEach((e,i)=>{{if(Math.abs(e.weight)<1e-9)return;
  els.push({{group:'edges',data:{{id:'e'+i,source:'n'+e.src,target:'n'+e.dst}}}});}});
cytoscape({{container:document.getElementById('cy'),elements:els,
  style:[{{selector:'node',style:{{'background-color':'data(color)','label':'data(label)',
    'color':'#fff','font-size':11,'text-valign':'center','width':26,'height':26}}}},
   {{selector:'node[pheno = "output"]',style:{{'width':34,'height':34,'shape':'round-rectangle'}}}},
   {{selector:'edge',style:{{'width':1.4,'line-color':'#39506b','curve-style':'bezier',
    'target-arrow-shape':'triangle','target-arrow-color':'#39506b','arrow-scale':0.8}}}}],
  layout:{{name:'preset'}}, minZoom:0.3, maxZoom:3}});
</script></body></html>"""


def render(sub, out: Path) -> None:
    snap = capture_full(sub.arena)
    d = snap.to_dict()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_HTML.format(
        nb=N_BRANCH,
        cells=json.dumps(d["cells"]),
        edges=json.dumps(d["edges"]),
    ))


def main() -> None:
    sub = build_structure()
    a = sub.arena
    n_dend = sum(1 for cid in a.alive_ids().tolist() if a.phenotype_cache[cid].item() == DENDRITE)
    print(f"cells={len(a.alive_ids())}  dendrite-branches={n_dend}  edges={a.edge_cursor}")
    print(f"ranks present: {sorted(set(int(a.rank[c].item()) for c in a.alive_ids().tolist()))}")
    out = Path("runs/viz_demo.html")
    render(sub, out)
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
