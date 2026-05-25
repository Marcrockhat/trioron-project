"""Export snapshots to a self-contained HTML viewer.  See spec §7.3."""
from __future__ import annotations

import json
from pathlib import Path


_VIEWER_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Trioron Substrate Viewer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape-cose-bilkent/4.1.0/cytoscape-cose-bilkent.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', monospace; background: #0d1117; color: #c9d1d9; }}
#app {{ display: flex; height: 100vh; }}
#cy {{ flex: 1; background: #0d1117; }}
#panel {{ width: 340px; overflow-y: auto; padding: 16px; background: #161b22;
          border-left: 1px solid #30363d; font-size: 13px; }}
#timeline {{ position: fixed; bottom: 0; left: 0; right: 340px; height: 52px;
             background: #161b22; border-top: 1px solid #30363d;
             display: flex; align-items: center; padding: 0 20px; gap: 12px; }}
#timeline input[type=range] {{ flex: 1; }}
#timeline button {{ background: #238636; color: #fff; border: none; padding: 6px 14px;
                    border-radius: 6px; cursor: pointer; font-size: 13px; }}
#timeline button:hover {{ background: #2ea043; }}
h2 {{ margin: 12px 0 6px; font-size: 14px; color: #58a6ff; }}
.stat {{ color: #8b949e; font-size: 12px; }}
.info-row {{ padding: 4px 0; border-bottom: 1px solid #21262d; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }}
.legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 11px; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
#snap-label {{ color: #58a6ff; font-weight: bold; white-space: nowrap; min-width: 120px; }}
#selected {{ margin-top: 12px; }}
</style>
</head>
<body>
<div id="app">
  <div id="cy"></div>
  <div id="panel">
    <h2>Trioron v2.0</h2>
    <div id="info"></div>
    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:#58a6ff"></div>Perception</div>
      <div class="legend-item"><div class="legend-dot" style="background:#f0883e"></div>Output</div>
      <div class="legend-item"><div class="legend-dot" style="background:#7ee787"></div>CONV</div>
      <div class="legend-item"><div class="legend-dot" style="background:#d2a8ff"></div>ATTENTION</div>
      <div class="legend-item"><div class="legend-dot" style="background:#8b949e"></div>LINEAR</div>
      <div class="legend-item"><div class="legend-dot" style="background:#f85149"></div>Stem</div>
    </div>
    <h2>Selected</h2>
    <div id="selected" class="stat">Click a node</div>
    <h2>Events</h2>
    <div id="events" class="stat"></div>
  </div>
</div>
<div id="timeline">
  <button id="btn-prev">&lt;</button>
  <input type="range" id="scrubber" min="0" max="0" value="0">
  <button id="btn-next">&gt;</button>
  <span id="snap-label">t000</span>
</div>

<script>
const SNAPSHOTS = {snapshots_json};

const PHENO_COLORS = {{
  'conv': '#7ee787', 'attention': '#d2a8ff', 'linear': '#8b949e',
  'recurrent': '#3fb950', 'dendrite': '#db61a2'
}};

function cellColor(c) {{
  if (c.epigenome === 0 && c.phenotype === 'linear') return '#f85149'; // stem
  const epi = c.epigenome;
  if (epi & (1 << 5)) return '#58a6ff'; // perception
  if (epi & (1 << 6)) return '#f0883e'; // output
  return PHENO_COLORS[c.phenotype] || '#8b949e';
}}

function cellShape(c) {{
  const epi = c.epigenome;
  if (epi & (1 << 5)) return 'diamond';     // perception
  if (epi & (1 << 6)) return 'triangle';     // output
  if (c.epigenome === 0) return 'star';       // stem
  if (!c.forward_inclusion) return 'diamond'; // astrocyte
  return 'ellipse';
}}

function cellSize(c) {{
  const epi = c.epigenome;
  if (epi & (1 << 5)) return 6;   // perception: tiny
  if (epi & (1 << 6)) return 18;  // output: big
  return 10 + Math.min(10, c.engagement * 15);
}}

let cy = null;
let currentIdx = 0;

function loadSnapshot(idx) {{
  currentIdx = idx;
  const snap = SNAPSHOTS[idx];
  if (!snap) return;

  document.getElementById('snap-label').textContent =
    `[${{idx}}] t${{snap.task_index}} ${{snap.kind}} — ${{snap.cells.length}} cells`;

  // Info panel
  const nPerc = snap.cells.filter(c => c.epigenome & (1 << 5)).length;
  const nOut = snap.cells.filter(c => c.epigenome & (1 << 6)).length;
  const nInt = snap.cells.length - nPerc - nOut;
  document.getElementById('info').innerHTML = `
    <div class="info-row">Cells: ${{snap.cells.length}} (perc=${{nPerc}} int=${{nInt}} out=${{nOut}})</div>
    <div class="info-row">Edges: ${{snap.edges ? snap.edges.length : '—'}}</div>
    <div class="info-row">Task: ${{snap.task_index}}</div>
  `;

  // Events
  const evts = snap.events_since_last_snapshot || [];
  document.getElementById('events').innerHTML = evts.length
    ? evts.map(e => `<div class="info-row">${{e.type}} cell=${{e.cell_id}}</div>`).join('')
    : '<div>No events</div>';

  // Filter: skip perception cells if > 100 (too many to render)
  let cells = snap.cells;
  let edges = snap.edges || [];
  const skipPerc = nPerc > 100;
  if (skipPerc) {{
    const percIds = new Set(cells.filter(c => c.epigenome & (1 << 5)).map(c => c.id));
    cells = cells.filter(c => !(c.epigenome & (1 << 5)));
    edges = edges.filter(e => !percIds.has(e.src) && !percIds.has(e.dst));
  }}

  // Build cytoscape elements
  const elements = [];
  const nodeIds = new Set();

  cells.forEach(c => {{
    nodeIds.add(c.id);
    elements.push({{
      group: 'nodes',
      data: {{
        id: 'n' + c.id,
        label: c.id,
        color: cellColor(c),
        shape: cellShape(c),
        size: cellSize(c),
        opacity: c.state === 'dormant' ? 0.4 : 1.0,
        z: c.position[2],
        ...c,
      }}
    }});
  }});

  edges.forEach((e, i) => {{
    if (nodeIds.has(e.src) && nodeIds.has(e.dst)) {{
      elements.push({{
        group: 'edges',
        data: {{
          id: 'e' + i,
          source: 'n' + e.src,
          target: 'n' + e.dst,
          weight: e.weight,
        }}
      }});
    }}
  }});

  if (cy) cy.destroy();

  cy = cytoscape({{
    container: document.getElementById('cy'),
    elements: elements,
    style: [
      {{
        selector: 'node',
        style: {{
          'background-color': 'data(color)',
          'shape': 'data(shape)',
          'width': 'data(size)',
          'height': 'data(size)',
          'opacity': 'data(opacity)',
          'label': '',
          'border-width': 0,
        }}
      }},
      {{
        selector: 'node:selected',
        style: {{
          'border-width': 2,
          'border-color': '#fff',
          'label': 'data(label)',
          'font-size': 10,
          'color': '#fff',
        }}
      }},
      {{
        selector: 'edge',
        style: {{
          'width': 0.5,
          'line-color': function(e) {{
            return e.data('weight') > 0 ? 'rgba(80,180,80,0.25)' : 'rgba(180,80,80,0.25)';
          }},
          'curve-style': 'bezier',
          'target-arrow-shape': 'triangle',
          'target-arrow-color': 'rgba(100,100,100,0.3)',
          'arrow-scale': 0.4,
        }}
      }}
    ],
    layout: {{
      name: 'cose-bilkent',
      quality: 'draft',
      nodeDimensionsIncludeLabels: false,
      idealEdgeLength: 60,
      nodeRepulsion: 8000,
      gravity: 0.25,
      gravityRange: 3.8,
      animate: false,
      tile: true,
    }},
    minZoom: 0.1,
    maxZoom: 5,
  }});

  cy.on('tap', 'node', function(evt) {{
    const d = evt.target.data();
    document.getElementById('selected').innerHTML = `
      <div class="info-row"><b>#${{d.label}}</b> ${{d.phenotype}} [${{d.state}}]</div>
      <div class="info-row">Position: (${{d.position.map(v=>v.toFixed(2)).join(', ')}})</div>
      <div class="info-row">Rank: ${{d.rank}} | Engagement: ${{d.engagement.toFixed(4)}}</div>
      <div class="info-row">Utility: ${{d.utility.toFixed(4)}} | Age: ${{d.age}}</div>
      <div class="info-row">Parent: ${{d.parent}} | Lineage: ${{d.lineage_root}}</div>
      <div class="info-row">Epigenome: 0x${{d.epigenome.toString(16)}}</div>
    `;
  }});
}}

// Timeline controls
const scrubber = document.getElementById('scrubber');
scrubber.max = Math.max(0, SNAPSHOTS.length - 1);
scrubber.addEventListener('input', e => loadSnapshot(parseInt(e.target.value)));
document.getElementById('btn-prev').addEventListener('click', () => {{
  if (currentIdx > 0) {{ scrubber.value = currentIdx - 1; loadSnapshot(currentIdx - 1); }}
}});
document.getElementById('btn-next').addEventListener('click', () => {{
  if (currentIdx < SNAPSHOTS.length - 1) {{ scrubber.value = currentIdx + 1; loadSnapshot(currentIdx + 1); }}
}});

loadSnapshot(0);
</script>
</body>
</html>
"""


def export_html(
    snapshot_dir: str | Path,
    output: str | Path,
    structures: dict | None = None,
) -> Path:
    """Bundle snapshots into a Cytoscape.js network viewer."""
    snap_dir = Path(snapshot_dir)
    out_path = Path(output)

    snapshots = []
    for f in sorted(snap_dir.glob("*.json")):
        snapshots.append(json.loads(f.read_text()))

    html = _VIEWER_TEMPLATE.format(
        snapshots_json=json.dumps(snapshots),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
