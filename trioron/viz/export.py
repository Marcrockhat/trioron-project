"""Export snapshots to a self-contained HTML viewer.  See spec §7.3."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


_VIEWER_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Trioron Substrate Viewer</title>
<style>
body {{ margin: 0; font-family: monospace; background: #1a1a2e; color: #e0e0e0; }}
#container {{ display: flex; height: 100vh; }}
#scene {{ flex: 1; position: relative; }}
#panel {{ width: 320px; overflow-y: auto; padding: 12px; background: #16213e;
          border-left: 1px solid #333; font-size: 13px; }}
#timeline {{ position: fixed; bottom: 0; left: 0; right: 320px; height: 48px;
             background: #0f3460; display: flex; align-items: center; padding: 0 16px; }}
#timeline input {{ flex: 1; }}
#timeline label {{ margin-left: 12px; white-space: nowrap; }}
canvas {{ display: block; }}
h2 {{ margin: 8px 0; font-size: 15px; color: #e94560; }}
.cell-list {{ max-height: 400px; overflow-y: auto; }}
.cell-item {{ padding: 4px 0; border-bottom: 1px solid #222; cursor: pointer; }}
.cell-item:hover {{ background: #1a1a4e; }}
.stat {{ color: #888; }}
</style>
</head>
<body>
<div id="container">
  <div id="scene">
    <canvas id="canvas"></canvas>
  </div>
  <div id="panel">
    <h2>Trioron v2.0 Viewer</h2>
    <div id="info">Loading...</div>
    <h2>Cells</h2>
    <div id="cell-list" class="cell-list"></div>
    <h2>Structures</h2>
    <div id="structures"></div>
  </div>
</div>
<div id="timeline">
  <input type="range" id="scrubber" min="0" max="0" value="0">
  <label id="snap-label">t000</label>
</div>

<script>
const SNAPSHOTS = {snapshots_json};
const STRUCTURES = {structures_json};

let currentIdx = 0;

function renderSnapshot(idx) {{
  currentIdx = idx;
  const snap = SNAPSHOTS[idx];
  if (!snap) return;

  document.getElementById('snap-label').textContent =
    `t${{String(snap.task_index).padStart(3,'0')}} [${{snap.kind}}] ${{snap.cells.length}} cells`;

  const info = document.getElementById('info');
  info.innerHTML = `
    <div>Task: ${{snap.task_index}}</div>
    <div>Cells: ${{snap.cells.length}}</div>
    <div>Edges: ${{snap.edges ? snap.edges.length : '(delta)'}}</div>
    <div>Kind: ${{snap.kind}}</div>
    <div>Time: ${{snap.timestamp_iso || '-'}}</div>
  `;

  const list = document.getElementById('cell-list');
  list.innerHTML = snap.cells.map(c => `
    <div class="cell-item" title="id=${{c.id}}">
      <b>#${{c.id}}</b> ${{c.phenotype}} [${{c.state}}]
      <span class="stat">eng=${{c.engagement.toFixed(3)}} util=${{c.utility.toFixed(4)}} rank=${{c.rank}}</span>
    </div>
  `).join('');

  const stDiv = document.getElementById('structures');
  const snapStructs = STRUCTURES.structures_by_snapshot?.[idx];
  if (snapStructs && snapStructs.structures.length > 0) {{
    stDiv.innerHTML = snapStructs.structures.map(s =>
      `<div class="cell-item"><b>${{s.type}}</b> ${{s.label}} (${{s.cell_ids.length}} cells)</div>`
    ).join('');
  }} else {{
    stDiv.innerHTML = '<div class="stat">No structures detected</div>';
  }}

  renderCanvas(snap);
}}

function renderCanvas(snap) {{
  const canvas = document.getElementById('canvas');
  const parent = canvas.parentElement;
  canvas.width = parent.clientWidth;
  canvas.height = parent.clientHeight - 48;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const w = canvas.width;
  const h = canvas.height;
  const pad = 40;

  const phenoColors = {{
    linear: '#888', attention: '#4488ff', conv: '#ff4444',
    recurrent: '#44cc44', dendrite: '#aa44ff'
  }};

  // Draw edges
  if (snap.edges) {{
    ctx.strokeStyle = 'rgba(100,100,100,0.3)';
    ctx.lineWidth = 0.5;
    const cellMap = {{}};
    snap.cells.forEach(c => cellMap[c.id] = c);
    snap.edges.forEach(e => {{
      const s = cellMap[e.src], d = cellMap[e.dst];
      if (!s || !d) return;
      ctx.beginPath();
      ctx.moveTo(pad + s.position[0] * (w - 2*pad), pad + s.position[1] * (h - 2*pad));
      ctx.lineTo(pad + d.position[0] * (w - 2*pad), pad + d.position[1] * (h - 2*pad));
      ctx.stroke();
    }});
  }}

  // Draw cells
  snap.cells.forEach(c => {{
    const x = pad + c.position[0] * (w - 2*pad);
    const y = pad + c.position[1] * (h - 2*pad);
    const r = c.state === 'dormant' ? 3 : 5;
    const color = phenoColors[c.phenotype] || '#888';
    const alpha = c.state === 'dormant' ? 0.4 : 1.0;

    ctx.beginPath();
    if (c.forward_inclusion) {{
      ctx.arc(x, y, r, 0, Math.PI * 2);
    }} else {{
      // Diamond for astrocytes
      ctx.moveTo(x, y - r); ctx.lineTo(x + r, y);
      ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath();
    }}
    ctx.fillStyle = color;
    ctx.globalAlpha = alpha;
    ctx.fill();
    ctx.globalAlpha = 1.0;
  }});
}}

// Timeline scrubber
const scrubber = document.getElementById('scrubber');
scrubber.max = Math.max(0, SNAPSHOTS.length - 1);
scrubber.addEventListener('input', e => renderSnapshot(parseInt(e.target.value)));

window.addEventListener('resize', () => renderSnapshot(currentIdx));
renderSnapshot(0);
</script>
</body>
</html>
"""


def export_html(
    snapshot_dir: str | Path,
    output: str | Path,
    structures: dict | None = None,
) -> Path:
    """Bundle snapshots + structures into a self-contained HTML viewer."""
    snap_dir = Path(snapshot_dir)
    out_path = Path(output)

    snapshots = []
    for f in sorted(snap_dir.glob("*.json")):
        snapshots.append(json.loads(f.read_text()))

    structs = structures or {"structures_by_snapshot": []}

    html = _VIEWER_TEMPLATE.format(
        snapshots_json=json.dumps(snapshots),
        structures_json=json.dumps(structs),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
