#!/usr/bin/env python3
"""
Render a rich, interactive evolution report (HTML) from a TurboGEPA run.

Inputs:  .turbo_gepa/evolution/<run_id>.json  (written by DefaultAdapter)
Outputs: .turbo_gepa/evolution/<run_id>.html  (self-contained, CDN assets)

Features:
- Interactive graph (Cytoscape + Dagre) with pan/zoom
- Click a node to inspect fingerprint, shard, quality, and full prompt
- Search + filters (substring, rung %, minimum quality)
- Progress chart: best quality vs elapsed time (Chart.js)

Usage:
  source .envrc && source .venv/bin/activate && \
  python scripts/render_evolution.py --input .turbo_gepa/evolution/<run_id>.json

If --input is omitted, the most recent evolution JSON is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _latest_evolution_json(root: Path) -> Path | None:
    root.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_graph_data(parent_children: Dict[str, List[str]], lineage: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Create Cytoscape elements and a details map from lineage and edges."""
    qualities: Dict[str, float] = {}
    statuses: Dict[str, str] = {}
    shards: Dict[str, float] = {}
    prompts: Dict[str, str] = {}
    prompts_full: Dict[str, str] = {}

    for item in lineage:
        fp = str(item.get("fingerprint"))
        q = item.get("quality")
        if isinstance(q, (int, float)):
            qualities[fp] = float(q)
        s = item.get("status")
        if isinstance(s, str):
            statuses[fp] = s
        sf = item.get("shard_fraction")
        if isinstance(sf, (int, float)):
            shards[fp] = float(sf)
        p = item.get("prompt")
        if isinstance(p, str):
            prompts[fp] = p
        pf = item.get("prompt_full")
        if isinstance(pf, str):
            prompts_full[fp] = pf

    # Collect unique node ids
    fps: set[str] = set(qualities.keys()) | set(shards.keys()) | set(prompts.keys()) | set(parent_children.keys())
    for children in parent_children.values():
        fps.update(children)

    def node_label(fp: str) -> str:
        q = qualities.get(fp)
        shard = shards.get(fp)
        label_q = f" q={q:.3f}" if isinstance(q, float) else ""
        label_s = f" ({int(shard*100)}%)" if isinstance(shard, float) else ""
        return f"{fp[:8]}{label_q}{label_s}"

    elements: List[Dict[str, Any]] = []
    details: Dict[str, Dict[str, Any]] = {}

    for fp in fps:
        sid = fp[:8]
        q = qualities.get(fp)
        rung = (shards.get(fp) or 0.0) * 100.0
        elements.append({
            "data": {
                "id": sid,
                "label": node_label(fp),
                "status": statuses.get(fp, "other"),
                "rung": rung,
                "scored": 1 if (isinstance(q, (int, float)) or rung > 0) else 0,
            }
        })
        details[sid] = {
            "fingerprint": fp,
            "quality": qualities.get(fp),
            "status": statuses.get(fp),
            "shard_fraction": shards.get(fp),
            "prompt": prompts.get(fp, ""),
            "prompt_full": prompts_full.get(fp, prompts.get(fp, "")),
        }

    for parent, children in parent_children.items():
        p = parent[:8]
        for child in children:
            c = child[:8]
            elements.append({"data": {"id": f"{p}-{c}", "source": p, "target": c}})

    return elements, details


def _render_html(elements: List[Dict[str, Any]], title: str, details: Dict[str, Dict[str, Any]], timeline: List[Dict[str, Any]]) -> str:
    """Return a self-contained HTML page rendering a Cytoscape graph."""
    template = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #0B1020; color: #F2F5F9; }}
    header {{ padding: 12px 16px; border-bottom: 1px solid #1F2A44; background: #0E1530; }}
    .container {{ padding: 16px; display: grid; grid-template-columns: 2.2fr 1fr; gap: 16px; align-items: start; }}
    .legend {{ margin-bottom: 12px; font-size: 14px; }}
    .legend span {{ display: inline-block; padding: 4px 8px; border-radius: 6px; margin-right: 8px; }}
    .lg-promoted {{ background: #A3D977; color: #102A00; }}
    .lg-inflight {{ background: #F7D070; color: #3A2A00; }}
    .lg-other {{ background: #DDE3EA; color: #1F2A37; }}
    .panel {{ background: #0E1530; border: 1px solid #1F2A44; border-radius: 8px; padding: 12px; }}
    .panel h3 {{ margin: 0 0 8px 0; font-size: 16px; }}
    .kv {{ font-size: 13px; line-height: 1.4; }}
    .kv dt {{ color: #9FB0C0; }}
    .kv dd {{ margin: 0 0 6px 0; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; white-space: pre-wrap; word-break: break-word; }}
    #graph {{ height: 520px; border-radius: 8px; background: #0B1020; border: 1px solid #1F2A44; }}
    .controls {{ display: flex; gap: 8px; margin: 8px 0 12px; }}
    .controls input {{ background: #0B1530; border: 1px solid #1F2A44; color: #E4EDF6; padding: 6px 8px; border-radius: 6px; }}
  </style>
  <script src=\"https://unpkg.com/cytoscape@3.28.0/dist/cytoscape.min.js\"></script>
  <script src=\"https://unpkg.com/dagre@0.8.5/dist/dagre.min.js\"></script>
  <script src=\"https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js\"></script>
</head>
<body>
  <header><strong>TurboGEPA Evolution</strong> — {title}</header>
  <div class=\"container\"> 
    <div>
      <div class=\"legend\">
        <span class=\"lg-promoted\">promoted</span>
        <span class=\"lg-inflight\">in_flight</span>
        <span class=\"lg-other\">other</span>
        <span style=\"opacity:.8\">(label shows fingerprint, q, rung)</span>
      </div>
      <div class=\"panel\">
        <div class=\"controls\">
          <input id=\"search\" placeholder=\"Search fingerprint or prompt...\" />
          <input id=\"rung\" placeholder=\"Filter rung % (e.g., 100)\" />
          <input id=\"minq\" placeholder=\"Min q (e.g., 0.7)\" />
        </div>
        <div id=\"graph\"></div>
      </div>
      <div class=\"panel\">
        <h3>Progress</h3>
        <canvas id=\"qTime\" height=\"120\"></canvas>
      </div>
    </div>
    <div class=\"panel\" id=\"details\">
      <h3>Candidate details</h3>
      <dl class=\"kv\">
        <dt>Fingerprint</dt><dd id=\"d-fp\"></dd>
        <dt>Quality</dt><dd id=\"d-q\"></dd>
        <dt>Shard</dt><dd id=\"d-s\"></dd>
      </dl>
      <div class=\"mono\" id=\"d-prompt\"></div>
    </div>
  </div>
  <script>
    const DETAILS = {details_json};
    const TL = {timeline_json};
    const ELEMENTS = {elements_json};

    if (window.cytoscape) {{
      try {{ cytoscape.use(window.cytoscapeDagre); }} catch (e) {{ /* ignore */ }}
      const cy = cytoscape({{
        container: document.getElementById('graph'),
        elements: ELEMENTS,
        layout: {{ name: 'dagre', nodeSep: 20, rankSep: 50, rankDir: 'LR' }},
        style: [
          {{ selector: 'node', style: {{ 'background-color': '#22324E', 'label': 'data(label)', 'color': '#E4EDF6', 'text-valign': 'center', 'text-halign': 'center', 'shape': 'round-rectangle', 'border-width': 1, 'border-color': '#5B6B7A', 'font-size': 10, 'padding': 6 }} }},
          {{ selector: 'node[status = "promoted"]', style: {{ 'background-color': '#A3D977', 'border-color': '#335C0D', 'color': '#102A00' }} }},
          {{ selector: 'node[status = "in_flight"]', style: {{ 'background-color': '#F7D070', 'border-color': '#8B5E00', 'color': '#3A2A00' }} }},
          {{ selector: 'node[scored = 0]', style: {{ 'opacity': 0.25 }} }},
          {{ selector: 'node[rung >= 99]', style: {{ 'border-width': 2, 'border-color': '#6BE675' }} }},
          {{ selector: 'node[rung >= 50][rung < 99]', style: {{ 'border-width': 2, 'border-color': '#6EC5FF' }} }},
          {{ selector: 'edge', style: {{ 'width': 1, 'line-color': '#3C4A5E', 'target-arrow-color': '#3C4A5E', 'target-arrow-shape': 'vee', 'curve-style': 'bezier' }} }}
        ],
      }});

      const showDetails = (id) => {{
        const d = DETAILS[id];
        if (!d) return;
        document.getElementById('d-fp').textContent = d.fingerprint || id;
        document.getElementById('d-q').textContent = (d.quality!=null? d.quality.toFixed(3):'-');
        document.getElementById('d-s').textContent = (d.shard_fraction!=null? (d.shard_fraction*100).toFixed(0)+'%':'-');
        document.getElementById('d-prompt').textContent = d.prompt_full || d.prompt || '';
      }};

      cy.on('tap', 'node', (evt) => {{ showDetails(evt.target.id()); }});
      cy.ready(() => {{
        // Default-select highest rung then highest quality
        const nodes = cy.nodes().map(n => n.id());
        const best = nodes
          .map(id => ({{ id, rung: (DETAILS[id]?.shard_fraction||0), q: (DETAILS[id]?.quality||0) }}))
          .sort((a,b) => (b.rung - a.rung) || (b.q - a.q))[0];
        if (best) showDetails(best.id);
      }});

      // Simple search/filter controls
      const q = document.getElementById('search');
      const rung = document.getElementById('rung');
      const minq = document.getElementById('minq');
      const applyFilter = () => {{
        const qv = (q.value||'').toLowerCase();
        const rungv = parseFloat(rung.value);
        const minqv = parseFloat(minq.value);
        cy.nodes().forEach(n => {{
          const id = n.id();
          const d = DETAILS[id]||{{}};
          let ok = true;
          if (qv) {{
            ok = (d.fingerprint||'').toLowerCase().includes(qv) || (d.prompt_full||'').toLowerCase().includes(qv);
          }}
          if (ok && !Number.isNaN(rungv)) {{
            ok = Math.round((d.shard_fraction||0)*100) === Math.round(rungv);
          }}
          if (ok && !Number.isNaN(minqv)) {{
            ok = (d.quality||0) >= minqv;
          }}
          n.style('opacity', ok ? 1.0 : 0.1);
        }});
      }};
      q.addEventListener('input', applyFilter);
      rung.addEventListener('input', applyFilter);
      minq.addEventListener('input', applyFilter);
    }}

    // Progress chart: best quality vs elapsed
    (function() {{
      try {{
        const ctx = document.getElementById('qTime');
        const labels = TL.map(p => (p.elapsed||0).toFixed(1));
        const data = TL.map(p => p.best_quality||0);
        new Chart(ctx, {{ type: 'line', data: {{ labels, datasets: [{{ label: 'Best quality', data, borderColor: '#6EC5FF', tension: 0.2 }}] }}, options: {{ scales: {{ y: {{ min: 0, max: 1 }} }} }} }});
      }} catch (e) {{ console.error('Chart error', e); }}
    }})();
  </script>
</body>
</html>
"""
    import json as _json
    return template.format(
        title=title,
        details_json=_json.dumps(details, ensure_ascii=False),
        timeline_json=_json.dumps(timeline or [], ensure_ascii=False),
        elements_json=_json.dumps(elements, ensure_ascii=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render TurboGEPA evolution graph (HTML)")
    parser.add_argument("--input", type=Path, default=None, help="Path to evolution JSON (.turbo_gepa/evolution/<run_id>.json)")
    parser.add_argument("--output", type=Path, default=None, help="Optional output HTML path")
    args = parser.parse_args()

    evo_root = Path(".turbo_gepa/evolution")
    src = args.input or _latest_evolution_json(evo_root)
    if not src or not Path(src).exists():
        raise SystemExit("No evolution JSON found. Run a TurboGEPA job first.")

    payload = _load_payload(Path(src))
    run_id = str(payload.get("run_id") or Path(src).stem)
    evo_stats = payload.get("evolution_stats") or {}
    lineage = payload.get("lineage") or []
    parent_children = evo_stats.get("parent_children") or {}

    elements, details = _build_graph_data(parent_children, lineage)
    title = f"Run {run_id}"
    html = _render_html(elements, title, details, payload.get("timeline") or [])

    out = args.output or evo_root / f"{run_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"✅ Wrote evolution HTML → {out}")


if __name__ == "__main__":
    main()
