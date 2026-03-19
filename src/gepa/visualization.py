# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Candidate tree visualization for GEPA optimization runs.

Generates Graphviz DOT and self-contained HTML visualizations of the
candidate lineage tree.  No external dependencies are required.

Both :class:`~gepa.core.state.GEPAState` and :class:`~gepa.core.result.GEPAResult`
are supported via thin wrappers that extract the required data and delegate
to shared ``*_from_data`` functions.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from gepa.core.state import GEPAState, ProgramIdx


def _escape(text: str) -> str:
    """Escape text for safe inclusion in DOT labels and HTML tooltips."""
    return html.escape(text, quote=True)


# ---------------------------------------------------------------------------
# Data-driven core (works with raw lists/dicts — no GEPAState required)
# ---------------------------------------------------------------------------


def candidate_tree_dot_from_data(
    candidates: Sequence[dict[str, str]],
    parents: Sequence[Sequence[int | None]],
    val_scores: Sequence[float],
    pareto_front_programs: Mapping[Any, set[ProgramIdx]],
) -> str:
    """Generate a Graphviz DOT string from raw optimization data.

    Args:
        candidates: List of candidate dicts (index → component name → text).
        parents: ``parents[i]`` is a list of parent indices for candidate *i*.
        val_scores: Per-candidate aggregate validation score.
        pareto_front_programs: Per-val-example best candidate sets.

    Returns:
        A Graphviz DOT string.
    """
    from gepa.gepa_utils import find_dominator_programs

    n = len(candidates)
    best_idx = max(range(n), key=lambda i: val_scores[i]) if n > 0 else 0
    dominator_ids = set(find_dominator_programs(pareto_front_programs, list(val_scores)))

    dot_lines = [
        "digraph G {",
        "    rankdir=TB;",
        "    node [style=filled, shape=circle, fontsize=14, width=0.6, height=0.6];",
    ]

    for idx in range(n):
        score = val_scores[idx]
        candidate = candidates[idx]
        pars = parents[idx]

        # Tooltip
        tooltip_parts = [f"Candidate {idx}"]
        tooltip_parts.append(f"Val Score: {score:.4f}")
        parent_str = ", ".join(str(p) for p in pars if p is not None) or "seed"
        tooltip_parts.append(f"Parent(s): {parent_str}")
        if idx == best_idx:
            tooltip_parts.append("Role: BEST")
        elif idx in dominator_ids:
            tooltip_parts.append("Role: Pareto Front")
        elif idx == 0:
            tooltip_parts.append("Role: Seed")
        tooltip_parts.append("")
        for comp_name, comp_text in sorted(candidate.items()):
            tooltip_parts.append(f"--- {comp_name} ---")
            tooltip_parts.append(comp_text)

        tooltip = _escape("\n".join(tooltip_parts))
        label = f"{idx}\\n({score:.2f})"

        if idx == best_idx:
            color = "cyan"
        elif idx in dominator_ids:
            color = "orange"
        else:
            color = "lightgray"

        dot_lines.append(f'    {idx} [label="{label}", fillcolor={color}, tooltip=" "];')

    for child in range(n):
        for parent in parents[child]:
            if parent is not None:
                dot_lines.append(f"    {parent} -> {child};")

    dot_lines.append("}")
    return "\n".join(dot_lines)


def candidate_tree_html_from_data(
    candidates: Sequence[dict[str, str]],
    parents: Sequence[Sequence[int | None]],
    val_scores: Sequence[float],
    pareto_front_programs: Mapping[Any, set[ProgramIdx]],
) -> str:
    """Generate a self-contained HTML page from raw optimization data.

    The page uses ``@viz-js/viz`` loaded from a CDN to render the DOT
    graph client-side.  Hovering over a node shows a rich tooltip with
    the full candidate text and metadata.

    Args:
        candidates: List of candidate dicts.
        parents: Lineage list.
        val_scores: Per-candidate aggregate val scores.
        pareto_front_programs: Pareto front mapping.

    Returns:
        A self-contained HTML string.
    """
    from gepa.gepa_utils import find_dominator_programs

    n = len(candidates)
    best_idx = max(range(n), key=lambda i: val_scores[i]) if n > 0 else 0
    dominator_ids = set(find_dominator_programs(pareto_front_programs, list(val_scores)))

    nodes_json_parts: list[str] = []
    for idx in range(n):
        candidate = candidates[idx]
        pars = parents[idx]
        parent_str = ", ".join(str(p) for p in pars if p is not None) or "seed"

        if idx == best_idx:
            role = "Best"
        elif idx in dominator_ids:
            role = "Pareto Front"
        elif idx == 0:
            role = "Seed"
        else:
            role = ""

        node_data = {
            "idx": idx,
            "score": round(val_scores[idx], 4),
            "parents": parent_str,
            "role": role,
            "components": dict(sorted(candidate.items())),
        }
        nodes_json_parts.append(json.dumps(node_data))

    nodes_json = "[" + ",\n".join(nodes_json_parts) + "]"
    dot_string = candidate_tree_dot_from_data(candidates, parents, val_scores, pareto_front_programs)
    dot_escaped = dot_string.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    return _HTML_TEMPLATE.replace("__DOT_STRING__", dot_escaped).replace("__NODES_JSON__", nodes_json)


# ---------------------------------------------------------------------------
# GEPAState convenience wrappers
# ---------------------------------------------------------------------------


def candidate_tree_dot(state: GEPAState) -> str:
    """Generate a Graphviz DOT string from a :class:`GEPAState`."""
    return candidate_tree_dot_from_data(
        candidates=state.program_candidates,
        parents=state.parent_program_for_candidate,
        val_scores=state.program_full_scores_val_set,
        pareto_front_programs=state.program_at_pareto_front_valset,
    )


def candidate_tree_html(state: GEPAState) -> str:
    """Generate a self-contained HTML page from a :class:`GEPAState`."""
    return candidate_tree_html_from_data(
        candidates=state.program_candidates,
        parents=state.parent_program_for_candidate,
        val_scores=state.program_full_scores_val_set,
        pareto_front_programs=state.program_at_pareto_front_valset,
    )


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GEPA Candidate Tree</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { height: 100%; width: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8f9fa; display: flex; flex-direction: column; }
  #header { background: #fff; border-bottom: 1px solid #dee2e6; padding: 12px 20px; display: flex; align-items: center; gap: 16px; flex-shrink: 0; }
  #header h1 { font-size: 18px; font-weight: 600; color: #212529; }
  .legend { display: flex; gap: 14px; font-size: 13px; color: #495057; }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .legend-dot { width: 12px; height: 12px; border-radius: 50%; border: 1px solid #adb5bd; }
  #graph-container { flex: 1 1 auto; width: 100%; overflow: auto; padding: 20px; text-align: center; }
  #graph-container svg { width: 100%; height: 100%; }
  #tooltip {
    display: none; position: fixed; background: #fff; border: 1px solid #dee2e6;
    border-radius: 8px; padding: 14px 16px; max-width: 560px; max-height: 70vh; overflow-y: auto;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18); z-index: 1000; font-size: 13px; line-height: 1.5;
  }
  #tooltip .tt-header { font-weight: 700; font-size: 15px; margin-bottom: 6px; color: #212529; }
  #tooltip .tt-meta { color: #6c757d; margin-bottom: 4px; }
  #tooltip .tt-hint { color: #adb5bd; font-size: 11px; font-style: italic; margin-bottom: 10px; }
  #tooltip .tt-comp-name { font-weight: 600; color: #495057; margin-top: 10px; border-bottom: 1px solid #e9ecef; padding-bottom: 2px; }
  #tooltip .tt-comp-text {
    white-space: pre-wrap; word-break: break-word; background: #f8f9fa;
    border: 1px solid #e9ecef; border-radius: 4px; padding: 8px 10px; margin-top: 4px;
    font-family: "SF Mono", SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px;
    max-height: 200px; overflow-y: auto;
  }
  .role-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-left: 6px; }
  .role-best { background: #00e5ff33; color: #006064; }
  .role-pareto { background: #ff980033; color: #e65100; }
  .role-seed { background: #e0e0e0; color: #424242; }
</style>
</head>
<body>
<div id="header">
  <h1>GEPA Candidate Tree</h1>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:cyan"></div> Best</div>
    <div class="legend-item"><div class="legend-dot" style="background:orange"></div> Pareto Front</div>
    <div class="legend-item"><div class="legend-dot" style="background:lightgray"></div> Other</div>
  </div>
</div>
<div id="graph-container"><p>Loading graph&hellip;</p></div>
<div id="tooltip"></div>

<script type="module">
const NODES = __NODES_JSON__;
const DOT = `__DOT_STRING__`;

const nodeMap = {};
NODES.forEach(n => { nodeMap[n.idx] = n; });

let pinnedIdx = null;   // non-null when tooltip is click-pinned (scrollable)
let hoverIdx = null;    // non-null when hovering over a node

function renderTooltip(idx) {
  const n = nodeMap[idx];
  if (!n) return "";
  let roleBadge = "";
  if (n.role === "Best") roleBadge = '<span class="role-badge role-best">BEST</span>';
  else if (n.role === "Pareto Front") roleBadge = '<span class="role-badge role-pareto">PARETO</span>';
  else if (n.role === "Seed") roleBadge = '<span class="role-badge role-seed">SEED</span>';

  let comps = "";
  for (const [name, text] of Object.entries(n.components)) {
    const escaped = text.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    comps += '<div class="tt-comp-name">' + name + '</div><div class="tt-comp-text">' + escaped + '</div>';
  }

  return '<div class="tt-header">Candidate ' + n.idx + roleBadge + '</div>' +
    '<div class="tt-meta">Score: <strong>' + n.score + '</strong>&nbsp;&nbsp;|&nbsp;&nbsp;Parent(s): ' + n.parents +
    '</div><div class="tt-hint">' + (pinnedIdx === idx ? 'Click node again to dismiss' : 'Click to pin &amp; scroll') + '</div>' +
    comps;
}

function positionTooltip(x, y) {
  const tt = document.getElementById("tooltip");
  let tx = x + 16, ty = y + 16;
  const r = tt.getBoundingClientRect();
  if (tx + r.width > window.innerWidth - 8) tx = x - r.width - 16;
  if (ty + r.height > window.innerHeight - 8) ty = y - r.height - 16;
  if (tx < 8) tx = 8;
  if (ty < 8) ty = 8;
  tt.style.left = tx + "px";
  tt.style.top = ty + "px";
}

function showTooltip(idx, x, y) {
  const tt = document.getElementById("tooltip");
  tt.innerHTML = renderTooltip(idx);
  tt.style.display = "block";
  positionTooltip(x, y);
}

function hideTooltip() {
  pinnedIdx = null;
  hoverIdx = null;
  document.getElementById("tooltip").style.display = "none";
}

// Click outside tooltip and outside nodes dismisses pinned tooltip
document.addEventListener("mousedown", function(e) {
  const tt = document.getElementById("tooltip");
  if (pinnedIdx !== null && tt.style.display === "block"
      && !tt.contains(e.target) && !e.target.closest(".node")) {
    hideTooltip();
  }
});

async function render() {
  const { instance } = await import("https://cdn.jsdelivr.net/npm/@viz-js/viz@3.11.0/lib/viz-standalone.mjs");
  const viz = await instance();
  const svg = viz.renderSVGElement(DOT);
  const container = document.getElementById("graph-container");
  container.innerHTML = "";
  container.appendChild(svg);

  // Attach hover listeners and strip native <title> tooltips to avoid double tooltips
  svg.querySelectorAll(".node").forEach(node => {
    const title = node.querySelector("title");
    if (!title) return;
    const idx = parseInt(title.textContent, 10);
    if (isNaN(idx)) return;
    title.remove();  // remove native SVG tooltip
    node.style.cursor = "pointer";
    // Hover: show tooltip following the mouse (non-interactive)
    node.addEventListener("mouseenter", e => {
      if (pinnedIdx !== null) return;  // don't override a pinned tooltip
      hoverIdx = idx;
      showTooltip(idx, e.clientX, e.clientY);
    });
    node.addEventListener("mousemove", e => {
      if (pinnedIdx !== null || hoverIdx !== idx) return;
      positionTooltip(e.clientX, e.clientY);
    });
    node.addEventListener("mouseleave", () => {
      if (pinnedIdx !== null) return;
      hoverIdx = null;
      document.getElementById("tooltip").style.display = "none";
    });
    // Click: pin the tooltip so user can scroll it
    node.addEventListener("click", e => {
      e.stopPropagation();
      if (pinnedIdx === idx) { hideTooltip(); return; }
      pinnedIdx = idx;
      showTooltip(idx, e.clientX, e.clientY);
    });
  });
  // Also strip <title> from edges and the graph itself
  svg.querySelectorAll(".edge title").forEach(t => t.remove());
  const graphTitle = svg.querySelector(":scope > title");
  if (graphTitle) graphTitle.remove();
}

render().catch(err => {
  document.getElementById("graph-container").innerHTML =
    "<p style='color:red'>Failed to render graph: " + err.message + "</p>" +
    "<pre>" + DOT.replace(/</g,"&lt;") + "</pre>";
});
</script>
</body>
</html>
"""
