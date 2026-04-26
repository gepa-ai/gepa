#!/usr/bin/env python3
"""
Standalone experiment: visualize the architecture of one ARC-AGI candidate.

Usage:
    python viz_experiment.py              # show candidate 5 (best)
    python viz_experiment.py --idx 0     # seed agent
    python viz_experiment.py --all       # all 10 candidates side-by-side
    python viz_experiment.py --save      # save arch_experiment.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# ── matplotlib setup ─────────────────────────────────────────────────────────
mpl_dir = Path(".mplconfig")
mpl_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))

import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.patches import FancyBboxPatch


# ── colour palette ────────────────────────────────────────────────────────────
STYLE_COLORS: dict[str, tuple[str, str]] = {
    "io": ("#e0f2fe", "#0284c7"),  # sky blue
    "llm": ("#ede9fe", "#7c3aed"),  # violet
    "check": ("#fef3c7", "#d97706"),  # amber
    "fix": ("#fee2e2", "#dc2626"),  # red
    "fallback": ("#fce7f3", "#db2777"),  # pink
    "exec": ("#dcfce7", "#16a34a"),  # green
}

BG = "#f1f5f9"  # slate-100 — slightly warmer than pure white
ARROW = "#64748b"  # slate-500
TEXT_HI = "#0f172a"  # slate-950
TEXT_LO = "#64748b"  # slate-500


# ── stage inference ───────────────────────────────────────────────────────────


def infer_stages(code: str) -> list[dict]:
    """Return an ordered list of pipeline stage dicts for *code*."""
    has_code_gen = "def transform" in code or "```python" in code
    has_exec = "exec(" in code
    has_validation = has_exec and (
        "train_outputs" in code
        or "success_count" in code
        or "execution_feedback" in code
        or ("training" in code.lower() and "exec(" in code)
    )
    has_fix_loop = (
        "max_fix" in code
        or "fix_attempt" in code
        or ("for _ in range" in code and "fix" in code.lower())
    )
    has_fallback = "fallback" in code.lower() or (
        code.count("llm(") >= 2 and has_code_gen
    )
    has_multi_hyp = "for _ in range" in code and "llm(" in code
    has_analyst = (
        "analyst" in code.lower()
        or "analysis_prompt" in code
        or "induction_prompt" in code
        or "induction_res" in code
    )

    def stage(label: str, sub: str, style: str) -> dict:
        fc, ec = STYLE_COLORS[style]
        return {"label": label, "sub": sub, "color": fc, "border": ec, "style": style}

    stages: list[dict] = [stage("INPUT", "train pairs · test grids", "io")]

    if has_analyst and has_code_gen:
        stages.append(stage("Analyst", "LLM · describe pattern & rules", "llm"))

    if has_code_gen:
        if has_multi_hyp:
            stages.append(
                stage("Programmer ×N", "LLM · write transform(grid) ×N", "llm")
            )
        else:
            stages.append(stage("Programmer", "LLM · write transform(grid)", "llm"))

    if has_validation:
        stages.append(stage("Validator", "exec code on all training pairs", "check"))
        if has_fix_loop:
            stages.append(stage("Fixer  ≤2×", "LLM · fix with error feedback", "fix"))
    elif has_exec:
        stages.append(stage("Execute", "run transform() on test input", "exec"))

    if not has_code_gen:
        stages.append(stage("LLM Predict", "direct one-shot JSON output", "llm"))
    elif has_fallback:
        stages.append(stage("Fallback", "LLM · direct JSON predict", "fallback"))

    stages.append(stage("OUTPUT", "up to 2 attempts per test input", "io"))
    return stages


# ── drawing ───────────────────────────────────────────────────────────────────

# Layout constants
BOX_H = 0.42  # box height
GAP = 0.040  # gap between boxes — wide enough to see arrows clearly
PAD_X = 0.016  # left/right margin


def _layout(n: int) -> tuple[float, float]:
    """(box_w, left_x) so n boxes + gaps fill [PAD_X, 1-PAD_X]."""
    box_w = (1 - 2 * PAD_X - (n - 1) * GAP) / n
    return box_w, PAD_X


def _cx(i: int, box_w: float) -> float:
    return PAD_X + i * (box_w + GAP) + box_w / 2


def _lx(i: int, box_w: float) -> float:
    return PAD_X + i * (box_w + GAP)


def _rx(i: int, box_w: float) -> float:
    return _lx(i, box_w) + box_w


def draw_arch(
    ax: matplotlib.axes.Axes,
    stages: list[dict],
    candidate_idx: int,
    score: float | None,
    event_type: str = "accepted",
) -> None:
    """Render a clean horizontal flowchart onto *ax*."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)

    n = len(stages)
    box_w, _ = _layout(n)

    # Vertical layout — always vertically centre the full composition
    has_retry = any(s["style"] == "fix" for s in stages)
    has_fallback = any(s["style"] == "fallback" for s in stages)

    RETRY_LANE = 0.13  # height of retry arc lane above boxes
    FB_LANE = 0.28  # height of fallback arc lane above boxes (stacked on retry)
    arc_above = FB_LANE if has_fallback else RETRY_LANE if has_retry else 0.0

    total_h = BOX_H + arc_above  # total composition height
    BOX_BY = (1.0 - total_h) / 2.0  # centre vertically
    BOX_TY = BOX_BY + BOX_H
    box_cy = BOX_BY + BOX_H / 2.0
    RETRY_Y = BOX_TY + RETRY_LANE
    FB_Y = BOX_TY + FB_LANE

    # ── Boxes ─────────────────────────────────────────────────────────────────
    for i, stage in enumerate(stages):
        lx = _lx(i, box_w)
        cx = _cx(i, box_w)
        fc, ec = stage["color"], stage["border"]

        # soft drop shadow
        ax.add_patch(
            FancyBboxPatch(
                (lx + 0.003, BOX_BY - 0.007),
                box_w,
                BOX_H,
                boxstyle="round,pad=0.012",
                facecolor="#00000018",
                edgecolor="none",
                zorder=1,
            )
        )
        # box with slightly thicker border
        ax.add_patch(
            FancyBboxPatch(
                (lx, BOX_BY),
                box_w,
                BOX_H,
                boxstyle="round,pad=0.012",
                facecolor=fc,
                edgecolor=ec,
                linewidth=2.0,
                zorder=2,
            )
        )
        # bold label
        ax.text(
            cx,
            box_cy + BOX_H * 0.12,
            stage["label"],
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=TEXT_HI,
            zorder=3,
        )
        # italic sublabel
        ax.text(
            cx,
            box_cy - BOX_H * 0.20,
            stage["sub"],
            ha="center",
            va="center",
            fontsize=5.5,
            color=TEXT_LO,
            style="italic",
            linespacing=1.3,
            zorder=3,
        )

    # ── Right-pointing arrows ─────────────────────────────────────────────────
    for i in range(1, n):
        ax.annotate(
            "",
            xy=(_lx(i, box_w) - 0.002, box_cy),
            xytext=(_rx(i - 1, box_w) + 0.002, box_cy),
            arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=2.5, mutation_scale=18),
            zorder=4,
        )

    # ── Retry arc: Fixer → up → dashed left → arrowhead into Validator ────────
    fix_idx = next((i for i, s in enumerate(stages) if s["style"] == "fix"), None)
    if fix_idx is not None and fix_idx >= 2:
        val_idx = fix_idx - 1
        cx_f = _cx(fix_idx, box_w)
        cx_v = _cx(val_idx, box_w)
        ax.plot(cx_f, BOX_TY, "o", color="#dc2626", ms=6, zorder=6)
        ax.plot([cx_f, cx_f], [BOX_TY, RETRY_Y], color="#dc2626", lw=1.8, zorder=4)
        ax.plot(
            [cx_f, cx_v],
            [RETRY_Y, RETRY_Y],
            color="#dc2626",
            lw=1.8,
            ls=(0, (6, 3)),
            zorder=4,
        )
        ax.annotate(
            "",
            xy=(cx_v, BOX_TY + 0.003),
            xytext=(cx_v, RETRY_Y),
            arrowprops=dict(
                arrowstyle="-|>", color="#dc2626", lw=1.8, mutation_scale=14
            ),
            zorder=5,
        )
        ax.text(
            (cx_f + cx_v) / 2,
            RETRY_Y + 0.014,
            "retry",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#dc2626",
            zorder=5,
        )

    # ── Fallback arc: INPUT → up → dashed right → arrowhead into Fallback ─────
    fb_idx = next((i for i, s in enumerate(stages) if s["style"] == "fallback"), None)
    if fb_idx is not None and fb_idx > 1:
        cx_in = _cx(0, box_w)
        cx_fb = _cx(fb_idx, box_w)
        ax.plot(cx_in, BOX_TY, "o", color="#db2777", ms=6, zorder=6)
        ax.plot([cx_in, cx_in], [BOX_TY, FB_Y], color="#db2777", lw=1.8, zorder=4)
        ax.plot(
            [cx_in, cx_fb],
            [FB_Y, FB_Y],
            color="#db2777",
            lw=1.8,
            ls=(0, (6, 3)),
            zorder=4,
        )
        ax.annotate(
            "",
            xy=(cx_fb, BOX_TY + 0.003),
            xytext=(cx_fb, FB_Y),
            arrowprops=dict(
                arrowstyle="-|>", color="#db2777", lw=1.8, mutation_scale=14
            ),
            zorder=5,
        )
        ax.text(
            (cx_in + cx_fb) / 2,
            FB_Y + 0.014,
            "parallel fallback",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#db2777",
            zorder=5,
        )


# ── main ──────────────────────────────────────────────────────────────────────


def load_all_candidates() -> list[dict]:
    """
    Return all 10 candidates (0–9) with code, score, and event type.
    Candidate 0 (seed) only exists in accepted_candidates.json.
    Candidates 1–9 come from all_candidates.json.
    """
    accepted = json.loads(Path("accepted_candidates.json").read_text())
    events = json.loads(Path("all_candidates.json").read_text())

    by_idx: dict[int, dict] = {}

    # Seed from accepted (candidate 0)
    for a in accepted:
        idx = a["candidate_idx"]
        by_idx[idx] = {
            "candidate_idx": idx,
            "score": a["score"],
            "event_type": "accepted",
            "agent_code": a["agent_code"],
        }

    # Rest from all_candidates events (proposed_agent_code takes priority)
    for e in events:
        idx = e.get("candidate_idx")
        code = e.get("proposed_agent_code") or e.get("agent_code") or ""
        if idx is None or not code:
            continue
        if idx not in by_idx:
            by_idx[idx] = {
                "candidate_idx": idx,
                "score": e.get("score"),
                "event_type": e.get("event_type", "unknown"),
                "agent_code": code,
            }

    return sorted(by_idx.values(), key=lambda c: c["candidate_idx"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--idx", type=int, default=5, help="Candidate index to visualize"
    )
    parser.add_argument(
        "--save", action="store_true", help="Save instead of displaying"
    )
    parser.add_argument(
        "--each", action="store_true", help="Save each candidate to arch_candidates/"
    )
    parser.add_argument(
        "--grid", action="store_true", help="Bird's-eye grid of ALL candidates"
    )
    args = parser.parse_args()

    all_cands = load_all_candidates()
    print(f"Loaded {len(all_cands)} unique candidates from all_candidates.json")

    # ── Save every candidate as its own PNG ───────────────────────────────────
    if args.each:
        out_dir = Path("arch_candidates")
        out_dir.mkdir(exist_ok=True)
        for cand in all_cands:
            stages = infer_stages(cand["agent_code"])
            fig, ax = plt.subplots(figsize=(11, 3.8))
            fig.patch.set_facecolor(BG)
            draw_arch(
                ax,
                stages,
                cand["candidate_idx"],
                cand["score"],
                event_type=cand["event_type"],
            )
            plt.tight_layout(pad=0.4)
            out = out_dir / f"candidate_{cand['candidate_idx']:02d}.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved → {out}")
        print("Done.")
        return

    # ── Bird's-eye grid ───────────────────────────────────────────────────────
    if args.grid:
        n = len(all_cands)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 9, rows * 3.5))
        fig.patch.set_facecolor(BG)
        fig.suptitle(
            "All Candidates — Architecture Overview",
            fontsize=13,
            fontweight="bold",
            color=TEXT_HI,
            y=1.005,
        )
        for ax_cell, cand in zip(axes.flat, all_cands):
            stages = infer_stages(cand["agent_code"])
            draw_arch(
                ax_cell,
                stages,
                cand["candidate_idx"],
                cand["score"],
                event_type=cand["event_type"],
            )
        for ax_unused in axes.flat[n:]:
            ax_unused.axis("off")
        plt.tight_layout(pad=0.8, h_pad=1.2)
        if args.save:
            out = Path("arch_all_candidates.png")
            fig.savefig(out, dpi=100, bbox_inches="tight")
            print(f"Saved → {out}")
        else:
            plt.show()
        return

    # ── Single candidate ──────────────────────────────────────────────────────
    cand = next((c for c in all_cands if c["candidate_idx"] == args.idx), None)
    if cand is None:
        print(
            f"Candidate {args.idx} not found. Available: {[c['candidate_idx'] for c in all_cands]}"
        )
        return

    stages = infer_stages(cand["agent_code"])
    print(f"Candidate {args.idx}  event={cand['event_type']}  score={cand['score']}")
    for s in stages:
        print(f"  [{s['style']:8s}]  {s['label']}")

    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor(BG)
    draw_arch(
        ax, stages, cand["candidate_idx"], cand["score"], event_type=cand["event_type"]
    )
    plt.tight_layout(pad=0.4)

    if args.save:
        out = Path("arch_experiment.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
