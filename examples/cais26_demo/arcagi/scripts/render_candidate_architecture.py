#!/usr/bin/env python3
"""
Render a compact architecture diagram for an ARC-AGI candidate as a PNG.

Defaults to the highest-scoring accepted candidate we discussed:
candidate_idx=5.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    title: str
    body: str
    color: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render candidate architecture PNG")
    parser.add_argument(
        "--all-events",
        type=Path,
        default=Path("all_candidates.json"),
        help="Path to all_candidates.json",
    )
    parser.add_argument(
        "--candidate-idx",
        type=int,
        default=5,
        help="Candidate index to render",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("candidate_5_architecture.png"),
        help="Output PNG path",
    )
    return parser.parse_args()


def setup_matplotlib():
    mpl_dir = Path(".mplconfig")
    mpl_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_dir.resolve()))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    return plt, FancyArrowPatch, FancyBboxPatch


def load_candidate(events_path: Path, candidate_idx: int) -> dict:
    events = json.loads(events_path.read_text())
    for event in events:
        if event.get("candidate_idx") == candidate_idx and event.get("agent_code"):
            return event
    raise SystemExit(f"Could not find accepted candidate_idx={candidate_idx} in {events_path}")


def build_candidate_5_stages() -> list[Stage]:
    return [
        Stage(
            title="Input + Examples",
            body="train_inputs\ntrain_outputs\ntest_inputs",
            color="#dbeafe",
        ),
        Stage(
            title="Generate Code",
            body="Infer the rule\nAsk LLM to write\n`transform(grid)`",
            color="#ede9fe",
        ),
        Stage(
            title="Validate + Fix",
            body="Run on all train pairs\nIf it fails, repair with\nexecution feedback\nup to 2 times",
            color="#fef3c7",
        ),
        Stage(
            title="Run Best Code",
            body="If functional,\nuse code output\nas attempt 1",
            color="#dcfce7",
        ),
        Stage(
            title="Fallback Predict",
            body="Ask LLM for direct\nJSON predictions\nfor attempt 2 / backup",
            color="#fee2e2",
        ),
        Stage(
            title="Output",
            body="Return up to 2 attempts\nper test input",
            color="#ccfbf1",
        ),
    ]


def summarize_candidate(event: dict) -> tuple[str, str]:
    score = event.get("score")
    header = f"Candidate {event['candidate_idx']}  |  score={score:.3f}" if score is not None else f"Candidate {event['candidate_idx']}"
    subtitle = "Code generation with train-set validation and a fallback direct predictor"
    return header, subtitle


def draw_box(ax, patch_cls, x: float, y: float, w: float, h: float, stage: Stage) -> None:
    patch = patch_cls(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.6,
        facecolor=stage.color,
        edgecolor="#0f172a",
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.69,
        stage.title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        family="DejaVu Sans",
        color="#0f172a",
    )
    ax.text(
        x + w / 2,
        y + h * 0.34,
        stage.body,
        ha="center",
        va="center",
        fontsize=10,
        family="DejaVu Sans Mono",
        color="#1f2937",
    )


def draw_arrow(ax, arrow_cls, start: tuple[float, float], end: tuple[float, float], label: str = "") -> None:
    arrow = arrow_cls(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color="#334155",
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    if label:
        x_mid = (start[0] + end[0]) / 2
        y_mid = (start[1] + end[1]) / 2
        ax.text(
            x_mid,
            y_mid + 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            family="DejaVu Sans",
            color="#334155",
        )


def render_candidate_5(event: dict, output_path: Path) -> None:
    plt, FancyArrowPatch, FancyBboxPatch = setup_matplotlib()
    stages = build_candidate_5_stages()
    header, subtitle = summarize_candidate(event)

    fig = plt.figure(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.05,
        0.94,
        "ARC-AGI Agent Architecture",
        fontsize=24,
        fontweight="bold",
        family="DejaVu Sans",
        color="#020617",
    )
    ax.text(
        0.05,
        0.90,
        header,
        fontsize=13,
        family="DejaVu Sans Mono",
        color="#0f766e",
    )
    ax.text(
        0.05,
        0.87,
        subtitle,
        fontsize=12,
        family="DejaVu Sans",
        color="#475569",
    )

    positions = {
        0: (0.06, 0.54, 0.16, 0.18),
        1: (0.27, 0.54, 0.16, 0.18),
        2: (0.48, 0.54, 0.16, 0.18),
        3: (0.69, 0.54, 0.16, 0.18),
        4: (0.27, 0.25, 0.16, 0.18),
        5: (0.48, 0.10, 0.16, 0.16),
    }

    for idx, stage in enumerate(stages):
        draw_box(ax, FancyBboxPatch, *positions[idx], stage)

    draw_arrow(ax, FancyArrowPatch, (0.22, 0.63), (0.27, 0.63))
    draw_arrow(ax, FancyArrowPatch, (0.43, 0.63), (0.48, 0.63))
    draw_arrow(ax, FancyArrowPatch, (0.64, 0.63), (0.69, 0.63))
    draw_arrow(ax, FancyArrowPatch, (0.77, 0.54), (0.58, 0.26), label="attempt 1")
    draw_arrow(ax, FancyArrowPatch, (0.22, 0.54), (0.35, 0.43), label="fallback")
    draw_arrow(ax, FancyArrowPatch, (0.43, 0.25), (0.48, 0.18), label="attempt 2")

    ax.text(
        0.05,
        0.84,
        "Simple view",
        fontsize=13,
        fontweight="bold",
        family="DejaVu Sans",
        color="#0f172a",
    )
    ax.text(
        0.05,
        0.76,
        "Main path: inputs -> generate code -> validate/fix -> run best code -> output\n"
        "Side path: inputs -> fallback predict -> output",
        fontsize=11,
        family="DejaVu Sans Mono",
        color="#334155",
        va="bottom",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    event = load_candidate(args.all_events, args.candidate_idx)
    if args.candidate_idx != 5:
        raise SystemExit("This renderer currently has a hand-tuned diagram layout for candidate_idx=5.")
    render_candidate_5(event, args.output)
    print(f"Saved architecture diagram to {args.output}")


if __name__ == "__main__":
    main()
