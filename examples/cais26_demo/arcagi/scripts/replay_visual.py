#!/usr/bin/env python3
"""
Visual replay for ARC-AGI candidate exploration.

Uses:
- accepted_candidates.json
- all_candidates.json

Two panels (stacked):
  Top:    validation score over time
  Bottom: architecture diagram of the current best accepted candidate
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual replay for ARC-AGI candidates")
    parser.add_argument(
        "--accepted",
        type=Path,
        default=Path("accepted_candidates.json"),
        help="Path to accepted_candidates.json",
    )
    parser.add_argument(
        "--all-events",
        type=Path,
        default=Path("all_candidates.json"),
        help="Path to all_candidates.json",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between replay steps",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run headlessly for verification",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Only replay the first N events",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="Only show accepted candidates discovered at or before this metric-call cutoff",
    )
    return parser.parse_args()


def truncate_code(
    code_text: str | None, max_lines: int = 26, max_chars: int = 1800
) -> str:
    if not code_text:
        return "<no code available>"
    lines = code_text.strip().splitlines()
    clipped = "\n".join(lines[:max_lines])
    if len(clipped) > max_chars:
        clipped = clipped[: max_chars - 3].rstrip() + "..."
    elif len(lines) > max_lines:
        clipped += "\n..."
    return clipped


def infer_event_metric_calls(accepted: list[dict], events: list[dict]) -> list[int]:
    cumulative = []
    total = 0
    for event in events:
        total += len(event.get("subsample_ids") or [])
        total += len(event.get("evaluated_val_indices") or [])
        cumulative.append(total)

    accepted_metric_calls = {
        item["candidate_idx"]: item["metric_calls_by_discovery"] for item in accepted
    }
    offsets = []
    for raw_total, event in zip(cumulative, events):
        candidate_idx = event.get("candidate_idx")
        if candidate_idx is None:
            continue
        accepted_total = accepted_metric_calls.get(candidate_idx)
        if accepted_total is not None:
            offsets.append(accepted_total - raw_total)

    offset = offsets[0] if offsets else 0
    return [raw_total + offset for raw_total in cumulative]


def filter_by_metric_calls(
    accepted: list[dict], events: list[dict], max_metric_calls: int | None
) -> tuple[list[dict], list[dict]]:
    if max_metric_calls is None:
        return accepted, events

    accepted_filtered = [
        item
        for item in accepted
        if item["metric_calls_by_discovery"] <= max_metric_calls
    ]
    if not accepted_filtered:
        return [accepted[0]], []

    discovery_by_candidate = {
        item["candidate_idx"]: item["metric_calls_by_discovery"] for item in accepted
    }
    first_event_past_cutoff: int | None = None
    for idx, event in enumerate(events):
        candidate_idx = event.get("candidate_idx")
        if candidate_idx is None:
            continue
        discovery_calls = discovery_by_candidate.get(candidate_idx)
        if discovery_calls is not None and discovery_calls > max_metric_calls:
            first_event_past_cutoff = idx
            break

    if first_event_past_cutoff is None:
        return accepted_filtered, events
    return accepted_filtered, events[:first_event_past_cutoff]


def setup_matplotlib(no_show: bool):
    mpl_dir = Path(".mplconfig")
    mpl_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_dir.resolve()))
    if no_show:
        os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    return plt, Line2D


def get_screen_geometry() -> tuple[int, int] | None:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        return None


def pretty_print(event: dict, metric_call: int) -> None:
    reason = event.get("event_reason", event["event_type"])
    print("=" * 96)
    print(
        f"iteration={event['iteration']:>2} | "
        f"metric_calls={metric_call:>4} | "
        f"type={event['event_type']:<12} | "
        f"parent={event['selected_program_candidate']} "
        f"({event['selected_program_score'] * 100:.1f}%)"
        + (
            f" | proposal={event['candidate_idx']} ({event['score'] * 100:.1f}%)"
            if event["candidate_idx"] is not None and event["score"] is not None
            else ""
        )
    )
    print(
        f"Reason: {reason} | "
        f"subsample parent={event['subsample_score_sum']}"
        + (
            f" -> proposal={event['new_subsample_score_sum']}"
            if event["new_subsample_score_sum"] is not None
            else ""
        )
    )
    code = (
        event.get("agent_code")
        or event.get("proposed_agent_code")
        or event.get("selected_program_code")
    )
    label = (
        "Accepted candidate code"
        if event.get("agent_code")
        else "Parent code reference"
    )
    print(f"{label}:")
    print(truncate_code(code))
    print()


# ---------------------------------------------------------------------------
# Architecture helpers — imported from viz_experiment
# ---------------------------------------------------------------------------


def _import_viz():
    """Lazy-import infer_stages and draw_arch from viz_experiment.py."""
    import importlib.util, sys

    spec = importlib.util.spec_from_file_location(
        "viz_experiment",
        Path(__file__).parent / "viz_experiment.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.infer_stages, mod.draw_arch


def render(
    accepted: list[dict], events: list[dict], delay: float, no_show: bool
) -> None:
    plt, Line2D = setup_matplotlib(no_show)
    infer_stages, draw_arch = _import_viz()

    screen = None if no_show else get_screen_geometry()
    if screen is not None:
        screen_width, screen_height = screen
        fig_w = max(screen_width // 2, 1200) / 100
        fig_h = max(screen_height, 900) / 100
    else:
        fig_w, fig_h = 14, 10

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)

    # ── Two rows: score chart (top 65%) + arch diagram (bottom 35%) ───────────
    # add_axes([left, bottom, width, height]) in figure fraction
    ax_score = fig.add_axes([0.06, 0.38, 0.91, 0.58])  # top panel
    ax_arch = fig.add_axes([0.01, 0.01, 0.98, 0.34])  # bottom panel

    accepted_scores = [item["score"] for item in accepted]
    candidate_scores = [
        event["score"]
        if event["score"] is not None
        else event["selected_program_score"]
        for event in events
    ]
    y_min = min(accepted_scores + candidate_scores)
    y_max = max(accepted_scores + candidate_scores)
    pad = max((y_max - y_min) * 0.1, 0.02)
    ax_score.set_title("Validation Score (ARC-AGI1, Gemini 3 Flash)")
    event_metric_calls = infer_event_metric_calls(accepted, events)

    ax_score.set_xlabel("Metric Calls")
    ax_score.set_ylabel("Validation Score (%)")
    ax_score.grid(True, alpha=0.25)
    x_min = min(event_metric_calls)
    x_max = max(event_metric_calls)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    ax_score.set_xlim(x_min, x_max)
    ax_score.set_ylim(y_min - pad, y_max + pad)
    (best_line,) = ax_score.plot(
        [], [], color="#0f766e", linewidth=2.5, label="best-so-far validation score"
    )
    candidate_points = ax_score.scatter([], [], color="#9a3412", s=45, alpha=0.6)
    accepted_points = ax_score.scatter([], [], color="#115e59", s=60)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#0f766e",
            linewidth=2.5,
            label="Best-so-far validation score",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="#dc2626",
            markersize=7,
            label="Rejected candidate",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="#2563eb",
            markersize=7,
            label="Skipped candidate",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="#115e59",
            markersize=7,
            label="Accepted full-validation candidate",
        ),
    ]
    ax_score.legend(handles=legend_handles, loc="lower right")
    ax_score.yaxis.set_major_formatter(lambda value, _pos: f"{value * 100:.0f}%")
    accepted_annotations = []

    colors = {
        "accepted": "#16a34a",
        "rejected": "#dc2626",
        "perfect_skip": "#2563eb",
        "unknown_candidate": "#a16207",
    }
    status_text = ax_score.text(
        0.02,
        0.95,
        "",
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
        transform=ax_score.transAxes,
    )
    if not no_show:
        plt.ion()
        plt.show(block=False)
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None and screen is not None:
            try:
                fig.set_size_inches(fig_w, fig_h, forward=True)
            except Exception:
                pass
            plt.pause(0.01)
            try:
                manager.resize(int(fig_w * 100), int(fig_h * 100))
            except Exception:
                try:
                    manager.window.wm_geometry(
                        f"{int(fig_w * 100)}x{int(fig_h * 100)}+0+0"
                    )
                except Exception:
                    pass
        plt.pause(0.01)

    best_history_x: list[int] = []
    best_history_y: list[float] = []
    candidate_history_x: list[int] = []
    candidate_history_y: list[float] = []
    accepted_history_x: list[int] = []
    accepted_history_y: list[float] = []
    accepted_reasons: list[str] = []
    running_best = accepted[0]["score"]

    # Draw initial arch panel for candidate 0 (seed)
    _last_arch_idx: list[int | None] = [None]

    def refresh_arch(
        cand_idx: int | None,
        code: str,
        score: float | None,
        event_type: str = "accepted",
    ) -> None:
        if cand_idx == _last_arch_idx[0]:
            return  # same candidate — skip redraw
        _last_arch_idx[0] = cand_idx
        ax_arch.clear()
        stages = infer_stages(code)
        draw_arch(ax_arch, stages, cand_idx, score, event_type=event_type)

    refresh_arch(
        accepted[0]["candidate_idx"], accepted[0]["agent_code"], accepted[0]["score"]
    )

    for idx, event in enumerate(events, start=1):
        current_score = (
            event["score"]
            if event["score"] is not None
            else event["selected_program_score"]
        )
        current_metric_call = event_metric_calls[idx - 1]
        pretty_print(event, current_metric_call)
        candidate_history_x.append(current_metric_call)
        candidate_history_y.append(current_score)
        running_best = max(running_best, current_score)
        best_history_x.append(current_metric_call)
        best_history_y.append(running_best)

        if event["event_type"] == "accepted" and event["score"] is not None:
            accepted_history_x.append(current_metric_call)
            accepted_history_y.append(event["score"])
            accepted_reasons.append(event.get("event_reason", event["event_type"]))

        # Update arch panel only for accepted candidates (rejected/skipped
        # events don't log their proposed code, so there's nothing new to show).
        cand_idx = event.get("candidate_idx")
        if cand_idx is not None:
            code = event.get("proposed_agent_code") or event.get("agent_code") or ""
            if code:
                refresh_arch(
                    cand_idx,
                    code,
                    event.get("score"),
                    event_type=event.get("event_type", "unknown"),
                )

        best_line.set_data(best_history_x, best_history_y)
        candidate_points.set_offsets(
            list(zip(candidate_history_x, candidate_history_y))
        )
        if accepted_history_x:
            accepted_points.set_offsets(
                list(zip(accepted_history_x, accepted_history_y))
            )
        else:
            accepted_points.set_offsets([[float("nan"), float("nan")]])

        for annotation in accepted_annotations:
            annotation.remove()
        accepted_annotations.clear()
        for i, (x_val, y_val) in enumerate(zip(accepted_history_x, accepted_history_y)):
            accepted_annotations.append(
                ax_score.annotate(
                    f"{y_val * 100:.1f}%",
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color="#115e59",
                )
            )

        status_text.set_text(f"{event.get('event_reason', event['event_type'])}")

        ax_score.collections[0].set_color(
            [colors.get(events[j]["event_type"], "#6b7280") for j in range(idx)]
        )

        fig.canvas.draw_idle()
        if not no_show:
            fig.canvas.flush_events()
            plt.pause(delay)

    if no_show:
        print("ARC-AGI replay prepared successfully in no-show mode.")
        plt.close(fig)
        return

    plt.ioff()
    plt.show()


def main() -> None:
    args = parse_args()
    accepted = json.loads(args.accepted.read_text())
    events = json.loads(args.all_events.read_text())
    accepted, events = filter_by_metric_calls(accepted, events, args.max_metric_calls)
    if args.max_steps is not None:
        events = events[: args.max_steps]

    render(accepted, events, delay=args.delay, no_show=args.no_show)


if __name__ == "__main__":
    main()
