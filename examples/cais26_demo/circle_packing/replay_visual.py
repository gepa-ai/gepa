#!/usr/bin/env python3
"""
Standalone visual replay for the circle packing demo.

This reads the saved best-so-far states from output/state_tracker_logs.json and
plays them back with:
- pretty terminal summaries
- a score graph over metric calls
- a circle packing visualization

It does not depend on gepa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual replay for circle packing")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("output/state_tracker_logs.json"),
        help="Path to state_tracker_logs.json",
    )
    parser.add_argument(
        "--best-output-dir",
        type=Path,
        default=Path("output/generated_best_outputs_valset/task_0"),
        help="Directory containing generated best output JSON files",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between frames",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run without opening a window (useful for verification)",
    )
    parser.add_argument(
        "--fake-non-improving",
        action="store_true",
        help="Use synthetic code snippets and side info for non-improving steps",
    )
    parser.add_argument(
        "--fake-model",
        type=str,
        default="mercury-2",
        help="Model to use for synthetic non-improving snippets",
    )
    parser.add_argument(
        "--fake-prefix-steps",
        type=int,
        default=0,
        help="Use synthetic code snippets and side info for the first N replay steps",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Only replay the first N steps",
    )
    return parser.parse_args()


@dataclass
class Frame:
    sequence_index: int
    frame_index: int
    metric_calls: int
    best_score: float
    delta: float
    improved: bool
    source: str
    circles: list[list[float]]
    code_text: str
    code_score: float | None
    refiner_score: float | None
    validation_text: str | None


def infer_source(entry: dict) -> str:
    if "best_artifact_refined_code" in entry:
        return "refined"
    if "best_artifact_code" in entry:
        return "code"
    return "unknown"


def get_code_text(entry: dict) -> str:
    if "best_artifact_refined_code" in entry:
        return entry["best_artifact_refined_code"]
    if "best_artifact_code" in entry:
        return entry["best_artifact_code"]
    return ""


def load_best_output_lookup(best_output_dir: Path) -> dict[float, dict]:
    lookup: dict[float, dict] = {}
    if not best_output_dir.exists():
        return lookup

    for path in sorted(best_output_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        score = payload.get("best_score")
        if isinstance(score, (int, float)):
            lookup[float(score)] = payload
    return lookup


def match_best_output(score: float, lookup: dict[float, dict]) -> dict | None:
    for key, payload in lookup.items():
        if abs(key - score) <= 1e-12:
            return payload
    return None


def load_frames(log_path: Path, best_output_dir: Path) -> tuple[list[dict], list[Frame]]:
    data = json.loads(log_path.read_text())
    best_output_lookup = load_best_output_lookup(best_output_dir)

    all_entries: list[dict] = []
    frames: list[Frame] = []
    previous_best = None

    for idx, entry in enumerate(data):
        circles = json.loads(entry["best_solution"]) if entry.get("best_solution") else []
        row = {
            "frame_index": idx,
            "metric_calls": entry["metric_calls"],
            "best_score": float(entry["best_score"]),
            "circles": circles,
            "source": infer_source(entry),
        }
        all_entries.append(row)

        improved = previous_best is None or row["best_score"] > previous_best + 1e-15
        matched_output = match_best_output(row["best_score"], best_output_lookup)
        frames.append(
            Frame(
                sequence_index=len(frames) + 1,
                frame_index=idx,
                metric_calls=row["metric_calls"],
                best_score=row["best_score"],
                delta=0.0 if previous_best is None else row["best_score"] - previous_best,
                improved=improved,
                source=row["source"],
                circles=circles,
                code_text=get_code_text(entry),
                code_score=matched_output.get("code_score") if matched_output else None,
                refiner_score=matched_output.get("refiner_score") if matched_output else None,
                validation_text=None,
            )
        )
        previous_best = row["best_score"]

    return all_entries, frames


def synthetic_scores(frame: Frame) -> tuple[float | None, float | None, str]:
    seed = int(hashlib.sha256(f"{frame.metric_calls}:{frame.source}".encode()).hexdigest()[:12], 16)
    rng = random.Random(seed)
    gap = max(0.003, min(0.08, frame.best_score * 0.02))
    candidate = frame.best_score - rng.uniform(gap * 0.25, gap)
    candidate = max(0.0, candidate)

    if frame.source == "refined":
        baseline = max(0.0, candidate - rng.uniform(0.0005, 0.01))
        return baseline, candidate, "valid packing; candidate did not beat incumbent"
    return candidate, None, "valid packing; incumbent remained better"


def call_mercury(prompt: str, model: str) -> str:
    api_key = os.environ.get("INCEPTION_API_KEY")
    if not api_key:
        raise RuntimeError("INCEPTION_API_KEY is not set")

    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "model": model,
        }
    )

    result = subprocess.run(
        [
            "curl",
            "-s",
            "-X",
            "POST",
            "https://api.inceptionlabs.ai/v1/chat/completions",
            "-H",
            f"Authorization: Bearer {api_key}",
            "-H",
            "Content-Type: application/json",
            "-d",
            payload,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    response = json.loads(result.stdout)
    return response["choices"][0]["message"]["content"].strip()


def fallback_fake_code(frame: Frame) -> str:
    return (
        "import numpy as np\n"
        "import time\n\n"
        "def main(timeout, current_best_solution):\n"
        "    n = 26\n"
        "    start = time.time()\n"
        "    rng = np.random.default_rng()\n"
        "    centers = current_best_solution[:, :2].copy() if current_best_solution is not None else rng.uniform(0.08, 0.92, size=(n, 2))\n"
        "    centers = np.clip(centers + rng.normal(0, 0.01, size=(n, 2)), 1e-3, 1 - 1e-3)\n"
        "    radii = np.minimum.reduce([centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]])\n"
        "    for i in range(n):\n"
        "        for j in range(i + 1, n):\n"
        "            d = np.linalg.norm(centers[i] - centers[j])\n"
        "            s = radii[i] + radii[j]\n"
        "            if s > d and s > 0:\n"
        "                scale = max(d / s, 0.5)\n"
        "                radii[i] *= scale\n"
        "                radii[j] *= scale\n"
        "    circles = np.hstack([centers, np.maximum(radii, 1e-4).reshape(-1, 1)])\n"
        "    score = float(np.sum(circles[:, 2]))\n"
        "    return {'circles': circles, 'all_scores': [score]}\n"
    )


def is_acceptable_fake_code(code_text: str) -> bool:
    required = [
        "def main(timeout, current_best_solution):",
        "n = 26",
        "return {'circles': circles, 'all_scores': [score]}",
    ]
    forbidden = [
        "__main__",
        "metric_call=",
        "current_best_score",
        "print(",
        "solve(",
        "pack(",
        "area=",
    ]

    if any(token not in code_text for token in required):
        return False
    if any(token in code_text for token in forbidden):
        return False
    if len(code_text.splitlines()) < 12:
        return False
    return True


def make_fake_code(frame: Frame, model: str) -> str:
    prompt = (
        "Write plausible Python candidate code for a circle packing solver.\n"
        "Return code only.\n"
        "Hard requirements:\n"
        "- Must define exactly: def main(timeout, current_best_solution):\n"
        "- Must use n = 26\n"
        "- Must return {'circles': circles, 'all_scores': [score]} or a similar list of scores\n"
        "- No top-level execution\n"
        "- No __main__ block\n"
        "- No print statements\n"
        "- No toy examples with 5 or 12 circles\n"
        "- No metric_call variables or commentary text\n"
        "- Keep it plausible as an in-progress optimization attempt, around 18 to 35 lines\n"
        "- Prefer numpy-based code and simple local-search / repair logic\n\n"
        f"Context:\nsource={frame.source}\n"
        f"current_best_score={frame.best_score:.12f}\n"
        f"style_reference:\n{truncate_code(frame.code_text, max_lines=18, max_chars=1000)}"
    )
    candidate = call_mercury(prompt, model)
    candidate = candidate.replace("```python", "").replace("```", "").strip()
    if is_acceptable_fake_code(candidate):
        return candidate
    return fallback_fake_code(frame)


def apply_fake_prefix(frames: list[Frame], count: int, model: str) -> None:
    for frame in frames[:count]:
        frame.code_score, frame.refiner_score, frame.validation_text = synthetic_scores(frame)
        try:
            frame.code_text = make_fake_code(frame, model)
        except Exception:
            frame.code_text = fallback_fake_code(frame)
            if frame.validation_text is None:
                frame.validation_text = "valid packing; exploring an alternative candidate"


def apply_fake_non_improving(frames: list[Frame], model: str) -> None:
    for frame in frames:
        if frame.improved:
            frame.validation_text = "best-so-far state"
            continue

        frame.code_score, frame.refiner_score, frame.validation_text = synthetic_scores(frame)
        try:
            frame.code_text = make_fake_code(frame, model)
        except Exception:
            frame.code_text = fallback_fake_code(frame)
            if frame.validation_text is None:
                frame.validation_text = "valid packing; candidate did not beat incumbent"


def truncate_code(code_text: str, max_lines: int = 14, max_chars: int = 900) -> str:
    snippet = code_text.strip()
    if not snippet:
        return "<no code available>"

    lines = snippet.splitlines()
    clipped = "\n".join(lines[:max_lines])
    if len(clipped) > max_chars:
        clipped = clipped[: max_chars - 3].rstrip() + "..."
    elif len(lines) > max_lines:
        clipped = clipped.rstrip() + "\n..."
    return clipped


def pretty_print(frame: Frame, total_frames: int) -> None:
    if frame.sequence_index == 1:
        delta_text = "n/a"
    elif not frame.improved:
        delta_text = "+0.000000"
    elif abs(frame.delta) >= 1e-4:
        delta_text = f"+{frame.delta:.6f}"
    else:
        delta_text = f"+{frame.delta:.2e}"
    code_score = "n/a" if frame.code_score is None else f"{frame.code_score:.12f}"
    refiner_score = "n/a" if frame.refiner_score is None else f"{frame.refiner_score:.12f}"
    print("=" * 88)
    print(
        f"metric_call={frame.metric_calls:>3} | "
        f"source={frame.source:<7} | "
        f"best_score={frame.best_score:.12f} | "
        f"delta={delta_text}"
    )
    print(
        f"Side info: code_score={code_score} | "
        f"refiner_score={refiner_score} | "
        f"circles={len(frame.circles)} | "
        f"validation={frame.validation_text or 'n/a'}"
    )
    print("Code snippet:")
    print(truncate_code(frame.code_text, max_lines=22, max_chars=1600))
    print()


def setup_matplotlib(no_show: bool):
    mpl_dir = Path(".mplconfig")
    mpl_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_dir.resolve()))

    if no_show:
        os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    return plt, Circle, Rectangle


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


def render_replay(all_entries: list[dict], frames: list[Frame], delay: float, no_show: bool) -> None:
    plt, Circle, Rectangle = setup_matplotlib(no_show)

    screen = None if no_show else get_screen_geometry()
    if screen is not None:
        screen_width, screen_height = screen
        half_width = max(screen_width // 2, 1100)
        full_height = max(screen_height, 900)
        fig = plt.figure(figsize=(half_width / 100, full_height / 100), dpi=100)
    else:
        half_width = 1200
        full_height = 1000
        fig = plt.figure(figsize=(12, 10), dpi=100)
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.28)
    ax_score = fig.add_subplot(grid[0])
    ax_pack = fig.add_subplot(grid[1])

    metric_calls = [row["metric_calls"] for row in all_entries]
    scores = [row["best_score"] for row in all_entries]
    min_call, max_call = min(metric_calls), max(metric_calls)
    min_score, max_score = min(scores), max(scores)
    score_pad = max((max_score - min_score) * 0.08, 0.01)

    ax_score.set_title("Circle Packing Score (GPT 5)")
    ax_score.set_xlabel("Metric Call")
    ax_score.set_ylabel("Best Score")
    ax_score.grid(True, alpha=0.25)
    ax_score.set_xlim(min_call, max_call)
    ax_score.set_ylim(min_score - score_pad, max_score + score_pad)
    score_line, = ax_score.plot([], [], color="#1d4ed8", linewidth=2.5)
    score_point, = ax_score.plot([], [], "o", color="#1e3a8a", markersize=8)

    ax_pack.set_title("Best Circle Packing Layout")
    ax_pack.set_xlim(0, 1)
    ax_pack.set_ylim(0, 1)
    ax_pack.set_aspect("equal")
    ax_pack.set_xticks([])
    ax_pack.set_yticks([])
    ax_pack.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=1.5))

    status_text = fig.text(
        0.02,
        0.98,
        "",
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    circle_artists: list = []

    if not no_show:
        plt.ion()
        plt.show(block=False)
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None and screen is not None:
            try:
                fig.set_size_inches(half_width / 72, full_height / 72, forward=True)
            except Exception:
                pass
            plt.pause(0.01)
            try:
                manager.resize(half_width, full_height)
            except Exception:
                try:
                    fig.set_size_inches(half_width / 100, full_height / 100, forward=True)
                except Exception:
                    pass
                try:
                    manager.window.wm_geometry(f"{half_width}x{full_height}+0+0")
                except Exception:
                    pass
        plt.pause(3.0)
        time.sleep(3.0)

    total_frames = len(frames)
    for idx, frame in enumerate(frames, start=1):
        pretty_print(frame, total_frames)

        upto = [row for row in all_entries if row["metric_calls"] <= frame.metric_calls]
        upto_calls = [row["metric_calls"] for row in upto]
        upto_scores = [row["best_score"] for row in upto]
        score_line.set_data(upto_calls, upto_scores)
        score_point.set_data([frame.metric_calls], [frame.best_score])

        for artist in circle_artists:
            artist.remove()
        circle_artists.clear()

        colors = plt.cm.viridis_r(
            [i / max(1, len(frame.circles) - 1) for i in range(len(frame.circles))]
        )
        # Draw larger circles first so smaller ones remain visible.
        ordered = sorted(zip(frame.circles, colors), key=lambda item: item[0][2], reverse=True)
        for (x, y, r), color in ordered:
            circle = Circle(
                (x, y),
                r,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.72,
            )
            ax_pack.add_patch(circle)
            circle_artists.append(circle)

        status_text.set_text(
            f"metric_call: {frame.metric_calls}\n"
            f"source:      {frame.source}\n"
            f"best_score:  {frame.best_score:.12f}\n"
            f"delta:       "
            f"{'n/a' if idx == 1 else ('+0.000000' if not frame.improved else (f'+{frame.delta:.6f}' if abs(frame.delta) >= 1e-4 else f'+{frame.delta:.2e}'))}"
        )

        fig.canvas.draw_idle()
        if not no_show:
            plt.pause(delay)

    if no_show:
        print("Replay prepared successfully in no-show mode.")
        plt.close(fig)
        return

    plt.ioff()
    plt.show()


def main() -> None:
    args = parse_args()
    all_entries, frames = load_frames(args.log, args.best_output_dir)

    if args.max_steps is not None:
        frames = frames[: args.max_steps]
        if frames:
            max_metric_call = frames[-1].metric_calls
            all_entries = [row for row in all_entries if row["metric_calls"] <= max_metric_call]

    if args.fake_prefix_steps > 0:
        apply_fake_prefix(frames, args.fake_prefix_steps, args.fake_model)

    if args.fake_non_improving:
        apply_fake_non_improving(frames, args.fake_model)
    else:
        for frame in frames:
            if frame.validation_text is None:
                frame.validation_text = "best-so-far state"

    if not frames:
        raise SystemExit(f"No replay frames found in {args.log}")

    render_replay(all_entries, frames, delay=args.delay, no_show=args.no_show)


if __name__ == "__main__":
    main()
