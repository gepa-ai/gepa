"""Regenerate the omni blog-post figures that carry the pipeline name.

Produces (into the 2026-05-28-optimize-anything-omni post's images/ dir):

- ``omni_bar.png`` — headline bar chart: standalone engines vs the composed
  omni pipeline on Frontier-CS.
- ``omni_design.png`` — the two-phase pipeline diagram (also the post's
  social-preview image).

The other two figures in the post (``optimizer_variance.png``,
``unstuck_frontier_cs.png``) carry no pipeline-name text and are untouched.

Usage:
    uv run --with matplotlib python docs/scripts/blog_omni_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

IMAGES = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "blog"
    / "posts"
    / "2026-05-28-optimize-anything-omni"
    / "images"
)

PIPELINE = "omni"

NAVY_BAR = "#26466b"
GRAY_BAR = "#c4c4c4"
NAVY_LINE = "#152d70"
GOLD = "#b87d2a"
FILL_LIGHT = "#f8faff"
FILL_DARK = "#eaf1fe"


def make_bar() -> None:
    """Headline bar: GEPA / AutoResearch / Meta-Harness standalone vs omni."""
    labels = ["GEPA", "AutoResearch", "Meta-Harness", PIPELINE]
    scores = [43.8, 55.4, 50.9, 63.2]
    colors = [GRAY_BAR, GRAY_BAR, GRAY_BAR, NAVY_BAR]

    fig, ax = plt.subplots(figsize=(12.64, 7.83), dpi=100)
    bars = ax.bar(labels, scores, width=0.6, color=colors, zorder=3)

    for bar, score in zip(bars, scores, strict=True):
        ax.annotate(
            f"{score}",
            (bar.get_x() + bar.get_width() / 2, score),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=21,
            fontweight="bold",
        )

    ax.set_ylabel("Mean score (10 problems, $20 budget)", fontsize=20)
    ax.set_ylim(0, 80)
    ax.set_yticks(range(0, 81, 10))
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(axis="y", linestyle="--", color="#dddddd", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(
        handles=[
            Patch(color=GRAY_BAR, label="Standalone ($20)"),
            Patch(color=NAVY_BAR, label=f"{PIPELINE} ($20)"),
        ],
        loc="upper left",
        fontsize=20,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(IMAGES / "omni_bar.png")
    plt.close(fig)


def _box(ax, cx: float, cy: float, w: float, h: float, title: str, money: str, fill: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=24",
            facecolor=fill,
            edgecolor=NAVY_LINE,
            linewidth=4.2,
            zorder=3,
        )
    )
    if money:
        # y-axis is inverted (y down): smaller y = visually higher.
        ax.text(
            cx, cy - 36, title, ha="center", va="center", fontsize=30, fontweight="bold", color="#111111", zorder=4
        )
        ax.text(cx, cy + 34, money, ha="center", va="center", fontsize=26, fontweight="bold", color=GOLD, zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=30, fontweight="bold", color="#111111", zorder=4)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], rad: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=100,
            linewidth=5.5,
            color=NAVY_LINE,
            zorder=2,
            shrinkA=0,
            shrinkB=4,
        )
    )


def make_design() -> None:
    """Two-phase pipeline diagram: parallel explore -> pick best -> fresh continuation."""
    fig, ax = plt.subplots(figsize=(31.56, 9.32), dpi=100)
    ax.set_xlim(0, 3156)
    ax.set_ylim(932, 0)  # y down, matching image coordinates
    ax.axis("off")

    rows = [133, 461, 792]
    engines = ["GEPA", "AutoResearch", "Meta-Harness"]

    # Phase 1: engines in parallel.
    for y, engine in zip(rows, engines, strict=True):
        _box(ax, 292, y, 490, 158, engine, "$5", FILL_LIGHT)

    # Pick best.
    _box(ax, 1205, rows[1], 450, 182, "Pick Best", "", FILL_LIGHT)

    # Phase 2: fresh continuation engines and the named variants.
    for y, engine in zip(rows, engines, strict=True):
        _box(ax, 1983, y, 490, 158, f"Fresh {engine}", "$5", FILL_LIGHT)
        _box(ax, 2830, y, 580, 158, f"{PIPELINE}-{engine}", "$20", FILL_DARK)

    # Converging arrows into Pick Best (heads fanned slightly so all three
    # stay visible), diverging out to the fresh engines.
    for y, rad, fan in zip(rows, (0.16, 0.0, -0.16), (-36, 0, 36), strict=True):
        _arrow(ax, (537, y), (978, rows[1] + fan), rad)
        _arrow(ax, (1432, rows[1]), (1736, y), -rad)
        _arrow(ax, (2228, y), (2538, y), 0.0)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(IMAGES / "omni_design.png")
    plt.close(fig)


if __name__ == "__main__":
    make_bar()
    make_design()
    print(f"wrote omni_bar.png and omni_design.png to {IMAGES}")
