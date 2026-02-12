"""
Optimize SVG code to depict "a pelican riding a bicycle".

Single-instance mode (no dataset). A VLM scores the rendered SVG, and the
rendered PNG is passed back as an Image in side_info so the reflection LM
can see it and propose better SVG.

Requirements:
    pip install cairosvg litellm

Usage:
    uv run python examples/svg_optimization/main.py
    uv run python examples/svg_optimization/main.py --model openai/gpt-4o --eval-model openai/gpt-4o
"""

from __future__ import annotations

import argparse
import base64
import re
import tempfile

import cairosvg
import litellm

from gepa.image import Image
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)

GOAL = "a pelican riding a bicycle"

SEED_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <rect width="512" height="512" fill="lightblue"/>
  <circle cx="170" cy="380" r="50" fill="none" stroke="black" stroke-width="3"/>
  <circle cx="340" cy="380" r="50" fill="none" stroke="black" stroke-width="3"/>
  <ellipse cx="255" cy="250" rx="40" ry="30" fill="white" stroke="black" stroke-width="2"/>
  <circle cx="270" cy="220" r="15" fill="white" stroke="black" stroke-width="2"/>
  <line x1="285" y1="220" x2="320" y2="215" stroke="orange" stroke-width="3"/>
</svg>"""

EVAL_PROMPT = f"""\
You are evaluating an SVG rendering of: '{GOAL}'.

Score each axis from 0 to 10 and provide brief feedback for each.

Respond in EXACTLY this format (one line per axis, then a Feedback section):
Pelican: N/10
Bicycle: N/10
Background: N/10
Integration: N/10

Feedback:
<brief specific feedback on what to improve for each axis>
"""


def render_svg_to_png(svg_code: str) -> bytes | None:
    """Render SVG to PNG bytes via cairosvg. Returns None on failure."""
    try:
        return cairosvg.svg2png(bytestring=svg_code.encode("utf-8"), output_width=512, output_height=512)
    except Exception:
        return None


def _parse_axis_score(text: str, label: str) -> float:
    """Extract 'Label: N/10' from VLM output, return 0-1 float."""
    m = re.search(rf"{label}:\s*(\d+(?:\.\d+)?)\s*/\s*10", text, re.IGNORECASE)
    return float(m.group(1)) / 10.0 if m else 0.0


def make_evaluator(eval_model: str):
    """Return an evaluator that scores rendered SVGs on multiple axes via a VLM."""

    def evaluator(candidate: dict[str, str]) -> tuple[float, SideInfo]:
        svg_code = candidate["svg_code"]
        png_bytes = render_svg_to_png(svg_code)

        if png_bytes is None:
            return 0.0, {
                "Feedback": f"SVG rendering failed. Ensure valid SVG syntax.\nSVG:\n{svg_code}",
                "scores": {"pelican": 0.0, "bicycle": 0.0, "background": 0.0, "integration": 0.0},
            }

        # Save PNG so side_info can reference it as an Image
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="svg_eval_")
        tmp.write(png_bytes)
        tmp.close()

        # Ask VLM to evaluate the rendering on multiple axes
        b64 = base64.b64encode(png_bytes).decode()
        resp = litellm.completion(
            model=eval_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": EVAL_PROMPT},
                    ],
                }
            ],
        )
        vlm_text = resp.choices[0].message.content or ""

        # Parse per-axis scores
        pelican = _parse_axis_score(vlm_text, "Pelican")
        bicycle = _parse_axis_score(vlm_text, "Bicycle")
        background = _parse_axis_score(vlm_text, "Background")
        integration = _parse_axis_score(vlm_text, "Integration")

        # Aggregate: weighted average (integration matters most)
        score = 0.25 * pelican + 0.25 * bicycle + 0.15 * background + 0.35 * integration

        return score, {
            "RenderedSVG": Image(path=tmp.name),
            "Feedback": vlm_text,
            "scores": {
                "pelican": pelican,
                "bicycle": bicycle,
                "background": background,
                "integration": integration,
            },
        }

    return evaluator


def main():
    parser = argparse.ArgumentParser(description="Optimize SVG: pelican riding a bicycle")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="Reflection VLM model")
    parser.add_argument("--eval-model", default="openai/gpt-4o-mini", help="Evaluation VLM model")
    parser.add_argument("--max-calls", type=int, default=15, help="Max evaluator calls")
    args = parser.parse_args()

    config = GEPAConfig(
        engine=EngineConfig(max_metric_calls=args.max_calls, cache_evaluation=False),
        reflection=ReflectionConfig(
            reflection_lm=args.model,
            reflection_minibatch_size=1,
        ),
        refiner=None,
    )

    print(f"Goal: {GOAL}")
    print(f"Reflection VLM: {args.model} | Eval VLM: {args.eval_model}")
    print(f"Max evaluator calls: {args.max_calls}")
    print("=" * 60)

    result = optimize_anything(
        seed_candidate={"svg_code": SEED_SVG},
        evaluator=make_evaluator(args.eval_model),
        objective=(
            f"Optimize the SVG code to create a clear, detailed illustration of '{GOAL}'. "
            "Output ONLY valid SVG code (starting with <svg, ending with </svg>). "
            "Use viewBox='0 0 512 512'. The rendered image is shown in side_info — "
            "use the visual feedback and the VLM critique to improve shapes, colors, "
            "proportions, and composition each iteration."
        ),
        config=config,
    )

    best_svg = result.best_candidate["svg_code"]
    best_score = result.val_aggregate_scores[result.best_idx]

    with open("best_pelican.svg", "w") as f:
        f.write(best_svg)

    print("\n" + "=" * 60)
    print(f"Best score : {best_score:.2f}")
    print(f"Total evals: {result.total_metric_calls}")
    print("Saved best SVG to best_pelican.svg")


if __name__ == "__main__":
    main()
