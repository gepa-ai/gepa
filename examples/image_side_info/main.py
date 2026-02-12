"""
Minimal e2e example: optimize_anything with images in side_info.

Demonstrates passing Image objects through the evaluation → reflection loop
so that a VLM (vision-capable LLM) can analyze visual feedback.

Scenario
--------
We are "optimizing a drawing instruction" for a target color.  The evaluator:
  1. Parses the candidate's RGB values.
  2. Generates a tiny solid-color PNG.
  3. Returns the PNG as an ``Image`` in ``side_info`` together with text feedback.

The reflection LM (a VLM) sees both the textual feedback *and* the rendered
image, and proposes improved parameters.

Usage
-----
    # With a real VLM (requires OPENAI_API_KEY or equivalent):
    uv run python examples/image_side_info/main.py

    # Override the model:
    uv run python examples/image_side_info/main.py --model openai/gpt-4o

    # Dry run with a local mock LM (no API key needed) — just verifies the
    # image plumbing works end-to-end:
    uv run python examples/image_side_info/main.py --mock
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import tempfile
import zlib

from gepa.image import Image
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)

# ---------------------------------------------------------------------------
# Target colour (what we want the candidate to converge to)
# ---------------------------------------------------------------------------
TARGET_R, TARGET_G, TARGET_B = 30, 180, 90  # a nice green


# ---------------------------------------------------------------------------
# Tiny PNG generator (pure-Python, no Pillow needed)
# ---------------------------------------------------------------------------

def _make_solid_png(r: int, g: int, b: int, size: int = 16) -> bytes:
    """Return a ``size x size`` solid-colour PNG as raw bytes."""

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    scanline = b"\x00" + bytes([r, g, b]) * size  # filter-none + pixels
    raw = scanline * size
    idat = zlib.compress(raw)

    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluator(candidate: dict[str, str], **kwargs) -> tuple[float, SideInfo]:
    """Score a candidate RGB colour and return a rendered image in side_info."""

    # Parse candidate
    try:
        params = json.loads(candidate["color_params"])
        r = max(0, min(255, int(params["r"])))
        g = max(0, min(255, int(params["g"])))
        b = max(0, min(255, int(params["b"])))
    except Exception as e:
        return 0.0, {"Error": f"Could not parse color_params as JSON with r/g/b keys: {e}"}

    # Score: inverse Euclidean distance in RGB space (higher = better, max ≈1.0)
    dist = math.sqrt((r - TARGET_R) ** 2 + (g - TARGET_G) ** 2 + (b - TARGET_B) ** 2)
    max_dist = math.sqrt(255**2 * 3)  # ≈441
    score = 1.0 - dist / max_dist

    # Render a tiny PNG of the candidate colour
    png_bytes = _make_solid_png(r, g, b)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="candidate_color_")
    tmp.write(png_bytes)
    tmp.close()

    # Also render the target for reference
    target_png = _make_solid_png(TARGET_R, TARGET_G, TARGET_B)
    tmp_target = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="target_color_")
    tmp_target.write(target_png)
    tmp_target.close()

    side_info: SideInfo = {
        "CandidateColor": f"rgb({r}, {g}, {b})",
        # "TargetColor": f"rgb({TARGET_R}, {TARGET_G}, {TARGET_B})",
        # "EuclideanDistance": f"{dist:.1f}",
        "CandidateRendering": Image(path=tmp.name),
        "TargetRendering": Image(path=tmp_target.name),
        "Feedback": (
            f"The candidate colour rgb({r},{g},{b}) is {dist:.0f} units away from the "
            # f"target rgb({TARGET_R},{TARGET_G},{TARGET_B}).  "
            "Look at both images and adjust r, g, b to make the candidate match the target."
        ),
    }
    return score, side_info


# ---------------------------------------------------------------------------
# Mock reflection LM (for --mock mode)
# ---------------------------------------------------------------------------

def make_mock_reflection_lm():
    """Return a mock LM that verifies it receives multimodal messages and
    proposes a slightly-better candidate."""
    call_num = 0

    def _mock_lm(prompt):
        nonlocal call_num
        call_num += 1

        if isinstance(prompt, list):
            # Multimodal message — inspect it
            msg = prompt[0]
            content_parts = msg["content"]
            text_parts = [p for p in content_parts if p["type"] == "text"]
            image_parts = [p for p in content_parts if p["type"] == "image_url"]
            print(f"\n  [mock LM call #{call_num}]")
            print(f"    Received multimodal message: {len(text_parts)} text part(s), {len(image_parts)} image(s)")
            print(f"    Text length: {len(text_parts[0]['text'])} chars")
            for i, ip in enumerate(image_parts, 1):
                url = ip["image_url"]["url"]
                if url.startswith("data:"):
                    print(f"    Image {i}: data URI ({len(url)} chars, mime={url.split(';')[0].split(':')[1]})")
                else:
                    print(f"    Image {i}: URL {url[:80]}...")
        else:
            print(f"\n  [mock LM call #{call_num}] Received plain text prompt ({len(prompt)} chars)")

        # Propose something closer to the target each time
        new_params = json.dumps({"r": TARGET_R + 10 - call_num, "g": TARGET_G - 5 + call_num, "b": TARGET_B})
        return f"```\n{new_params}\n```"

    return _mock_lm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E2E example: images in side_info")
    parser.add_argument("--mock", action="store_true", help="Use a local mock LM (no API key needed)")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="LiteLLM model name for reflection")
    parser.add_argument("--max-calls", type=int, default=10, help="Max evaluator calls")
    args = parser.parse_args()

    # Pick reflection LM
    if args.mock:
        reflection_lm = make_mock_reflection_lm()
        print("Using MOCK reflection LM (no API calls)")
    else:
        reflection_lm = args.model  # type: ignore[assignment]
        print(f"Using real VLM: {args.model}")

    seed = {"color_params": json.dumps({"r": 200, "g": 50, "b": 200})}

    config = GEPAConfig(
        engine=EngineConfig(max_metric_calls=args.max_calls, cache_evaluation=False),
        reflection=ReflectionConfig(
            reflection_lm=reflection_lm,
            reflection_minibatch_size=1,
        ),
        refiner=None,
    )

    print("=" * 60)
    print("Optimizing RGB colour to match target")
    print(f"  Target: rgb({TARGET_R}, {TARGET_G}, {TARGET_B})")
    print(f"  Seed  : {seed['color_params']}")
    print("=" * 60)

    result = optimize_anything(
        seed_candidate=seed,
        evaluator=evaluator,
        objective=(
            "Find the RGB colour (r, g, b integers 0-255) that matches the target. "
            "The side_info includes rendered images of the candidate and target colours — "
            "use them to guide your adjustments. Output ONLY a JSON object like "
            '{"r": N, "g": N, "b": N}.'
        ),
        config=config,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Best candidate : {result.best_candidate['color_params']}")
    print(f"  Best score     : {result.val_aggregate_scores[result.best_idx]:.4f}")
    print(f"  Total evals    : {result.total_metric_calls}")


if __name__ == "__main__":
    main()
