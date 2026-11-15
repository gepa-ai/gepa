# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for Image support in side_info and multimodal reflection prompts."""

import base64
import os
import tempfile

import pytest

from gepa.image import Image
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)
from gepa.strategies.instruction_proposal import InstructionProposalSignature

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal valid 1x1 red-pixel PNG (69 bytes).
_TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
_TINY_PNG_BYTES = base64.b64decode(_TINY_PNG_B64)


def _write_tiny_png(path: str) -> None:
    with open(path, "wb") as f:
        f.write(_TINY_PNG_BYTES)


# ---------------------------------------------------------------------------
# Image dataclass construction & validation
# ---------------------------------------------------------------------------


class TestImageConstruction:
    def test_url(self):
        img = Image(url="https://example.com/img.png")
        assert img.url == "https://example.com/img.png"
        assert img.path is None and img.base64_data is None

    def test_path(self):
        img = Image(path="/tmp/img.png")
        assert img.path == "/tmp/img.png"

    def test_base64(self):
        img = Image(base64_data=_TINY_PNG_B64, media_type="image/png")
        assert img.base64_data == _TINY_PNG_B64

    def test_no_source_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            Image()

    def test_multiple_sources_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            Image(url="https://x.com/a.png", path="/tmp/a.png")

    def test_base64_without_media_type_raises(self):
        with pytest.raises(ValueError, match="media_type is required"):
            Image(base64_data=_TINY_PNG_B64)


# ---------------------------------------------------------------------------
# to_openai_content_part (litellm / VLM provider compatibility)
# ---------------------------------------------------------------------------


class TestToOpenAIContentPart:
    """Verify the output matches the OpenAI vision API content-part schema.

    The ``{"type": "image_url", "image_url": {"url": ...}}`` format is the
    standard accepted by litellm for OpenAI, Anthropic, Google, Mistral, etc.
    """

    def test_url_image(self):
        img = Image(url="https://example.com/photo.jpg")
        part = img.to_openai_content_part()
        assert part == {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}

    def test_data_uri(self):
        data_uri = f"data:image/png;base64,{_TINY_PNG_B64}"
        img = Image(url=data_uri)
        part = img.to_openai_content_part()
        assert part["image_url"]["url"] == data_uri

    def test_path_image(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_TINY_PNG_BYTES)
            tmp_path = f.name
        try:
            img = Image(path=tmp_path)
            part = img.to_openai_content_part()
            assert part["type"] == "image_url"
            url = part["image_url"]["url"]
            assert url.startswith("data:image/png;base64,")
            # Roundtrip: the embedded data should decode back to the original bytes.
            encoded = url.split(",", 1)[1]
            assert base64.b64decode(encoded) == _TINY_PNG_BYTES
        finally:
            os.unlink(tmp_path)

    def test_path_jpeg_media_type_inferred(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"not-really-a-jpeg")
            tmp_path = f.name
        try:
            img = Image(path=tmp_path)
            part = img.to_openai_content_part()
            assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")
        finally:
            os.unlink(tmp_path)

    def test_path_explicit_media_type_override(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_TINY_PNG_BYTES)
            tmp_path = f.name
        try:
            img = Image(path=tmp_path, media_type="image/webp")
            part = img.to_openai_content_part()
            assert part["image_url"]["url"].startswith("data:image/webp;base64,")
        finally:
            os.unlink(tmp_path)

    def test_base64_image(self):
        img = Image(base64_data=_TINY_PNG_B64, media_type="image/png")
        part = img.to_openai_content_part()
        assert part == {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"}}

    def test_content_part_schema_shape(self):
        """All three modes must produce the exact two-level dict structure
        expected by litellm / OpenAI / Anthropic / Google VLM APIs."""
        for img in [
            Image(url="https://x.com/a.png"),
            Image(base64_data=_TINY_PNG_B64, media_type="image/png"),
        ]:
            part = img.to_openai_content_part()
            assert set(part.keys()) == {"type", "image_url"}
            assert part["type"] == "image_url"
            assert isinstance(part["image_url"], dict)
            assert "url" in part["image_url"]
            assert isinstance(part["image_url"]["url"], str)


# ---------------------------------------------------------------------------
# Prompt renderer: text-only (backward compat) vs multimodal
# ---------------------------------------------------------------------------


class TestPromptRendererImages:
    """Test InstructionProposalSignature.prompt_renderer with Image objects."""

    def _render(self, dataset, prompt_template=None):
        return InstructionProposalSignature.prompt_renderer(
            {
                "current_instruction_doc": "Do the thing.",
                "dataset_with_feedback": dataset,
                "prompt_template": prompt_template,
            }
        )

    def test_no_images_returns_string(self):
        result = self._render([{"Input": "hello", "Score": 0.5}])
        assert isinstance(result, str)
        assert "hello" in result

    def test_with_images_returns_messages_list(self):
        img = Image(base64_data=_TINY_PNG_B64, media_type="image/png")
        result = self._render([{"Input": "hello", "Visual": img}])

        # Must be a list of message dicts (OpenAI format)
        assert isinstance(result, list)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "user"
        content = msg["content"]
        assert isinstance(content, list)

        # First element is the text prompt
        assert content[0]["type"] == "text"
        assert "Do the thing." in content[0]["text"]
        assert "[IMAGE-1" in content[0]["text"]

        # Second element is the image
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_multiple_images(self):
        img1 = Image(url="https://example.com/a.png")
        img2 = Image(base64_data=_TINY_PNG_B64, media_type="image/png")
        result = self._render([
            {"Input": "x", "Chart": img1},
            {"Input": "y", "Plot": img2},
        ])
        content = result[0]["content"]
        # text + 2 images
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert "[IMAGE-1" in content[0]["text"]
        assert "[IMAGE-2" in content[0]["text"]
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "image_url"

    def test_nested_images(self):
        img = Image(base64_data=_TINY_PNG_B64, media_type="image/png")
        result = self._render([{"Data": {"nested": {"deep": img}}}])
        content = result[0]["content"]
        assert len(content) == 2  # text + 1 image
        assert "[IMAGE-1" in content[0]["text"]

    def test_images_in_list_values(self):
        img = Image(base64_data=_TINY_PNG_B64, media_type="image/png")
        result = self._render([{"Frames": [img, img]}])
        content = result[0]["content"]
        assert len(content) == 3  # text + 2 images
        assert "[IMAGE-1" in content[0]["text"]
        assert "[IMAGE-2" in content[0]["text"]

    def test_image_count_header(self):
        img = Image(url="https://example.com/x.png")
        result = self._render([{"A": img}])
        text = result[0]["content"][0]["text"]
        assert "1 image(s)" in text

    def test_mixed_images_and_text(self):
        """Images and regular text coexist; text values are not corrupted."""
        img = Image(base64_data=_TINY_PNG_B64, media_type="image/png")
        result = self._render([{"Feedback": "too dark", "Screenshot": img, "Score": 0.3}])
        text = result[0]["content"][0]["text"]
        assert "too dark" in text
        assert "0.3" in text


# ---------------------------------------------------------------------------
# E2E: optimize_anything with images in side_info
# ---------------------------------------------------------------------------


class TestOptimizeAnythingWithImage:
    """Integration test: ensure images flow through the full optimization loop."""

    def test_image_reaches_reflection_lm(self):
        """Evaluator returns Image in side_info; the mock reflection LM
        receives a multimodal messages list with the image content part."""
        reflection_calls: list = []

        def mock_reflection_lm(prompt):
            reflection_calls.append(prompt)
            return "```\nimproved prompt\n```"

        img = Image(base64_data=_TINY_PNG_B64, media_type="image/png")

        def evaluator(candidate, **kwargs):
            score = 1.0 if "improved" in candidate["instructions"] else 0.5
            side_info = {
                "Input": "draw a red circle",
                "Rendering": img,
                "Feedback": "Circle is blue instead of red",
            }
            return score, side_info

        result = optimize_anything(
            seed_candidate={"instructions": "draw shapes"},
            evaluator=evaluator,
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=2, cache_evaluation=False),
                reflection=ReflectionConfig(
                    reflection_lm=mock_reflection_lm,
                    reflection_minibatch_size=1,
                ),
            ),
        )

        assert result is not None

        # At least one reflection call should be multimodal (a messages list)
        multimodal_calls = [c for c in reflection_calls if isinstance(c, list)]
        assert len(multimodal_calls) >= 1, (
            "Expected at least one multimodal reflection call, but all calls were plain strings"
        )

        # Verify structure of the multimodal call
        msg = multimodal_calls[0]
        assert msg[0]["role"] == "user"
        content = msg[0]["content"]

        text_parts = [p for p in content if p["type"] == "text"]
        image_parts = [p for p in content if p["type"] == "image_url"]

        assert len(text_parts) == 1
        assert len(image_parts) >= 1

        # The text must contain the side_info fields
        text = text_parts[0]["text"]
        assert "draw shapes" in text  # current instruction
        assert "Circle is blue" in text  # feedback from side_info
        assert "[IMAGE-" in text  # placeholder for the image

        # The image part must be a valid data URI
        img_url = image_parts[0]["image_url"]["url"]
        assert img_url.startswith("data:image/png;base64,")

    def test_no_image_backward_compat(self):
        """Without images, the reflection LM receives a plain string (not messages list)."""
        reflection_calls: list = []

        def mock_reflection_lm(prompt):
            reflection_calls.append(prompt)
            return "```\nimproved\n```"

        def evaluator(candidate, **kwargs):
            return 0.5, {"Feedback": "needs work"}

        optimize_anything(
            seed_candidate={"instructions": "do stuff"},
            evaluator=evaluator,
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=2, cache_evaluation=False),
                reflection=ReflectionConfig(
                    reflection_lm=mock_reflection_lm,
                    reflection_minibatch_size=1,
                ),
            ),
        )

        # All calls should be plain strings
        assert all(isinstance(c, str) for c in reflection_calls), (
            "Expected plain string prompts when no images are present"
        )
