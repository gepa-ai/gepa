# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Image wrapper for passing visual data through side_info to reflection VLMs."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

_MEDIA_TYPE_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}


def _guess_media_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _MEDIA_TYPE_BY_EXT.get(ext, "image/png")


@dataclass
class Image:
    """Image data for inclusion in ``side_info``, enabling VLM-based reflection.

    Wrap image data in this class and include it as a value (or nested value)
    anywhere inside the ``side_info`` dict returned by your evaluator.  When
    GEPA builds the reflection prompt, ``Image`` objects are automatically
    converted to the OpenAI vision content-part format and passed to the
    reflection LM as inline images.

    This enables a powerful visual feedback loop: your evaluator renders an
    artifact (SVG, 3D model, chart, etc.), passes the rendered image back as
    ASI, and a vision-capable proposer can literally *see* what it's improving.
    Requires a VLM as the ``reflection_lm`` (e.g. ``"vertex_ai/gemini-3-flash-preview"``).

    Provide **exactly one** of ``url``, ``path``, or ``base64_data``.

    Args:
        url: A URL pointing to the image, **or** a ``data:`` URI
            (``data:image/png;base64,...``).
        path: A local filesystem path to an image file.  The file is read and
            base64-encoded when the reflection prompt is constructed.
        base64_data: Raw base64-encoded image bytes.  Requires ``media_type``.
        media_type: MIME type (e.g. ``"image/png"``).  Inferred from ``path``
            extension when using *path*; **required** when using *base64_data*.

    Examples::

        # Rendered SVG feedback for visual optimization
        image_b64 = render_svg_to_png(candidate["svg_code"])
        side_info = {
            "RenderedSVG": Image(base64_data=image_b64, media_type="image/png"),
            "Feedback": vlm_feedback,
        }

        # File-based image feedback
        side_info = {
            "Input": "design a logo",
            "RenderedOutput": Image(path="/tmp/logo_v3.png"),
            "Feedback": "The colors are too muted",
        }
    """

    url: str | None = None
    path: str | None = None
    base64_data: str | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        sources = sum(x is not None for x in [self.url, self.path, self.base64_data])
        if sources != 1:
            raise ValueError("Exactly one of url, path, or base64_data must be provided.")
        if self.base64_data is not None and self.media_type is None:
            raise ValueError("media_type is required when using base64_data.")

    def to_openai_content_part(self) -> dict[str, Any]:
        """Convert to an OpenAI-compatible ``image_url`` content-part dict.

        Returns a dict of the form::

            {"type": "image_url", "image_url": {"url": "..."}}

        For *path*-based images the file is read and base64-encoded inline.
        """
        if self.url is not None:
            return {"type": "image_url", "image_url": {"url": self.url}}

        if self.path is not None:
            mt = self.media_type or _guess_media_type(self.path)
            with open(self.path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            return {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{data}"}}

        # base64_data
        assert self.base64_data is not None and self.media_type is not None
        return {"type": "image_url", "image_url": {"url": f"data:{self.media_type};base64,{self.base64_data}"}}
