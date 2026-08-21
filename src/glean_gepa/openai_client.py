"""OpenAI-compatible client for Glean's QE LLM proxy."""

from __future__ import annotations

import hashlib
import ssl

import openai
import truststore


def get_perfeval_secret(project: str) -> str:
    """Derive the QE perf-eval secret for a GCP project."""
    return "perfeval_" + hashlib.sha256(project.encode()).hexdigest()


def create_qe_openai_client(
    instance: str,
    *,
    timeout_seconds: float = 600.0,
    max_retries: int = 5,
) -> openai.OpenAI:
    """Create a QE client that validates TLS with the operating system trust store."""
    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return openai.OpenAI(
        base_url=f"https://{instance}-be.glean.com/qe/llm",
        api_key="dummy",  # QE authenticates with perf_eval_secret in the request body.
        timeout=timeout_seconds,
        max_retries=max_retries,
        http_client=openai.DefaultHttpxClient(verify=ssl_context),
    )


def format_exception_chain(exc: BaseException) -> str:
    """Render nested transport failures without losing the actionable root cause."""
    messages: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " <- ".join(messages)
