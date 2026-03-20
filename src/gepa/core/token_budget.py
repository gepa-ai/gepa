# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


_WARNING_THRESHOLD = 0.8
FALLBACK_TOKEN_COUNTER_MODEL = "openai/gpt-5"


def count_candidate_tokens(candidate: dict[str, str], token_counter_model: str) -> int:
    """Count the total number of tokens across all text components of a candidate.

    Uses litellm.token_counter when available for accurate counts,
    otherwise falls back to a character-based heuristic (~4 chars per token).
    """
    total_text = "\n\n".join(candidate.values())

    try:
        import litellm

        return litellm.token_counter(model=token_counter_model, messages=[{"role": "user", "content": total_text}])
    except Exception:
        return max(1, len(total_text) // 4)


def check_candidate_token_limit(
    candidate: dict[str, str],
    max_candidate_tokens: int,
    token_counter_model: str,
) -> tuple[int, bool, bool]:
    """Check a candidate against the token limit.

    Returns:
        (token_count, exceeds_limit, exceeds_warning_threshold)
    """
    token_count = count_candidate_tokens(candidate, token_counter_model=token_counter_model)
    exceeds = token_count > max_candidate_tokens
    warning = token_count > max_candidate_tokens * _WARNING_THRESHOLD
    return token_count, exceeds, warning


def build_candidate_token_context(
    candidate: dict[str, str], max_candidate_tokens: int, token_counter_model: str
) -> str:
    """Build a context string describing the token limit for the reflection LM."""
    current_tokens = count_candidate_tokens(candidate, token_counter_model=token_counter_model)
    remaining = max(0, max_candidate_tokens - current_tokens)
    pct = current_tokens / max_candidate_tokens * 100

    return (
        f"IMPORTANT — Token Limit: The current candidate uses {current_tokens} tokens "
        f"({pct:.0f}% of the {max_candidate_tokens} token limit, {remaining} tokens remaining). "
        f"Your proposed replacement MUST stay within {max_candidate_tokens} total tokens. "
        f"Prefer concise, targeted improvements and avoid unnecessary verbosity."
    )
