"""Create a reflection language model callable for use with GEPA."""

from gepa.proposer.reflective_mutation.base import LanguageModel


def make_reflection_lm(model: str) -> LanguageModel:
    """Return a callable that sends prompts to the given LLM and returns text.

    Args:
        model: Model name understood by litellm (e.g. "gpt-4o-mini") or a
               Gemini model name (e.g. "gemini-1.5-flash").
    """
    import litellm  # type: ignore

    gemini_client = None
    if "gemini" in model:
        from google import genai  # type: ignore
        from google.genai.types import HttpOptions  # type: ignore

        gemini_client = genai.Client(http_options=HttpOptions(api_version="v1"))

    def call_lm(prompt: str | list[dict[str, str]]) -> str:
        if "gemini" in model and gemini_client is not None:
            response = gemini_client.models.generate_content(
                model=model,
                contents=_to_string(prompt),
            )
            return response.text or ""
        else:
            messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            completion = litellm.completion(model=model, messages=messages)
            return completion.choices[0].message.content or ""  # type: ignore

    return call_lm


def _to_string(prompt: str | list[dict[str, str]]) -> str:
    if isinstance(prompt, list):
        return "\n".join(f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in prompt)
    return prompt
