# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from gepa.image import Image
from gepa.proposer.reflective_mutation.base import (
    LanguageModel,
    Signature,
    SignatureAdapter,
    SignatureParseResult,
)


class InstructionProposalError(ValueError):
    """Raised when a reflection output is known to be incomplete."""


@dataclass(frozen=True)
class ProposalParseResult:
    """A parsed proposal or a diagnostic explaining why it should be skipped."""

    text: str | None
    error: str | None = None

    def require(self, context: str = "Could not parse instruction proposal") -> str:
        if self.text is None:
            detail = f": {self.error}" if self.error else ""
            raise InstructionProposalError(f"{context}{detail}")
        return self.text


class ProposalAdapter:
    """Format a signature normally and parse its output compatibly.

    A complete outer fence is authoritative. Without one, the historical GEPA
    salvage behavior is retained unless the response carries positive evidence
    that generation was truncated. This matters because an ordinary unfenced
    instruction and an untagged truncated response are otherwise
    indistinguishable from text alone.
    """

    def __init__(self, output_key: str):
        self.output_key = output_key

    def format(self, signature: type[Signature], input_dict: Mapping[str, Any]) -> str | list[dict[str, Any]]:
        return signature.prompt_renderer(input_dict)

    @staticmethod
    def _has_fence_pair(lm_out: str) -> bool:
        return (lm_out.find("```") + 3) < lm_out.rfind("```")

    @staticmethod
    def _finish_reason(lm_out: str) -> str | None:
        finish_reason = getattr(lm_out, "finish_reason", None)
        return finish_reason if isinstance(finish_reason, str) else None

    @classmethod
    def _is_known_truncated(cls, signature: type[Signature], lm_out: str) -> bool:
        if cls._has_fence_pair(lm_out):
            return False
        if cls._finish_reason(lm_out) in {"length", "max_tokens"}:
            return True

        stripped = lm_out.lstrip()
        reasoning_tags = getattr(signature, "reasoning_tags", ("think",))
        return any(
            stripped.startswith(f"<{tag}>") and lm_out.count(f"<{tag}>") > lm_out.count(f"</{tag}>")
            for tag in reasoning_tags
        )

    def parse(self, signature: type[Signature], lm_out: str) -> SignatureParseResult:
        if self._is_known_truncated(signature, lm_out):
            reason = self._finish_reason(lm_out)
            detail = f"finish_reason={reason!r}" if reason is not None else "unterminated reasoning block"
            return SignatureParseResult.failure(f"reflection output is incomplete ({detail})")

        start = lm_out.find("```") + 3
        end = lm_out.rfind("```")
        if self._has_fence_pair(lm_out):
            content = lm_out[start:end]
            match = re.match(r"^\S*\n", content)
            if match:
                content = content[match.end() :]
            value = content.strip()
        else:
            # Preserve GEPA's original permissive contract for custom LMs.
            value = lm_out.strip()
            if value.startswith("```"):
                match = re.match(r"^```\S*\n?", value)
                if match:
                    value = value[match.end() :].strip()
            elif value.endswith("```"):
                value = value[:-3].strip()

        return SignatureParseResult.success({self.output_key: value})


_INSTRUCTION_PROPOSAL_ADAPTER = ProposalAdapter("new_instruction")


def parse_proposal(signature: type[Signature], lm_out: str) -> ProposalParseResult:
    """Parse one single-output proposal without raising on an invalid completion."""
    adapter = signature.adapter
    if adapter is None:
        raise TypeError(f"{signature.__name__} does not define a proposal adapter")
    if len(signature.output_keys) != 1:
        raise TypeError(f"{signature.__name__} must define exactly one output key")

    parsed = adapter.parse(signature, lm_out)
    if parsed.output is None:
        return ProposalParseResult(text=None, error=parsed.error)
    return ProposalParseResult(text=parsed.output[signature.output_keys[0]])


def run_proposal(
    signature: type[Signature], lm: LanguageModel, input_dict: Mapping[str, Any]
) -> tuple[ProposalParseResult, str | list[dict[str, Any]], str]:
    """Render, complete, and parse a proposal through the non-throwing facade."""
    prompt = signature.prompt_renderer(input_dict)
    raw_output = lm(prompt).strip()
    return parse_proposal(signature, raw_output), prompt, raw_output


class InstructionProposalSignature(Signature):
    default_prompt_template = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_param>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<side_info>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""

    input_keys: ClassVar[list[str]] = ["current_instruction_doc", "dataset_with_feedback", "prompt_template"]
    output_keys: ClassVar[list[str]] = ["new_instruction"]
    adapter: ClassVar[SignatureAdapter | None] = _INSTRUCTION_PROPOSAL_ADAPTER
    reasoning_tags: ClassVar[tuple[str, ...]] = ("think",)

    @classmethod
    def validate_prompt_template(cls, prompt_template: str | None) -> None:
        if prompt_template is None:
            return
        missing_placeholders = [
            placeholder for placeholder in ("<curr_param>", "<side_info>") if placeholder not in prompt_template
        ]
        if missing_placeholders:
            raise ValueError(f"Missing placeholder(s) in prompt template: {', '.join(missing_placeholders)}")

    @classmethod
    def prompt_renderer(cls, input_dict: Mapping[str, Any]) -> str | list[dict[str, Any]]:
        current_instruction = input_dict.get("current_instruction_doc")
        if not isinstance(current_instruction, str):
            raise TypeError("current_instruction_doc must be a string")

        dataset = input_dict.get("dataset_with_feedback")
        if not isinstance(dataset, Sequence) or isinstance(dataset, str | bytes):
            raise TypeError("dataset_with_feedback must be a sequence of records")

        def format_samples(samples: Sequence[Mapping[str, Any]]) -> tuple[str, list[Image]]:
            """Render samples as markdown, extracting any Image objects.

            Returns:
                A tuple of (formatted_text, collected_images).  Image objects
                are replaced with ``[IMAGE-N]`` placeholders in the text.
            """
            collected_images: list[Image] = []

            def render_value(value: Any, level: int = 3) -> str:
                # level controls markdown header depth (###, ####, etc.)
                if isinstance(value, Image):
                    collected_images.append(value)
                    return f"[IMAGE-{len(collected_images)} — see visual content]\n\n"
                elif isinstance(value, dict):
                    s = ""
                    for k, v in value.items():
                        s += f"{'#' * level} {k}\n"
                        s += render_value(v, min(level + 1, 6))
                    if not value:
                        s += "\n"
                    return s
                elif isinstance(value, list | tuple):
                    s = ""
                    for i, item in enumerate(value):
                        s += f"{'#' * level} Item {i + 1}\n"
                        s += render_value(item, min(level + 1, 6))
                    if not value:
                        s += "\n"
                    return s
                else:
                    return f"{str(value).strip()}\n\n"

            def convert_sample_to_markdown(sample: Mapping[str, Any], examplenum: int) -> str:
                s = f"# Example {examplenum}\n"
                for key, val in sample.items():
                    s += f"## {key}\n"
                    s += render_value(val, level=3)
                return s

            text = "\n\n".join(convert_sample_to_markdown(sample, i + 1) for i, sample in enumerate(samples))
            return text, collected_images

        prompt_template = input_dict.get("prompt_template")
        if prompt_template is None:
            prompt_template = cls.default_prompt_template

        cls.validate_prompt_template(prompt_template)

        formatted_text, images = format_samples(dataset)

        if images:
            formatted_text = (
                f"The evaluation data below includes visual content ({len(images)} image(s)). "
                "Analyze both the text and images when suggesting improvements.\n\n" + formatted_text
            )

        prompt = prompt_template.replace("<curr_param>", current_instruction)
        prompt = prompt.replace("<side_info>", formatted_text)

        # When images are present, return an OpenAI-compatible multimodal
        # messages list so the reflection LM receives the images inline.
        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in images:
                content.append(img.to_openai_content_part())
            return [{"role": "user", "content": content}]

        return prompt

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        """Compatibility boundary for callers that require a proposal or error."""
        return {"new_instruction": parse_proposal(cls, lm_out).require()}
