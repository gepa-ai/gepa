# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from gepa.image import Image
from gepa.proposer.reflective_mutation.base import LanguageModel, Signature


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
        def extract_instruction_text() -> str:
            # Find the first and last backtick positions (if any)
            start = lm_out.find("```") + 3
            end = lm_out.rfind("```")

            # Handle if the first and last backticks are the same or overlap
            if start >= end:
                # Handle incomplete blocks
                stripped = lm_out.strip()
                if stripped.startswith("```"):
                    # Remove opening ``` and optional language specifier
                    match = re.match(r"^```\S*\n?", lm_out)
                    if match:
                        return lm_out[match.end() :].strip()
                elif stripped.endswith("```"):
                    # Remove closing ```
                    return stripped[:-3].strip()
                return stripped

            # Skip optional language specifier
            content = lm_out[start:end]
            match = re.match(r"^\S*\n", content)
            if match:
                content = content[match.end() :]

            return content.strip()

        return {"new_instruction": extract_instruction_text()}


class InstructionEditSignature(InstructionProposalSignature):
    """Asks the LLM to output search/replace edit blocks instead of rewriting
    the entire instruction. This saves output tokens for long candidates and
    preserves parts of the instruction that are already working well.

    The LLM outputs one or more edit blocks in this format::

        <<<<<<< SEARCH
        exact text to find
        =======
        replacement text
        >>>>>>> REPLACE

    The output_extractor applies each block sequentially to the original text.
    If no valid edit blocks are found, falls back to extracting a full rewrite
    from ``` blocks (same as InstructionProposalSignature).
    """

    default_prompt_template = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_param>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<side_info>
```

Your task is to improve the existing instruction by making targeted edits. Do NOT rewrite the instruction from scratch.

Read all the assistant responses and the corresponding feedback. Identify what needs to change and make minimal, targeted edits.

Express your edits as one or more SEARCH/REPLACE blocks. Each block finds an exact passage in the current instruction and replaces it:

<<<<<<< SEARCH
exact text from the current instruction to find
=======
replacement text
>>>>>>> REPLACE

Rules:
- The SEARCH section must match the current instruction EXACTLY (including whitespace and newlines).
- You can use multiple SEARCH/REPLACE blocks to make several edits.
- To insert new text, use a SEARCH block that matches the text just before where you want to insert, and include that text plus the new content in REPLACE.
- To delete text, use an empty REPLACE section."""

    @classmethod
    def _apply_edits(cls, original: str, lm_out: str) -> str | None:
        """Parse SEARCH/REPLACE blocks from lm_out and apply them to original.

        Returns the edited text, or None if no valid edit blocks were found.
        """
        pattern = re.compile(
            r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
            re.DOTALL,
        )
        matches = list(pattern.finditer(lm_out))
        if not matches:
            return None

        result = original
        for match in matches:
            search_text = match.group(1)
            replace_text = match.group(2)
            if search_text in result:
                result = result.replace(search_text, replace_text, 1)
        return result

    @classmethod
    def run(cls, lm: "LanguageModel", input_dict: Mapping[str, Any]) -> dict[str, str]:
        original_instruction = input_dict.get("current_instruction_doc", "")
        full_prompt = cls.prompt_renderer(input_dict)
        lm_res = lm(full_prompt)
        lm_out = lm_res.strip()

        # Try to apply search/replace edits
        edited = cls._apply_edits(str(original_instruction), lm_out)
        if edited is not None:
            return {"new_instruction": edited}

        # Fallback: extract full rewrite from ``` blocks (same as parent)
        return cls.output_extractor(lm_out)
