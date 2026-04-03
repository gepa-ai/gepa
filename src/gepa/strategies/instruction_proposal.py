# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import difflib
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


_EDIT_BLOCK_PATTERN = re.compile(
    r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)


def _fuzzy_find_and_replace(text: str, search: str, replace: str) -> tuple[str, bool]:
    """Try to find *search* in *text* with progressively looser matching.

    Returns (new_text, matched).  If no match is found at any level the
    original text is returned with matched=False.

    Matching levels tried in order:
    1. Exact match (fastest)
    2. Whitespace-normalized match — collapse runs of whitespace on both
       sides before comparing, then replace the original span.
    3. Line-level fuzzy match — use difflib.SequenceMatcher to find the
       best-matching contiguous block of lines (ratio >= 0.6).
    """
    if not search:
        return text, False

    # --- Level 1: exact ---
    if search in text:
        return text.replace(search, replace, 1), True

    # --- Level 2: whitespace-normalized ---
    def _ws_normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    norm_search = _ws_normalize(search)
    if not norm_search:
        return text, False

    # Slide a window over the text to find a span whose normalized form
    # matches.  We use a line-based approach to keep it efficient.
    text_lines = text.split("\n")
    search_lines = search.split("\n")
    search_line_count = len(search_lines)

    for start_idx in range(len(text_lines) - search_line_count + 1):
        candidate = "\n".join(text_lines[start_idx : start_idx + search_line_count])
        if _ws_normalize(candidate) == norm_search:
            before = "\n".join(text_lines[:start_idx])
            after = "\n".join(text_lines[start_idx + search_line_count :])
            parts = [p for p in (before, replace, after) if p]
            return "\n".join(parts), True

    # --- Level 3: line-level fuzzy match ---
    best_ratio = 0.0
    best_start = -1
    best_end = -1
    # Try window sizes around the search line count (±30%)
    min_window = max(1, int(search_line_count * 0.7))
    max_window = min(len(text_lines), int(search_line_count * 1.3) + 1)

    for window_size in range(min_window, max_window + 1):
        for start_idx in range(len(text_lines) - window_size + 1):
            candidate_lines = text_lines[start_idx : start_idx + window_size]
            # For single-line comparisons, compare at character level;
            # for multi-line, compare at line level.
            if window_size == 1 and search_line_count == 1:
                ratio = difflib.SequenceMatcher(None, search_lines[0], candidate_lines[0]).ratio()
            else:
                ratio = difflib.SequenceMatcher(None, search_lines, candidate_lines).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start_idx
                best_end = start_idx + window_size

    if best_ratio >= 0.6 and best_start >= 0:
        before = "\n".join(text_lines[:best_start])
        after = "\n".join(text_lines[best_end:])
        parts = [p for p in (before, replace, after) if p]
        return "\n".join(parts), True

    return text, False


class InstructionEditSignature(InstructionProposalSignature):
    """Asks the LLM to output search/replace edit blocks instead of rewriting
    the entire instruction.  This saves output tokens for long candidates and
    preserves parts of the instruction that are already working well.

    The LLM outputs one or more edit blocks in this format::

        <<<<<<< SEARCH
        exact text to find
        =======
        replacement text
        >>>>>>> REPLACE

    Each block is applied sequentially.  If an exact match fails, a fuzzy
    matching cascade is attempted (whitespace-normalized, then line-level
    similarity).  If a block still cannot be matched, the LLM is retried
    once with error feedback.  If no valid edit blocks are found at all,
    falls back to extracting a full rewrite from ``` blocks.
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
    def _apply_edits_with_retry(
        cls,
        original_instruction: str,
        lm_out: str,
        lm: "LanguageModel",
        max_retries: int = 1,
    ) -> tuple[dict[str, str], str]:
        """Apply SEARCH/REPLACE edits with retry on failure, returning the result and final raw output.

        Shared by both ``run()`` and ``run_with_metadata()`` to avoid duplicating
        the retry + fallback logic.

        Returns:
            (result_dict, final_raw_lm_output) where result_dict has key ``"new_instruction"``.
        """
        edited, failed = cls._apply_edits(original_instruction, lm_out)

        retries = 0
        while failed and retries < max_retries:
            retries += 1
            failed_summary = "\n\n".join(
                f"SEARCH block {i + 1} not found:\n```\n{s}\n```" for i, s in enumerate(failed)
            )
            retry_prompt = (
                f"Your previous edit contained {len(failed)} SEARCH block(s) that could not be "
                f"matched against the current instruction (even with fuzzy matching).\n\n"
                f"{failed_summary}\n\n"
                f"The current instruction is:\n```\n{edited if edited is not None else original_instruction}\n```\n\n"
                "Please provide corrected SEARCH/REPLACE blocks. The SEARCH text must appear "
                "verbatim in the current instruction above."
            )
            lm_out = lm(retry_prompt).strip()
            source = edited if edited is not None else original_instruction
            edited, failed = cls._apply_edits(source, lm_out)

        if edited is not None:
            return {"new_instruction": edited}, lm_out

        # Fallback: extract full rewrite from ``` blocks (same as parent)
        return cls.output_extractor(lm_out), lm_out

    @classmethod
    def run_with_metadata(
        cls, lm: "LanguageModel", input_dict: Mapping[str, Any]
    ) -> tuple[dict[str, str], str | list[dict[str, Any]], str]:
        """Like ``run()``, but also returns the rendered prompt and raw LM output.

        Overrides the base class to route through the SEARCH/REPLACE parsing,
        retry logic, and fuzzy matching pipeline.
        """
        full_prompt = cls.prompt_renderer(input_dict)

        # For multimodal prompts (list of message dicts with images), extract
        # the text component for the LM call — the LM receives the full
        # multimodal prompt, but we need the text for logging.
        if isinstance(full_prompt, list):
            lm_res = lm(full_prompt)
        else:
            lm_res = lm(full_prompt)
        raw_lm_out = lm_res.strip()

        original_instruction = str(input_dict.get("current_instruction_doc", ""))
        result, final_raw = cls._apply_edits_with_retry(original_instruction, raw_lm_out, lm)
        return result, full_prompt, final_raw

    @classmethod
    def _apply_edits(cls, original: str, lm_out: str) -> tuple[str | None, list[str]]:
        """Parse SEARCH/REPLACE blocks from *lm_out* and apply them to *original*.

        Returns ``(edited_text, failed_searches)`` where *edited_text* is
        ``None`` when no edit blocks were found at all, and
        *failed_searches* lists SEARCH strings that could not be matched
        even after fuzzy fallback.
        """
        matches = list(_EDIT_BLOCK_PATTERN.finditer(lm_out))
        if not matches:
            return None, []

        result = original
        failed: list[str] = []
        for match in matches:
            search_text = match.group(1)
            replace_text = match.group(2)
            new_result, matched = _fuzzy_find_and_replace(result, search_text, replace_text)
            if matched:
                result = new_result
            else:
                failed.append(search_text)

        return result, failed

    @classmethod
    def run(
        cls,
        lm: LanguageModel,
        input_dict: Mapping[str, Any],
        *,
        max_retries: int = 1,
    ) -> dict[str, str]:
        original_instruction = str(input_dict.get("current_instruction_doc", ""))
        full_prompt = cls.prompt_renderer(input_dict)
        lm_out = lm(full_prompt).strip()
        result, _ = cls._apply_edits_with_retry(original_instruction, lm_out, lm, max_retries)
        return result
