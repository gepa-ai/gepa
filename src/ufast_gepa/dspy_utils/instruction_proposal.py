"""
Instruction proposal prompt for LLM-based reflection.

Adapted from GEPA's InstructionProposalSignature for uFast-GEPA's async architecture.
"""

import re
from typing import Any, Dict, List


class InstructionProposalPrompt:
    """
    LLM prompt template for generating improved instructions from feedback.

    This is adapted from GEPA's InstructionProposalSignature to work with
    uFast-GEPA's async reflection system.
    """

    TEMPLATE = """I provided an assistant with the following instructions to perform a task for me:
```
{current_instruction}
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
{formatted_dataset}
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""

    @staticmethod
    def format_dataset(samples: List[Dict[str, Any]]) -> str:
        """
        Format a list of feedback samples into markdown.

        Args:
            samples: List of dicts with "Inputs", "Generated Outputs", "Feedback" keys

        Returns:
            Formatted markdown string
        """

        def render_value(value: Any, level: int = 3) -> str:
            """Recursively render a value as markdown."""
            if isinstance(value, dict):
                s = ""
                for k, v in value.items():
                    s += f"{'#' * level} {k}\n"
                    s += render_value(v, min(level + 1, 6))
                if not value:
                    s += "\n"
                return s
            elif isinstance(value, (list, tuple)):
                s = ""
                for i, item in enumerate(value):
                    s += f"{'#' * level} Item {i + 1}\n"
                    s += render_value(item, min(level + 1, 6))
                if not value:
                    s += "\n"
                return s
            else:
                return f"{str(value).strip()}\n\n"

        def convert_sample_to_markdown(sample: Dict[str, Any], examplenum: int) -> str:
            """Convert a single sample to markdown."""
            s = f"# Example {examplenum}\n"
            for key, val in sample.items():
                s += f"## {key}\n"
                s += render_value(val, level=3)
            return s

        return "\n\n".join(
            convert_sample_to_markdown(sample, i + 1) for i, sample in enumerate(samples)
        )

    @staticmethod
    def build_prompt(current_instruction: str, dataset_with_feedback: List[Dict[str, Any]]) -> str:
        """
        Build the full LLM prompt for instruction proposal.

        Args:
            current_instruction: The current instruction text
            dataset_with_feedback: List of feedback samples

        Returns:
            Full prompt string
        """
        return InstructionProposalPrompt.TEMPLATE.format(
            current_instruction=current_instruction,
            formatted_dataset=InstructionProposalPrompt.format_dataset(dataset_with_feedback),
        )

    @staticmethod
    def extract_instruction(lm_output: str) -> str:
        """
        Extract the instruction from LLM output.

        Handles various formats:
        - Text between ``` blocks
        - Text with opening ``` but no closing
        - Text with closing ``` but no opening
        - Plain text

        Args:
            lm_output: Raw LLM response

        Returns:
            Extracted instruction text
        """
        # Find the first and last backtick positions
        start = lm_output.find("```") + 3
        end = lm_output.rfind("```")

        # Handle incomplete blocks
        if start >= end:
            stripped = lm_output.strip()
            if stripped.startswith("```"):
                # Remove opening ``` and optional language specifier
                match = re.match(r"^```\S*\n?", lm_output)
                if match:
                    return lm_output[match.end() :].strip()
            elif stripped.endswith("```"):
                # Remove closing ```
                return stripped[:-3].strip()
            return stripped

        # Extract content between backticks
        content = lm_output[start:end]

        # Skip optional language specifier (e.g., ```python\n)
        match = re.match(r"^\S*\n", content)
        if match:
            content = content[match.end() :]

        return content.strip()
