# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import yaml
from typing import Any

from gepa.proposer.reflective_mutation.base import Signature


class EvolveProgramProposalSignature(Signature):
    prompt_template = """You are an expert programmer helping to improve code through evolution.

Here's the current program:
```
<curr_program>
```

Here are examples of the program's execution on different inputs, along with feedback on how it could be improved:
```
<dataset_with_feedback>
```

Your task is to propose an improved version of the code that addresses the feedback and performs better.

IMPORTANT: The program contains an EVOLVE-BLOCK section (marked with # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END). You should ONLY propose the improved code that should go INSIDE this block. Do NOT include:
- The EVOLVE-BLOCK markers themselves
- Code outside the block (like evaluate_function, run_search, or if __name__ == "__main__" blocks)
- Fixed code that appears after the EVOLVE-BLOCK-END marker

Only propose the evolvable code that should replace the content between the markers. The rest of the program will be preserved automatically.

CRITICAL REQUIREMENTS:
- You must provide a COMPLETE, self-contained code block that can be inserted into the EVOLVE-BLOCK
- The code must include the full function definition (def function_name(...):)
- The code must include all necessary logic and a return statement
- Do NOT provide partial code snippets, comments-only sections, or incomplete implementations
- The code should be syntactically valid Python that can run independently when inserted into the block

Analyze the current code, the execution examples, and the feedback carefully. Identify patterns in the failures and areas for improvement.

Provide the improved code (just the evolvable block content) within ``` blocks."""
    
    input_keys = ["curr_program", "dataset_with_feedback"]
    output_keys = ["new_program"]

    @classmethod
    def prompt_renderer(cls, input_dict: dict[str, str]) -> str:
        def format_samples(samples):
            # Serialize the samples list to YAML for concise, structured representation
            yaml_str = yaml.dump(samples, sort_keys=False, default_flow_style=False, indent=2, allow_unicode=True)
            return yaml_str

        prompt = cls.prompt_template
        prompt = prompt.replace("<curr_program>", input_dict["curr_program"])
        prompt = prompt.replace("<dataset_with_feedback>", format_samples(input_dict["dataset_with_feedback"]))
        return prompt

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        """ Extracts code from LLM output """
        import logging
        
        lm_out = lm_out.strip()
        
        # Find first opening ```
        start_idx = lm_out.find("```")
        if start_idx == -1:
            # No code block markers, return full output
            logging.warning("No code block markers found in LLM output, using full output")
            return {"new_program": lm_out}
        
        # Find the NEXT closing ``` after the opening one
        # (not the last one, to handle cases with multiple blocks or explanations)
        code_start = start_idx + 3  # Skip past opening ```
        
        # Check for language identifier (e.g., ```python)
        remaining_after_start = lm_out[code_start:].strip()
        if remaining_after_start.startswith("python"):
            code_start += 6
        elif remaining_after_start.startswith("py"):
            code_start += 2
        
        # Find the closing ``` that matches this opening block
        # Start searching from after the opening marker
        search_start = start_idx + 3
        end_idx = lm_out.find("```", search_start)
        
        if end_idx == -1:
            # No closing marker found - response may be truncated
            logging.warning("No closing code block marker found; LLM response may be truncated")
            # Extract what we have, but log a warning
            new_program = lm_out[code_start:].strip()
        else:
            # Extract content between opening and closing markers
            new_program = lm_out[code_start:end_idx].strip()
        
        # Fallback if extracted code is empty
        if not new_program:
            return {"new_program": lm_out}
        
        return {"new_program": new_program}

    @classmethod
    def from_config(cls, prompt_config: dict[str, Any]) -> type:
        """
        Create a custom signature class using system message from config.
        
        Args:
            prompt_config: Dictionary from config["prompt"], may contain:
                - system_message: Optional system message to prepend to the default template
        
        Returns:
            A Signature class configured with the system message prepended, or the default class if no system_message
        """
        system_message = prompt_config.get("system_message", "")
        
        if not system_message:
            # No system_message, return default class
            return cls
        
        # Create custom class to prepend system_message to default template
        prompt_template = f"{system_message}\n\n{cls.prompt_template}"
        
        # Capture prompt_template in closure
        captured_template = prompt_template
        
        # Create a new class with the custom template
        class CustomEvolveProgramProposalSignature(Signature):
            prompt_template = captured_template
            input_keys = cls.input_keys
            output_keys = cls.output_keys
            
            @classmethod
            def prompt_renderer(cls, input_dict: dict[str, str]) -> str:
                def format_samples(samples):
                    yaml_str = yaml.dump(samples, sort_keys=False, default_flow_style=False, indent=2, allow_unicode=True)
                    return yaml_str
                
                rendered = captured_template  # Use the captured prompt_template from closure
                # Support both placeholder formats for compatibility
                rendered = rendered.replace("<current_instruction_doc>", input_dict.get("curr_program", ""))
                rendered = rendered.replace("<curr_program>", input_dict.get("curr_program", ""))
                rendered = rendered.replace("<dataset_with_feedback>", format_samples(input_dict.get("dataset_with_feedback", [])))
                return rendered
            
            @classmethod
            def output_extractor(cls, lm_out: str) -> dict[str, str]:
                # Reuse the default output extractor
                return EvolveProgramProposalSignature.output_extractor(lm_out)
        
        return CustomEvolveProgramProposalSignature
