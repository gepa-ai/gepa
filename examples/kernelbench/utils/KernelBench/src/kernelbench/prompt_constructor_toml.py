# src/prompt_constructor_toml.py | toml based prompt constructor
import os
import runpy
import tomli  
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from kernelbench.utils import read_file, get_package_resource_path, resolve_path, REPO_TOP_PATH

"""
TOML-based prompt constructor for managing prompt templates and configurations.
This module provides a way to load and compose prompt templates from a TOML configuration file.

You can easily check some of the prompt templates we have provided and create your own.
"""

# Resolve paths using the helper from utils
PROMPTS_TOML = get_package_resource_path("prompts/prompts.toml")
GPU_SPECS_PY = get_package_resource_path("prompts/hardware/gpu_specs.py")
HARDWARE_COMPONENT_KEYS = [
    "hardware_header",
    "hardware_specs",
    "hardware_definitions",
    "hardware_best_practices",
]

@dataclass
class PromptConfig:
    """
    Configuration wrapper for prompts.toml data.
    
    This class holds the parsed TOML file data and provides ways to navigate 
    the nested structure and compose prompt templates.
    
    The TOML file has a  structure like:
        [backends.cuda]
        [options.few_shot]
        [templates.common.arch_block]
    
    This class makes it easy to look up values in that hierarchy.
    """
    data: Dict[str, Any]  # The raw parsed TOML data as nested dictionaries

    @classmethod
    def from_toml(cls, path: str) -> "PromptConfig":
        """
        Load and parse a TOML configuration file.
        
        Args:
            path: Filesystem path to the prompts.toml file
            
        Returns:
            PromptConfig instance with parsed data
        """
        with open(path, "rb") as f:
            data = tomli.load(f)
        return cls(data)

    def compose_blocks(self, keys: List[str]) -> str:
        """
        Look up and concatenate multiple template blocks using dotted key paths.
        
        This method navigates the nested TOML structure using dotted notation
        (e.g., "templates.common.arch_block") to find template strings, then
        concatenates them together with newlines.
        
        Args:
            keys: List of dotted key paths (e.g., ["templates.common.arch_block"])
                  Each key is split on "." and used to traverse the nested dict.
                  
        Returns:
            Concatenated string of all template blocks, each separated by newlines
        """
        text_parts = []
        for key in keys:
            # Navigate through the nested dictionary structure
            node: Any = self.data
            for part in key.split("."):
                if part not in node:
                    raise KeyError(f"compose key not found: {key}")
                node = node[part]
            
            # Ensure we found a string template, not another dict/list
            if not isinstance(node, str):
                raise TypeError(f"compose key must resolve to string: {key}")
            
            text_parts.append(node.strip() + "\n")
        
        return "\n".join(text_parts).strip() + "\n"

def _gpu_context_from_gpu_specs(py_path: str, gpu_name: str) -> Dict[str, str]:
    """
    Load GPU_* dicts from the GPU specs file (no exec of raw strings; use runpy).
    Expected globals:
      - GPU_SPEC_INFO: dict[str, dict]
      - GPU_DEFINITIONS: dict[str, str]
      - GPU_BEST_PRACTICES: list[str]  OR {"list": [...]} for compatibility
    """
    mod = runpy.run_path(py_path)
    spec_info = mod.get("GPU_SPEC_INFO", {})
    definitions = mod.get("GPU_DEFINITIONS", {})
    best = mod.get("GPU_BEST_PRACTICES", [])

    if not spec_info or not definitions or best is None:
        raise ValueError("GPU_SPEC_INFO / GPU_DEFINITIONS / GPU_BEST_PRACTICES missing in gpu specs .py")

    if isinstance(best, dict) and "list" in best:
        best = best["list"]

    if gpu_name not in spec_info:
        raise KeyError(f"GPU name {gpu_name} not found in GPU_SPEC_INFO")

    curr = spec_info[gpu_name]
    gpu_architecture = curr.get("GPU Architecture", "Unknown")
    specs_bullets = "\n".join([f"- We have {v} of {k}." for k, v in curr.items() if k != "GPU Architecture"])
    defs_bullets = "\n".join([f"- {k}: {v}" for k, v in definitions.items()])
    best_bullets = "\n".join([f"- {x}" for x in (best or [])])

    return {
        "gpu_name": gpu_name,
        "gpu_architecture": gpu_architecture,
        "gpu_specs_bullets": specs_bullets,
        "gpu_definitions_bullets": defs_bullets,
        "gpu_best_practices_bullets": best_bullets,
    }

def render_prompt_by_option(
    *,
    prompts_toml: str,
    backend: str,
    option: str,
    context: Dict[str, str],
    gpu_specs_py: Optional[str] = None,
    gpu_name: Optional[str] = None,
    precision: Optional[str] = None,
    include_hardware: bool = False,
    components_override: Optional[List[str]] = None,
) -> str:
    """
    Render a prompt using backends.X and options.Y structure from TOML.
    
    Args:
        prompts_toml: Path to the prompts.toml file
        backend: The kernel backend (triton, cuda, cute, tilelang)
        option: The prompt option (zero_shot, one_shot, few_shot)
                - zero_shot: No examples (model learns from description only)
                - one_shot: Single example
                - few_shot: Multiple examples if available for backend, otherwise falls back to one_shot
        context: Variables to fill in the prompt template
        gpu_specs_py: Optional path to GPU specs Python file (required if hardware info is included)
        gpu_name: Optional GPU name (required if hardware info is included)
        precision: Optional precision string (fp32, fp16, bf16) - defaults to fp32 if not provided
        include_hardware: Whether to inject hardware guidance blocks after the examples section
        components_override: When provided, users can arrange prompt components from the toml
                             file in any order they want.
                             Components must exist under templates.common or be hardware_* entries.
    
    Returns:
        The rendered prompt string
    """
    cfg = PromptConfig.from_toml(prompts_toml)
    
    # Get backend-specific content
    try:
        backend_data = cfg.data["backends"][backend]
    except KeyError:
        raise KeyError(f"Unknown backend: {backend}")
    
    # Get option configuration
    try:
        option_data = cfg.data["options"][option]
    except KeyError:
        raise KeyError(f"Unknown option: {option}")

    component_sequence = list(components_override or option_data["components"])
    if include_hardware:
        if components_override is None:
            insert_idx = component_sequence.index("arch_block") if "arch_block" in component_sequence else len(component_sequence)
            component_sequence[insert_idx:insert_idx] = HARDWARE_COMPONENT_KEYS
        else:
            # Custom sequences must explicitly have hardware blocks present in their prompt if they 
            # have set they are including hardware info. 
            if not any(component in HARDWARE_COMPONENT_KEYS for component in component_sequence):
                raise ValueError(
                    "components_override must contain at least one hardware_* entry when include_hardware=True"
                )
    
    # Get shared templates
    shared = cfg.data.get("shared", {})
    backend_display = backend_data.get("backend_display", backend.upper())
    
    # Fill in shared templates with backend-specific terms
    problem_statement = shared.get("problem_statement", "").format(backend_display=backend_display)
    instruction = shared.get("instruction", "").format(backend_display=backend_display)
    
    # Add backend-specific content to context
    context = {
        **context,
        "backend": backend.upper() if backend in ["cuda", "cute"] else backend.capitalize(),
        "backend_display": backend_display,
        "problem_statement": problem_statement,
        "instruction": instruction,
    }
    
    # Load precision details if provided
    if precision:
        try:
            precision_data = cfg.data["precision"][precision]
            context["precision_display"] = precision_data.get("precision_display", precision.upper())
        except KeyError:
            raise KeyError(f"Unknown precision: {precision}. Must be one of: fp32, fp16, bf16")
    else:
        # Default to fp32 if not specified
        default_precision = cfg.data.get("meta", {}).get("default_precision", "fp32")
        precision_data = cfg.data["precision"].get(default_precision, {})
        context["precision_display"] = precision_data.get("precision_display", "FP32 (32-bit floating point)")
    
    # Load example files if requested. Supports loading one shot or few shot examples. 
    requires_example = option_data.get("requires_example")
    if requires_example:
        example_entry_template = cfg.compose_blocks(["templates.common.example_entry_template"]).strip()
        intro_one_shot = cfg.compose_blocks(["templates.common.example_intro_one_shot"]).strip()
        intro_few_shot = cfg.compose_blocks(["templates.common.example_intro_few_shot"]).strip()
        intro_one_shot = intro_one_shot.format(
            backend_display=backend_display
        )
        intro_few_shot = intro_few_shot.format(
            backend_display=backend_display
        )

        def render_example_entry(input_code: str, output_code: str, example_label: str) -> str:
            return example_entry_template.format(
                example_label=example_label,
                input_code=input_code,
                output_code=output_code,
                backend_display=backend_display,
            )

        examples_entries: List[str] = []
        examples_intro = intro_one_shot

        if requires_example == "few_shot":
            # Try to load few-shot examples if available
            few_shot_examples = backend_data.get("few_shot_examples")

            if few_shot_examples and len(few_shot_examples) > 0:
                # Use multiple examples (true few-shot)
                examples_intro = intro_few_shot
                for i, (input_path, output_path) in enumerate(few_shot_examples, 1):
                    input_code = read_file(resolve_path(input_path))
                    output_code = read_file(resolve_path(output_path))
                    examples_entries.append(
                        render_example_entry(input_code, output_code, f"Example {i}:")
                    )
            else:
                # Fall back to one-shot
                ex_arch_path = resolve_path(
                    backend_data.get("few_shot_example_arch") or shared.get("few_shot_example_arch")
                )
                ex_new_path = resolve_path(backend_data["one_shot_new_arch"])
                input_code = read_file(ex_arch_path)
                output_code = read_file(ex_new_path)
                examples_entries.append(
                    render_example_entry(input_code, output_code, "Example:")
                )

        elif requires_example == "one_shot":
            # Always use one-shot
            ex_arch_path = resolve_path(
                backend_data.get("few_shot_example_arch") or shared.get("few_shot_example_arch")
            )
            ex_new_path = resolve_path(backend_data["one_shot_new_arch"])
            input_code = read_file(ex_arch_path)
            output_code = read_file(ex_new_path)
            examples_entries.append(
                render_example_entry(input_code, output_code, "Example:")
            )

        if not examples_entries:
            raise ValueError(f"No example entries could be constructed for option '{option}'.")

        context["examples_intro"] = examples_intro
        context["examples_entries"] = "\n\n".join(examples_entries).strip()
    
    # Load GPU details if requested
    if option_data.get("requires_gpu") or include_hardware:
        if not (gpu_specs_py and gpu_name):
            raise ValueError(
                f"Hardware info requested for option '{option}'; provide gpu_specs_py and gpu_name"
            )
        context = {**context, **_gpu_context_from_gpu_specs(resolve_path(gpu_specs_py), gpu_name)}
    
    # Builds the prompt from the components in the toml file. 
    prompt_parts = []
    for component in component_sequence:
        if component == "problem_statement":
            # Use the already-formatted problem_statement from context
            prompt_parts.append(context["problem_statement"])
        elif component == "instruction":
            # Use the already-formatted instruction from context
            prompt_parts.append(context["instruction"])
        elif component.startswith("hardware_"):
            # Hardware components from templates.hardware
            template_key = f"templates.hardware.{component}"
            prompt_parts.append(cfg.compose_blocks([template_key]))
        else:
            # Other components from templates.common
            template_key = f"templates.common.{component}"
            prompt_parts.append(cfg.compose_blocks([template_key]))
    
    prompt_text = "\n".join(prompt_parts).strip() + "\n"
    
    try:
        return prompt_text.format(**context).strip() + "\n"
    except KeyError as e:
        raise KeyError(f"Missing placeholder in context: {e.args[0]}. Available: {list(context.keys())}") from e

# -------------------------------------------------------------------------
# High-level convenience functions
# -------------------------------------------------------------------------

def get_prompt_for_backend(
    ref_arch_src: str,
    backend: str = "triton",
    option: str = "one_shot",
    precision: Optional[str] = None,
    include_hardware: bool = False,
    gpu_name: Optional[str] = None,
) -> str:
    """
    Generate a prompt for a specific backend and option.
    
    Args:
        ref_arch_src: The reference architecture source code
        backend: The kernel backend (triton, cuda, cute, tilelang)
        option: The prompt option (zero_shot, one_shot, few_shot)
        precision: Optional precision (fp32, fp16, bf16) - defaults to fp32 if not provided
        include_hardware: When True, append hardware guidance blocks (requires gpu_name)
        gpu_name: GPU identifier used when include_hardware is True (e.g., "A100")
    """
    return render_prompt_by_option(
        prompts_toml=PROMPTS_TOML,
        backend=backend.lower(),
        option=option.lower(),
        context={"ref_arch_src": ref_arch_src},
        precision=precision,
        include_hardware=include_hardware,
        gpu_specs_py=GPU_SPECS_PY if include_hardware else None,
        gpu_name=gpu_name,
    )


def get_custom_prompt(
    custom_key: str,
    *,
    ref_arch_src: str,
    backend: str,
    option: str,
    precision: Optional[str] = None,
    include_hardware: bool = False,
    gpu_name: Optional[str] = None,
    prompts_toml: str = PROMPTS_TOML,
) -> str:
    """
    Render a prompt defined under [custom_prompts.<custom_key>] in prompts.toml.
    Must still provide backend/option/precision settings just like
    get_prompt_for_backend. 
    """
    if not ref_arch_src:
        raise ValueError(f"Custom prompt '{custom_key}' requires ref_arch_src.")
    cfg = PromptConfig.from_toml(prompts_toml)
    try:
        custom_cfg: Dict[str, Any] = cfg.data["custom_prompts"][custom_key]
    except KeyError as exc:
        raise KeyError(f"Unknown custom prompt: {custom_key}") from exc

    components_override = custom_cfg.get("components")

    return render_prompt_by_option(
        prompts_toml=prompts_toml,
        backend=backend.lower(),
        option=option.lower(),
        context={"ref_arch_src": ref_arch_src},
        precision=precision,
        include_hardware=include_hardware,
        gpu_specs_py=GPU_SPECS_PY if include_hardware else None,
        gpu_name=gpu_name,
        components_override=components_override,
    )

__all__ = [
    "get_prompt_for_backend",
    "get_custom_prompt",
    "get_prompt_with_hardware",
    "render_prompt_by_option",
    "PromptConfig",
]


def log_prompt(prompt: str, dir_path: str, file_name: str):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, file_name), "w") as f:
        f.write(prompt)

def test_prompt():
    """
    Demonstrate baseline, few-shot, DSL, hardware-aware, and custom prompt
    generation. Customize the reference architecture or custom_prompt_key
    if you want to try different inputs.
    """
    ref_arch_src = read_file(os.path.join(REPO_TOP_PATH, "KernelBench", "level1", "1_Square_matrix_multiplication_.py"))
    assert len(ref_arch_src) > 0, "ref_arch_src is empty"   

    print("Testing prompt construction...")
    scratch_dir = os.path.join(REPO_TOP_PATH, "scratch")
    # baseline prompt
    baseline_prompt = get_prompt_for_backend(
        ref_arch_src=ref_arch_src,
        backend="cuda",
        option="one_shot",
        precision="fp32",
        # GPU platform agnostic for baseline
    )
    log_prompt(baseline_prompt, os.path.join(scratch_dir), "baseline_prompt.txt")

    # few shot prompt
    few_shot_prompt = get_prompt_for_backend(
        ref_arch_src=ref_arch_src,
        backend="cuda",
        option="few_shot",
        precision="fp32",
    )
    log_prompt(few_shot_prompt, os.path.join(scratch_dir), "few_shot_prompt.txt")

    # DSL prompt
    dsl_prompt = get_prompt_for_backend(
        ref_arch_src=ref_arch_src,
        backend="triton",
        option="one_shot",
        precision="fp32",
    )
    log_prompt(dsl_prompt, os.path.join(scratch_dir), "dsl_prompt.txt")

    # hardware prompt
    hardware_prompt = get_prompt_for_backend(
        ref_arch_src=ref_arch_src,
        backend="cute",
        option="one_shot",
        precision="fp32",
        include_hardware=True,
        gpu_name="L40S",
    )
    log_prompt(hardware_prompt, os.path.join(scratch_dir), "hardware_prompt.txt")

    # custom prompt defined in prompts.toml
    custom_prompt = get_custom_prompt(
        # the key is whatever you name the prompt in the custom_prompts section of the toml file
        custom_key="custom",
        
        ref_arch_src=ref_arch_src,
        backend="triton",
        option="one_shot",
        precision="fp32",
        include_hardware=True,
        gpu_name="L40S",
    )
    log_prompt(custom_prompt, os.path.join(scratch_dir), "custom_prompt.txt")
    
if __name__ == "__main__":
    test_prompt()