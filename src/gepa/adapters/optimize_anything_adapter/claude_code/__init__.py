"""Claude Code support for the optimize_anything adapter."""

from gepa.adapters.optimize_anything_adapter.claude_code.agent import (
    AgenticProposer,
    EvalRecorderCallback,
    GlobalNote,
    NoteUpdater,
    build_mutation_prompt,
)
from gepa.adapters.optimize_anything_adapter.claude_code.prompts import (
    CC_AGENTIC_MUTATION_SYSTEM_PROMPT,
    CC_RLM_SYSTEM_PROMPT,
    build_note_update_prompt,
)
from gepa.adapters.optimize_anything_adapter.claude_code.runtime import (
    extract_text_from_stream_json,
    make_claude_code_lm,
    parse_stream_json,
)

__all__ = [
    "AgenticProposer",
    "CC_AGENTIC_MUTATION_SYSTEM_PROMPT",
    "CC_RLM_SYSTEM_PROMPT",
    "EvalRecorderCallback",
    "GlobalNote",
    "NoteUpdater",
    "build_mutation_prompt",
    "build_note_update_prompt",
    "extract_text_from_stream_json",
    "make_claude_code_lm",
    "parse_stream_json",
]
