import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from collections import defaultdict
from .al_adapter import RunTrace, ToolEvent, Candidate, ModuleSpec, MODULES
import random
# === Helpers ===

def _safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None

def _parse_debug_info_list(step: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Returns mapping: key -> [values...]
    because keys can repeat.
    """
    out = defaultdict(list)
    for item in step.get("debugInfo", []) or []:
        k = item.get("key")
        v = item.get("value")
        if k is not None:
            out[k].append(v)
    return out

def _tool_type_from_name(name: str) -> str:
    n = (name or "").lower()
    if "glean" in n:
        return "Glean"
    if "web" in n:
        return "web"
    if "mcp" in n:
        return "MCP"
    return "internal"


# === Main conversion ===

def convert_workflow_trace_to_runtrace(raw: Dict[str, Any]) -> RunTrace:
    """
    Converts your raw dict (actResponse + workflowResponseInfo) to RunTrace.
    Best-effort parsing; designed to be robust.
    """
    answer = (raw.get("actResponse") or "").strip()

    # Walk steps
    wfi = raw.get("workflowResponseInfo", {}) or {}
    steps = (wfi.get("steps") or [])

    tool_events: List[ToolEvent] = []
    total_latency = 0
    input_tokens = 0
    output_tokens = 0

    # Heuristics for loops/calls/errors
    loop_like_step_count = 0
    tool_calls = 0
    tool_errors = 0

    for step in steps:
        step_id = step.get("stepId", "")

        # latency
        dbg = _parse_debug_info_list(step)
        for k, vs in dbg.items():
            # keys end with STEP_LATENCY_MILLIS in your sample
            if k.endswith("STEP_LATENCY_MILLIS"):
                for v in vs:
                    try:
                        total_latency += int(v)
                    except Exception:
                        pass

        # count "loop" steps
        # your trace has INTERNAL_LOOPING_PYAGENT – treat that as 1 loop occurrence
        if "LOOP" in step_id.upper() or "PYAGENT" in step_id.upper():
            loop_like_step_count += 1

        # Prefer answer from PYTHON_AGENTIC_TOOL_RESPONSE if actResponse missing
        if not answer:
            for k, vs in dbg.items():
                if k.endswith("PYTHON_AGENTIC_TOOL_RESPONSE") and vs:
                    answer = (vs[-1] or "").strip()

        # Tool calls: look at ACTION_SUMMARY (some are JSON strings)
        for k, vs in dbg.items():
            if k.endswith("ACTION_SUMMARY"):
                for v in vs:
                    j = _safe_json_loads(v) if isinstance(v, str) else None
                    if not j:
                        continue

                    # Example patterns in your trace:
                    # {"tool_name":"Glean Search","tool_input":{...}}
                    # or {"last_executed_agent":"Glean Assistant"} (not a tool call)
                    tool_name = j.get("tool_name")
                    if tool_name:
                        # Parse function name if present
                        # sometimes summary contains tool_id or tool_input_raw with "name"
                        # We'll look for common fields:
                        name = j.get("tool_id") or tool_name
                        # Some systems embed a called function name under tool_input_raw or elsewhere
                        raw_in = j.get("tool_input_raw") or ""
                        called = None
                        if isinstance(raw_in, str):
                            # tool_input_raw might be JSON; ignore if not
                            _ = _safe_json_loads(raw_in)

                        # fallback: if tool_name says "Glean Search", name as "glean_search"
                        if "glean" in tool_name.lower() and "search" in tool_name.lower():
                            called = "glean_search"
                        final_name = called or name

                        # Decide ok/error best-effort
                        ok = True
                        tool_calls += 1

                        tool_events.append(
                            ToolEvent(
                                tool_type=_tool_type_from_name(final_name),
                                name=final_name,
                                ok=ok,
                            )
                        )

        # Token usage: you have an "LLM_USAGE" json string under INTERNAL_LOOPING_PYAGENT
        for k, vs in dbg.items():
            if k.endswith("LLM_USAGE"):
                for v in vs:
                    j = _safe_json_loads(v) if isinstance(v, str) else None
                    if not j:
                        continue
                    input_tokens += int(j.get("input_tokens", 0) or 0)
                    output_tokens += int(j.get("output_tokens", 0) or 0)

        # Also there are "usage" blobs inside PYTHON_AGENT_RAW_LLM_RESPONSE sometimes
        for k, vs in dbg.items():
            if k.endswith("PYTHON_AGENT_RAW_LLM_RESPONSE"):
                for v in vs:
                    j = _safe_json_loads(v) if isinstance(v, str) else None
                    if not j:
                        continue
                    usage = j.get("usage") or {}
                    input_tokens += int(usage.get("input_tokens", 0) or 0)
                    output_tokens += int(usage.get("output_tokens", 0) or 0)

        # Error detection (best-effort): scan keys/values for status != null or explicit error keys
        for k, vs in dbg.items():
            if "ERROR" in k.upper():
                tool_errors += 1
            for v in vs:
                if isinstance(v, str) and ("error" in v.lower() and "no_error" not in v.lower()):
                    # avoid over-counting; keep conservative
                    pass

    # If we saw tool events but tool_calls stayed 0 due to parsing differences, fix it
    if tool_calls == 0:
        tool_calls = len(tool_events)

    tool_errors = max(tool_errors, len([t for t in tool_events if not t.ok]))

    # loops heuristic: at least 1 if we saw the internal looping step
    num_loops = max(1 if loop_like_step_count else 0, loop_like_step_count)

    return RunTrace(
        answer=answer,
        tool_events=tool_events,
        num_loops=num_loops,
        num_tool_calls=tool_calls,
        num_tool_errors=tool_errors,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=total_latency if total_latency > 0 else None,
    )


def crossover(a: Candidate, b: Candidate, module_specs: Dict[str, ModuleSpec], global_cap: int) -> Candidate:
    new_modules = dict(a.prompt_modules)
    # randomly swap some modules from b
    for mid in MODULES:
        if mid in b.prompt_modules and random.random() < 0.5:
            new_modules[mid] = b.prompt_modules[mid]
    return Candidate(
        model=a.model,
        prompt_modules=new_modules,
        module_specs=module_specs,
        global_token_cap=global_cap,
        baseline_prompt_hash=a.baseline_prompt_hash,
        parent_id=a.candidate_id,
    )

def apply_single_module_edit(parent: Candidate, module_name: str, new_text: str) -> Candidate:
    pm = dict(parent.prompt_modules)
    pm[module_name] = new_text
    return Candidate(
        model=parent.model,
        prompt_modules=pm,
        module_specs=parent.module_specs,
        global_token_cap=parent.global_token_cap,
        baseline_prompt_hash=parent.baseline_prompt_hash,
        parent_id=parent.candidate_id,
    )