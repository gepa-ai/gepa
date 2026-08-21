from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, NotRequired, TypedDict

from gepa.core.adapter import EvaluationBatch
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.focused_evalset import (
    DEFAULT_BUCKET_TYPE,
    ensure_focused_eval_set,
)
from glean_gepa.prompt import compile_encoded_prompt
from glean_gepa.shell_tool_error_util import (
    HIGH_SIGNAL_VERIFY_OBJECTIVE,
    SHELL_SUCCESS_OBJECTIVE,
    EvalRunShellToolErrorAnalysis,
    fetch_eval_run_shell_tool_error_analysis,
    shell_error_free_rate,
)

JudgingMode = Literal["teacher_student", "single_model"]
TEACHER_STUDENT_PRIMARY_OBJECTIVE = "correctness"

PRIMARY_OBJECTIVE_BY_MODE: dict[JudgingMode, str] = {
    "single_model": SHELL_SUCCESS_OBJECTIVE,
    "teacher_student": TEACHER_STUDENT_PRIMARY_OBJECTIVE,
}

DEFAULT_FRONTIER_TYPE_BY_MODE: dict[JudgingMode, str] = {
    "single_model": "objective",
    "teacher_student": "hybrid",
}


def get_screening_score(eval_batch: GleanEvaluationBatch, judging_mode: JudgingMode) -> float:
    """Return the primary score used to screen offspring and compare parent/child."""
    if eval_batch.summary is None:
        return float("-inf")
    if judging_mode == "single_model":
        high_signal_count = int(eval_batch.summary.get("high_signal_entry_count", 0))
        verify_rate = eval_batch.summary.get(HIGH_SIGNAL_VERIFY_OBJECTIVE)
        if high_signal_count > 0 and verify_rate is not None:
            return float(verify_rate)
    objective = PRIMARY_OBJECTIVE_BY_MODE[judging_mode]
    return eval_batch.summary.get(objective, float("-inf"))


def log_shell_tool_error_analysis(analysis: EvalRunShellToolErrorAnalysis) -> None:
    """Log the fetched shell-tool error rate and recent error details."""
    aggregate = analysis.aggregate
    print(
        f"[Shell Tool] Fetched error rate for eval {analysis.eval_id}: "
        f"{aggregate.shell_error_pct:.2f}% "
        f"({aggregate.shell_errors}/{aggregate.shell_executions})"
    )
    for example in aggregate.recent_error_examples:
        if example.error_str:
            print(f"[Shell Tool] Error for eval {analysis.eval_id}: {example.error_str}")


# ---------------------------
# 1) Prompt modules + candidate
# ---------------------------

MODULES = [
    "WRITING_CODE",
]

@dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    kind: str                  # "free_text" | "enum_knob"
    token_budget: int

@dataclass
class Candidate:
    model: str                 # "claude" | "gemini" etc
    prompt_modules: dict[str, str]  # Single editable key: {"WRITING_CODE": "..."}
    module_specs: dict[str, ModuleSpec]
    global_token_cap: int      # relative to baseline prompt for that model
    baseline_prompt_hash: str  # used to define "relative cap"

    # bookkeeping for GEPA loop
    parent_id: str | None = None
    candidate_id: str = field(default_factory=lambda: hashlib.md5(str(random.random()).encode()).hexdigest()[:10])

def approx_token_len(text: str) -> int:
    # Replace with your tokenizer (tiktoken/cl100k, etc). Keep fast for screening.
    return max(1, len(text) // 4)

def total_prompt_tokens(candidate: Candidate) -> int:
    return sum(approx_token_len(candidate.prompt_modules.get(m, "")) for m in candidate.prompt_modules)

def within_prompt_budget(candidate: Candidate) -> bool:
    return total_prompt_tokens(candidate) <= candidate.global_token_cap and all(
        approx_token_len(candidate.prompt_modules.get(mid, "")) <= candidate.module_specs[mid].token_budget
        for mid in candidate.module_specs
        if mid in candidate.prompt_modules
    )

# ---------------------------
# 2) Eval records & metrics
# ---------------------------
class ALDataInst(TypedDict):
    eval_set_name: str
    eval_set_version: str
    deployment_ids: list[str]
    status: str

class ALRolloutOutput(TypedDict):
    deployment_id: str
    query: str
    # Student execution results
    student_answer: str
    student_tool_events: list[str]
    student_loops: int
    student_tool_calls: int
    student_tool_errors: int
    student_input_tokens: int
    student_output_tokens: int
    student_latency_ms: int | None

    # Teacher execution results (for comparison)
    teacher_answer: str
    teacher_tool_events: list[str]
    teacher_loops: int
    teacher_tool_calls: int
    teacher_input_tokens: int
    teacher_output_tokens: int

    # Entry metadata
    entry_id: str
    shell_error_messages: NotRequired[list[str]]
    student_eval_run_id: NotRequired[str]

class ALTrajectory(TypedDict):
    # Input data
    data: ALDataInst
    # Full execution output
    output: ALRolloutOutput
    # Score
    score: float
    # Objective scores
    objective_scores: dict[str, float]

class TraceInfo(TypedDict):
    """Trace information for a single evaluation run.

    Fields:
        eval_id: Evaluation run ID
        trace_id: Trace ID for retrieving detailed trace
        finish_time_millis: Timestamp when the run finished (in milliseconds)
        deployment_id: Deployment ID
        correctness_score: Correctness score for this run
        spans: Detailed trace spans (optional, fetched separately)

        # Execution details (parsed from metadata)
        query: str
        answer: str
        tool_events: List[Dict[str, Any]]  # Serialized ToolEvent objects
        num_loops: int
        num_tool_calls: int
        num_tool_errors: int
        input_tokens: int
        output_tokens: int
        latency_ms: NotRequired[int]
    """
    eval_id: str
    trace_id: str
    finish_time_millis: int
    deployment_id: str
    correctness_score: float
    spans: NotRequired[list[dict[str, Any]]]

    # Execution details
    query: str
    answer: str
    tool_events: list[dict[str, Any]]
    num_loops: int
    num_tool_calls: int
    num_tool_errors: int
    input_tokens: int
    output_tokens: int
    latency_ms: NotRequired[int]

class ReflectiveExampleInputs(TypedDict):
    """Input information for a reflective example."""
    eval_set: str
    entry_id: str
    deployment_id: str
    query: str

class ReflectiveExampleOutputs(TypedDict):
    """Generated outputs for a reflective example."""
    student_answer: str
    teacher_answer: str
    student_tools: list[str]
    teacher_tools: list[str]

class ReflectiveExampleMetrics(TypedDict):
    """Metrics for a reflective example."""
    score: float
    shell_success_rate: NotRequired[float]
    correctness: NotRequired[float]

# TypedDict with keys containing spaces must use functional form
ReflectiveExample = TypedDict("ReflectiveExample", {
    "Inputs": ReflectiveExampleInputs,
    "Generated Outputs": ReflectiveExampleOutputs,
    "Feedback": str,
    "Metrics": ReflectiveExampleMetrics,
})

@dataclass
class JudgeResult:
    correctness: float         # 0..1
    tool_alignment: float
    grounding: float
    rationale: str
    traces: dict[str, list[TraceInfo]] | None = None  # Maps entry_id -> list of trace_info dicts

# ---------------------------
# 3) Teacher cache + runner interfaces
# ---------------------------

class ALRunner:
    """
    Triggers eval runs and manages eval run IDs for the judge.

    Uses evalcli to create eval runs and poll until completion.
    All execution data (answers, tool events, tokens, etc.) is retrieved later by the Judge
    via evalcli analyze commands.
    """
    def __init__(
        self,
        evalcli: EvalCliClient,
        deployment_ids: list[str] | None = None,
        cache_file: str | None = None,
    ):
        self.evalcli = evalcli
        self.deployment_ids = deployment_ids or ["scio-prod"]
        self.cache_file = os.path.expanduser(cache_file) if cache_file else None

        # Track eval run IDs: cache_key -> eval_run_id
        self._eval_run_ids: dict[tuple[str, str, str, str, str], str] = {}

        # Load cached eval run IDs if cache file exists
        if self.cache_file:
            self._load_cache()

    def _load_cache(self) -> None:
        """Load eval run IDs cache from file."""
        if not self.cache_file:
            return

        try:
            import os
            if os.path.exists(self.cache_file):
                with open(self.cache_file) as f:
                    data = json.load(f)
                    # Convert string keys back to tuples
                    self._eval_run_ids = {
                        tuple(json.loads(k)): v
                        for k, v in data.items()
                    }
                    print(f"Loaded {len(self._eval_run_ids)} eval run IDs from cache")
        except Exception as e:
            print(f"Failed to load cache from {self.cache_file}: {e}")
            self._eval_run_ids = {}

    def _save_cache(self) -> None:
        """Save eval run IDs cache to file."""
        if not self.cache_file:
            return

        try:
            import os
            # Create directory if it doesn't exist
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            # Convert tuple keys to strings for JSON serialization
            data = {
                json.dumps(list(k)): v
                for k, v in self._eval_run_ids.items()
            }

            with open(self.cache_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(self._eval_run_ids)} eval run IDs to cache")
        except Exception as e:
            print(f"Failed to save cache to {self.cache_file}: {e}")

    def _build_sc_params(self, model: str, system_prompt: str) -> str:
        """Build scParams based on model type."""
        # Base configuration shared by both models
        base_params = [
            "ro.scholastic_required=true",
            "db.disable_usr=true",
            "db.filter_query_debug_results=true",
            "db.filter_bad_query_jiras=true",
            "db.include_final_scores=true",
            "ro.ro.fetch_supplemental_results=false",
            "db.ranking_only=true",
            "db.debug_mode=1",
            "co.debug_only_disabled_tools_list=gmail_search;outlook_search;respond;think",
            "co.use_eval_cache_for_llm=true",
            "db.get_doc_metadata=true",
            "co.lo.fail_on_tool_failure=1",
            "co.lo.enable_agent_recommendation=false",
            "ro.feso.slso.drop_slack_native=false",
            "ro.feso.slso.rts_count=0",
            "ro.feso.slso.skip_inline_rts=true",
            "co.py_agent_route_override=o3_agentic_loop",
            "co.web_search_preference_order=bravewebsearch;openaiwebsearch;googlegeminiwebsearch",
            "co.disable_full_document_content=true",
            "ro.enable_code_file_matches=1",
            "db.pyagents_use_qe_for_responses_api=1",
            "co.lo.reasoning_effort=medium",
            "wo.plan_only_dry_run_for_write_actions=true",
            "wo.plan_only_dry_run_for_write_actions_in_actas=true",
        ]

        # Model-specific configuration
        if model == "claude":
            base_params.append("co.lo.oai_model_for_agentic_loop=CLAUDE_4_5_SONNET_20250929")
        elif model != "gpt" and model != "fast":
            raise ValueError(f"Unknown model: {model}")

        # Add system prompt override if provided (and not the placeholder)
        if system_prompt and system_prompt != "<<TEACHER_PROD_PROMPT>>":
            # system_prompt should already be the compiled sc parameter from compile_system_prompt
            base_params.append(system_prompt)

        return ",".join(base_params)

    def run(
        self,
        model: str,
        system_prompt: str,
        eval_set_name: str,
        eval_set_version: str,
        deployment_ids: list[str],
        run_label: str = "gepa",
    ) -> str:
        """
        Trigger an eval run and return the eval_run_id.

        Args:
            model: "claude" or "gpt"
            system_prompt: Compiled system prompt (sc parameter string from compile_system_prompt)
            eval_set_name: Name of the eval set
            eval_set_version: Version of the eval set
            deployment_ids: List of deployment IDs to use
            run_label: Prefix for eval run id / cache key (e.g. gepa vs verify_<hash>)

        Returns:
            eval_run_id string
        """
        system_prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:16]
        cache_key = (model, system_prompt_hash, eval_set_name, eval_set_version, run_label)

        if cache_key in self._eval_run_ids:
            return self._eval_run_ids[cache_key]

        eval_id = f"{run_label}_{model}_{system_prompt_hash}_{int(time.time())}"

        sc_params = self._build_sc_params(model, system_prompt)
        eval_params = "experimental_queue=temp-system-prompt-optimization"
        if model == "fast":
            eval_params += ",gleanchat_agent=FAST"
        else:
            eval_params += ",gleanchat_agent=ADVANCED"

        print(f"Creating eval run {eval_id} for {eval_set_name}:{eval_set_version}...")
        created_id = self.evalcli.create_eval_run(
            eval_run_id=eval_id,
            eval_set_name=eval_set_name,
            eval_set_version=eval_set_version,
            deployment_ids=deployment_ids,
            description=f"GEPA eval run for {eval_set_name}:{eval_set_version}",
            sc_params=sc_params,
            eval_params=eval_params,
        )
        self.evalcli.wait_for_eval_run(created_id)
        print(f"Eval run {created_id} completed successfully")

        self._eval_run_ids[cache_key] = created_id
        self._save_cache()

        return created_id


    def get_eval_run_id(
        self,
        model: str,
        system_prompt: str,
        eval_set_name: str,
        eval_set_version: str,
        run_label: str = "gepa",
    ) -> str | None:
        """Get the eval run ID for a given cache key."""
        system_prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:16]
        cache_key = (model, system_prompt_hash, eval_set_name, eval_set_version, run_label)
        return self._eval_run_ids.get(cache_key)

class Judge:
    """
    LLM judge that compares teacher vs student using evalcli.

    Flow:
        1. Create judge run via evalcli judge create
        2. Poll until judge run SUCCEEDED
        3. Fetch analysis view/details/trace via evalcli analyze
    """
    def __init__(self, evalcli: EvalCliClient):
        self.evalcli = evalcli
        # Cache judge results: (teacher_eval_id, student_eval_id) -> JudgeResult
        self._judge_cache: dict[tuple[str, str], JudgeResult] = {}

    def judge(
        self,
        teacher_eval_id: str,
        student_eval_id: str,
        skip_trigger: bool = False,
    ) -> JudgeResult:
        """
        Run LLM judge comparison between teacher and student.

        Args:
            teacher_eval_id: Teacher eval run ID
            student_eval_id: Student eval run ID
            skip_trigger: If True, skip triggering the judge and go straight to fetching results
        """
        cache_key = (teacher_eval_id, student_eval_id)
        if cache_key in self._judge_cache:
            print(f"Using cached judge result for {teacher_eval_id} vs {student_eval_id}")
            return self._judge_cache[cache_key]

        if not skip_trigger:
            print(f"Triggering judge run for {teacher_eval_id} vs {student_eval_id}...")
            judge_run_id = self.evalcli.create_judge_run(
                student_eval_id=student_eval_id,
                teacher_eval_id=teacher_eval_id,
            )
            self.evalcli.wait_for_judge_run(judge_run_id)
        else:
            print(f"Skipping judge trigger (already triggered), fetching results for {teacher_eval_id} vs {student_eval_id}.")

        judge_result = self._get_full_judge_results(student_eval_id, teacher_eval_id)

        self._judge_cache[cache_key] = judge_result
        return judge_result

    def _get_full_judge_results(self, student_eval_id: str, teacher_eval_id: str) -> JudgeResult:
        result_data = self.evalcli.get_analysis_view(student_eval_id, teacher_eval_id)

        entries = result_data.get("entries", [])

        deployments = []
        entry_details = []
        durations_map = {}
        loop_counts_map = {}
        input_tokens_map = {}
        output_tokens_map = {}
        tools_invocations_map = {}
        for entry in entries:
            has_error = False
            for eval_run_entry in entry.get("evalRunEntries", []):
                if eval_run_entry.get("errorMessage"):
                    has_error = True
                    break
            if has_error:
                continue
            entry_id = entry.get("entryId")
            deployment_id = entry.get("deploymentId")
            deployments.append(deployment_id)
            entry_details.append({
                "deploymentId": deployment_id,
                "entryId": entry_id,
                "evalRunIds": [student_eval_id, teacher_eval_id]
            })
            for eval_run_entry in entry.get("evalRunEntries", []):
                eval_run_id = eval_run_entry.get("evalRunId")
                durations_map[(entry_id, eval_run_id)] = eval_run_entry.get("duration", 0)
                loop_counts_map[(entry_id, eval_run_id)] = eval_run_entry.get("metadata", {}).get("loopCount", 0)
                input_tokens_map[(entry_id, eval_run_id)] = eval_run_entry.get("metadata", {}).get("uncachedInputTokens", 0)
                output_tokens_map[(entry_id, eval_run_id)] = eval_run_entry.get("metadata", {}).get("outputTokens", 0)
                tools_invocations_map[(entry_id, eval_run_id)] = eval_run_entry.get("metadata", {}).get("toolsInvoked", [])


        # Get detailed run information including traces, grouped by deployment
        trace_map: defaultdict[str, list[TraceInfo]] = defaultdict(list)
        details_by_deployment: dict[str, list[str]] = defaultdict(list)
        for entry_detail in entry_details:
            details_by_deployment[entry_detail["deploymentId"]].append(entry_detail["entryId"])

        details_data: list[dict[str, Any]] = []
        for deployment_id, entry_ids in details_by_deployment.items():
            details_data.extend(
                self.evalcli.get_analysis_details(
                    entry_ids=entry_ids,
                    eval_run_ids=[student_eval_id, teacher_eval_id],
                    deployment_id=deployment_id,
                )
            )

        # Extract trace information for each (entryId, evalId) pair
        for item in details_data:
            if item.get("error"):
                print(f"Error in details data: {item.get('error')}")
                continue
            entry_id = item.get("evalSetEntry", {}).get("id")
            deployment_id = item.get("evalSetEntry", {}).get("deploymentId")
            run_responses = _normalize_run_responses(item.get("runResponses"))
            trace_infos: list[TraceInfo] = []

            correctness_scores = {}
            for judge_entry in item.get("judgeRunEntries") or []:
                if judge_entry.get("errorMessage"):
                    print(f"Error in judge entry: {judge_entry.get('errorMessage')}")
                    continue
                outputs = judge_entry.get("outputs", [])
                eval_id = judge_entry.get("evalRunId")
                if outputs is None:
                    continue
                for output in outputs:
                    if output.get("name") == "CORRECTNESS":
                        correctness_scores[(entry_id, eval_id)] = output.get("score", 0)
            for run_response in run_responses:
                if not run_response.get("output"):
                    print(f"Error in run response: {run_response.get('errorMessage')}")
                    continue
                eval_id = run_response.get("runId")
                trace_id = run_response.get("outputTrace", {}).get("id", "")
                metadata = run_response.get("metadata", {})
                finish_time_ms = metadata.get("finishTimeMillis", "")
                answer = run_response.get("output").get("chatResponseInfo", {}).get("actResponse", "")
                query = item.get("evalSetEntry", {}).get("input", {}).get("query")
                print(f"Got query: {query}")

                if entry_id and eval_id and trace_id and finish_time_ms:
                    trace_info: TraceInfo = {
                        "eval_id": eval_id,
                        "trace_id": trace_id,
                        "finish_time_millis": finish_time_ms,
                        "deployment_id": deployment_id,
                        # Execution details
                        "query": query,
                        "answer": answer,
                        "tool_events": tools_invocations_map.get((entry_id, eval_id), []),
                        "num_loops": loop_counts_map.get((entry_id, eval_id), 0),
                        "num_tool_calls": len(tools_invocations_map.get((entry_id, eval_id), [])),
                        "num_tool_errors": 0,  # Not provided in metadata
                        "input_tokens": input_tokens_map.get((entry_id, eval_id), 0),
                        "output_tokens": output_tokens_map.get((entry_id, eval_id), 0),
                    }
                    duration = durations_map.get((entry_id, eval_id), 0)
                    if duration > 0:
                        trace_info["latency_ms"] = duration
                    if eval_id == student_eval_id and correctness_scores.get((entry_id, eval_id)) is not None:
                        trace_info["correctness_score"] = correctness_scores.get((entry_id, eval_id))
                        trace_infos.append(trace_info)
                    elif eval_id == teacher_eval_id:
                        trace_infos.append(trace_info)

            trace_map[entry_id] = trace_infos

        # Fetch detailed trace for each trace ID
        for _entry_id, trace_infos in trace_map.items():
            for trace_info in trace_infos:
                eval_id = trace_info.get("eval_id")
                trace_id = trace_info.get("trace_id")
                finish_time_ms = trace_info.get("finish_time_millis")
                deployment_id = trace_info.get("deployment_id")

                if trace_id and finish_time_ms and deployment_id:
                    start_time_ms = finish_time_ms - 3600000
                    end_time_ms = finish_time_ms

                    detailed_trace = self.evalcli.get_analysis_trace(
                        deployment_id=deployment_id,
                        trace_id=trace_id,
                        start_time_millis=start_time_ms,
                        end_time_millis=end_time_ms,
                    )

                    trace_info["spans"] = detailed_trace.get("trace", {}).get("spans")

        # Average correctness across all entries
        correctness_score_list = []
        for _entry_id, trace_infos in trace_map.items():
            for trace_info in trace_infos:
                if trace_info.get("eval_id") == student_eval_id:
                    curr_score = 0
                    if trace_info.get("correctness_score"):
                        curr_score = trace_info.get("correctness_score")
                    else:
                        print(f'Get a none correctness score for trace {trace_info.get("eval_id")}')
                    correctness_score_list.append(curr_score)
        correctness = sum(correctness_score_list) / len(correctness_score_list) if correctness_score_list else 0.0
        tool_alignment = get_tool_alignment(trace_map, student_eval_id, teacher_eval_id)

        # Use correctness as proxy for other metrics
        grounding = correctness
        rationale = f"Correctness: {correctness:.2f} (avg of {len(correctness_score_list)} entries)"

        return JudgeResult(
            correctness=correctness,
            grounding=grounding,
            tool_alignment=tool_alignment,
            rationale=rationale,
            traces=dict(trace_map),
        )

def _normalize_run_responses(run_responses: Any) -> list[dict[str, Any]]:
    if isinstance(run_responses, dict):
        return [response for response in run_responses.values() if isinstance(response, dict)]
    if isinstance(run_responses, list):
        return [response for response in run_responses if isinstance(response, dict)]
    return []

def get_tool_alignment(trace_map: dict[str, list[TraceInfo]], student_eval_id: str, teacher_eval_id: str) -> float:
    tool_alignment_scores = []
    for _entry_id, trace_infos in trace_map.items():
        student_run = None
        teacher_run = None
        for trace_info in trace_infos:
            eval_id = trace_info.get("eval_id")
            if eval_id == student_eval_id:
                student_run = trace_info["spans"]
            elif eval_id == teacher_eval_id:
                teacher_run = trace_info["spans"]
        if not student_run or not teacher_run:
            return 0
        tool_alignment_scores.append(get_tool_alignment_from_traces(student_run, teacher_run))
    return sum(tool_alignment_scores) / len(tool_alignment_scores) if tool_alignment_scores else 0.0


# TODO(Cathy) Get cleaner traces than what the api current returns
def get_tool_alignment_from_traces(student_trace_spans: list[dict[str, Any]], teacher_trace_spans: list[dict[str, Any]]) -> float:
    student_tool_usages = []
    teacher_tool_usages = []
    for span in student_trace_spans:
        if "Execute Action" in span["name"]:
            tool_usage = parse_tool_usage(span)
            if not tool_usage:
                continue
            student_tool_usages.append(tool_usage)
    for span in teacher_trace_spans:
        if "Execute Action" in span["name"]:
            tool_usage = parse_tool_usage(span)
            if not tool_usage:
                continue
            teacher_tool_usages.append(tool_usage)
    # print(f"Student tool usages: {student_tool_usages}")
    # print(f"Teacher tool usages: {teacher_tool_usages}")
    final_tool_alignment_score = 0
    for i in range(len(student_tool_usages)):
        if i >= len(teacher_tool_usages):
            final_tool_alignment_score -= 0.1 * (len(student_tool_usages) - len(teacher_tool_usages))
            break
        if student_tool_usages[i][0] == teacher_tool_usages[i][0]:
            final_tool_alignment_score += 0.2
        else:
            final_tool_alignment_score -= 0.1
    return final_tool_alignment_score

def process_raw_typed_value(raw: str | dict):
    # Parse the outer Python-literal string
    if isinstance(raw, dict):
        outer = raw
    else:
        outer = ast.literal_eval(raw)

    # Handle non-string typed values
    if outer.get("intValue") is not None:
        return outer["intValue"]
    if outer.get("boolValue") is not None:
        return outer["boolValue"]

    str_value = outer.get("strValue")
    if str_value is None:
        print("No strValue")
        return None

    # Try parsing strValue directly as JSON
    parsed = json.loads(str_value)

    # If parsed["input"] is itself a JSON string, parse it too
    if isinstance(parsed, dict) and isinstance(parsed.get("input"), str):
        try:
            parsed_input = json.loads(parsed["input"])
            return parsed_input
        except (TypeError, ValueError, json.JSONDecodeError):
            print(f"Could not parse input: {parsed['input']}")
            return parsed

    return parsed

def extract_tool_names_from_spans(spans: list[dict[str, Any]] | None) -> list[str]:
    """Extract tool names from trace spans."""
    if not spans:
        return []
    names = []
    for span in spans:
        if "Execute Action" in span.get("name", ""):
            parts = span["name"].split(": ")
            if len(parts) >= 2 and parts[1] != "Personal Knowledge Vault Retrieve":
                names.append(parts[1])
    return names


def parse_tool_usage(span: dict) -> tuple[str, dict] | None:
    parts = span["name"].split(": ")
    if len(parts) < 2:
        print(f"Invalid span name: {span['name']}")
        return None
    tool_name = span["name"].split(": ")[1]
    tool_inputs = span["attributes"]["input"]
    if tool_name == "Personal Knowledge Vault Retrieve":
        return None
    try:
        tool_inputs = process_raw_typed_value(tool_inputs)
        del tool_inputs["id"]
        return (tool_name, tool_inputs)
    except (KeyError, TypeError, ValueError, SyntaxError, json.JSONDecodeError):
        return None

# ---------------------------
# 4) AssistantALAdapter (core)
# ---------------------------

@dataclass
class Thresholds:
    quality_min: float
    tools_min: float
    max_student_tokens: int

class AssistantALAdapter:
    def __init__(
        self,
        runner: ALRunner,
        teacher_model: str,
        thresholds: Thresholds,
        student_model: str,
        *,
        judging_mode: JudgingMode = "single_model",
        judge: Judge | None = None,
        bigquery_client: Any | None = None,
        shell_error_lookback_days: int = 7,
        focused_eval_set_bucket_type: str = DEFAULT_BUCKET_TYPE,
        cache_file: str | None = None,
    ):
        if judging_mode == "teacher_student" and judge is None:
            raise ValueError("judge is required for teacher_student judging mode")
        if judging_mode == "single_model" and bigquery_client is None:
            raise ValueError("bigquery_client is required for single_model judging mode")

        self.runner = runner
        self.judging_mode = judging_mode
        self.judge = judge
        self.bigquery_client = bigquery_client
        self.shell_error_lookback_days = shell_error_lookback_days
        self.focused_eval_set_bucket_type = focused_eval_set_bucket_type
        self.teacher_model = teacher_model
        self.thresholds = thresholds
        self.student_model = student_model
        self.cache_file = os.path.expanduser(cache_file) if cache_file else None

        # module freezing memory: module -> count of "not relevant" in consecutive generations
        self._module_irrelevant_streak: dict[tuple[str, str], int] = defaultdict(int)  # (candidate_family, module) -> streak

        # good options pool per module (to avoid losing good parts)
        # TODO(Cathy) populate the good_module_options
        self.good_module_options: dict[str, list[str]] = defaultdict(list)

        # Eval ID cache: (eval_set_name, eval_set_version, model, prompt_hash) -> eval_id
        self._eval_cache: dict[tuple[str, str, str, str], str] = {}

        # Judge triggered cache: (teacher_eval_id, student_eval_id) -> triggered
        self._judge_triggered: set[tuple[str, str]] = set()

        # Load cache if file exists
        if self.cache_file:
            self._load_cache()

    def _load_cache(self) -> None:
        """Load eval ID cache and judge triggered cache from file."""
        if not self.cache_file:
            return

        try:
            import os
            if os.path.exists(self.cache_file):
                with open(self.cache_file) as f:
                    data = json.load(f)

                    # Load eval cache
                    eval_cache_data = data.get("eval_cache", {})
                    self._eval_cache = {
                        tuple(json.loads(k)): v
                        for k, v in eval_cache_data.items()
                    }

                    # Load judge triggered cache
                    judge_triggered_data = data.get("judge_triggered", [])
                    self._judge_triggered = {
                        tuple(item) for item in judge_triggered_data
                    }

                    print(f"[AssistantALAdapter] Loaded {len(self._eval_cache)} eval IDs and {len(self._judge_triggered)} judge triggers from cache: {self.cache_file}")
        except Exception as e:
            print(f"[AssistantALAdapter] Failed to load cache from {self.cache_file}: {e}")
            self._eval_cache = {}
            self._judge_triggered = set()

    def _save_cache(self) -> None:
        """Save eval ID cache and judge triggered cache to file."""
        if not self.cache_file:
            return

        try:
            import os
            # Create directory if it doesn't exist
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            # Build cache data structure
            data = {
                "eval_cache": {
                    json.dumps(list(k)): v
                    for k, v in self._eval_cache.items()
                },
                "judge_triggered": [
                    list(pair) for pair in self._judge_triggered
                ]
            }

            with open(self.cache_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[AssistantALAdapter] Saved {len(self._eval_cache)} eval IDs and {len(self._judge_triggered)} judge triggers to cache: {self.cache_file}")
        except Exception as e:
            print(f"[AssistantALAdapter] Failed to save cache to {self.cache_file}: {e}")

    def _extract_per_entry_metrics(
        self,
        judge_result: JudgeResult,
        student_eval_id: str,
        teacher_eval_id: str
    ) -> dict[str, dict[str, float]]:
        """Extract per-entry metrics from judge result traces.

        Returns:
            Dict mapping entry_id -> {correctness, tool_alignment, grounding}
        """
        per_entry_metrics = {}

        if not judge_result.traces:
            return per_entry_metrics

        # Process each entry's traces
        for entry_id, trace_infos in judge_result.traces.items():
            student_trace_info = None
            teacher_trace_info = None

            # Find student and teacher trace infos for this entry
            for trace_info in trace_infos:
                eval_id = trace_info.get("eval_id")
                if eval_id == student_eval_id:
                    student_trace_info = trace_info
                elif eval_id == teacher_eval_id:
                    teacher_trace_info = trace_info

            if not student_trace_info:
                continue

            # Extract correctness from student trace
            correctness = student_trace_info.get("correctness_score", 0.0)

            # Compute tool alignment from spans if available
            tool_alignment = get_tool_alignment_from_traces(
                student_trace_info["spans"],
                teacher_trace_info["spans"]
            )

            # Use correctness as proxy for grounding (could be improved with actual grounding metric)
            grounding = correctness

            per_entry_metrics[entry_id] = {
                "correctness": correctness,
                "tool_alignment": tool_alignment,
                "grounding": grounding
            }

        return per_entry_metrics

    def evaluate(
        self,
        batch: list[ALDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate candidate on eval set(s)."""
        if self.judging_mode == "single_model":
            return self._evaluate_with_shell_error_rate(batch, candidate, capture_traces)
        return self._evaluate_with_judge(batch, candidate, capture_traces)

    def _get_or_run_student_eval(
        self,
        *,
        eval_set_name: str,
        eval_set_version: str,
        deployment_ids: list[str],
        system_prompt: str,
        run_label: str = "gepa",
    ) -> str:
        student_prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:16]
        student_cache_key = (
            eval_set_name,
            eval_set_version,
            self.student_model,
            student_prompt_hash,
            run_label,
        )

        student_eval_id = self._eval_cache.get(student_cache_key)
        if student_eval_id:
            print(f"[Cache HIT] Using cached student eval_id: {student_eval_id} ({run_label})")
            return student_eval_id

        student_eval_id = self.runner.run(
            self.student_model,
            system_prompt=system_prompt,
            eval_set_name=eval_set_name,
            eval_set_version=eval_set_version,
            deployment_ids=deployment_ids,
            run_label=run_label,
        )
        self._eval_cache[student_cache_key] = student_eval_id
        self._save_cache()
        print(f"[Cache MISS] Started and cached student eval_id: {student_eval_id} ({run_label})")
        return student_eval_id

    def _verify_high_signal_entries(
        self,
        *,
        eval_set_name: str,
        eval_set_version: str,
        deployment_ids: list[str],
        system_prompt: str,
        entry_ids: Sequence[str],
    ) -> float:
        """Re-run only the high-signal entries and report how many come back error-free.

        The subset is uploaded as its own small eval set, so the re-run costs a handful of
        entries instead of the whole eval set.
        """
        focused = ensure_focused_eval_set(
            self.runner.evalcli,
            base_eval_set_name=eval_set_name,
            base_eval_set_version=eval_set_version,
            deployment_ids=deployment_ids,
            entry_ids=entry_ids,
            bucket_type=self.focused_eval_set_bucket_type,
        )
        if focused is None:
            print("[Verify] Skipping high-signal verification: no focused eval set available")
            return 1.0

        verify_eval_id = self._get_or_run_student_eval(
            eval_set_name=focused.name,
            eval_set_version=focused.version,
            deployment_ids=deployment_ids,
            system_prompt=system_prompt,
            run_label="verify",
        )
        verify_analysis = fetch_eval_run_shell_tool_error_analysis(
            self.bigquery_client,
            eval_id=verify_eval_id,
            lookback_days=self.shell_error_lookback_days,
        )
        log_shell_tool_error_analysis(verify_analysis)
        pass_rate = shell_error_free_rate(verify_analysis.per_entry)
        print(
            f"[Verify] {focused.name}:{focused.version} run {verify_eval_id} "
            f"error-free rate {pass_rate:.2f} over {len(verify_analysis.per_entry)} entries"
        )
        return pass_rate

    def _evaluate_with_shell_error_rate(
        self,
        batch: list[ALDataInst],
        candidate: dict[str, str],
        capture_traces: bool,
    ) -> EvaluationBatch:
        if not batch:
            return GleanEvaluationBatch(
                outputs=[],
                scores=[],
                trajectories=None,
                objective_scores=[],
                summary=None,
            )

        system_prompt = compile_encoded_prompt(candidate)
        all_outputs: list[ALRolloutOutput] = []
        all_scores: list[float] = []
        all_trajectories: list[ALTrajectory] = [] if capture_traces else None
        all_objective_scores: list[dict[str, float]] = []
        summary_shell_rates: list[float] = []
        summary_verify_rates: list[float] = []
        total_high_signal_entries = 0

        for al_data_inst in batch:
            eval_set_version = al_data_inst.get("eval_set_version", "")
            eval_set_name = al_data_inst.get("eval_set_name", "")
            deployment_ids = al_data_inst.get("deployment_ids", [])

            student_eval_id = self._get_or_run_student_eval(
                eval_set_name=eval_set_name,
                eval_set_version=eval_set_version,
                deployment_ids=deployment_ids,
                system_prompt=system_prompt,
            )

            analysis = fetch_eval_run_shell_tool_error_analysis(
                self.bigquery_client,
                eval_id=student_eval_id,
                lookback_days=self.shell_error_lookback_days,
            )
            log_shell_tool_error_analysis(analysis)
            high_signal_entry_ids = analysis.high_signal_entry_ids
            verify_pass_rate = 1.0
            if high_signal_entry_ids:
                verify_pass_rate = self._verify_high_signal_entries(
                    eval_set_name=eval_set_name,
                    eval_set_version=eval_set_version,
                    deployment_ids=deployment_ids,
                    system_prompt=system_prompt,
                    entry_ids=high_signal_entry_ids,
                )

            summary_shell_rates.append(analysis.aggregate.shell_success_rate)
            summary_verify_rates.append(verify_pass_rate)
            total_high_signal_entries += len(high_signal_entry_ids)

            if not high_signal_entry_ids:
                shell_error_messages = [
                    example.error_str
                    for example in analysis.aggregate.recent_error_examples
                    if example.error_str
                ]
                output: ALRolloutOutput = {
                    "deployment_id": deployment_ids[0] if deployment_ids else "",
                    "query": f"{eval_set_name}:{eval_set_version}",
                    "student_answer": "",
                    "student_tool_events": [],
                    "student_loops": 0,
                    "student_tool_calls": analysis.aggregate.shell_executions,
                    "student_tool_errors": analysis.aggregate.shell_errors,
                    "student_input_tokens": 0,
                    "student_output_tokens": 0,
                    "teacher_answer": "",
                    "teacher_tool_events": [],
                    "teacher_loops": 0,
                    "teacher_tool_calls": 0,
                    "teacher_input_tokens": 0,
                    "teacher_output_tokens": 0,
                    "entry_id": f"{eval_set_name}:{eval_set_version}",
                    "shell_error_messages": shell_error_messages,
                    "student_eval_run_id": student_eval_id,
                }
                all_outputs.append(output)
                all_scores.append(analysis.aggregate.shell_success_rate)
                objective_score = {
                    SHELL_SUCCESS_OBJECTIVE: analysis.aggregate.shell_success_rate,
                    HIGH_SIGNAL_VERIFY_OBJECTIVE: verify_pass_rate,
                }
                all_objective_scores.append(objective_score)
                if capture_traces and all_trajectories is not None:
                    all_trajectories.append(
                        {
                            "data": al_data_inst,
                            "output": output,
                            "score": analysis.aggregate.shell_success_rate,
                            "objective_scores": objective_score,
                        }
                    )
                continue

            for entry_id in high_signal_entry_ids:
                entry_metrics = analysis.per_entry[entry_id]
                shell_error_messages = [
                    example.error_str
                    for example in entry_metrics.recent_error_examples
                    if example.error_str
                ]
                entry_output: ALRolloutOutput = {
                    "deployment_id": deployment_ids[0] if deployment_ids else "",
                    "query": f"{eval_set_name}:{eval_set_version} entry={entry_id}",
                    "student_answer": "",
                    "student_tool_events": [],
                    "student_loops": 0,
                    "student_tool_calls": entry_metrics.shell_executions,
                    "student_tool_errors": entry_metrics.shell_errors,
                    "student_input_tokens": 0,
                    "student_output_tokens": 0,
                    "teacher_answer": "",
                    "teacher_tool_events": [],
                    "teacher_loops": 0,
                    "teacher_tool_calls": 0,
                    "teacher_input_tokens": 0,
                    "teacher_output_tokens": 0,
                    "entry_id": entry_id,
                    "shell_error_messages": shell_error_messages,
                    "student_eval_run_id": student_eval_id,
                }
                all_outputs.append(entry_output)
                entry_score = entry_metrics.shell_success_rate
                all_scores.append(entry_score)
                entry_objective_score = {
                    SHELL_SUCCESS_OBJECTIVE: entry_metrics.shell_success_rate,
                    HIGH_SIGNAL_VERIFY_OBJECTIVE: verify_pass_rate,
                }
                all_objective_scores.append(entry_objective_score)
                if capture_traces and all_trajectories is not None:
                    all_trajectories.append(
                        {
                            "data": al_data_inst,
                            "output": entry_output,
                            "score": entry_score,
                            "objective_scores": entry_objective_score,
                        }
                    )

        summary = None
        if summary_shell_rates:
            summary = {
                SHELL_SUCCESS_OBJECTIVE: sum(summary_shell_rates) / len(summary_shell_rates),
                HIGH_SIGNAL_VERIFY_OBJECTIVE: sum(summary_verify_rates) / len(summary_verify_rates),
                "high_signal_entry_count": float(total_high_signal_entries),
            }

        return GleanEvaluationBatch(
            outputs=all_outputs,
            scores=all_scores,
            trajectories=all_trajectories,
            objective_scores=all_objective_scores,
            summary=summary,
        )

    def _evaluate_with_judge(
        self,
        batch: list[ALDataInst],
        candidate: dict[str, str],
        capture_traces: bool,
    ) -> EvaluationBatch:
        if self.judge is None:
            raise RuntimeError("Judge is required for judge-based evaluation")

        # Handle empty batch
        if not batch:
            return GleanEvaluationBatch(
                outputs=[],
                scores=[],
                trajectories=None,
                objective_scores=[],
                summary=None
            )

        # Compile system prompt from candidate
        system_prompt = compile_encoded_prompt(candidate)

        # Collect results across all eval sets
        all_outputs: list[ALRolloutOutput] = []
        all_scores: list[float] = []
        all_trajectories: list[ALTrajectory] = [] if capture_traces else None
        all_objective_scores: list[dict[str, float]] = []

        for al_data_inst in batch:
            eval_set_version = al_data_inst.get("eval_set_version", "")
            eval_set_name = al_data_inst.get("eval_set_name", "")
            deployment_ids = al_data_inst.get("deployment_ids", [])

            # Create cache keys for teacher and student
            teacher_prompt_hash = hashlib.md5(b"<<TEACHER_PROD_PROMPT>>").hexdigest()[:16]
            student_prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:16]

            teacher_cache_key = (eval_set_name, eval_set_version, self.teacher_model, teacher_prompt_hash)
            student_cache_key = (eval_set_name, eval_set_version, self.student_model, student_prompt_hash)

            # Check cache for teacher eval
            teacher_eval_id = self._eval_cache.get(teacher_cache_key)
            if teacher_eval_id:
                print(f"[Cache HIT] Using cached teacher eval_id: {teacher_eval_id}")
            else:
                # Trigger teacher eval run
                teacher_eval_id = self.runner.run(
                    self.teacher_model,
                    system_prompt="<<TEACHER_PROD_PROMPT>>",
                    eval_set_name=eval_set_name,
                    eval_set_version=eval_set_version,
                    deployment_ids=deployment_ids
                )
                # Cache and save immediately
                self._eval_cache[teacher_cache_key] = teacher_eval_id
                self._save_cache()
                print(f"[Cache MISS] Started and cached teacher eval_id: {teacher_eval_id}")

            # Check cache for student eval
            student_eval_id = self._eval_cache.get(student_cache_key)
            if student_eval_id:
                print(f"[Cache HIT] Using cached student eval_id: {student_eval_id}")
            else:
                # Trigger student eval run
                student_eval_id = self.runner.run(
                    self.student_model,
                    system_prompt=system_prompt,
                    eval_set_name=eval_set_name,
                    eval_set_version=eval_set_version,
                    deployment_ids=deployment_ids
                )
                # Cache and save immediately
                self._eval_cache[student_cache_key] = student_eval_id
                self._save_cache()
                print(f"[Cache MISS] Started and cached student eval_id: {student_eval_id}")
            # teacher_eval_id = "gepa_gpt_3070257bbe5f1340_1774652253"
            # student_eval_id = "gepa_fast_1ad33e85e6067b04_1774652258"

            # Check if judge has been triggered for this pair
            judge_cache_key = (teacher_eval_id, student_eval_id)
            skip_trigger = judge_cache_key in self._judge_triggered

            if skip_trigger:
                print(f"[Judge Cache HIT] Judge already triggered for {teacher_eval_id} vs {student_eval_id}")
            else:
                print(f"[Judge Cache MISS] Will trigger judge for {teacher_eval_id} vs {student_eval_id}")
                # Mark as triggered and save immediately
                self._judge_triggered.add(judge_cache_key)
                self._save_cache()

            # Run judge to compare teacher vs student
            judge_result = self.judge.judge(teacher_eval_id, student_eval_id, skip_trigger=skip_trigger)

            # Build per-entry metrics map from judge traces
            per_entry_metrics = self._extract_per_entry_metrics(
                judge_result, student_eval_id, teacher_eval_id
            )

            # Process each entry in the eval set (from judge traces)
            if not judge_result.traces:
                continue

            for entry_id, trace_infos in judge_result.traces.items():
                # Find student and teacher traces for this entry
                student_trace = None
                teacher_trace = None
                for trace_info in trace_infos:
                    if trace_info["eval_id"] == student_eval_id:
                        student_trace = trace_info
                    elif trace_info["eval_id"] == teacher_eval_id:
                        teacher_trace = trace_info

                if student_trace is None or teacher_trace is None:
                    continue

                # Get per-entry metrics or fall back to aggregate
                entry_metrics = per_entry_metrics.get(entry_id, {
                    "correctness": judge_result.correctness,
                    "tool_alignment": judge_result.tool_alignment,
                    "grounding": judge_result.grounding
                })

                # Create comprehensive output with full execution details
                output: ALRolloutOutput = {
                    # Student execution
                    "deployment_id": student_trace["deployment_id"],
                    "query": student_trace["query"],
                    "student_answer": student_trace["answer"],
                    "student_tool_events": extract_tool_names_from_spans(student_trace.get("spans")),
                    "student_loops": student_trace["num_loops"],
                    "student_tool_calls": len(extract_tool_names_from_spans(student_trace.get("spans"))),
                    "student_tool_errors": student_trace["num_tool_errors"],
                    "student_input_tokens": student_trace["input_tokens"],
                    "student_output_tokens": student_trace["output_tokens"],
                    "student_latency_ms": student_trace.get("latency_ms"),

                    # Teacher execution
                    "teacher_answer": teacher_trace["answer"],
                    "teacher_tool_events": extract_tool_names_from_spans(teacher_trace.get("spans")),
                    "teacher_loops": teacher_trace["num_loops"],
                    "teacher_tool_calls": teacher_trace["num_tool_calls"],
                    "teacher_input_tokens": teacher_trace["input_tokens"],
                    "teacher_output_tokens": teacher_trace["output_tokens"],

                    # Metadata
                    "entry_id": entry_id,
                }
                all_outputs.append(output)

                # Create score (weighted combination of metrics)
                score = (
                    0.5 * entry_metrics["correctness"] +
                    0.3 * entry_metrics["tool_alignment"] +
                    0.2 * entry_metrics["grounding"]
                )
                all_scores.append(score)

                # Create objective scores for multi-objective optimization
                objective_score = {
                    "correctness": entry_metrics["correctness"],
                    "tool_alignment": entry_metrics["tool_alignment"],
                    "grounding": entry_metrics["grounding"],
                    "tokens": float(student_trace["input_tokens"] + student_trace["output_tokens"]),
                    "loops": float(student_trace["num_loops"]),
                    "tool_errors": float(student_trace["num_tool_errors"]),
                }
                all_objective_scores.append(objective_score)

                # Create trajectory if requested
                if capture_traces and all_trajectories is not None:
                    trajectory: ALTrajectory = {
                        "data": al_data_inst,
                        "output": output,
                        "score": score,
                        "objective_scores": objective_score,
                    }
                    all_trajectories.append(trajectory)

        # Compute summary by averaging objective scores across all dimensions
        summary = None
        if all_objective_scores:
            summary = {}
            # Get all unique dimension names
            all_dims = set()
            for obj_score in all_objective_scores:
                all_dims.update(obj_score.keys())

            # Average each dimension
            for dim in all_dims:
                values = [obj_score.get(dim, 0.0) for obj_score in all_objective_scores if dim in obj_score]
                summary[dim] = sum(values) / len(values) if values else 0.0

        return GleanEvaluationBatch(
            outputs=all_outputs,
            scores=all_scores,
            trajectories=all_trajectories,
            objective_scores=all_objective_scores,
            summary=summary
        )

    # ---------------------------
    # 5) High-signal reflective dataset selection (per module)
    # ---------------------------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[ALTrajectory, ALRolloutOutput],
        components_to_update: list[str],
        k: int, # max return
    ) -> dict[str, list[ReflectiveExample]]:
        """
        Build reflective dataset from evaluation results, selecting examples with lowest scores.

        Args:
            candidate: Current candidate prompt modules
            eval_batch: Results from evaluate() with trajectories
            components_to_update: List of component names to generate datasets for

        Returns:
            Dict mapping component_name -> list of reflective examples
        """
        if not eval_batch.trajectories:
            return {comp: [] for comp in components_to_update}

        result: dict[str, list[ReflectiveExample]] = {}

        for component_name in components_to_update:
            # Compute module-specific relevance scores for each example
            scored_trajectories = []

            for idx, trajectory in enumerate(eval_batch.trajectories):
                relevance = self._compute_module_relevance(
                    component_name,
                    trajectory,
                    eval_batch.scores[idx]
                )
                scored_trajectories.append((relevance, idx, trajectory))

            # Sort by relevance (higher = more relevant for improvement)
            # Then by score (lower = worse performance, needs more attention)
            scored_trajectories.sort(key=lambda x: (-x[0], x[2]["score"]))

            # Select diverse examples with poor performance
            reflective_examples = []
            seen_patterns = set()

            for _relevance, _idx, trajectory in scored_trajectories:
                # Create a pattern signature for diversity
                pattern = self._create_failure_pattern(component_name, trajectory)

                # Allow some duplicates near the end to fill quota
                # TODO(Cathy) check with claude why it was < k-2
                if pattern in seen_patterns and len(reflective_examples) > k - 3:
                    continue

                # Build reflective example in standard format
                example = self._build_reflective_example(
                    component_name,
                    trajectory,
                    candidate
                )

                reflective_examples.append(example)
                seen_patterns.add(pattern)

                if len(reflective_examples) >= k:
                    break

            result[component_name] = reflective_examples

        return result

    # TODO(Cathy): Implement this
    def _compute_module_relevance(
        self,
        module_name: str,
        trajectory: ALTrajectory,
        score: float
    ) -> float:
        """Compute how relevant this example is for improving the given module."""
        return 1.0

    def _check_primary_tool_mismatch(self, output: ALRolloutOutput) -> bool:
        """Check if student and teacher used different primary tools."""
        student_tools = output["student_tool_events"]
        teacher_tools = output["teacher_tool_events"]

        if not student_tools or not teacher_tools:
            return len(student_tools) != len(teacher_tools)

        # Primary tool = first tool used
        student_primary = student_tools[0]
        teacher_primary = teacher_tools[0]

        return student_primary != teacher_primary

    def _create_failure_pattern(
        self,
        component_name: str,
        trajectory: ALTrajectory
    ) -> tuple:
        """Create a signature for clustering similar failures."""
        output = trajectory["output"]
        objective_scores = trajectory.get("objective_scores", {})

        if self.judging_mode == "single_model":
            shell_success_rate = objective_scores.get(SHELL_SUCCESS_OBJECTIVE, 1.0)
            verify_pass = objective_scores.get(HIGH_SIGNAL_VERIFY_OBJECTIVE, 1.0)
            shell_error_messages = output.get("shell_error_messages", [])
            return (
                int(shell_success_rate < 0.9),
                int(verify_pass < 1.0),
                int(output["student_tool_errors"] > 0),
                len(shell_error_messages),
            )

        correctness = objective_scores.get("correctness", 1.0)
        tool_mismatch = self._check_primary_tool_mismatch(output)
        return (
            int(correctness < 0.7),
            int(tool_mismatch),
            int(output["student_tool_errors"] > 0),
        )

    def _build_reflective_example(
        self,
        component_name: str,
        trajectory: ALTrajectory,
        candidate: dict[str, str]
    ) -> ReflectiveExample:
        """Build a reflective example in the standard format for instruction proposal."""
        output = trajectory["output"]

        # Determine which tool types were used
        student_tools = output["student_tool_events"]
        teacher_tools = output["teacher_tool_events"]

        feedback_parts = []
        objective_scores = trajectory.get("objective_scores", {})
        metrics: ReflectiveExampleMetrics = {"score": trajectory["score"]}

        if self.judging_mode == "single_model":
            shell_success_rate = objective_scores.get(SHELL_SUCCESS_OBJECTIVE, 1.0)
            metrics["shell_success_rate"] = shell_success_rate

            if shell_success_rate < 0.9:
                feedback_parts.append(
                    f"Shell success rate issue: Student scored {shell_success_rate:.2f} "
                    f"({output['student_tool_errors']} shell errors out of {output['student_tool_calls']} executions)."
                )

            shell_error_messages = output.get("shell_error_messages", [])
            if shell_error_messages:
                feedback_parts.append(
                    "Recent shell errors: " + "; ".join(shell_error_messages[:5])
                )
            elif output["student_tool_errors"] > 0:
                feedback_parts.append(
                    f"Tool errors: Student encountered {output['student_tool_errors']} shell tool errors."
                )

            default_feedback = "General shell tool reliability issue."
        else:
            correctness = objective_scores.get("correctness", trajectory["score"])
            tool_alignment = objective_scores.get("tool_alignment", 0.0)
            metrics["correctness"] = correctness

            if correctness < 0.7:
                feedback_parts.append(
                    f"Correctness issue: Student scored {correctness:.2f} vs teacher baseline."
                )
            if self._check_primary_tool_mismatch(output):
                feedback_parts.append(
                    f"Tool mismatch: student used {student_tools[:3]} vs teacher {teacher_tools[:3]}."
                )
            if tool_alignment < 0.7:
                feedback_parts.append(f"Tool alignment issue: score={tool_alignment:.2f}.")

            default_feedback = "General teacher/student divergence."

        feedback = " ".join(feedback_parts) if feedback_parts else default_feedback

        return {
            "Inputs": {
                "eval_set": trajectory["data"]["eval_set_name"],
                "entry_id": output["entry_id"],
                "deployment_id": output["deployment_id"],
                "query": output["query"],
            },
            "Generated Outputs": {
                "student_answer": output["student_answer"],
                "teacher_answer": output["teacher_answer"],
                "student_tools": student_tools,
                "teacher_tools": teacher_tools,
            },
            "Feedback": feedback,
            "Metrics": metrics,
        }

    # ---------------------------
    # 6) Reflection prompt templates (module-aware)
    # ---------------------------

    def reflection_prompt(self, module_name: str) -> str:
        if module_name == "WRITING_CODE":
            if self.judging_mode == "single_model":
                return (
                    "Focus ONLY on coding instructions that affect shell tool reliability: SDK call patterns, "
                    "ToolResult handling, parallelism via asyncio.gather, sandbox rules, and when to print vs extract. "
                    "Use shell error examples as evidence. Propose minimal deltas."
                )
            return (
                "Focus ONLY on coding instructions: SDK call patterns, ToolResult handling, "
                "parallelism via asyncio.gather, sandbox rules, and when to print vs extract. "
                "Use teacher/student tool divergences as evidence. Propose minimal deltas."
            )
        return "Focus only on this module's responsibilities."

    # ---------------------------
    # 7) propose_new_texts: merge + patch deltas
    # ---------------------------

    def propose_new_texts(
        self,
        reflection_llm: Callable[[str], str],  # plug your LLM call
        candidate: Candidate,
        components_to_update: list[str],
        reflective_examples: list[ReflectiveExample],
        max_variants: int = 3,
    ) -> tuple[list[str], bool]:
        """
        Returns (new_module_texts, module_marked_irrelevant)
        Must:
          1) merge reflections across examples
          2) diagnose recurring failure modes tied to module
          3) propose small patch with before/after
          4) brief why
          5) option: not relevant
        """
        current = candidate.prompt_modules.get(components_to_update[0], "")

        ex_blocks = []
        for r in reflective_examples:
            metrics = r["Metrics"]
            if self.judging_mode == "single_model":
                metric_line = f"shell_success_rate={metrics.get('shell_success_rate', metrics['score']):.2f}"
            else:
                metric_line = (
                    f"score={metrics['score']:.2f}, "
                    f"correctness={metrics.get('correctness', metrics['score']):.2f}"
                )
            ex_blocks.append(
                f"---\n"
                f"QUERY: {r['Inputs']['query']}\n"
                f"TEACHER_ANSWER: {r['Generated Outputs']['teacher_answer']}\n"
                f"STUDENT_ANSWER: {r['Generated Outputs']['student_answer']}\n"
                f"TEACHER_TOOLS: {r['Generated Outputs']['teacher_tools']}\n"
                f"STUDENT_TOOLS: {r['Generated Outputs']['student_tools']}\n"
                f"METRICS: {metric_line}\n"
                f"FEEDBACK: {r['Feedback']}\n"
            )
        if len(components_to_update) != 1:
            return [], False
        module_name = components_to_update[0]
        failure_label = (
            "HIGH-SIGNAL FAILURES"
            if self.judging_mode == "single_model"
            else "HIGH-SIGNAL FAILURES (teacher vs student)"
        )
        prompt = (
            f"You are optimizing ONLY the module {module_name}.\n"
            f"MODULE RESPONSIBILITY:\n{self.reflection_prompt(module_name)}\n\n"
            f"CURRENT MODULE TEXT:\n<<<\n{current}\n>>>\n\n"
            f"{failure_label}:\n{''.join(ex_blocks)}\n\n"
            f"Task:\n"
            f"1) Identify recurring failure modes that are plausibly caused by {module_name}.\n"
            f"2) If {module_name} is NOT relevant, output exactly: NOT_RELEVANT\n"
            f"3) Otherwise propose 1-2 SMALL patches (delta edits), each with:\n"
            f"   - BEFORE: quoted snippet from current module\n"
            f"   - AFTER: revised snippet\n"
            f"   - WHY: one sentence\n"
            f"4) Keep token budget in mind; do not bloat.\n"
        )

        raw = reflection_llm(prompt).strip()
        if raw == "NOT_RELEVANT":
            return [], True

        # Simple parser strategy:
        # In production, parse structured JSON/YAML, or ask LLM to output patches in JSON.
        # Here we just return one consolidated rewrite request for a second pass:
        consolidate_prompt = (
            f"Consolidate the following patch suggestions into up to {max_variants} candidate rewrites "
            f"of the module {module_name}. Preserve good behavior, incorporate consistent changes only. "
            f"Output each variant separated by '\n===VARIANT===\n'.\n\n"
            f"CURRENT:\n<<<\n{current}\n>>>\n\n"
            f"SUGGESTIONS:\n{raw}\n"
        )
        consolidated = reflection_llm(consolidate_prompt).strip()
        variants = [v.strip() for v in consolidated.split("===VARIANT===") if v.strip()]
        return variants[:max_variants], False
