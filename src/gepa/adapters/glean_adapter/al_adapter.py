from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import random
import re
import requests
import time
from typing import Any, Callable, Dict, List, NotRequired, Optional, Tuple, TypedDict

from anyio import sleep
from gepa.adapters.glean_adapter.prompt import compile_encoded_prompt
from gepa.core.adapter import EvaluationBatch


# ---------------------------
# 1) Prompt modules + candidate
# ---------------------------

MODULES = [
    "GLOBAL_ROLE",
    "PERSISTENCE",
    "FORMATTING",
    "TOOL_USAGE_1",
    "TOOL_USAGE_2",
    "TOOL_USAGE_3",
    "TOOL_USAGE_4",
]

@dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    kind: str                  # "free_text" | "enum_knob"
    token_budget: int

@dataclass
class Candidate:
    model: str                 # "claude" | "gemini" etc
    prompt_modules: Dict[str, str]  # Flattened: {"GLOBAL_ROLE": "...", "TOOL_USAGE_1": "...", ...}
    module_specs: Dict[str, ModuleSpec]
    global_token_cap: int      # relative to baseline prompt for that model
    baseline_prompt_hash: str  # used to define "relative cap"

    # bookkeeping for GEPA loop
    parent_id: Optional[str] = None
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
    deployment_ids: List[str]
    status: str

class ALRolloutOutput(TypedDict):
    deployment_id: str
    query: str
    # Student execution results
    student_answer: str
    student_tool_events: List[str]
    student_loops: int
    student_tool_calls: int
    student_tool_errors: int
    student_input_tokens: int
    student_output_tokens: int
    student_latency_ms: Optional[int]

    # Teacher execution results (for comparison)
    teacher_answer: str
    teacher_tool_events: List[str]
    teacher_loops: int
    teacher_tool_calls: int
    teacher_input_tokens: int
    teacher_output_tokens: int

    # Entry metadata
    entry_id: str

class ALTrajectory(TypedDict):
    # Input data
    data: ALDataInst
    # Full execution output
    output: ALRolloutOutput
    # Score
    score: float
    # Objective scores
    objective_scores: Dict[str, float]

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
    spans: NotRequired[List[Dict[str, Any]]]

    # Execution details
    query: str
    answer: str
    tool_events: List[Dict[str, Any]]
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
    correctness: float
    tool_alignment: float
    grounding: float
    score: float

# TypedDict with keys containing spaces must use functional form
ReflectiveExample = TypedDict('ReflectiveExample', {
    'Inputs': ReflectiveExampleInputs,
    'Generated Outputs': ReflectiveExampleOutputs,
    'Feedback': str,
    'Metrics': ReflectiveExampleMetrics,
})

@dataclass
class ToolEvent:
    tool_type: str             # "GST" | "web" | "MCP" | ...
    name: str
    ok: bool
    input_tokens: int = 0
    output_tokens: int = 0

@dataclass
class JudgeResult:
    correctness: float         # 0..1
    tool_alignment: float
    grounding: float
    rationale: str
    traces: Optional[Dict[str, List[TraceInfo]]] = None  # Maps entry_id -> list of trace_info dicts

# ---------------------------
# 3) Teacher cache + runner interfaces
# ---------------------------

class ALRunner:
    """
    Triggers eval runs and manages eval run IDs for the judge.

    The runner just triggers eval runs via POST and tracks the eval_run_ids.
    All execution data (answers, tool events, tokens, etc.) is retrieved later by the Judge
    from the same analysis/view endpoint.
    """
    def __init__(
        self,
        api_url: str = "https://apps-gke.glean.com/debug/cortex/evalruns",
        cookie: Optional[str] = None,
        deployment_ids: Optional[List[str]] = None,
        cache_file: Optional[str] = None,
    ):
        self.api_url = api_url
        self.cookie = cookie
        self.deployment_ids = deployment_ids or ["scio-prod"]
        self.cache_file = cache_file

        # Track eval run IDs: cache_key -> eval_run_id
        self._eval_run_ids: Dict[Tuple[str, str, str, str], str] = {}

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
                with open(self.cache_file, 'r') as f:
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

            with open(self.cache_file, 'w') as f:
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

    def run(self, model: str, system_prompt: str, eval_set_name: str, eval_set_version: str, deployment_ids: list[str]) -> str:
        """
        Trigger an eval run and return the eval_run_id.

        Args:
            model: "claude" or "gpt"
            system_prompt: Compiled system prompt (sc parameter string from compile_system_prompt)
            eval_set_name: Name of the eval set
            eval_set_version: Version of the eval set
            deployment_ids: List of deployment IDs to use

        Returns:
            eval_run_id string
        """
        # Create cache key based on eval set and system prompt
        system_prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:16]
        cache_key = (model, system_prompt_hash, eval_set_name, eval_set_version)

        # Check if we've already run this eval set for this candidate
        if cache_key in self._eval_run_ids:
            return self._eval_run_ids[cache_key]

        # Need to run the eval set - generate unique eval_id
        eval_id = f"gepa_{model}_{system_prompt_hash}_{int(time.time())}"

        # Build scParams
        sc_params = self._build_sc_params(model, system_prompt)
        eval_params = "experimental_queue=eval-experimental-2"
        if model == 'fast':
            eval_params += f",gleanchat_agent=FAST"
        else:
            eval_params += f",gleanchat_agent=ADVANCED"

        # Build request body
        body = {
            "deploymentIds": deployment_ids,
            "deploymentOption": "CUSTOM",
            "description": "",
            "evalSetName": eval_set_name,
            "evalSetVersion": eval_set_version,
            "id": eval_id,
            "runConfig": {
                "evalParams": "experimental_queue=eval-experimental-2",
                "runnerType": "GLEAN_CHAT",
                "scParams": sc_params,
            }
        }

        # Build headers
        headers = {
            "accept": "*/*",
            "content-type": "application/json",
            "origin": "https://dev.glean.com",
            "referer": "https://dev.glean.com/",
            "sec-ch-ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        }

        if self.cookie:
            headers["cookie"] = self.cookie
        else:
            raise ValueError("No cookie provided for ALRunner")

        # Step 1: Make request to create the eval run
        print(f"Creating eval run {eval_id} for {eval_set_name}:{eval_set_version}...")
        response = requests.post(
            self.api_url,
            headers=headers,
            json=body,
            timeout=300,
        )
        response.raise_for_status()
        print(f"Eval run {eval_id} triggered successfully")

        # Store eval run ID for later judge runs
        self._eval_run_ids[cache_key] = eval_id
        self._save_cache()

        return eval_id


    def get_eval_run_id(self, model: str, system_prompt: str, eval_set_name: str, eval_set_version: str) -> Optional[str]:
        """Get the eval run ID for a given cache key."""
        system_prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:16]
        cache_key = (model, system_prompt_hash, eval_set_name, eval_set_version)
        return self._eval_run_ids.get(cache_key)

class Judge:
    """
    LLM judge that compares teacher vs student:
      correctness, grounding, rationale

    Usage:
        judge = Judge(
            api_url="https://apps-gke.glean.com/debug/cortex/judgeruns/batch",
            cookie="your_cookie_string"
        )

    Flow:
        1. Trigger judge run: POST /judgeruns/batch with teacher and student eval IDs
        2. Wait 15 minutes for judge run to complete
        3. Get pairwise metrics: POST /metrics/evalruns/pairwise with baseEvalId and testEvalId
        4. Parse and cache results by (teacher_eval_id, student_eval_id)

    Note:
        - Judge results are cached to avoid reruns
        - Uses pairwise metrics endpoint to get comparison results
    """
    def __init__(
        self,
        api_url: str = "https://apps-gke.glean.com/debug/cortex/judgeruns/batch",
        cookie: Optional[str] = None,
    ):
        self.api_url = api_url
        self.cookie = cookie
        # Cache judge results: (teacher_eval_id, student_eval_id) -> JudgeResult
        self._judge_cache: Dict[Tuple[str, str], JudgeResult] = {}

    def judge(
        self,
        teacher_eval_id: str,
        student_eval_id: str,
        skip_trigger: bool = False,
    ) -> JudgeResult:
        """
        Run LLM judge comparison between teacher and student.
        If eval IDs are provided, triggers actual judge run via API and caches results.
        Otherwise returns dummy result.

        Args:
            teacher_eval_id: Teacher eval run ID
            student_eval_id: Student eval run ID
            skip_trigger: If True, skip triggering the judge and go straight to fetching results
        """
        # Check cache
        cache_key = (teacher_eval_id, student_eval_id)
        if cache_key in self._judge_cache:
            print(f"Using cached judge result for {teacher_eval_id} vs {student_eval_id}")
            return self._judge_cache[cache_key]

        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://dev.glean.com",
            "referer": "https://dev.glean.com/",
            "sec-ch-ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        }

        if self.cookie:
            headers["cookie"] = self.cookie

        if not skip_trigger:
            # Trigger judge run
            print(f"Triggering judge run for {teacher_eval_id} vs {student_eval_id}...")

            # Build request body
            body = [{
                "baseEvalRunId": teacher_eval_id,
                "config": {
                    "inputMappings": [
                        {"entryType": "TEST", "name": "Query", "path": "Query", "sourceType": "EVAL_SET_PROTO"},
                        {"entryType": "TEST", "name": "Response", "path": "EvalChatResponseInfo.ActResponse", "sourceType": "EVAL_RUN_OUTPUT"},
                        {"entryType": "BASE", "name": "CanonicalAnswer", "path": "EvalChatResponseInfo.ActResponse", "sourceType": "EVAL_RUN_OUTPUT"}
                    ],
                    "judgeType": "CORRECTNESS",
                    "runParameters": {
                        "Judge Type": "DIRECT_CORRECTNESS",
                        "Llm model": "default",
                        "Use Cache": "true"
                    }
                },
                "evalRunId": student_eval_id
            }]

            # Trigger judge run
            response = requests.post(
                self.api_url,
                headers=headers,
                json=body,
                timeout=300,
            )
            response.raise_for_status()
            print(f"Judge run triggered successfully")
            # TODO(Cathy): Add a better way of detecting that the judge is done.
            print("Waiting for judge to finish by sleeping for 10 mins")
            time.sleep(600)
        else:
            print(f"Skipping judge trigger (already triggered), fetching results for {teacher_eval_id} vs {student_eval_id}.")

        judge_result = self._get_full_judge_results(student_eval_id, teacher_eval_id, headers)

        # Cache the result
        self._judge_cache[cache_key] = judge_result
        return judge_result

    def _get_full_judge_results(self, student_eval_id: str, teacher_eval_id: str, headers: Dict[str, str]) -> JudgeResult:

        # Get judge results from analysis/view endpoint
        analysis_url = "https://apps-gke.glean.com/debug/cortex/analysis/view"

        analysis_response = requests.get(
            analysis_url,
            params={"testRunIds": student_eval_id, "baseRunId": teacher_eval_id},
            headers=headers,
            timeout=300,
        )
        analysis_response.raise_for_status()
        result_data = analysis_response.json()

        entries = result_data.get("entries", [])

        deployments = []
        entry_details = []
        durations_map = dict()
        loop_counts_map = dict()
        input_tokens_map = dict()
        output_tokens_map = dict()
        tools_invocations_map = dict()
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


        # Get detailed run information including traces
        trace_map: defaultdict[str, list[TraceInfo]] = defaultdict(list)  # Maps entry_id -> list of trace_info
        details_url = "https://apps-gke.glean.com/debug/cortex/analysis/view/details"
        details_response = requests.post(
            details_url,
            headers=headers,
            json=entry_details,
            timeout=300,
        )
        details_response.raise_for_status()
        details_data = details_response.json()

        # Extract trace information for each (entryId, evalId) pair
        for item in details_data:
            if item.get("error"):
                print(f"Error in details data: {item.get('error')}")
                continue
            entry_id = item.get("evalSetEntry", {}).get("id")
            deployment_id = item.get("evalSetEntry", {}).get("deploymentId")
            run_responses = item.get("runResponses", [])
            trace_infos: list[TraceInfo] = []

            correctness_scores = dict()
            for judge_entry in item.get("judgeRunEntries"):
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
                print(f'Got query: {query}')

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
        for entry_id, trace_infos in trace_map.items():
            for trace_info in trace_infos:
                eval_id = trace_info.get("eval_id")
                trace_id = trace_info.get("trace_id")
                finish_time_ms = trace_info.get("finish_time_millis")
                deployment_id = trace_info.get("deployment_id")

                if trace_id and finish_time_ms and deployment_id:
                    # Start time is 1 hour (3600000 ms) before finish time
                    start_time_ms = finish_time_ms - 3600000
                    end_time_ms = finish_time_ms

                    trace_url = "https://apps-gke.glean.com/debug/cortex/analysis/view/trace"
                    trace_response = requests.get(
                        trace_url,
                        params={
                            "deploymentId": deployment_id,
                            "traceId": trace_id,
                            "startTimeMillis": start_time_ms,
                            "endTimeMillis": end_time_ms
                        },
                        headers=headers,
                        timeout=60,
                    )
                    trace_response.raise_for_status()
                    detailed_trace = trace_response.json()

                    # Store the detailed trace
                    trace_info["spans"] = detailed_trace.get("trace", {}).get("spans")

        # Average correctness across all entries
        correctness_score_list = []
        for entry_id, trace_infos in trace_map.items():
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

def render_system_prompt(candidate: Candidate) -> str:
    # Deterministic formatting: stable module order
    parts = []
    for mid in MODULES:
        if mid in candidate.prompt_modules:
            parts.append(f"## {mid}\n{candidate.prompt_modules[mid].strip()}\n")
    return "\n".join(parts).strip()


def compile_system_prompt(candidate: Candidate) -> str:
    """Compile candidate prompt_modules into encoded system prompt parameter."""
    return compile_encoded_prompt(candidate.prompt_modules)

def get_tool_alignment(trace_map: Dict[str, List[TraceInfo]], student_eval_id: str, teacher_eval_id: str) -> float:
    tool_alignment_scores = []
    for entry_id, trace_infos in trace_map.items():
        student_run = None
        teacher_run = None
        for trace_info in trace_infos:
            eval_id = trace_info.get("eval_id")
            if eval_id == student_eval_id:
                student_run = trace_info['spans']
            elif eval_id == teacher_eval_id:
                teacher_run = trace_info['spans']
        if not student_run or not teacher_run:
            return 0
        tool_alignment_scores.append(get_tool_alignment_from_traces(student_run, teacher_run))
    return sum(tool_alignment_scores) / len(tool_alignment_scores)


# TODO(Cathy) Get cleaner traces than what the api current returns
def get_tool_alignment_from_traces(student_trace_spans: List[Dict[str, Any]], teacher_trace_spans: List[Dict[str, Any]]) -> float:
    student_tool_usages = []
    teacher_tool_usages = []
    for span in student_trace_spans:
        if "Execute Action" in span['name']:
            tool_usage = parse_tool_usage(span)
            if not tool_usage:
                continue
            student_tool_usages.append(tool_usage)
    for span in teacher_trace_spans:
        if "Execute Action" in span['name']:
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
            input = json.loads(parsed["input"])
            return input
        except:
            print(f"Could not parse input: {parsed["input"]}")
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


def parse_tool_usage(span: dict) -> tuple[str, dict]:
    parts = span['name'].split(": ")
    if len(parts) < 2:
        print(f"Invalid span name: {span['name']}")
        return None
    tool_name = span['name'].split(": ")[1]
    tool_inputs = span['attributes']['input']
    if tool_name == "Personal Knowledge Vault Retrieve":
        return None
    try:
        tool_inputs = process_raw_typed_value(tool_inputs)
        del tool_inputs['id']
        return (tool_name, tool_inputs)
    except:
        return None

# ---------------------------
# 4) AssistantALAdapter (core)
# ---------------------------

@dataclass
class Thresholds:
    quality_min: float
    tools_min: float
    max_student_tokens: int

@dataclass
class EvalSummary:
    # multi-objective means we keep all
    quality: float
    tool_align: float
    loops: float
    tokens: float
    # additional observability
    fail_rate: float
    avg_tool_errors: float

class AssistantALAdapter:
    def __init__(
        self,
        runner: ALRunner,
        judge: Judge,
        teacher_model: str,
        thresholds: Thresholds,
        student_model: str,
        cache_file: Optional[str] = None,
    ):
        self.runner = runner
        self.judge = judge
        self.teacher_model = teacher_model
        self.thresholds = thresholds
        self.student_model = student_model
        self.cache_file = cache_file

        # module freezing memory: module -> count of "not relevant" in consecutive generations
        self._module_irrelevant_streak: Dict[Tuple[str, str], int] = defaultdict(int)  # (candidate_family, module) -> streak

        # good options pool per module (to avoid losing good parts)
        # TODO(Cathy) populate the good_module_options
        self.good_module_options: Dict[str, List[str]] = defaultdict(list)

        # Eval ID cache: (eval_set_name, eval_set_version, model, prompt_hash) -> eval_id
        self._eval_cache: Dict[Tuple[str, str, str, str], str] = {}

        # Judge triggered cache: (teacher_eval_id, student_eval_id) -> triggered
        self._judge_triggered: set[Tuple[str, str]] = set()

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
                with open(self.cache_file, 'r') as f:
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

            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[AssistantALAdapter] Saved {len(self._eval_cache)} eval IDs and {len(self._judge_triggered)} judge triggers to cache: {self.cache_file}")
        except Exception as e:
            print(f"[AssistantALAdapter] Failed to save cache to {self.cache_file}: {e}")

    def _extract_per_entry_metrics(
        self,
        judge_result: JudgeResult,
        student_eval_id: str,
        teacher_eval_id: str
    ) -> Dict[str, Dict[str, float]]:
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
        """Evaluate candidate on eval set(s).

        Args:
            batch: List of eval set specifications
            candidate: Flattened candidate dict[str, str]
            capture_traces: Whether to capture traces for reflection

        Returns:
            EvaluationBatch with per-query outputs, scores, and optional trajectories
        """
        # Handle empty batch
        if not batch:
            return EvaluationBatch(
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
            teacher_prompt_hash = hashlib.md5("<<TEACHER_PROD_PROMPT>>".encode()).hexdigest()[:16]
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
                # TODO(Cathy): wait till the evals are done running
                print("Waiting for evals to finish by sleeping for 15 mins")
                time.sleep(900)
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

        return EvaluationBatch(
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
    ) -> Dict[str, List[ReflectiveExample]]:
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

        result: Dict[str, List[ReflectiveExample]] = {}

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

            for relevance, idx, trajectory in scored_trajectories:
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
        correctness = trajectory.get("objective_scores", {})["correctness"]
        tool_alignment = trajectory.get("objective_scores", {})["tool_alignment"]
        grounding = trajectory.get("objective_scores", {})["grounding"]

        output = trajectory["output"]

        return (
            int(self._check_primary_tool_mismatch(output)),
            int(correctness < 0.75),
            int(grounding < 0.75),
            int(output["student_tool_errors"] > 0),
            int(output["student_input_tokens"] > 10000),
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

        # Build feedback based on metrics
        feedback_parts = []

        if trajectory.get("objective_scores", {})["correctness"] < 0.75:
            feedback_parts.append(
                f"Correctness issue: Student scored {trajectory.get("objective_scores", {})['correctness']:.2f}. "
            )

        if trajectory.get("objective_scores", {})["tool_alignment"] < 0.5:
            feedback_parts.append(
                f"Tool alignment issue: Student used {student_tools}, "
                f"teacher used {teacher_tools}."
            )

        if output["student_tool_errors"] > 0:
            feedback_parts.append(
                f"Tool errors: Student encountered {output['student_tool_errors']} tool errors."
            )

        if trajectory.get("objective_scores", {})["grounding"] < 0.8:
            feedback_parts.append(
                f"Grounding issue: Response may not be well-grounded in retrieved information."
            )

        feedback = " ".join(feedback_parts) if feedback_parts else "General quality issue."

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
            "Metrics": {
                "correctness": trajectory.get("objective_scores", {})["correctness"],
                "tool_alignment": trajectory.get("objective_scores", {})["tool_alignment"],
                "grounding": trajectory.get("objective_scores", {})["grounding"],
                "score": trajectory["score"],
            }
        }

    # ---------------------------
    # 6) Reflection prompt templates (module-aware)
    # ---------------------------

    def reflection_prompt(self, module_name: str) -> str:
        # Doc: customize per module to reduce cross-contamination
        if module_name == "GLOBAL_ROLE":
            return (
                "Focus ONLY on behavior framing: how the assistant should approach the user query. "
                "Use evidence from examples. Propose small patch(es) to improve appropriateness."
            )
        if module_name.startswith("TOOL_USAGE"):
            # All TOOL_USAGE parts handle tool selection and usage guidelines
            return (
                "Focus ONLY on tool usage guidelines: when/why to use tools, parameter selection, parallel execution, etc. "
                "Prioritize tool choice divergences from teacher. Propose minimal deltas."
            )
        if module_name == "PERSISTENCE":
            return (
                "Focus ONLY on how many variations to try, how to handle unclear intent, and how to avoid hallucinations. "
                "Adjust exploration vs assumptions conservatively."
            )
        if module_name == "FORMATTING":
            return (
                "Focus ONLY on markdown/citations/artifacts triggering. NO behavioral changes. "
                "Propose concrete formatting rule edits."
            )
        return "Focus only on this module’s responsibilities."

    # ---------------------------
    # 7) propose_new_texts: merge + patch deltas
    # ---------------------------

    def propose_new_texts(
        self,
        reflection_llm: Callable[[str], str],  # plug your LLM call
        candidate: Candidate,
        components_to_update: list[str],
        reflective_examples: List[ReflectiveExample],
        max_variants: int = 3,
    ) -> Tuple[List[str], bool]:
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

        # Build reflective dataset payload (minimal but high-signal)
        ex_blocks = []
        for r in reflective_examples:
            print(f"Processing reflective example: {r}")
            ex_blocks.append(
                f"---\n"
                f"QUERY: {r['Inputs']['query']}\n"
                f"TEACHER_ANSWER: {r['Generated Outputs']['teacher_answer']}\n"
                f"STUDENT_ANSWER: {r['Generated Outputs']['student_answer']}\n"
                f"TEACHER_TOOLS: {r['Generated Outputs']['teacher_tools']}\n"
                f"STUDENT_TOOLS: {r['Generated Outputs']['student_tools']}\n"
                f"JUDGE: correctness={r['Metrics']['correctness']:.2f}, grounding={r['Metrics']['grounding']:.2f}\n"
                # f"FEEDBACK: {r['Feedback']}\n"
            )
        if len(components_to_update) != 1:
            return None
        module_name = components_to_update[0]
        prompt = (
            f"You are optimizing ONLY the module {module_name}.\n"
            f"MODULE RESPONSIBILITY:\n{self.reflection_prompt(module_name)}\n\n"
            f"CURRENT MODULE TEXT:\n<<<\n{current}\n>>>\n\n"
            f"HIGH-SIGNAL FAILURES (teacher vs student):\n{''.join(ex_blocks)}\n\n"
            f"Task:\n"
            f"1) Identify recurring failure modes that are plausibly caused by {module_name}.\n"
            f"2) If {module_name} is NOT relevant, output exactly: NOT_RELEVANT\n"
            f"3) Otherwise propose 1-2 SMALL patches (delta edits), each with:\n"
            f"   - BEFORE: quoted snippet from current module\n"
            f"   - AFTER: revised snippet\n"
            f"   - WHY: one sentence\n"
            f"4) Keep token budget in mind; do not bloat.\n"
        )
        print("***BEGIN PROMPT***")
        print(prompt)
        print("***END OF PROMPT***")
        time.sleep(1000)

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


# ---------------------------
# 8) Pareto frontier management
# ---------------------------

def dominates(a: EvalSummary, b: EvalSummary) -> bool:
    """
    Multi-objective:
      - maximize quality
      - maximize tool_align
      - minimize loops
      - minimize tokens
    a dominates b if >= on quality/tool_align and <= on loops/tokens and strictly better in at least one.
    """
    better_or_eq = (
        a.quality >= b.quality and
        a.tool_align >= b.tool_align and
        a.loops <= b.loops and
        a.tokens <= b.tokens
    )
    strictly_better = (
        a.quality > b.quality or
        a.tool_align > b.tool_align or
        a.loops < b.loops or
        a.tokens < b.tokens
    )
    return better_or_eq and strictly_better

@dataclass
class Frontier:
    members: Dict[str, EvalSummary] = field(default_factory=dict)

    def add_if_pareto(self, cand_id: str, summ: EvalSummary) -> None:
        # Remove dominated existing members; reject if dominated by any
        for oid, os in list(self.members.items()):
            if dominates(os, summ):
                return
            if dominates(summ, os):
                del self.members[oid]
        self.members[cand_id] = summ

    def best_per_dimension(self) -> Dict[str, str]:
        # Doc: explicitly keep 4 candidates: best correctness(quality), best tools, best tokens, best loops
        if not self.members:
            return {}
        best_quality = max(self.members.items(), key=lambda kv: kv[1].quality)[0]
        best_tools = max(self.members.items(), key=lambda kv: kv[1].tool_align)[0]
        best_tokens = min(self.members.items(), key=lambda kv: kv[1].tokens)[0]
        best_loops = min(self.members.items(), key=lambda kv: kv[1].loops)[0]
        return {"quality": best_quality, "tools": best_tools, "tokens": best_tokens, "loops": best_loops}