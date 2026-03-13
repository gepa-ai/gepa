from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import random
import requests
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

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
    feedback_summary: str

class ALTrajectory(TypedDict):
    data: ALDataInst
    feedback: ALRolloutOutput

@dataclass
class ToolEvent:
    tool_type: str             # "GST" | "web" | "MCP" | ...
    name: str
    ok: bool
    input_tokens: int = 0
    output_tokens: int = 0

@dataclass
class RunTrace:
    answer: str
    tool_events: List[ToolEvent]
    num_loops: int
    num_tool_calls: int
    num_tool_errors: int
    input_tokens: int
    output_tokens: int
    latency_ms: Optional[int] = None

@dataclass
class JudgeResult:
    correctness: float         # 0..1
    completeness: float
    grounding: float
    safety: float
    rationale: str

@dataclass
class EvalResult:
    example_id: str
    teacher: RunTrace
    student: RunTrace
    judge: JudgeResult

    # tool alignment metrics
    primary_tool_match: float
    tool_dist_distance: float  # lower is better

    # aggregated behavior/efficiency
    loops: int
    tool_calls: int
    tool_errors: int

    # tokens/cost
    prompt_tokens: int
    student_total_tokens: int  # student input+output tokens

    # scalar for quick comparisons
    quality_score: float       # derived from judge fields
    tool_align_score: float    # derived from match + distance

    # diagnostic tags (for clustering failures)
    tags: Dict[str, Any] = field(default_factory=dict)

def primary_tool_type(trace: RunTrace) -> Optional[str]:
    if not trace.tool_events:
        return None
    # primary = first tool type used (or most frequent—pick one; doc says primary tool type match rate)
    return trace.tool_events[0].tool_type

def tool_type_distribution(trace: RunTrace) -> Dict[str, int]:
    d = defaultdict(int)
    for ev in trace.tool_events:
        d[ev.tool_type] += 1
    return dict(d)

def l1_dist(d1: Dict[str, int], d2: Dict[str, int]) -> float:
    keys = set(d1) | set(d2)
    return float(sum(abs(d1.get(k, 0) - d2.get(k, 0)) for k in keys))

def aggregate_quality(j: JudgeResult) -> float:
    # You can weight these; doc calls out correctness, completeness, grounding, safety.
    return float(0.35*j.correctness + 0.25*j.completeness + 0.20*j.grounding + 0.20*j.safety)

def tool_alignment_score(primary_match: float, dist_distance: float) -> float:
    # Higher better; normalize distance a bit.
    return float(0.7*primary_match + 0.3*(1.0 / (1.0 + dist_distance)))

# ---------------------------
# 3) Teacher cache + runner interfaces
# ---------------------------
from .utils import convert_workflow_trace_to_runtrace

class TeacherCache:
    def __init__(self):
        self._cache: Dict[str, RunTrace] = {}

    def get(self, key: str) -> Optional[RunTrace]:
        return self._cache.get(key)

    def put(self, key: str, trace: RunTrace) -> None:
        self._cache[key] = trace

class ALRunner:
    """
    Your integration point:
      run(model=..., system_prompt=..., query=..., env=...) -> RunTrace
    Must run full AL loops and return tool traces + token counts + tool errors, etc.

    Note: The Glean API runs entire eval sets, not individual queries.
    This runner caches results from eval set runs to avoid redundant API calls.
    """
    def __init__(
        self,
        api_url: str = "https://apps-gke.glean.com/debug/cortex/evalruns",
        cookie: Optional[str] = None,
        eval_set_name: str = "Glean Chat Multiturn V2 Small",
        eval_set_version: str = "20260308",
        deployment_ids: Optional[List[str]] = None,
    ):
        self.api_url = api_url
        self.cookie = cookie
        self.eval_set_name = eval_set_name
        self.eval_set_version = eval_set_version
        self.deployment_ids = deployment_ids or ["scio-prod"]

        # Cache for eval set results: (model, system_prompt_hash, eval_set_name, eval_set_version) -> {query: RunTrace}
        self._eval_set_cache: Dict[Tuple[str, str, str, str], Dict[str, RunTrace]] = {}
        # Track eval run IDs for judge runs: cache_key -> eval_run_id
        self._eval_run_ids: Dict[Tuple[str, str, str, str], str] = {}

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
        elif model != "gpt":
            raise ValueError(f"Unknown model: {model}")

        # Add system prompt override if provided (and not the placeholder)
        if system_prompt and system_prompt != "<<TEACHER_PROD_PROMPT>>":
            # system_prompt should already be the compiled sc parameter from compile_system_prompt
            base_params.append(system_prompt)

        return ",".join(base_params)

    def run(self, model: str, system_prompt: str, env: Dict[str, Any]) -> RunTrace:
        """
        Run a single query through Glean's eval API.

        Note: The Glean API runs entire eval sets. This method caches results from the eval set
        run and returns the specific query result.

        Args:
            model: "claude" or "gpt"
            system_prompt: Compiled system prompt (sc parameter string from compile_system_prompt)
            query: User query to evaluate (or "RUN_EVAL_SET:name:version" to run entire set)
            env: Environment config (deployment_ids, eval_set_name, eval_set_version, etc.)

        Returns:
            RunTrace with results for this specific query
        """
        # Allow overriding eval set per run (for train vs val vs test)
        eval_set_name = env.get("eval_set_name", self.eval_set_name)
        eval_set_version = env.get("eval_set_version", self.eval_set_version)

        # Create cache key based on eval set and system prompt
        system_prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:16]
        cache_key = (model, system_prompt_hash, eval_set_name, eval_set_version)

        # (TODO) Check if we've already run this eval set for this candidate

        # Need to run the eval set - generate unique eval_id
        eval_id = f"gepa_{model}_{system_prompt_hash}_{int(time.time())}"

        # Build scParams
        sc_params = self._build_sc_params(model, system_prompt)

        # Get deployment IDs from env or use defaults
        deployment_ids = env.get("deployment_ids", self.deployment_ids)

        # Build request body
        body = {
            "deploymentIds": deployment_ids,
            "deploymentOption": "CUSTOM",
            "description": "",
            "evalSetName": eval_set_name,
            "evalSetVersion": eval_set_version,
            "id": eval_id,
            "runConfig": {
                "evalParams": "gleanchat_agent=ADVANCED,experimental_queue=eval-experimental-2",
                "runnerType": "GLEAN_CHAT",
                "scParams": sc_params,
            }
        }

        # Build headers
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

        # Step 1: Make request to create the eval run
        try:
            print(f"Creating eval run {eval_id} for {eval_set_name}:{eval_set_version}...")
            response = requests.post(
                self.api_url,
                headers=headers,
                json=body,
                timeout=120,
            )
            response.raise_for_status()
            print(f"Eval run {eval_id} created successfully")

            # Step 2: Poll for results
            analysis_url = self.api_url.replace("/evalruns", "/analysis/view")

            print(f"Waiting 5 min for eval run {eval_id} to complete...")
            # wait for 5 minutes if eval_set_name ends in small
            if "Small" in eval_set_name:
                time.sleep(300) # 5 minutes
            else:
                time.sleep(600)

            # Fetch results
            analysis_response = requests.get(
                analysis_url,
                params={"testRunIds": eval_id},
                headers=headers,
                timeout=30,
            )
            analysis_response.raise_for_status()
            result = analysis_response.json()

            # Check if results are ready
            # The response structure contains the eval results
            if result and isinstance(result, dict):
                # Extract eval results - the structure may vary, adjust as needed
                test_runs = result.get("testRuns", [])
                if test_runs:
                    # Found results
                    test_run = test_runs[0]
                    eval_results = test_run.get("evalSet", [])

                    # Cache all query results from this eval set run
                    query_results: Dict[str, RunTrace] = {}
                    for entry in eval_results:
                        entry_query = entry.get("query", "")
                        if entry_query:
                            eval_chat_response_info = entry.get("evalChatResponseInfo", {})
                            query_results[entry_query] = convert_workflow_trace_to_runtrace(eval_chat_response_info)

                    # Store in cache
                    self._eval_set_cache[cache_key] = query_results
                    # Store eval run ID for judge runs
                    self._eval_run_ids[cache_key] = eval_id

                    # Query not found in results
                    return RunTrace(
                        answer="",
                        tool_events=[],
                        num_loops=0,
                        num_tool_calls=0,
                        num_tool_errors=1,
                        input_tokens=0,
                        output_tokens=0,
                    )

            # Timeout waiting for results
            print(f"Timeout waiting for eval run {eval_id} to complete")
            return RunTrace(
                answer=f"TIMEOUT: Eval run did not complete within 300s",
                tool_events=[],
                num_loops=0,
                num_tool_calls=0,
                num_tool_errors=1,
                input_tokens=0,
                output_tokens=0,
            )

        except Exception as e:
            # Return error trace on failure
            print(f"Error running eval set {eval_set_name}:{eval_set_version}: {e}")
            print(f"Eval id that failed: {eval_id}")
            return RunTrace(
                answer=f"ERROR: {str(e)}",
                tool_events=[],
                num_loops=0,
                num_tool_calls=0,
                num_tool_errors=1,
                input_tokens=0,
                output_tokens=0,
            )

    def get_eval_run_id(self, model: str, system_prompt: str, eval_set_name: str, eval_set_version: str) -> Optional[str]:
        """Get the eval run ID for a given cache key."""
        system_prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:16]
        cache_key = (model, system_prompt_hash, eval_set_name, eval_set_version)
        return self._eval_run_ids.get(cache_key)

class Judge:
    """
    LLM judge that compares teacher vs student:
      correctness, completeness, grounding, safety + rationale

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
        teacher_eval_id: Optional[str] = None,
        student_eval_id: Optional[str] = None,
    ) -> JudgeResult:
        """
        Run LLM judge comparison between teacher and student.
        If eval IDs are provided, triggers actual judge run via API and caches results.
        Otherwise returns dummy result.
        """
        # If no eval IDs provided, return dummy result (for backward compatibility)
        if teacher_eval_id is None or student_eval_id is None:
            return JudgeResult(
                correctness=0.8,
                completeness=0.8,
                grounding=0.8,
                safety=0.9,
                rationale=""
            )

        # Check cache
        cache_key = (teacher_eval_id, student_eval_id)
        if cache_key in self._judge_cache:
            print(f"Using cached judge result for {teacher_eval_id} vs {student_eval_id}")
            return self._judge_cache[cache_key]

        # Trigger judge run
        print(f"Triggering judge run for {teacher_eval_id} vs {student_eval_id}...")

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

        # Build request body
        body = [{
            "baseEvalRunId": teacher_eval_id,
            "config": {
                "inputMappings": [
                    {"entryType": "TEST", "name": "Query", "path": "Query", "sourceType": "EVAL_SET_PROTO"},
                    {"entryType": "TEST", "name": "Response", "path": "EvalChatResponseInfo.ActResponse", "sourceType": "EVAL_RUN_OUTPUT"},
                    {"entryType": "BASE", "name": "CanonicalAnswer", "path": "EvalChatResponseInfo.ActResponse", "sourceType": "EVAL_SET_PROTO"}
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

        try:
            # Trigger judge run
            response = requests.post(
                self.api_url,
                headers=headers,
                json=body,
                timeout=120,
            )
            response.raise_for_status()
            print(f"Judge run triggered successfully")

            # Wait 15 minutes for results (matching eval run wait time)
            print(f"Waiting 60 minutes for judge run to complete...")
            time.sleep(600)

            # Get pairwise metrics comparing teacher vs student
            metrics_url = "https://apps-gke.glean.com/debug/cortex/metrics/evalruns/pairwise"

            metrics_body = {
                "baseEvalId": teacher_eval_id,
                "testEvalId": student_eval_id
            }

            metrics_response = requests.post(
                metrics_url,
                headers=headers,
                json=metrics_body,
                timeout=30,
            )
            metrics_response.raise_for_status()
            result_data = metrics_response.json()

            print(f"Got pairwise metrics: {str(result_data)[:500]}...")
            judge_result = self._parse_judge_results(result_data)

            # Cache the result
            self._judge_cache[cache_key] = judge_result
            return judge_result

        except Exception as e:
            print(f"Error running judge: {e}")
            # Return default result on error
            default_result = JudgeResult(
                correctness=0.5,
                completeness=0.5,
                grounding=0.5,
                safety=0.5,
                rationale=f"ERROR: {str(e)}"
            )
            self._judge_cache[cache_key] = default_result
            return default_result

    def _parse_judge_results(self, result_data: Dict[str, Any]) -> JudgeResult:
        """Parse pairwise metrics API response into JudgeResult.

        The pairwise metrics endpoint returns comparison metrics between base (teacher) and test (student) eval runs.
        """
        try:
            # The pairwise metrics response structure - adjust based on actual API
            # Typical structure might be:
            # {
            #   "metrics": {...},
            #   "winRate": ...,
            #   "averageScores": {...},
            #   etc.
            # }

            # Try different possible keys for the metrics
            metrics = result_data.get("metrics", result_data.get("averageScores", {}))

            # Parse scores - try common key patterns
            correctness = float(
                metrics.get("correctness") or
                metrics.get("averageCorrectness") or
                result_data.get("correctness") or
                result_data.get("winRate", 0.5)
            )

            # For now, use correctness as proxy for other metrics until we see actual response
            completeness = float(metrics.get("completeness", correctness))
            grounding = float(metrics.get("grounding", correctness))
            safety = float(metrics.get("safety", 0.9))  # Default to high safety

            # Build rationale from available data
            rationale_parts = []
            if "winRate" in result_data:
                rationale_parts.append(f"Win rate: {result_data['winRate']}")
            if "ties" in result_data:
                rationale_parts.append(f"Ties: {result_data['ties']}")
            rationale = "; ".join(rationale_parts) if rationale_parts else "Pairwise comparison completed"

            return JudgeResult(
                correctness=correctness,
                completeness=completeness,
                grounding=grounding,
                safety=safety,
                rationale=rationale
            )
        except Exception as e:
            print(f"Error parsing pairwise metrics: {e}")
            print(f"Full response: {result_data}")
            return JudgeResult(
                correctness=0.5,
                completeness=0.5,
                grounding=0.5,
                safety=0.5,
                rationale=f"Parse error: {str(e)}"
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

@dataclass
class CandidateEval:
    candidate_id: str
    summary: EvalSummary
    results: Optional[List[EvalResult]] = None  # only present if capture_traces True or for reflection

class AssistantALAdapter:
    def __init__(
        self,
        runner: ALRunner,
        judge: Judge,
        teacher_cache: TeacherCache,
        teacher_model: str,
        thresholds: Thresholds,
        eval_set_name: str,
        eval_set_version: str,
        student_model: str,
    ):
        self.runner = runner
        self.judge = judge
        self.teacher_cache = teacher_cache
        self.teacher_model = teacher_model
        self.thresholds = thresholds
        self.eval_set_name = eval_set_name
        self.eval_set_version = eval_set_version
        self.student_model = student_model

        # module freezing memory: module -> count of "not relevant" in consecutive generations
        self._module_irrelevant_streak: Dict[Tuple[str, str], int] = defaultdict(int)  # (candidate_family, module) -> streak

        # good options pool per module (to avoid losing good parts)
        self.good_module_options: Dict[str, List[str]] = defaultdict(list)

        # Cache for evaluated candidates
        self._candidate_cache: Dict[str, CandidateEval] = {}

    def evaluate(
        self,
        batch: list[ALDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> CandidateEval:
        """Evaluate candidate on eval set.

        Args:
            batch: Ignored (for GEPA compatibility)
            candidate: Flattened candidate dict[str, str]
            capture_traces: Whether to capture traces for reflection

        Returns:
            CandidateEval with summary and optional results
        """
        # Create candidate hash for caching
        cand_hash = hashlib.md5(json.dumps(candidate, sort_keys=True).encode()).hexdigest()
        cache_key = f"{cand_hash}_{self.eval_set_name}_{self.eval_set_version}"

        if cache_key in self._candidate_cache:
            return self._candidate_cache[cache_key]

        # Compile system prompt from candidate
        system_prompt = compile_encoded_prompt(candidate)

        # Run teacher eval set (cached)
        teacher_cache_key = f"teacher_{self.eval_set_name}_{self.eval_set_version}"
        if not self.teacher_cache._cache.get(teacher_cache_key):
            teacher_trace = self.runner.run(
                self.teacher_model,
                system_prompt="<<TEACHER_PROD_PROMPT>>",
                env={"eval_set_name": self.eval_set_name, "eval_set_version": self.eval_set_version}
            )
            self.teacher_cache.put(teacher_cache_key, teacher_trace)

        # Run student eval set
        student_trace = self.runner.run(
            self.student_model,
            system_prompt=system_prompt,
            env={"eval_set_name": self.eval_set_name, "eval_set_version": self.eval_set_version}
        )

        # Get eval run IDs for judge
        teacher_eval_id = self.runner.get_eval_run_id(
            self.teacher_model, "<<TEACHER_PROD_PROMPT>>", self.eval_set_name, self.eval_set_version
        )
        student_eval_id = self.runner.get_eval_run_id(
            self.student_model, system_prompt, self.eval_set_name, self.eval_set_version
        )

        # Run judge to compare teacher vs student
        judge_result = self.judge.judge(teacher_eval_id, student_eval_id)

        # Create summary from judge results
        summary = EvalSummary(
            quality=judge_result.correctness,
            tool_align=0.8,  # Placeholder - judge doesn't provide this
            loops=1,  # Placeholder
            tokens=1000,  # Placeholder
            fail_rate=1.0 - judge_result.correctness,
            avg_tool_errors=0,  # Placeholder
        )

        result = CandidateEval(candidate_id=cand_hash, summary=summary, results=None)
        self._candidate_cache[cache_key] = result
        return result

    # ---------------------------
    # 5) High-signal reflective dataset selection (per module)
    # ---------------------------

    def make_reflective_dataset(
        self,
        candidate: Candidate,
        executions: List[EvalResult],
        module_name: str,
        k: int = 8,
    ) -> List[EvalResult]:
        """
        Doc requirement:
          - small set of high-signal examples where candidate did clearly worse than teacher
          - per module per candidate
        """
        # Define "clearly worse" vs teacher by combining:
        #  - low quality score (judge)
        #  - tool mismatch (esp for TOOL_ROUTING)
        #  - formatting/response guideline mismatches (approx via judge dimensions/tags)
        def module_relevance_score(r: EvalResult) -> float:
            if module_name.startswith("TOOL_USAGE"):
                # Tool usage modules: focus on tool selection and alignment
                return float(r.tags["primary_tool_mismatch"]) * 2.0 + (1.0 - r.tool_align_score)
            if module_name == "FORMATTING":
                # Formatting: focus on completeness, grounding, tool errors
                return float(r.judge.completeness < 0.75) + float(r.judge.grounding < 0.75) + float(r.tags["tool_errors"])
            if module_name == "PERSISTENCE":
                # Persistence: focus on correctness, grounding, exploration
                return float(r.judge.correctness < 0.75) + float(r.judge.grounding < 0.75) + float(r.loops < 2) * 0.2
            if module_name == "GLOBAL_ROLE":
                # Global role: focus on tone, approach, completeness, safety
                return float(r.judge.completeness < 0.75) + float(r.judge.safety < 0.85)
            return 1.0 - r.quality_score

        # rank by "badness" + module relevance
        ranked = sorted(
            executions,
            key=lambda r: (-(module_relevance_score(r)), r.quality_score),
        )

        # diversity: avoid picking all the same failure tag
        picked: List[EvalResult] = []
        seen_sig = set()
        for r in ranked:
            # Build a coarse signature for failure mode
            sig = (
                int(r.tags.get("primary_tool_mismatch", False)),
                int(r.tags.get("low_correctness", False)),
                int(r.tags.get("low_grounding", False)),
                int(r.tags.get("tool_errors", False)),
                int(r.tags.get("over_budget", False)),
            )
            if sig in seen_sig and len(picked) < k - 2:
                continue
            picked.append(r)
            seen_sig.add(sig)
            if len(picked) >= k:
                break

        return picked

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
        module_name: str,
        reflective_examples: List[EvalResult],
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
        current = candidate.prompt_modules.get(module_name, "")

        # Build reflective dataset payload (minimal but high-signal)
        ex_blocks = []
        for r in reflective_examples:
            ex_blocks.append(
                f"---\n"
                f"QUERY: {r.student.answer and ''}\n"  # you likely want original query here; keep placeholder minimal
                f"TEACHER_ANSWER: {r.teacher.answer}\n"
                f"STUDENT_ANSWER: {r.student.answer}\n"
                f"TEACHER_TOOLS: {[e.tool_type for e in r.teacher.tool_events]}\n"
                f"STUDENT_TOOLS: {[e.tool_type for e in r.student.tool_events]}\n"
                f"JUDGE: correctness={r.judge.correctness:.2f}, completeness={r.judge.completeness:.2f}, grounding={r.judge.grounding:.2f}, safety={r.judge.safety:.2f}\n"
                f"RATIONALE: {r.judge.rationale}\n"
            )

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

# ---------------------------
# 9) Candidate generation: single-module edits + crossover + good pool
# ---------------------------

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
