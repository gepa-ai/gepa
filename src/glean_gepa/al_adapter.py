from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from datetime import time as datetime_time
from typing import Any, Callable, NotRequired, TypedDict, cast

from gepa.core.adapter import EvaluationBatch
from glean_gepa.adapter_types import (
    ALDataInst,
    ALRolloutOutput,
    ALTrajectory,
)
from glean_gepa.batch import EvalRunIds, GleanEvaluationBatch
from glean_gepa.debug import debug_print
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.reflection_sampling import deduplicate_reflective_examples
from glean_gepa.shell_tool_error_util import (
    EvalRunShellToolErrorAnalysis,
    parse_shell_tool_error_entry_metrics,
    parse_shell_tool_error_metrics,
)

EVAL_ANALYSIS_CACHE_SCHEMA_VERSION = 9


def _write_json_atomically(path: str, data: Any) -> None:
    """Replace a JSON cache only after a complete same-directory write."""
    cache_path = os.path.abspath(path)
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=cache_dir or ".", delete=False) as temp_file:
            temp_path = temp_file.name
            json.dump(data, temp_file, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, cache_path)
    except Exception:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise


# The scParams backing the Coding Harness evalcli preset. Keep nested values
# percent-encoded: scParams itself is comma-delimited, so decoding them would
# change the meaning of the nested agentic-loop configuration.
CODING_HARNESS_SC_PARAMS = (
    "co.lo.mo.slwo.disabled=1,"
    "ro.scholastic_required=true,"
    "db.disable_usr=true,"
    "db.filter_query_debug_results=true,"
    "db.filter_bad_query_jiras=true,"
    "db.include_final_scores=true,"
    "ro.ro.fetch_supplemental_results=false,"
    "db.ranking_only=true,"
    "db.debug_mode=1,"
    "co.debug_only_disabled_tools_list=gmail_search;outlook_search;respond;think;curl,"
    "db.get_doc_metadata=true,"
    "co.lo.enable_agent_recommendation=false,"
    "ro.feso.slso.drop_slack_native=false,"
    "ro.feso.slso.rts_count=0,"
    "ro.feso.slso.skip_inline_rts=true,"
    "co.disable_full_document_content=true,"
    "wo.plan_only_dry_run_for_write_actions=true,"
    "wo.plan_only_dry_run_for_write_actions_in_actas=true,"
    "co.internal_looping_pyagent_default_route_override=coding_agent_loop,"
    "co.lo.cao.agentic_loop_sc_params=co.so.enable_for_agentic_loop%3D1%2C"
    "co.so.enable_programmatic_tool_calling%3D1%2C"
    "co.so.ptc_allowed_tools%3D_none%2C"
    "co.so.ptc_only_tools%3Dglean_search%253Bglean_document_reader%253Buser_activity_retrieve%253B"
    "email_search_v2%253Bmeeting_lookup%253Bdiscover%253Btodo_write%253Bemployee_search%253B"
    "ask_user_questions%253Bcode_search%253Bimage_generation%253Bcode_repository_agent%253B"
    "web_search%253Bretrieve_personalized_writing_context%253Bcreate_image_collection%253Bcreate_ppt%253B"
    "create_presentation_pdf%253Bsalesforce_context_selector%253Bjira_schema_reader%2C"
    "co.so.enable_dynamic_tools_in_ptc%3D1%2C"
    "co.lo.enable_todo%3D1%2C"
    "co.enable_skill_reader_for_o3_agentic_loop%3D1%2C"
    "co.cito.enable_mcp_citations_prompt%3D1%2C"
    "co.lo.enable_artifacts%3Dtrue%2C"
    "co.lo.artifact_per_type_skills%3Dtrue%2C"
    "co.so.enable_approval_required_tools_in_ptc%3D1%2C"
    "acto.enable_preview_and_cta_in_hitl_banner%3D1%2C"
    "co.so.enable_ptc_tool_output_to_afs%3D1%2C"
    "co.lo.enable_ask_user_questions%3D1%2C"
    "co.use_discovery_layer%3D1%2C"
    "co.so.enable_dev_message_at_max_turns%3Dtrue%2C"
    "co.so.compact_max_chars%3D50000%2C"
    "co.so.compact_max_chars_per_tool%3Dglean_document_reader%3A320000%3Bglean_search%3A160000%3B"
    "code_search%3A160000%3Bemail_search_v2%3A160000%3Buser_activity_retrieve%3A160000%3B"
    "meeting_lookup%3A160000%2C"
    "co.so.upload_llm_rendered_to_afs%3Dtrue%2C"
    "co.lo.cao.use_auto_mode%3Dtrue%2C"
    "co.lo.cao.use_knowledge_tool%3D0%2C"
    "co.lo.suppress_shell_step_when_detailed_message%3D1%2C"
    "co.so.omit_file_path_when_no_compaction%3Dtrue%2C"
    "co.enable_run_tool_action_summary_event_response%3D1%2C"
    "co.lo.cao.bare_tool_configs%3DCode%2520Search%253ASearch%2520company%2520code%252C%2520configs%252C%2520schemas%252C%2520commits%252C%2520and%2520snippets.%2520Primary%2520tool%2520for%2520any%2520technical%2520question%2520about%2520implementation%2520ground%2520truth%252C%2520system%2520behavior%252C%2520code%2520locations%252C%2520bugs%252C%2520errors%252C%2520config%2520keys%252C%2520APIs%252C%2520data%2520models%252C%2520feature%2520flags%252C%2520or%2520how%252Fwhere%252Fwhy%2520something%2520is%2520implemented.%253BUser%2520Activity%2520Retrieve%253ARetrieves%2520all%2520cross-app%2520work%2520activities%2520over%2520a%2520date%2520range%253BMap%253ARuns%2520parallel%2520subtasks%2520across%2520multiple%2520inputs.%2520Use%2520sparingly.%253BCode%2520Repository%2520Agent%253AMakes%2520code%2520changes%2520and%2520creates%2520PRs%253BEmail%2520Search%2520V2%253ASearches%2520over%2520the%2520user%2527s%2520emails%253BSandbox%2520File%2520View%253AView%2520image%2520files%2520%28.jpg%252F.jpeg%252F.png%29%2520from%2520the%2520sandbox%2520for%2520visual%2520inspection%253BData%2520Analysis%253AUse%2520for%2520ANY%2520SQL%252C%2520analytics%252C%2520BI%252C%2520metrics%252C%2520KPI%252C%2520reporting%252C%2520pipeline%252FCRM%252C%2520or%2520data-lookup%2520query.%2520Returns%2520a%2520routing%2520playbook%2520across%2520the%2520connected%2520data%2520sources.%2520Always%2520run%2520it%2520first%252C%2520before%2520drilling%2520into%2520a%2520specific%2520data%2520source.%2520It%2520requires%2520zero%2520args%2520%25E2%2580%2594%2520just%2520use%2520%2560print%2528asyncio.run%2528data_analysis%2528%2529%2529%2529%2560%2520directly%2520%2528skip%2520%2560help%2560for%2520this%2520tool%2529.%2C"
    "co.lo.warm_start_discover_skills_header_override%3DSuggested%2520skills%2520%2528read%2520only%2520if%2520CLEARLY%2520relevant%2529%253A%2C"
    "llmo.per_prompt_overrides.intermediary_updates_instructions%3DIyMjIEludGVybWVkaWFyeSBVcGRhdGVzCldoaWxlIHdvcmtpbmcsIGVtaXQgQlJJRUYgYW5kIHNpbXBsZSAoMS0yIHNlbnRlbmNlKSBzdGF0dXMgdXBkYXRlcyBhdCBiaWcgbWlsZXN0b25lcy4KUnVsZXM6Ci0gVXBkYXRlcyBhcmUgb2JzZXJ2YXRpb25hbCwgbm90IGRlY2lzaW9uYWwuIERvIG5vdCB1c2UgdXBkYXRlcyB0byBkZXRlcm1pbmUgd2hhdCB0byBkbyBuZXh0LgotIERvIG5vdCBuYXJyYXRlIHRvb2wgY2FsbCBvciBza2lsbCB1c2FnZS4gVXBkYXRlIG9ubHkgd2hlbiB5b3UgZm91bmQgc3Vic3RhbnRpYWwgaW5mb3JtYXRpb24uCi0gQWZ0ZXIgY29tcGxldGluZyBhIHNlYXJjaCBvciB0b29sIGNhbGwsIGNoZWNrIHdoZXRoZXIgdGhlIHVzZXIncyBvcmlnaW5hbCByZXF1ZXN0IGlzIGZ1bGx5IHNhdGlzZmllZCBiZWZvcmUgY29uc2lkZXJpbmcgdGhlIHRhc2sgZG9uZS4gRmluZGluZyBpbmZvcm1hdGlvbiBpcyBub3QgdGhlIHNhbWUgYXMgZGVsaXZlcmluZyB0aGUgZmluYWwgb3V0cHV0Lg%3D%3D%2C"
    "co.lo.cao.hide_shell_tool_output_file_path%3Dtrue%2C"
    "co.lo.cao.enable_split_tool_sdk%3D1%2C"
    "co.use_gateway_for_discovery%3D1%2C"
    "co.lo.cao.bare_tools_redlist%3Dsalesforce_context_selector%253Bjira_schema_reader"
)


def log_shell_tool_error_analysis(analysis: EvalRunShellToolErrorAnalysis) -> None:
    """Log the fetched shell-tool error rate and recent error details."""
    aggregate = analysis.aggregate
    print(
        f"[Shell Tool] Fetched error rate for eval {analysis.eval_id}: "
        f"{aggregate.shell_error_pct:.2f}% "
        f"({aggregate.shell_errors}/{aggregate.shell_executions})"
    )
    for example in aggregate.recent_error_examples:
        if example.action_input:
            debug_print(f"[Shell Tool] Action input for eval {analysis.eval_id}: {example.action_input}")
        if example.error_str:
            debug_print(f"[Shell Tool] Error for eval {analysis.eval_id}: {example.error_str}")


# ---------------------------
# 1) Prompt modules + candidate
# ---------------------------

MODULES = [
    "WRITING_CODE",
]


@dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    kind: str  # "free_text" | "enum_knob"
    token_budget: int


@dataclass
class Candidate:
    model: str  # "claude" | "gemini" etc
    prompt_modules: dict[str, str]  # Single editable key: {"WRITING_CODE": "..."}
    module_specs: dict[str, ModuleSpec]
    global_token_cap: int  # relative to baseline prompt for that model
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
    eval_run_id: NotRequired[str]
    eval_trace_id: NotRequired[str]


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
ReflectiveExample = TypedDict(
    "ReflectiveExample",
    {
        "Inputs": ReflectiveExampleInputs,
        "Generated Outputs": ReflectiveExampleOutputs,
        "Action Inputs": list[str],
        "Execution Errors": list[str],
        "Feedback": str,
        "Metrics": ReflectiveExampleMetrics,
    },
)

EvaluateFn = Callable[[list[ALDataInst], dict[str, str], bool], GleanEvaluationBatch]
FailurePatternFn = Callable[[str, Any], tuple[Any, ...]]
ReflectiveExampleFn = Callable[[str, Any, dict[str, str]], ReflectiveExample]
ReflectionPromptFn = Callable[[str], str]
ReflectiveMetricsFn = Callable[[ReflectiveExampleMetrics], str | None]


@dataclass
class JudgeResult:
    correctness: float  # 0..1
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
        eval_run_timeout_sec: int | None = None,
    ):
        self.evalcli = evalcli
        self.deployment_ids = deployment_ids or ["scio-prod"]
        self.cache_file = os.path.expanduser(cache_file) if cache_file else None
        self.eval_run_timeout_sec = eval_run_timeout_sec
        self._cache_lock = threading.RLock()

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
            with self._cache_lock:
                if not os.path.exists(self.cache_file):
                    return
                with open(self.cache_file) as f:
                    data = json.load(f)
                # Convert string keys back to tuples
                self._eval_run_ids = {tuple(json.loads(k)): v for k, v in data.items()}
                print(f"Loaded {len(self._eval_run_ids)} eval run IDs from cache")
        except Exception as e:
            print(f"Failed to load cache from {self.cache_file}: {e}")
            self._eval_run_ids = {}

    def _save_cache(self) -> None:
        """Save eval run IDs cache to file."""
        if not self.cache_file:
            return

        try:
            with self._cache_lock:
                # Convert tuple keys to strings for JSON serialization
                data = {json.dumps(list(k)): v for k, v in self._eval_run_ids.items()}
                _write_json_atomically(self.cache_file, data)
                print(f"Saved {len(self._eval_run_ids)} eval run IDs to cache")
        except Exception as e:
            print(f"Failed to save cache to {self.cache_file}: {e}")

    def _build_sc_params(self, model: str, system_prompt: str) -> str:
        """Build scParams based on model type."""
        # Start with the exact Coding Harness preset. It is a single string
        # because nested values intentionally contain encoded commas.
        base_params = [CODING_HARNESS_SC_PARAMS]

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
        on_created: Callable[[str], None] | None = None,
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
            on_created: Optional callback invoked after the eval run is created and cached,
                but before polling for completion.

        Returns:
            eval_run_id string
        """
        system_prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:16]
        cache_key = (model, system_prompt_hash, eval_set_name, eval_set_version, run_label)

        with self._cache_lock:
            if cache_key in self._eval_run_ids:
                cached_eval_id = self._eval_run_ids[cache_key]
                print(
                    f"[Eval run cache HIT] Using cached student eval_id: {cached_eval_id} "
                    f"for {eval_set_name}:{eval_set_version} ({run_label})"
                )
                return cached_eval_id

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
        with self._cache_lock:
            self._eval_run_ids[cache_key] = created_id
            self._save_cache()
        print(
            f"[Eval run cache] Cached student eval_id at creation: {created_id} "
            f"for {eval_set_name}:{eval_set_version} ({run_label})"
        )
        if on_created is not None:
            on_created(created_id)

        if self.eval_run_timeout_sec is None:
            self.evalcli.wait_for_eval_run(created_id)
        else:
            self.evalcli.wait_for_eval_run(created_id, timeout_sec=self.eval_run_timeout_sec)
        print(f"Eval run {created_id} completed successfully")

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
        with self._cache_lock:
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
            print(
                f"Skipping judge trigger (already triggered), fetching results for {teacher_eval_id} vs {student_eval_id}."
            )

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
            entry_details.append(
                {"deploymentId": deployment_id, "entryId": entry_id, "evalRunIds": [student_eval_id, teacher_eval_id]}
            )
            for eval_run_entry in entry.get("evalRunEntries", []):
                eval_run_id = eval_run_entry.get("evalRunId")
                durations_map[(entry_id, eval_run_id)] = eval_run_entry.get("duration", 0)
                loop_counts_map[(entry_id, eval_run_id)] = eval_run_entry.get("metadata", {}).get("loopCount", 0)
                input_tokens_map[(entry_id, eval_run_id)] = eval_run_entry.get("metadata", {}).get(
                    "uncachedInputTokens", 0
                )
                output_tokens_map[(entry_id, eval_run_id)] = eval_run_entry.get("metadata", {}).get("outputTokens", 0)
                tools_invocations_map[(entry_id, eval_run_id)] = eval_run_entry.get("metadata", {}).get(
                    "toolsInvoked", []
                )

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
                        print(f"Get a none correctness score for trace {trace_info.get('eval_id')}")
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
def get_tool_alignment_from_traces(
    student_trace_spans: list[dict[str, Any]], teacher_trace_spans: list[dict[str, Any]]
) -> float:
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


def _typed_str_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        str_value = value.get("strValue")
        return str_value if isinstance(str_value, str) else None
    return None


def extract_shell_action_inputs(detailed_trace: dict[str, Any]) -> dict[str, str]:
    """Map Shell action run IDs to their serialized action inputs."""
    action_inputs: dict[str, str] = {}
    spans = detailed_trace.get("trace", {}).get("spans") or []
    for span in spans:
        if span.get("name") not in ("Execute Action: Shell", "Execute Action: Shell Tool"):
            continue
        attributes = span.get("attributes") or {}
        raw_input = _typed_str_value(attributes.get("input"))
        raw_gle = _typed_str_value(attributes.get("span.gle"))
        if not raw_input or not raw_gle:
            continue
        try:
            input_payload = json.loads(raw_input)
            gle_payload = json.loads(raw_gle)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        action_input = input_payload.get("action_input")
        action_run_id = (gle_payload.get("action") or {}).get("action_run_id")
        if isinstance(action_run_id, str) and isinstance(action_input, str) and action_input:
            action_inputs[action_run_id] = action_input
    return action_inputs


def _timestamp_millis(value: str | None) -> int | None:
    if not value:
        return None
    normalized = value.replace(" UTC", "+00:00")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def enrich_shell_error_action_inputs(
    evalcli: EvalCliClient,
    analysis: EvalRunShellToolErrorAnalysis,
) -> EvalRunShellToolErrorAnalysis:
    """Fetch detailed traces and attach serialized Shell inputs to failed actions."""
    examples = list(analysis.aggregate.recent_error_examples)
    for metrics in analysis.per_entry.values():
        examples.extend(metrics.recent_error_examples)

    grouped: defaultdict[tuple[str, str], list[Any]] = defaultdict(list)
    for example in examples:
        if example.project_id and example.trace_id and example.action_run_id and not example.action_input:
            grouped[(example.project_id, example.trace_id)].append(example)
    if not grouped:
        return analysis

    resolved: dict[tuple[str, str, str], str] = {}
    for (deployment_id, trace_id), trace_examples in grouped.items():
        timestamps = [
            timestamp
            for example in trace_examples
            for timestamp in [_timestamp_millis(example.started_at)]
            if timestamp is not None
        ]
        if timestamps:
            start_time_millis = min(timestamps) - int(timedelta(hours=1).total_seconds() * 1000)
            end_time_millis = max(timestamps) + 1000
        else:
            start_dt = datetime.combine(analysis.start_date, datetime_time.min, tzinfo=timezone.utc)
            end_dt = datetime.combine(analysis.end_date + timedelta(days=1), datetime_time.min, tzinfo=timezone.utc)
            start_time_millis = int(start_dt.timestamp() * 1000)
            end_time_millis = int(end_dt.timestamp() * 1000)
        try:
            detailed_trace = evalcli.get_analysis_trace(
                deployment_id=deployment_id,
                trace_id=trace_id,
                start_time_millis=start_time_millis,
                end_time_millis=end_time_millis,
            )
        except Exception as exc:
            print(f"[Shell Tool] Failed to fetch action inputs for trace {trace_id}: {exc}")
            continue
        for action_run_id, action_input in extract_shell_action_inputs(detailed_trace).items():
            resolved[(deployment_id, trace_id, action_run_id)] = action_input

    def enrich_example(example: Any) -> Any:
        if example.action_input or not (example.project_id and example.trace_id and example.action_run_id):
            return example
        action_input = resolved.get((example.project_id, example.trace_id, example.action_run_id))
        return replace(example, action_input=action_input) if action_input else example

    if not resolved:
        return analysis

    aggregate = replace(
        analysis.aggregate,
        recent_error_examples=tuple(enrich_example(example) for example in analysis.aggregate.recent_error_examples),
    )
    per_entry = {
        entry_id: replace(
            metrics,
            recent_error_examples=tuple(enrich_example(example) for example in metrics.recent_error_examples),
        )
        for entry_id, metrics in analysis.per_entry.items()
    }
    return replace(analysis, aggregate=aggregate, per_entry=per_entry)


# ---------------------------
# 4) Shared adapter internals
# ---------------------------


@dataclass
class Thresholds:
    quality_min: float
    tools_min: float
    max_student_tokens: int


class GleanAdapterBase:
    supports_high_signal_eval = False
    def __init__(
        self,
        runner: ALRunner,
        thresholds: Thresholds,
        student_model: str,
        *,
        evaluate_fn: EvaluateFn,
        failure_pattern_fn: FailurePatternFn,
        reflective_example_fn: ReflectiveExampleFn,
        reflection_prompt_fn: ReflectionPromptFn,
        reflective_metrics_fn: ReflectiveMetricsFn,
        failure_label: str,
        primary_objective: str,
        default_frontier_type: str,
        cache_file: str | None = None,
    ):
        self.runner = runner
        self.thresholds = thresholds
        self.student_model = student_model
        self.primary_objective = primary_objective
        self.default_frontier_type = default_frontier_type
        self.cache_file = os.path.expanduser(cache_file) if cache_file else None
        self._cache_lock = threading.RLock()
        self._evaluate_fn = evaluate_fn
        self._failure_pattern_fn = failure_pattern_fn
        self._reflective_example_fn = reflective_example_fn
        self._reflection_prompt_fn = reflection_prompt_fn
        self._reflective_metrics_fn = reflective_metrics_fn
        self._failure_label = failure_label

        # module freezing memory: module -> count of "not relevant" in consecutive generations
        self._module_irrelevant_streak: dict[tuple[str, str], int] = defaultdict(
            int
        )  # (candidate_family, module) -> streak

        # good options pool per module (to avoid losing good parts)
        # TODO(Cathy) populate the good_module_options
        self.good_module_options: dict[str, list[str]] = defaultdict(list)

        # These caches are keyed by the immutable eval run ID so they remain useful
        # when the adapter cache is loaded in a later process.
        self._eval_analysis_cache: dict[str, EvalRunShellToolErrorAnalysis] = {}

        # Judge triggered cache: (teacher_eval_id, student_eval_id) -> triggered
        self._judge_triggered: set[tuple[str, str]] = set()

        # Load cache if file exists
        if self.cache_file:
            self._load_cache()

    def _load_cache(self) -> None:
        """Load analysis and judge-trigger state from the adapter cache."""
        if not self.cache_file:
            return

        try:
            with self._cache_lock:
                if not os.path.exists(self.cache_file):
                    return
                with open(self.cache_file) as f:
                    data = json.load(f)

                # Load judge triggered cache
                judge_triggered_data = data.get("judge_triggered", [])
                self._judge_triggered = {tuple(item) for item in judge_triggered_data}

                self._load_eval_analysis_cache(data.get("eval_analysis_cache", {}))
                print(
                    f"[GleanAdapter] Loaded {len(self._eval_analysis_cache)} error analyses and "
                    f"{len(self._judge_triggered)} judge triggers from cache: {self.cache_file}"
                )
        except Exception as e:
            print(f"[GleanAdapter] Failed to load cache from {self.cache_file}: {e}")
            self._judge_triggered = set()
            self._eval_analysis_cache = {}

    def _save_cache(self) -> None:
        """Save analysis and judge-trigger state to the adapter cache."""
        if not self.cache_file:
            return

        try:
            with self._cache_lock:
                # Build cache data structure
                data = {
                    "judge_triggered": [list(pair) for pair in self._judge_triggered],
                    "eval_analysis_cache": {
                        eval_id: self._serialize_eval_analysis(analysis)
                        for eval_id, analysis in self._eval_analysis_cache.items()
                    },
                }
                _write_json_atomically(self.cache_file, data)
                print(
                    f"[GleanAdapter] Saved {len(self._eval_analysis_cache)} error analyses and "
                    f"{len(self._judge_triggered)} judge triggers to cache: {self.cache_file}"
                )
        except Exception as e:
            print(f"[GleanAdapter] Failed to save cache to {self.cache_file}: {e}")

    @staticmethod
    def _serialize_eval_analysis(analysis: EvalRunShellToolErrorAnalysis) -> dict[str, Any]:
        def metrics_dict(metrics: Any) -> dict[str, Any]:
            return {
                "eval_id": getattr(metrics, "eval_id", None),
                "entry_id": getattr(metrics, "entry_id", None),
                "shell_executions": metrics.shell_executions,
                "shell_errors": metrics.shell_errors,
                "shell_error_rate": metrics.shell_error_rate,
                "shell_error_pct": metrics.shell_error_pct,
                "trace_ids": list(getattr(metrics, "trace_ids", ())),
                "session_tracking_tokens": list(getattr(metrics, "session_tracking_tokens", ())),
                # Preserve the trace error details because they are diagnostic
                # input for the reflection model, not merely logging metadata.
                "recent_error_examples": [asdict(example) for example in metrics.recent_error_examples],
            }

        return {
            "schema_version": EVAL_ANALYSIS_CACHE_SCHEMA_VERSION,
            "eval_id": analysis.eval_id,
            "start_date": analysis.start_date.isoformat(),
            "end_date": analysis.end_date.isoformat(),
            "aggregate": metrics_dict(analysis.aggregate),
            "per_entry": {entry_id: metrics_dict(metrics) for entry_id, metrics in analysis.per_entry.items()},
            "high_signal_entry_ids": list(analysis.high_signal_entry_ids),
        }

    def _load_eval_analysis_cache(self, raw_cache: Any) -> None:
        if not isinstance(raw_cache, dict):
            return
        for eval_id, raw in raw_cache.items():
            try:
                if not isinstance(raw, dict):
                    continue
                if raw.get("schema_version") != EVAL_ANALYSIS_CACHE_SCHEMA_VERSION:
                    print(f"[Cache] Refreshing legacy shell error analysis for eval_id: {eval_id}")
                    continue
                aggregate = parse_shell_tool_error_metrics(raw["aggregate"])
                if aggregate.shell_executions == 0:
                    print(f"[Cache] Refreshing provisional 0/0 shell analysis for eval_id: {eval_id}")
                    continue
                per_entry = {
                    entry_id: parse_shell_tool_error_entry_metrics(metrics)
                    for entry_id, metrics in (raw.get("per_entry") or {}).items()
                }
                self._eval_analysis_cache[str(eval_id)] = EvalRunShellToolErrorAnalysis(
                    eval_id=str(raw.get("eval_id") or eval_id),
                    start_date=date.fromisoformat(raw["start_date"]),
                    end_date=date.fromisoformat(raw["end_date"]),
                    aggregate=aggregate,
                    per_entry=per_entry,
                    high_signal_entry_ids=tuple(raw.get("high_signal_entry_ids") or ()),
                )
            except (KeyError, TypeError, ValueError):
                continue

    def evaluate(
        self,
        batch: list[ALDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> GleanEvaluationBatch:
        """Evaluate candidate on eval set(s)."""
        return self._evaluate_fn(batch, candidate, capture_traces)

    def batch_evaluate(
        self,
        items: list[tuple[dict[str, str], list[ALDataInst]]],
        *,
        capture_traces: bool = True,
    ) -> list[GleanEvaluationBatch]:
        """Run independent candidate evaluations concurrently, preserving input order."""
        if len(items) < 2:
            return [self.evaluate(batch, candidate, capture_traces=capture_traces) for candidate, batch in items]

        with ThreadPoolExecutor(max_workers=len(items)) as executor:
            futures = [
                executor.submit(self.evaluate, batch, candidate, capture_traces=capture_traces)
                for candidate, batch in items
            ]
            return [future.result() for future in futures]

    def prepare_high_signal_batch(self, batch: list[ALDataInst]) -> list[ALDataInst] | None:
        """Prepare a high-signal batch once before its child candidates are screened."""
        return batch

    def attach_cached_eval_run_ids(
        self, batch: list[ALDataInst], eval_run_ids: list[EvalRunIds]
    ) -> list[ALDataInst]:
        """Attach persisted eval IDs to matching eval-set items for reuse."""
        cached_by_eval_set = {
            (record["eval_set_name"], record["eval_set_version"]): record for record in eval_run_ids
        }
        attached: list[ALDataInst] = []
        for data in batch:
            record = cached_by_eval_set.get((data["eval_set_name"], data["eval_set_version"]))
            if record is None:
                attached.append(data)
                continue
            cached_data = dict(data)
            cached_data["cached_student_eval_run_id"] = record["student_eval_run_id"]
            if teacher_eval_run_id := record.get("teacher_eval_run_id"):
                cached_data["cached_teacher_eval_run_id"] = teacher_eval_run_id
            attached.append(cast(ALDataInst, cached_data))
        return attached

    def get_screening_score(self, eval_batch: GleanEvaluationBatch) -> float:
        """Return the objective selected by the concrete adapter for screening."""
        if eval_batch.summary is None:
            return float("-inf")
        return eval_batch.summary.get(self.primary_objective, float("-inf"))

    def high_signal_batch(self, eval_batch: GleanEvaluationBatch) -> list[ALDataInst]:
        """Return eval-set configs narrowed to the entries that failed for a parent."""
        grouped: dict[tuple[str, str, tuple[str, ...]], list[str]] = defaultdict(list)
        for trajectory in eval_batch.trajectories or []:
            data = trajectory["data"]
            output = trajectory["output"]
            if trajectory["score"] >= 1.0:
                continue
            key = (data["eval_set_name"], data["eval_set_version"], tuple(data["deployment_ids"]))
            entry_id = output.get("entry_id")
            if entry_id and entry_id not in grouped[key]:
                grouped[key].append(entry_id)

        return [
            {
                "eval_set_name": eval_set_name,
                "eval_set_version": eval_set_version,
                "deployment_ids": list(deployment_ids),
                "status": "active",
                "eval_entry_ids": entry_ids,
            }
            for (eval_set_name, eval_set_version, deployment_ids), entry_ids in grouped.items()
        ]

    def high_signal_fix_rate(
        self, parent_eval: GleanEvaluationBatch, child_eval: GleanEvaluationBatch
    ) -> float:
        """Fraction of focused entries that are error-free for the child."""
        parent_failure_count = sum(1 for trajectory in parent_eval.trajectories or [] if trajectory["score"] < 1.0)
        if not parent_failure_count:
            return 0.0
        primary_objective = getattr(self, "primary_objective", None)
        if primary_objective and child_eval.summary is not None:
            # Focused single-model evals calculate this from all requested
            # entries, including any entry missing from the trace query.
            return child_eval.summary.get(primary_objective, 0.0)
        child_trajectories = child_eval.trajectories or []
        if not child_trajectories:
            return 0.0
        fixed_count = sum(1 for trajectory in child_trajectories if trajectory["score"] >= 1.0)
        return fixed_count / len(child_trajectories)

    def _get_or_run_student_eval(
        self,
        *,
        eval_set_name: str,
        eval_set_version: str,
        deployment_ids: list[str],
        system_prompt: str,
        run_label: str = "gepa",
    ) -> str:
        return self.runner.run(
            self.student_model,
            system_prompt=system_prompt,
            eval_set_name=eval_set_name,
            eval_set_version=eval_set_version,
            deployment_ids=deployment_ids,
            run_label=run_label,
        )

    # ---------------------------
    # 5) High-signal reflective dataset selection (per module)
    # ---------------------------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[ALTrajectory, ALRolloutOutput],
        components_to_update: list[str],
        k: int | None,  # max return; None includes all examples
        error_hamming_distance_k: int | None = None,
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
                relevance = self._compute_module_relevance(component_name, trajectory, eval_batch.scores[idx])
                scored_trajectories.append((relevance, idx, trajectory))

            # Sort by relevance (higher = more relevant for improvement)
            # Then by score (lower = worse performance, needs more attention)
            scored_trajectories.sort(key=lambda x: (-x[0], x[2]["score"]))

            # Select diverse examples with poor performance
            reflective_examples = []
            seen_patterns = set()

            for _relevance, _idx, trajectory in scored_trajectories:
                # Create a pattern signature for diversity
                pattern = self._failure_pattern_fn(component_name, trajectory)

                # Allow some duplicates near the end to fill quota
                # TODO(Cathy) check with claude why it was < k-2
                if k is not None and pattern in seen_patterns and len(reflective_examples) > k - 3:
                    continue

                # Build reflective example in standard format
                example = self._reflective_example_fn(component_name, trajectory, candidate)

                reflective_examples.append(example)
                seen_patterns.add(pattern)

                if k is not None and len(reflective_examples) >= k:
                    break

            if error_hamming_distance_k is not None:
                before_dedupe = len(reflective_examples)
                reflective_examples = deduplicate_reflective_examples(
                    reflective_examples,
                    k=error_hamming_distance_k,
                    log=print,
                )
                removed = before_dedupe - len(reflective_examples)
                if removed:
                    print(
                        f"Reflection sampling removed {removed} near-duplicate {component_name} example(s) "
                        f"within Hamming distance {error_hamming_distance_k}."
                    )

            result[component_name] = reflective_examples

        return result

    # TODO(Cathy): Implement this
    def _compute_module_relevance(self, module_name: str, trajectory: ALTrajectory, score: float) -> float:
        """Compute how relevant this example is for improving the given module."""
        return 1.0

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
        Returns (new_module_texts, module_marked_irrelevant). The sole
        WRITING_CODE module is always relevant, so the second value is False.
        Must:
          1) merge reflections across examples
          2) diagnose recurring failure modes tied to module
          3) propose small patch with before/after
          4) brief why
        """
        current = candidate.prompt_modules.get(components_to_update[0], "")

        ex_blocks = []
        for r in reflective_examples:
            metric_line = self._reflective_metrics_fn(r["Metrics"])
            metric_section = f"METRICS: {metric_line}\n" if metric_line else ""
            generated_outputs = r["Generated Outputs"]
            output_lines = []
            if eval_trace_id := r["Inputs"].get("eval_trace_id"):
                output_lines.append(f"EVAL_TRACE_ID: {eval_trace_id}\n")
            for label, value in (
                ("TEACHER_ANSWER", generated_outputs["teacher_answer"]),
                ("STUDENT_ANSWER", generated_outputs["student_answer"]),
                ("TEACHER_TOOLS", generated_outputs["teacher_tools"]),
                ("STUDENT_TOOLS", generated_outputs["student_tools"]),
            ):
                if value:
                    output_lines.append(f"{label}: {value}\n")
            for action_input in r["Action Inputs"]:
                output_lines.append(f"ACTION_INPUT: {action_input}\n")
            ex_blocks.append(
                f"---\n"
                f"QUERY: {r['Inputs']['query']}\n"
                f"{''.join(output_lines)}"
                f"EXECUTION_ERRORS: {r['Execution Errors']}\n"
                f"{metric_section}"
                f"FEEDBACK: {r['Feedback']}\n"
            )
        if len(components_to_update) != 1:
            return [], False
        module_name = components_to_update[0]
        prompt = (
            f"You are optimizing ONLY the module {module_name}.\n"
            f"MODULE RESPONSIBILITY:\n{self._reflection_prompt_fn(module_name)}\n\n"
            f"CURRENT MODULE TEXT:\n<<<\n{current}\n>>>\n\n"
            f"{self._failure_label}:\n{''.join(ex_blocks)}\n\n"
            f"Task:\n"
            f"1) Identify recurring failure modes that are plausibly caused by {module_name}.\n"
            f"2) Propose 1-2 SMALL patches (delta edits), each with:\n"
            f"   - BEFORE: quoted snippet from current module\n"
            f"   - AFTER: revised snippet\n"
            f"   - WHY: one sentence\n"
            f"3) Every supplied example is relevant evidence for {module_name}; use it to propose a variant.\n"
            f"4) Make only generalizable changes; do not overfit to individual examples.\n"
            f"5) Keep the revised module succinct: each candidate must be strictly less than 1.1 times "
            f"the current module's character length.\n"
        )

        raw = reflection_llm(prompt).strip()
        if raw.upper() == "NOT_RELEVANT" or not raw:
            raw = (
                "No module-specific diagnosis was returned. Propose a conservative rewrite that addresses "
                "the supplied failure evidence while preserving the current instructions."
            )

        # Simple parser strategy:
        # In production, parse structured JSON/YAML, or ask LLM to output patches in JSON.
        # Here we just return one consolidated rewrite request for a second pass:
        consolidate_prompt = (
            f"Consolidate the following patch suggestions into up to {max_variants} candidate rewrites "
            f"of the module {module_name}. Preserve good behavior, incorporate consistent changes only, "
            f"and make only generalizable changes. Each variant must be succinct and strictly less than 1.1 "
            f"times the current module's character length. "
            f"Output each variant separated by '\n===VARIANT===\n'.\n\n"
            f"CURRENT:\n<<<\n{current}\n>>>\n\n"
            f"EVIDENCE (every example is relevant):\n{''.join(ex_blocks)}\n\n"
            f"SUGGESTIONS:\n{raw}\n"
        )
        consolidated = reflection_llm(consolidate_prompt).strip()
        variants = [
            variant
            for variant in (v.strip() for v in consolidated.split("===VARIANT==="))
            if variant and variant.upper() != "NOT_RELEVANT"
        ]
        return variants[:max_variants], False
