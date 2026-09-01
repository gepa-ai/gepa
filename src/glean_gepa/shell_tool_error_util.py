"""Shell-tool error metrics derived from Glean evaluation spans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence

DEFAULT_AGENTS_SPAN_TABLE = "scio-apps.scrubbed_agentspan.scrubbed_agentspan_*"
DEFAULT_EVALSET_ENTRIES_TABLE = "scio-apps.fact.evalset_entries"
DEFAULT_EVAL_WORKFLOW_RUNS_TABLE = "scio-apps.fact.eval_workflow_runs"
SHELL_SUCCESS_OBJECTIVE = "shell_success_rate"
SHELL_SPAN_NAMES = ("Execute Action: Shell", "Execute Action: Shell Tool")
SHELL_ACTION_IDS = ("Shell", "Shell Tool")
FAILED_PROVIDER_STATUSES = frozenset({"failed", "error"})


@dataclass(frozen=True)
class QueryParameter:
    name: str
    type_: str
    value: str | list[str]


@dataclass(frozen=True)
class ShellToolErrorExample:
    started_at: str | None
    project_id: str | None
    entry_id: str | None
    eval_id: str | None
    run_id: str | None
    trace_id: str | None
    span_id: str | None
    span_name: str | None
    action_id: str | None
    action_status: str | None
    span_status: str | None
    provider_status: str | None
    output_status_code: str | None
    error_str: str | None
    session_tracking_token: str | None = None
    action_run_id: str | None = None
    action_input: str | None = None


@dataclass(frozen=True)
class ShellToolErrorEntryMetrics:
    entry_id: str
    shell_executions: int
    shell_errors: int
    shell_error_rate: float
    shell_error_pct: float
    recent_error_examples: tuple[ShellToolErrorExample, ...]
    trace_ids: tuple[str, ...] = ()
    session_tracking_tokens: tuple[str, ...] = ()

    @property
    def shell_success_rate(self) -> float:
        return 1.0 - self.shell_error_rate

    @property
    def has_shell_error(self) -> bool:
        return self.shell_errors > 0


@dataclass(frozen=True)
class ShellToolErrorMetrics:
    eval_id: str
    shell_executions: int
    shell_errors: int
    shell_error_rate: float
    shell_error_pct: float
    recent_error_examples: tuple[ShellToolErrorExample, ...]

    @property
    def shell_success_rate(self) -> float:
        """Higher-is-better score for GEPA objective tracking."""
        return 1.0 - self.shell_error_rate


@dataclass(frozen=True)
class EvalRunShellToolErrorAnalysis:
    eval_id: str
    start_date: date
    end_date: date
    aggregate: ShellToolErrorMetrics
    per_entry: dict[str, ShellToolErrorEntryMetrics]
    high_signal_entry_ids: tuple[str, ...]


def is_shell_tool_error(
    *,
    action_status: str | None,
    span_status: str | None,
    output_status_code: str | None,
    provider_status: str | None,
) -> bool:
    provider = (provider_status or "").lower()
    return (
        action_status == "ERROR"
        or span_status == "ERROR"
        or output_status_code == "ERROR"
        or provider in FAILED_PROVIDER_STATUSES
    )


def _shell_span_filter_sql(table_alias: str = "") -> str:
    prefix = f"{table_alias}." if table_alias else ""
    span_names = ", ".join(f"'{name}'" for name in SHELL_SPAN_NAMES)
    action_ids = ", ".join(f"'{action_id}'" for action_id in SHELL_ACTION_IDS)
    return (
        f"({prefix}jsonPayload.span_info.span_name IN ({span_names}) "
        f"OR {prefix}jsonPayload.action.action_id IN ({action_ids}))"
    )


def _shell_spans_select_sql() -> str:
    shell_filter = _shell_span_filter_sql()
    return f"""
  SELECT
    jsonPayload.context.eval.eval_id AS eval_id,
    COALESCE(
      NULLIF(jsonPayload.action.action_run_id, ''),
      NULLIF(jsonPayload.context.agent_trace.span_id, ''),
      -- Some eval spans omit both IDs. Count the observed span instead of
      -- letting COUNT(DISTINCT NULL) turn a real execution into 0/0.
      TO_JSON_STRING(jsonPayload)
    ) AS shell_execution_id,
    jsonPayload.context.eval.entry_uuid AS entry_uuid,
    CAST(jsonPayload.context.eval.entry_id AS STRING) AS entry_id,
    resource.labels.project_id AS project_id,
    jsonPayload.context.workflow.run_id AS run_id,
    jsonPayload.context.agent_trace.trace_id AS trace_id,
    jsonPayload.span_info.session_info.session_tracking_token AS session_tracking_token,
    jsonPayload.context.agent_trace.span_id AS span_id,
    NULLIF(jsonPayload.action.action_run_id, '') AS action_run_id,
    jsonPayload.span_info.span_name AS span_name,
    jsonPayload.action.action_id AS action_id,
    jsonPayload.action.execution_status AS action_status,
    COALESCE(
      NULLIF(jsonPayload.action.error_str, ''),
      NULLIF(jsonPayload.span_info.execution_status.message, ''),
      NULLIF(jsonPayload.span_info.execution_status.user_message, '')
    ) AS error_str,
    jsonPayload.span_info.execution_status.code AS span_status,
    (
      SELECT o.value
      FROM UNNEST(jsonPayload.span_info.outputs) o
      WHERE o.name = 'status'
      LIMIT 1
    ) AS provider_status,
    (
      SELECT o.value
      FROM UNNEST(jsonPayload.span_info.outputs) o
      WHERE o.name = 'status_code'
      LIMIT 1
    ) AS output_status_code,
    SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64) AS start_ms
  FROM `{{agentspan_table}}`
  WHERE PARSE_DATE('%Y%m%d', _TABLE_SUFFIX)
    BETWEEN @start_date AND @end_date
    AND jsonPayload.context.eval.eval_id = @eval_id
    AND {shell_filter}
    AND jsonPayload.action.execution_mode = 'EXECUTE'
"""


def build_eval_run_time_bounds_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> str:
    """Find min/max shell span timestamps for an eval run inside the lookback window."""
    shell_filter = _shell_span_filter_sql()
    return f"""
SELECT
  MIN(SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64)) AS min_start_ms,
  MAX(SAFE_CAST(jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64)) AS max_start_ms
FROM `{agentspan_table}`
WHERE PARSE_DATE('%Y%m%d', _TABLE_SUFFIX)
  BETWEEN @search_start_date AND @search_end_date
  AND jsonPayload.context.eval.eval_id = @eval_id
  AND {shell_filter}
  AND jsonPayload.action.execution_mode = 'EXECUTE'
""".strip()


def build_shell_tool_error_per_entry_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
    entry_ids: Sequence[str] | None = None,
    include_error_examples: bool = True,
) -> str:
    """Build SQL for per-entry shell tool error metrics scoped to one eval run."""
    entry_filter = ""
    if entry_ids:
        entry_filter = "AND COALESCE(entry_id, entry_uuid) IN UNNEST(@entry_ids)"
    shell_spans = _shell_spans_select_sql().format(agentspan_table=agentspan_table)
    error_detail_columns = """
    , ARRAY_AGG(
      IF(
        is_error,
        STRUCT(
          TIMESTAMP_MILLIS(start_ms) AS started_at,
          project_id,
          entry_key AS entry_id,
          eval_id,
          run_id,
          trace_id,
          session_tracking_token,
          span_id,
          span_name,
          action_id,
          action_run_id,
          action_status,
          span_status,
          provider_status,
          output_status_code,
          error_str
        ),
        NULL
      )
      IGNORE NULLS
      ORDER BY start_ms DESC
      LIMIT 10
    ) AS recent_error_examples,
    ARRAY_AGG(
      IF(is_error, trace_id, NULL)
      IGNORE NULLS
      ORDER BY start_ms DESC
      LIMIT 10
    ) AS trace_ids,
    ARRAY_AGG(
      IF(is_error, session_tracking_token, NULL)
      IGNORE NULLS
      ORDER BY start_ms DESC
      LIMIT 10
    ) AS session_tracking_tokens
""" if include_error_examples else ""
    return f"""
WITH shell_spans AS (
{shell_spans}
),
classified AS (
  SELECT
    -- Prefer the explicit eval entry ID when present, otherwise use the
    -- trace-side eval entry UUID. Current eval runs populate entry_uuid.
    COALESCE(entry_id, entry_uuid) AS entry_key,
    *,
    (
      action_status = 'ERROR'
      OR span_status = 'ERROR'
      OR output_status_code = 'ERROR'
      OR LOWER(provider_status) IN ('failed', 'error')
    ) AS is_error
  FROM shell_spans
  WHERE COALESCE(entry_id, entry_uuid) IS NOT NULL
  {entry_filter}
),
per_entry AS (
  SELECT
    entry_key AS entry_id,
    COUNT(DISTINCT shell_execution_id) AS shell_executions,
    COUNT(DISTINCT IF(is_error, shell_execution_id, NULL)) AS shell_errors,
    SAFE_DIVIDE(
      COUNT(DISTINCT IF(is_error, shell_execution_id, NULL)),
      COUNT(DISTINCT shell_execution_id)
    ) AS shell_error_rate,
    ROUND(
      100 * SAFE_DIVIDE(
        COUNT(DISTINCT IF(is_error, shell_execution_id, NULL)),
        COUNT(DISTINCT shell_execution_id)
      ),
      2
    ) AS shell_error_pct{error_detail_columns}
  FROM classified
  GROUP BY entry_key
)
SELECT * FROM per_entry
ORDER BY shell_errors DESC, shell_executions DESC
""".strip()


def build_shell_tool_error_rate_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> str:
    """Build SQL to measure aggregate shell tool error rate for a single eval run."""
    shell_spans = _shell_spans_select_sql().format(agentspan_table=agentspan_table)
    return f"""
WITH shell_spans AS (
{shell_spans}
),
classified AS (
  SELECT
    *,
    (
      action_status = 'ERROR'
      OR span_status = 'ERROR'
      OR output_status_code = 'ERROR'
      OR LOWER(provider_status) IN ('failed', 'error')
    ) AS is_error
  FROM shell_spans
)
SELECT
  eval_id,
  COUNT(DISTINCT shell_execution_id) AS shell_executions,
  COUNT(DISTINCT IF(is_error, shell_execution_id, NULL)) AS shell_errors,
  SAFE_DIVIDE(
    COUNT(DISTINCT IF(is_error, shell_execution_id, NULL)),
    COUNT(DISTINCT shell_execution_id)
  ) AS shell_error_rate,
  ROUND(
    100 * SAFE_DIVIDE(
      COUNT(DISTINCT IF(is_error, shell_execution_id, NULL)),
      COUNT(DISTINCT shell_execution_id)
    ),
    2
  ) AS shell_error_pct,
  ARRAY_AGG(
    IF(
      is_error,
      STRUCT(
        TIMESTAMP_MILLIS(start_ms) AS started_at,
        project_id,
        COALESCE(entry_uuid, entry_id) AS entry_id,
        eval_id,
        run_id,
        trace_id,
        span_id,
        span_name,
        action_id,
        action_run_id,
        action_status,
        span_status,
        provider_status,
        output_status_code,
        error_str
      ),
      NULL
    )
    IGNORE NULLS
    ORDER BY start_ms DESC
    LIMIT 25
  ) AS recent_error_examples
FROM classified
GROUP BY eval_id
""".strip()


def build_high_signal_source_entries_query(
    *,
    evalset_entries_table: str = DEFAULT_EVALSET_ENTRIES_TABLE,
    eval_workflow_runs_table: str = DEFAULT_EVAL_WORKFLOW_RUNS_TABLE,
    deployment_ids: Sequence[str] | None = None,
) -> str:
    """Resolve source evalset rows from an eval run's trace-side entry UUIDs."""
    deployment_filter = ""
    if deployment_ids:
        deployment_filter = "AND project_id IN UNNEST(@deployment_ids)"
    return f"""
WITH runtime_entries AS (
  SELECT
    entry_uuid,
    ARRAY_AGG(stt IGNORE NULLS ORDER BY workflow_start_timestamp DESC LIMIT 1)[SAFE_OFFSET(0)] AS stt
  FROM `{eval_workflow_runs_table}`
  WHERE eval_id = @eval_run_id
    AND entry_uuid IN UNNEST(@entry_uuids)
  GROUP BY entry_uuid
)
SELECT
  runtime_entries.entry_uuid AS id,
  evalset_entries.project_id AS deploymentId,
  evalset_entries.stt,
  evalset_entries.workflow_run_id AS runId,
  evalset_entries.query_ts,
  evalset_entries.datepartition AS source_date
FROM runtime_entries
JOIN `{evalset_entries_table}` AS evalset_entries USING (stt)
WHERE evalset_entries.eval_set_name = @eval_set_name
  AND evalset_entries.eval_set_version = @eval_set_version
  AND LENGTH(stt) > 0
  AND LENGTH(evalset_entries.workflow_run_id) > 0
  {deployment_filter}
QUALIFY ROW_NUMBER() OVER (PARTITION BY runtime_entries.entry_uuid ORDER BY evalset_entries.log_ts DESC) = 1
ORDER BY id
""".strip()


def build_high_signal_trace_query(
    *,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> str:
    """Join resolved source entries to their non-eval historical traces."""
    return f"""
WITH source_entries AS (
  SELECT
    @entry_ids[OFFSET(i)] AS id,
    @deployment_ids[OFFSET(i)] AS deploymentId,
    @session_tracking_tokens[OFFSET(i)] AS stt,
    @workflow_run_ids[OFFSET(i)] AS runId
  FROM UNNEST(GENERATE_ARRAY(0, ARRAY_LENGTH(@entry_ids) - 1)) AS i
)
SELECT
  e.id,
  e.deploymentId,
  e.stt,
  e.runId,
  a.jsonPayload.context.agent_trace.trace_id AS traceId
FROM source_entries e
JOIN `{agentspan_table}` a
  ON (
    a.jsonPayload.context.workflow.run_id = e.runId
    OR (
      e.runId = ''
      AND a.jsonPayload.span_info.session_info.session_tracking_token = e.stt
    )
  )
WHERE _TABLE_SUFFIX BETWEEN @start_suffix AND @end_suffix
  AND a.jsonPayload.context.eval.eval_id IS NULL
  AND a.jsonPayload.context.agent_trace.trace_id IS NOT NULL
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY e.id
  ORDER BY SAFE_CAST(a.jsonPayload.span_info.start_end_timestamps.start_time_millis AS INT64) DESC
) = 1
ORDER BY e.stt
""".strip()


def fetch_high_signal_evalset_entries(
    client: Any,
    *,
    eval_set_name: str,
    eval_set_version: str,
    eval_run_id: str,
    entry_ids: Sequence[str],
    deployment_ids: Sequence[str] | None = None,
    evalset_entries_table: str = DEFAULT_EVALSET_ENTRIES_TABLE,
    eval_workflow_runs_table: str = DEFAULT_EVAL_WORKFLOW_RUNS_TABLE,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> list[dict[str, Any]]:
    """Fetch upload-ready source entries from the parent eval's trace-side entry UUIDs."""
    entry_uuids = sorted(set(entry_ids))
    if not entry_uuids:
        return []
    params = [
        QueryParameter("eval_set_name", "STRING", eval_set_name),
        QueryParameter("eval_set_version", "STRING", eval_set_version),
        QueryParameter("eval_run_id", "STRING", eval_run_id),
        QueryParameter("entry_uuids", "STRING", entry_uuids),
    ]
    if deployment_ids:
        params.append(QueryParameter("deployment_ids", "STRING", list(deployment_ids)))
    source_rows = client.query(
        build_high_signal_source_entries_query(
            evalset_entries_table=evalset_entries_table,
            eval_workflow_runs_table=eval_workflow_runs_table,
            deployment_ids=deployment_ids,
        ),
        params=params,
    )
    source_rows = [row for row in source_rows if row.get("id") and row.get("deploymentId") and row.get("stt")]
    if not source_rows:
        return []

    source_dates = [
        parsed_date
        for row in source_rows
        for parsed_date in [_parse_bigquery_date(row.get("query_ts") or row.get("source_date"))]
        if parsed_date is not None
    ]
    if not source_dates:
        return []
    start_suffix = (min(source_dates) - timedelta(days=1)).strftime("%Y%m%d")
    end_suffix = (max(source_dates) + timedelta(days=1)).strftime("%Y%m%d")
    trace_rows = client.query(
        build_high_signal_trace_query(agentspan_table=agentspan_table),
        params=[
            QueryParameter("entry_ids", "STRING", [str(row["id"]) for row in source_rows]),
            QueryParameter(
                "deployment_ids",
                "STRING",
                [str(row["deploymentId"]) for row in source_rows],
            ),
            QueryParameter(
                "session_tracking_tokens",
                "STRING",
                [str(row["stt"]) for row in source_rows],
            ),
            QueryParameter(
                "workflow_run_ids",
                "STRING",
                [str(row.get("runId") or "") for row in source_rows],
            ),
            QueryParameter("start_suffix", "STRING", start_suffix),
            QueryParameter("end_suffix", "STRING", end_suffix),
        ],
    )
    return [
        {
            "id": str(row["id"]),
            "deploymentId": str(row["deploymentId"]),
            "stt": str(row["stt"]),
            "runId": str(row["runId"]),
            "traceId": str(row["traceId"]),
        }
        for row in trace_rows
        if row.get("id") and row.get("deploymentId") and row.get("stt") and row.get("runId") and row.get("traceId")
    ]


def _parse_bigquery_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def default_date_range(*, lookback_days: int = 7, end_date: date | None = None) -> tuple[date, date]:
    resolved_end = end_date or date.today()
    return resolved_end - timedelta(days=lookback_days), resolved_end


def build_eval_run_search_params(
    *,
    eval_id: str,
    lookback_days: int = 7,
    end_date: date | None = None,
) -> list[QueryParameter]:
    search_start, search_end = default_date_range(lookback_days=lookback_days, end_date=end_date)
    return [
        QueryParameter("eval_id", "STRING", eval_id),
        QueryParameter("search_start_date", "DATE", search_start.isoformat()),
        QueryParameter("search_end_date", "DATE", search_end.isoformat()),
    ]


def build_shell_tool_error_query_params(
    *,
    eval_id: str,
    start_date: date,
    end_date: date,
    entry_ids: Sequence[str] | None = None,
) -> list[QueryParameter]:
    params: list[QueryParameter] = [
        QueryParameter("eval_id", "STRING", eval_id),
        QueryParameter("start_date", "DATE", start_date.isoformat()),
        QueryParameter("end_date", "DATE", end_date.isoformat()),
    ]
    if entry_ids:
        params.append(QueryParameter("entry_ids", "STRING", list(entry_ids)))
    return params


def resolve_eval_run_date_range(
    bounds_row: dict[str, Any] | None,
    *,
    lookback_days: int,
    end_date: date | None = None,
) -> tuple[date, date] | None:
    if not bounds_row:
        return None
    min_ms = bounds_row.get("min_start_ms")
    max_ms = bounds_row.get("max_start_ms")
    if min_ms is None or max_ms is None:
        return None

    # `_TABLE_SUFFIX` is a UTC date. Convert the bounds in UTC too; using the
    # host timezone can shift a just-after-midnight span into the prior day,
    # causing the aggregate query to scan a different shard and return 0/0.
    min_date = datetime.fromtimestamp(int(min_ms) / 1000, tz=timezone.utc).date()
    max_date = datetime.fromtimestamp(int(max_ms) / 1000, tz=timezone.utc).date()
    search_end = end_date or date.today()
    search_start = search_end - timedelta(days=lookback_days)
    start_date = max(min_date, search_start)
    end_date_resolved = min(max_date, search_end)
    if start_date > end_date_resolved:
        return None
    return start_date, end_date_resolved


def parse_shell_tool_error_example(raw: dict[str, Any]) -> ShellToolErrorExample:
    return ShellToolErrorExample(
        started_at=_stringify_value(raw.get("started_at")),
        project_id=_optional_str(raw.get("project_id")),
        entry_id=_optional_str(raw.get("entry_id")),
        eval_id=_optional_str(raw.get("eval_id")),
        run_id=_optional_str(raw.get("run_id")),
        trace_id=_optional_str(raw.get("trace_id")),
        session_tracking_token=_optional_str(raw.get("session_tracking_token")),
        span_id=_optional_str(raw.get("span_id")),
        span_name=_optional_str(raw.get("span_name")),
        action_id=_optional_str(raw.get("action_id")),
        action_run_id=_optional_str(raw.get("action_run_id")),
        action_input=_optional_str(raw.get("action_input")),
        action_status=_optional_str(raw.get("action_status")),
        span_status=_optional_str(raw.get("span_status")),
        provider_status=_optional_str(raw.get("provider_status")),
        output_status_code=_optional_str(raw.get("output_status_code")),
        error_str=_optional_str(raw.get("error_str")),
    )


def parse_shell_tool_error_entry_metrics(row: dict[str, Any]) -> ShellToolErrorEntryMetrics:
    examples_raw = row.get("recent_error_examples") or []
    examples = tuple(parse_shell_tool_error_example(example) for example in examples_raw)
    shell_executions = int(row.get("shell_executions") or 0)
    shell_errors = int(row.get("shell_errors") or 0)
    shell_error_rate = float(row.get("shell_error_rate") or 0.0)
    shell_error_pct = float(row.get("shell_error_pct") or (shell_error_rate * 100))
    trace_ids = tuple(str(trace_id) for trace_id in (row.get("trace_ids") or []) if trace_id)
    session_tracking_tokens = tuple(str(token) for token in (row.get("session_tracking_tokens") or []) if token)
    return ShellToolErrorEntryMetrics(
        entry_id=str(row.get("entry_id") or ""),
        shell_executions=shell_executions,
        shell_errors=shell_errors,
        shell_error_rate=shell_error_rate,
        shell_error_pct=shell_error_pct,
        recent_error_examples=examples,
        trace_ids=trace_ids,
        session_tracking_tokens=session_tracking_tokens,
    )


def parse_shell_tool_error_metrics(row: dict[str, Any]) -> ShellToolErrorMetrics:
    examples_raw = row.get("recent_error_examples") or []
    examples = tuple(parse_shell_tool_error_example(example) for example in examples_raw)
    shell_executions = int(row.get("shell_executions") or 0)
    shell_errors = int(row.get("shell_errors") or 0)
    shell_error_rate = float(row.get("shell_error_rate") or 0.0)
    shell_error_pct = float(row.get("shell_error_pct") or (shell_error_rate * 100))
    return ShellToolErrorMetrics(
        eval_id=str(row.get("eval_id") or ""),
        shell_executions=shell_executions,
        shell_errors=shell_errors,
        shell_error_rate=shell_error_rate,
        shell_error_pct=shell_error_pct,
        recent_error_examples=examples,
    )


def aggregate_entry_metrics(
    eval_id: str,
    per_entry: dict[str, ShellToolErrorEntryMetrics],
) -> ShellToolErrorMetrics:
    if not per_entry:
        return empty_shell_tool_error_metrics(eval_id)
    shell_executions = sum(entry.shell_executions for entry in per_entry.values())
    shell_errors = sum(entry.shell_errors for entry in per_entry.values())
    shell_error_rate = shell_errors / shell_executions if shell_executions else 0.0
    recent_examples: list[ShellToolErrorExample] = []
    for entry in per_entry.values():
        recent_examples.extend(entry.recent_error_examples)
    recent_examples = recent_examples[:25]
    return ShellToolErrorMetrics(
        eval_id=eval_id,
        shell_executions=shell_executions,
        shell_errors=shell_errors,
        shell_error_rate=shell_error_rate,
        shell_error_pct=shell_error_rate * 100,
        recent_error_examples=tuple(recent_examples),
    )


def high_signal_entry_ids(per_entry: dict[str, ShellToolErrorEntryMetrics]) -> tuple[str, ...]:
    return tuple(sorted(entry_id for entry_id, metrics in per_entry.items() if metrics.has_shell_error))


def shell_error_free_rate(per_entry: dict[str, ShellToolErrorEntryMetrics]) -> float:
    """Fraction of observed entries with no shell tool errors.

    Used to score a re-run of the high-signal subset. The focused eval set assigns fresh
    entry uuids, so the re-run is scored on its own entries rather than matched by id.
    """
    if not per_entry:
        return 1.0
    passing = sum(1 for metrics in per_entry.values() if not metrics.has_shell_error)
    return passing / len(per_entry)


def fetch_eval_run_shell_tool_error_analysis(
    client: Any,
    *,
    eval_id: str,
    lookback_days: int = 7,
    end_date: date | None = None,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
    include_error_examples: bool = True,
) -> EvalRunShellToolErrorAnalysis:
    bounds_rows = client.query(
        build_eval_run_time_bounds_query(agentspan_table=agentspan_table),
        params=build_eval_run_search_params(
            eval_id=eval_id,
            lookback_days=lookback_days,
            end_date=end_date,
        ),
    )
    date_range = resolve_eval_run_date_range(
        bounds_rows[0] if bounds_rows else None,
        lookback_days=lookback_days,
        end_date=end_date,
    )
    if date_range is None:
        empty = empty_shell_tool_error_metrics(eval_id)
        return EvalRunShellToolErrorAnalysis(
            eval_id=eval_id,
            start_date=default_date_range(lookback_days=lookback_days, end_date=end_date)[0],
            end_date=default_date_range(lookback_days=lookback_days, end_date=end_date)[1],
            aggregate=empty,
            per_entry={},
            high_signal_entry_ids=(),
        )

    start_date, resolved_end = date_range
    # Aggregate counts must cover every matching shell span. Per-entry metrics
    # deliberately require eval-entry attribution and therefore omit spans that
    # lack an entry id/uuid; using them as the aggregate made those omitted
    # spans look like a perfect 0/0 run.
    aggregate_rows = client.query(
        build_shell_tool_error_rate_query(agentspan_table=agentspan_table),
        params=build_shell_tool_error_query_params(
            eval_id=eval_id,
            start_date=start_date,
            end_date=resolved_end,
        ),
    )
    aggregate = parse_shell_tool_error_metrics(aggregate_rows[0]) if aggregate_rows else empty_shell_tool_error_metrics(eval_id)
    per_entry_rows = client.query(
        build_shell_tool_error_per_entry_query(
            agentspan_table=agentspan_table,
            include_error_examples=include_error_examples,
        ),
        params=build_shell_tool_error_query_params(
            eval_id=eval_id,
            start_date=start_date,
            end_date=resolved_end,
        ),
    )
    per_entry = {
        metrics.entry_id: metrics
        for row in per_entry_rows
        for metrics in [parse_shell_tool_error_entry_metrics(row)]
        if metrics.entry_id
    }
    return EvalRunShellToolErrorAnalysis(
        eval_id=eval_id,
        start_date=start_date,
        end_date=resolved_end,
        aggregate=aggregate,
        per_entry=per_entry,
        high_signal_entry_ids=high_signal_entry_ids(per_entry),
    )


def fetch_shell_tool_error_metrics_for_entries(
    client: Any,
    *,
    eval_id: str,
    entry_ids: Sequence[str],
    start_date: date,
    end_date: date,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> dict[str, ShellToolErrorEntryMetrics]:
    if not entry_ids:
        return {}
    rows = client.query(
        build_shell_tool_error_per_entry_query(
            agentspan_table=agentspan_table,
            entry_ids=entry_ids,
        ),
        params=build_shell_tool_error_query_params(
            eval_id=eval_id,
            start_date=start_date,
            end_date=end_date,
            entry_ids=entry_ids,
        ),
    )
    return {
        metrics.entry_id: metrics
        for row in rows
        for metrics in [parse_shell_tool_error_entry_metrics(row)]
        if metrics.entry_id
    }


def fetch_shell_tool_error_metrics(
    client: Any,
    *,
    eval_id: str,
    lookback_days: int = 7,
    end_date: date | None = None,
    agentspan_table: str = DEFAULT_AGENTS_SPAN_TABLE,
) -> ShellToolErrorMetrics:
    analysis = fetch_eval_run_shell_tool_error_analysis(
        client,
        eval_id=eval_id,
        lookback_days=lookback_days,
        end_date=end_date,
        agentspan_table=agentspan_table,
    )
    return analysis.aggregate


def empty_shell_tool_error_metrics(eval_id: str) -> ShellToolErrorMetrics:
    return ShellToolErrorMetrics(
        eval_id=eval_id,
        shell_executions=0,
        shell_errors=0,
        shell_error_rate=0.0,
        shell_error_pct=0.0,
        recent_error_examples=(),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _stringify_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
