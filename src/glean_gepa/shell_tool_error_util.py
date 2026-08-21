"""Shell-tool error metrics derived from Glean evaluation spans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Sequence

DEFAULT_AGENTS_SPAN_TABLE = "scio-apps.scrubbed_agentspan.scrubbed_agentspan_*"
SHELL_SUCCESS_OBJECTIVE = "shell_success_rate"
HIGH_SIGNAL_VERIFY_OBJECTIVE = "high_signal_verify_pass_rate"
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


@dataclass(frozen=True)
class ShellToolErrorEntryMetrics:
    entry_id: str
    shell_executions: int
    shell_errors: int
    shell_error_rate: float
    shell_error_pct: float
    recent_error_examples: tuple[ShellToolErrorExample, ...]

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
      jsonPayload.action.action_run_id,
      jsonPayload.context.agent_trace.span_id
    ) AS shell_execution_id,
    jsonPayload.context.eval.entry_uuid AS entry_uuid,
    CAST(jsonPayload.context.eval.entry_id AS STRING) AS entry_id,
    resource.labels.project_id AS project_id,
    jsonPayload.context.workflow.run_id AS run_id,
    jsonPayload.context.agent_trace.trace_id AS trace_id,
    jsonPayload.context.agent_trace.span_id AS span_id,
    jsonPayload.span_info.span_name AS span_name,
    jsonPayload.action.action_id AS action_id,
    jsonPayload.action.execution_status AS action_status,
    jsonPayload.action.error_str AS error_str,
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
) -> str:
    """Build SQL for per-entry shell tool error metrics scoped to one eval run."""
    entry_filter = ""
    if entry_ids:
        entry_filter = "AND COALESCE(entry_uuid, entry_id) IN UNNEST(@entry_ids)"
    shell_spans = _shell_spans_select_sql().format(agentspan_table=agentspan_table)
    return f"""
WITH shell_spans AS (
{shell_spans}
),
classified AS (
  SELECT
    COALESCE(entry_uuid, entry_id) AS entry_key,
    *,
    (
      action_status = 'ERROR'
      OR span_status = 'ERROR'
      OR output_status_code = 'ERROR'
      OR LOWER(provider_status) IN ('failed', 'error')
    ) AS is_error
  FROM shell_spans
  WHERE COALESCE(entry_uuid, entry_id) IS NOT NULL
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
    ) AS shell_error_pct,
    ARRAY_AGG(
      IF(
        is_error,
        STRUCT(
          TIMESTAMP_MILLIS(start_ms) AS started_at,
          project_id,
          entry_key AS entry_id,
          eval_id,
          run_id,
          trace_id,
          span_id,
          span_name,
          action_id,
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
    ) AS recent_error_examples
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

    min_date = date.fromtimestamp(int(min_ms) / 1000)
    max_date = date.fromtimestamp(int(max_ms) / 1000)
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
        span_id=_optional_str(raw.get("span_id")),
        span_name=_optional_str(raw.get("span_name")),
        action_id=_optional_str(raw.get("action_id")),
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
    return ShellToolErrorEntryMetrics(
        entry_id=str(row.get("entry_id") or ""),
        shell_executions=shell_executions,
        shell_errors=shell_errors,
        shell_error_rate=shell_error_rate,
        shell_error_pct=shell_error_pct,
        recent_error_examples=examples,
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
    return tuple(
        sorted(entry_id for entry_id, metrics in per_entry.items() if metrics.has_shell_error)
    )


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
    per_entry_rows = client.query(
        build_shell_tool_error_per_entry_query(agentspan_table=agentspan_table),
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
    aggregate = aggregate_entry_metrics(eval_id, per_entry)
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
