"""BigQuery access used by the Glean evaluator."""

from __future__ import annotations

import os
from typing import Any, Protocol, Sequence

DEFAULT_BIGQUERY_PROJECT = "scio-apps"


class BigQueryError(RuntimeError):
    pass


class BigQueryQueryClient(Protocol):
    def query(self, sql: str, job_config: Any | None = None) -> Any: ...


class BigQueryClient:
    """Thin wrapper around google-cloud-bigquery for parameterized SQL execution."""

    def __init__(
        self,
        *,
        project_id: str | None = None,
        client: BigQueryQueryClient | None = None,
    ):
        self.project_id = project_id or os.environ.get("BIGQUERY_PROJECT", DEFAULT_BIGQUERY_PROJECT)
        self._client = client

    def _get_client(self) -> BigQueryQueryClient:
        if self._client is not None:
            return self._client
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            raise BigQueryError(
                "google-cloud-bigquery is required for BigQueryClient. "
                "Install with: uv sync --extra glean"
            ) from exc

        self._client = bigquery.Client(project=self.project_id)
        return self._client

    def query(
        self,
        sql: str,
        *,
        params: Sequence[Any] | None = None,
    ) -> list[dict[str, Any]]:
        from google.cloud import bigquery

        job_config = None
        if params:
            bq_params = []
            for param in params:
                if hasattr(param, "name") and hasattr(param, "type_"):
                    if isinstance(param.value, list):
                        bq_params.append(bigquery.ArrayQueryParameter(param.name, param.type_, param.value))
                    else:
                        bq_params.append(
                            bigquery.ScalarQueryParameter(param.name, param.type_, param.value)
                        )
                else:
                    bq_params.append(param)
            job_config = bigquery.QueryJobConfig(query_parameters=bq_params)

        try:
            query_job = self._get_client().query(sql, job_config=job_config)
            rows = query_job.result()
        except Exception as exc:
            raise BigQueryError(f"BigQuery query failed: {exc}") from exc

        return [dict(row.items()) for row in rows]
