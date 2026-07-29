import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from gepa.oa.budget import BudgetTracker
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.engines.meta_harness import MetaHarnessEngine
from gepa.oa.eval_server import EvalServer
from gepa.oa.task import Task
from gepa.optimize_anything import _run_engine


@pytest.fixture(autouse=True)
def _skip_claude_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gepa.oa.engines.meta_harness.preflight_claude_engine", lambda *a, **k: None)


def test_meta_harness_persists_final_cost_summary_after_failed_proposer(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    task = Task(name="smoke", seed_candidate="seed")
    server = EvalServer(
        task,
        lambda candidate: (1.0, {"cost": 0.05}),
        BudgetTracker(max_evals=5),
        output_dir=output_dir,
    )
    server.evaluate("seed")
    engine = MetaHarnessEngine(
        OptimizeAnythingConfig(
            engine="meta_harness",
            sandbox=False,
            run_dir=str(tmp_path / "work"),
            engine_config={"max_iterations": 1},
        )
    )
    failed_proposer = subprocess.CompletedProcess(
        args=["claude"],
        returncode=1,
        stdout=json.dumps(
            {
                "total_cost_usd": 0.4,
                "adapter_cost_status": "standard_tier_upper_estimate_from_observed_usage",
            }
        ),
        stderr="failed",
    )

    with patch("gepa.oa.engines.meta_harness.subprocess.run", return_value=failed_proposer):
        result = _run_engine(server, engine, owns_server=True)

    summary = json.loads((output_dir / "summary.json").read_text())
    assert result.metadata["adapter_cost"] == pytest.approx(0.4)
    assert result.metadata["adapter_cost_status"] == "standard_tier_upper_estimate_from_observed_usage"
    assert result.metadata["total_cost"] == pytest.approx(0.5)
    assert result.metadata["meta_harness"]["stop_reason"] == "proposer_failed"
    assert summary["eval_cost_usd"] == pytest.approx(0.1)
    assert summary["adapter_cost_usd"] == pytest.approx(0.4)
    assert summary["total_cost_usd"] == pytest.approx(result.metadata["total_cost"])
    assert summary["total_cost"] == pytest.approx(result.metadata["total_cost"])
    assert summary["adapter_cost_status"] == "standard_tier_upper_estimate_from_observed_usage"
