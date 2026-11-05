import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gepa.core.state import GEPAState
from gepa.evidence.config import SequentialEvidenceConfig, digest
from gepa.evidence.controller import EvidenceController
from test_sequential_evidence_controller import Adapter, controller, proposal
from test_sequential_evidence_native import ScriptedAdapter, run


CRASH_SCRIPT = r'''
import os, sys
sys.path.insert(0, "tests")
from pathlib import Path
from test_sequential_evidence_native import ScriptedAdapter, run
from gepa.evidence.controller import EvidenceController
from gepa.core.state import GEPAState
stage, directory = sys.argv[1:]
if stage == "seed":
    class CrashAdapter(ScriptedAdapter):
        def evaluate(self, batch, candidate, capture_traces=False):
            os._exit(72)
    adapter = CrashAdapter()
elif stage == "partial":
    class CrashAdapter(ScriptedAdapter):
        def evaluate_replication(self, batch, candidate, *, trial_context, capture_traces):
            if trial_context.side == "child":
                os._exit(72)
            return super().evaluate_replication(batch, candidate, trial_context=trial_context, capture_traces=capture_traces)
    adapter = CrashAdapter()
else:
    adapter = ScriptedAdapter()
    if stage == "decision":
        original = EvidenceController.compare
        def compare(self, *args, **kwargs):
            original(self, *args, **kwargs)
            os._exit(72)
        EvidenceController.compare = compare
    elif stage == "selection":
        original = EvidenceController.select
        def select(self, *args, **kwargs):
            original(self, *args, **kwargs)
            os._exit(72)
        EvidenceController.select = select
    elif stage == "pool":
        original = GEPAState.save
        def save(self, *args, **kwargs):
            original(self, *args, **kwargs)
            if len(self.program_candidates) > 1:
                os._exit(72)
        GEPAState.save = save
run(Path(directory), adapter, budget=300, stop_callbacks=lambda state: len(state.program_candidates) > 1)
'''


@pytest.mark.parametrize("stage", ["seed", "partial", "decision", "selection", "pool"])
def test_actual_process_failure_reconciles_native_checkpoint_and_journal(tmp_path, stage):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", CRASH_SCRIPT, stage, str(tmp_path)],
                            cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 72, result.stdout + result.stderr
    journal = json.loads((tmp_path / "sequential_evidence.json").read_text())["payload"]
    before = sum(r["actual_cost"] if r["actual_cost"] is not None else r["upper_cost"]
                 for r in journal["guard"]["records"])
    raw = ScriptedAdapter()
    # Resume selected validation only; no new proposal should conceal recovery.
    recovered = run(tmp_path, raw, budget=301, stop_callbacks=lambda state: True)
    assert recovered.total_metric_calls == before + raw.executions
    assert recovered.total_metric_calls <= 301
    assert raw.replications == 0
    if stage == "partial":
        report = recovered.sequential_evidence["comparisons"][0]
        assert report["status"] == "inconclusive"
        assert report["completed_rounds"] == 0
        assert report["metric_calls"] == 2
        assert report["observation_counts"]["abandoned"] == 1
    elif stage == "decision":
        assert len(recovered.candidates) == 1
    elif stage == "pool":
        assert len(recovered.candidates) == 2 and raw.executions == 0
        assert recovered.sequential_evidence["comparisons"][0]["status"] == "promoted"
    elif stage == "seed":
        assert len(recovered.candidates) == 1 and raw.executions == 1
        assert any(r["status"] == "abandoned" for r in
                   json.loads((tmp_path / "sequential_evidence.json").read_text())["payload"]["guard"]["records"])


def rewrite_payload(tmp_path, mutate):
    path = tmp_path / "sequential_evidence.json"
    envelope = json.loads(path.read_text())
    mutate(envelope["payload"])
    envelope["checksum"] = digest(envelope["payload"])
    path.write_text(json.dumps(envelope))


@pytest.mark.parametrize("mutation", ["mean", "bound", "candidate", "round", "observation", "cost"])
def test_semantically_corrupt_saved_evidence_fails_closed(tmp_path, mutation):
    control = controller(tmp_path)
    control.compare(proposal(), {"p": "parent"})
    def mutate(payload):
        comparison = payload["comparisons"][0]
        if mutation == "mean": comparison["moments_parent"][0]["mean"] = 0.75
        if mutation == "bound": comparison["lower"] = 0.99
        if mutation == "candidate": comparison["parent"] = {"p": "different"}
        if mutation == "round": comparison["completed_rounds"] = 0
        if mutation == "observation": comparison["observations"][0]["identity"]["candidate"] = "wrong"
        if mutation == "cost": comparison["metric_calls"] = 1
    rewrite_payload(tmp_path, mutate)
    with pytest.raises(ValueError):
        controller(tmp_path)


def test_legacy_checkpoint_enabling_evidence_starts_no_historical_replicates(tmp_path):
    import gepa
    from test_sequential_evidence_native import Quiet
    ordinary = gepa.optimize({"program": "lambda x: 0"}, [1], [2], ScriptedAdapter(), max_metric_calls=4,
                            reflection_minibatch_size=1, skip_perfect_score=False, logger=Quiet(), run_dir=str(tmp_path))
    result = run(tmp_path, budget=8, stop_callbacks=lambda state: True)
    assert result.total_metric_calls == ordinary.total_metric_calls
    assert result.sequential_evidence["comparisons"] == []


def test_native_checkpoint_ahead_of_journal_rejected(tmp_path):
    run(tmp_path, budget=3)
    state = GEPAState.load(str(tmp_path))
    state.total_num_evals += 1
    state.save(str(tmp_path))
    with pytest.raises(ValueError, match="ahead"):
        run(tmp_path, budget=20)
