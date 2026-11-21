# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from subprocess import CompletedProcess

import pytest

from gepa.gepa_launcher import ReflectionConfig
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.proposers import claude_code_agent
from gepa.oa.proposers.claude_code_agent import ClaudeCodeAgentProposer, _extract_fenced
from gepa.optimize_anything import optimize_anything


def test_extract_fenced_uses_last_complete_block():
    body = "Explanation:\n```text\nexample\n```\nFinal proposal:\n```markdown\nactual proposal\n```"

    assert _extract_fenced(body) == "actual proposal"


def test_extract_fenced_preserves_unfenced_file_compatibility():
    assert _extract_fenced("  direct proposal  ") == "direct proposal"


def test_extract_fenced_rejects_visibly_truncated_file():
    assert _extract_fenced("<think>unfinished reasoning") == ""


@pytest.mark.parametrize("sandbox", [False, True])
def test_later_proposals_receive_parent_text_without_state_export(tmp_path, monkeypatch, sandbox):
    observed_parents = []

    def run_offline_proposal(self, subdir, components, stems, **kwargs):
        (component,) = components
        stem = stems[component]
        parent = subdir / "parent"
        parent_file = parent / "components" / f"{stem}.txt"
        observed_parents.append(parent_file.read_text(encoding="utf-8"))
        if kwargs["parent_iter_id"] != "seed":
            assert (parent / "plan.md").read_text() == "Expand the parent text."
        (subdir / "plan.md").write_text("Expand the parent text.")
        (subdir / "new" / f"{stem}.md").write_text("```\n" + observed_parents[-1] + " more\n```\n", encoding="utf-8")
        return CompletedProcess(["offline-proposer"], 0, stdout='{"usage": {}}', stderr="")

    monkeypatch.setattr(claude_code_agent, "preflight_claude_engine", lambda *args, **kwargs: None)
    monkeypatch.setattr(claude_code_agent, "copy_session_transcript", lambda *args, **kwargs: None)
    monkeypatch.setattr(ClaudeCodeAgentProposer, "_run_claude", run_offline_proposal)
    proposer = ClaudeCodeAgentProposer("offline", tmp_path, sandbox=sandbox)

    result = optimize_anything(
        "seed text",
        evaluator=lambda candidate: (len(candidate) / 100, {}),
        config=OptimizeAnythingConfig(
            engine="gepa",
            max_evals=12,
            stop_at_score=0.24,
            run_dir=str(tmp_path),
            engine_config={"reflection": ReflectionConfig(custom_candidate_proposer=proposer, reflection_lm=None)},
        ),
    )

    assert observed_parents == ["seed text", "seed text more", "seed text more more"]
    assert result.best_candidate == "seed text more more more"
    assert not (tmp_path / "pareto").exists()
