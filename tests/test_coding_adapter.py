"""Tests for the coding adapter: GitRepo, CodingAdapter, and coding mode in optimize_anything."""

import os
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from gepa.adapters.coding_adapter.git_repo import GitRepo
from gepa.adapters.coding_adapter.coding_agent import CodingAgentProtocol
from gepa.optimize_anything import CodeCandidate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary git repo with some files and return its path."""
    repo = tmp_path / "repo"
    repo.mkdir()

    (repo / "main.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "utils.py").write_text("def helper():\n    pass\n")

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo,
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "gepa", "GIT_AUTHOR_EMAIL": "gepa@gepa", "GIT_COMMITTER_NAME": "gepa", "GIT_COMMITTER_EMAIL": "gepa@gepa"},
    )
    subprocess.run(["git", "checkout", "-B", "base"], cwd=repo, check=True, capture_output=True)
    return str(repo)


@pytest.fixture
def temp_non_git_dir(tmp_path):
    """Create a temporary directory with files but NO git repo."""
    d = tmp_path / "project"
    d.mkdir()
    (d / "app.py").write_text("print('hello')\n")
    (d / "config.yaml").write_text("key: value\n")
    return str(d)


# ---------------------------------------------------------------------------
# GitRepo tests
# ---------------------------------------------------------------------------


class TestGitRepo:
    def test_init_valid_repo(self, temp_git_repo):
        repo = GitRepo(temp_git_repo)
        assert repo.repo_path == temp_git_repo

    def test_init_invalid_dir(self, tmp_path):
        with pytest.raises(subprocess.CalledProcessError):
            GitRepo(str(tmp_path))

    def test_current_branch(self, temp_git_repo):
        repo = GitRepo(temp_git_repo)
        assert repo.current_branch() == "base"

    def test_create_and_checkout_branch(self, temp_git_repo):
        repo = GitRepo(temp_git_repo)
        repo.create_branch("feature", "base")
        repo.checkout("feature")
        assert repo.current_branch() == "feature"

    def test_create_branch_overwrites_existing(self, temp_git_repo):
        repo = GitRepo(temp_git_repo)
        repo.create_branch("feature", "base")
        repo.checkout("feature")

        # Modify and commit on feature
        repo.write_file("new.py", "x = 1\n")
        repo.commit_all("add new.py")

        # Recreate feature from base — should lose the commit
        repo.checkout("base")
        repo.create_branch("feature", "base")
        repo.checkout("feature")
        assert not os.path.exists(os.path.join(temp_git_repo, "new.py"))

    def test_branch_exists(self, temp_git_repo):
        repo = GitRepo(temp_git_repo)
        assert repo.branch_exists("base")
        assert not repo.branch_exists("nonexistent")

    def test_commit_all(self, temp_git_repo):
        repo = GitRepo(temp_git_repo)
        repo.create_branch("work", "base")
        repo.checkout("work")

        # No changes → returns False
        assert repo.commit_all("empty") is False

        # Make a change → returns True
        repo.write_file("main.py", "def add(a, b):\n    return a + b + 0\n")
        assert repo.commit_all("tweak") is True

    def test_get_diff(self, temp_git_repo):
        repo = GitRepo(temp_git_repo)
        repo.create_branch("work", "base")
        repo.checkout("work")
        repo.write_file("main.py", "def multiply(a, b):\n    return a * b\n")
        repo.commit_all("change function")

        diff = repo.get_diff("base", "work")
        assert "multiply" in diff
        assert "add" in diff  # removed line

    def test_read_file(self, temp_git_repo):
        repo = GitRepo(temp_git_repo)
        content = repo.read_file("main.py")
        assert "def add" in content

    def test_read_file_from_branch(self, temp_git_repo):
        repo = GitRepo(temp_git_repo)
        repo.create_branch("work", "base")
        repo.checkout("work")
        repo.write_file("main.py", "CHANGED")
        repo.commit_all("change")

        # Read from base (should be original)
        original = repo.read_file("main.py", branch="base")
        assert "def add" in original

        # Read from work (should be changed)
        changed = repo.read_file("main.py", branch="work")
        assert changed == "CHANGED"

    def test_write_file_creates_dirs(self, temp_git_repo):
        repo = GitRepo(temp_git_repo)
        repo.write_file("sub/dir/file.py", "content")
        assert os.path.exists(os.path.join(temp_git_repo, "sub", "dir", "file.py"))

    def test_list_files(self, temp_git_repo):
        repo = GitRepo(temp_git_repo)
        files = repo.list_files()
        assert "main.py" in files
        assert "utils.py" in files

    def test_list_files_with_patterns(self, temp_git_repo):
        repo = GitRepo(temp_git_repo)
        files = repo.list_files(patterns=["*.py"])
        assert "main.py" in files

    def test_has_uncommitted_changes(self, temp_git_repo):
        repo = GitRepo(temp_git_repo)
        assert not repo.has_uncommitted_changes()
        repo.write_file("new_file.py", "x = 1")
        assert repo.has_uncommitted_changes()


class TestGitRepoEnsure:
    def test_ensure_repo_existing(self, temp_git_repo):
        """ensure_repo on an existing git repo should return GitRepo without reinit."""
        repo = GitRepo.ensure_repo(temp_git_repo, initial_branch="base")
        assert repo.current_branch() == "base"
        # Files should still be there
        assert "main.py" in repo.list_files()

    def test_ensure_repo_non_git(self, temp_non_git_dir):
        """ensure_repo on a plain directory should git init and commit."""
        repo = GitRepo.ensure_repo(temp_non_git_dir, initial_branch="base")
        assert repo.current_branch() == "base"
        files = repo.list_files()
        assert "app.py" in files
        assert "config.yaml" in files
        # Should have a commit
        assert not repo.has_uncommitted_changes()

    def test_ensure_repo_creates_initial_branch(self, temp_git_repo):
        """ensure_repo should create the initial branch if it doesn't exist."""
        repo = GitRepo.ensure_repo(temp_git_repo, initial_branch="custom_base")
        assert repo.branch_exists("custom_base")


# ---------------------------------------------------------------------------
# CodeCandidate tests
# ---------------------------------------------------------------------------


class TestCodeCandidate:
    def test_single_repo_string(self):
        cc = CodeCandidate(repo_paths="/path/to/repo")
        assert cc.repo_paths == "/path/to/repo"
        assert cc.base_branch == "main"
        assert cc.coding_agent == "bash"

    def test_multi_repo_list(self):
        cc = CodeCandidate(repo_paths=["/path/a", "/path/b"])
        assert cc.repo_paths == ["/path/a", "/path/b"]

    def test_custom_agent(self):
        cc = CodeCandidate(repo_paths="/repo", coding_agent="claude_code")
        assert cc.coding_agent == "claude_code"

    def test_custom_model(self):
        cc = CodeCandidate(repo_paths="/repo", model="anthropic/claude-sonnet-4-6")
        assert cc.model == "anthropic/claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# CodingAdapter integration tests
# ---------------------------------------------------------------------------


class TestCodingAdapterIntegration:
    """Test CodingAdapter through optimize_anything with a mock coding agent."""

    def _make_mock_agent(self):
        """Create a mock coding agent that makes a simple change."""

        class MockAgent:
            def __init__(self):
                self.call_count = 0

            def propose(self, repo, base_branch, feedback, objective, background):
                self.call_count += 1
                # Make a simple change: append a comment to main.py
                content = repo.read_file("main.py")
                repo.write_file("main.py", content + f"\n# optimization {self.call_count}\n")
                return True

        return MockAgent()

    def test_coding_adapter_evaluate(self, temp_git_repo):
        """CodingAdapter should checkout branch before calling evaluator."""
        from gepa.adapters.coding_adapter import CodingAdapter, GitRepo

        repo = GitRepo(temp_git_repo)
        mock_agent = self._make_mock_agent()

        # Track what evaluator sees
        eval_calls = []

        def evaluator(candidate, **kwargs):
            # candidate should be {repo_path: branch_name}
            eval_calls.append(candidate)
            return 1.0

        from gepa.optimize_anything import EvaluatorWrapper

        wrapped = EvaluatorWrapper(evaluator, single_instance_mode=True, str_candidate_mode=False)

        adapter = CodingAdapter(
            evaluator=wrapped,
            coding_agent=mock_agent,
            repos={temp_git_repo: repo},
            base_branches={temp_git_repo: "base"},
            objective="test",
        )

        candidate = {temp_git_repo: "base"}
        result = adapter.evaluate([None], candidate, capture_traces=True)

        assert len(result.scores) == 1
        assert result.scores[0] == 1.0
        assert len(eval_calls) == 1
        assert eval_calls[0] == candidate

    def test_coding_adapter_propose(self, temp_git_repo):
        """CodingAdapter.propose_new_texts should create a new branch with changes."""
        from gepa.adapters.coding_adapter import CodingAdapter, GitRepo

        repo = GitRepo(temp_git_repo)
        mock_agent = self._make_mock_agent()

        def evaluator(candidate, **kwargs):
            return 1.0

        from gepa.optimize_anything import EvaluatorWrapper

        wrapped = EvaluatorWrapper(evaluator, single_instance_mode=True, str_candidate_mode=False)

        adapter = CodingAdapter(
            evaluator=wrapped,
            coding_agent=mock_agent,
            repos={temp_git_repo: repo},
            base_branches={temp_git_repo: "base"},
            objective="test",
        )

        candidate = {temp_git_repo: "base"}
        new_texts = adapter.propose_new_texts(
            candidate,
            reflective_dataset={temp_git_repo: [{"Score": 1.0, "Feedback": "good"}]},
            components_to_update=[temp_git_repo],
        )

        # Should return new branch name
        assert temp_git_repo in new_texts
        new_branch = new_texts[temp_git_repo]
        assert new_branch.startswith("gepa/iter_")

        # Branch should exist with the change
        assert repo.branch_exists(new_branch)
        content = repo.read_file("main.py", branch=new_branch)
        assert "optimization 1" in content

        # Agent should have been called once
        assert mock_agent.call_count == 1

    def test_coding_adapter_reflective_dataset(self, temp_git_repo):
        """make_reflective_dataset should include code diffs and side_info."""
        from gepa.adapters.coding_adapter import CodingAdapter, GitRepo
        from gepa.core.adapter import EvaluationBatch

        repo = GitRepo(temp_git_repo)
        mock_agent = self._make_mock_agent()

        def evaluator(candidate, **kwargs):
            return 1.0

        from gepa.optimize_anything import EvaluatorWrapper

        wrapped = EvaluatorWrapper(evaluator, single_instance_mode=True, str_candidate_mode=False)

        adapter = CodingAdapter(
            evaluator=wrapped,
            coding_agent=mock_agent,
            repos={temp_git_repo: repo},
            base_branches={temp_git_repo: "base"},
            objective="test",
        )

        # Create a branch with changes
        repo.create_branch("gepa/iter_1", "base")
        repo.checkout("gepa/iter_1")
        repo.write_file("main.py", "def fast_add(a, b):\n    return a + b\n")
        repo.commit_all("optimize")

        candidate = {temp_git_repo: "gepa/iter_1"}
        eval_batch = EvaluationBatch(
            outputs=[(1.5, None)],
            scores=[1.5],
            trajectories=[{"Feedback": "1.5x speedup", "execution_time": 0.3}],
        )

        dataset = adapter.make_reflective_dataset(candidate, eval_batch, [temp_git_repo])

        assert temp_git_repo in dataset
        records = dataset[temp_git_repo]
        assert len(records) == 1
        record = records[0]
        assert "Code Diff from Base" in record
        assert "fast_add" in record["Code Diff from Base"]
        assert record["Score"] == 1.5
        assert record["Feedback"] == "1.5x speedup"
        assert record["execution_time"] == 0.3

    def test_end_to_end_optimize_anything(self, temp_git_repo):
        """Full end-to-end test of optimize_anything in coding mode."""
        from gepa.optimize_anything import (
            CodeCandidate,
            EngineConfig,
            GEPAConfig,
            optimize_anything,
        )

        mock_agent = self._make_mock_agent()

        # Simple evaluator: score based on number of lines in main.py
        def evaluator(candidate):
            repo_path = list(candidate.keys())[0]
            with open(os.path.join(repo_path, "main.py")) as f:
                lines = len(f.readlines())
            return float(lines), {"lines": lines}

        result = optimize_anything(
            seed_candidate=CodeCandidate(
                repo_paths=temp_git_repo,
                base_branch="base",
                coding_agent=mock_agent,
                branch_prefix="gepa",
            ),
            evaluator=evaluator,
            objective="Add more lines to main.py",
            config=GEPAConfig(
                engine=EngineConfig(
                    max_metric_calls=5,
                    frontier_type="instance",
                ),
            ),
        )

        # Should have explored at least 1 candidate beyond seed
        assert result.num_candidates >= 1

        # Best candidate should be a dict with the repo path
        best = result.best_candidate
        assert isinstance(best, dict)
        assert temp_git_repo in best

    def test_multi_repo(self, tmp_path):
        """Test multi-repo optimization with two repos."""
        # Create two repos
        repo_a = tmp_path / "repo_a"
        repo_a.mkdir()
        (repo_a / "a.py").write_text("x = 1\n")

        repo_b = tmp_path / "repo_b"
        repo_b.mkdir()
        (repo_b / "b.py").write_text("y = 2\n")

        mock_agent = self._make_mock_agent()

        def evaluator(candidate):
            return 1.0

        from gepa.optimize_anything import (
            CodeCandidate,
            EngineConfig,
            GEPAConfig,
            optimize_anything,
        )

        result = optimize_anything(
            seed_candidate=CodeCandidate(
                repo_paths=[str(repo_a), str(repo_b)],
                base_branch="base",
                coding_agent=mock_agent,
                branch_prefix="gepa",
            ),
            evaluator=evaluator,
            objective="Optimize both repos",
            config=GEPAConfig(
                engine=EngineConfig(
                    max_metric_calls=3,
                    frontier_type="instance",
                ),
            ),
        )

        best = result.best_candidate
        assert isinstance(best, dict)
        assert str(repo_a) in best
        assert str(repo_b) in best
