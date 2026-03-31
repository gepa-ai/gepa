"""Git repository operations for the coding adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitRepo:
    """Thin wrapper around git CLI for branch and file operations."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        # Validate it's a git repo
        self._run(["git", "rev-parse", "--git-dir"])

    @staticmethod
    def ensure_repo(repo_path: str, initial_branch: str = "base") -> GitRepo:
        """Ensure a directory is a git repo. If not, initialize it.

        For non-git directories, runs ``git init``, commits all existing files,
        and creates a named branch at that commit.

        Returns a GitRepo instance.
        """
        resolved = str(Path(repo_path).resolve())
        check = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=resolved,
            capture_output=True,
            text=True,
        )
        if check.returncode == 0:
            # Already a git repo
            repo = GitRepo(resolved)
            # Ensure the initial branch exists (create if needed)
            if not repo.branch_exists(initial_branch):
                repo._run(["git", "checkout", "-B", initial_branch])
            return repo

        # Not a git repo — initialize
        subprocess.run(["git", "init"], cwd=resolved, check=True, capture_output=True)
        subprocess.run(["git", "add", "-A"], cwd=resolved, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial commit (gepa)"],
            cwd=resolved,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", "-B", initial_branch],
            cwd=resolved,
            check=True,
            capture_output=True,
        )
        return GitRepo(resolved)

    def _run(
        self,
        cmd: list[str],
        *,
        capture: bool = True,
        check: bool = True,
        cwd: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            cwd=cwd or self.repo_path,
            capture_output=capture,
            text=True,
            check=check,
        )

    def current_branch(self) -> str:
        result = self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        return result.stdout.strip()

    def branch_exists(self, name: str) -> bool:
        result = self._run(["git", "rev-parse", "--verify", name], check=False)
        return result.returncode == 0

    def create_branch(self, name: str, from_branch: str) -> None:
        """Create a new branch. If it already exists, delete and recreate."""
        if self.branch_exists(name):
            self._run(["git", "branch", "-D", name])
        self._run(["git", "branch", name, from_branch])

    def checkout(self, branch: str) -> None:
        self._run(["git", "checkout", branch])

    def commit_all(self, message: str) -> bool:
        """Stage all changes and commit. Returns True if a commit was created."""
        self._run(["git", "add", "-A"])
        result = self._run(["git", "diff", "--cached", "--quiet"], check=False)
        if result.returncode == 0:
            return False
        self._run(["git", "commit", "-m", message])
        return True

    def get_diff(self, base: str, head: str) -> str:
        result = self._run(["git", "diff", base, head])
        return result.stdout

    def read_file(self, path: str, branch: str | None = None) -> str:
        """Read a file's contents. If branch is given, read from that branch without checkout."""
        if branch is not None:
            result = self._run(["git", "show", f"{branch}:{path}"])
            return result.stdout
        file_path = Path(self.repo_path) / path
        return file_path.read_text()

    def write_file(self, path: str, content: str) -> None:
        file_path = Path(self.repo_path) / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    def list_files(self, branch: str | None = None, patterns: list[str] | None = None) -> list[str]:
        """List tracked files, optionally filtered by glob patterns."""
        ref = branch or "HEAD"
        cmd = ["git", "ls-tree", "-r", "--name-only", ref]
        result = self._run(cmd)
        files = result.stdout.strip().splitlines()
        if patterns:
            import fnmatch

            filtered: list[str] = []
            for f in files:
                if any(fnmatch.fnmatch(f, p) for p in patterns):
                    filtered.append(f)
            return filtered
        return files

    def has_uncommitted_changes(self) -> bool:
        result = self._run(["git", "status", "--porcelain"])
        return bool(result.stdout.strip())
