"""Claude Code agent proposer for GEPA (file-based, ``custom_candidate_proposer``).

Drops into GEPA via ``ReflectionConfig.custom_candidate_proposer`` — **not**
``reflection_lm``. The reflection-LM hook is a text-in / text-out callable and
has no structured access to the per-iteration ``reflective_dataset``;
``custom_candidate_proposer`` does, with signature::

    (candidate, reflective_dataset, components_to_update) -> dict[str, str]

That's the contract this module implements. Each call:

1. Allocates a unique subdir under ``<run_dir>/proposals/`` (timestamp + pid +
   uuid) so concurrent proposers or multiple processes sharing the same
   ``run_dir`` can't clobber each other.
2. Writes the parent candidate components, the reflective dataset, and the
   components-to-update list as structured files in that subdir. Nothing goes
   into the prompt body — ``claude`` reads everything from disk.
3. Launches ``claude --print`` with ``cwd=run_dir`` (or a tempdir mirror, when
   sandboxed) and a short wrapper prompt pointing at the subdir. Sandbox
   scopes file tools to the visible tree so Claude can also browse
   ``iterations/``, ``pareto/``, ``history.md`` for history.
4. Reads one file per updated component back from ``<subdir>/new/``. The file
   body is parsed for a fenced ```` ``` ```` block (same convention
   ``InstructionProposalSignature.output_extractor`` uses). Missing files fall
   back to the original candidate text so GEPA's acceptance criterion just
   rejects the no-op proposal.
5. Copies the agent's ``<subdir>/plan.md`` into
   ``iterations/<iter_id>/plan.md`` and rebuilds ``<run_dir>/history.md``
   from the iterations tree so the next proposer call sees a chronological
   log of every prior iteration's short plan + score outcome.

Needs ``EngineConfig.write_agent_state=True`` on the GEPA side so the
``run_dir`` actually contains the readable ``iterations/`` and ``pareto/``
trees this proposer reads. The :class:`~gepa.omni.backends.gepa.GepaBackend`
wiring auto-sets that default whenever this proposer is selected.

Learns the current ``iteration`` (trace ``state.i``) and
``parent_iteration_id`` via the ``metadata`` kwarg on the
:class:`~gepa.core.adapter.ProposalFn` contract — GEPA's
``reflective_mutation`` populates that dict on every call.

Sandbox: OS jail (bwrap on Linux) + Claude file-tool deny-list
(:data:`DENY_WEB_TOOLS`), both scoped to a per-call jail mirror. Network is
shared with the host so the eval server stays reachable. Bash is **on** —
the OS jail already confines shell commands, and grep/diff/jq/python let
the agent analyze past candidates and reflective datasets far more
effectively than the Claude file tools alone. Write isolation across
concurrent proposers is convention-only (claude is *instructed* to stay
inside its own ``proposals/<subdir>/``); a stricter per-subdir write jail
or a git worktree per proposal would be needed to enforce this.

Budget + cost tracking: raises :class:`gepa.omni.budget.BudgetExhausted`
when cumulative spend hits ``max_budget_usd``, and exposes ``total_cost``
/ ``total_tokens_in`` / ``total_tokens_out`` so existing reflection-cost
GEPA callbacks plug in unchanged (they only need those three attributes,
not a full LM protocol).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gepa.omni.backends.claude_utils import copy_session_transcript
from gepa.omni.budget import BudgetExhausted
from gepa.omni.sandbox import DENY_WEB_TOOLS, bwrap_prefix, claude_settings_args

_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _extract_fenced(text: str) -> str:
    """Pull the last fenced ```` ``` ```` block out of ``text``.

    Matches the convention in
    :func:`gepa.strategies.instruction_proposal.InstructionProposalSignature.output_extractor`.
    If there's no complete fence, falls back to the raw text (stripped). If
    the content is preceded by an optional ``<language>`` spec, that line is
    dropped.
    """
    matches = _FENCE_RE.findall(text)
    if matches:
        return matches[-1].strip()
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def _safe_component_filename(name: str, idx: int) -> str:
    """Derive a safe on-disk filename for a component name.

    Mirrors :meth:`gepa.core.state.GEPAState._save_components` so the
    parent-iteration component files this proposer reads from
    ``iterations/NNNNN/components/`` use the exact same stems.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return safe if safe else f"component_{idx:02d}"


class ClaudeCodeAgentProposer:
    """GEPA ``custom_candidate_proposer`` backed by the Claude Code CLI.

    Matches :class:`gepa.core.adapter.ProposalFn` — call signature
    ``(candidate, reflective_dataset, components_to_update) -> dict[str, str]``.

    Parallel-safe: each call writes under its own unique subdir of
    ``<run_dir>/proposals/``.
    """

    def __init__(
        self,
        model: str,
        run_dir: str | Path,
        *,
        objective: str | None = None,
        background: str | None = None,
        max_budget_usd: float | None = None,
        max_thinking_tokens: int | None = None,
        effort: str | None = None,
        sandbox: bool = True,
        subdir_prefix: str = "proposals",
    ) -> None:
        self.model = model
        self.run_dir = Path(run_dir)
        self.objective = objective
        self.background = background
        self.max_budget_usd = max_budget_usd
        self.max_thinking_tokens = max_thinking_tokens
        self.effort = effort
        self.sandbox = sandbox
        self.subdir_prefix = subdir_prefix
        self._total_cost: float = 0.0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0
        self._lock = threading.Lock()
        self._task_md_materialized = False

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return self._total_tokens_out

    def _ensure_task_md(self) -> None:
        """Write ``<run_dir>/task.md`` once per run.

        Mirrors the ``program.md`` pattern used by
        :class:`~gepa.omni.backends.claude_code.ClaudeCodeBackend`: long task
        context is materialized on disk and referenced by path from the CLI
        prompt. One file (not a dir) so the agent incurs a single read per
        iteration instead of one-per-section.

        Idempotent and safe across parallel first-calls — file writes are
        last-writer-wins with identical content.
        """
        if self._task_md_materialized:
            return
        if self.objective is None and self.background is None:
            self._task_md_materialized = True
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        sections: list[str] = ["# Task"]
        if self.objective:
            sections.append("## Objective\n" + self.objective.strip())
        if self.background:
            sections.append("## Background\n" + self.background.strip())
        (self.run_dir / "task.md").write_text("\n\n".join(sections) + "\n")
        self._task_md_materialized = True

    def _build_jail_mirror(self, subdir_real: Path, jail_root: Path) -> Path:
        """Mirror per-call read-only state into a TMPDIR-rooted jail.

        Layout: ``<jail_root>/{task.md,history.md,proposals/<sub.name>/...}``.
        The agent reads/writes inside the jail; ``_sync_jail_outputs`` copies
        ``new/*`` and ``plan.md`` back to ``subdir_real`` on return.
        """
        for fname in ("task.md", "history.md"):
            src = self.run_dir / fname
            if src.exists():
                shutil.copy2(src, jail_root / fname)
        jail_subdir = jail_root / "proposals" / subdir_real.name
        jail_subdir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(subdir_real, jail_subdir)
        return jail_subdir

    def _sync_jail_outputs(self, jail_subdir: Path, subdir_real: Path) -> None:
        """Copy the agent's prompt-mandated outputs (``new/*``, ``plan.md``)
        from the jail back to ``subdir_real``. Anything else is dropped with
        the tempdir."""
        new_src = jail_subdir / "new"
        new_dst = subdir_real / "new"
        if new_src.is_dir():
            new_dst.mkdir(parents=True, exist_ok=True)
            for f in new_src.iterdir():
                if f.is_file():
                    shutil.copy2(f, new_dst / f.name)
        plan_src = jail_subdir / "plan.md"
        if plan_src.exists():
            shutil.copy2(plan_src, subdir_real / "plan.md")

    def _allocate_subdir(self, iteration: int | None) -> Path:
        """Pick a collision-free subdir for this call under ``<run_dir>/<prefix>/``.

        Shape: ``iter_NNNNN-<uuid8>`` when the iteration id is known
        (standard path — supplied via ``metadata["iteration"]`` which is
        the 1-indexed on-disk iteration id); falls back to
        ``YYYYmmddTHHMMSS-<pid>-<uuid8>`` otherwise. The uuid suffix
        disambiguates parallel proposals that share the same iteration id
        (``num_parallel_proposals > 1``).
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        base = self.run_dir / self.subdir_prefix
        base.mkdir(parents=True, exist_ok=True)
        if iteration is not None:
            name = f"iter_{iteration:05d}-{uuid.uuid4().hex[:8]}"
        else:
            ts = time.strftime("%Y%m%dT%H%M%S")
            name = f"{ts}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        subdir = base / name
        subdir.mkdir(parents=True, exist_ok=False)
        return subdir

    def _copy_parent_iteration(self, parent_dir: Path, dest: Path) -> None:
        """Copy the slim parts of the parent iteration folder into ``dest``.

        Agent-locality convenience: copy ``meta.json``, ``val_scores.json``,
        ``plan.md`` and ``components/`` so the claude process can read the
        parent's program + reasoning next to its own working files without
        hopping across the run_dir. Intentionally excludes ``outputs/`` and
        ``trajectories/`` — those can be many MBs per iteration on adapters
        whose rollouts are big, and the agent can still reach them via
        ``iterations/<parent_id>/`` if it actually wants them.
        """
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("meta.json", "val_scores.json", "plan.md"):
            src = parent_dir / name
            if src.exists():
                (dest / name).write_text(src.read_text())
        comps_src = parent_dir / "components"
        if comps_src.is_dir():
            comps_dst = dest / "components"
            comps_dst.mkdir(parents=True, exist_ok=True)
            for f in comps_src.iterdir():
                if f.is_file():
                    (comps_dst / f.name).write_text(f.read_text())

    def _materialize(
        self,
        subdir: Path,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        parent_iteration_dir: Path | None,
    ) -> dict[str, str]:
        """Write reflection dataset + parent copy + index files into ``subdir``.

        Returns a map ``{component_name: safe_filename_stem}`` used to locate
        output files on disk. The parent candidate text is *not* re-written
        per-component — instead the whole parent iteration folder (slim
        parts) is copied into ``<subdir>/parent/`` via
        :meth:`_copy_parent_iteration`, so the agent has access to the
        parent's ``plan.md`` and scores as well as its component texts.
        """
        name_to_stem: dict[str, str] = {}
        used: set[str] = set()
        for i, name in enumerate(sorted(candidate.keys())):
            stem = _safe_component_filename(name, i)
            j = 0
            base_stem = stem
            while stem in used:
                j += 1
                stem = f"{base_stem}_{j}"
            used.add(stem)
            name_to_stem[name] = stem

        if parent_iteration_dir is not None:
            self._copy_parent_iteration(parent_iteration_dir, subdir / "parent")
        else:
            # First iteration or state unreadable: materialize parent texts
            # directly so the agent still has something to read, with no
            # ``plan.md`` / ``meta.json`` (absence is the signal).
            fallback_parent = subdir / "parent"
            fallback_parent.mkdir(parents=True, exist_ok=True)
            comps_dir = fallback_parent / "components"
            comps_dir.mkdir(parents=True, exist_ok=True)
            for name, text in candidate.items():
                (comps_dir / f"{name_to_stem[name]}.txt").write_text(text)

        reflection_dir = subdir / "reflection"
        reflection_dir.mkdir()
        for name in components_to_update:
            records = list(reflective_dataset.get(name, []))
            (reflection_dir / f"{name_to_stem[name]}.json").write_text(json.dumps(records, indent=2, default=str))

        new_dir = subdir / "new"
        new_dir.mkdir()

        return name_to_stem

    def _wrapper_prompt(
        self,
        subdir: Path,
        components: list[str],
        stems: dict[str, str],
        *,
        first_iteration: bool,
        parent_iter_id: int | None,
        root_dir: Path | None = None,
    ) -> str:
        """Program.md-style proposal brief.

        Long task context (objective, background, scoring rubric) lives in
        ``<root>/task.md`` rather than the CLI prompt. The agent is pointed
        at ``history.md`` and ``parent/`` for context. ``root_dir`` is the
        per-call jail mirror under TMPDIR when sandboxed, ``self.run_dir``
        otherwise; either way the prompt uses absolute paths.
        """
        run_abs = (root_dir or self.run_dir).resolve()
        sub_abs = subdir.resolve()
        task_md_exists = (self.objective is not None) or (self.background is not None)
        files_to_produce = "\n".join(f"- `{sub_abs}/new/{stems[name]}.md`  (component {name!r})" for name in components)
        task_section = (
            f"## Task\nThe problem statement and **scoring rubric** are in "
            f"`{run_abs}/task.md`. Read it first — it defines what a higher "
            "score means and, for graded problems, what the achievable range "
            "actually is. Do not assume a score of 1.0 is perfect unless "
            "`task.md` says so.\n\n"
            if task_md_exists
            else ""
        )
        if first_iteration:
            history_section = (
                f"## History\nThis is the first proposal in this run — there "
                f"is no `history.md` yet. Your `{sub_abs}/parent/` directory "
                "contains the **seed program** (no `plan.md`).\n\n"
            )
        else:
            history_section = (
                f"## History\nRead `{run_abs}/history.md` **first**. It's a "
                "chronological log of every prior iteration's short plan and "
                "score outcome (accepted or rejected). Use it to avoid "
                "repeating strategies that have already been tried and to "
                "build on what worked.\n\n"
            )
        # Enumerate concrete input paths per component, same pattern the
        # Outputs section uses. Passing literal ``<stem>`` placeholders
        # confused the agent — it Read the placeholder filename verbatim and
        # errored out. We have the real filenames in ``stems``; use them.
        parent_component_lines = "\n".join(
            f"   - `{sub_abs}/parent/components/{stems[name]}.txt`  (component {name!r})" for name in sorted(stems)
        )
        reflection_lines = "\n".join(
            f"   - `{sub_abs}/reflection/{stems[name]}.json`  (component {name!r})" for name in components
        )
        parent_section = (
            f"3. `{sub_abs}/parent/` — parent iteration (`plan.md`, "
            f"`meta.json`, `val_scores.json`, plus component files):\n"
            f"{parent_component_lines}"
        )
        # Iterations/ and pareto/ only exist on disk in the persistent
        # ``run_dir``. When sandboxed (``root_dir`` != ``self.run_dir``),
        # they're not mirrored into the jail — task.md, history.md, and the
        # per-call subdir are sufficient context. Drop those references.
        sandboxed = root_dir is not None and root_dir != self.run_dir
        if not sandboxed and parent_iter_id is not None:
            parent_section += (
                f"\n   For full rollout outputs see "
                f"`{run_abs}/iterations/{parent_iter_id:05d}/outputs/` + "
                f"`{run_abs}/iterations/{parent_iter_id:05d}/trajectories/`."
            )
        wider_state_section = (
            ""
            if sandboxed
            else (
                f"\n## Wider state (browse when useful)\n"
                f"- `{run_abs}/iterations/NNNNN/` — every past iteration "
                f"(accepted *and* rejected), with `meta.json`, "
                f"`components/`, `plan.md`, `reflective_dataset.json`, "
                f"`trace.json`. `00000/` is always the seed.\n"
                f"- `{run_abs}/pareto/` — current Pareto frontier(s) "
                f"keyed by iteration id.\n"
            )
        )
        return f"""\
You are proposing improved text for one or more components of a program.
Your goal is to raise the program's aggregate score on
future evaluations, based on the reflective feedback from past runs.

All inputs and outputs for **this** proposal call live under:
  `{sub_abs}/`
The shared run directory is at:
  `{run_abs}/`
Use these absolute paths in every Read/Write/Bash call — your cwd is
an isolated tempdir, not the run directory.

{task_section}{history_section}## Inputs (read in this order)
1. `{run_abs}/task.md` — problem statement + scoring rubric.
2. `{run_abs}/history.md` — chronological log of prior iterations' plans and scores.
{parent_section}
4. Per-example feedback for each component to update:
{reflection_lines}
{wider_state_section}

## Outputs you must produce
One improved-text file per component:
{files_to_produce}

Wrap the new component text in a single ```…``` fenced block. Anything
outside the block is treated as rationale and discarded by the engine.

**Also** write a short plan to `{sub_abs}/plan.md` — **≤50 words**, plain
prose. Describe what you're changing and why. This file is concatenated
with every prior iteration's plan into `history.md` so future proposers
can see your reasoning; keep it tight.

## Rules
- Write only inside `{sub_abs}/new/` and `{sub_abs}/plan.md`.
- Do not modify other iterations, state files, `task.md`, `history.md`,
  or another proposal's directory.
- Do not attempt to run any evaluations or verifications.
  The evaluation is done after you finished (by the outer engine).
"""

    def _read_new_texts(
        self,
        subdir: Path,
        candidate: dict[str, str],
        components_to_update: list[str],
        stems: dict[str, str],
    ) -> dict[str, str]:
        """Read one proposed text per updated component from ``<subdir>/new/``.

        Missing or empty files fall back to the parent candidate text — GEPA's
        acceptance criterion then rejects the no-op proposal on its own.
        """
        out: dict[str, str] = {}
        new_dir = subdir / "new"
        for name in components_to_update:
            stem = stems[name]
            path = new_dir / f"{stem}.md"
            if not path.exists():
                out[name] = candidate[name]
                continue
            body = path.read_text()
            extracted = _extract_fenced(body)
            out[name] = extracted if extracted else candidate[name]
        return out

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, str]:
        """Propose new component texts via the Claude Code CLI.

        ``metadata`` is an open context dict supplied by GEPA's reflective
        mutation path. Keys read (all optional):

        * ``"iteration"`` — the proposal's trace index (``state.i``). Used to
          name the proposals subdir, derive the on-disk iteration id for
          ``plan.md`` persistence, and decide whether ``history.md`` is
          worth rebuilding.
        * ``"parent_iteration_id"`` — on-disk iteration id of the parent
          candidate under ``iterations/NNNNN/``. Used to locate the parent
          iteration folder without scanning.

        When ``metadata`` is ``None`` or the keys are absent (direct
        non-reflective callers, tests, tooling) we fall back to UUID-only
        subdir naming and skip ``history.md`` + ``plan.md`` bookkeeping.
        """
        if self.max_budget_usd is not None and self._total_cost >= self.max_budget_usd:
            raise BudgetExhausted(
                f"claude_code_agent proposer spent ${self._total_cost:.2f} >= cap ${self.max_budget_usd:.2f}"
            )

        if not components_to_update:
            return {}

        meta = dict(metadata) if metadata else {}
        iteration_raw = meta.get("iteration")
        parent_iter_raw = meta.get("parent_iteration_id")
        iteration = int(iteration_raw) if iteration_raw is not None else None
        parent_iter_id = int(parent_iter_raw) if parent_iter_raw is not None else None

        self._ensure_task_md()
        # Regenerate history.md from the iterations/ tree every call —
        # state.save() at the top of each GEPA turn means everything
        # strictly before this iter is already on disk.
        if iteration is not None:
            self._rebuild_history_md()

        subdir = self._allocate_subdir(iteration)
        parent_dir: Path | None = None
        if parent_iter_id is not None:
            iter_dir = self.run_dir / "iterations" / f"{parent_iter_id:05d}"
            if iter_dir.is_dir():
                parent_dir = iter_dir
        stems = self._materialize(
            subdir,
            candidate,
            reflective_dataset,
            components_to_update,
            parent_iteration_dir=parent_dir,
        )
        # Seed owns iteration id 0; the very first loop proposal runs at id 1.
        first_iteration = iteration == 1
        session_id = str(uuid.uuid4())

        env = {**os.environ}
        env.pop("CLAUDECODE", None)
        env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")
        if self.max_thinking_tokens is not None:
            env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"
            env["MAX_THINKING_TOKENS"] = str(self.max_thinking_tokens)

        # Sandbox path: run claude inside a TMPDIR-rooted jail mirror of
        # this call's read-only state. bwrap and outer cwd are scoped to the
        # jail, never run_dir — closes the CLAUDE.md walk-up channel and
        # keeps sibling proposer outputs invisible. Outputs sync back to
        # ``subdir`` after claude returns.
        proc: subprocess.CompletedProcess[str]
        if self.sandbox:
            with tempfile.TemporaryDirectory(prefix="omni_gepa_cc_agent_") as jail_root_str:
                jail_root = Path(jail_root_str)
                agent_subdir = self._build_jail_mirror(subdir, jail_root)
                proc = self._run_claude(
                    agent_subdir,
                    components_to_update,
                    stems,
                    first_iteration=first_iteration,
                    parent_iter_id=parent_iter_id,
                    jail_root=jail_root,
                    env=env,
                    session_id=session_id,
                )
                self._sync_jail_outputs(agent_subdir, subdir)
        else:
            proc = self._run_claude(
                subdir,
                components_to_update,
                stems,
                first_iteration=first_iteration,
                parent_iter_id=parent_iter_id,
                jail_root=None,
                env=env,
                session_id=session_id,
            )

        copy_session_transcript(self.run_dir, session_id, subdir)

        payload: dict[str, Any] = {}
        stdout = (proc.stdout or "").strip()
        parse_error: str | None = None
        if stdout:
            try:
                payload = json.loads(stdout)
            except (json.JSONDecodeError, ValueError) as e:
                parse_error = f"{type(e).__name__}: {e}"
                payload = {}

        try:
            cost = float(payload.get("total_cost_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        usage = payload.get("usage") or {}
        try:
            tokens_in = int(usage.get("input_tokens", 0) or 0)
            tokens_out = int(usage.get("output_tokens", 0) or 0)
        except (TypeError, ValueError):
            tokens_in = tokens_out = 0

        with self._lock:
            self._total_cost += cost
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out

        # Surface claude failures. Without this, a non-zero exit (e.g. API
        # rate limit, sandbox setup failure, auth error) silently reports
        # zero cost/tokens and the proposer falls back to the parent
        # candidate — indistinguishable from a proposer that ran and
        # declined to change anything.
        is_error_payload = bool(payload.get("is_error"))
        empty_payload = not payload and not stdout
        if proc.returncode != 0 or parse_error or is_error_payload or empty_payload:
            stderr_tail = (proc.stderr or "")[-4000:]
            stdout_tail = (proc.stdout or "")[-4000:]
            reason = (
                f"returncode={proc.returncode}"
                + (f" parse_error={parse_error}" if parse_error else "")
                + (" is_error=true" if is_error_payload else "")
                + (" empty_output=true" if empty_payload else "")
            )
            try:
                (subdir / "claude_failure.json").write_text(
                    json.dumps(
                        {
                            "returncode": proc.returncode,
                            "parse_error": parse_error,
                            "is_error": is_error_payload,
                            "stderr_tail": stderr_tail,
                            "stdout_tail": stdout_tail,
                            "cost": cost,
                            "tokens_in": tokens_in,
                            "tokens_out": tokens_out,
                        },
                        indent=2,
                    )
                )
            except OSError:
                pass
            print(
                f"[ClaudeCodeAgentProposer] claude failure ({reason}) "
                f"subdir={subdir.name}; stderr: {stderr_tail.strip()[:400]}",
                file=sys.stderr,
                flush=True,
            )

        new_texts = self._read_new_texts(subdir, candidate, components_to_update, stems)

        # Copy the agent's plan.md into iterations/<iter_id>/plan.md so the
        # next proposer call — which regenerates history.md from disk —
        # sees it. ``iterations/<iter_id>/`` may not exist yet: GEPA's
        # state.save() writes it at the top of the next outer-loop turn,
        # but we create the dir here so the write is ordering-safe (state
        # save uses ``os.makedirs(..., exist_ok=True)``).
        if iteration is not None:
            self._persist_plan_md(subdir, iteration)

        return new_texts

    def _run_claude(
        self,
        agent_subdir: Path,
        components_to_update: list[str],
        stems: dict[str, str],
        *,
        first_iteration: bool,
        parent_iter_id: int | None,
        jail_root: Path | None,
        env: dict[str, str],
        session_id: str,
    ) -> subprocess.CompletedProcess[str]:
        """Build argv + invoke ``claude --print`` in the agent's working dir.

        ``jail_root`` is the bwrap-bound dir when sandboxed; ``None`` when
        not. ``agent_subdir`` is the per-call subdir the prompt points at —
        either inside the jail (sandboxed) or the real run_dir subdir.
        """
        wrapper = self._wrapper_prompt(
            agent_subdir,
            components_to_update,
            stems,
            first_iteration=first_iteration,
            parent_iter_id=parent_iter_id,
            root_dir=jail_root,
        )
        cwd = jail_root if jail_root is not None else self.run_dir
        cmd: list[str] = bwrap_prefix(cwd) if jail_root is not None else []
        cmd += [
            "claude",
            "--print",
            wrapper,
            "--output-format",
            "json",
            "--model",
            self.model,
            "--session-id",
            session_id,
            "--permission-mode",
            "bypassPermissions",
            DENY_WEB_TOOLS,
        ]
        if jail_root is not None:
            cmd.extend(claude_settings_args(jail_root))  # macOS Seatbelt fallback
        if self.max_thinking_tokens is None and self.effort is not None:
            cmd.extend(["--effort", self.effort])
        if self.max_budget_usd is not None:
            remaining = max(0.01, self.max_budget_usd - self._total_cost)
            cmd.extend(["--max-budget-usd", f"{remaining:.4f}"])

        return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)

    def __repr__(self) -> str:
        return (
            f"ClaudeCodeAgentProposer(model={self.model!r}, "
            f"run_dir={str(self.run_dir)!r}, cost=${self._total_cost:.2f})"
        )

    # ------------------------------------------------------------------
    # Plan + history bookkeeping
    # ------------------------------------------------------------------

    def _persist_plan_md(self, subdir: Path, iteration: int) -> None:
        """Copy the agent's ``plan.md`` into ``iterations/<iter_id>/plan.md``.

        Best-effort — if the agent didn't write a plan, silently skip.
        ``iterations/<iter_id>/`` is created if missing; GEPA's state.save
        populates the rest of that directory (meta.json, components/, ...)
        on the next outer-loop save pass without conflict.
        """
        plan_src = subdir / "plan.md"
        if not plan_src.exists():
            return
        dest = self.run_dir / "iterations" / f"{iteration:05d}" / "plan.md"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(plan_src.read_text())
        except OSError as e:
            print(
                f"[ClaudeCodeAgentProposer] failed to persist plan.md for iteration {iteration}: {e}",
                file=sys.stderr,
                flush=True,
            )

    def _rebuild_history_md(self) -> None:
        """Rewrite ``<run_dir>/history.md`` from iterations/*/ on disk.

        Produces one chronological block per completed iteration::

            ### Iteration N — accepted (parent: M)
            score: 0.42 -> 0.51 (subsample)
            plan:
            <verbatim plan body>

        Seed (``iterations/00000/``) is skipped: it has no plan and no
        parent. Merge (multi-parent) iterations show the first parent only
        for now.
        """
        iters_root = self.run_dir / "iterations"
        if not iters_root.exists():
            return
        blocks: list[str] = []
        for d in sorted(iters_root.iterdir()):
            if not d.is_dir():
                continue
            try:
                iter_id = int(d.name)
            except ValueError:
                continue
            if iter_id == 0:
                continue
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            accepted = meta.get("accepted")
            status = "accepted" if accepted else ("rejected" if accepted is False else "pending")
            parent_ids = meta.get("parent_iteration_ids") or []
            parent_str = f"parent: {parent_ids[0]}" if parent_ids else "parent: ?"
            before = meta.get("subsample_scores_before")
            after = meta.get("subsample_scores_after")
            if isinstance(before, list) and isinstance(after, list) and before and after:
                score_line = f"score: {sum(before):.4g} -> {sum(after):.4g} (subsample)"
            elif isinstance(after, list) and after:
                score_line = f"score: -> {sum(after):.4g}"
            else:
                score_line = "score: (unknown)"
            plan_path = d / "plan.md"
            plan_body: str | None = None
            if plan_path.exists():
                try:
                    plan_body = plan_path.read_text().strip()
                except OSError:
                    plan_body = None
            header = f"### Iteration {iter_id} — {status} ({parent_str})"
            parts = [header, score_line]
            if plan_body:
                parts.append("plan:")
                parts.append(plan_body)
            else:
                parts.append("plan: (none)")
            blocks.append("\n".join(parts))
        history_path = self.run_dir / "history.md"
        try:
            if blocks:
                history_path.write_text("\n\n".join(blocks) + "\n")
            elif history_path.exists():
                history_path.unlink()
        except OSError as e:
            print(
                f"[ClaudeCodeAgentProposer] failed to rebuild history.md: {e}",
                file=sys.stderr,
                flush=True,
            )
