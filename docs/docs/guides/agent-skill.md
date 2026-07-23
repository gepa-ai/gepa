# Using the GEPA Agent Skill

GEPA ships as an **[Agent Skill](https://agentskills.io/)** — a self-contained instruction bundle
(`SKILL.md` + references + templates) that teaches a coding agent how to use
[`gepa.optimize_anything`](../api/optimize_anything/optimize_anything.md) correctly: how to pick the optimization
mode, write a feedback-rich evaluator, size the budget, choose an engine, and avoid the common
pitfalls. It lives in this repository at
[`.claude/skills/gepa-optimize-anything/`](https://github.com/gepa-ai/gepa/tree/main/.claude/skills/gepa-optimize-anything).

## When an agent should use it

When you ask an agent to **auto-optimize, tune, or search over any scorable text artifact** — a
prompt, a program/CUDA kernel, a config, a regex/SQL query, an agent scaffold, or an encoded search
solution — where you can provide a score (an objective metric or an LLM-as-judge rating) and, ideally,
feedback. The skill covers all three modes (single-task, multi-task, generalization) and the engines
exposed by `optimize_anything` (GEPA, best-of-N, AutoResearch, MetaHarness).

## Installation

**In a clone of this repo** — no install needed. Claude Code auto-discovers the skill, and so do other
agents that read `.claude/skills/` (Cursor, VS Code/Copilot, Codex, Gemini CLI, Goose, …). Just ask
the agent to optimize something.

**In any other project** — install it as a Claude Code plugin from this repo's marketplace:

```bash
/plugin marketplace add gepa-ai/gepa
/plugin install gepa-optimize-anything@gepa
```

The skill then loads in every project, namespaced as `gepa-optimize-anything`.

## Other agents / ecosystems

The same `SKILL.md` works unmodified in any agent that reads the `.claude/skills/` convention. For
agents that use a different skills directory (e.g. Windsurf's `.windsurf/skills/`), copy or symlink the
folder. For agents with no skills concept, point them at this page and at the
[`optimize_anything` reference](../api/optimize_anything/optimize_anything.md).

## What's inside the skill

- `SKILL.md` — the entry point: overview, the three modes, budget sizing, termination, a minimal
  runnable example, and the critical gotchas.
- `references/` — deeper docs: the API surface, writing evaluators (feedback design, LLM-as-judge,
  multi-objective), experiment tracking, and a full gotchas list.
- `templates/optimize_prompt.py` — a copy-paste starting point.
- `scripts/preflight.py` — validates credentials and prerequisites before a long run.
