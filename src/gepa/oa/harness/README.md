# OA agent harness abstraction (draft)

**Status: draft / RFC.** This branch makes the agent runtime behind
`optimize_anything`'s subprocess engines a config knob instead of a
hardcoded `claude --print` invocation, and adds
[Omnigent](https://omnigent.ai) as the second implementation — which brings
every backend Omnigent wraps (Claude Code, Codex, Goose, Qwen, Kimi,
Hermes, Pi, Cursor, Copilot, ACP agents) plus Omnigent's own OS sandboxing.
It is intended to (a) show the target shape inside gepa and (b) surface a
concrete ask-list for the Omnigent team.

## Usage

```python
gepa.optimize_anything(
    ...,
    config=OptimizeAnythingConfig(
        engine="meta_harness",
        harness={"type": "omnigent", "backend": "codex"},  # or "claude-code" (default)
    ),
)
```

## What an engine needs from an agent run

| Signal | claude-code source | omnigent source |
|---|---|---|
| cost (budget) | envelope `total_cost_usd` | session `total_cost_usd` |
| success/failure | exit code + `is_error` | session `status` / `last_task_error` |
| tokens | envelope `usage` | session `usage_by_model` |
| candidate artifacts | files in work dir | files in work dir (session `workspace`) |
| iteration/resume | `--session-id`/`--resume` | same Omnigent conversation, next message |
| transcript (diagnostic) | `~/.claude/projects/<slug>/<id>.jsonl` | items API export + native transcript via `external_session_id` |

## Sandboxing

Exactly one layer owns isolation. `harness="claude-code"` keeps gepa's
bwrap/Seatbelt jail from `oa/sandbox.py`. `harness="omnigent"` delegates to
Omnigent's sandbox via the agent bundle's `os_env.sandbox` block
(`sandbox=True` → Omnigent platform default; `sandbox_type="..."` → explicit
type/provider, including Omnigent's remote sandbox providers) — gepa never
wraps Omnigent processes in bwrap, because Omnigent's own bwrap seccomp
profile denies nested `CLONE_NEW*`.

## Removing the Claude-CLI paths (goal state)

The interface (`AgentRunSpec` → `AgentRunResult`) was shaped to absorb all
three inline claude call sites; `ClaudeCodeHarness` already contains the
unified argv/env/envelope logic. Migration status:

- [x] `engines/meta_harness.py` `_run_proposer` — routed through
  `harness.run()` on this branch (the reference migration).
- [ ] `engines/autoresearch.py` `_run_claude` — maps to
  `run(spec(resume=...))`; its budget-kill Popen polling loop needs either
  a harness-level `timeout_seconds` + smaller per-call budgets (as here) or
  a streaming hook on the interface.
- [ ] `proposers/claude_code_agent.py` `_run_claude` — maps directly; the
  proposer grows a `harness=` ctor kwarg.
- [ ] `oa/sandbox.py` claude-specific helpers fold into
  `harness/claude_code.py` once the three sites are migrated;
  `preflight_claude_engine` call sites move to `harness.preflight()`.

## Ask-list for the Omnigent team

Ordered by how much they block a non-draft integration:

1. **A supported local-embedded mode.** Spawning a usable server+runner
   pair currently requires internal surfaces (`omnigent.runner._entry`,
   `token_bound_runner_id`, `OMNIGENT_RUNNER_TUNNEL_TOKEN`), ported here
   from Omnigent's own test infra. A public `LocalRuntime()` (or
   `omnigent_client.LocalServer` growing a runner) would delete the most
   fragile code in `harness/omnigent.py`.
2. **Machine-readable one-shot mode** (`omni run -p ... --output-format
   json` printing result/session_id/cost/status): would let minimal
   integrations skip the SDK entirely.
3. **Restart-safe session status** — snapshot `status` is served from a
   process-local cache; a failed run can read back `"idle"`. Upstream PR:
   `fix/session-snapshot-live-status` (Shangyint/omnigent).
4. **`external_session_id` for headless backends** — the pointer to the
   wrapped agent's native transcript. Upstream PR:
   `feat/headless-external-session-id` (6 backends).
5. **Item-stream fidelity** so Omnigent's record can replace native
   transcripts entirely: persist SDK-native tool-call names/args (today
   they orphan), persist reasoning items (schema exists, no producer),
   emit `created_at` in the items API/export.
6. **Pre-emptive per-session budget cap** at session create (the
   `--max-budget-usd` equivalent); spend-cap policies exist but aren't on
   the REST policy allowlist.
7. **Session-create-time tool denylist** (WebFetch/WebSearch equivalent)
   without authoring a custom policy.
8. **Sub-agent recursion in `session export`** (children are separate
   sessions the export doesn't follow).

## Testing status

`ClaudeCodeHarness` reproduces the existing subprocess behavior and is
covered by the meta_harness path. `OmnigentHarness` is written against
Omnigent's SDK/test-infra recipes but has **not** been run end-to-end; it
is the concrete artifact for the conversation above, not yet a supported
backend.
