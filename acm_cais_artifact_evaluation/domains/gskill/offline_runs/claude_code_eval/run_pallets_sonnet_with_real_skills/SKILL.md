---
name: pallets-jinja-bugfix
description: Expert bug-fixing skills optimized for the pallets/jinja repository. Contains repo-specific patterns, diagnostic strategies, and step-by-step workflows for efficiently finding and resolving bugs in this codebase. Use when debugging issues, fixing failing tests, or resolving reported bugs.
---

TERMINAL / BASH WORKFLOW
- Always operate on the main repo at /testbed. Before/after changes, confirm with:
  - git -C /testbed status
  - git -C /testbed diff
  (Avoid running git in nested directories like /testbed/src/jinja2 unless you’ve confirmed /testbed itself isn’t the git root.)
- Quick codebase navigation patterns:
  - ls /testbed/src/jinja2
  - find /testbed/src/jinja2 -maxdepth 1 -type f | sort
  - grep -RIn "revindex" /testbed/src/jinja2
  - grep -RIn "parse_filter" /testbed/src/jinja2
- When output is too long, slice it:
  - sed -n '900,980p' /testbed/src/jinja2/parser.py
  - head -n 60 file ; tail -n 60 file

REPO-SPECIFIC KNOWLEDGE (JINJA2 IN THIS REPO)
- The package under test is the in-repo implementation: /testbed/src/jinja2 (not the system/site-packages install).
  - Do NOT “fix” by moving/renaming /testbed/src/jinja2 or by pip-installing a different Jinja2 version; pytest expects to import the local package.
- Common bug locations:
  - Parsing / filters / tests: src/jinja2/parser.py
  - Loop variables (loop.index, loop.revindex, etc.): src/jinja2/runtime.py (LoopContext)
  - Environment compilation and template loading: src/jinja2/environment.py
  - Tokenization: src/jinja2/lexer.py
- Tests live under /testbed/tests. Use them to confirm behavior and prevent regressions.

DEBUGGING STRATEGY (REPEATABLE)
- Reproduce first with a minimal snippet using the local code:
  - python -c "from jinja2 import Environment; ..."
- If parsing fails, differentiate lexer vs parser:
  - env.lex(source) confirms tokens (pipe/name/lparen/etc.)
  - If lexer tokens look right, inspect parser methods (e.g., parse_filter) and compare logic to expected grammar.
- For loop variable bugs:
  - Render a tiny template that prints loop.index/index0/revindex/revindex0/last to pin down the exact off-by-one and where it originates.
  - Then search for the corresponding property in LoopContext (runtime.py).
- Do not apply runtime monkeypatches as the “fix”. Implement the correction in library source and add/adjust tests.

TESTING PRACTICES
- Run the smallest relevant tests while iterating:
  - pytest -q
  - pytest -q -k "revindex" (or other keyword)
- After fixing, run full suite:
  - pytest
- If a “fix” makes imports fail (ModuleNotFoundError), undo it; the repo’s local package must remain importable at /testbed/src/jinja2.

CODE EDITING / PATCH HYGIENE
- Prefer direct file edits in /testbed/src/jinja2/*.py and verify with:
  - python -m compileall /testbed/src/jinja2
- Ensure your work produces an actual patch in the main repo:
  - git -C /testbed diff must show changes before finishing.
  - If diff is empty, you likely edited the wrong location or used a command (like git checkout in a different git root) that didn’t affect /testbed.
- Avoid relying on git commits (may fail due to missing user.name/user.email and is unnecessary for evaluation). Just edit files.
- When inspecting changes from a suspicious commit, don’t revert via package removal. Instead, restore correct logic by editing the affected function and add a regression test.

COMMON MISTAKES TO AVOID (LEARNED FROM FAILURES)
- Don’t “fix” by renaming/removing /testbed/src/jinja2 (breaks pytest imports).
- Don’t “fix” by pip installing another version (tests target local source).
- Don’t stop at a successful manual snippet; always convert the workaround into a source change + tests, and confirm git diff in /testbed shows the patch.
- Don’t emit overly large diagnostic output; use sed/head/tail to focus on relevant lines.