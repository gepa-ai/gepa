# Git-commit candidate mode

Optimize a **repository** with GEPA: a candidate is a git commit SHA rather than
a text string. GEPA evolves the repo by committing edits from a coding agent, and
your evaluator scores a working tree checked out at each candidate SHA. The best
candidate is a commit you can `git checkout`.

## Pieces

| Piece | What it does |
|---|---|
| `GitWorktreePool` (`gepa.oa.repo_pool`) | K warm worktrees over the repo's own object store; cheap incremental checkout between SHAs, concurrent-lease isolation (readers share, writers exclusive), gc-pinning of scored commits |
| `GitAgentProposer` (`gepa.oa.proposers.git_agent`) | Leases the parent SHA, runs *your* coding agent to edit files in place, mints the child commit, and rejects any edit outside the manifest (out-of-manifest paths, symlinks, gitlinks) |
| your `evaluate(candidate, ...)` | Leases the candidate SHA and scores its worktree (build / run / judge) |

## Enabling it

Set `git_commit` on `OptimizeAnythingConfig`:

```python
config = OptimizeAnythingConfig(
    engine="gepa",
    git_commit={
        "repo_dir": "/path/to/repo",
        "manifest_globs": ["src"],        # the editable surface + anti-reward-hacking gate
        "worktree_pool": pool,             # a RepoCandidatePool (GitWorktreePool, or your own)
        "agent": my_coding_agent,          # (slot_dir, ctx) -> None; edits files in place
    },
)
```

The engine builds a `GitAgentProposer` from `agent` and turns off text-merge
(SHAs can't be merged by component text). To use your own proposer instead, set
`engine_config={"reflection": {"custom_candidate_proposer": ...}}` and omit
`agent`.

## Bring your own agent

The `agent` is any callable `(slot_dir: Path, ctx: ProposalContext) -> None`. It
edits files in the leased worktree; the pool commits the manifest afterwards.
`ctx` carries `reflective_dataset` (per-example scores/feedback from the last
evaluation), `parent_sha`, and `manifest_globs`. Bind a real coding CLI by
shelling out with `cwd=slot_dir`, or drop in a scripted mutation for tests.

## Run the demo

```bash
python examples/git_mode/optimize_repo.py
```

It uses a scripted agent (increment an integer in a file), so it runs with no API
key and prints the seed and best commit SHAs.
