# optimize

## Resuming from `run_dir`

If `run_dir` already contains `gepa_state.bin`, GEPA loads that state and continues:

- The saved seed is **not** re-evaluated on the full valset.
- A different `seed_candidate` (one not already in the saved pool) is full-valset-evaluated and added as a new candidate parented on the saved seed, like an accepted proposal.
- The same `seed_candidate` as a saved candidate does no extra seed evaluation.

::: gepa.api.optimize
    handler: python
    options:
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
        members_order: source
        show_signature_annotations: true
