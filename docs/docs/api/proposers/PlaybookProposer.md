# PlaybookProposer

`PlaybookProposer` grows instruction components by appending rule bullets based
on reflective examples. Existing component text is preserved verbatim. Pass it
through the existing `custom_candidate_proposer` hook:

```python
import gepa
from gepa.lm import LM
from gepa.proposer import PlaybookProposer

proposer = PlaybookProposer(
    reflection_lm=LM("openai/gpt-4.1"),
    max_additions_per_step=3,
    avoid_redundancy=True,
)
result = gepa.optimize(
    seed_candidate={"system_prompt": "Answer the question correctly."},
    trainset=trainset,
    valset=valset,
    adapter=adapter,  # An adapter without its own propose_new_texts implementation.
    custom_candidate_proposer=proposer,
    max_metric_calls=100,
)
```

The LM is supplied to the proposer explicitly: custom proposers own proposal
generation and do not receive the optimizer's `reflection_lm` argument. A plain
callable implementing the `LanguageModel` protocol also works.

The response must be JSON, for example
`{"system_prompt": ["Check units before returning a numerical answer."]}`.
The proposer appends `- Check units before returning a numerical answer.` locally;
the LM never supplies a replacement for the existing text. The addition limit
applies across all selected components in a single proposal. Empty lists or
omitted keys leave the corresponding components unchanged. Components without
reflective examples are skipped. Invalid responses raise `ValueError` and use
the optimizer's existing `raise_on_exception` handling.

With `avoid_redundancy=True`, exact duplicate lines/bullets are skipped locally
and the prompt asks the LM to avoid paraphrases. This does not guarantee semantic
uniqueness. Appending also does not guarantee a better score: the usual
evaluation and acceptance criteria still decide whether to keep the candidate.
Long runs can grow the prompt, so choose an appropriate evaluation budget.

::: gepa.proposer.PlaybookProposer
    handler: python
    options:
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
