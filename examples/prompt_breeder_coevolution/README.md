# PromptBreeder Co-Evolution Example

This example shows how to couple GEPA's task-prompt evolution with PromptBreeder's reflection-genome evolution on a small ambiguity-resolution task.

It demonstrates:

- task-prompt mutation with GEPA
- reflection-process mutation with PromptBreeder
- acceptance-gated mutator inheritance
- checkpointable breeder state

Run it after configuring model credentials for your provider:

```bash
python main.py
```

The optimizer will evolve a system prompt that handles ambiguous language more cautiously while simultaneously evolving the reflection instructions used to propose future prompt changes.
