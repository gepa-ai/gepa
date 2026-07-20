"""Minimal GEPA + PromptBreeder co-evolution example."""

import gepa

trainset = [
    {
        "input": "He walked to the bank.",
        "answer": "The sentence is ambiguous between a financial bank and a river bank.",
        "additional_context": {
            "policy": "When context is insufficient, identify plausible senses and ask a targeted clarification.",
        },
    },
    {
        "input": "Jordan told Alex that he was late.",
        "answer": "The pronoun 'he' has an unclear referent.",
        "additional_context": {
            "policy": "Do not silently choose a referent when evidence is insufficient.",
        },
    },
]

seed_candidate = {
    "system_prompt": (
        "Analyze the student's sentence and answer accurately. "
        "Use the available context and explain uncertainty when necessary."
    )
}

result = gepa.optimize(
    seed_candidate=seed_candidate,
    trainset=trainset,
    valset=trainset,
    task_lm="openai/gpt-4.1-mini",
    reflection_lm="openai/gpt-4.1",
    prompt_breeder_config=gepa.PromptBreederConfig(
        operator_weights={
            "zero_order": 0.25,
            "hypermutation": 0.45,
            "lineage": 0.30,
        },
        lineage_size=6,
        elite_pool_size=8,
        exploration_rate=0.15,
        seed=7,
    ),
    max_metric_calls=100,
    run_dir="runs/ambiguity_prompt_breeder",
)

print(result.best_candidate)
