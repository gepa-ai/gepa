"""
Reproduction for GEPA #97:
Reflective dataset contains potentially redundant data for ReAct agent.

This is a lightweight unit-level reproduction that does not call an LLM.
It builds a fake eval_batch.trajectories object shaped like what
DspyAdapter.make_reflective_dataset expects, then shows that the function
keeps multiple cumulative trace entries in Program Trace.

Issue:
https://github.com/gepa-ai/gepa/issues/97
"""

from gepa.adapters.dspy_full_program_adapter.full_program_adapter import DspyAdapter


class FakeSignature:
    def equals(self, other):
        return True


class FakeModule:
    signature = FakeSignature()


class FakePredictor:
    signature = FakeSignature()


class FakeProgram:
    def named_predictors(self):
        return [("react.react", FakePredictor())]


class FakeExample:
    def inputs(self):
        return {"question": "What is 2+2?"}


class FakeEvalBatch:
    pass


adapter = object.__new__(DspyAdapter)
adapter.build_program = lambda candidate: (FakeProgram(), None)

# Trace 1: first step only
trace_1_inputs = {
    "question": "What is 2+2?",
}
trace_1_outputs = {
    "next_thought": "I should use the calculator.",
    "next_tool_name": "calculator",
    "next_args": "2+2",
}

# Trace 2: cumulative trace that repeats prior context/output information
trace_2_inputs = {
    "question": "What is 2+2?",
    "previous_step": "I should use the calculator. calculator 2+2",
}
trace_2_outputs = {
    "observation": "4",
    "next_thought": "Now I can answer.",
}

# Trace 3: final cumulative trace that again carries earlier information forward
trace_3_inputs = {
    "question": "What is 2+2?",
    "full_trajectory_so_far": (
        "I should use the calculator. calculator 2+2. "
        "observation: 4. Now I can answer."
    ),
}
trace_3_outputs = {
    "answer": "4",
}

eval_batch = FakeEvalBatch()
eval_batch.trajectories = [
    {
        "trace": [
            (FakeModule(), trace_1_inputs, trace_1_outputs),
            (FakeModule(), trace_2_inputs, trace_2_outputs),
            (FakeModule(), trace_3_inputs, trace_3_outputs),
        ],
        "example": FakeExample(),
        "prediction": {"answer": "4"},
        "score": 1.0,
    }
]

reflective_dataset = DspyAdapter.make_reflective_dataset(
    adapter,
    candidate={"program": "fake program"},
    eval_batch=eval_batch,
    components_to_update=["program"],
)

program_trace = reflective_dataset["program"][0]["Program Trace"]

print("GEPA #97 reproduction")
print("=" * 60)
print(f"Number of trace entries in reflective dataset: {len(program_trace)}")
print()

for i, entry in enumerate(program_trace, start=1):
    print(f"TRACE ENTRY {i}: {entry['Called Module']}")
    print(f"Inputs: {entry['Inputs']}")
    print(f"Generated Outputs: {entry['Generated Outputs']}")
    print()

print("Observation:")
print(
    "make_reflective_dataset keeps all 3 trace entries in Program Trace. "
    "For cumulative/ReAct-style traces, later entries may already include "
    "information from earlier steps, which causes repeated context to be sent "
    "to the reflection LM."
)
