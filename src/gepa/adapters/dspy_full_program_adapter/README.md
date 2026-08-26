# DSPy Full Program Adapter

This adapter lets GEPA evolve entire DSPy programs—including signatures, modules, and control flow—not just prompts or instructions.

## Usage

First, install DSPy (version 3.0.2 or higher) with: `pip install 'dspy>=3.0.2'`.

Now, let's use GEPA to generate a DSPy program to solve MATH (benchmark). We start with a very very simple DSPy program `dspy.ChainOfThought("question -> answer")`:
```python
from gepa import optimize
from gepa.adapters.dspy_full_program_adapter.full_program_adapter import DspyAdapter
import dspy

# Standard DSPy metric function
def metric_fn(example, pred, trace=None):
    ...

# Start with a basic program. This code block must export a `program` that shows how the task should be performed
seed_program = """import dspy
program = dspy.ChainOfThought("question -> answer")"""

# Run optimization
reflection_lm = dspy.LM(model="openai/gpt-4.1", max_tokens=32000) # <-- This LM will only be used to propose new DSPy programs
result = optimize(
    seed_candidate={"program": seed_program},
    trainset=train_data,
    valset=val_data,
    adapter=DspyAdapter(
        task_lm=dspy.LM(model="openai/gpt-4.1-nano", max_tokens=32000), # <-- This LM will be used for the downstream task
        metric_fn=metric_fn,
        reflection_lm=lambda x: reflection_lm(x)[0],
    ),
    max_metric_calls=2000,
)

# Get the evolved program
optimized_program_code = result.best_candidate["program"]
print(optimized_program_code)
```

Using dspy.ChainOfThought with GPT-4.1 Nano achieves a score of **67%**, while the following GEPA-optimized program boosts performance to **93%**!
```
import dspy
from typing import Optional

class MathQAReasoningSignature(dspy.Signature):
    """
    Solve the given math word problem step by step, showing all necessary reasoning and calculations.
    - First, provide a clear, detailed, and logically ordered reasoning chain, using equations and algebraic steps as needed.
    - Then, extract the final answer in the required format, strictly following these rules:
        * If the answer should be a number, output only the number (no units, unless explicitly requested).
        * If the answer should be an algebraic expression, output it in LaTeX math mode (e.g., \frac{h^2}{m}).
        * Do not include explanatory text, units, or extra formatting in the answer field unless the question explicitly requests it.
    Common pitfalls:
        - Including units when not required.
        - Restating the answer with extra words or formatting.
        - Failing to simplify expressions or extract the final answer.
    Edge cases:
        - If the answer is a sum or list, output only the final value(s) as required.
        - If the answer is an expression, ensure it is fully simplified.
    Successful strategies:
        - Use step-by-step algebraic manipulation.
        - Double-check the final answer for correct format and content.
    """
    question: str = dspy.InputField(desc="A math word problem to solve.")
    reasoning: str = dspy.OutputField(desc="Step-by-step solution, with equations and logic.")
    answer: str = dspy.OutputField(desc="Final answer, strictly in the required format (see instructions).")

class MathQAExtractSignature(dspy.Signature):
    """
    Given a math word problem and a detailed step-by-step solution, extract ONLY the final answer in the required format.
    - If the answer should be a number, output only the number (no units, unless explicitly requested).
    - If the answer should be an algebraic expression, output it in LaTeX math mode (e.g., \frac{h^2}{m}).
    - Do not include explanatory text, units, or extra formatting in the answer field unless the question explicitly requests it.
    - If the answer is a sum or list, output only the final value(s) as required.
    """
    question: str = dspy.InputField(desc="The original math word problem.")
    reasoning: str = dspy.InputField(desc="A detailed, step-by-step solution to the problem.")
    answer: str = dspy.OutputField(desc="Final answer, strictly in the required format.")

class MathQAModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.reasoner = dspy.ChainOfThought(MathQAReasoningSignature)
        self.extractor = dspy.Predict(MathQAExtractSignature)

    def forward(self, question: str):
        reasoning_pred = self.reasoner(question=question)
        extract_pred = self.extractor(question=question, reasoning=reasoning_pred.reasoning)
        return dspy.Prediction(
            reasoning=reasoning_pred.reasoning,
            answer=extract_pred.answer
        )

program = MathQAModule()
```

A fully executable notebook to run this example is in [src/gepa/examples/dspy_full_program_evolution/example.ipynb](../../examples/dspy_full_program_evolution/example.ipynb)

## Reflective traces

`make_reflective_dataset` builds the examples that the reflection LM sees. Each example includes the program-level inputs and outputs, plus a `Program Trace` of selected predictor calls, in execution order.

- Consecutive calls to the same predictor are collapsed to the last call only when ReAct's `trajectory` string grew as a prefix. Independent repeats are kept, including History inputs (empty, shared-prefix, growing conversations, and unrelated sessions of different lengths) and prefix-related or newline-appended strings on other fields such as `context` or `document`.
- Interleaved calls such as generate, critique, generate-again are kept in that order. The first generation is not dropped, and the critique is not moved after the revision.
- A `FailedPrediction` (format parse error) stays in the trace so the raw completion is visible. Calls before and after it are still selected.

That removes the repeated ReAct prefixes that previously sent the same trajectory into reflection many times. It does not drop a later module in a pipeline (for example the reasoner and extractor in the MATH program above). A later call does not always contain the earlier module's inputs and outputs, so both stay in the trace.

