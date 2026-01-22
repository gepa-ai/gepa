BACKGROUND = """
Signatures:
- Signatures define tasks declaratively through input/output fields and explicit instructions.
- They serve as blueprints for what the LM needs to accomplish.

Signature Types:
- Simple signatures: Specified as strings like "input1, ..., inputN -> output1, ..., outputM" (e.g., "topic -> tweet").
- Typed signatures: Create a subclass of dspy.Signature with a detailed docstring that includes task instructions, common pitfalls, edge cases, and successful strategies. Define fields using dspy.InputField(desc="...", type=...) and dspy.OutputField(desc="...", type=...) with pydantic types such as str, List[str], Literal["option1", "option2"], or custom classes.

Modules:
- Modules specify __how__ to solve the task defined by a signature.
- They are composable units inspired by PyTorch layers, using language models to process inputs and produce outputs.
- Inputs are provided as keyword arguments matching the signature's input fields.
- Outputs are returned as dspy.Prediction objects containing the signature's output fields.
- Key built-in modules:
  - dspy.Predict(signature): Performs a single LM call to directly generate the outputs from the inputs.
  - dspy.ChainOfThought(signature): Performs a single LM call that first generates a reasoning chain, then the outputs (adds a 'reasoning' field to the prediction).
  - Other options: dspy.ReAct(signature) for reasoning and acting, or custom chains.
- Custom modules: Subclass dspy.Module. In __init__, compose sub-modules (e.g., other Predict or ChainOfThought instances). In forward(self, **kwargs), define the data flow: call sub-modules, execute Python logic if needed, and return dspy.Prediction with the output fields.

Example Usage:
```
# Simple signature
simple_signature = "question -> answer"

# Typed signature
class ComplexSignature(dspy.Signature):
    \"\"\"
    <Detailed instructions for completing the task: Include steps, common pitfalls, edge cases, successful strategies. Include domain knowledge...>
    \"\"\"
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="Concise and accurate answer")

# Built-in module
simple_program = dspy.Predict(simple_signature)  # or dspy.ChainOfThought(ComplexSignature)

# Custom module
class ComplexModule(dspy.Module):
    def __init__(self):
        self.reasoner = dspy.ChainOfThought("question -> intermediate_answer")
        self.finalizer = dspy.Predict("intermediate_answer -> answer")
    
    def forward(self, question: str):
        intermediate = self.reasoner(question=question)
        final = self.finalizer(intermediate_answer=intermediate.intermediate_answer)
        return dspy.Prediction(answer=final.answer, reasoning=intermediate.reasoning) # dspy.ChainOfThought returns 'reasoning' in addition to the signature outputs.

complex_program = ComplexModule()
```

DSPy Improvement Strategies:
1. Analyze traces for LM overload: If a single call struggles (e.g., skips steps or hallucinates), decompose into multi-step modules with ChainOfThought or custom logic for stepwise reasoning.
2. Avoid over-decomposition: If the program is too fragmented, consolidate related steps into fewer modules for efficiency and coherence.
3. Refine signatures: Enhance docstrings with actionable guidance from traces—address specific errors, incorporate domain knowledge, document edge cases, and suggest reasoning patterns. Ensure docstrings are self-contained, as the LM won't have access external traces during runtime.
4. Balance LM and Python: Use Python for symbolic/logical operations (e.g., loops, conditionals); delegate complex reasoning or generation to LM calls.
5. Incorporate control flow: Add loops, conditionals, sub-modules in custom modules if the task requires iteration (e.g., multi-turn reasoning, selection, voting, etc.).
6. Leverage LM strengths: For code-heavy tasks, define signatures with 'code' outputs, extract and execute the generated code in the module's forward pass.

Assignment:
- Think step-by-step: First, deeply analyze the current code, traces, and feedback to identify failure modes, strengths, and opportunities.
- Create a concise checklist (3-7 bullets) outlining your high-level improvement plan, focusing on conceptual changes (e.g., "Decompose step X into a multi-stage module").
- Then, propose a drop-in replacement code that instantiates an improved 'program' object.
- Ensure the code is modular, efficient, and directly addresses feedback.
- Output everything in a single code block using triple backticks—no additional explanations, comments, or language markers outside the block.
- The code must be a valid, self-contained Python script with all necessary imports, definitions, and assignment to 'program'.

Output Format:
- Start with the checklist in plain text (3-7 short bullets).
- Follow immediately with one code block in triple backticks containing the complete Python code, including assigning a `program` object.
"""
