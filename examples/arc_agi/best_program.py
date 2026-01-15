program = r'''
import dspy
from typing import List, Optional, Any, Dict, Tuple, Callable
import pydantic
import re
import traceback
import copy

MATRIX = List[List[int]]

class TrainingExample(pydantic.BaseModel):
    input: MATRIX
    output: MATRIX

class SolveTaskSignature(dspy.Signature):
    """
    Solve ARC-style grid transformations by learning a function from examples.

    Inputs:
    - training_examples: A list of (input, output) grid pairs that demonstrate the task. Grids are integer matrices.
    - test_inputs: Grids to transform using the learned task.

    Output:
    - test_outputs: Exact grids corresponding to each test input.

    Approach:
    - Induce a general, deterministic transformation as Python code: def transform(grid: List[List[int]]) -> List[List[int]].
    - Common patterns:
      1) Separator rows/columns: entire rows/cols of a single color often partition the grid; keep separators unchanged.
      2) Block-wise aggregation: when grid is partitioned into kxk blocks by separators, fill each block by a statistic (e.g., majority including 0) inferred from examples.
      3) Mask algebra: when two subgrids (e.g., split by a row of constant color) define masks, combine with boolean logic (OR/AND/XOR) and map nonzero to a target color.
      4) Noise cleanup: replace a minority/noise color based on orthogonal neighbor majority of a primary color; otherwise drop to background.
    - Pitfalls:
      - Ensure exact equality with all training outputs; no partial credit.
      - Handle arbitrary grid sizes consistent with the pattern inferred from training pairs.
      - Do not hardcode coordinates or sizes; infer from structure (e.g., positions of separator lines).
      - Preserve separator rows/columns exactly when present.
    - Constraints:
      - Pure Python on lists; no imports; deterministic; O(n*m) to O(n*m*small) time.
    """
    training_examples: List[TrainingExample] = dspy.InputField(desc="Input/output example pairs describing the task.")
    test_inputs: List[MATRIX] = dspy.InputField(desc="Inputs to transform after learning from examples.")
    test_outputs: List[MATRIX] = dspy.OutputField(desc="Outputs for test_inputs produced by the learned transform.")

class SynthesizeTransform(dspy.Signature):
    """
    Write valid, self-contained Python code that defines:
        def transform(grid: List[List[int]]) -> List[List[int]]:
            ...
            return out_grid

    Requirements:
    - Use only built-in Python (lists/loops/dicts/sets); no imports.
    - May define small helper functions above transform.
    - Must be general to similar-sized grids and structures; do NOT hardcode absolute indices from training examples.
    - Preserve separator rows/columns if present.
    - For block aggregation, infer block sizes from constant-color separator lines/columns.
    - For mask logic, infer how to combine masks and the output color mapping from examples.
    - For noise cleanup, infer primary vs. noise colors and neighbor rules from examples.

    Guidance from previous attempt:
    {hint}

    Return ONLY code text containing the def transform(...) function (and optional helpers), nothing else.
    """
    training_examples: List[TrainingExample] = dspy.InputField(desc="Training pairs to infer the rule.")
    hint: str = dspy.InputField(desc="Feedback on prior failures and additional guidance.")
    code: str = dspy.OutputField(desc="Python code that defines transform(grid) and helper functions.")

def _extract_code_block(s: str) -> str:
    if s is None:
        return ""
    # Try to extract triple-backticked python code if present
    m = re.findall(r"```(?:python)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    return s.strip()

def _load_transform_func(code: str) -> Tuple[Optional[Callable[[MATRIX], MATRIX]], Optional[str]]:
    try:
        # Sanitize and ensure 'def transform(' exists
        if "def transform(" not in code:
            return None, "No transform(grid) function found."
        safe_globals: Dict[str, Any] = {
            "__builtins__": {
                "range": range,
                "len": len,
                "min": min,
                "max": max,
                "sum": sum,
                "enumerate": enumerate,
                "abs": abs,
                "all": all,
                "any": any,
                "sorted": sorted,
                "zip": zip,
                "set": set,
                "list": list,
                "dict": dict
            }
        }
        safe_locals: Dict[str, Any] = {}
        exec(code, safe_globals, safe_locals)
        fn = safe_locals.get("transform") or safe_globals.get("transform")
        if not callable(fn):
            return None, "transform is not callable."
        return fn, None
    except Exception as e:
        return None, f"Code exec error: {e}\n{traceback.format_exc()}"

def _grids_equal(a: MATRIX, b: MATRIX) -> bool:
    if len(a) != len(b):
        return False
    for r in range(len(a)):
        if a[r] != b[r]:
            return False
    return True

def _summarize_mismatch(gold: MATRIX, pred: MATRIX, max_points: int = 10) -> str:
    pts = []
    R = min(len(gold), len(pred))
    for r in range(R):
        C = min(len(gold[r]), len(pred[r]))
        for c in range(C):
            if gold[r][c] != pred[r][c]:
                pts.append(f"({r},{c}): expected {gold[r][c]}, got {pred[r][c]}")
                if len(pts) >= max_points:
                    break
        if len(pts) >= max_points:
            break
    return "; ".join(pts) if pts else "shape or structural mismatch"

class CodeSynthesisSolver(dspy.Module):
    def __init__(self, attempts: int = 4):
        super().__init__()
        self.attempts = attempts
        self.codegen = dspy.ChainOfThought(SynthesizeTransform)

    def _verify_on_training(self, fn: Callable[[MATRIX], MATRIX], training_examples: List[TrainingExample]) -> Tuple[bool, Optional[str]]:
        failures = []
        for idx, ex in enumerate(training_examples):
            try:
                pred = fn(copy.deepcopy(ex.input))
            except Exception as e:
                return False, f"Runtime error on example {idx}: {e}"
            if not _grids_equal(pred, ex.output):
                mm = _summarize_mismatch(ex.output, pred)
                failures.append(f"Ex {idx} mismatch: {mm}")
                if len(failures) >= 3:
                    break
        if failures:
            return False, " | ".join(failures)
        return True, None

    def forward(self, training_examples: List[TrainingExample], test_inputs: List[MATRIX]) -> dspy.Prediction:
        hint = (
            "Focus on inferring a general rule from ALL training pairs. "
            "Verify your transform reproduces every training output exactly before finalizing."
        )
        last_error = None
        for attempt in range(1, self.attempts + 1):
            pred = self.codegen(training_examples=training_examples, hint=hint)
            code_text = pred.code if hasattr(pred, "code") and pred.code else ""
            code_text = _extract_code_block(code_text)
            fn, load_err = _load_transform_func(code_text)
            if fn is None:
                last_error = load_err or "Unknown code loading error."
                hint = (
                    f"Attempt {attempt} failed to load transform: {last_error}. "
                    "Return ONLY valid Python code that defines def transform(grid)."
                )
                continue
            ok, err = self._verify_on_training(fn, training_examples)
            if ok:
                # Apply to test inputs
                outputs: List[MATRIX] = []
                for i, g in enumerate(test_inputs):
                    try:
                        out = fn(copy.deepcopy(g))
                    except Exception as e:
                        # If test-time error occurs, treat as failure and refine
                        last_error = f"Runtime error on test input {i}: {e}"
                        ok = False
                        break
                    outputs.append(out)
                if ok:
                    return dspy.Prediction(test_outputs=outputs)
                else:
                    hint = (
                        f"Transform passed training but failed on test due to: {last_error}. "
                        "Make the transform more robust while preserving training behavior."
                    )
                    continue
            else:
                last_error = err or "Mismatch without details."
                hint = (
                    f"Attempt {attempt} produced incorrect outputs on training: {last_error}. "
                    "Refine the code: ensure exact equality, preserve separators, infer block sizes, "
                    "and generalize the rule."
                )

        # Fallback: identity transform to ensure a return (last resort)
        outputs = [copy.deepcopy(g) for g in test_inputs]
        return dspy.Prediction(test_outputs=outputs)

# Instantiate the improved program
program = CodeSynthesisSolver(attempts=5)
'''
