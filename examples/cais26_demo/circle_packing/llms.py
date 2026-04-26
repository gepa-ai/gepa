"""
LLM configurations for circle packing direct code optimization.

This module contains:
- DSPy signature for code reflection (direct optimization)
- Code reflection instructions
- Code proposer function
"""

import dspy


CIRCLE_PACKING_BACKGROUND = """
Make BREAKTHROUGH improvements by trying fundamentally different approaches.

Pack 26 non-overlapping circles inside a UNIT SQUARE [0,1] x [0,1].
Goal: MAXIMIZE the sum of all circle radii.

SCORING: Sum of all circle radii (higher is better!)

CRITICAL CODE FORMAT:
- Function name MUST be: `def main(timeout, current_best_solution):`
- `current_best_solution` is a numpy array of shape (N, 3) or None.
- Return a dictionary with:
    - 'circles': numpy array shape (N, 3) where each row is (x, y, radius)
    - 'all_scores': list of floats (even if just one score)

CRITICAL CONSTRAINTS:
1. All circles fully inside [0,1]×[0,1]: 0 ≤ x-r, x+r ≤ 1 and 0 ≤ y-r, y+r ≤ 1
2. No overlaps: distance between centers ≥ sum of radii
3. All radii positive
4. N is 26 for this problem.

INNOVATION STRATEGIES:
1. **Algorithmic diversity**: Physics-based, optimization-based, geometric, hybrid, meta-heuristics
2. **Geometric insights**: Hexagonal patterns, corner utilization, variable radii
3. **Optimization techniques**: Multiple restarts, hierarchical approaches, gradient-free methods
4. **Hyperparameter auto-tuning**: Use optuna/hyperopt to find best parameters automatically
5. Imagine you have all the packages available in the environment already and freely explore any of the packages you need.

ANALYSIS STRATEGY:
1. If scores plateau → try fundamentally different algorithm
2. If errors persist → address root cause, don't just patch
3. Synthesize insights from refinement successes into NEW innovations
4. The refiner LLM will handle the refinement process using the `refiner_prompt`, so you focus on making a big leap in the global strategy.

OUTPUT REQUIREMENTS:
- Return ONLY executable Python code (no markdown, no explanations)
- Focus on BREAKTHROUGH ideas, not incremental tweaks
- Be bold and innovative!
"""


# ============================================================================
# INITIAL REFINEMENT PROMPT
# ============================================================================


SEED_REFINEMENT_PROMPT = """You are an expert mathematician and computational geometry specialist.

YOUR TASK: Given the current code and its execution results (provided in `code_results`), generate improved code that fixes errors and increases the score.

DOMAIN KNOWLEDGE FOR CIRCLE PACKING:
1. **Optimization strategies**: Multiple restarts, better initialization, adaptive step sizes, hybrid local+global search
2. **Geometric improvements**: Better initial layouts, exploit symmetry, special corner/edge placement strategies
3. **Circle size tuning**: Variable vs uniform radii, size gradients (larger center vs edges or vice versa)
4. **Algorithm switching**: If physics-based fails, try optimization-based. If greedy fails, try force simulation
5. **Constraint handling**: Penalty methods, projection methods, constrained optimization (scipy.optimize.minimize with bounds)
6. **Hyperparameter tuning**: Use optuna/hyperopt for algorithmic parameters (force constants, step sizes, iterations)

CRITICAL CONSTRAINTS (must be preserved):
- Function MUST be `def main(timeout, current_best_solution):`
- All circles fully inside [0,1]×[0,1]: 0 ≤ x-r, x+r ≤ 1 and 0 ≤ y-r, y+r ≤ 1
- No overlaps: distance between centers ≥ sum of radii
- All radii positive, return exactly N circles with shape (N, 3)
- Return a dict with 'circles' and 'all_scores' keys.

REFINER SHOULD:
- Build on what worked in the initial code (don't throw away good ideas)
- Carefully analyze `code_results` (stdout, stderr, errors) to identify why the previous attempt failed or scored poorly.
- Use `current_best_solution` (passed as argument) to warm-start if it helps.
- Make targeted improvements rather than complete rewrites (unless necessary)
- Return the whole refined code in executable format.
"""


# ============================================================================
# DYNAMIC CODE REFINER (Used by Fitness Function)
# ============================================================================


class RefinerSignature(dspy.Signature):
    """Refine the code based on its evaluation results by fixing the errors and improving the performance."""

    refiner_prompt = dspy.InputField(desc="Instructions for how to refine the code")
    code_to_improve = dspy.InputField(desc="Code to improve")
    code_results = dspy.InputField(
        desc="Evaluation results of the code to improve by fixing the errors and improving the performance"
    )
    refined_code = dspy.OutputField(
        desc="Next iteration of improved code based on the evaluation results"
    )


REFINEMENT_PROMPT_BACKGROUND = """
**Your Role: Evolve the Local Improver LLM's prompt (refiner_prompt)**
**What Makes a Good Refinement Prompt:**
1. **Clear priorities**: "First fix execution errors, then satisfy constraints, then optimize the score"
2. **Result utilization**: "Instruct the refiner to use `code_results` (stdout, errors) to diagnose issues"
3. **Warm-starting**: "Encourage the refiner to use the previous best solution if available"
4. **Incremental approach**: "Make targeted improvements, preserve what works"
5. **Precision**: "Ensure the refiner follows the exact function signature and return format"

**Key Refinement Strategies to Guide:**
- Optimization: better initialization, parameter tuning, more iterations
- Algorithm switching: when to try completely different approach
- Constraint fixing: overlap removal, boundary violation fixes

**Analysis of Results:**
Look for:
- Consistent improvement → prompt is working, keep reinforcing successful strategies
- Plateaus → need stronger guidance to escape local optima
- Regressions → too aggressive, need more "preserve what works" guidance

**Output:**
- Return ONLY the improved refinement prompt text
- Make it actionable, specific, and clear
- Guide the refiner to be methodical and incremental
"""
