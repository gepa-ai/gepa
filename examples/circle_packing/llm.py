REFLECTION_PROMPT_TEMPLATE = """
You are an expert mathematician and computational geometry specialist optimizing circle packing algorithms.
Your task is to propose a new, improved circle packing algorithm that will lead to better circle packing results.
The below is the current circle packing algorithm:
```
<curr_param>
```

Below is evaluation data showing how this algorithm performed across multiple test cases.
The data contains performance metrics, diagnostic information, and other relevant details from the evaluation:
```
<side_info>
```

Carefully analyze all the evaluation data provided above. Look for patterns that indicate what works and what doesn't. 
Pay special attention to:
- Performance metrics and how they correlate with parameter behavior
- Recurring issues, errors, or failure patterns across multiple test cases
- Successful patterns or behaviors that should be preserved or enhanced
- Any domain-specific requirements, constraints, or factual information revealed in the evaluation data
- Specific technical details that are crucial for understanding the parameter's role

Based on your analysis, propose a new circle packing algorithm that addresses the identified issues while maintaining or improving upon what works well. Your proposal should be directly informed by the patterns and insights from the evaluation data.

Do not change the code inside __name__ == "__main__" block.
You have all the packages available in the enviornment already.
You're even highly encouraged to develop a sophisticated algorithm using optimization frameworks.
You can evaluate your circles using the `evaluate_circles(circles, num_circles)` function.
`num_circles` will be always available in the environment. So no need for fallback.
"circles" should be a numpy array of shape (num_circles, 3) where each row is (x, y, radius).
Your code may call `evaluate_circles` function multiple times but not more than `total_evaluation_budgets` times.
`total_evaluation_budgets` will be always available in the environment. So no need for fallback.

⚠️ CRITICAL BUDGET ENFORCEMENT:
- You MUST strictly respect the `total_evaluation_budgets` limit. If your code calls `evaluate_circles` more than `total_evaluation_budgets` times, the code will be INVALIDATED and receive a score of 0.0 (extremely bad for maximization).
- You MUST track the number of calls to `evaluate_circles` and ensure it never exceeds `total_evaluation_budgets`.
- Exceeding the budget will result in immediate code invalidation regardless of solution quality.

Available packages:
- numpy
- scipy
- pandas
- scikit-learn
- shapely (computational geometry)
- networkx (graph algorithms)
- sympy (symbolic mathematics)
- cvxpy (convex optimization)
- optuna (hyperparameter optimization)
- pyomo (optimization modeling)
- pulp (linear programming)
- numba (JIT compilation for performance)
- jax (automatic differentiation and optimization)

Always include the immutable code block without changing or deleting it.

Provide the new circle packing algorithm within ``` blocks.
Your output should be a standalone, executable Python code.
"""
