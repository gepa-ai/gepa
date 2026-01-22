"""Prompts for gepa_blog blackbox optimization using objective/background pattern."""

OBJECTIVE = "Evolve Python code that minimizes a blackbox objective function using the available evaluation budget efficiently."

BACKGROUND = """
You are optimizing code that solves blackbox minimization problems (lower is better).

## Function Signature
```python
def solve(objective_function, config, prev_best_x=None):
    # config contains: bounds (array of [min, max] per dim), dim (int), budget (int)
    # prev_best_x: warm-start from previous best solution (numpy array or None)
    # Returns: best x found (numpy array of shape (dim,))
```

## Code Requirements
- Return the best solution found as a numpy array of shape (dim,)
- Use `objective_function(x)` to evaluate candidates (counts against budget, lower score is better)
- Stay within `config['budget']` calls
- Use `prev_best_x` to warm-start search if available

## Mutation Strategies
1. Hyperparameter tuning (learning rates, population sizes, iterations)
2. Algorithm changes (evolutionary, gradient-free, bayesian optimization)
3. Initialization strategies (random, latin hypercube, from prev_best_x)
4. Hybrid approaches (local + global search)
5. Exploitation vs exploration balance

## Output Format
Provide the improved code in a single code block with triple backticks.
"""
