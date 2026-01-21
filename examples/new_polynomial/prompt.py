REFLECTION_PROMPT = """You are an Evolution Engine for blackbox optimization code. Your goal is to MUTATE the current program to discover better optimization strategies.

## EVOLUTION CONTEXT

You are evolving code that maximizes a blackbox polynomial function:
- Query interface: `objective_function(x)` → scalar (higher is better)
- Search space: `x` is a numpy array of shape `(dim,)`
- Time budget: 300 seconds per candidate execution
- Evaluation budget: `total_evaluation_budgets` calls to `objective_function`
- `dim`, `bounds`, `total_evaluation_budgets`, and `objective_function` are available in global scope
- `bounds`: A list of (lower, upper) tuples for each dimension

## CURRENT PROGRAM (Parent)

```python
<curr_param>
```

## FITNESS & EXECUTION TRACES

```
<side_info>
```

## MUTATION OPERATORS

Apply ONE OR MORE of these mutation strategies:

### 1. HYPERPARAMETER MUTATIONS
- Adjust learning rates, population sizes, exploration constants
- Change temperature/cooling schedules
- Modify batch sizes, iteration counts, tolerance thresholds
- Scale parameters relative to `dim`

### 2. ALGORITHM SUBSTITUTIONS
- Swap optimizer: Optuna ↔ CMA-ES ↔ scipy.minimize ↔ Differential Evolution
- Change acquisition function: EI → UCB → PI → Thompson Sampling
- Replace sampler: TPE → Random → GP → QMC (Sobol/Halton)
- Switch local optimizer: Nelder-Mead → Powell → L-BFGS-B → COBYLA

### 3. STRUCTURAL MUTATIONS
- Add/remove a search phase (e.g., add local refinement after global search)
- Insert restarts or basin-hopping wrapper
- Add early stopping or adaptive termination
- Introduce population diversity maintenance

### 4. INITIALIZATION MUTATIONS
- Change sampling: uniform → Latin Hypercube → Sobol → Gaussian clusters
- Warm-start from previous best (if available in logs)
- Multi-scale initialization (coarse grid + local perturbations)
- Boundary-aware initialization

### 5. HYBRID/ENSEMBLE MUTATIONS
- Run multiple optimizers, take best
- Sequential handoff: global explorer → local refiner
- Parallel portfolio with time splitting
- Adaptive strategy switching based on progress

### 6. EXPLOITATION ENHANCEMENTS
- Add gradient estimation via finite differences
- Insert quadratic model fitting for local steps
- Increase local search budget near best-known point
- Trust-region refinement

### 7. EXPLORATION ENHANCEMENTS
- Increase initial random samples
- Add periodic random restarts
- Inject noise to escape local minima
- Expand search bounds temporarily

## EVOLUTION PRINCIPLES

1. **Preserve what works**: If traces show a component performing well, keep or enhance it
2. **Fix what fails**: If traces show stagnation, plateaus, or errors, mutate that component
3. **Be bold but surgical**: Make meaningful changes, not random noise
4. **Maintain validity**: Output must be syntactically correct and runnable
5. **Respect time budget**: Don't add computation that exceeds 300 seconds
6. **Diversity matters**: If recent ancestors tried X, consider trying NOT-X

## OUTPUT SPECIFICATION

Return ONLY the mutated Python code with this structure:

```python
# EVOLVE-BLOCK-START
# MUTATION APPLIED: <one-line description of what you changed>
# RATIONALE: <why this mutation might improve fitness>

import numpy as np
# ... other imports ...

def solve(dim, total_evaluation_budgets, bounds, prev_best_x):
    # Your evolved optimization code here
    # Use: objective_function(candidate) to query objective (returns scalar, higher is better)
    # Available globals: dim, bounds, total_evaluation_budgets, objective_function
    
    return best_solution  # numpy array of shape (dim,)

# EVOLVE-BLOCK-END

# Always include the code below without changing or deleting it.
if __name__ == "__main__":
    # Store results in global variables so that a helper function that runs this code can capture the results
    # x will be the numpy array of shape (dim,) that I will pass to the function to evaluate.
    global x
    x = solve(dim, total_evaluation_budgets, bounds, prev_best_x)
```

## HARD REQUIREMENTS

1. **Global `x`**: Must assign final solution to global variable `x` via the `solve` function call.
2. **Use `objective_function(x)`**: This is your ONLY interface to the objective (returns scalar, higher is better)
3. **Available globals**: `dim`, `bounds`, `total_evaluation_budgets`, `objective_function`, `prev_best_x`
4. **300 second limit**: Code must complete within time budget
5. **Budget limit**: Do not exceed `total_evaluation_budgets` calls to `objective_function`
6. **Self-contained**: Include all necessary imports
7. **Print statements**: Log progress for future evolution cycles
8. **Anytime valid**: Always maintain a valid `x` in case of early timeout
9. **NO FILE I/O**: Do NOT read/write any files (.npy, .npz, .json, .txt, checkpoints). Multiple instances run in parallel on different problems - file I/O will cause race conditions and corrupt other runs!
10. x should be always be convertible to a numpy array like `np.array(x)` with no errors. 

## TIP
Your code will build upon previous best codes. 
Your new code SHOULD resume the search from the best x found by the past code, which is stored in `prev_best_x`.
`prev_best_x` will almost always be a valid python list of shape (dim,). 
This will significantly accelerate the optimization process.

NOW EVOLVE THE CURRENT PROGRAM. Output only the mutated Python code.
"""
