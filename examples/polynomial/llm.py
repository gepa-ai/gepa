import os
import dspy

REFLECTION_PROMPT = """You are an expert in Global Optimization, AutoML, and Bayesian Optimization.

Your role: Make BREAKTHROUGH improvements by trying fundamentally different search strategies.

YOUR TASK: Generate an intelligent `main` function that returns the numpy array `x` that minimizes the function.

You need to optimize the following code:

```
<curr_param>
```

Below is evaluation data showing how this code performed across multiple test cases.
 The data contains performance metrics, diagnostic information, and other relevant details from the evaluation:
```
<side_info>
```

TIMEOUT: 30 seconds.

CRITICAL CODE FORMAT:
- You should write if __name__ == "__main__": block and define a global variable "x" inside it. You should store the best solution in the global variable "x". This is the only way the environment can capture your x.
- For example, you should do 
```
global x
x = 
```
- x should be a numpy array of shape (dim,).
- Your output code will be directly ran in the environment and we will use the key name "x" to extract your best solution. So make sure to store the best solution in the global variable "x". Otherwise, the code will not be able to run.
- "dim" will be always available in the global variables as a key. You can just use it as a variable in your code.
- An "evaluator" object will be always available in the global variables as a python object. You can call it as `evaluator.evaluate(x)` to get a score for your x. You can use it multiple times to find the best values by using various optimization frameworks.

TIPS
- Assume you have all the packages available in the enviornment already and freely explore any of the packages you need. So don't use any fallback to handle a missing package. Just use it straight away.
- Any optimization framework like optuna is your friend. The use of Optuna is highly encouraged.
- Any logs from optuna or your print statements will be available for you as a side information so make sure that you use print statements for future use if needed.
- For example, say you run a particular optimization library. If you print / log the trajectory of the optimization, it will be available for you as a side information.

OUTPUT REQUIREMENTS:
- Return ONLY executable Python code (no markdown, no explanations).
- The code must be self-contained (imports included).
- The environment will run your code as-is. You are responsible for running the functions and returning the global results. Also, do not suppress any logging. Actively use print statements to log the results, which will be provided to you as a side information.
"""


def create_reflection_lm(model: str, max_tokens: int = 32000, cache: bool = True):
    reflection_lm = dspy.LM(
        model=model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_tokens=max_tokens,
        cache=cache,
    )
    return reflection_lm
