# Polynomial Minimization on Evalset

This example uses GEPA to optimize code that minimizes blackbox functions from [evalset](https://github.com/sigopt/evalset/tree/main).

## Files

- **main.py** - Entry point that sets up GEPA optimization. Creates a dataset from benchmark problems, configures the optimization engine, and runs the optimization loop.

- **config.py** - Handles command-line argument parsing (LLM model, optimization level, log directory, etc.) and generates log directory paths.

- **evaluator.py** - Creates the fitness function that executes candidate code on optimization problems. Handles code execution with timeouts, extracts results, and computes scores based on how close the solution is to the true minimum.

- **evalset.py** - Collection of benchmark optimization test functions (Ackley, Rosenbrock, Rastrigin, etc.) from the "Stratified Analysis of Bayesian Optimization Methods" paper. Each function has properties like dimension, bounds, true minimum location, and classifiers (unimodal, multimodal, noisy, etc.).

- **llm.py** - Defines the reflection prompt template that guides the LLM to improve optimization code, and provides a function to create the reflection language model instance.

## Usage

```bash
python examples/polynomial/main.py
```