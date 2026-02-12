# Creating Adapters

!!! tip "Most users don't need a custom adapter"
    The [`optimize_anything`](quickstart.md) API handles most use cases — just write an evaluator function.
    Custom adapters are for advanced scenarios where you need full control over batch evaluation,
    trace capture, or reflective dataset formatting.

GEPA can optimize any system consisting of text components by implementing the `GEPAAdapter` protocol. This guide explains how to create custom adapters.

## The GEPAAdapter Protocol

Every adapter must implement two methods:

```python
from gepa.core.adapter import GEPAAdapter, EvaluationBatch

class MyAdapter(GEPAAdapter[DataInst, Trajectory, RolloutOutput]):
    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        """Execute the system and return scores."""
        ...
    
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        """Build dataset for reflection."""
        ...
```

## Step-by-Step Guide

### Step 1: Define Your Types

First, define the types your adapter will use:

```python
from dataclasses import dataclass
from typing import Any

# Your input data type
@dataclass
class TaskInput:
    question: str
    context: str
    expected_answer: str

# Trajectory captures execution details
@dataclass  
class ExecutionTrace:
    prompt_used: str
    model_response: str
    intermediate_steps: list[str]

# Output from your system
@dataclass
class TaskOutput:
    answer: str
    confidence: float
```

### Step 2: Implement `evaluate`

The `evaluate` method runs your system on a batch of inputs:

```python
from gepa.core.adapter import EvaluationBatch

class MyAdapter:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def evaluate(
        self,
        batch: list[TaskInput],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[ExecutionTrace, TaskOutput]:
        outputs = []
        scores = []
        trajectories = [] if capture_traces else None
        
        for task in batch:
            # Build prompt using candidate components
            prompt = candidate["system_prompt"] + "\n" + task.question
            
            # Run your system
            response = self._call_model(prompt)
            
            # Parse output
            output = TaskOutput(answer=response, confidence=0.9)
            outputs.append(output)
            
            # Compute score (higher is better)
            score = 1.0 if output.answer == task.expected_answer else 0.0
            scores.append(score)
            
            # Capture trace if requested
            if capture_traces:
                trace = ExecutionTrace(
                    prompt_used=prompt,
                    model_response=response,
                    intermediate_steps=[],
                )
                trajectories.append(trace)
        
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )
```

### Step 3: Implement `make_reflective_dataset`

This method creates data for the reflection LLM to propose improvements:

```python
def make_reflective_dataset(
    self,
    candidate: dict[str, str],
    eval_batch: EvaluationBatch[ExecutionTrace, TaskOutput],
    components_to_update: list[str],
) -> dict[str, list[dict]]:
    """Build a reflective dataset for each component."""
    
    dataset = {}
    
    for component_name in components_to_update:
        component_data = []
        
        for i, trace in enumerate(eval_batch.trajectories):
            record = {
                "Inputs": {
                    "prompt": trace.prompt_used,
                },
                "Generated Outputs": {
                    "response": trace.model_response,
                },
                "Feedback": self._generate_feedback(
                    trace, 
                    eval_batch.outputs[i],
                    eval_batch.scores[i],
                ),
            }
            component_data.append(record)
        
        dataset[component_name] = component_data
    
    return dataset

def _generate_feedback(self, trace, output, score):
    """Generate helpful feedback for the reflection LLM."""
    if score == 1.0:
        return "Correct! The answer matched the expected output."
    else:
        return f"Incorrect. The model answered '{output.answer}' but this was wrong."
```

## Best Practices

### 1. Rich Feedback

The more informative your feedback, the better GEPA can optimize:

```python
def _generate_feedback(self, trace, output, expected, score):
    feedback_parts = []
    
    # Include the score
    feedback_parts.append(f"Score: {score}")
    
    # Explain what went wrong
    if score < 1.0:
        feedback_parts.append(f"Expected: {expected}")
        feedback_parts.append(f"Got: {output.answer}")
        
        # Add specific error analysis
        if len(output.answer) > 100:
            feedback_parts.append("Issue: Response too verbose")
        if expected.lower() not in output.answer.lower():
            feedback_parts.append("Issue: Key information missing")
    
    return "\n".join(feedback_parts)
```

### 2. Error Handling

Handle failures gracefully:

```python
def evaluate(self, batch, candidate, capture_traces=False):
    outputs, scores, trajectories = [], [], []
    
    for task in batch:
        try:
            output = self._run_task(task, candidate)
            score = self._compute_score(output, task)
        except Exception as e:
            # Return a failed result rather than raising
            output = TaskOutput(answer="ERROR", confidence=0.0)
            score = 0.0
            if capture_traces:
                trajectories.append(ExecutionTrace(
                    error=str(e),
                    # ... capture what you can
                ))
        
        outputs.append(output)
        scores.append(score)
    
    return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)
```

### 3. Multi-Objective Optimization

Support multiple objectives:

```python
def evaluate(self, batch, candidate, capture_traces=False):
    # ... evaluation logic ...
    
    objective_scores = []
    for output in outputs:
        objective_scores.append({
            "accuracy": 1.0 if output.correct else 0.0,
            "latency": 1.0 / (1.0 + output.latency),  # Lower is better, inverted
            "cost": 1.0 / (1.0 + output.token_count),
        })
    
    return EvaluationBatch(
        outputs=outputs,
        scores=scores,
        trajectories=trajectories,
        objective_scores=objective_scores,  # Multi-objective support
    )
```

## Example: Complete Adapter

Here's a complete example adapter:

```python
from dataclasses import dataclass
from typing import Any
import litellm
from gepa.core.adapter import GEPAAdapter, EvaluationBatch

@dataclass
class QAInput:
    question: str
    answer: str

@dataclass
class QATrace:
    prompt: str
    response: str

@dataclass
class QAOutput:
    answer: str

class SimpleQAAdapter(GEPAAdapter[QAInput, QATrace, QAOutput]):
    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.model = model
    
    def evaluate(
        self,
        batch: list[QAInput],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[QATrace, QAOutput]:
        outputs, scores = [], []
        trajectories = [] if capture_traces else None
        
        for item in batch:
            # Build prompt
            prompt = f"{candidate['system_prompt']}\n\nQuestion: {item.question}"
            
            # Call model
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content
            
            # Score
            output = QAOutput(answer=answer)
            score = 1.0 if item.answer.lower() in answer.lower() else 0.0
            
            outputs.append(output)
            scores.append(score)
            
            if capture_traces:
                trajectories.append(QATrace(prompt=prompt, response=answer))
        
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )
    
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[QATrace, QAOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        dataset = {"system_prompt": []}
        
        for i, trace in enumerate(eval_batch.trajectories or []):
            dataset["system_prompt"].append({
                "Inputs": {"question": trace.prompt.split("Question: ")[-1]},
                "Generated Outputs": {"answer": trace.response},
                "Feedback": f"Score: {eval_batch.scores[i]}"
            })
        
        return dataset

# Usage
adapter = SimpleQAAdapter(model="openai/gpt-4o-mini")
result = gepa.optimize(
    seed_candidate={"system_prompt": "Answer questions accurately."},
    trainset=trainset,
    adapter=adapter,
    reflection_lm="openai/gpt-4o",
    max_metric_calls=50,
)
```

## Built-in Adapters

GEPA provides several ready-to-use adapters for common use cases:

| Adapter | Description | Use Case |
|---------|-------------|----------|
| [DefaultAdapter](../api/adapters/DefaultAdapter.md) | Simple adapter for prompt optimization with any LLM | General prompt tuning, Q&A systems |
| [DSPy Adapter](../api/adapters/DSPyAdapter.md) | Optimizes DSPy program instructions and prompts | DSPy module optimization |
| [DSPy Full Program Adapter](../api/adapters/DSPyFullProgramAdapter.md) | Evolves entire DSPy programs including structure | Full program evolution, architecture search |
| [RAG Adapter](../api/adapters/RAGAdapter.md) | Optimizes RAG pipeline components | Retrieval-augmented generation systems |
| [MCP Adapter](../api/adapters/MCPAdapter.md) | Optimizes MCP tool descriptions and system prompts | Tool-using agents, MCP servers |
| [TerminalBench Adapter](../api/adapters/TerminalBenchAdapter.md) | Optimizes agents for terminal-based tasks | CLI agents, shell automation |

### When to Use Each Adapter

- **DefaultAdapter**: Start here for simple prompt optimization tasks. Works with any LLM via litellm.

- **DSPy Adapter**: Use when you have a DSPy program and want to optimize the instructions for individual predictors while keeping the program structure fixed.

- **DSPy Full Program Adapter**: Use when you want GEPA to evolve the entire DSPy program, including its structure and module composition.

- **RAG Adapter**: Use for optimizing retrieval-augmented generation systems. Supports multiple vector stores (ChromaDB, Weaviate, Qdrant, Milvus, etc.) and optimizes query reformulation, context synthesis, and answer generation prompts.

- **MCP Adapter**: Use for optimizing Model Context Protocol tool usage. Supports both local (stdio) and remote (SSE/StreamableHTTP) MCP servers.

- **TerminalBench Adapter**: Use for optimizing agents that interact with terminal/shell environments.

## Next Steps

- See the [API Reference](../api/core/GEPAAdapter.md) for complete `GEPAAdapter` protocol documentation
- Explore the built-in adapters above for your specific use case
- Read the [DefaultAdapter](../api/adapters/DefaultAdapter.md) source for a reference implementation
