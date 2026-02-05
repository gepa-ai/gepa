"""Test LLM tracking on a single ARC problem."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from examples.arc_agi.main import SEED_AGENT_CODE, load_arc_dataset
from examples.arc_agi.evaluate import run_agent


# More complex candidate that makes multiple LLM calls
MULTI_CALL_AGENT = '''
import json
import re

def solve(train_in, train_out, test_in, llm):
    """Solve ARC with multiple LLM calls: analyze, hypothesize, predict."""
    def fmt(g):
        return "\\n".join(" ".join(str(c) for c in row) for row in g)

    def parse_grid(text):
        try:
            return json.loads(text)
        except:
            lines = [l.strip() for l in text.strip().split("\\n") if l.strip()]
            return [[int(x) for x in re.findall(r"\\d+", line)] for line in lines]

    examples = "\\n".join(f"In:\\n{fmt(i)}\\nOut:\\n{fmt(o)}" for i, o in zip(train_in, train_out))

    # Call 1: Analyze the pattern
    analysis = llm(f"""Analyze this ARC puzzle. What pattern transforms input to output?

{examples}

Describe the transformation rule in 2-3 sentences.""")

    # Call 2: Predict outputs
    all_inputs = train_in + test_in
    predictions_prompt = f"""Based on this rule: {analysis}

Apply it to predict outputs for these inputs:
""" + "\\n".join(f"Input {i}:\\n{fmt(inp)}" for i, inp in enumerate(all_inputs))
    predictions_prompt += "\\n\\nReturn each output as a JSON list of lists."

    resp = llm(predictions_prompt)

    # Parse predictions
    grids = re.findall(r"\\[\\[.*?\\]\\]", resp.replace("\\n", ""))
    preds = [parse_grid(g) for g in grids]

    n_train = len(train_in)
    return {
        "train": preds[:n_train],
        "test": [[p] for p in preds[n_train:]],
    }
'''


def test_llm_tracking():
    """Test that LLM calls are tracked correctly."""
    # Load one problem
    train_set, _, _ = load_arc_dataset(seed=0)
    ex = train_set[0]

    print(f"Problem: {ex.problem_id}")
    print(f"  Train examples: {len(ex.train_in)}")
    print(f"  Test examples: {len(ex.test_in)}")

    # Run agent
    result = run_agent(
        agent_code=SEED_AGENT_CODE,
        train_in=ex.train_in,
        train_out=ex.train_out,
        test_in=ex.test_in,
        test_out=ex.test_out,
        model_id="openrouter/google/gemini-3-flash-preview",
        max_llm_calls=5,
    )

    print(f"\n=== Results ===")
    print(f"Training score: {result['training_score']:.0%}")
    print(f"Test score: {result['test_score']:.0%}")
    print(f"Error: {result['error']}")

    # Check LLM tracking
    llm = result["llm"]
    side_info = llm.get_side_info()

    print(f"\n=== LLM Tracking ===")
    print(f"LLM calls: {side_info['llm_calls']}/{side_info['llm_budget']}")
    print(f"Total tokens: {side_info['total_tokens']}")
    print(f"Total cost: ${side_info['total_cost']:.6f}")

    # Verify trajectory
    trajectory = side_info["trajectory"]
    print(f"Trajectory entries: {len(trajectory)}")

    for i, call in enumerate(trajectory):
        print(f"\n{'='*60}")
        print(f"CALL {i+1}")
        print(f"{'='*60}")
        print(f"\n--- PROMPT ({len(call['prompt'])} chars) ---")
        print(call['prompt'])
        print(f"\n--- REASONING ({len(call.get('reasoning') or '')} chars) ---")
        print(call.get('reasoning') or 'None')
        print(f"\n--- RESPONSE ({len(call['response'])} chars) ---")
        print(call['response'])
        print(f"\n--- TOKENS: {call['tokens']}, COST: ${call['cost']:.6f} ---")

    # Assertions
    assert side_info["llm_calls"] > 0, "Should have made at least one LLM call"
    assert side_info["llm_calls"] == len(trajectory), "Trajectory should match call count"
    assert all("prompt" in c and "response" in c for c in trajectory), "Each call should have prompt and response"

    print("\n✓ LLM tracking test passed!")


def test_multi_call_agent():
    """Test tracking with a more complex agent that makes multiple LLM calls."""
    train_set, _, _ = load_arc_dataset(seed=0)
    ex = train_set[0]

    print(f"Problem: {ex.problem_id}")
    print(f"  Train examples: {len(ex.train_in)}")
    print(f"  Test examples: {len(ex.test_in)}")

    # Run multi-call agent with gemini-3-flash-preview
    result = run_agent(
        agent_code=MULTI_CALL_AGENT,
        train_in=ex.train_in,
        train_out=ex.train_out,
        test_in=ex.test_in,
        test_out=ex.test_out,
        model_id="openrouter/google/gemini-3-flash-preview",
        max_llm_calls=5,
    )

    print(f"\n=== Results ===")
    print(f"Training score: {result['training_score']:.0%}")
    print(f"Test score: {result['test_score']:.0%}")
    print(f"Error: {result['error']}")

    llm = result["llm"]
    side_info = llm.get_side_info()

    print(f"\n=== LLM Tracking ===")
    print(f"LLM calls: {side_info['llm_calls']}/{side_info['llm_budget']}")
    print(f"Total tokens: {side_info['total_tokens']}")
    print(f"Total cost: ${side_info['total_cost']:.6f}")

    trajectory = side_info["trajectory"]
    print(f"Trajectory entries: {len(trajectory)}")

    for i, call in enumerate(trajectory):
        print(f"\n{'='*60}")
        print(f"CALL {i+1}")
        print(f"{'='*60}")
        print(f"\n--- PROMPT ({len(call['prompt'])} chars) ---")
        print(call['prompt'])
        print(f"\n--- REASONING ({len(call.get('reasoning') or '')} chars) ---")
        print(call.get('reasoning') or 'None')
        print(f"\n--- RESPONSE ({len(call['response'])} chars) ---")
        print(call['response'])
        print(f"\n--- TOKENS: {call['tokens']}, COST: ${call['cost']:.6f} ---")

    assert side_info["llm_calls"] >= 2, "Should have made at least 2 LLM calls"
    print("\n✓ Multi-call agent tracking test passed!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        test_multi_call_agent()
    else:
        test_llm_tracking()
