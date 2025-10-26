#!/usr/bin/env python3
"""
Example 3: Evolving API Schemas for Better Developer Experience

This example demonstrates using GEPA to evolve API request/response schemas.
Starting from a basic schema, GEPA discovers improvements that make the API
more intuitive, type-safe, and easier to use.

This showcases GEPA's ability to evolve structured text formats beyond just
prompts and code - including JSON schemas, API specs, database schemas, etc.
"""

import json
from typing import Any

# Simulated API usage scenarios (what developers try to do with the API)
API_SCENARIOS = [
    {
        "task": "Create a new user account",
        "developer_intent": {
            "action": "create_user",
            "data": {
                "name": "John Doe",
                "email": "john@example.com",
                "age": 30,
                "preferences": {"theme": "dark", "notifications": True},
            },
        },
        "expected_outcome": "User created successfully with ID",
        "pain_points": [
            "unclear if email is required or optional",
            "no validation hints for email format",
            "preferences structure is undocumented",
        ],
    },
    {
        "task": "Update user preferences",
        "developer_intent": {
            "action": "update_user",
            "user_id": "12345",
            "updates": {"preferences": {"theme": "light"}},
        },
        "expected_outcome": "Preferences updated, other fields unchanged",
        "pain_points": [
            "unclear if partial updates are supported",
            "no clear distinction between required and optional fields",
            "unclear if this replaces or merges preferences",
        ],
    },
    {
        "task": "Query users with filters",
        "developer_intent": {
            "action": "list_users",
            "filters": {"age_min": 18, "age_max": 65, "email_domain": "example.com"},
            "pagination": {"limit": 10, "offset": 0},
        },
        "expected_outcome": "List of users matching filters with pagination",
        "pain_points": [
            "unclear what filtering options are available",
            "no documentation of pagination parameters",
            "unclear if filters can be combined",
        ],
    },
    {
        "task": "Delete a user",
        "developer_intent": {"action": "delete_user", "user_id": "12345", "soft_delete": True},
        "expected_outcome": "User marked as deleted (soft delete)",
        "pain_points": [
            "unclear if soft delete is supported",
            "no indication of required vs optional fields",
            "unclear what happens to user data after deletion",
        ],
    },
    {
        "task": "Batch create users",
        "developer_intent": {
            "action": "batch_create_users",
            "users": [{"name": "Alice", "email": "alice@example.com"}, {"name": "Bob", "email": "bob@example.com"}],
        },
        "expected_outcome": "Multiple users created, returns IDs for each",
        "pain_points": [
            "unclear if batch operations are supported",
            "no indication of how errors are handled per-item",
            "unclear if operation is atomic or partial",
        ],
    },
]


def evaluate_schema(candidate: dict[str, str], batch: list[dict[str, Any]]) -> list[dict]:
    """
    Evaluate an API schema on developer usage scenarios.

    Scoring criteria:
    - Clarity: Does the schema clearly indicate required vs optional fields?
    - Completeness: Does it document all necessary fields and types?
    - Usability: Does it guide developers toward correct usage?
    - Flexibility: Does it support common use cases?
    """
    results = []
    schema_doc = candidate["api_schema"]

    for scenario in batch:
        task = scenario["task"]
        intent = scenario["developer_intent"]
        pain_points = scenario["pain_points"]

        # Evaluate schema quality by checking if it addresses pain points
        score = 0.0
        feedback_parts = []

        # Parse schema (assume JSON format)
        try:
            # Check if schema is valid JSON first
            if schema_doc.strip().startswith("{"):
                schema = json.loads(schema_doc)
            else:
                # Might be markdown or other format, extract JSON
                import re

                json_match = re.search(r"```json\s*(\{.*?\})\s*```", schema_doc, re.DOTALL)
                if json_match:
                    schema = json.loads(json_match.group(1))
                else:
                    raise ValueError("Could not find valid JSON in schema")

            # Evaluate schema against scenario
            action = intent["action"]

            # Check if action is documented
            if action in schema.get("endpoints", {}):
                score += 0.3
                endpoint = schema["endpoints"][action]

                # Check if required fields are clearly marked
                if "required_fields" in endpoint or "required" in endpoint:
                    score += 0.2
                    feedback_parts.append("✓ Required fields are documented")
                else:
                    feedback_parts.append("✗ Required fields are not clearly marked")

                # Check if field types are documented
                if "fields" in endpoint or "properties" in endpoint:
                    score += 0.2
                    feedback_parts.append("✓ Field types are documented")
                else:
                    feedback_parts.append("✗ Field types are not documented")

                # Check if examples are provided
                if "example" in endpoint or "examples" in endpoint:
                    score += 0.2
                    feedback_parts.append("✓ Examples provided")
                else:
                    feedback_parts.append("✗ No examples provided")

                # Check if validation rules are documented
                if any(k in endpoint for k in ["validation", "constraints", "format"]):
                    score += 0.1
                    feedback_parts.append("✓ Validation rules documented")
                else:
                    feedback_parts.append("✗ No validation rules")
            else:
                feedback_parts.append(f"✗ Endpoint '{action}' is not documented in schema")

            # Bonus: Check if pain points are addressed
            schema_text = json.dumps(schema, indent=2).lower()
            addressed_points = sum(1 for pp in pain_points if any(keyword in schema_text for keyword in pp.split()[:3]))
            if addressed_points > 0:
                bonus = 0.2 * (addressed_points / len(pain_points))
                score += bonus
                feedback_parts.append(f"✓ Addresses {addressed_points}/{len(pain_points)} pain points")

            feedback = f"Task: {task}\n" + "\n".join(feedback_parts)

        except Exception as e:
            score = 0.0
            feedback = f"Task: {task}\n✗ Schema parsing failed: {e!s}\n"
            feedback += "Schema must be valid JSON or contain JSON in markdown code block."

        results.append(
            {
                "score": min(1.0, score),  # Cap at 1.0
                "context_and_feedback": {
                    "inputs": f"Developer Task: {task}",
                    "outputs": f"Schema evaluation score: {score:.2f}",
                    "feedback": feedback,
                    "pain_points": ", ".join(pain_points),
                },
            }
        )

    return results


# Initial basic schema (intentionally incomplete)
SEED_SCHEMA = """
{
  "api_version": "1.0",
  "endpoints": {
    "create_user": {
      "method": "POST",
      "fields": ["name", "email"]
    },
    "update_user": {
      "method": "PUT",
      "fields": ["user_id", "updates"]
    },
    "delete_user": {
      "method": "DELETE",
      "fields": ["user_id"]
    }
  }
}
"""


def main():
    """Run API schema evolution using GEPA."""

    from gepa import evolve

    print("=" * 80)
    print("GEPA Evolve-Anything Example: API Schema Evolution")
    print("=" * 80)
    print(f"\nEvaluating schema on {len(API_SCENARIOS)} developer usage scenarios")
    print("\nInitial Schema:")
    print("-" * 80)
    print(SEED_SCHEMA)
    print("-" * 80)
    print("\nStarting evolution...\n")

    # Run GEPA evolution
    result = evolve(
        seed_candidate={"api_schema": SEED_SCHEMA},
        trainset=API_SCENARIOS,
        evaluate=evaluate_schema,
        reflection_prompt="""You are evolving an API schema to improve developer experience.

A good API schema should:
1. **Clarity**: Clearly mark required vs optional fields
2. **Completeness**: Document all fields with types and descriptions
3. **Validation**: Specify constraints (format, ranges, etc.)
4. **Examples**: Provide example requests for each endpoint
5. **Error Handling**: Document error cases and responses
6. **Flexibility**: Support common variations (partial updates, filtering, etc.)

Based on the feedback, identify what's missing:
- Are required fields marked?
- Are field types and formats documented?
- Are examples provided?
- Are validation rules clear?
- Are pain points addressed?

Common improvements:
- Add "required_fields" lists
- Add "fields" with type information and descriptions
- Add "example" requests showing correct usage
- Add "validation" rules (email format, age ranges, etc.)
- Document optional features (soft_delete, pagination, etc.)
- Add response schemas

Provide only the improved JSON schema (may wrap in markdown).""",
        num_iterations=25,
        minibatch_size=3,
        teacher_lm="openai/gpt-4o",
        random_seed=42,
        output_dir="./api_schema_evolution_output",
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("Evolution Complete!")
    print("=" * 80)
    print(f"\nBest Score: {result['best_score']:.3f}")
    print("\nEvolved API Schema:")
    print("-" * 80)
    print(result["best_candidate"]["api_schema"])
    print("-" * 80)

    # Test on all scenarios
    print("\n" + "=" * 80)
    print("Testing Evolved Schema on All Scenarios:")
    print("=" * 80)

    eval_results = evaluate_schema(result["best_candidate"], API_SCENARIOS)

    avg_score = sum(r["score"] for r in eval_results) / len(eval_results)
    print(f"\nOverall Schema Quality: {avg_score:.3f} / 1.0")

    for scenario, eval_result in zip(API_SCENARIOS, eval_results, strict=False):
        status = "✓" if eval_result["score"] > 0.7 else "◐" if eval_result["score"] > 0.4 else "✗"
        print(f"\n{status} {scenario['task']}: {eval_result['score']:.2f}")
        print(f"   {eval_result['context_and_feedback']['feedback'][:100]}...")


if __name__ == "__main__":
    main()
