# GEPA-Evolved ARC-AGI Agent Architecture

**89.5% accuracy** (vs 32.5% seed) | **$0.14/task** with Gemini 3 Flash

```mermaid
flowchart LR
    subgraph INPUT["📥 INPUT"]
        I["train_inputs\ntrain_outputs\ntest_inputs"]
    end

    subgraph STAGE1["🔵 Programmer"]
        P1["Prompt: Expert ARC solver\nWrite transform(grid)"]
        LLM1["🤖 Generate"]
        EXTRACT["Extract code"]
    end

    subgraph STAGE2["🟠 Validator"]
        EXEC["Test on training"]
        CHECK{"Pass?"}
        P2["Prompt: Fix with feedback"]
        LLM2["🤖 Fix"]
    end

    subgraph STAGE3["🟢 Execute"]
        RUN["Run on test"]
    end

    subgraph STAGE4["🟣 Fallback"]
        P3["Prompt: Direct predict"]
        LLM3["🤖 Predict"]
    end

    subgraph OUTPUT["✅ OUTPUT"]
        COMBINE["2 attempts per test"]
    end

    I --> P1 --> LLM1 --> EXTRACT --> EXEC --> CHECK
    CHECK -->|Yes| RUN --> COMBINE
    CHECK -->|No| P2 --> LLM2 -->|"max 2x"| EXEC
    I --> P3 --> LLM3 --> COMBINE

    style INPUT fill:#E3F2FD,stroke:#1976D2
    style STAGE1 fill:#E1F5FE,stroke:#0288D1
    style STAGE2 fill:#FFF3E0,stroke:#F57C00
    style STAGE3 fill:#E8F5E9,stroke:#388E3C
    style STAGE4 fill:#F3E5F5,stroke:#7B1FA2
    style OUTPUT fill:#E8F5E9,stroke:#2D936C
```

---

## Prompt Summaries

| Stage | Prompt | Key Elements |
|-------|--------|--------------|
| **Programmer** | Generate `transform(grid)` | Role: "expert ARC solver", hints (objects, symmetry, colors), numpy template |
| **Fixer** | Fix failed code | Previous code + "expected X, got Y" feedback, re-show examples |
| **Fallback** | Direct prediction | No code, JSON output format, backup for code failures |

---

## Why This Architecture Works

| Component | Seed Agent | Evolved Agent | Impact |
|-----------|------------|---------------|--------|
| **Approach** | Direct prediction | Code synthesis | +40% accuracy |
| **Validation** | None | Test on ALL training | Catches bugs |
| **Error handling** | None | Feedback loop (2x) | Fixes edge cases |
| **Attempts** | 1 per test | 2 per test | Uses full quota |
| **Fallback** | None | Direct LLM backup | Never returns empty |

## LLM Call Pattern

```
solve() called
│
├─► LLM Call 1: Programmer (generate transform code)
│
├─► [If validation fails]
│   └─► LLM Call 2: Fixer (fix code with error feedback)
│       └─► [If still fails]
│           └─► LLM Call 3: Fixer (second attempt)
│
└─► LLM Call 4: Fallback (direct prediction for 2nd attempt)

Total: 2-4 LLM calls per problem
```

## Key Insight: Test-Driven Development for LLMs

GEPA discovered that **validating code against training examples before submission** is crucial. This is essentially **test-driven development** applied to LLM code generation:

1. Generate code
2. Run tests (training examples)
3. If tests fail, show errors to LLM
4. LLM fixes code
5. Repeat until tests pass or max retries

This pattern could generalize to other code generation tasks beyond ARC-AGI.
