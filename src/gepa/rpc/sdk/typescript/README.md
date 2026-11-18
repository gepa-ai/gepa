# @gepa/sdk

TypeScript client for the [GEPA] cross-language
optimizer. Lets you run GEPA optimizations from Node.js, providing your
evaluator as a regular async function instead of a Python adapter.

## Install

```bash
npm install @gepa/sdk
```

## Usage

```ts
import { Client } from "@gepa/sdk";

const client = new Client({ target: "localhost:50051" });

const result = await client.optimize({
  runId: "demo-1",
  seedCandidate: { instructions: "You are a helpful assistant." },
  trainset: [
    { id: "1", fields: { input: "What is 2+2?", answer: "4" } },
    { id: "2", fields: { input: "Capital of France?", answer: "Paris" } },
  ],
  maxMetricCalls: 30,

  evaluate: async ({ candidate, batch, captureTraces }) => {
    // Your native evaluator runs here.
    return {
      outputs: batch.map(() => "..."),
      scores: batch.map(() => 0.5),
      trajectories: captureTraces ? batch.map(() => ({ /* ... */ })) : undefined,
    };
  },

  makeReflectiveDataset: async ({ componentsToUpdate, trajectories }) => {
    const out: Record<string, ReflectiveEntry[]> = {};
    for (const comp of componentsToUpdate) {
      out[comp] = trajectories.map((t) => ({
        inputs: t.inputFields,
        generatedOutput: t.output,
        feedback: t.feedback,
      }));
    }
    return out;
  },

  onProgress: (u) =>
    console.log(`${u.metricCallsUsed}/${u.maxMetricCalls} best=${u.bestScore}`),
});

console.log("best:", result.bestCandidate, "score:", result.bestScore);
client.close();
```

## How it works

The SDK opens a long-lived bidirectional gRPC stream against a `gepa-rpc`
server. The server runs `gepa.optimize()` in-process and uses your `evaluate`
and `makeReflectiveDataset` callbacks as the adapter — every score comes from
your code, not from Python.

## Protocol source of truth

The `.proto` file lives at the repo root (`proto/gepa.proto`). `npm run build`
runs `scripts/sync-proto.sh` first to copy the canonical version into this
package.
