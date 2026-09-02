/**
 * Minimal end-to-end example.
 *
 * Prereqs:
 *   1. The gepa-rpc server is running on localhost:50051
 *      (from the repo root: `gepa-rpc --port 50051`).
 *   2. The reflection model the server uses is reachable
 *      (the server hardcodes "gpt-5"; configure your provider creds
 *      so litellm can call it).
 *
 * Run:
 *   npx tsx examples/basic.ts
 *
 * The "evaluate" function below is a stand-in: it just checks whether the
 * candidate's instruction text contains a keyword. In a real workload you
 * would call your own model / pipeline here.
 */

import { Client, type Example } from "../src";

const trainset: Example[] = [
  { id: "1", fields: { input: "What is 2+2?", answer: "4" } },
  { id: "2", fields: { input: "Capital of France?", answer: "Paris" } },
  { id: "3", fields: { input: "Color of the sky?", answer: "blue" } },
];

async function main(): Promise<void> {
  const client = new Client({ target: "localhost:50051" });

  try {
    const result = await client.optimize({
      runId: `demo-${Date.now()}`,
      seedCandidate: {
        instructions: "Answer the question in one word.",
      },
      trainset,
      maxMetricCalls: 20,

      evaluate: async ({ candidate, batch, captureTraces }) => {
        const instructions = candidate.instructions ?? "";
        const outputs: string[] = [];
        const scores: number[] = [];
        const trajectories = captureTraces ? [] as Array<{
          inputFields: Record<string, string>;
          output: string;
          feedback: string;
        }> : undefined;

        for (const ex of batch) {
          const expected = ex.fields.answer ?? "";
          // Stand-in "model": echo the instruction prefix + expected token if
          // the instruction mentions "answer", otherwise just the input.
          const output = instructions.toLowerCase().includes("answer")
            ? expected
            : ex.fields.input ?? "";
          const correct = output === expected;
          outputs.push(output);
          scores.push(correct ? 1.0 : 0.0);
          if (trajectories) {
            trajectories.push({
              inputFields: ex.fields,
              output,
              feedback: correct
                ? `Correct: produced "${expected}".`
                : `Wrong: expected "${expected}" but got "${output}".`,
            });
          }
        }
        return { outputs, scores, trajectories };
      },

      makeReflectiveDataset: async ({ componentsToUpdate, trajectories }) => {
        const out: Record<string, Array<{
          inputs: Record<string, string>;
          generatedOutput: string;
          feedback: string;
        }>> = {};
        for (const comp of componentsToUpdate) {
          out[comp] = trajectories.map((t) => ({
            inputs: t.inputFields,
            generatedOutput: t.output,
            feedback: t.feedback,
          }));
        }
        return out;
      },

      onProgress: (u) => {
        console.log(
          `[progress] ${u.metricCallsUsed}/${u.maxMetricCalls} ` +
            `best=${u.bestScore.toFixed(3)}`,
        );
      },
    });

    console.log("\noptimization complete");
    console.log("  runId:", result.runId);
    console.log("  bestScore:", result.bestScore);
    console.log("  bestCandidate:", result.bestCandidate);
  } finally {
    client.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
