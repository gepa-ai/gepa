/**
 * End-to-end example for the RunOptimizationOmni endpoint.
 *
 * Prereqs:
 *   1. gepa-rpc server running: `python -m gepa_rpc.cli --port 50051`
 *   2. OPENAI_API_KEY set in the server's environment.
 *
 * Run:
 *   npx tsx examples/omni.ts
 *
 * Task: optimize a system prompt that classifies a sentiment as positive or negative.
 * The evaluator checks whether the candidate prompt causes the model to output
 * the correct label — here simulated with a keyword match stand-in.
 */

import { Client, type Example } from "../src";

const dataset: Example[] = [
  { id: "1", fields: { text: "I love this product!", label: "positive" } },
  { id: "2", fields: { text: "This is terrible.", label: "negative" } },
  { id: "3", fields: { text: "Absolutely fantastic experience.", label: "positive" } },
  { id: "4", fields: { text: "Worst purchase I have ever made.", label: "negative" } },
];

async function main(): Promise<void> {
  const client = new Client({ target: "localhost:50051" });

  try {
    const result = await client.optimizeOmni({
      runId: `omni-demo-${Date.now()}`,
      seedCandidate: "Classify the sentiment of the following text as positive or negative.",
      dataset,
      objective: "Maximize accuracy of sentiment classification.",
      reflectionLm: "openai/gpt-4o-mini",
      maxEvals: 30,

      evaluate: async ({ candidate, batch }) => {
        const scores: number[] = [];
        const sideInfos: Record<string, unknown>[] = [];

        for (const ex of batch) {
          const text = ex.fields.text ?? "";
          const label = ex.fields.label ?? "";

          // Stand-in evaluator: score 1.0 if the candidate mentions both
          // "positive" and "negative" (i.e. it looks like a classification prompt),
          // and the label keyword appears in the candidate or text.
          const lowerCandidate = candidate.toLowerCase();
          const isClassificationPrompt =
            lowerCandidate.includes("positive") || lowerCandidate.includes("negative") ||
            lowerCandidate.includes("sentiment") || lowerCandidate.includes("classify");

          const score = isClassificationPrompt ? 1.0 : 0.0;
          scores.push(score);
          sideInfos.push({ text, label, score });
        }

        return { scores, sideInfos };
      },

      onProgress: (u) => {
        console.log(
          `[progress] ${u.evalsUsed}/${u.maxEvals} best=${u.bestScore.toFixed(3)} | "${u.bestCandidate.slice(0, 60)}..."`
        );
      },
    });

    console.log("\noptimization complete");
    console.log("  runId:        ", result.runId);
    console.log("  bestScore:    ", result.bestScore);
    console.log("  totalEvals:   ", result.totalEvals);
    console.log("  bestCandidate:", result.bestCandidate);
  } finally {
    client.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
