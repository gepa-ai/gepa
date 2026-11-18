/**
 * Exercises the real wire protocol for the non-Omni optimize() path --
 * request/response shapes, trajectories, and the reflective-dataset round
 * trip -- against a real gepa-rpc server subprocess.
 *
 * Unlike e2e.ts this needs no recorded LLM cache: the server runs with
 * gepa.optimize patched to a deterministic fake, so no real or replayed LLM
 * calls happen. e2e.ts covers the Omni path for real against a recorded
 * cache; this covers the non-Omni path's wire protocol, which neither SDK's
 * tests covered before.
 *
 * Run: npx tsx tests/optimize_mocked.ts
 */
import { Client, type Example } from "../src";
import { withMockServer } from "./_harness";

const TRAINSET: Example[] = [
  { id: "1", fields: { x: "a" } },
  { id: "2", fields: { x: "b" } },
];
const SEED = { instructions: "do the thing" };

async function main(): Promise<void> {
  await withMockServer("mocked", async (port) => {
    const client = new Client({ target: `localhost:${port}` });
    let sawCaptureTraces = false;
    let sawReflective = false;
    try {
      const result = await client.optimize({
        runId: `mocked-${Date.now()}`,
        seedCandidate: SEED,
        trainset: TRAINSET,
        maxMetricCalls: 10,
        evaluate: async ({ batch, captureTraces }) => {
          if (captureTraces) sawCaptureTraces = true;
          return {
            outputs: batch.map(() => "out"),
            scores: batch.map(() => 1.0),
            trajectories: captureTraces
              ? batch.map((ex) => ({ inputFields: ex.fields, output: "out", feedback: "good" }))
              : undefined,
          };
        },
        makeReflectiveDataset: async ({ trajectories }) => {
          sawReflective = true;
          if (trajectories.length !== TRAINSET.length) {
            throw new Error(`expected ${TRAINSET.length} trajectories, got ${trajectories.length}`);
          }
          return { instructions: [] };
        },
      });

      if (result.bestCandidate.instructions !== SEED.instructions) {
        throw new Error(`unexpected bestCandidate: ${JSON.stringify(result.bestCandidate)}`);
      }
      if (result.bestScore !== 1) throw new Error(`unexpected bestScore: ${result.bestScore}`);
    } finally {
      client.close();
    }

    if (!sawCaptureTraces) throw new Error("server never requested trajectories");
    if (!sawReflective) throw new Error("make_reflective_dataset was never called");
    console.log("PASS: optimize() non-Omni wire protocol (TypeScript, mocked optimizer)");
  });
}

main().catch((err) => {
  console.error("FAIL:", err);
  process.exit(1);
});
