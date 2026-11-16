/**
 * Verifies close()/AbortSignal/timeoutMs actually settle an in-flight
 * optimize() call, against the mocked server from optimize_mocked.ts. Each
 * case starts a run whose evaluate() never resolves, so the promise can only
 * settle via the mechanism under test -- before this, close() mid-run left
 * the promise pending forever and there was no cancellation API at all.
 *
 * Run: npx tsx tests/cancellation.ts
 */
import { Client, type Example } from "../src";
import { withMockServer } from "./_harness";

const TRAINSET: Example[] = [{ id: "1", fields: { x: "a" } }];
const SEED = { instructions: "do the thing" };

function hangingCall(client: Client, extra: Record<string, unknown> = {}) {
  return client.optimize({
    runId: `cancel-${Date.now()}-${Math.floor(Math.random() * 1e9)}`,
    seedCandidate: SEED,
    trainset: TRAINSET,
    maxMetricCalls: 10,
    // Never resolves: the server stays blocked waiting on this response.
    evaluate: () => new Promise(() => {}),
    makeReflectiveDataset: async () => ({ instructions: [] }),
    ...extra,
  });
}

async function expectReject(promise: Promise<unknown>, pattern: RegExp, what: string) {
  await promise.then(
    () => {
      throw new Error(`expected ${what} to reject the pending call, but it resolved`);
    },
    (err) => {
      if (!(err instanceof Error) || !pattern.test(err.message)) {
        throw new Error(`unexpected rejection from ${what}: ${err}`);
      }
    },
  );
}

async function main(): Promise<void> {
  await withMockServer("cancel", async (port) => {
    // close() mid-call
    let client = new Client({ target: `localhost:${port}` });
    let p = hangingCall(client);
    await new Promise((r) => setTimeout(r, 300));
    client.close();
    await expectReject(p, /close/i, "close()");
    console.log("PASS: close() settles a pending optimize() call");

    // AbortSignal
    client = new Client({ target: `localhost:${port}` });
    const controller = new AbortController();
    p = hangingCall(client, { signal: controller.signal });
    setTimeout(() => controller.abort(), 300);
    try {
      await expectReject(p, /abort/i, "AbortSignal");
    } finally {
      client.close();
    }
    console.log("PASS: AbortSignal rejects a pending optimize() call");

    // timeoutMs
    client = new Client({ target: `localhost:${port}` });
    const start = Date.now();
    p = hangingCall(client, { timeoutMs: 300 });
    try {
      await expectReject(p, /timed out/i, "timeoutMs");
    } finally {
      client.close();
    }
    const elapsed = Date.now() - start;
    if (elapsed > 5000) throw new Error(`timeoutMs took far too long: ${elapsed}ms`);
    console.log("PASS: timeoutMs rejects a pending optimize() call promptly");
  });
}

main().catch((err) => {
  console.error("FAIL:", err);
  process.exit(1);
});
