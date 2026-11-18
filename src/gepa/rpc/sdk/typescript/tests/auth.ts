/**
 * Verifies the SDK can talk to a server started with --auth-token, and is
 * correctly rejected without it. Before this the SDK had no way to send
 * `authorization` metadata at all, so a hardened server was unreachable from
 * the official client -- the server-side feature was unusable in practice.
 *
 * Run: npx tsx tests/auth.ts
 */
import { spawn, ChildProcess } from "node:child_process";
import { Client, type Example } from "../src";
import { freePort, waitForPort, killProcess, PYTHON } from "./_harness";

const TRAINSET: Example[] = [{ id: "1", fields: { x: "a" } }];
const SEED = { instructions: "do the thing" };
const TOKEN = "s3cret-token";

function startAuthServer(port: number): ChildProcess {
  return spawn(
    PYTHON,
    ["-m", "gepa.rpc.testing.mock_optimize_server", "--port", String(port),
     "--runs-dir", "/tmp/gepa-rpc-ts-auth", "--auth-token", TOKEN],
    { stdio: "pipe" },
  );
}

function runOnce(client: Client, runId: string) {
  return client.optimize({
    runId,
    seedCandidate: SEED,
    trainset: TRAINSET,
    maxMetricCalls: 10,
    evaluate: async ({ batch, captureTraces }) => ({
      outputs: batch.map(() => "out"),
      scores: batch.map(() => 1.0),
      trajectories: captureTraces
        ? batch.map((ex) => ({ inputFields: ex.fields, output: "out", feedback: "ok" }))
        : undefined,
    }),
    makeReflectiveDataset: async () => ({ instructions: [] }),
  });
}

async function main(): Promise<void> {
  const port = await freePort();
  const proc = startAuthServer(port);
  const log: string[] = [];
  proc.stdout?.on("data", (c) => log.push(c.toString()));
  proc.stderr?.on("data", (c) => log.push(c.toString()));

  try {
    await waitForPort(port);

    // Without a token: must be rejected.
    const anon = new Client({ target: `localhost:${port}` });
    try {
      await runOnce(anon, `auth-missing-${Date.now()}`);
      throw new Error("expected the server to reject an unauthenticated call");
    } catch (err) {
      const m = (err as Error).message ?? "";
      if (!/UNAUTHENTICATED|auth token/i.test(m)) {
        throw new Error(`expected UNAUTHENTICATED, got: ${m}`);
      }
    } finally {
      anon.close();
    }
    console.log("PASS: unauthenticated call is rejected");

    // With the wrong token: must also be rejected.
    const wrong = new Client({ target: `localhost:${port}`, authToken: "nope" });
    try {
      await runOnce(wrong, `auth-wrong-${Date.now()}`);
      throw new Error("expected the server to reject a wrong token");
    } catch (err) {
      const m = (err as Error).message ?? "";
      if (!/UNAUTHENTICATED|auth token/i.test(m)) {
        throw new Error(`expected UNAUTHENTICATED, got: ${m}`);
      }
    } finally {
      wrong.close();
    }
    console.log("PASS: wrong token is rejected");

    // With the right token: the whole run must succeed.
    const authed = new Client({ target: `localhost:${port}`, authToken: TOKEN });
    try {
      const result = await runOnce(authed, `auth-ok-${Date.now()}`);
      if (result.bestCandidate.instructions !== SEED.instructions) {
        throw new Error(`unexpected result: ${JSON.stringify(result.bestCandidate)}`);
      }
      // Cancel is unary -- prove metadata rides on that path too.
      const c = await authed.cancel("no-such-run");
      if (c.accepted !== false) throw new Error("expected accepted=false for an unknown run");
    } finally {
      authed.close();
    }
    console.log("PASS: correct token authenticates streaming and unary calls");
  } catch (err) {
    console.error("--- server output ---\n" + log.join(""));
    throw err;
  } finally {
    await killProcess(proc);
  }
}

main().catch((err) => {
  console.error("FAIL:", err);
  process.exit(1);
});
