/**
 * Real end-to-end test: runs an actual GEPA optimization through the real
 * TypeScript client and a real gepa-rpc server subprocess, with no mocking
 * of gepa.optimize/optimize_anything anywhere in the path.
 *
 * Ports tests/test_rpc_e2e/test_rpc_e2e.py's design exactly (same dataset,
 * seed candidate, and grading) and reuses its recorded cache and golden
 * file, since the wire protocol and task logic are identical regardless of
 * client language. Set RECORD_TESTS=true to regenerate them against a real
 * OPENAI_API_KEY (only needs to be done once, from any one language).
 *
 * Run: npx tsx tests/e2e.ts
 */

import { spawn, ChildProcess } from "node:child_process";
import { createServer, createConnection } from "node:net";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { Client, type Example } from "../src";

const RECORD = (process.env.RECORD_TESTS ?? "false").toLowerCase() === "true";
const PYTHON = process.env.GEPA_RPC_PYTHON ?? "python3";

const E2E_DIR = join(__dirname, "..", "..", "..", "..", "..", "..", "tests", "test_rpc_e2e");
const CACHE_FILE = join(E2E_DIR, "llm_cache.json");
const GOLDEN_FILE = join(E2E_DIR, "optimized_candidate.txt");

const TASK_MODEL = "openai/gpt-4.1-nano";
const REFLECTION_MODEL = "openai/gpt-4.1-nano";

const DATASET: Example[] = [
  { id: "1", fields: { text: "I love this product!", label: "positive" } },
  { id: "2", fields: { text: "This is terrible.", label: "negative" } },
  { id: "3", fields: { text: "Oh great, another Monday.", label: "negative" } },
  { id: "4", fields: { text: "Well, that could have gone better.", label: "negative" } },
  { id: "5", fields: { text: "Not bad at all, actually.", label: "positive" } },
  { id: "6", fields: { text: "I guess it's fine, whatever.", label: "negative,neutral" } },
];

const SEED_CANDIDATE = "Classify the sentiment of the following text.";
const OBJECTIVE = "Maximize accuracy of sentiment classification.";

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.listen(0, () => {
      const address = srv.address();
      if (address === null || typeof address === "string") {
        reject(new Error("could not determine free port"));
        return;
      }
      const port = address.port;
      srv.close(() => resolve(port));
    });
  });
}

function waitForPort(port: number, timeoutMs = 10000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const socket = createConnection({ host: "127.0.0.1", port }, () => {
        socket.end();
        resolve();
      });
      socket.on("error", () => {
        socket.destroy();
        if (Date.now() > deadline) {
          reject(new Error(`nothing listening on 127.0.0.1:${port} after ${timeoutMs}ms`));
        } else {
          setTimeout(attempt, 100);
        }
      });
    };
    attempt();
  });
}

function grade(answer: string, acceptedLabels: string): number {
  const trimmed = answer.trim();
  const firstWord = trimmed.length > 0 ? trimmed.split(/\s+/)[0].replace(/[.,!"']/g, "").toLowerCase() : "";
  const accepted = acceptedLabels.toLowerCase().split(",");
  return accepted.includes(firstWord) ? 1.0 : 0.0;
}

async function callTaskLm(stubPort: number, candidate: string, text: string): Promise<string> {
  const prompt = `${candidate}\n\nText: ${text}`;
  const resp = await fetch(`http://127.0.0.1:${stubPort}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: TASK_MODEL, messages: [{ role: "user", content: prompt }] }),
  });
  const payload = await resp.json();
  return payload.choices[0].message.content;
}

async function killProcess(proc: ChildProcess): Promise<void> {
  if (proc.exitCode !== null || proc.killed) return;
  proc.kill("SIGTERM");
  await new Promise<void>((resolve) => {
    const timer = setTimeout(() => {
      proc.kill("SIGKILL");
      resolve();
    }, 5000);
    proc.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}

function bufferOutput(proc: ChildProcess, buf: string[]): void {
  proc.stdout?.on("data", (chunk) => buf.push(chunk.toString()));
  proc.stderr?.on("data", (chunk) => buf.push(chunk.toString()));
}

async function main(): Promise<void> {
  const stubPort = await freePort();
  const serverPort = await freePort();
  const stubLog: string[] = [];
  const serverLog: string[] = [];

  const stubArgs = ["-m", "gepa.rpc.testing.fake_llm_server", "--port", String(stubPort), "--cache-file", CACHE_FILE];
  if (RECORD) stubArgs.push("--record");
  const stubProc = spawn(PYTHON, stubArgs, { stdio: "pipe" });
  bufferOutput(stubProc, stubLog);

  const runsDir = mkdtempSync(join(tmpdir(), "gepa-rpc-ts-e2e-"));
  const serverProc = spawn(PYTHON, ["-m", "gepa.rpc.cli", "--port", String(serverPort), "--runs-dir", runsDir], {
    stdio: "pipe",
    env: {
      ...process.env,
      OPENAI_API_BASE: `http://127.0.0.1:${stubPort}`,
      OPENAI_API_KEY: process.env.OPENAI_API_KEY ?? "sk-fake-e2e-test-key",
    },
  });
  bufferOutput(serverProc, serverLog);

  try {
    await waitForPort(stubPort);
    await waitForPort(serverPort);

    const client = new Client({ target: `localhost:${serverPort}` });
    let bestCandidate: string;
    try {
      const result = await client.optimizeOmni({
        runId: `rpc-e2e-sentiment-ts-${Date.now()}`,
        seedCandidate: SEED_CANDIDATE,
        dataset: DATASET,
        objective: OBJECTIVE,
        reflectionLm: REFLECTION_MODEL,
        maxEvals: 30,

        evaluate: async ({ candidate, batch }) => {
          const scores: number[] = [];
          for (const ex of batch) {
            const answer = await callTaskLm(stubPort, candidate, ex.fields.text ?? "");
            scores.push(grade(answer, ex.fields.label ?? ""));
          }
          return { scores, sideInfos: batch.map(() => ({})) };
        },
      });
      bestCandidate = result.bestCandidate;
    } finally {
      client.close();
    }

    if (RECORD) {
      writeFileSync(GOLDEN_FILE, bestCandidate);
      if (typeof bestCandidate !== "string" || bestCandidate.length === 0) {
        throw new Error("RECORD_TESTS run produced an empty best candidate");
      }
      console.log("RECORD_TESTS: wrote golden file:", GOLDEN_FILE);
    } else {
      const expected = readFileSync(GOLDEN_FILE, "utf-8");
      if (bestCandidate !== expected) {
        throw new Error(
          `best candidate does not match golden file.\n  expected: ${JSON.stringify(expected)}\n  actual:   ${JSON.stringify(bestCandidate)}`
        );
      }
    }

    console.log("PASS: test_real_optimization_through_omni (TypeScript)");
  } catch (err) {
    console.error("--- fake_llm_server output ---\n" + stubLog.join(""));
    console.error("--- gepa-rpc server output ---\n" + serverLog.join(""));
    throw err;
  } finally {
    await Promise.all([killProcess(serverProc), killProcess(stubProc)]);
  }
}

main().catch((err) => {
  console.error("FAIL:", err);
  process.exit(1);
});