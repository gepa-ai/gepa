/** Shared subprocess/port plumbing for the SDK tests. */
import { spawn, ChildProcess } from "node:child_process";
import { createServer, createConnection } from "node:net";

export const PYTHON = process.env.GEPA_RPC_PYTHON ?? "python3";

export function freePort(): Promise<number> {
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

export function waitForPort(port: number, timeoutMs = 10000): Promise<void> {
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

export async function killProcess(proc: ChildProcess): Promise<void> {
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

/** Runs `fn` against a real gepa-rpc server whose optimizer is the
 * deterministic fake in gepa.rpc.testing.mock_optimize_server. */
export async function withMockServer<T>(label: string, fn: (port: number) => Promise<T>): Promise<T> {
  const port = await freePort();
  const log: string[] = [];
  const proc = spawn(
    PYTHON,
    ["-m", "gepa.rpc.testing.mock_optimize_server", "--port", String(port),
     "--runs-dir", `/tmp/gepa-rpc-ts-${label}`],
    { stdio: "pipe" },
  );
  proc.stdout?.on("data", (c) => log.push(c.toString()));
  proc.stderr?.on("data", (c) => log.push(c.toString()));
  try {
    await waitForPort(port);
    return await fn(port);
  } catch (err) {
    console.error("--- gepa-rpc server output ---\n" + log.join(""));
    throw err;
  } finally {
    await killProcess(proc);
  }
}
