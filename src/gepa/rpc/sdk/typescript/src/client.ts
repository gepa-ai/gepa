import * as path from "path";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";

import type {
  ClientOptions,
  ErrorKind,
  EvaluateBatchArgs,
  EvaluateBatchResult,
  Example,
  OmniBestEval,
  OmniEvaluateBatchArgs,
  OmniEvaluateBatchResult,
  OmniOptimizationState,
  OmniProgressUpdate,
  OptimizeOmniOptions,
  OptimizeOmniResult,
  OptimizeOptions,
  OptimizeResult,
  ProgressUpdate,
  ReflectiveDatasetArgs,
  ReflectiveDatasetResult,
  Trajectory,
} from "./types";
import { OptimizationError } from "./types";
import type {
  TypedDuplexStream,
  WireEvaluateBatchRequest,
  WireMap,
  WireOmniEvaluateBatchRequest,
  WireOmniProgressUpdate,
  WireOmniServerMessage,
  WireOptimizationError,
  WireProgressUpdate,
  WireReflectiveDatasetRequest,
  WireServerMessage,
} from "./wire";

const PROTO_PATH = path.resolve(__dirname, "..", "proto", "gepa.proto");

/** The dynamically-loaded gRPC service. @grpc/proto-loader has no static
 * types to offer here (it parses gepa.proto at runtime), so this interface is
 * the one deliberate boundary we can't derive -- everything past it is typed
 * against wire.ts's hand-written shapes. */
interface GEPAServiceClient {
  RunOptimization(): TypedDuplexStream<unknown, WireServerMessage>;
  RunOptimizationOmni(): TypedDuplexStream<unknown, WireOmniServerMessage>;
  Cancel(
    request: { run_id: string },
    callback: (
      err: grpc.ServiceError | null,
      resp?: { accepted: boolean; message: string },
    ) => void,
  ): void;
}

type ServiceCtor = new (target: string, credentials: grpc.ChannelCredentials) => GEPAServiceClient;

// Loaded lazily on first Client construction rather than at module import.
// protoLoader.loadSync does blocking file I/O, which previously ran the moment
// anything imported this module -- including bundlers doing static analysis
// that never construct a Client at all.
let cachedServiceCtor: ServiceCtor | null = null;

function getServiceCtor(): ServiceCtor {
  if (cachedServiceCtor) return cachedServiceCtor;
  const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
    keepCase: true,
    longs: Number,
    enums: String,
    defaults: true,
    oneofs: true,
  });
  const proto = grpc.loadPackageDefinition(packageDefinition) as unknown as {
    gepa_rpc: { GEPAService: ServiceCtor };
  };
  cachedServiceCtor = proto.gepa_rpc.GEPAService;
  return cachedServiceCtor;
}

function errorKindFromWire(kind: WireOptimizationError["error_kind"] | undefined): ErrorKind {
  switch (kind) {
    case "EVALUATOR_ERROR":
      return "evaluator_error";
    case "INTERNAL_ERROR":
      return "internal_error";
    case "CANCELLED":
      return "cancelled";
    default:
      return "unspecified";
  }
}

function mapToObject(m: WireMap | undefined): Record<string, string> {
  if (!m) return {};
  if (m instanceof Map) {
    const o: Record<string, string> = {};
    for (const [k, v] of m) o[k] = String(v);
    return o;
  }
  const o: Record<string, string> = {};
  for (const k of Object.keys(m)) o[k] = String(m[k]);
  return o;
}

function parseSideInfo(raw: string): unknown {
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return { raw };
  }
}

function toProgressUpdate(p: WireProgressUpdate): ProgressUpdate {
  return {
    metricCallsUsed: p.metric_calls_used,
    maxMetricCalls: p.max_metric_calls,
    bestScore: p.best_score,
    bestCandidate: mapToObject(p.best_candidate),
  };
}

function toOmniProgressUpdate(p: WireOmniProgressUpdate): OmniProgressUpdate {
  return {
    evalsUsed: p.evals_used,
    maxEvals: p.max_evals,
    bestScore: p.best_score,
    bestCandidate: p.best_candidate,
  };
}

/** Wires `signal`/`timeoutMs` to cancel the call and reject, at most once.
 * Returns a cleanup to detach the listener and clear the timer. */
function wireAbort(
  label: string,
  cancelCall: () => void,
  onAbort: (reason: Error) => void,
  signal: AbortSignal | undefined,
  timeoutMs: number | undefined,
): () => void {
  let fired = false;
  const fire = (reason: Error) => {
    if (fired) return;
    fired = true;
    cancelCall();
    onAbort(reason);
  };

  const onSignalAbort = () => fire(new Error(`${label} aborted via AbortSignal`));
  signal?.addEventListener("abort", onSignalAbort);

  let timer: ReturnType<typeof setTimeout> | undefined;
  if (timeoutMs !== undefined) {
    timer = setTimeout(() => fire(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs);
  }

  return () => {
    signal?.removeEventListener("abort", onSignalAbort);
    if (timer) clearTimeout(timer);
  };
}

export class Client {
  private grpcClient: GEPAServiceClient;
  private pendingCalls = new Set<{ cancel: () => void; reject: (err: Error) => void }>();

  constructor(opts: ClientOptions) {
    const credentials = opts.credentials ?? grpc.credentials.createInsecure();
    const Ctor = getServiceCtor();
    this.grpcClient = new Ctor(opts.target, credentials);
  }

  /** Closes the channel and rejects every call still in flight. Previously a
   * pending optimize()/optimizeOmni() promise was left unresolved forever if
   * close() was called mid-run. */
  close(): void {
    const closedErr = new Error("Client.close() called while a call was still in flight");
    for (const pending of this.pendingCalls) {
      pending.cancel();
      pending.reject(closedErr);
    }
    this.pendingCalls.clear();
    grpc.closeClient(this.grpcClient as unknown as grpc.Client);
  }

  /** Asks the server to cancel a run by id. Resolves with `accepted: false`
   * (not a rejection) when the run is unknown or already finished, since
   * cancelling a finished run is a no-op rather than a failure. */
  cancel(runId: string): Promise<{ accepted: boolean; message: string }> {
    return new Promise((resolve, reject) => {
      this.grpcClient.Cancel({ run_id: runId }, (err, resp) => {
        if (err) reject(err);
        else resolve(resp!);
      });
    });
  }

  optimize(opts: OptimizeOptions): Promise<OptimizeResult> {
    return new Promise<OptimizeResult>((resolve, reject) => {
      const call = this.grpcClient.RunOptimization();
      let settled = false;

      const settle = (fn: () => void) => {
        if (settled) return;
        settled = true;
        cleanupAbort();
        this.pendingCalls.delete(pendingEntry);
        try {
          call.end();
        } catch {
          // already closed
        }
        fn();
      };
      const doResolve = (r: OptimizeResult) => settle(() => resolve(r));
      const doReject = (e: Error) => settle(() => reject(e));

      const pendingEntry = { cancel: () => call.cancel(), reject: doReject };
      this.pendingCalls.add(pendingEntry);
      const cleanupAbort = wireAbort(
        "optimize()", () => call.cancel(), doReject, opts.signal, opts.timeoutMs,
      );

      call.on("data", (msg) => {
        if (msg.evaluate_batch_request) {
          this.handleEvaluate(call, msg.evaluate_batch_request, opts.evaluate).catch(doReject);
        } else if (msg.reflective_dataset_request) {
          this.handleReflective(
            call, msg.reflective_dataset_request, opts.makeReflectiveDataset,
          ).catch(doReject);
        } else if (msg.progress_update) {
          if (opts.onProgress) {
            try {
              opts.onProgress(toProgressUpdate(msg.progress_update));
            } catch (err) {
              doReject(err as Error);
            }
          }
        } else if (msg.optimization_complete) {
          const c = msg.optimization_complete;
          doResolve({
            runId: c.run_id,
            bestCandidate: mapToObject(c.best_candidate),
            bestScore: c.best_score,
          });
        } else if (msg.optimization_error) {
          const e = msg.optimization_error;
          doReject(new OptimizationError(e.run_id, e.message, errorKindFromWire(e.error_kind)));
        }
      });

      call.on("error", (err) => doReject(err));
      call.on("end", () =>
        doReject(new Error("server closed the stream before sending optimization_complete")),
      );

      call.write({
        start_request: {
          run_id: opts.runId,
          seed_candidate: opts.seedCandidate,
          trainset: opts.trainset.map((e) => ({ id: e.id, fields: e.fields })),
          valset: opts.valset?.map((e) => ({ id: e.id, fields: e.fields })) ?? [],
          reflection_lm: opts.reflectionLm ?? "",
          max_metric_calls: opts.maxMetricCalls,
        },
      });
    });
  }

  optimizeOmni(opts: OptimizeOmniOptions): Promise<OptimizeOmniResult> {
    return new Promise<OptimizeOmniResult>((resolve, reject) => {
      const call = this.grpcClient.RunOptimizationOmni();
      let settled = false;

      const settle = (fn: () => void) => {
        if (settled) return;
        settled = true;
        cleanupAbort();
        this.pendingCalls.delete(pendingEntry);
        try {
          call.end();
        } catch {
          // already closed
        }
        fn();
      };
      const doResolve = (r: OptimizeOmniResult) => settle(() => resolve(r));
      const doReject = (e: Error) => settle(() => reject(e));

      const pendingEntry = { cancel: () => call.cancel(), reject: doReject };
      this.pendingCalls.add(pendingEntry);
      const cleanupAbort = wireAbort(
        "optimizeOmni()", () => call.cancel(), doReject, opts.signal, opts.timeoutMs,
      );

      call.on("data", (msg) => {
        if (msg.evaluate_batch_request) {
          this.handleOmniEvaluate(call, msg.evaluate_batch_request, opts.evaluate).catch(doReject);
        } else if (msg.progress_update) {
          if (opts.onProgress) {
            try {
              opts.onProgress(toOmniProgressUpdate(msg.progress_update));
            } catch (err) {
              doReject(err as Error);
            }
          }
        } else if (msg.optimization_complete) {
          const c = msg.optimization_complete;
          doResolve({
            runId: c.run_id,
            bestCandidate: c.best_candidate,
            bestScore: c.best_score,
            totalEvals: c.total_evals,
          });
        } else if (msg.optimization_error) {
          const e = msg.optimization_error;
          doReject(new OptimizationError(e.run_id, e.message, errorKindFromWire(e.error_kind)));
        }
      });

      call.on("error", (err) => doReject(err));
      call.on("end", () =>
        doReject(new Error("server closed the stream before sending optimization_complete")),
      );

      call.write({
        start_request: {
          run_id: opts.runId,
          seed_candidate: opts.seedCandidate ?? "",
          dataset: (opts.dataset ?? []).map((e) => ({ id: e.id, fields: e.fields })),
          valset: (opts.valset ?? []).map((e) => ({ id: e.id, fields: e.fields })),
          objective: opts.objective ?? "",
          background: opts.background ?? "",
          reflection_lm: opts.reflectionLm ?? "",
          engine: opts.engine ?? "",
          max_evals: opts.maxEvals ?? 0,
        },
      });
    });
  }

  private async handleOmniEvaluate(
    call: TypedDuplexStream<unknown, WireOmniServerMessage>,
    req: WireOmniEvaluateBatchRequest,
    handler: (args: OmniEvaluateBatchArgs) => Promise<OmniEvaluateBatchResult>,
  ): Promise<void> {
    const args: OmniEvaluateBatchArgs = {
      requestId: req.request_id,
      candidate: req.candidate,
      batch: (req.batch ?? []).map((e): Example => ({ id: e.id, fields: mapToObject(e.fields) })),
      optStates: (req.opt_states ?? []).map(
        (s): OmniOptimizationState => ({
          bestExampleEvals: (s.best_example_evals ?? []).map(
            (e): OmniBestEval => ({ score: e.score, sideInfo: parseSideInfo(e.side_info) }),
          ),
        }),
      ),
    };
    const result = await handler(args);
    call.write({
      evaluate_batch_response: {
        request_id: args.requestId,
        scores: result.scores,
        side_infos: (result.sideInfos ?? []).map((s) => (s !== undefined ? JSON.stringify(s) : "null")),
      },
    });
  }

  private async handleEvaluate(
    call: TypedDuplexStream<unknown, WireServerMessage>,
    req: WireEvaluateBatchRequest,
    handler: (args: EvaluateBatchArgs) => Promise<EvaluateBatchResult>,
  ): Promise<void> {
    const args: EvaluateBatchArgs = {
      requestId: req.request_id,
      candidate: mapToObject(req.candidate),
      batch: (req.batch ?? []).map((e): Example => ({ id: e.id, fields: mapToObject(e.fields) })),
      captureTraces: !!req.capture_traces,
    };
    const result = await handler(args);
    call.write({
      evaluate_batch_response: {
        request_id: args.requestId,
        outputs: result.outputs,
        scores: result.scores,
        trajectories: (result.trajectories ?? []).map((t) => ({
          input_fields: t.inputFields,
          output: t.output,
          feedback: t.feedback,
        })),
      },
    });
  }

  private async handleReflective(
    call: TypedDuplexStream<unknown, WireServerMessage>,
    req: WireReflectiveDatasetRequest,
    handler: (args: ReflectiveDatasetArgs) => Promise<ReflectiveDatasetResult>,
  ): Promise<void> {
    const args: ReflectiveDatasetArgs = {
      requestId: req.request_id,
      candidate: mapToObject(req.candidate),
      componentsToUpdate: req.components_to_update ?? [],
      trajectories: (req.trajectories ?? []).map(
        (t): Trajectory => ({
          inputFields: mapToObject(t.input_fields),
          output: t.output,
          feedback: t.feedback,
        }),
      ),
    };
    const result = await handler(args);
    const reflective_data: Record<string, { entries: unknown[] }> = {};
    for (const [comp, entries] of Object.entries(result)) {
      reflective_data[comp] = {
        entries: entries.map((e) => ({
          inputs: e.inputs,
          generated_output: e.generatedOutput,
          feedback: e.feedback,
        })),
      };
    }
    call.write({
      reflective_dataset_response: { request_id: args.requestId, reflective_data },
    });
  }
}
