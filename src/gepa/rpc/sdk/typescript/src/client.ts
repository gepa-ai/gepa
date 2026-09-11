import * as path from "path";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";

import type {
  ClientOptions,
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

const PROTO_PATH = path.resolve(__dirname, "..", "proto", "gepa.proto");

const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
  keepCase: true,
  longs: Number,
  enums: String,
  defaults: true,
  oneofs: true,
});

const proto = (grpc.loadPackageDefinition(packageDefinition) as any).gepa_rpc;

export class Client {
  private grpcClient: any;

  constructor(opts: ClientOptions) {
    const credentials = opts.credentials ?? grpc.credentials.createInsecure();
    this.grpcClient = new proto.GEPAService(opts.target, credentials);
  }

  close(): void {
    grpc.closeClient(this.grpcClient);
  }

  optimize(opts: OptimizeOptions): Promise<OptimizeResult> {
    return new Promise<OptimizeResult>((resolve, reject) => {
      const call = this.grpcClient.RunOptimization() as grpc.ClientDuplexStream<
        any,
        any
      >;
      let settled = false;

      const settle = (fn: () => void) => {
        if (settled) return;
        settled = true;
        try {
          call.end();
        } catch {
          // already closed
        }
        fn();
      };

      call.on("data", (msg: any) => {
        if (msg.evaluate_batch_request) {
          this.handleEvaluate(call, msg.evaluate_batch_request, opts.evaluate)
            .catch((err) => settle(() => reject(err)));
        } else if (msg.reflective_dataset_request) {
          this.handleReflective(
            call,
            msg.reflective_dataset_request,
            opts.makeReflectiveDataset,
          ).catch((err) => settle(() => reject(err)));
        } else if (msg.progress_update) {
          if (opts.onProgress) {
            try {
              opts.onProgress(toProgressUpdate(msg.progress_update));
            } catch (err) {
              settle(() => reject(err));
            }
          }
        } else if (msg.optimization_complete) {
          const c = msg.optimization_complete;
          settle(() =>
            resolve({
              runId: c.run_id,
              bestCandidate: mapToObject(c.best_candidate),
              bestScore: c.best_score,
            }),
          );
        } else if (msg.optimization_error) {
          const e = msg.optimization_error;
          settle(() =>
            reject(
              new Error(
                `optimization ${e.run_id || "(unknown)"} failed: ${e.message}`,
              ),
            ),
          );
        }
      });

      call.on("error", (err: Error) => {
        settle(() => reject(err));
      });

      call.on("end", () => {
        settle(() =>
          reject(
            new Error(
              "server closed the stream before sending optimization_complete",
            ),
          ),
        );
      });

      call.write({
        start_request: {
          run_id: opts.runId,
          seed_candidate: opts.seedCandidate,
          trainset: opts.trainset.map((e) => ({ id: e.id, fields: e.fields })),
          valset:
            opts.valset?.map((e) => ({ id: e.id, fields: e.fields })) ?? [],
          reflection_lm: opts.reflectionLm ?? "",
          max_metric_calls: opts.maxMetricCalls,
        },
      });
    });
  }

  optimizeOmni(opts: OptimizeOmniOptions): Promise<OptimizeOmniResult> {
    return new Promise<OptimizeOmniResult>((resolve, reject) => {
      const call = this.grpcClient.RunOptimizationOmni() as grpc.ClientDuplexStream<
        any,
        any
      >;
      let settled = false;

      const settle = (fn: () => void) => {
        if (settled) return;
        settled = true;
        try {
          call.end();
        } catch {
          // already closed
        }
        fn();
      };

      call.on("data", (msg: any) => {
        if (msg.evaluate_batch_request) {
          this.handleOmniEvaluate(call, msg.evaluate_batch_request, opts.evaluate)
            .catch((err) => settle(() => reject(err)));
        } else if (msg.progress_update) {
          if (opts.onProgress) {
            try {
              opts.onProgress(toOmniProgressUpdate(msg.progress_update));
            } catch (err) {
              settle(() => reject(err));
            }
          }
        } else if (msg.optimization_complete) {
          const c = msg.optimization_complete;
          settle(() =>
            resolve({
              runId: c.run_id,
              bestCandidate: c.best_candidate,
              bestScore: c.best_score,
              totalEvals: c.total_evals,
            }),
          );
        } else if (msg.optimization_error) {
          const e = msg.optimization_error;
          settle(() =>
            reject(
              new Error(
                `optimization ${e.run_id || "(unknown)"} failed: ${e.message}`,
              ),
            ),
          );
        }
      });

      call.on("error", (err: Error) => {
        settle(() => reject(err));
      });

      call.on("end", () => {
        settle(() =>
          reject(
            new Error(
              "server closed the stream before sending optimization_complete",
            ),
          ),
        );
      });

      call.write({
        start_request: {
          run_id: opts.runId,
          seed_candidate: opts.seedCandidate ?? "",
          dataset: (opts.dataset ?? []).map((e) => ({ id: e.id, fields: e.fields })),
          valset: (opts.valset ?? []).map((e) => ({ id: e.id, fields: e.fields })),
          objective: opts.objective ?? "",
          reflection_lm: opts.reflectionLm ?? "",
          engine: opts.engine ?? "",
          max_evals: opts.maxEvals ?? 0,
        },
      });
    });
  }

  private async handleOmniEvaluate(
    call: grpc.ClientDuplexStream<any, any>,
    req: any,
    handler: (args: OmniEvaluateBatchArgs) => Promise<OmniEvaluateBatchResult>,
  ): Promise<void> {
    const args: OmniEvaluateBatchArgs = {
      requestId: req.request_id,
      candidate: req.candidate,
      batch: (req.batch ?? []).map(
        (e: any): Example => ({
          id: e.id,
          fields: mapToObject(e.fields),
        }),
      ),
      optStates: (req.opt_states ?? []).map(
        (s: any): OmniOptimizationState => ({
          bestExampleEvals: (s.best_example_evals ?? []).map(
            (e: any): OmniBestEval => ({
              score: e.score,
              sideInfo: parseSideInfo(e.side_info),
            }),
          ),
        }),
      ),
    };
    const result = await handler(args);
    call.write({
      evaluate_batch_response: {
        request_id: args.requestId,
        scores: result.scores,
        side_infos: (result.sideInfos ?? []).map((s) => s !== undefined ? JSON.stringify(s) : "null"),
      },
    });
  }

  private async handleEvaluate(
    call: grpc.ClientDuplexStream<any, any>,
    req: any,
    handler: (args: EvaluateBatchArgs) => Promise<EvaluateBatchResult>,
  ): Promise<void> {
    const args: EvaluateBatchArgs = {
      requestId: req.request_id,
      candidate: mapToObject(req.candidate),
      batch: (req.batch ?? []).map(
        (e: any): Example => ({
          id: e.id,
          fields: mapToObject(e.fields),
        }),
      ),
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
    call: grpc.ClientDuplexStream<any, any>,
    req: any,
    handler: (
      args: ReflectiveDatasetArgs,
    ) => Promise<ReflectiveDatasetResult>,
  ): Promise<void> {
    const args: ReflectiveDatasetArgs = {
      requestId: req.request_id,
      candidate: mapToObject(req.candidate),
      componentsToUpdate: req.components_to_update ?? [],
      trajectories: (req.trajectories ?? []).map(
        (t: any): Trajectory => ({
          inputFields: mapToObject(t.input_fields),
          output: t.output,
          feedback: t.feedback,
        }),
      ),
    };
    const result = await handler(args);
    const reflective_data: Record<string, { entries: any[] }> = {};
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
      reflective_dataset_response: {
        request_id: args.requestId,
        reflective_data,
      },
    });
  }
}

function parseSideInfo(raw: string): unknown {
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return { raw };
  }
}

function mapToObject(m: any): Record<string, string> {
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

function toProgressUpdate(p: any): ProgressUpdate {
  return {
    metricCallsUsed: p.metric_calls_used,
    maxMetricCalls: p.max_metric_calls,
    bestScore: p.best_score,
    bestCandidate: mapToObject(p.best_candidate),
  };
}

function toOmniProgressUpdate(p: any): OmniProgressUpdate {
  return {
    evalsUsed: p.evals_used,
    maxEvals: p.max_evals,
    bestScore: p.best_score,
    bestCandidate: p.best_candidate,
  };
}
