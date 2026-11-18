/**
 * Shapes of the raw messages @grpc/proto-loader produces on the wire
 * (snake_case, since client.ts loads with keepCase: true).
 *
 * Hand-written rather than generated: @grpc/proto-loader parses gepa.proto at
 * runtime, so there's no .d.ts to generate from without introducing a second
 * codegen tool (ts-proto) -- a bigger change than this pass covers. Keeping
 * these in one file next to the proto means a future field change is a
 * one-line diff here rather than a silent `any` passthrough.
 */

export interface WireExample {
  id: string;
  fields: Record<string, string>;
}

export interface WireTrajectory {
  input_fields: Record<string, string>;
  output: string;
  feedback: string;
}

/** proto3 map fields arrive as a Map or a plain object depending on options. */
export type WireMap = Map<string, string> | Record<string, string>;

export interface WireEvaluateBatchRequest {
  request_id: string;
  candidate: WireMap;
  batch: WireExample[];
  capture_traces: boolean;
}

export interface WireReflectiveDatasetRequest {
  request_id: string;
  candidate: WireMap;
  components_to_update: string[];
  trajectories: WireTrajectory[];
}

export interface WireProgressUpdate {
  metric_calls_used: number;
  max_metric_calls: number;
  best_score: number;
  best_candidate: WireMap;
}

export interface WireOptimizationComplete {
  run_id: string;
  best_candidate: WireMap;
  best_score: number;
}

export interface WireOptimizationError {
  run_id: string;
  message: string;
  error_kind: "UNSPECIFIED" | "EVALUATOR_ERROR" | "INTERNAL_ERROR" | "CANCELLED";
}

export interface WireServerMessage {
  evaluate_batch_request?: WireEvaluateBatchRequest;
  reflective_dataset_request?: WireReflectiveDatasetRequest;
  progress_update?: WireProgressUpdate;
  optimization_complete?: WireOptimizationComplete;
  optimization_error?: WireOptimizationError;
}

export interface WireOmniOptimizationState {
  best_example_evals: { score: number; side_info: string }[];
}

export interface WireOmniEvaluateBatchRequest {
  request_id: string;
  candidate: string;
  batch: WireExample[];
  opt_states: WireOmniOptimizationState[];
}

export interface WireOmniProgressUpdate {
  evals_used: number;
  max_evals: number;
  best_score: number;
  best_candidate: string;
}

export interface WireOmniOptimizationComplete {
  run_id: string;
  best_candidate: string;
  best_score: number;
  total_evals: number;
}

export interface WireOmniServerMessage {
  evaluate_batch_request?: WireOmniEvaluateBatchRequest;
  progress_update?: WireOmniProgressUpdate;
  optimization_complete?: WireOmniOptimizationComplete;
  optimization_error?: WireOptimizationError;
}

/** The subset of grpc.ClientDuplexStream this client relies on, typed against
 * our wire shapes instead of grpc-js's generic `any`. */
export interface TypedDuplexStream<Req, Res> {
  write(message: Req): boolean;
  end(): void;
  cancel(): void;
  on(event: "data", listener: (msg: Res) => void): this;
  on(event: "error", listener: (err: Error) => void): this;
  on(event: "end", listener: () => void): this;
}

/** Client -> server report that the caller's own evaluate /
 * makeReflectiveDataset callback threw. */
export interface WireEvaluatorFailure {
  request_id: string;
  error_type: string;
  message: string;
}
