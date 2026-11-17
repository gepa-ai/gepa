import type { ChannelCredentials } from "@grpc/grpc-js";

/** Distinguishes the client's own evaluator mistake from a server/transport
 * failure -- mirrors OptimizationError.error_kind in gepa.proto. */
export type ErrorKind = "evaluator_error" | "internal_error" | "cancelled" | "unspecified";

/** Thrown by optimize()/optimizeOmni() when the server reports an
 * optimization_error. `errorKind` lets a caller tell its own callback's bug
 * apart from a server-side failure without parsing the message string. */
export class OptimizationError extends Error {
  readonly runId: string;
  readonly errorKind: ErrorKind;

  constructor(runId: string, message: string, errorKind: ErrorKind) {
    super(`optimization ${runId || "(unknown)"} failed: ${message}`);
    this.name = "OptimizationError";
    this.runId = runId;
    this.errorKind = errorKind;
  }
}

export interface Example {
  id: string;
  fields: Record<string, string>;
}

export interface Trajectory {
  inputFields: Record<string, string>;
  output: string;
  feedback: string;
}

export interface ReflectiveEntry {
  inputs: Record<string, string>;
  generatedOutput: string;
  feedback: string;
}

export interface EvaluateBatchArgs {
  requestId: string;
  candidate: Record<string, string>;
  batch: Example[];
  captureTraces: boolean;
}

export interface EvaluateBatchResult {
  outputs: string[];
  scores: number[];
  trajectories?: Trajectory[];
}

export interface ReflectiveDatasetArgs {
  requestId: string;
  candidate: Record<string, string>;
  componentsToUpdate: string[];
  trajectories: Trajectory[];
}

export type ReflectiveDatasetResult = Record<string, ReflectiveEntry[]>;

export interface ProgressUpdate {
  metricCallsUsed: number;
  maxMetricCalls: number;
  bestScore: number;
  bestCandidate: Record<string, string>;
}

export interface OptimizeOptions {
  runId: string;
  seedCandidate: Record<string, string>;
  trainset: Example[];
  valset?: Example[];
  reflectionLm?: string;
  maxMetricCalls: number;

  evaluate: (args: EvaluateBatchArgs) => Promise<EvaluateBatchResult>;
  makeReflectiveDataset: (
    args: ReflectiveDatasetArgs,
  ) => Promise<ReflectiveDatasetResult>;
  onProgress?: (update: ProgressUpdate) => void;

  /** Aborts the call: rejects the returned promise and cancels the underlying
   * stream. The server keeps running until it notices -- use Client.cancel()
   * to ask it to stop too. */
  signal?: AbortSignal;
  /** Milliseconds to wait for the whole call before rejecting. Affects only
   * what this client observes; the server is unaffected. */
  timeoutMs?: number;
}

export interface OptimizeResult {
  runId: string;
  bestCandidate: Record<string, string>;
  bestScore: number;
}

export interface ClientOptions {
  target: string;
  credentials?: ChannelCredentials;
}

// ------------------------------------------------------------------ Omni types

export interface OmniBestEval {
  score: number;
  sideInfo: unknown;
}

export interface OmniOptimizationState {
  /** Top-K best prior evaluations for this example, sorted descending by score. */
  bestExampleEvals: OmniBestEval[];
}

export interface OmniEvaluateBatchArgs {
  requestId: string;
  /** String candidate (prompt, code, instructions, etc.) */
  candidate: string;
  batch: Example[];
  /**
   * Per-example warm-start history, aligned 1:1 with `batch`. Empty when the
   * server didn't send opt_states (e.g. an engine that doesn't produce them).
   */
  optStates: OmniOptimizationState[];
}

export interface OmniEvaluateBatchResult {
  scores: number[];
  /** Per-example side-info objects (free-form, serialised to JSON by the server). */
  sideInfos?: Record<string, unknown>[];
}

export interface OmniProgressUpdate {
  evalsUsed: number;
  maxEvals: number;
  bestScore: number;
  bestCandidate: string;
}

export interface OptimizeOmniOptions {
  runId: string;
  /** Initial candidate string to optimise from. */
  seedCandidate?: string;
  dataset?: Example[];
  valset?: Example[];
  objective?: string;
  /** Long-form task context (problem statement, evaluation rules) passed
   * straight through to optimize_anything's own `background` parameter. */
  background?: string;
  reflectionLm?: string;
  /**
   * Optimize Anything backend: "gepa" (default), "autoresearch", "best_of_n",
   * or "meta_harness". Only "gepa" applies reflectionLm and streams
   * onProgress updates today; other engines run to completion and resolve
   * with the final result only.
   */
  engine?: string;
  maxEvals?: number;

  evaluate: (args: OmniEvaluateBatchArgs) => Promise<OmniEvaluateBatchResult>;
  onProgress?: (update: OmniProgressUpdate) => void;

  /** Aborts the call: rejects the returned promise and cancels the underlying
   * stream. The server keeps running until it notices -- use Client.cancel()
   * to ask it to stop too. */
  signal?: AbortSignal;
  /** Milliseconds to wait for the whole call before rejecting. Affects only
   * what this client observes; the server is unaffected. */
  timeoutMs?: number;
}

export interface OptimizeOmniResult {
  runId: string;
  bestCandidate: string;
  bestScore: number;
  totalEvals: number;
}
