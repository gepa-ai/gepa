# gepa.rpc

A bidirectional gRPC interface in front of GEPA so Rust and JavaScript developers can drive prompt optimization while providing **native-language evaluators** -- no Python required on the client side.

Two endpoints are available:

- **`RunOptimization`** wraps `gepa.optimize`. The client supplies structured `(trainset, valset, seed_candidate)` and implements `evaluate` + `makeReflectiveDataset`. Full multi-component prompt optimization.
- **`RunOptimizationOmni`** wraps `optimize_anything`. The client supplies a single `evaluate(candidate, example) -> (score, side_info)` callback. Simpler contract; selectable across all engines registered in `gepa` (`gepa`, `autoresearch`, `best_of_n`, `meta_harness`) via `OmniStartRequest.engine` (defaults to `"gepa"`). **Caveat:** only the `gepa` engine applies `reflection_lm` and streams `OmniProgressUpdate`s today -- other engines run to completion with their own defaults and report the final `OmniOptimizationComplete` only (no live progress).

```
            ┌──────────────────────────┐                 ┌──────────────────────────┐
            │  client (TS / Rust)      │                 │  gepa-rpc server (py)    │
            │                          │  StartRequest   │                          │
  user ───▶ │  client.optimize({       │ ──────────────▶ │  GEPAServicer            │
            │    evaluate,             │                 │     │                    │
            │    makeReflectiveDataset │ ◀── Eval req ── │     ▼                    │
            │  })                      │ ── Eval resp ─▶ │  gepa.optimize(          │
            │                          │                 │     adapter=             │
            │                          │ ◀── Progress ── │     RemoteAdapter)       │
            │                          │ ◀─ Complete ─── │                          │
            └──────────────────────────┘                 └──────────────────────────┘

            ┌──────────────────────────┐                 ┌──────────────────────────┐
            │  client (TS / Rust)      │                 │  gepa-rpc server (py)    │
            │                          │  StartRequest   │                          │
  user ───▶ │  client.optimizeOmni({   │ ──────────────▶ │  GEPAServicer            │
            │    evaluate,             │                 │     │                    │
            │  })                      │ ◀── Eval req ── │     ▼                    │
            │                          │ ── Eval resp ─▶ │  optimize_anything(      │
            │                          │                 │     batch_evaluator=     │
            │                          │ ◀── Progress ── │     OmniRemoteEvaluator) │
            │                          │ ◀─ Complete ─── │                          │
            └──────────────────────────┘                 └──────────────────────────┘
```

## Quickstart

### 1. Start the server

```bash
pip install "gepa[rpc] @ git+https://github.com/gepa-ai/gepa.git@main"
# once a release including gepa.rpc is on PyPI, this becomes: pip install "gepa[rpc]"
gepa-rpc --port 50051 --runs-dir ./runs
```

State per run is checkpointed under `./runs/<run_id>/` so reconnecting with the same `run_id` resumes from the last saved iteration.

### 2. Drive it from TypeScript

```bash
cd src/gepa/rpc/sdk/typescript
npm install && npm run build
npx tsx examples/basic.ts
```

`examples/basic.ts` walks through the full `client.optimize()` API with a stand-in evaluator; `examples/omni.ts` does the same for `client.optimizeOmni()`.

### 3. Drive it from Rust

```bash
cd src/gepa/rpc/sdk/rust
cargo run --example basic
cargo run --example omni
```

`examples/basic.rs` walks through `Client::optimize()`; `examples/omni.rs` does the same for `Client::optimize_omni()`.

### 4. Run the RPC integration tests

```bash
pip install -e ".[dev,rpc]"
pytest tests/test_rpc_integration.py -v
```

Tests spin up a real gRPC server and drive both endpoints end-to-end with a mock optimizer (no LLM calls needed). This test module is skipped automatically if `grpcio` isn't installed (i.e. when the `rpc` extra wasn't requested), so it never breaks the default `gepa` test run.

### 5. Build the Docker image

```bash
docker build -f src/gepa/rpc/Dockerfile -t gepa-rpc:latest .   # from the repo root
docker run -p 50051:50051 -e OPENAI_API_KEY=$OPENAI_API_KEY gepa-rpc:latest
```

## Repo layout

```
src/gepa/rpc/
  proto/gepa.proto              canonical service + message definitions
  generated/                    protoc output (committed)
  conversions.py                RemoteExample / RemoteTrajectory dataclasses
  adapter.py                    RemoteAdapter (GEPAAdapter) + OmniRemoteEvaluator
  servicer.py                   GEPAServicer -- RunOptimization + RunOptimizationOmni handlers
  server.py                     build_server() / serve()
  cli.py                        `gepa-rpc` console script
  scripts/compile_proto.sh      regenerates generated/ from proto/gepa.proto
  Dockerfile                    build with the repo root as context (see Quickstart)
  sdk/typescript/                @gepa/sdk npm package
    src/{types,client,index}.ts
    examples/{basic,omni}.ts
    proto/gepa.proto             synced from ../../proto via scripts/sync-proto.sh
  sdk/rust/                      gepa-sdk Rust crate
    src/{client,types,error}.rs
    examples/{basic,omni}.rs
    proto/gepa.proto             synced from ../../proto via scripts/compile_proto.sh
tests/test_rpc_integration.py    gRPC integration tests (pytest, skipped without the rpc extra)
```

## Notes

- `reflection_lm` defaults to `"openai/gpt-5.1"`. Override it per-run via `StartRequest.reflection_lm` or `OmniStartRequest.reflection_lm`.
- `OmniStartRequest.engine` selects the Optimize Anything backend (`"gepa"` if unset). Only `gepa` gets `reflection_lm` applied and streams progress; `autoresearch`/`best_of_n`/`meta_harness` run with their own defaults and only report the final result. See the `RunOptimizationOmni` caveat above.
- Disconnect-resume relies on `gepa.optimize`'s built-in `run_dir` checkpointing. Re-issue `RunOptimization` with the same `run_id` to resume.
- The TypeScript SDK uses `@grpc/proto-loader` at runtime; user-facing types are hand-written in `sdk/typescript/src/types.ts`.
- `OmniRemoteEvaluator` groups `(candidate, example)` pairs by candidate before sending each batch request, matching the `batch_evaluator` contract from `optimize_anything`.
- `OmniEvaluateBatchRequest.opt_states` carries each example's top-K best prior evaluations (warm-start history), aligned 1:1 with `batch`, when the engine provides it. Empty on an example's first evaluation.
- The `rpc` extra depends on `litellm`/`tenacity` directly rather than via `gepa[full]`/`gepa[confidence]` -- the base `gepa` package doesn't pull litellm in on its own, and reflection_lm calls fail immediately without it.
- No authentication or TLS on the gRPC server, and `max_evals`/`max_metric_calls` are unenforced if the client omits them -- anyone with network access to the port can start runs that spend the server operator's LLM budget. Put this behind your own auth/network boundary before exposing it beyond a trusted network.
