//! Exercises the real wire protocol for the non-Omni optimize() path --
//! request/response shapes, trajectories, and the reflective-dataset round
//! trip -- against a real gepa-rpc server subprocess.
//!
//! Unlike e2e.rs this needs no recorded LLM cache: the server runs with
//! gepa.optimize patched to a deterministic fake (see
//! gepa.rpc.testing.mock_optimize_server), so no real or replayed LLM calls
//! happen. e2e.rs covers the Omni path for real; this covers the non-Omni
//! path, which neither SDK's tests covered before.

use std::collections::HashMap;
use std::env;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use gepa_sdk::{Client, EvalRequest, EvalResult, Example, OptimizeOpts, ReflectiveEntry, ReflectiveResult};
use tokio::net::{TcpListener, TcpStream};
use tokio::process::Command;
use tokio::time::timeout;

async fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap().port()
}

async fn wait_for_port(port: u16, timeout_ms: u64) {
    let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms);
    loop {
        if TcpStream::connect(("127.0.0.1", port)).await.is_ok() {
            return;
        }
        if tokio::time::Instant::now() > deadline {
            panic!("nothing listening on 127.0.0.1:{port} after {timeout_ms}ms");
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test]
async fn test_optimize_non_omni_wire_protocol() {
    let python = env::var("GEPA_RPC_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let port = free_port().await;
    let runs_dir = std::env::temp_dir().join("gepa-rpc-rust-mocked");

    let mut server = Command::new(&python)
        .args([
            "-m", "gepa.rpc.testing.mock_optimize_server",
            "--port", &port.to_string(),
            "--runs-dir", &runs_dir.to_string_lossy(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn mock_optimize_server");

    let result = run(port).await;

    let _ = server.start_kill();
    let _ = timeout(Duration::from_secs(5), server.wait()).await;

    if let Err(e) = result {
        panic!("{e}");
    }
}

async fn run(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    wait_for_port(port, 10_000).await;

    let seed: HashMap<String, String> =
        HashMap::from([("instructions".to_string(), "do the thing".to_string())]);
    let trainset = vec![
        Example::new("1", HashMap::from([("x".to_string(), "a".to_string())])),
        Example::new("2", HashMap::from([("x".to_string(), "b".to_string())])),
    ];
    let expected_batch = trainset.len();

    let saw_capture_traces = Arc::new(AtomicBool::new(false));
    let saw_reflective = Arc::new(AtomicBool::new(false));
    let traces_flag = saw_capture_traces.clone();
    let reflective_flag = saw_reflective.clone();

    let client = Client::new(format!("localhost:{port}"))?;
    let result = client
        .optimize(OptimizeOpts {
            run_id: "rust-mocked".to_string(),
            seed_candidate: seed.clone(),
            trainset,
            valset: None,
            reflection_lm: None,
            max_metric_calls: 10,
            evaluate: move |req: EvalRequest| {
                let traces_flag = traces_flag.clone();
                async move {
                    if req.capture_traces {
                        traces_flag.store(true, Ordering::SeqCst);
                    }
                    let outputs: Vec<String> = req.batch.iter().map(|_| "out".to_string()).collect();
                    let scores: Vec<f64> = req.batch.iter().map(|_| 1.0).collect();
                    let trajectories = if req.capture_traces {
                        Some(
                            req.batch
                                .iter()
                                .map(|ex| gepa_sdk::Trajectory::new(ex.fields.clone(), "out", "good"))
                                .collect(),
                        )
                    } else {
                        None
                    };
                    Ok(EvalResult::new(outputs, scores, trajectories))
                }
            },
            make_reflective_dataset: move |req: gepa_sdk::ReflectiveRequest| {
                let reflective_flag = reflective_flag.clone();
                async move {
                    reflective_flag.store(true, Ordering::SeqCst);
                    assert_eq!(
                        req.trajectories.len(), expected_batch,
                        "server sent the wrong number of trajectories",
                    );
                    let mut out = ReflectiveResult::new();
                    for comp in &req.components_to_update {
                        out.insert(
                            comp.clone(),
                            vec![ReflectiveEntry::new(HashMap::new(), "", "")],
                        );
                    }
                    Ok(out)
                }
            },
            on_progress: None,
        })
        .await?;

    assert_eq!(result.best_candidate.get("instructions"), Some("do the thing"));
    assert_eq!(result.best_score, 1.0);
    assert!(saw_capture_traces.load(Ordering::SeqCst), "server never requested trajectories");
    assert!(saw_reflective.load(Ordering::SeqCst), "make_reflective_dataset was never called");

    println!("PASS: optimize() non-Omni wire protocol (Rust, mocked optimizer)");
    Ok(())
}
