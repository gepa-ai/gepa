//! Verifies the Rust SDK can reach a server started with --auth-token, and is
//! rejected without it. Before this the SDK had no way to send `authorization`
//! metadata, so a hardened server was unreachable from the official client.

use std::collections::HashMap;
use std::env;
use std::process::Stdio;
use std::time::Duration;

use gepa_sdk::{Client, EvalRequest, EvalResult, Example, GEPAError, OptimizeOpts, ReflectiveResult};
use tokio::net::{TcpListener, TcpStream};
use tokio::process::Command;
use tokio::time::timeout;

const TOKEN: &str = "s3cret-token";

async fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    l.local_addr().unwrap().port()
}

async fn wait_for_port(port: u16, ms: u64) {
    let deadline = tokio::time::Instant::now() + Duration::from_millis(ms);
    loop {
        if TcpStream::connect(("127.0.0.1", port)).await.is_ok() {
            return;
        }
        if tokio::time::Instant::now() > deadline {
            panic!("nothing listening on 127.0.0.1:{port} after {ms}ms");
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn opts(run_id: &str) -> OptimizeOpts<
    impl FnMut(EvalRequest) -> std::future::Ready<Result<EvalResult, GEPAError>>,
    impl FnMut(gepa_sdk::ReflectiveRequest) -> std::future::Ready<Result<ReflectiveResult, GEPAError>>,
> {
    OptimizeOpts {
        run_id: run_id.to_string(),
        seed_candidate: HashMap::from([("instructions".to_string(), "do the thing".to_string())]),
        trainset: vec![Example::new("1", HashMap::from([("x".to_string(), "a".to_string())]))],
        valset: None,
        reflection_lm: None,
        max_metric_calls: 10,
        evaluate: |req: EvalRequest| {
            let n = req.batch.len();
            let trajectories = if req.capture_traces {
                Some(
                    req.batch
                        .iter()
                        .map(|ex| gepa_sdk::Trajectory::new(ex.fields.clone(), "out", "ok"))
                        .collect(),
                )
            } else {
                None
            };
            std::future::ready(Ok(EvalResult::new(
                vec!["out".to_string(); n],
                vec![1.0; n],
                trajectories,
            )))
        },
        make_reflective_dataset: |_req: gepa_sdk::ReflectiveRequest| {
            std::future::ready(Ok(ReflectiveResult::new()))
        },
        on_progress: None,
    }
}

#[tokio::test]
async fn test_auth_token_is_required_and_sufficient() {
    let python = env::var("GEPA_RPC_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let port = free_port().await;
    let runs_dir = std::env::temp_dir().join("gepa-rpc-rust-auth");

    let mut server = Command::new(&python)
        .args([
            "-m", "gepa.rpc.testing.mock_optimize_server",
            "--port", &port.to_string(),
            "--runs-dir", &runs_dir.to_string_lossy(),
            "--auth-token", TOKEN,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn mock_optimize_server");

    let outcome = run(port).await;

    let _ = server.start_kill();
    let _ = timeout(Duration::from_secs(5), server.wait()).await;
    if let Err(e) = outcome {
        panic!("{e}");
    }
}

async fn run(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    wait_for_port(port, 10_000).await;

    // No token -> rejected.
    let anon = Client::new(format!("localhost:{port}"))?;
    match anon.optimize(opts("auth-missing")).await {
        Err(GEPAError::Status(msg)) => {
            assert!(
                msg.contains("Unauthenticated") || msg.to_lowercase().contains("auth"),
                "expected an auth failure, got: {msg}"
            );
        }
        other => return Err(format!("expected rejection without a token, got {other:?}").into()),
    }
    println!("PASS: unauthenticated call is rejected");

    // Wrong token -> rejected.
    let wrong = Client::new(format!("localhost:{port}"))?.with_auth_token("nope")?;
    match wrong.optimize(opts("auth-wrong")).await {
        Err(GEPAError::Status(_)) => {}
        other => return Err(format!("expected rejection with a wrong token, got {other:?}").into()),
    }
    println!("PASS: wrong token is rejected");

    // Correct token -> the run succeeds.
    let authed = Client::new(format!("localhost:{port}"))?.with_auth_token(TOKEN)?;
    let result = authed.optimize(opts("auth-ok")).await?;
    assert_eq!(result.best_candidate.get("instructions"), Some("do the thing"));
    println!("PASS: correct token authenticates the streaming call");
    Ok(())
}
