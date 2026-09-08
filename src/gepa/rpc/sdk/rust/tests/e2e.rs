//! Real end-to-end test: runs an actual GEPA optimization through the real
//! Rust client and a real gepa-rpc server subprocess, with no mocking of
//! gepa.optimize/optimize_anything anywhere in the path.
//!
//! Ports tests/test_rpc_e2e/test_rpc_e2e.py's design exactly (same dataset,
//! seed candidate, and grading) and reuses its recorded cache and golden
//! file, since the wire protocol and task logic are identical regardless of
//! client language. Set RECORD_TESTS=true to regenerate them against a real
//! OPENAI_API_KEY (only needs to be done once, from any one language).

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use gepa_sdk::{Client, Example, OmniEvalRequest, OmniEvalResult, OmniOptimizeOpts};
use tokio::io::AsyncReadExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::process::{Child, Command};
use tokio::time::timeout;

const TASK_MODEL: &str = "openai/gpt-4.1-nano";
const REFLECTION_MODEL: &str = "openai/gpt-4.1-nano";

fn e2e_dir() -> PathBuf {
    // src/gepa/rpc/sdk/rust/tests -> repo root -> tests/test_rpc_e2e
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../../tests/test_rpc_e2e")
}

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

fn grade(answer: &str, accepted_labels: &str) -> f32 {
    let first_word = answer
        .trim()
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(|c| c == '.' || c == ',' || c == '!' || c == '"' || c == '\'')
        .to_lowercase();
    accepted_labels
        .to_lowercase()
        .split(',')
        .any(|label| label == first_word)
        .then_some(1.0)
        .unwrap_or(0.0)
}

async fn call_task_lm(
    http: &reqwest::Client,
    stub_port: u16,
    candidate: &str,
    text: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let prompt = format!("{candidate}\n\nText: {text}");
    let resp: serde_json::Value = http
        .post(format!("http://127.0.0.1:{stub_port}/chat/completions"))
        .json(&serde_json::json!({
            "model": TASK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        }))
        .send()
        .await?
        .json()
        .await?;
    Ok(resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string())
}

async fn read_all(mut reader: impl tokio::io::AsyncRead + Unpin) -> String {
    let mut buf = String::new();
    let _ = reader.read_to_string(&mut buf).await;
    buf
}

async fn kill(mut child: Child) {
    let _ = child.start_kill();
    let _ = timeout(Duration::from_secs(5), child.wait()).await;
}

fn dataset() -> Vec<Example> {
    let ex = |id: &str, text: &str, label: &str| Example {
        id: id.to_string(),
        fields: HashMap::from([
            ("text".to_string(), text.to_string()),
            ("label".to_string(), label.to_string()),
        ]),
    };
    vec![
        ex("1", "I love this product!", "positive"),
        ex("2", "This is terrible.", "negative"),
        ex("3", "Oh great, another Monday.", "negative"),
        ex("4", "Well, that could have gone better.", "negative"),
        ex("5", "Not bad at all, actually.", "positive"),
        ex("6", "I guess it's fine, whatever.", "negative,neutral"),
    ]
}

#[tokio::test]
async fn test_real_optimization_through_omni() {
    let record = env::var("RECORD_TESTS").unwrap_or_default().to_lowercase() == "true";
    let python = env::var("GEPA_RPC_PYTHON").unwrap_or_else(|_| "python3".to_string());

    let dir = e2e_dir();
    let cache_file = dir.join("llm_cache.json");
    let golden_file = dir.join("optimized_candidate.txt");

    if !record && !cache_file.exists() {
        panic!("Cache file not found: {cache_file:?}. Run with RECORD_TESTS=true to generate it.");
    }

    let stub_port = free_port().await;
    let server_port = free_port().await;

    let mut stub_args = vec![
        "-m".to_string(),
        "gepa.rpc.testing.fake_llm_server".to_string(),
        "--port".to_string(),
        stub_port.to_string(),
        "--cache-file".to_string(),
        cache_file.to_string_lossy().to_string(),
    ];
    if record {
        stub_args.push("--record".to_string());
    }
    let mut stub_child = Command::new(&python)
        .args(&stub_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn fake_llm_server");
    let stub_stdout = stub_child.stdout.take().unwrap();
    let stub_stderr = stub_child.stderr.take().unwrap();

    let runs_dir = std::env::temp_dir().join(format!(
        "gepa-rpc-rust-e2e-{}",
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis()
    ));
    fs::create_dir_all(&runs_dir).unwrap();

    let mut server_child = Command::new(&python)
        .args([
            "-m",
            "gepa.rpc.cli",
            "--port",
            &server_port.to_string(),
            "--runs-dir",
            &runs_dir.to_string_lossy(),
        ])
        .env("OPENAI_API_BASE", format!("http://127.0.0.1:{stub_port}"))
        .env(
            "OPENAI_API_KEY",
            env::var("OPENAI_API_KEY").unwrap_or_else(|_| "sk-fake-e2e-test-key".to_string()),
        )
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn gepa-rpc server");
    let server_stdout = server_child.stdout.take().unwrap();
    let server_stderr = server_child.stderr.take().unwrap();

    let result = run_test(record, stub_port, server_port, &golden_file).await;

    if let Err(e) = &result {
        eprintln!("--- fake_llm_server stdout ---\n{}", read_all(stub_stdout).await);
        eprintln!("--- fake_llm_server stderr ---\n{}", read_all(stub_stderr).await);
        eprintln!("--- gepa-rpc server stdout ---\n{}", read_all(server_stdout).await);
        eprintln!("--- gepa-rpc server stderr ---\n{}", read_all(server_stderr).await);
        kill(server_child).await;
        kill(stub_child).await;
        panic!("{e}");
    }

    kill(server_child).await;
    kill(stub_child).await;
}

async fn run_test(
    record: bool,
    stub_port: u16,
    server_port: u16,
    golden_file: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    wait_for_port(stub_port, 10_000).await;
    wait_for_port(server_port, 10_000).await;

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let client = Client::new(format!("localhost:{server_port}"));
    let millis = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

    let result = client
        .optimize_omni(OmniOptimizeOpts {
            run_id: format!("rpc-e2e-sentiment-rust-{millis}"),
            seed_candidate: Some("Classify the sentiment of the following text.".to_string()),
            dataset: Some(dataset()),
            valset: None,
            objective: Some("Maximize accuracy of sentiment classification.".to_string()),
            reflection_lm: Some(REFLECTION_MODEL.to_string()),
            engine: None,
            max_evals: 30,
            evaluate: |req: OmniEvalRequest| {
                let http = http.clone();
                async move {
                    let mut scores = Vec::with_capacity(req.batch.len());
                    for ex in &req.batch {
                        let text = ex.fields.get("text").map(String::as_str).unwrap_or("");
                        let label = ex.fields.get("label").map(String::as_str).unwrap_or("");
                        let answer = call_task_lm(&http, stub_port, &req.candidate, text)
                            .await
                            .map_err(|e| gepa_sdk::GEPAError::OptimizationFailed(e.to_string()))?;
                        scores.push(grade(&answer, label));
                    }
                    Ok(OmniEvalResult {
                        scores,
                        side_infos: None,
                    })
                }
            },
            on_progress: None,
        })
        .await?;

    if record {
        fs::write(golden_file, &result.best_candidate)?;
        if result.best_candidate.is_empty() {
            return Err("RECORD_TESTS run produced an empty best candidate".into());
        }
        println!("RECORD_TESTS: wrote golden file: {golden_file:?}");
    } else {
        let expected = fs::read_to_string(golden_file)?;
        if result.best_candidate != expected {
            return Err(format!(
                "best candidate does not match golden file.\n  expected: {expected:?}\n  actual:   {:?}",
                result.best_candidate
            )
            .into());
        }
    }

    println!("PASS: test_real_optimization_through_omni (Rust)");
    Ok(())
}
