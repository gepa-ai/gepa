use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use gepa_sdk::{Client, EvalResult, Example, OptimizeOpts, ReflectiveEntry, ReflectiveResult};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let make_example = |id: &str, input: &str, answer: &str| Example {
        id: id.to_string(),
        fields: HashMap::from([
            ("input".to_string(), input.to_string()),
            ("answer".to_string(), answer.to_string()),
        ]),
    };

    let trainset = vec![
        make_example("1", "What is 2+2?", "4"),
        make_example("2", "Capital of France?", "Paris"),
        make_example("3", "Color of the sky?", "blue"),
    ];

    let client = Client::new("localhost:50051");

    let millis = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

    let result = client
        .optimize(OptimizeOpts {
            run_id: format!("demo-{millis}"),
            seed_candidate: HashMap::from([(
                "instructions".to_string(),
                "Answer the question in one word.".to_string(),
            )]),
            trainset,
            valset: None,
            max_metric_calls: 20,

            evaluate: |req: gepa_sdk::EvalRequest| async move {
                let instructions =
                    req.candidate.get("instructions").unwrap_or("").to_lowercase();
                let mut outputs = Vec::new();
                let mut scores = Vec::new();
                let mut trajectories = Vec::new();

                for ex in &req.batch {
                    let expected =
                        ex.fields.get("answer").map(|s| s.as_str()).unwrap_or("");
                    let output = if instructions.contains("answer") {
                        expected.to_string()
                    } else {
                        ex.fields.get("input").cloned().unwrap_or_default()
                    };
                    let correct = output == expected;
                    if req.capture_traces {
                        trajectories.push(gepa_sdk::Trajectory {
                            input_fields: ex.fields.clone(),
                            output: output.clone(),
                            feedback: if correct {
                                format!("Correct: produced \"{}\".", expected)
                            } else {
                                format!("Wrong: expected \"{}\" but got \"{}\".", expected, output)
                            },
                        });
                    }
                    outputs.push(output);
                    scores.push(if correct { 1.0 } else { 0.0 });
                }

                Ok(EvalResult {
                    outputs,
                    scores,
                    trajectories: if req.capture_traces { Some(trajectories) } else { None },
                })
            },

            make_reflective_dataset: |req: gepa_sdk::ReflectiveRequest| async move {
                let mut result = ReflectiveResult::new();
                for comp in &req.components_to_update {
                    let entries = req
                        .trajectories
                        .iter()
                        .map(|t| ReflectiveEntry {
                            inputs: t.input_fields.clone(),
                            generated_output: t.output.clone(),
                            feedback: t.feedback.clone(),
                        })
                        .collect();
                    result.insert(comp.clone(), entries);
                }
                Ok(result)
            },

            on_progress: Some(Box::new(|u| {
                println!(
                    "[progress] {}/{} best={:.3}",
                    u.metric_calls_used, u.max_metric_calls, u.best_score
                );
            })),
        })
        .await?;

    println!("best candidate: {:?}", result.best_candidate.as_map());
    println!("best score: {}", result.best_score);

    Ok(())
}
