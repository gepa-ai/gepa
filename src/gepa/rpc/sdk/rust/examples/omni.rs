use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use gepa_sdk::{Client, Example, OmniEvalResult, OmniOptimizeOpts};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let make_example = |id: &str, text: &str, label: &str| Example {
        id: id.to_string(),
        fields: HashMap::from([
            ("text".to_string(), text.to_string()),
            ("label".to_string(), label.to_string()),
        ]),
    };

    let dataset = vec![
        make_example("1", "I love this product!", "positive"),
        make_example("2", "This is terrible.", "negative"),
        make_example("3", "Absolutely fantastic experience.", "positive"),
        make_example("4", "Worst purchase I have ever made.", "negative"),
    ];

    let client = Client::new("localhost:50051");

    let millis = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

    let result = client
        .optimize_omni(OmniOptimizeOpts {
            run_id: format!("omni-demo-{millis}"),
            seed_candidate: Some(
                "Classify the sentiment of the following text as positive or negative."
                    .to_string(),
            ),
            dataset: Some(dataset),
            valset: None,
            objective: Some("Maximize accuracy of sentiment classification.".to_string()),
            reflection_lm: Some("openai/gpt-4o-mini".to_string()),
            engine: None, // defaults to "gepa"
            max_evals: 30,

            evaluate: |req: gepa_sdk::OmniEvalRequest| async move {
                let lower_candidate = req.candidate.to_lowercase();
                let is_classification_prompt = lower_candidate.contains("positive")
                    || lower_candidate.contains("negative")
                    || lower_candidate.contains("sentiment")
                    || lower_candidate.contains("classify");

                let scores = req
                    .batch
                    .iter()
                    .map(|_| if is_classification_prompt { 1.0 } else { 0.0 })
                    .collect();

                Ok(OmniEvalResult {
                    scores,
                    side_infos: None,
                })
            },

            on_progress: Some(Box::new(|u| {
                println!(
                    "[progress] {}/{} best={:.3} | \"{}...\"",
                    u.evals_used,
                    u.max_evals,
                    u.best_score,
                    &u.best_candidate.chars().take(60).collect::<String>()
                );
            })),
        })
        .await?;

    println!("\noptimization complete");
    println!("  runId:        {}", result.run_id);
    println!("  bestScore:    {}", result.best_score);
    println!("  totalEvals:   {}", result.total_evals);
    println!("  bestCandidate:{}", result.best_candidate);

    Ok(())
}
