use std::collections::HashMap;

pub struct Example {
    pub id: String,
    pub fields: HashMap<String, String>,
}

pub struct Candidate {
    inner: HashMap<String, String>,
}

impl Candidate {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.inner.get(key).map(|s| s.as_str())
    }

    pub fn as_map(&self) -> &HashMap<String, String> {
        &self.inner
    }

    pub(crate) fn from_map(map: HashMap<String, String>) -> Self {
        Self { inner: map }
    }
}

pub struct Trajectory {
    pub input_fields: HashMap<String, String>,
    pub output: String,
    pub feedback: String,
}

pub struct EvalRequest {
    pub request_id: String,
    pub candidate: Candidate,
    pub batch: Vec<Example>,
    pub capture_traces: bool,
}

pub struct EvalResult {
    pub outputs: Vec<String>,
    pub scores: Vec<f32>,
    pub trajectories: Option<Vec<Trajectory>>,
}

pub struct ReflectiveEntry {
    pub inputs: HashMap<String, String>,
    pub generated_output: String,
    pub feedback: String,
}

pub struct ReflectiveRequest {
    pub request_id: String,
    pub candidate: Candidate,
    pub components_to_update: Vec<String>,
    pub trajectories: Vec<Trajectory>,
}

pub type ReflectiveResult = HashMap<String, Vec<ReflectiveEntry>>;

pub struct ProgressUpdate {
    pub metric_calls_used: i32,
    pub max_metric_calls: i32,
    pub best_score: f32,
    pub best_candidate: Candidate,
}

pub struct OptimizeResult {
    pub run_id: String,
    pub best_candidate: Candidate,
    pub best_score: f32,
}

pub struct OptimizeOpts<E, M> {
    pub run_id: String,
    pub seed_candidate: HashMap<String, String>,
    pub trainset: Vec<Example>,
    pub valset: Option<Vec<Example>>,
    pub max_metric_calls: i32,
    pub evaluate: E,
    pub make_reflective_dataset: M,
    pub on_progress: Option<Box<dyn Fn(ProgressUpdate) + Send>>,
}

// ------------------------------------------------------------------ Omni types

pub struct OmniBestEval {
    pub score: f32,
    /// JSON-encoded side-info string (or empty).
    pub side_info: String,
}

pub struct OmniOptState {
    /// Top-K best prior evaluations for this example, sorted descending by score.
    pub best_example_evals: Vec<OmniBestEval>,
}

pub struct OmniEvalRequest {
    pub request_id: String,
    /// String candidate (prompt, code, instructions, etc.)
    pub candidate: String,
    pub batch: Vec<Example>,
    /// Per-example warm-start history, aligned 1:1 with `batch`. Empty when
    /// the server didn't send opt_states.
    pub opt_states: Vec<OmniOptState>,
}

pub struct OmniEvalResult {
    pub scores: Vec<f32>,
    /// Optional per-example side-info, one JSON-encoded string per example
    /// (or an empty string for none). Omitted entries are treated as `{}`
    /// server-side.
    pub side_infos: Option<Vec<String>>,
}

pub struct OmniProgressUpdate {
    pub evals_used: i32,
    pub max_evals: i32,
    pub best_score: f32,
    pub best_candidate: String,
}

pub struct OmniOptimizeResult {
    pub run_id: String,
    pub best_candidate: String,
    pub best_score: f32,
    pub total_evals: i32,
}

pub struct OmniOptimizeOpts<E> {
    pub run_id: String,
    /// Initial candidate string to optimize from. `None` for seedless mode.
    pub seed_candidate: Option<String>,
    pub dataset: Option<Vec<Example>>,
    pub valset: Option<Vec<Example>>,
    pub objective: Option<String>,
    pub reflection_lm: Option<String>,
    /// Optimize Anything backend: "gepa" (default), "autoresearch",
    /// "best_of_n", or "meta_harness". Only "gepa" applies `reflection_lm`
    /// and streams `on_progress` updates today; other engines run to
    /// completion and resolve with the final result only.
    pub engine: Option<String>,
    pub max_evals: i32,
    pub evaluate: E,
    pub on_progress: Option<Box<dyn Fn(OmniProgressUpdate) + Send>>,
}
