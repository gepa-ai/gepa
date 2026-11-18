use std::collections::HashMap;

// The data types below are #[non_exhaustive] + Default: callers only ever
// *receive* them, so adding a field later shouldn't break their code. The two
// Opts structs are excluded -- callers must build those with required
// closures (`evaluate`), so marking them #[non_exhaustive] would make them
// unconstructable outside this crate without also adding a builder, which is
// beyond this pass. Types callers DO construct get an explicit `new()`, since
// #[non_exhaustive] blocks struct-literal syntax cross-crate (and
// `..Default::default()` does not work around that).

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct Example {
    pub id: String,
    pub fields: HashMap<String, String>,
}

#[derive(Debug, Clone, Default)]
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

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct Trajectory {
    pub input_fields: HashMap<String, String>,
    pub output: String,
    pub feedback: String,
}

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct EvalRequest {
    pub request_id: String,
    pub candidate: Candidate,
    pub batch: Vec<Example>,
    pub capture_traces: bool,
}

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct EvalResult {
    pub outputs: Vec<String>,
    pub scores: Vec<f64>,
    pub trajectories: Option<Vec<Trajectory>>,
}

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ReflectiveEntry {
    pub inputs: HashMap<String, String>,
    pub generated_output: String,
    pub feedback: String,
}

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ReflectiveRequest {
    pub request_id: String,
    pub candidate: Candidate,
    pub components_to_update: Vec<String>,
    pub trajectories: Vec<Trajectory>,
}

pub type ReflectiveResult = HashMap<String, Vec<ReflectiveEntry>>;

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ProgressUpdate {
    pub metric_calls_used: i32,
    pub max_metric_calls: i32,
    pub best_score: f64,
    pub best_candidate: Candidate,
}

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct OptimizeResult {
    pub run_id: String,
    pub best_candidate: Candidate,
    pub best_score: f64,
}

pub struct OptimizeOpts<E, M> {
    pub run_id: String,
    pub seed_candidate: HashMap<String, String>,
    pub trainset: Vec<Example>,
    pub valset: Option<Vec<Example>>,
    /// reflection_lm for this run; `None` uses the server's own default.
    pub reflection_lm: Option<String>,
    pub max_metric_calls: i32,
    pub evaluate: E,
    pub make_reflective_dataset: M,
    pub on_progress: Option<Box<dyn Fn(ProgressUpdate) + Send>>,
}

// ------------------------------------------------------------------ Omni types

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct OmniBestEval {
    pub score: f64,
    /// JSON-encoded side-info string (or empty).
    pub side_info: String,
}

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct OmniOptState {
    /// Top-K best prior evaluations for this example, sorted descending by score.
    pub best_example_evals: Vec<OmniBestEval>,
}

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct OmniEvalRequest {
    pub request_id: String,
    /// String candidate (prompt, code, instructions, etc.)
    pub candidate: String,
    pub batch: Vec<Example>,
    /// Per-example warm-start history, aligned 1:1 with `batch`. Empty when
    /// the server didn't send opt_states.
    pub opt_states: Vec<OmniOptState>,
}

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct OmniEvalResult {
    pub scores: Vec<f64>,
    /// Optional per-example side-info, one JSON-encoded string per example
    /// (or an empty string for none). Omitted entries are treated as `{}`
    /// server-side.
    pub side_infos: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct OmniProgressUpdate {
    pub evals_used: i32,
    pub max_evals: i32,
    pub best_score: f64,
    pub best_candidate: String,
}

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct OmniOptimizeResult {
    pub run_id: String,
    pub best_candidate: String,
    pub best_score: f64,
    pub total_evals: i32,
}

pub struct OmniOptimizeOpts<E> {
    pub run_id: String,
    /// Initial candidate string to optimize from. `None` for seedless mode.
    pub seed_candidate: Option<String>,
    pub dataset: Option<Vec<Example>>,
    pub valset: Option<Vec<Example>>,
    pub objective: Option<String>,
    /// Long-form task context (problem statement, evaluation rules) passed
    /// straight through to optimize_anything's own `background` parameter.
    pub background: Option<String>,
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

// ------------------------------------------------- constructors for callers

impl Example {
    /// `#[non_exhaustive]` means struct-literal syntax only works inside this
    /// crate, so callers assembling a trainset/dataset need this.
    pub fn new(id: impl Into<String>, fields: HashMap<String, String>) -> Self {
        Self { id: id.into(), fields }
    }
}

impl Trajectory {
    pub fn new(
        input_fields: HashMap<String, String>,
        output: impl Into<String>,
        feedback: impl Into<String>,
    ) -> Self {
        Self { input_fields, output: output.into(), feedback: feedback.into() }
    }
}

impl EvalResult {
    pub fn new(outputs: Vec<String>, scores: Vec<f64>, trajectories: Option<Vec<Trajectory>>) -> Self {
        Self { outputs, scores, trajectories }
    }
}

impl ReflectiveEntry {
    pub fn new(
        inputs: HashMap<String, String>,
        generated_output: impl Into<String>,
        feedback: impl Into<String>,
    ) -> Self {
        Self { inputs, generated_output: generated_output.into(), feedback: feedback.into() }
    }
}

impl OmniEvalResult {
    pub fn new(scores: Vec<f64>, side_infos: Option<Vec<String>>) -> Self {
        Self { scores, side_infos }
    }
}
