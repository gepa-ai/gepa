mod generated {
    tonic::include_proto!("gepa_rpc");
}

mod client;
mod error;
mod types;

pub use client::Client;
pub use error::GEPAError;
pub use types::{
    Candidate, EvalRequest, EvalResult, Example, OmniBestEval, OmniEvalRequest, OmniEvalResult,
    OmniOptState, OmniOptimizeOpts, OmniOptimizeResult, OmniProgressUpdate, OptimizeOpts,
    OptimizeResult, ProgressUpdate, ReflectiveEntry, ReflectiveRequest, ReflectiveResult,
    Trajectory,
};
