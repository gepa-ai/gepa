use thiserror::Error;

// Deliberately NOT #[non_exhaustive]: callers *construct* this (e.g. to
// return their own evaluate() failure as a GEPAError), not merely match on
// it. That differs from the public structs in types.rs, which callers only
// ever receive and which therefore are #[non_exhaustive].
#[derive(Debug, Error)]
pub enum GEPAError {
    #[error("invalid server address: {0}")]
    InvalidAddress(String),
    #[error("gRPC transport error: {0}")]
    Transport(String),
    #[error("gRPC status: {0}")]
    Status(String),
    #[error("stream closed unexpectedly")]
    StreamClosed,
    /// The server reported a failure that was NOT the evaluator callback's
    /// fault -- an optimizer bug, or a transport/protocol issue surfaced
    /// through the stream. See `EvaluatorError` for the other case.
    #[error("optimization failed: {0}")]
    OptimizationFailed(String),
    /// The server reported that the evaluator or reflective-dataset callback
    /// raised, or returned something malformed (e.g. a mismatched score
    /// count). Distinct from `OptimizationFailed` so callers can tell their
    /// own bug from a server-side one without parsing the message string.
    #[error("evaluator callback error: {0}")]
    EvaluatorError(String),
    #[error("channel send error")]
    ChannelSend,
}

// Hand-written rather than #[from] on the variants, so tonic's types never
// appear in this crate's public error API: a caller matching on GEPAError
// needs no tonic dependency of its own, and this crate can change its tonic
// version without that being an observable breaking change to GEPAError.
impl From<tonic::transport::Error> for GEPAError {
    fn from(e: tonic::transport::Error) -> Self {
        GEPAError::Transport(e.to_string())
    }
}

impl From<tonic::Status> for GEPAError {
    fn from(e: tonic::Status) -> Self {
        GEPAError::Status(e.to_string())
    }
}
