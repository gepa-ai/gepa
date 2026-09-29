use thiserror::Error;

#[derive(Debug, Error)]
pub enum GEPAError {
    #[error("invalid server address: {0}")]
    InvalidAddress(String),
    #[error("gRPC transport error: {0}")]
    Transport(#[from] tonic::transport::Error),
    #[error("gRPC status: {0}")]
    Status(#[from] tonic::Status),
    #[error("stream closed unexpectedly")]
    StreamClosed,
    #[error("optimization failed: {0}")]
    OptimizationFailed(String),
    #[error("channel send error")]
    ChannelSend,
}
