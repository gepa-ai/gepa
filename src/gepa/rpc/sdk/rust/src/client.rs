use std::future::Future;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Channel, ClientTlsConfig};
use tonic::Request;

use crate::error::GEPAError;
use crate::generated::{
    client_message::Payload as ClientPayload, gepa_service_client::GepaServiceClient,
    omni_client_message::Payload as OmniClientPayload,
    omni_server_message::Payload as OmniServerPayload,
    optimization_error::ErrorKind, server_message::Payload as ServerPayload, ClientMessage,
    EvaluateBatchResponse,
    Example as ProtoExample, OmniBestEval as ProtoOmniBestEval, OmniClientMessage,
    OmniEvaluateBatchResponse, OmniOptimizationState as ProtoOmniOptState, OmniStartRequest,
    OptimizationError as ProtoOptimizationError, ReflectiveComponentData,
    ReflectiveDatasetResponse, ReflectiveEntry as ProtoEntry, StartRequest,
    Trajectory as ProtoTrajectory,
};
use crate::types::{
    Candidate, EvalRequest, EvalResult, Example, OmniBestEval, OmniEvalRequest, OmniEvalResult,
    OmniOptState, OmniOptimizeOpts, OmniOptimizeResult, OmniProgressUpdate, OptimizeOpts,
    OptimizeResult, ProgressUpdate, ReflectiveRequest, ReflectiveResult,
    Trajectory,
};

/// Maps a server-reported OptimizationError onto the right GEPAError variant,
/// so "your callback was wrong" is distinguishable from "the server failed"
/// without parsing the message text.
fn error_from_optimization_error(e: ProtoOptimizationError) -> GEPAError {
    match ErrorKind::try_from(e.error_kind) {
        Ok(ErrorKind::EvaluatorError) => GEPAError::EvaluatorError(e.message),
        _ => GEPAError::OptimizationFailed(e.message),
    }
}

pub struct Client {
    // One Channel, built once and reused by every call. tonic's Channel is
    // cheap to clone (an internal handle, not a new socket), and
    // connect_lazy() means constructing it neither blocks nor requires the
    // server to be up yet -- the first real RPC establishes the connection.
    // Previously optimize()/optimize_omni() each ran
    // Channel::from_shared(..).connect().await on *every* invocation,
    // reconnecting from scratch each time.
    channel: Channel,
}

impl Client {
    /// Connects over plaintext HTTP/2. For a server started with
    /// --tls-cert/--tls-key, use [`Client::new_with_tls`].
    pub fn new(target: impl Into<String>) -> Result<Self, GEPAError> {
        let target = target.into();
        let channel = Channel::from_shared(format!("http://{target}"))
            .map_err(|e| GEPAError::InvalidAddress(e.to_string()))?
            .connect_lazy();
        Ok(Self { channel })
    }

    /// Connects over TLS. `tls_config` carries any custom CA (e.g. for a
    /// self-signed server certificate) via
    /// `ClientTlsConfig::new().ca_certificate(..)`; pass
    /// `ClientTlsConfig::new()` to trust the platform's default roots.
    pub fn new_with_tls(
        target: impl Into<String>,
        tls_config: ClientTlsConfig,
    ) -> Result<Self, GEPAError> {
        let target = target.into();
        let channel = Channel::from_shared(format!("https://{target}"))
            .map_err(|e| GEPAError::InvalidAddress(e.to_string()))?
            .tls_config(tls_config)
            .map_err(|e| GEPAError::InvalidAddress(e.to_string()))?
            .connect_lazy();
        Ok(Self { channel })
    }

    pub async fn optimize<E, M, EFut, MFut>(
        &self,
        mut opts: OptimizeOpts<E, M>,
    ) -> Result<OptimizeResult, GEPAError>
    where
        E: FnMut(EvalRequest) -> EFut,
        EFut: Future<Output = Result<EvalResult, GEPAError>> + Send,
        M: FnMut(ReflectiveRequest) -> MFut,
        MFut: Future<Output = Result<ReflectiveResult, GEPAError>> + Send,
    {
        let mut grpc_client = GepaServiceClient::new(self.channel.clone());

        let (tx, rx) = mpsc::channel::<ClientMessage>(32);
        let stream = ReceiverStream::new(rx);

        tx.send(ClientMessage {
            payload: Some(ClientPayload::StartRequest(StartRequest {
                run_id: opts.run_id.clone(),
                seed_candidate: opts.seed_candidate.clone(),
                trainset: opts.trainset.iter().map(example_to_proto).collect(),
                valset: opts
                    .valset
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .map(example_to_proto)
                    .collect(),
                reflection_lm: opts.reflection_lm.clone().unwrap_or_default(),
                max_metric_calls: opts.max_metric_calls,
            })),
        })
        .await
        .map_err(|_| GEPAError::ChannelSend)?;

        let mut response = grpc_client
            .run_optimization(Request::new(stream))
            .await?
            .into_inner();

        loop {
            let msg = response.message().await?;
            match msg.and_then(|m| m.payload) {
                None => return Err(GEPAError::StreamClosed),
                Some(ServerPayload::EvaluateBatchRequest(req)) => {
                    let args = EvalRequest {
                        request_id: req.request_id.clone(),
                        candidate: Candidate::from_map(req.candidate.clone()),
                        batch: req.batch.iter().map(proto_to_example).collect(),
                        capture_traces: req.capture_traces,
                    };
                    let result = (opts.evaluate)(args).await?;
                    tx.send(ClientMessage {
                        payload: Some(ClientPayload::EvaluateBatchResponse(
                            EvaluateBatchResponse {
                                request_id: req.request_id,
                                outputs: result.outputs,
                                scores: result.scores,
                                trajectories: result
                                    .trajectories
                                    .unwrap_or_default()
                                    .into_iter()
                                    .map(traj_to_proto)
                                    .collect(),
                            },
                        )),
                    })
                    .await
                    .map_err(|_| GEPAError::ChannelSend)?;
                }
                Some(ServerPayload::ReflectiveDatasetRequest(req)) => {
                    let args = ReflectiveRequest {
                        request_id: req.request_id.clone(),
                        candidate: Candidate::from_map(req.candidate.clone()),
                        components_to_update: req.components_to_update.clone(),
                        trajectories: req.trajectories.iter().map(proto_to_traj).collect(),
                    };
                    let result = (opts.make_reflective_dataset)(args).await?;
                    let reflective_data = result
                        .into_iter()
                        .map(|(comp, entries)| {
                            let proto_entries = entries
                                .into_iter()
                                .map(|e| ProtoEntry {
                                    inputs: e.inputs,
                                    generated_output: e.generated_output,
                                    feedback: e.feedback,
                                })
                                .collect();
                            (comp, ReflectiveComponentData { entries: proto_entries })
                        })
                        .collect();
                    tx.send(ClientMessage {
                        payload: Some(ClientPayload::ReflectiveDatasetResponse(
                            ReflectiveDatasetResponse {
                                request_id: req.request_id,
                                reflective_data,
                            },
                        )),
                    })
                    .await
                    .map_err(|_| GEPAError::ChannelSend)?;
                }
                Some(ServerPayload::ProgressUpdate(u)) => {
                    if let Some(ref cb) = opts.on_progress {
                        cb(ProgressUpdate {
                            metric_calls_used: u.metric_calls_used,
                            max_metric_calls: u.max_metric_calls,
                            best_score: u.best_score,
                            best_candidate: Candidate::from_map(u.best_candidate),
                        });
                    }
                }
                Some(ServerPayload::OptimizationComplete(c)) => {
                    return Ok(OptimizeResult {
                        run_id: c.run_id,
                        best_candidate: Candidate::from_map(c.best_candidate),
                        best_score: c.best_score,
                    });
                }
                Some(ServerPayload::OptimizationError(e)) => {
                    return Err(error_from_optimization_error(e));
                }
            }
        }
    }

    pub async fn optimize_omni<E, EFut>(
        &self,
        mut opts: OmniOptimizeOpts<E>,
    ) -> Result<OmniOptimizeResult, GEPAError>
    where
        E: FnMut(OmniEvalRequest) -> EFut,
        EFut: Future<Output = Result<OmniEvalResult, GEPAError>> + Send,
    {
        let mut grpc_client = GepaServiceClient::new(self.channel.clone());

        let (tx, rx) = mpsc::channel::<OmniClientMessage>(32);
        let stream = ReceiverStream::new(rx);

        tx.send(OmniClientMessage {
            payload: Some(OmniClientPayload::StartRequest(OmniStartRequest {
                run_id: opts.run_id.clone(),
                seed_candidate: opts.seed_candidate.clone().unwrap_or_default(),
                objective: opts.objective.clone().unwrap_or_default(),
                background: opts.background.clone().unwrap_or_default(),
                dataset: opts
                    .dataset
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .map(example_to_proto)
                    .collect(),
                valset: opts
                    .valset
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .map(example_to_proto)
                    .collect(),
                max_evals: opts.max_evals,
                reflection_lm: opts.reflection_lm.clone().unwrap_or_default(),
                engine: opts.engine.clone().unwrap_or_default(),
            })),
        })
        .await
        .map_err(|_| GEPAError::ChannelSend)?;

        let mut response = grpc_client
            .run_optimization_omni(Request::new(stream))
            .await?
            .into_inner();

        loop {
            let msg = response.message().await?;
            match msg.and_then(|m| m.payload) {
                None => return Err(GEPAError::StreamClosed),
                Some(OmniServerPayload::EvaluateBatchRequest(req)) => {
                    let args = OmniEvalRequest {
                        request_id: req.request_id.clone(),
                        candidate: req.candidate.clone(),
                        batch: req.batch.iter().map(proto_to_example).collect(),
                        opt_states: req.opt_states.iter().map(proto_to_opt_state).collect(),
                    };
                    let result = (opts.evaluate)(args).await?;
                    tx.send(OmniClientMessage {
                        payload: Some(OmniClientPayload::EvaluateBatchResponse(
                            OmniEvaluateBatchResponse {
                                request_id: req.request_id,
                                scores: result.scores,
                                side_infos: result.side_infos.unwrap_or_default(),
                            },
                        )),
                    })
                    .await
                    .map_err(|_| GEPAError::ChannelSend)?;
                }
                Some(OmniServerPayload::ProgressUpdate(u)) => {
                    if let Some(ref cb) = opts.on_progress {
                        cb(OmniProgressUpdate {
                            evals_used: u.evals_used,
                            max_evals: u.max_evals,
                            best_score: u.best_score,
                            best_candidate: u.best_candidate,
                        });
                    }
                }
                Some(OmniServerPayload::OptimizationComplete(c)) => {
                    return Ok(OmniOptimizeResult {
                        run_id: c.run_id,
                        best_candidate: c.best_candidate,
                        best_score: c.best_score,
                        total_evals: c.total_evals,
                    });
                }
                Some(OmniServerPayload::OptimizationError(e)) => {
                    return Err(error_from_optimization_error(e));
                }
            }
        }
    }
}

fn example_to_proto(e: &Example) -> ProtoExample {
    ProtoExample {
        id: e.id.clone(),
        fields: e.fields.clone(),
    }
}

fn proto_to_example(e: &ProtoExample) -> Example {
    Example {
        id: e.id.clone(),
        fields: e.fields.clone(),
    }
}

fn proto_to_best_eval(e: &ProtoOmniBestEval) -> OmniBestEval {
    OmniBestEval {
        score: e.score,
        side_info: e.side_info.clone(),
    }
}

fn proto_to_opt_state(s: &ProtoOmniOptState) -> OmniOptState {
    OmniOptState {
        best_example_evals: s.best_example_evals.iter().map(proto_to_best_eval).collect(),
    }
}

fn traj_to_proto(t: Trajectory) -> ProtoTrajectory {
    ProtoTrajectory {
        input_fields: t.input_fields,
        output: t.output,
        feedback: t.feedback,
    }
}

fn proto_to_traj(t: &ProtoTrajectory) -> Trajectory {
    Trajectory {
        input_fields: t.input_fields.clone(),
        output: t.output.clone(),
        feedback: t.feedback.clone(),
    }
}
