from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class OmniClientMessage(_message.Message):
    __slots__ = ("start_request", "evaluate_batch_response")
    START_REQUEST_FIELD_NUMBER: _ClassVar[int]
    EVALUATE_BATCH_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    start_request: OmniStartRequest
    evaluate_batch_response: OmniEvaluateBatchResponse
    def __init__(self, start_request: _Optional[_Union[OmniStartRequest, _Mapping]] = ..., evaluate_batch_response: _Optional[_Union[OmniEvaluateBatchResponse, _Mapping]] = ...) -> None: ...

class OmniServerMessage(_message.Message):
    __slots__ = ("evaluate_batch_request", "progress_update", "optimization_complete", "optimization_error")
    EVALUATE_BATCH_REQUEST_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_UPDATE_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZATION_COMPLETE_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZATION_ERROR_FIELD_NUMBER: _ClassVar[int]
    evaluate_batch_request: OmniEvaluateBatchRequest
    progress_update: OmniProgressUpdate
    optimization_complete: OmniOptimizationComplete
    optimization_error: OptimizationError
    def __init__(self, evaluate_batch_request: _Optional[_Union[OmniEvaluateBatchRequest, _Mapping]] = ..., progress_update: _Optional[_Union[OmniProgressUpdate, _Mapping]] = ..., optimization_complete: _Optional[_Union[OmniOptimizationComplete, _Mapping]] = ..., optimization_error: _Optional[_Union[OptimizationError, _Mapping]] = ...) -> None: ...

class ClientMessage(_message.Message):
    __slots__ = ("start_request", "evaluate_batch_response", "reflective_dataset_response")
    START_REQUEST_FIELD_NUMBER: _ClassVar[int]
    EVALUATE_BATCH_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    REFLECTIVE_DATASET_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    start_request: StartRequest
    evaluate_batch_response: EvaluateBatchResponse
    reflective_dataset_response: ReflectiveDatasetResponse
    def __init__(self, start_request: _Optional[_Union[StartRequest, _Mapping]] = ..., evaluate_batch_response: _Optional[_Union[EvaluateBatchResponse, _Mapping]] = ..., reflective_dataset_response: _Optional[_Union[ReflectiveDatasetResponse, _Mapping]] = ...) -> None: ...

class ServerMessage(_message.Message):
    __slots__ = ("evaluate_batch_request", "reflective_dataset_request", "progress_update", "optimization_complete", "optimization_error")
    EVALUATE_BATCH_REQUEST_FIELD_NUMBER: _ClassVar[int]
    REFLECTIVE_DATASET_REQUEST_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_UPDATE_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZATION_COMPLETE_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZATION_ERROR_FIELD_NUMBER: _ClassVar[int]
    evaluate_batch_request: EvaluateBatchRequest
    reflective_dataset_request: ReflectiveDatasetRequest
    progress_update: ProgressUpdate
    optimization_complete: OptimizationComplete
    optimization_error: OptimizationError
    def __init__(self, evaluate_batch_request: _Optional[_Union[EvaluateBatchRequest, _Mapping]] = ..., reflective_dataset_request: _Optional[_Union[ReflectiveDatasetRequest, _Mapping]] = ..., progress_update: _Optional[_Union[ProgressUpdate, _Mapping]] = ..., optimization_complete: _Optional[_Union[OptimizationComplete, _Mapping]] = ..., optimization_error: _Optional[_Union[OptimizationError, _Mapping]] = ...) -> None: ...

class OmniStartRequest(_message.Message):
    __slots__ = ("run_id", "seed_candidate", "objective", "dataset", "valset", "max_evals", "reflection_lm", "engine")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SEED_CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    OBJECTIVE_FIELD_NUMBER: _ClassVar[int]
    DATASET_FIELD_NUMBER: _ClassVar[int]
    VALSET_FIELD_NUMBER: _ClassVar[int]
    MAX_EVALS_FIELD_NUMBER: _ClassVar[int]
    REFLECTION_LM_FIELD_NUMBER: _ClassVar[int]
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    seed_candidate: str
    objective: str
    dataset: _containers.RepeatedCompositeFieldContainer[Example]
    valset: _containers.RepeatedCompositeFieldContainer[Example]
    max_evals: int
    reflection_lm: str
    engine: str
    def __init__(self, run_id: _Optional[str] = ..., seed_candidate: _Optional[str] = ..., objective: _Optional[str] = ..., dataset: _Optional[_Iterable[_Union[Example, _Mapping]]] = ..., valset: _Optional[_Iterable[_Union[Example, _Mapping]]] = ..., max_evals: _Optional[int] = ..., reflection_lm: _Optional[str] = ..., engine: _Optional[str] = ...) -> None: ...

class StartRequest(_message.Message):
    __slots__ = ("run_id", "seed_candidate", "trainset", "valset", "reflection_lm", "max_metric_calls")
    class SeedCandidateEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SEED_CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    TRAINSET_FIELD_NUMBER: _ClassVar[int]
    VALSET_FIELD_NUMBER: _ClassVar[int]
    REFLECTION_LM_FIELD_NUMBER: _ClassVar[int]
    MAX_METRIC_CALLS_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    seed_candidate: _containers.ScalarMap[str, str]
    trainset: _containers.RepeatedCompositeFieldContainer[Example]
    valset: _containers.RepeatedCompositeFieldContainer[Example]
    reflection_lm: str
    max_metric_calls: int
    def __init__(self, run_id: _Optional[str] = ..., seed_candidate: _Optional[_Mapping[str, str]] = ..., trainset: _Optional[_Iterable[_Union[Example, _Mapping]]] = ..., valset: _Optional[_Iterable[_Union[Example, _Mapping]]] = ..., reflection_lm: _Optional[str] = ..., max_metric_calls: _Optional[int] = ...) -> None: ...

class OmniEvaluateBatchRequest(_message.Message):
    __slots__ = ("request_id", "candidate", "batch", "opt_states")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    BATCH_FIELD_NUMBER: _ClassVar[int]
    OPT_STATES_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    candidate: str
    batch: _containers.RepeatedCompositeFieldContainer[Example]
    opt_states: _containers.RepeatedCompositeFieldContainer[OmniOptimizationState]
    def __init__(self, request_id: _Optional[str] = ..., candidate: _Optional[str] = ..., batch: _Optional[_Iterable[_Union[Example, _Mapping]]] = ..., opt_states: _Optional[_Iterable[_Union[OmniOptimizationState, _Mapping]]] = ...) -> None: ...

class OmniOptimizationState(_message.Message):
    __slots__ = ("best_example_evals",)
    BEST_EXAMPLE_EVALS_FIELD_NUMBER: _ClassVar[int]
    best_example_evals: _containers.RepeatedCompositeFieldContainer[OmniBestEval]
    def __init__(self, best_example_evals: _Optional[_Iterable[_Union[OmniBestEval, _Mapping]]] = ...) -> None: ...

class OmniBestEval(_message.Message):
    __slots__ = ("score", "side_info")
    SCORE_FIELD_NUMBER: _ClassVar[int]
    SIDE_INFO_FIELD_NUMBER: _ClassVar[int]
    score: float
    side_info: str
    def __init__(self, score: _Optional[float] = ..., side_info: _Optional[str] = ...) -> None: ...

class OmniEvaluateBatchResponse(_message.Message):
    __slots__ = ("request_id", "scores", "side_infos")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    SIDE_INFOS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    scores: _containers.RepeatedScalarFieldContainer[float]
    side_infos: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, request_id: _Optional[str] = ..., scores: _Optional[_Iterable[float]] = ..., side_infos: _Optional[_Iterable[str]] = ...) -> None: ...

class EvaluateBatchRequest(_message.Message):
    __slots__ = ("request_id", "candidate", "batch", "capture_traces")
    class CandidateEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    BATCH_FIELD_NUMBER: _ClassVar[int]
    CAPTURE_TRACES_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    candidate: _containers.ScalarMap[str, str]
    batch: _containers.RepeatedCompositeFieldContainer[Example]
    capture_traces: bool
    def __init__(self, request_id: _Optional[str] = ..., candidate: _Optional[_Mapping[str, str]] = ..., batch: _Optional[_Iterable[_Union[Example, _Mapping]]] = ..., capture_traces: _Optional[bool] = ...) -> None: ...

class EvaluateBatchResponse(_message.Message):
    __slots__ = ("request_id", "outputs", "scores", "trajectories")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    TRAJECTORIES_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    outputs: _containers.RepeatedScalarFieldContainer[str]
    scores: _containers.RepeatedScalarFieldContainer[float]
    trajectories: _containers.RepeatedCompositeFieldContainer[Trajectory]
    def __init__(self, request_id: _Optional[str] = ..., outputs: _Optional[_Iterable[str]] = ..., scores: _Optional[_Iterable[float]] = ..., trajectories: _Optional[_Iterable[_Union[Trajectory, _Mapping]]] = ...) -> None: ...

class ReflectiveDatasetRequest(_message.Message):
    __slots__ = ("request_id", "candidate", "components_to_update", "trajectories")
    class CandidateEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    COMPONENTS_TO_UPDATE_FIELD_NUMBER: _ClassVar[int]
    TRAJECTORIES_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    candidate: _containers.ScalarMap[str, str]
    components_to_update: _containers.RepeatedScalarFieldContainer[str]
    trajectories: _containers.RepeatedCompositeFieldContainer[Trajectory]
    def __init__(self, request_id: _Optional[str] = ..., candidate: _Optional[_Mapping[str, str]] = ..., components_to_update: _Optional[_Iterable[str]] = ..., trajectories: _Optional[_Iterable[_Union[Trajectory, _Mapping]]] = ...) -> None: ...

class ReflectiveDatasetResponse(_message.Message):
    __slots__ = ("request_id", "reflective_data")
    class ReflectiveDataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ReflectiveComponentData
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ReflectiveComponentData, _Mapping]] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    REFLECTIVE_DATA_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    reflective_data: _containers.MessageMap[str, ReflectiveComponentData]
    def __init__(self, request_id: _Optional[str] = ..., reflective_data: _Optional[_Mapping[str, ReflectiveComponentData]] = ...) -> None: ...

class OmniProgressUpdate(_message.Message):
    __slots__ = ("evals_used", "max_evals", "best_score", "best_candidate")
    EVALS_USED_FIELD_NUMBER: _ClassVar[int]
    MAX_EVALS_FIELD_NUMBER: _ClassVar[int]
    BEST_SCORE_FIELD_NUMBER: _ClassVar[int]
    BEST_CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    evals_used: int
    max_evals: int
    best_score: float
    best_candidate: str
    def __init__(self, evals_used: _Optional[int] = ..., max_evals: _Optional[int] = ..., best_score: _Optional[float] = ..., best_candidate: _Optional[str] = ...) -> None: ...

class OmniOptimizationComplete(_message.Message):
    __slots__ = ("run_id", "best_candidate", "best_score", "total_evals")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    BEST_CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    BEST_SCORE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_EVALS_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    best_candidate: str
    best_score: float
    total_evals: int
    def __init__(self, run_id: _Optional[str] = ..., best_candidate: _Optional[str] = ..., best_score: _Optional[float] = ..., total_evals: _Optional[int] = ...) -> None: ...

class ProgressUpdate(_message.Message):
    __slots__ = ("metric_calls_used", "max_metric_calls", "best_score", "best_candidate")
    class BestCandidateEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    METRIC_CALLS_USED_FIELD_NUMBER: _ClassVar[int]
    MAX_METRIC_CALLS_FIELD_NUMBER: _ClassVar[int]
    BEST_SCORE_FIELD_NUMBER: _ClassVar[int]
    BEST_CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    metric_calls_used: int
    max_metric_calls: int
    best_score: float
    best_candidate: _containers.ScalarMap[str, str]
    def __init__(self, metric_calls_used: _Optional[int] = ..., max_metric_calls: _Optional[int] = ..., best_score: _Optional[float] = ..., best_candidate: _Optional[_Mapping[str, str]] = ...) -> None: ...

class OptimizationComplete(_message.Message):
    __slots__ = ("run_id", "best_candidate", "best_score")
    class BestCandidateEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    BEST_CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    BEST_SCORE_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    best_candidate: _containers.ScalarMap[str, str]
    best_score: float
    def __init__(self, run_id: _Optional[str] = ..., best_candidate: _Optional[_Mapping[str, str]] = ..., best_score: _Optional[float] = ...) -> None: ...

class OptimizationError(_message.Message):
    __slots__ = ("run_id", "message")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    message: str
    def __init__(self, run_id: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class StatusRequest(_message.Message):
    __slots__ = ("run_id",)
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    def __init__(self, run_id: _Optional[str] = ...) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("run_id", "status", "message", "metric_calls_used")
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNKNOWN: _ClassVar[StatusResponse.Status]
        RUNNING: _ClassVar[StatusResponse.Status]
        COMPLETE: _ClassVar[StatusResponse.Status]
        FAILED: _ClassVar[StatusResponse.Status]
    UNKNOWN: StatusResponse.Status
    RUNNING: StatusResponse.Status
    COMPLETE: StatusResponse.Status
    FAILED: StatusResponse.Status
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    METRIC_CALLS_USED_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    status: StatusResponse.Status
    message: str
    metric_calls_used: int
    def __init__(self, run_id: _Optional[str] = ..., status: _Optional[_Union[StatusResponse.Status, str]] = ..., message: _Optional[str] = ..., metric_calls_used: _Optional[int] = ...) -> None: ...

class Example(_message.Message):
    __slots__ = ("id", "fields")
    class FieldsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    id: str
    fields: _containers.ScalarMap[str, str]
    def __init__(self, id: _Optional[str] = ..., fields: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Trajectory(_message.Message):
    __slots__ = ("input_fields", "output", "feedback")
    class InputFieldsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    INPUT_FIELDS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    FEEDBACK_FIELD_NUMBER: _ClassVar[int]
    input_fields: _containers.ScalarMap[str, str]
    output: str
    feedback: str
    def __init__(self, input_fields: _Optional[_Mapping[str, str]] = ..., output: _Optional[str] = ..., feedback: _Optional[str] = ...) -> None: ...

class ReflectiveComponentData(_message.Message):
    __slots__ = ("entries",)
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[ReflectiveEntry]
    def __init__(self, entries: _Optional[_Iterable[_Union[ReflectiveEntry, _Mapping]]] = ...) -> None: ...

class ReflectiveEntry(_message.Message):
    __slots__ = ("inputs", "generated_output", "feedback")
    class InputsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    GENERATED_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    FEEDBACK_FIELD_NUMBER: _ClassVar[int]
    inputs: _containers.ScalarMap[str, str]
    generated_output: str
    feedback: str
    def __init__(self, inputs: _Optional[_Mapping[str, str]] = ..., generated_output: _Optional[str] = ..., feedback: _Optional[str] = ...) -> None: ...
