# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

# Generic type aliases matching your original
RolloutOutput = TypeVar("RolloutOutput")
Trajectory = TypeVar("Trajectory")
DataInst = TypeVar("DataInst")

@dataclass
class EvaluationBatch(Generic[Trajectory, RolloutOutput]):
    """
    Container for the result of evaluating a proposed candidate on a batch of data.

    - outputs: raw per-example outputs from upon executing the candidate. GEPA does not interpret these;
      they are forwarded to other parts of the user's code or logging as-is.
    - scores: per-example numeric scores (floats). GEPA sums these for minibatch acceptance
      and averages them over the full validation set for tracking/pareto fronts.
    - trajectories: optional per-example traces used by make_reflective_dataset to build
      a reflective dataset (See `GEPAAdapter.make_reflective_dataset`). If capture_traces=True is passed to `evaluate`, trajectories
      should be provided and align one-to-one with `outputs` and `scores`.
    """
    outputs: list[RolloutOutput]
    scores: list[float]
    trajectories: list[Trajectory] | None = None

class ProposalFn(Protocol):
    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        - Given the current `candidate`, a reflective dataset (as returned by
          `GEPAAdapter.make_reflective_dataset`), and a list of component names to update,
          return a mapping component_name -> new component text (str). This allows the user
          to implement their own instruction proposal logic. For example, the user can use
          a different LLM, implement DSPy signatures, etc. Another example can be situations
          where 2 or more components need to be updated together (coupled updates).

        Returns
        - Dict[str, str] mapping component names to newly proposed component texts.
        """
        ...

class GEPAAdapter(Protocol[DataInst, Trajectory, RolloutOutput]):
    """
    GEPAAdapter is the single integration point between your system
    and the GEPA optimization engine. Implementers provide three responsibilities:

    The following are user-defined types that are not interpreted by GEPA but are used by the user's code
        to define the adapter:
    DataInst: User-defined type of input data to the program under optimization.
    Trajectory: User-defined type of trajectory data, which typically captures the
        different steps of the program candidate execution.
    RolloutOutput: User-defined type of output data from the program candidate.

    The following are the responsibilities of the adapter:
    1) Program construction and evaluation (evaluate):
       Given a batch of DataInst and a "candidate" program (mapping from named components
       -> component text), execute the program to produce per-example scores and
       optionally rich trajectories (capturing intermediate states) needed for reflection.

    2) Reflective dataset construction (make_reflective_dataset):
       Given the candidate, EvaluationBatch (trajectories, outputs, scores), and the list of components to update,
       produce a small JSON-serializable dataset for each component that you want to update. This
       dataset is fed to the teacher LM to propose improved component text.

    3) Optional instruction proposal (propose_new_texts):
       GEPA provides a default implementation (instruction_proposal.py) that serializes the reflective dataset
       to propose new component texts. However, users can implement their own proposal logic by implementing this method.
       This method receives the current candidate, the reflective dataset, and the list of components to update,
       and returns a mapping from component name to new component text.

    Key concepts and contracts:
    - candidate: Dict[str, str] mapping a named component of the system to its corresponding text.
    - scores: higher is better. GEPA uses:
      - minibatch: sum(scores) to compare old vs. new candidate (acceptance test),
      - full valset: mean(scores) for tracking and Pareto-front selection.
      Ensure your metric is calibrated accordingly or normalized to a consistent scale.
    - trajectories: opaque to GEPA (the engine never inspects them). They must be
      consumable by your own make_reflective_dataset implementation to extract the
      minimal context needed to produce meaningful feedback for every component of
      the system under optimization.
    - error handling: Never raise for individual example failures. Instead:
      - Return a valid `EvaluationBatch` with per-example failure scores (e.g., 0.0)
        when formatting/parsing fails. Even better if the trajectories are also populated
        with the failed example, including the error message, identifying the reason for the failure.
      - Reserve exceptions for unrecoverable, systemic failures (e.g., missing model,
        misconfigured program, schema mismatch).
      - If an exception is raised, the engine will log the error and proceed to the next iteration.
    """

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        """
        Run the program defined by `candidate` on a batch of data.

        Parameters
        - batch: list of task-specific inputs (DataInst).
        - candidate: mapping from component name -> component text. You must instantiate
          your full system with the component text for each component, and execute it on the batch.
        - capture_traces: when True, you must populate `EvaluationBatch.trajectories`
          with a per-example trajectory object that your `make_reflective_dataset` can
          later consume. When False, you may set trajectories=None to save time/memory.
          capture_traces=True is used by the reflective mutation proposer to build a reflective dataset.

        Returns
        - EvaluationBatch with:
          - outputs: raw per-example outputs (opaque to GEPA).
          - scores: per-example floats, length == len(batch). Higher is better.
          - trajectories:
              - if capture_traces=True: list[Trajectory] with length == len(batch).
              - if capture_traces=False: None.

        Scoring semantics
        - The engine uses sum(scores) on minibatches to decide whether to accept a
          candidate mutation and average(scores) over the full valset for tracking.
        - Prefer to return per-example scores, that can be aggregated via summation.
        - If an example fails (e.g., parse error), use a fallback score (e.g., 0.0).

        Correctness constraints
        - len(outputs) == len(scores) == len(batch)
        - If capture_traces=True: trajectories must be provided and len(trajectories) == len(batch)
        - Do not mutate `batch` or `candidate` in-place. Construct a fresh program
          instance or deep-copy as needed.
        """
        ...

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Build a small, JSON-serializable dataset (per component) to drive instruction
        refinement by a teacher LLM.

        Parameters
        - candidate: the same candidate evaluated in evaluate().
        - eval_batch: The result of evaluate(..., capture_traces=True) on
          the same batch. You should extract everything you need from eval_batch.trajectories
          (and optionally outputs/scores) to assemble concise, high-signal examples.
        - components_to_update: subset of component names for which the proposer has
          requested updates. At a time, GEPA identifies a subset of components to update.

        Returns
        - A dict: component_name -> list of dict records (the "reflective dataset").
          Each record should be JSON-serializable and is passed verbatim to the
          instruction proposal prompt. A recommended schema is:
            {
              "Inputs": Dict[str, str],             # Minimal, clean view of the inputs to the component
              "Generated Outputs": Dict[str, str] | str,  # Model outputs or raw text
              "Feedback": str                       # Feedback on the component's performance, including correct answer, error messages, etc.
            }
          You may include additional keys (e.g., "score", "rationale", "trace_id") if useful.

        Determinism
        - If you subsample trace instances, use a seeded RNG to keep runs reproducible.
        """
        ...

    propose_new_texts: ProposalFn | None = None
