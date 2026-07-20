"""
This file provides an example adapter allowing GEPA to optimize text components of DSPy programs (instructions and prompts).
The most up-to-date version of this file is in the DSPy repository: https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/gepa/gepa_utils.py
"""

import json
import logging
import random
from typing import Any, Callable, Protocol, TypedDict

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types import History
from dspy.adapters.types.base_type import Type
from dspy.adapters.types.tool import Tool
from dspy.evaluate import Evaluate
from dspy.primitives import Example, Module, Prediction
from dspy.teleprompt.bootstrap_trace import FailedPrediction, TraceData

from gepa import EvaluationBatch, GEPAAdapter
from gepa.core.adapter import ProposalFn
from gepa.proposer.reflective_mutation.reflection_lm import (
    build_textual_gradient_selection_prompt,
    extract_best_textual_gradient_index,
)
from gepa.strategies.instruction_proposal import InstructionProposalSignature

logger = logging.getLogger(__name__)


# Constants for module optimization
TOOL_MODULE_PREFIX = "tool_module"


class LoggerAdapter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log(self, x: str):
        self.logger.info(x)


DSPyTrace = list[tuple[Any, dict[str, Any], Prediction]]

ReflectiveExample = TypedDict(
    "ReflectiveExample",
    {
        "Inputs": dict[str, Any],
        "Generated Outputs": dict[str, Any] | str,
        "Feedback": str,
    },
)

ReflectiveExample.__doc__ = """
Structure of individual examples in the reflective dataset.

Each example contains the predictor inputs, generated outputs, and feedback from evaluation.
"""


class ScoreWithFeedback(Prediction):
    score: float
    feedback: str | None = None
    subscores: dict[str, float] | None = None


class PredictorFeedbackFn(Protocol):
    def __call__(
        self,
        predictor_output: dict[str, Any],
        predictor_inputs: dict[str, Any],
        module_inputs: Example,
        module_outputs: Prediction,
        captured_trace: DSPyTrace,
    ) -> ScoreWithFeedback:
        """
        This function is used to provide feedback to a specific predictor.
        The function is called with the following arguments:
        - predictor_output: The output of the predictor.
        - predictor_inputs: The inputs to the predictor.
        - module_inputs: The inputs to the whole program --- `Example`.
        - module_outputs: The outputs of the whole program --- `Prediction`.
        - captured_trace: The trace of the module's execution.
        # Shape of trace is: [predictor_invocation_idx -> Tuple[Predictor, PredictorInputs, Prediction]]
        # Each trace is a tuple of (Predictor, PredictorInputs, Prediction)

        The function should return a `ScoreWithFeedback` object.
        The feedback is a string that is used to guide the evolution of the predictor.
        """
        ...


class DspyAdapter(GEPAAdapter[Example, TraceData, Prediction]):
    def __init__(
        self,
        student_module,
        metric_fn: Callable,
        feedback_map: dict[str, Callable],
        failure_score=0.0,
        num_threads: int | None = None,
        add_format_failure_as_feedback: bool = False,
        rng: random.Random | None = None,
        reflection_lm=None,
        custom_instruction_proposer: "ProposalFn | None" = None,
        warn_on_score_mismatch: bool = True,
        enable_tool_optimization: bool = False,
        reflection_minibatch_size: int | None = None,
        best_of_n: int = 1,
        num_iterations: int = 1,
    ):
        self.student = student_module
        self.metric_fn = metric_fn
        self.feedback_map = feedback_map
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.rng = rng or random.Random(0)
        self.reflection_lm = reflection_lm
        self.custom_instruction_proposer = custom_instruction_proposer
        self.warn_on_score_mismatch = warn_on_score_mismatch
        self.enable_tool_optimization = enable_tool_optimization
        self.reflection_minibatch_size = reflection_minibatch_size
        self.configure_textual_gradient_search(best_of_n=best_of_n, num_iterations=num_iterations)

    def configure_textual_gradient_search(self, *, best_of_n: int = 1, num_iterations: int = 1) -> None:
        if best_of_n < 1:
            raise ValueError("best_of_n must be >= 1.")
        if num_iterations < 1:
            raise ValueError("num_iterations must be >= 1.")
        self.best_of_n = best_of_n
        self.num_iterations = num_iterations

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        reflection_lm = self.reflection_lm or dspy.settings.lm
        # If custom proposer provided, override everything with custom proposer
        if self.custom_instruction_proposer:
            with dspy.context(lm=reflection_lm):
                return self.custom_instruction_proposer(
                    candidate=candidate,
                    reflective_dataset=reflective_dataset,
                    components_to_update=components_to_update,
                )

        # Otherwise, route to appropriate proposers
        # Separate into two categories: tool-using modules (ReAct) vs regular instructions
        # TODO: Add generic tool module support when DSPy trace lineage is improved
        tool_components = []
        instruction_components = []

        for c in components_to_update:
            if c.startswith(TOOL_MODULE_PREFIX):
                tool_components.append(c)
            else:
                instruction_components.append(c)

        results: dict[str, str] = {}

        with dspy.context(lm=reflection_lm):
            if instruction_components:
                for name in instruction_components:
                    if name not in candidate or name not in reflective_dataset:
                        continue
                    base_instruction = candidate[name]
                    dataset_with_feedback = reflective_dataset[name]
                    results[name] = self._run_instruction_iterations(
                        name=name,
                        base_instruction=base_instruction,
                        dataset_with_feedback=dataset_with_feedback,
                        reflection_lm=reflection_lm,
                    )

            if tool_components:
                from gepa.adapters.dspy_adapter.instruction_proposal import ToolProposer

                tool_proposer = ToolProposer()
                for module_key in tool_components:
                    if module_key not in candidate or module_key not in reflective_dataset:
                        continue
                    current_config = candidate[module_key]
                    dataset_for_module = reflective_dataset[module_key]
                    results[module_key] = self._run_tool_iterations(
                        module_key=module_key,
                        base_config=current_config,
                        dataset_for_module=dataset_for_module,
                        tool_proposer=tool_proposer,
                        reflection_lm=reflection_lm,
                    )

        return results

    def _run_instruction_iterations(
        self,
        name: str,
        base_instruction: str,
        dataset_with_feedback: list[ReflectiveExample],
        reflection_lm,
    ) -> str:
        instruction = base_instruction
        for _ in range(self.num_iterations):
            proposals = [
                InstructionProposalSignature.run(
                    lm=(lambda x: self._call_reflection_model(reflection_lm, x)),
                    input_dict={
                        "current_instruction_doc": instruction,
                        "dataset_with_feedback": dataset_with_feedback,
                    },
                )["new_instruction"]
                for _ in range(self.best_of_n)
            ]
            instruction = self._select_best_candidate(
                component_name=name,
                component_type="instruction",
                previous_text=instruction,
                dataset_with_feedback=dataset_with_feedback,
                proposals=proposals,
                reflection_lm=reflection_lm,
            )
        return instruction

    def _run_tool_iterations(
        self,
        module_key: str,
        base_config: str,
        dataset_for_module: list[ReflectiveExample],
        tool_proposer,
        reflection_lm,
    ) -> str:
        config = base_config
        for _ in range(self.num_iterations):
            proposals = [
                tool_proposer(
                    candidate={module_key: config},
                    reflective_dataset={module_key: dataset_for_module},
                    components_to_update=[module_key],
                ).get(module_key, config)
                for _ in range(self.best_of_n)
            ]
            config = self._select_best_candidate(
                component_name=module_key,
                component_type="tool configuration",
                previous_text=config,
                dataset_with_feedback=dataset_for_module,
                proposals=proposals,
                reflection_lm=reflection_lm,
            )
        return config

    def _select_best_candidate(
        self,
        component_name: str,
        component_type: str,
        previous_text: str,
        dataset_with_feedback: list[ReflectiveExample],
        proposals: list[str],
        reflection_lm,
    ) -> str:
        proposals = [proposal for proposal in proposals if proposal is not None]
        if not proposals:
            return previous_text
        if len(proposals) == 1 or self.best_of_n == 1:
            return proposals[0]
        if reflection_lm is None:
            logger.warning(
                "best_of_n>1 requested for component '%s', but no reflection_lm is available. "
                "Falling back to the first candidate.",
                component_name,
            )
            return proposals[0]

        prompt = build_textual_gradient_selection_prompt(
            component_name=component_name,
            component_type=component_type,
            previous_text=previous_text,
            dataset_with_feedback=dataset_with_feedback,
            proposals=proposals,
        )
        try:
            response_text = self._call_reflection_model(reflection_lm, prompt)
            best_index = extract_best_textual_gradient_index(response_text, len(proposals))
        except Exception as exc:
            logger.warning(
                "Failed to pick best candidate for %s due to %s. Defaulting to the first proposal.",
                component_name,
                exc,
            )
            best_index = 1
        return proposals[best_index - 1]

    @staticmethod
    def _call_reflection_model(reflection_lm, prompt: str | list[dict[str, Any]]) -> str:
        raw_outputs = reflection_lm(prompt)
        if isinstance(raw_outputs, str):
            return raw_outputs
        first = raw_outputs[0] if raw_outputs else ""
        if isinstance(first, dict):
            return first.get("text") or first.get("content") or json.dumps(first)
        return str(first)

    def build_program(self, candidate: dict[str, str]):
        new_prog = self.student.deepcopy()

        # Start with plain string instructions from candidate
        predictor_candidates = {k: v for k, v in candidate.items() if not k.startswith(TOOL_MODULE_PREFIX)}

        tool_candidates = {}
        if self.enable_tool_optimization:
            for key, value in candidate.items():
                if not key.startswith(TOOL_MODULE_PREFIX):
                    continue

                config = json.loads(value)

                for pred_name, instruction in config.items():
                    if isinstance(instruction, str):
                        predictor_candidates[pred_name] = instruction

                tool_candidates.update(config.get("tools", {}))

        # Update predictor instructions
        for name, pred in new_prog.named_predictors():
            if name in predictor_candidates:
                pred.signature = pred.signature.with_instructions(predictor_candidates[name])

        # Update tool descriptions
        if tool_candidates:
            self._update_tool_descriptions(new_prog, tool_candidates)

        return new_prog

    def _update_tool_descriptions(self, program: Module, tool_candidates: dict[str, Any]) -> None:
        all_tools = self._collect_tools(program)

        for tool_name, tool_config in tool_candidates.items():
            if tool_name not in all_tools:
                logger.warning(
                    f"Skipping updates for tool:'{tool_name}' because it cannot be detected on the student program."
                )
                continue

            tool = all_tools[tool_name]

            # Update tool description if present.
            if tool_config.get("desc"):
                tool.desc = tool_config["desc"]

            # Update arg descriptions if present.
            args_schema = tool_config.get("args") or {}
            for arg_name, arg_schema in args_schema.items():
                if arg_schema.get("description") is not None:
                    tool.args[arg_name]["description"] = arg_schema["description"]

    def _collect_tools(self, module: Module) -> dict[str, Tool]:
        """Recursively collect all Tool instances from a module and its sub-modules."""
        all_tools = {}
        visited = set()

        def _collect_from_attribute(attr_value):
            if isinstance(attr_value, Tool):
                all_tools[attr_value.name] = attr_value
            elif isinstance(attr_value, dspy.Module):
                _traverse(attr_value)
            elif isinstance(attr_value, list | dict):
                items = attr_value if isinstance(attr_value, list) else attr_value.values()
                for item in items:
                    if isinstance(item, Tool):
                        all_tools[item.name] = item

        def _traverse(current_module):
            if id(current_module) in visited or not hasattr(current_module, "__dict__"):
                return
            visited.add(id(current_module))

            for attr_value in current_module.__dict__.values():
                _collect_from_attribute(attr_value)

        _traverse(module)
        return all_tools

    def evaluate(self, batch, candidate, capture_traces=False):
        program = self.build_program(candidate)
        callback_metadata = (
            {"metric_key": "eval_full"}
            if self.reflection_minibatch_size is None or len(batch) > self.reflection_minibatch_size
            else {"disable_logging": True}
        )

        outputs: list[Prediction] = []
        scores: list[float] = []
        subscores: list[dict[str, float]] = []
        trajs: list[TraceData] | None = None

        if capture_traces:
            # bootstrap_trace_data-like flow with trace capture
            from dspy.teleprompt import bootstrap_trace as bootstrap_trace_module

            trajs = bootstrap_trace_module.bootstrap_trace_data(
                program=program,
                dataset=batch,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                raise_on_error=False,
                capture_failed_parses=True,
                failure_score=self.failure_score,
                format_failure_score=self.failure_score,
                callback_metadata=callback_metadata,
            )
            for t in trajs:
                outputs.append(t["prediction"])
                score_val, subscore_dict = self._extract_score_and_subscores(t.get("score"))
                if score_val is None:
                    score_val = self.failure_score
                scores.append(score_val)
                subscores.append(subscore_dict)
        else:
            evaluator = Evaluate(
                devset=batch,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                return_all_scores=True,
                return_outputs=True,
                failure_score=self.failure_score,
                provide_traceback=True,
                max_errors=len(batch) * 100,
                callback_metadata=callback_metadata,
            )
            res = evaluator(program)
            outputs = [r[1] for r in res.results]
            raw_scores = [r[2] for r in res.results]
            for raw_score in raw_scores:
                score_val, subscore_dict = self._extract_score_and_subscores(raw_score)
                if score_val is None:
                    score_val = self.failure_score
                scores.append(score_val)
                subscores.append(subscore_dict)

        has_subscores = any(subscores)
        # Map DSPy "subscores" into GEPA objective score payloads.
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajs,
            objective_scores=subscores if has_subscores else None,
        )

    @staticmethod
    def _extract_score_and_subscores(score_obj: Any) -> tuple[float | None, dict[str, float]]:
        """Extract score and subscores from various score object formats."""
        if score_obj is None:
            return None, {}
        if isinstance(score_obj, dict):
            score_val = score_obj.get("score")
            subscores = score_obj.get("subscores") or {}
            return score_val, dict(subscores)
        if hasattr(score_obj, "score"):
            score_val = getattr(score_obj, "score", None)
            subscores = getattr(score_obj, "subscores", None) or {}
            return score_val, dict(subscores)
        try:
            return float(score_obj), {}
        except (TypeError, ValueError):
            return None, {}

    def make_reflective_dataset(
        self, candidate, eval_batch, components_to_update
    ) -> dict[str, list[ReflectiveExample]]:
        program = self.build_program(candidate)

        ret_d: dict[str, list[ReflectiveExample]] = {}

        for pred_name in components_to_update:
            # Extract predictor name from component key
            if pred_name.startswith(TOOL_MODULE_PREFIX):
                target_name = pred_name.removeprefix(f"{TOOL_MODULE_PREFIX}:")
            else:
                target_name = pred_name

            # Find the predictor object
            module = None
            for name, m in program.named_predictors():
                if name == target_name:
                    module = m
                    break
            assert module is not None, f"Predictor not found: {target_name}"

            # Create reflective examples from traces
            items: list[ReflectiveExample] = []
            for data in eval_batch.trajectories or []:
                trace = data["trace"]
                example = data["example"]
                prediction = data["prediction"]
                module_score_obj = data.get("score")
                module_score, _ = self._extract_score_and_subscores(module_score_obj)

                trace_instances = [t for t in trace if t[0].signature.equals(module.signature)]
                if not self.add_format_failure_as_feedback:
                    trace_instances = [t for t in trace_instances if not isinstance(t[2], FailedPrediction)]
                if len(trace_instances) == 0:
                    continue

                selected = None
                for t in trace_instances:
                    if isinstance(t[2], FailedPrediction):
                        selected = t
                        break

                if selected is None:
                    if isinstance(prediction, FailedPrediction):
                        continue
                    selected = self.rng.choice(trace_instances)

                inputs = selected[1]
                outputs = selected[2]

                new_inputs = {}
                new_outputs = {}

                contains_history = False
                history_key_name = None
                for input_key, input_val in inputs.items():
                    if isinstance(input_val, History):
                        contains_history = True
                        assert history_key_name is None
                        history_key_name = input_key

                if contains_history:
                    s = "```json\n"
                    for i, message in enumerate(inputs[history_key_name].messages):
                        s += f"  {i}: {message}\n"
                    s += "```"
                    new_inputs["Context"] = s

                for input_key, input_val in inputs.items():
                    if contains_history and input_key == history_key_name:
                        continue

                    if isinstance(input_val, Type) and self.custom_instruction_proposer is not None:
                        # Keep original object - will be properly formatted when sent to reflection LM
                        new_inputs[input_key] = input_val
                    else:
                        new_inputs[input_key] = str(input_val)

                if isinstance(outputs, FailedPrediction):
                    s = "Couldn't parse the output as per the expected output format. The model's raw response was:\n"
                    s += "```\n"
                    s += outputs.completion_text + "\n"
                    s += "```\n\n"
                    new_outputs = s
                else:
                    for output_key, output_val in outputs.items():
                        new_outputs[output_key] = str(output_val)

                d = {"Inputs": new_inputs, "Generated Outputs": new_outputs}
                if isinstance(outputs, FailedPrediction):
                    adapter = ChatAdapter()
                    structure_instruction = ""
                    for dd in adapter.format(module.signature, [], {}):
                        structure_instruction += dd["role"] + ": " + dd["content"] + "\n"
                    d["Feedback"] = "Your output failed to parse. Follow this structure:\n" + structure_instruction
                    # d['score'] = self.failure_score
                else:
                    # Use actual predictor name for feedback lookup
                    feedback_fn = self.feedback_map[target_name]
                    fb = feedback_fn(
                        predictor_output=outputs,
                        predictor_inputs=inputs,
                        module_inputs=example,
                        module_outputs=prediction,
                        captured_trace=trace,
                    )
                    if isinstance(fb, dict):
                        feedback_score = fb.get("score")
                        feedback_text = fb.get("feedback", "")
                    else:
                        feedback_score = getattr(fb, "score", None)
                        feedback_text = getattr(fb, "feedback", "")
                    d["Feedback"] = feedback_text
                    if module_score is not None and feedback_score is not None:
                        if abs(feedback_score - module_score) > 1e-8:
                            if self.warn_on_score_mismatch:
                                logger.warning(
                                    "The score returned by the metric with pred_name is different from the overall metric score. This can indicate 2 things: Either the metric is non-deterministic (e.g., LLM-as-judge, Semantic score, etc.) or the metric returned a score specific to pred_name that differs from the module level score. Currently, GEPA does not support predictor level scoring (support coming soon), and only requires a feedback text to be provided, which can be specific to the predictor or program level. GEPA will ignore the differing score returned, and instead use module level score. You can safely ignore this warning if using a semantic metric, however, if this mismatch is caused due to predictor scoring, please return module-level scores. To disable this warning, set warn_on_score_mismatch=False."
                                )
                                self.warn_on_score_mismatch = False

                items.append(d)

            if len(items) == 0:
                logger.warning(f"  No valid reflective examples found for {pred_name}")
                continue

            ret_d[pred_name] = items

        if len(ret_d) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d

    # Always return strings from the LM outputs
    # Even when it returns a dict with e.g., "text" and "reasoning" fields
    def stripped_lm_call(self, x: str) -> list[str]:
        raw_outputs = self.reflection_lm(x)
        if isinstance(raw_outputs, str):
            raw_outputs = [raw_outputs]
        outputs = []
        for raw_output in raw_outputs:
            if type(raw_output) == str:
                outputs.append(raw_output)
            elif type(raw_output) == dict:
                if "text" not in raw_output:
                    raise KeyError("Missing 'text' field in the output from the base LM!")
                outputs.append(raw_output["text"])
            else:
                raise TypeError("Unexpected output type from the base LM! Expected str or dict")

        return outputs

    # TODO: Generic tool module optimization - pending DSPy trace lineage improvements
    # Currently only ReAct modules are supported for tool optimization.
    # Re-enable _update_candidate_tools when DSPy provides better tool→trace lineage.
    #
    # def _update_candidate_tools(self, candidate, program, trajectories) -> None:
    #     """Extract dspy.Tool objects from traces for tool modules and update candidate["tools"]."""
    #     ...

    # TODO: The current DSPyAdapter implementation uses the GEPA default propose_new_texts.
    # We can potentially override this, to use the instruction proposal similar to MIPROv2.

    # def propose_new_texts(
    #     self,
    #     candidate: Dict[str, str],
    #     reflective_dataset: Dict[str, List[Dict[str, Any]]],
    #     components_to_update: List[str]
    # ) -> Dict[str, str]:
    #     if self.adapter.propose_new_texts is not None:
    #         return self.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update)

    #     from .instruction_proposal import InstructionProposalSignature
    #     new_texts: Dict[str, str] = {}
    #     for name in components_to_update:
    #         base_instruction = candidate[name]
    #         dataset_with_feedback = reflective_dataset[name]
    #         new_texts[name] = InstructionProposalSignature.run(
    #             lm=self.reflection_lm,
    #             input_dict={
    #                 "current_instruction_doc": base_instruction,
    #                 "dataset_with_feedback": dataset_with_feedback
    #             }
    #         )['new_instruction']
    #     return new_texts
