import random
from typing import Any, Callable

import dspy
from dspy.adapters.types import History
from dspy.evaluate import Evaluate
from dspy.primitives import Example, Prediction
from dspy.teleprompt.bootstrap_trace import TraceData

from gepa import EvaluationBatch, GEPAAdapter


class DspyAdapter(GEPAAdapter[Example, TraceData, Prediction]):
    def __init__(
        self,
        task_lm: dspy.LM,
        metric_fn: Callable,
        reflection_lm: dspy.LM,
        failure_score=0.0,
        num_threads: int | None = None,
        add_format_failure_as_feedback: bool = False,
        rng: random.Random | None = None,
    ):
        self.task_lm = task_lm
        self.metric_fn = metric_fn
        assert reflection_lm is not None, "DspyAdapter for full-program evolution requires a reflection_lm to be provided"
        self.reflection_lm = reflection_lm
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.rng = rng or random.Random(0)

    def build_program(self, candidate: dict[str, str]) -> tuple[dspy.Module, None] | tuple[None, str]:
        candidate_src = candidate["program"]
        context = {}
        o = self.load_dspy_program_from_code(candidate_src, context)
        return o

    def load_dspy_program_from_code(
        self,
        candidate_src: str,
        context: dict,
    ):
        try:
            compile(candidate_src, "<string>", "exec")
        except SyntaxError as e:
            # print(f"Syntax Error in original code {e}")
            # return None
            import traceback

            tb = traceback.format_exc()
            return None, f"Syntax Error in code: {e}\n{tb}"

        try:
            exec(candidate_src, context)  # expose to current namespace
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            return None, f"Error in executing code: {e}\n{tb}"

        dspy_program = context.get("program")

        if dspy_program is None:
            return (
                None,
                "Your code did not define a `program` object. Please define a `program` object which is an instance of `dspy.Module`, either directly by dspy.Predict or dspy.ChainOfThought, or by instantiating a class that inherits from `dspy.Module`.",
            )
        else:
            if not isinstance(dspy_program, dspy.Module):
                return (
                    None,
                    f"Your code defined a `program` object, but it is an instance of {type(dspy_program)}, not `dspy.Module`. Please define a `program` object which is an instance of `dspy.Module`, either directly by dspy.Predict or dspy.ChainOfThought, or by instantiating a class that inherits from `dspy.Module`.",
                )

        dspy_program.set_lm(self.task_lm)

        return dspy_program, None

    def evaluate(self, batch, candidate, capture_traces=False):
        program, feedback = self.build_program(candidate)

        if program is None:
            return EvaluationBatch(outputs=None, scores=[self.failure_score for _ in batch], trajectories=feedback)

        if capture_traces:
            # bootstrap_trace_data-like flow with trace capture
            from dspy.teleprompt.bootstrap_trace import bootstrap_trace_data

            trajs = bootstrap_trace_data(
                program=program,
                dataset=batch,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                raise_on_error=False,
                capture_failed_parses=True,
                failure_score=self.failure_score,
                format_failure_score=self.failure_score,
            )
            scores = []
            outputs = []
            for t in trajs:
                outputs.append(t["prediction"])
                if hasattr(t["prediction"], "__class__") and t.get("score") is None:
                    scores.append(self.failure_score)
                else:
                    score = t["score"]
                    if hasattr(score, "score"):
                        score = score["score"]
                    scores.append(score)
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs)
        else:
            evaluator = Evaluate(
                devset=batch,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                return_all_scores=True,
                failure_score=self.failure_score,
                provide_traceback=True,
                max_errors=len(batch) * 100,
            )
            res = evaluator(program)
            outputs = [r[1] for r in res.results]
            scores = [r[2] for r in res.results]
            scores = [s["score"] if hasattr(s, "score") else s for s in scores]
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        proposed_program, _ = self.build_program(candidate)

        assert set(components_to_update) == {"program"}, f"set(components_to_update) = {set(components_to_update)}"
        from dspy.teleprompt.bootstrap_trace import FailedPrediction

        ret_d: dict[str, list[dict[str, Any]]] = {}

        if isinstance(eval_batch.trajectories, str):
            feedback = eval_batch.trajectories
            return {"program": {"Feedback": feedback}}

        ########
        items: list[dict[str, Any]] = []
        for data in eval_batch.trajectories or []:
            example_data = {}
            trace = data["trace"]
            example = data["example"]
            example_data["Program Inputs"] = {**example.inputs()}
            prediction = data["prediction"]
            example_data["Program Outputs"] = {**prediction}
            module_score = data["score"]

            if hasattr(module_score, "feedback"):
                feedback_text = module_score["feedback"]
            else:
                feedback_text = None

            if hasattr(module_score, "score"):
                module_score = module_score["score"]

            trace_instances = trace

            if len(trace_instances) == 0:
                continue

            selected = None
            for t in trace_instances:
                if isinstance(t[2], FailedPrediction):
                    selected = t
                    break

            if selected is not None:
                trace_instances = [selected]

            trace_d = []
            example_data["Program Trace"] = trace_d
            for selected in trace_instances:
                inputs = selected[1]
                outputs = selected[2]

                pred_name = None
                for name, predictor in proposed_program.named_predictors():
                    if predictor.signature.equals(selected[0].signature):
                        pred_name = name
                        break
                assert pred_name is not None, f"Could not find predictor for {selected[0].signature}"

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

                d = {"Called Module": pred_name, "Inputs": new_inputs, "Generated Outputs": new_outputs}
                # if isinstance(outputs, FailedPrediction):
                #     adapter = ChatAdapter()
                #     structure_instruction = ""
                #     for dd in adapter.format(module.signature, [], {}):
                #         structure_instruction += dd["role"] + ": " + dd["content"] + "\n"
                #     d["Feedback"] = "Your output failed to parse. Follow this structure:\n" + structure_instruction
                #     # d['score'] = self.failure_score
                # else:
                # assert fb["score"] == module_score, f"Currently, GEPA only supports feedback functions that return the same score as the module's score. However, the module-level score is {module_score} and the feedback score is {fb.score}."
                # d['score'] = fb.score
                trace_d.append(d)

            if feedback_text is not None:
                example_data["Feedback"] = feedback_text

            items.append(example_data)

        if len(items) == 0:
            raise Exception("No valid predictions found for program.")

        ret_d["program"] = items

        ########
        if len(ret_d) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        from gepa.adapters.dspy_full_program_adapter.dspy_program_proposal_signature import DSPyProgramProposalSignature

        new_texts: dict[str, str] = {}
        for name in components_to_update:
            base_instruction = candidate[name]
            dataset_with_feedback = reflective_dataset[name]
            new_texts[name] = DSPyProgramProposalSignature.run(
                lm=self.reflection_lm,
                input_dict={"curr_program": base_instruction, "dataset_with_feedback": dataset_with_feedback},
            )["new_program"]
        return new_texts
