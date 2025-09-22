from __future__ import annotations

import ast
import json
import base64
import mimetypes
import io
import re
import importlib
from typing import Any, TypedDict, Callable
from PIL import Image as PILImage
from pydantic import BaseModel, Field

import logging
import os
from datetime import datetime

from gepa.core.adapter import EvaluationBatch, GEPAAdapter


class MultiModalDataInst(TypedDict):
    input: str
    images: list[Any] # could be URLs, paths, PIL Images, or bytes
    answer: str
    additional_context: dict[str, str]


class MultiModalTrajectory(TypedDict):
    data: MultiModalDataInst
    full_assistant_response: str


class MultiModalRolloutOutput(TypedDict):
    full_assistant_response: str

class MultiModalStructuredOutput(BaseModel):
    final_answer: str = Field(..., description="The final answer only (no extra words, units, punctuation beyond %).")
    solution_pad: str = Field(..., description="Step-by-step reasoning or explanation that led to the answer.")


class MultimodalAdapter(GEPAAdapter[MultiModalDataInst, MultiModalTrajectory, MultiModalRolloutOutput]):
    """
    Multimodal Adapter is a GEPAAdapter for any dataset that contains multimodal inputs.
    It is designed to handle a wide range of image tasks, including charts, image understanding, and more.
    """

    def __init__(
        self,
        model: str,
        failure_score: float = 0.0,
        api_base: str | None = "http://localhost:8050/v1",
        max_litellm_workers: int = 16,
        api_key: str | None = None,
        metric_func: Callable | None = None,
        task_type: str = "chartqa",
    ) -> None:
        import litellm

        self.model = model
        self.failure_score = failure_score
        self.litellm = litellm
        self.api_base = api_base
        self.api_key = api_key
        self.max_litellm_workers = max_litellm_workers
        self.metric_func = metric_func
        self.task_type = task_type

        # Setup logging to file
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"logs/{self.task_type}_{self.model.replace('/', '_')}_{timestamp}.log"
        with open(self.log_file, "w") as f:
            f.write(f"Evaluation log for {self.model} on {self.task_type}\n")
            f.write(f"Started at {datetime.now().isoformat()}\n\n")

    def _image_part(self, img: Any) -> dict[str, Any]:
        # Accept data URLs, http(s), local paths, PIL.Image, bytes
        if isinstance(img, str):
            if img.startswith(("data:", "http://", "https://")):
                return {"type": "image_url", "image_url": {"url": img}}
            mime, _ = mimetypes.guess_type(img)
            mime = mime or "image/png"
            with open(img, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        if isinstance(img, PILImage.Image):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        if isinstance(img, (bytes, bytearray)):
            b64 = base64.b64encode(bytes(img)).decode("utf-8")
            return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        return {"type": "text", "text": f"[Unsupported image type: {type(img)}]"}

    def _get_metric_function(self, task_type: str = None) -> Callable:
        """Dynamically load the appropriate metric function based on task type"""
        task = task_type or self.task_type
        
        metric_modules = {
            "chartqa": "gepa.examples.multimodal.chartqa.chartqa_metric",
        }
        
        if task not in metric_modules:
            task = "default"
            
        try:
            module_path = metric_modules[task]
            module = importlib.import_module(module_path)
            return getattr(module, "get_metric")()
        except (ImportError, AttributeError) as e:
            # Fall back to string matching as a default metric
            return lambda pred, refs: float(any(r.lower() in pred.lower() for r in refs))

    def evaluate(
        self,
        batch: list[MultiModalDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[MultiModalTrajectory, MultiModalRolloutOutput]:
        outputs: list[MultiModalRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[MultiModalTrajectory] | None = [] if capture_traces else None
        if not candidate:
            raise ValueError("Candidate must contain at least one component text.")
        system_content = next(iter(candidate.values()))

        litellm_requests: list[list[dict[str, Any]]] = []
        for data in batch:
            user_parts: list[dict[str, Any]] = [{"type": "text", "text": data["input"]}]
            for img in data.get("images", []):
                user_parts.append(self._image_part(img))
            litellm_requests.append(
                [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_parts},
                ]
            )

        try:
            responses = self.litellm.batch_completion(
                model=self.model,
                messages=litellm_requests,
                api_base=self.api_base,
                api_key=self.api_key,
                max_workers=self.max_litellm_workers,
                format=MultiModalStructuredOutput.model_json_schema(),
            )
            try:
                with open(self.log_file, "a") as log:
                    log.write("=== Raw Model Responses ===\n")
                    for i, response in enumerate(responses):
                        raw_content = response.choices[0].message.content if response.choices else "NO RESPONSE"
                        log.write(f"Response {i}: {raw_content[:500]}...\n\n")
            except Exception as e:
                with open(self.log_file, "a") as log:
                    log.write(f"Error logging responses: {str(e)}\n")
            
            # Make the parsing logic more robust:
            for data, response in zip(batch, responses, strict=False):
                try:
                    raw = (response.choices[0].message.content or "").strip()
                    with open(self.log_file, "a") as log:
                        log.write(f"Processing raw response: {raw[:100]}...\n")
                except Exception:
                    raw = ""
                    with open(self.log_file, "a") as log:
                        log.write("Failed to get raw response content\n")
        except Exception as e:
            raise RuntimeError(f"litellm batch_completion failed: {e}") from e

        for data, response in zip(batch, responses, strict=False):
            try:
                raw = (response.choices[0].message.content or "").strip()
            except Exception:
                raw = ""

            correct_format = True
            final_answer = ""
            solution_pad = ""
            try:
                parsed = ast.literal_eval(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("Not dict")
                solution_pad = str(parsed.get("solution_pad", "")).strip()
                final_answer = str(parsed.get("final_answer", "")).strip()
                if not final_answer:
                    correct_format = False
            except Exception:
                parsed = {"_raw": raw, "_error": "parse_failed"}
                correct_format = False

            if not final_answer:
                match = re.search(r"[Ff]inal\s*[Aa]nswer\s*[:\-]?\s*(.*)", raw)
                if match:
                    final_answer = match.group(1).strip()
                    correct_format = True 

            if correct_format and final_answer:
                full_assistant_response = f"Assistant's Solution: {solution_pad}\nFinal Answer: {final_answer}"
            else:
                full_assistant_response = (
                    "Assistant failed to respond with the correct answer or format.\n"
                    f"Raw Response: {parsed.get('_raw','')}"
                )


            add_ctx = data.get("additional_context", {})
            ref_json = add_ctx.get("reference_answers_json")
            if ref_json:
                try:
                    refs = json.loads(ref_json)
                    if not isinstance(refs, list):
                        refs = [str(refs)]
                except Exception:
                    refs = [data["answer"]]
            else:
                refs = [data["answer"]]

            metric_input = full_assistant_response  # metric extracts "Final Answer:" internally
            if self.metric_func:
                metric_fn = self.metric_func
            else:
                task_type = data.get("additional_context", {}).get("task_type", self.task_type)
                metric_fn = self._get_metric_function(task_type)
            metric_score = metric_fn.score(metric_input, refs)
            score = metric_score if metric_score > 0 else self.failure_score

            with open(self.log_file, "a") as log:
                log.write(f"Question: {data['input'][:200]}\n")
                log.write(f"Gold answer(s): {refs}\n")
                log.write(f"Predicted: '{final_answer}'\n")
                log.write(f"Score: {metric_score:.4f} (passing: {metric_score > 0})\n")
                if metric_score <= 0:
                    log.write(f"Failed response: {raw[:300]}...\n")
                log.write("-" * 80 + "\n")

            output = {"full_assistant_response": full_assistant_response}
            outputs.append(output)
            scores.append(score)
            if capture_traces:
                trajectories.append({"data": data, "full_assistant_response": full_assistant_response})

        nonzero = sum(1 for s in scores if s > 0)
        with open(self.log_file, "a") as log:
            log.write("\nSummary:\n")
            log.write(f"Batch results: {nonzero}/{len(scores)} correct ({nonzero/len(scores) if scores else 0:.2%})\n")
            log.write(f"Mean score: {sum(scores)/len(scores) if scores else 0:.4f}\n")
            log.write("=" * 80 + "\n\n")
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[MultiModalTrajectory, MultiModalRolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        assert len(components_to_update) == 1
        comp = components_to_update[0]

        if eval_batch.trajectories is None:
            raise ValueError("Trajectories required (capture_traces=True).")

        items: list[dict[str, Any]] = []
        for traj, score in zip(eval_batch.trajectories, eval_batch.scores, strict=False):
            data = traj["data"]
            generated_outputs = traj["full_assistant_response"]
            add_ctx = data.get("additional_context", {})
            add_ctx_str = "\n".join(f"{k}: {v}" for k, v in add_ctx.items()) if add_ctx else ""

            if score > 0.0:
                feedback = f"The generated response is correct. The final answer is: {data['answer']}."
            else:
                feedback = (
                    f"The generated response is incorrect. The correct answer is: {data['answer']}.\n"
                    "Refine the instruction to ensure structured dict output and exact answer."
                )
                if add_ctx_str:
                    feedback += f"\nAdditional context:\n{add_ctx_str}"

            items.append(
                {
                    "Inputs": {
                        "input": data["input"],
                        "images": data.get("images", []),
                        "additional_context": add_ctx,
                    },
                    "Generated Outputs": generated_outputs,
                    "Feedback": feedback,
                }
            )

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return {comp: items}