from __future__ import annotations

import ast
import json
import base64
import mimetypes
import io
import re
from typing import Any, TypedDict
from PIL import Image as PILImage
from pydantic import BaseModel, Field
from gepa.examples.multimodal.chartqa.chartqa_metric import get_metric 

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
import logging


class MultiModalDataInst(TypedDict):
    """
    User-defined type of input data to the program under optimization.
    """
    input: str
    images: list[Any]
    answer: str
    additional_context: dict[str, str]


class MultiModalTrajectory(TypedDict):
    data: MultiModalDataInst # The input data instance
    full_assistant_response: str # The full raw response from the model


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
        api_key: str | None = None
    ) -> None:
        import litellm

        self.model = model
        self.failure_score = failure_score
        self.litellm = litellm
        self.api_base = api_base
        self.api_key = api_key
        self.max_litellm_workers = max_litellm_workers
        self.log = logging.getLogger("MultimodalAdapter")
        self.relaxed_metric = get_metric()

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

    def evaluate(
        self,
        batch: list[MultiModalDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[MultiModalTrajectory, MultiModalRolloutOutput]:
        outputs: list[MultiModalRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[MultiModalTrajectory] | None = [] if capture_traces else None
        log_file = open("chartqa_eval_log.txt", "a", encoding="utf-8")
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
            metric_score = self.relaxed_metric.score(metric_input, refs) if correct_format else 0.0
            log_file.write(
                f"Q: {data['input']}\n"
                f"Pred: {final_answer}\n"
                f"Gold: {refs}\n"
                f"Score: {metric_score}\n"
                f"Raw: {raw}\n"
                "----\n"
            )
            score = metric_score if metric_score > 0 else self.failure_score

            output = {"full_assistant_response": full_assistant_response}
            outputs.append(output)
            scores.append(score)
            if capture_traces:
                trajectories.append({"data": data, "full_assistant_response": full_assistant_response})

        nonzero = sum(1 for s in scores if s > 0)
        self.log.info(
            "Multimodal evaluate (%s): batch=%d parsed_ok=%d nonzero=%d mean=%.4f",
            self.relaxed_metric.name,
            len(batch),
            sum(1 for o in outputs if "Final Answer:" in o["full_assistant_response"]),
            nonzero,
            (sum(scores) / max(1, len(scores))),
        )
        log_file.close()
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