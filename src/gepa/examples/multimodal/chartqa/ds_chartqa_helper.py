import io
import base64
import json
import logging
from typing import Any, TypedDict, Tuple
from datasets import load_dataset
from tqdm.auto import tqdm


log = logging.getLogger(__name__)

DATASET_NAME = "HuggingFaceM4/ChartQA"


class ChartQASample(TypedDict):
    prompt: str
    images: list[str]
    answer: str
    additional_context: dict[str, str]


def _pil_to_data_url(img) -> str:
    """Convert a PIL.Image to a data URL (PNG)."""
    buf = io.BytesIO()
    try:
        img.save(buf, format="PNG")
    except Exception as e:
        raise RuntimeError(f"Failed to encode image to PNG: {e}") from e
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _coerce_answer(val: Any) -> Tuple[str, list[str]]:
    """Return (first_answer, all_answers) from a label that may be str | list[str] | None."""
    if isinstance(val, list):
        answers = [str(x).strip() for x in val if str(x).strip()]
        first = answers[0] if answers else ""
        return first, answers
    if val is None:
        return "", []
    s = str(val).strip()
    return s, [s] if s else []


def _row_to_sample(row: dict[str, Any]) -> ChartQASample:
    """Map a HF ChartQA row to our adapter-ready dict."""
    try:
        img_data_url = _pil_to_data_url(row["image"])
    except Exception as e:
        log.warning(f"Image conversion failed; inserting placeholder. Error: {e}")
        img_data_url = "data:image/png;base64,"  # empty image placeholder

    prompt = str(row.get("query") or "").strip()
    ans_first, ans_list = _coerce_answer(row.get("label", []))
    return ChartQASample(
        prompt=prompt,
        images=[img_data_url],
        answer=ans_first,
        additional_context={"reference_answers_json": json.dumps(ans_list)},
    )


def _load_split(split_name: str, limit: int | None) -> list[ChartQASample]:
    log.info(f"Loading split '{split_name}' from {DATASET_NAME} (limit={limit})")
    ds = load_dataset(DATASET_NAME, split=split_name)
    total = len(ds) if hasattr(ds, "__len__") else None
    show_total = (
        min(limit, total) if (limit is not None and total is not None)
        else (limit if limit is not None else total)
    )
    out: list[ChartQASample] = []
    for i, row in tqdm(
        enumerate(ds),
        total=show_total,
        desc=f"ChartQA {split_name}",
        unit="ex",
        leave=False,
    ):
        if limit is not None and i >= limit:
            break
        out.append(_row_to_sample(row))
    log.info(f"Prepared {len(out)} examples for split '{split_name}'")
    return out


def init_dataset(
    root: str | None = None,
    train_ann: str | None = None,
    val_ann: str | None = None,
    test_ann: str | None = None,
    images_dir: str | None = None,
    limit: int | None = None,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[list[ChartQASample], list[ChartQASample], list[ChartQASample]]:
    """
    Load ChartQA from HF and return (train, val, test) as adapter-ready dicts.
    Per-split limits override the global limit if provided.
    """
    tr_lim = train_limit if train_limit is not None else limit
    va_lim = val_limit if val_limit is not None else limit
    te_lim = test_limit if test_limit is not None else limit

    log.info(
        f"Initializing ChartQA: train_limit={tr_lim}, val_limit={va_lim}, test_limit={te_lim}"
    )

    train = []
    try:
        train = _load_split("train", tr_lim)
    except Exception as e:
        log.warning(f"Failed to load train split: {e}")

    try:
        val = _load_split("validation", va_lim)
    except Exception:
        try:
            val = _load_split("val", va_lim)
        except Exception as e:
            log.warning(f"Failed to load validation split(s): {e}")
            val = []

    try:
        test = _load_split("test", te_lim)
    except Exception as e:
        log.warning(f"Failed to load test split: {e}")
        test = []

    log.info(f"Loaded splits: train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test