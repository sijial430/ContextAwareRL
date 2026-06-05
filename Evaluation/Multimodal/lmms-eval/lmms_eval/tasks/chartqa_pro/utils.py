import io
import re

from datasets import Dataset
from PIL import Image


def chartqa_pro_process_docs(dataset: Dataset) -> Dataset:
    """Expand multiple Q&A pairs per chart into individual samples."""
    expanded = []
    for doc in dataset:
        img_field = doc["image"]
        if isinstance(img_field, bytes):
            img_bytes = img_field
        else:
            buf = io.BytesIO()
            img_field.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        for q, a in zip(doc["Question"], doc["Answer"]):
            expanded.append(
                {
                    "image_bytes": img_bytes,
                    "question": q,
                    "answer": a,
                    "question_type": doc["Question Type"],
                }
            )
    return Dataset.from_list(expanded)


def chartqa_pro_doc_to_visual(doc: dict) -> list:
    img = Image.open(io.BytesIO(doc["image_bytes"]))
    return [img.convert("RGB")]


def chartqa_pro_doc_to_text(doc: dict, lmms_eval_specific_kwargs: dict = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "")
    return f"{pre}{doc['question']}{post}"


def _to_float(text: str) -> float | None:
    try:
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0
        return float(text)
    except ValueError:
        return None


def _anls_score(prediction: str, target: str, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (ANLS) with acceptance threshold."""
    pred = prediction.lower().strip()
    gt = target.lower().strip()
    if not pred and not gt:
        return 1.0
    # Compute edit distance via DP
    m, n = len(pred), len(gt)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if pred[i - 1] == gt[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    edit_dist = dp[n]
    max_len = max(m, n)
    nls = edit_dist / max_len  # 0 = identical, 1 = completely different
    return 0.0 if nls > threshold else 1.0 - nls


def _score_answer(prediction: str, target: str, question_type: str) -> float:
    """Score prediction against target based on question type."""
    pred = prediction.strip()
    gt = target.strip()
    q_type = (question_type or "").lower()

    # Numeric: 5% relative tolerance
    pred_f = _to_float(pred)
    gt_f = _to_float(gt)
    if pred_f is not None and gt_f:
        return 1.0 if abs(pred_f - gt_f) / abs(gt_f) <= 0.05 else 0.0

    # Fact-checking / multiple-choice: exact match
    if any(t in q_type for t in ("fact", "choice", "multi")):
        return 1.0 if pred.lower() == gt.lower() else 0.0

    # List answers: average element-wise ANLS
    if "," in gt:
        gt_items = [x.strip() for x in gt.split(",")]
        pred_items = [x.strip() for x in pred.split(",")]
        scores = [
            _anls_score(p, g)
            for p, g in zip(
                pred_items + [""] * max(0, len(gt_items) - len(pred_items)),
                gt_items,
            )
        ]
        return sum(scores) / len(scores)

    # Default: ANLS
    return _anls_score(pred, gt)


def chartqa_pro_process_results(doc: dict, results: list) -> dict:
    pred = results[0].strip()
    score = _score_answer(pred, doc["answer"], doc["question_type"])
    return {"accuracy": score}


def chartqa_pro_aggregate(results: list) -> float:
    return sum(results) / len(results) if results else 0.0
