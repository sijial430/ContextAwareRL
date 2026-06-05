import io
import os
import re
from collections import defaultdict

from loguru import logger as eval_logger
from PIL import Image

MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

COMPARE_PROMPT = """You are evaluating whether a predicted answer is correct given a ground truth answer for a chart understanding question.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {prediction}

Rules:
- Accept small numeric differences (e.g., 32.35 ≈ 32.34)
- Accept capitalization differences
- For dates/years: require exact match
- Accept mathematically equivalent expressions
- Ignore unit differences when comparing numeric values

Is the predicted answer correct? Reply with only "Yes" or "No"."""


def _get_openai_client():
    """Lazy-initialize the OpenAI client to avoid import-time failures."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    kwargs = {"api_key": api_key}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return OpenAI(**kwargs)


def _extract_answer(text: str) -> str:
    """Extract content from <answer>...</answer> tags, or fall back to full text."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _llm_judge(question: str, ground_truth: str, prediction: str, max_retries: int = 5) -> float:
    """Use GPT-4 to judge if prediction matches ground truth. Returns 1.0 or 0.0."""
    client = _get_openai_client()
    prompt = COMPARE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4,
            )
            answer = response.choices[0].message.content.strip().lower()
            if "yes" in answer:
                return 1.0
            if "no" in answer:
                return 0.0
        except Exception as e:
            eval_logger.warning(f"ChartMuseum LLM judge attempt {attempt + 1}/{max_retries} failed: {e}")
    eval_logger.error("ChartMuseum LLM judge failed after all retries, scoring 0.")
    return 0.0


def _find_image_dir() -> str:
    """Locate the ChartMuseum images directory, downloading if necessary.

    Search order:
    1. CHARTMUSEUM_IMAGE_DIR env var (explicit override)
    2. HF hub snapshot cache (already downloaded)
    3. Auto-download via snapshot_download into the HF hub cache
    """
    explicit = os.getenv("CHARTMUSEUM_IMAGE_DIR", "")
    if explicit and os.path.isdir(explicit):
        return explicit

    # Check if images already exist in the HF hub snapshot
    hub_cache = os.getenv(
        "HUGGINGFACE_HUB_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    )
    repo_snapshots = os.path.join(hub_cache, "datasets--lytang--ChartMuseum", "snapshots")
    if os.path.isdir(repo_snapshots):
        for commit_hash in os.listdir(repo_snapshots):
            candidate = os.path.join(repo_snapshots, commit_hash)
            if os.path.isdir(os.path.join(candidate, "images")):
                return candidate

    # Images not found — download them now
    eval_logger.info("ChartMuseum images not found locally. Downloading from HuggingFace Hub...")
    from huggingface_hub import snapshot_download

    snapshot_dir = snapshot_download(
        repo_id="lytang/ChartMuseum",
        repo_type="dataset",
        allow_patterns=["images/*"],
    )
    eval_logger.info(f"ChartMuseum images downloaded to: {snapshot_dir}")
    return snapshot_dir


_IMAGE_DIR: str | None = None


def _get_image_dir() -> str:
    global _IMAGE_DIR
    if _IMAGE_DIR is None:
        _IMAGE_DIR = _find_image_dir()
        eval_logger.debug(f"ChartMuseum image directory: {_IMAGE_DIR}")
    return _IMAGE_DIR


def _load_image(image_field) -> Image.Image:
    """Load a PIL image from various source types."""
    if hasattr(image_field, "convert"):
        return image_field.convert("RGB")
    if isinstance(image_field, bytes):
        return Image.open(io.BytesIO(image_field)).convert("RGB")
    if isinstance(image_field, str):
        if image_field.startswith(("http://", "https://")):
            import requests

            r = requests.get(image_field, timeout=30)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        path = os.path.join(_get_image_dir(), image_field)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ChartMuseum image not found: {path}\n"
                "Download images by running: sbatch download_chartmuseum_images.sh\n"
                "Or set CHARTMUSEUM_IMAGE_DIR to the directory containing the images/ folder."
            )
        return Image.open(path).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image_field)}")


def chartmuseum_doc_to_visual(doc: dict) -> list:
    return [_load_image(doc["image"])]


def chartmuseum_doc_to_text(doc: dict, lmms_eval_specific_kwargs: dict = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "")
    return f"{pre}{doc['question']}{post}"


def chartmuseum_process_results(doc: dict, results: list) -> dict:
    raw_pred = results[0].strip()
    pred = _extract_answer(raw_pred)
    score = _llm_judge(doc["question"], doc["answer"], pred)
    return {
        "accuracy": {"score": score, "reasoning_type": doc["reasoning_type"]},
    }


def chartmuseum_accuracy_aggregate(results: list) -> float:
    by_type: dict[str, list] = defaultdict(list)
    all_scores = []
    for r in results:
        score = r["score"]
        all_scores.append(score)
        by_type[r["reasoning_type"]].append(score)

    def avg(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    for rtype, scores in sorted(by_type.items()):
        print(f"  {rtype}: {avg(scores):.4f} ({len(scores)} samples)")

    return avg(all_scores)
