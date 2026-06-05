#!/usr/bin/env python3
"""
Run inference on SWE-bench using any HuggingFace causal language model.

Supports chat-format models (Instruct/Chat) like Qwen2.5-Coder-7B-Instruct.
Generates unified diff patches for each SWE-bench instance.

Usage:
    python -m swebench.inference.run_hf_model \
        --model_name_or_path /path/to/Qwen2.5-Coder-7B-Instruct \
        --dataset_path /path/to/SWE-bench_Verified \
        --output_dir ./outputs \
        --max_new_tokens 4096 \
        --temperature 0.0
"""

import json
import logging
import re
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from swebench.inference.make_datasets.utils import extract_diff

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are an expert software engineer. You will be given a problem statement describing \
a bug or feature request for an open-source Python repository. Your task is to generate \
a patch (in unified diff format) that resolves the issue.

Rules:
1. Output ONLY a single patch in unified diff format, wrapped in ```diff ... ``` markers.
2. The patch must be applicable via `git apply`.
3. Include correct file paths (a/path and b/path).
4. Include correct line numbers in @@ hunks.
5. Do not include any other text, explanation, or commentary outside the diff block.\
"""


def build_user_prompt(instance: dict) -> str:
    """Build the user prompt from a SWE-bench instance."""
    parts = []
    parts.append(f"## Repository: {instance['repo']}")
    parts.append(f"## Instance ID: {instance['instance_id']}")
    parts.append(f"## Version: {instance.get('version', 'unknown')}")
    parts.append("")
    parts.append("## Problem Statement")
    parts.append(instance["problem_statement"])

    hints = instance.get("hints_text", "")
    if hints and hints.strip():
        parts.append("")
        parts.append("## Hints")
        parts.append(hints)

    parts.append("")
    parts.append(
        "Please generate a patch in unified diff format to resolve the issue above. "
        "Wrap your patch in ```diff ... ``` markers."
    )
    return "\n".join(parts)


def load_model_and_tokenizer(model_name_or_path: str, dtype: str = "auto"):
    """Load model and tokenizer from HuggingFace."""
    logger.info(f"Loading tokenizer from {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {
        "auto": "auto",
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(dtype, "auto")

    logger.info(f"Loading model from {model_name_or_path} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info(f"Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'single device'}")
    return model, tokenizer


def load_swebench_dataset(dataset_path: str, split: str = "test"):
    """Load SWE-bench dataset from HuggingFace or local path."""
    logger.info(f"Loading dataset from {dataset_path}, split={split}")
    path = Path(dataset_path)

    if path.is_dir():
        # Check for parquet files
        parquet_files = list(path.rglob("*.parquet"))
        if parquet_files:
            ds = load_dataset("parquet", data_files=[str(f) for f in parquet_files], split="train")
        elif (path / split).exists():
            ds = load_from_disk(str(path / split))
        else:
            ds = load_dataset(str(path), split=split)
    else:
        ds = load_dataset(dataset_path, split=split)

    logger.info(f"Loaded {len(ds)} instances")
    return ds


def generate_patch(
    model,
    tokenizer,
    instance: dict,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 0.95,
) -> dict:
    """Generate a patch for a single SWE-bench instance."""
    user_prompt = build_user_prompt(instance)

    # Build chat messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Use tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback: simple concatenation
        input_text = f"System: {SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    input_len = input_ids.shape[1]

    logger.info(f"  Input: {input_len} tokens")

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
    elapsed = time.time() - start_time

    new_tokens = output_ids[0][input_len:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    num_new = len(new_tokens)
    logger.info(f"  Generated {num_new} tokens in {elapsed:.1f}s ({num_new/elapsed:.1f} tok/s)")

    # Extract diff from output
    model_patch = extract_diff(output_text)

    return {
        "instance_id": instance["instance_id"],
        "model_name_or_path": str(model.name_or_path) if hasattr(model, "name_or_path") else "unknown",
        "model_patch": model_patch if model_patch else "",
        "full_output": output_text,
    }


def get_output_file(output_dir: str, model_name_or_path: str, dataset_path: str, temperature: float, top_p: float) -> Path:
    """Construct output file path."""
    model_nick = Path(model_name_or_path).name
    dataset_nick = Path(dataset_path).name
    output_file = Path(output_dir) / f"{dataset_nick}__{model_nick}__temp-{temperature}__top-p-{top_p}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return output_file


def load_existing_ids(output_file: Path) -> set:
    """Load instance IDs already in the output file (for resumption)."""
    if not output_file.exists():
        return set()
    existing = set()
    with open(output_file) as f:
        for line in f:
            try:
                data = json.loads(line)
                existing.add(data["instance_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    logger.info(f"Found {len(existing)} existing predictions, will skip them")
    return existing


def main():
    parser = ArgumentParser(description="Run SWE-bench inference with any HuggingFace model")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to local model or HuggingFace model name")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to SWE-bench dataset (local dir or HF name)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Maximum new tokens to generate per instance")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, args.dtype)

    # Load dataset
    dataset = load_swebench_dataset(args.dataset_path, args.split)

    # Shard if requested
    if args.shard_id is not None and args.num_shards is not None:
        dataset = dataset.shard(args.num_shards, args.shard_id, contiguous=True)
        logger.info(f"Shard {args.shard_id}/{args.num_shards}: {len(dataset)} instances")

    # Output file
    output_file = get_output_file(
        args.output_dir, args.model_name_or_path, args.dataset_path,
        args.temperature, args.top_p,
    )
    logger.info(f"Output file: {output_file}")

    # Resume support
    existing_ids = load_existing_ids(output_file)

    # Generate predictions
    success = 0
    fail = 0
    with open(output_file, "a") as f:
        for idx in tqdm(range(len(dataset)), desc="Generating patches"):
            instance = dataset[idx]
            if instance["instance_id"] in existing_ids:
                continue

            logger.info(f"[{idx+1}/{len(dataset)}] {instance['instance_id']}")
            try:
                result = generate_patch(
                    model, tokenizer, instance,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                print(json.dumps(result), file=f, flush=True)
                success += 1
            except Exception as e:
                logger.error(f"Failed on {instance['instance_id']}: {e}")
                # Write empty patch so we don't retry on resume
                result = {
                    "instance_id": instance["instance_id"],
                    "model_name_or_path": args.model_name_or_path,
                    "model_patch": "",
                    "full_output": f"ERROR: {e}",
                }
                print(json.dumps(result), file=f, flush=True)
                fail += 1

    logger.info(f"Done! Success: {success}, Failed: {fail}")
    logger.info(f"Predictions saved to: {output_file}")


if __name__ == "__main__":
    main()
