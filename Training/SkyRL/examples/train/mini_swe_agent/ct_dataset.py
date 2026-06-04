"""ChooseTrajectoryDataset: pre-tokenized single-turn MCQ dataset for the
DPO-style choose-trajectory auxiliary loss.

Mirrors the inference pipeline in
`SWE-bench/Klear_mini_swe_agent_plus-trajectories-66k/eval_klear_prefill_logits.py`:

    1. Apply chat template with ``enable_thinking=False``.
    2. Append the stripped prefill (e.g. "The correct option is").
    3. Resolve the single-token continuations for " A" and " B" once per
       tokenizer and store their ids on the dataset.
    4. Pre-tokenize every row at init time, filter by ``max_prompt_length``.

The ``collate_fn`` returns a left-padded :class:`TrainingInputBatch` whose
metadata carries ``a_id`` / ``b_id``. The per-sample ``correct_is_a`` tensor
is chunkable by the SkyRL dispatch layer (metadata is replicated, not
sliced, so the boolean label must live in the tensor batch itself).
"""

from typing import List, Optional

import datasets as hf_datasets
import torch
from loguru import logger
from transformers import PreTrainedTokenizerBase

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch


def resolve_letter_tokens(tokenizer: PreTrainedTokenizerBase, prefill: str):
    """Resolve ``(base_prefill, a_ids, b_ids)`` for next-token A/B classification.

    ``a_ids`` / ``b_ids`` are lists of candidate token ids — we try both the
    space-prefixed (" A", " B") and bare ("A", "B") forms and keep whichever
    add exactly one token after ``base``. For a non-empty prefill ending in
    text, typically only the space-prefixed form is valid. For an empty
    prefill (predicting the first token right after the chat template), both
    may be valid, in which case the policy worker picks the higher-probability
    one per sample.
    """
    base = prefill.rstrip()
    base_ids = tokenizer(base, add_special_tokens=False).input_ids

    def _candidates(letter: str):
        cands: List[int] = []
        for suffix in (" " + letter, letter):
            ids = tokenizer(base + suffix, add_special_tokens=False).input_ids
            if len(ids) == len(base_ids) + 1:
                tid = ids[-1]
                if tid not in cands:
                    cands.append(tid)
        if not cands:
            raise ValueError(
                f'Neither {" " + letter!r} nor {letter!r} adds exactly 1 token '
                f'after prefill {base!r}. base_ids tail: {base_ids[-4:]}'
            )
        return cands

    return base, _candidates("A"), _candidates("B")


class ChooseTrajectoryDataset:
    def __init__(
        self,
        datasets: List[str],
        tokenizer: PreTrainedTokenizerBase,
        prefill: str = "The correct option is",
        max_prompt_length: int = 32768,
        enable_thinking: bool = False,
        prompt_key: str = "prompt",
        answer_key: str = "correct_answer",
        num_workers: int = 8,
    ):
        self.tokenizer = tokenizer
        self.prefill = prefill
        self.max_prompt_length = max_prompt_length
        self.enable_thinking = enable_thinking
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.num_workers = num_workers

        self.base_prefill, self.a_ids, self.b_ids = resolve_letter_tokens(tokenizer, prefill)
        logger.info(
            f"[CT] Normalized prefill: {self.base_prefill!r} "
            f"a_ids={self.a_ids} ({[tokenizer.decode([i]) for i in self.a_ids]!r}) "
            f"b_ids={self.b_ids} ({[tokenizer.decode([i]) for i in self.b_ids]!r})"
        )

        ds_list = []
        for source in datasets:
            ds = hf_datasets.load_dataset("json", data_files=source, keep_in_memory=True)["train"]
            ds_list.append(ds)
        self.dataframe = hf_datasets.concatenate_datasets(ds_list)
        logger.info(f"[CT] Loaded {len(self.dataframe)} raw rows")

        self.dataframe = self.dataframe.filter(
            lambda ex: str(ex.get(answer_key, "")).strip() in ("A", "B"),
            desc="[CT] filter correct_answer in {A,B}",
        )
        logger.info(f"[CT] After A/B answer filter: {len(self.dataframe)}")

        base_prefill = self.base_prefill
        enable_thinking_flag = self.enable_thinking

        def _tokenize(ex):
            messages = [{"role": "user", "content": ex[prompt_key]}]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking_flag,
            )
            prompt_text = prompt_text + base_prefill
            ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            return {
                "ct_input_ids": ids,
                "ct_len": len(ids),
                "ct_correct_is_a": 1 if str(ex[answer_key]).strip() == "A" else 0,
            }

        self.dataframe = self.dataframe.map(
            _tokenize,
            num_proc=num_workers,
            desc="[CT] tokenize (chat template + prefill)",
        )

        pre = len(self.dataframe)
        self.dataframe = self.dataframe.filter(
            lambda ex: ex["ct_len"] <= max_prompt_length,
            num_proc=num_workers,
            desc=f"[CT] filter overlong > {max_prompt_length}",
        )
        logger.info(f"[CT] After overlong filter: {pre} -> {len(self.dataframe)}")

        if len(self.dataframe) == 0:
            logger.warning("[CT] All rows were filtered out! CT dataset is empty.")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe[idx]
        return {
            "input_ids": row["ct_input_ids"],
            "correct_is_a": int(row["ct_correct_is_a"]),
        }

    def collate_fn(self, items: List[dict]) -> TrainingInputBatch:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")

        max_len = max(len(item["input_ids"]) for item in items)
        n = len(items)

        input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, max_len), dtype=torch.long)
        correct_is_a = torch.zeros((n,), dtype=torch.long)

        for i, item in enumerate(items):
            ids = item["input_ids"]
            L = len(ids)
            input_ids[i, max_len - L :] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, max_len - L :] = 1
            correct_is_a[i] = item["correct_is_a"]

        position_ids = torch.clamp(attention_mask.cumsum(dim=-1) - 1, min=0)

        batch = TrainingInputBatch(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "ct_correct_is_a": correct_is_a,
            }
        )
        batch.metadata = {
            "a_ids": [int(i) for i in self.a_ids],
            "b_ids": [int(i) for i in self.b_ids],
        }
        return batch
