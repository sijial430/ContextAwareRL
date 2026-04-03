# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


# Metadata
REWARD_NAME = "math"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    pattern_boxed = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    pattern_answer = re.compile(r"<think>.*</think>.*<answer>.*</answer>.*", re.DOTALL)
    pattern_boxed_thinking = re.compile(r".*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    pattern_answer_thinking = re.compile(r".*</think>.*<answer>.*</answer>.*", re.DOTALL)
    format_match = re.fullmatch(pattern_boxed, response) or re.fullmatch(pattern_answer, response) or re.fullmatch(pattern_boxed_thinking, response) or re.fullmatch(pattern_answer_thinking, response)
    return 1.0 if format_match  else 0.0


def extract_answer_content(response: str) -> str:
    """Extract content between the last <answer> and </answer> tags."""
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = pattern.findall(response)
    if matches:
        return matches[-1].strip()  
    return ""


def _normalize_answer(text: str) -> str:
    text = text.lower().strip()
    # Remove punctuation but keep decimal points, minus signs and percentage symbols
    text = re.sub(r"(?<!\d)[^\w\s.-]|[^\w\s%-](?!\d)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_answer_content(response)
    if not answer:
        return 0.0
    
    # First check with grade_answer
    if grade_answer(answer, ground_truth):
        return 1.0
    
    # Handle multiple choice answers: if ground_truth is a single letter (e.g., "A", "B")
    # and candidate answer starts with it (e.g., "A. the first image"), consider it correct
    ground_truth_stripped = ground_truth.strip()
    answer_stripped = answer.strip()
    if len(ground_truth_stripped) == 1 and ground_truth_stripped.isalpha():
        # Check if answer starts with the ground truth (case-sensitive, allowing for punctuation)
        pattern = re.compile(rf"^{re.escape(ground_truth_stripped)}\s*[\.\)\:\-]?\s*")
        if pattern.match(answer_stripped):
            return 1.0
    
    # If grade_answer returns False, do partial matching
    candidate_norm = _normalize_answer(answer)
    ground_truth_norm = _normalize_answer(ground_truth)
    
    if not candidate_norm:
        return 0.0
    
    if candidate_norm == ground_truth_norm:
        return 1.0
    
    # Partial matching: both answers should have at least 3 words and length > 10
    candidate_words = candidate_norm.split()
    ground_truth_words = ground_truth_norm.split()
    if (
        len(candidate_words) >= 3
        and len(ground_truth_words) >= 3
        and len(candidate_norm) > 10
        and len(ground_truth_norm) > 10
    ):
        if ground_truth_norm in candidate_norm or candidate_norm in ground_truth_norm:
            return 1.0
    
    return 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
