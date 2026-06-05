from lmms_eval.models.model_utils.reasoning_model_utils import parse_reasoning_model_answer


def extract_final_answer(text: str) -> str:
    """Extract the final answer from model output.

    Strips <think>...</think> blocks, then tries <answer> tags,
    then \\boxed{}, then returns the remaining text.
    """
    return parse_reasoning_model_answer(text)
