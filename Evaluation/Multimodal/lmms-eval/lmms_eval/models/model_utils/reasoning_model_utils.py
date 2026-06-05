import re


def _extract_boxed_content(text: str) -> str | None:
    """Extract content from the last \\boxed{...}, correctly handling nested braces."""
    # Find the last occurrence of \boxed{
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    # Unclosed brace — return whatever was accumulated
    return text[start:i].strip()


def parse_reasoning_model_answer(model_answer: str) -> str:
    # 1. Strip <think>...</think> block if present
    think_match = re.search(r"</think>\s*", model_answer, re.DOTALL)
    if think_match:
        model_answer = model_answer[think_match.end():]

    # 2. Try to extract from <answer>...</answer> tags
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", model_answer, re.DOTALL)
    if answer_match:
        return answer_match.group(1)

    # 3. Try to extract from \boxed{...}, handling nested braces correctly
    boxed = _extract_boxed_content(model_answer)
    if boxed is not None:
        return boxed

    # 4. Return the remaining text (after think block removal) stripped
    return model_answer.strip()
