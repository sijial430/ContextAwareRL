from typing import Any, Dict, Optional

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


_VALID_CHOICES = ("A", "B")


def extract_choice(text: str, choices: tuple = _VALID_CHOICES) -> Optional[str]:
    """Return the last uppercase choice letter that appears in `text`.

    The model is run with thinking enabled (max_generate_length=4096) so the
    response can contain a ``<think>...</think>`` block followed by the final
    letter. We simply take the rightmost occurrence of any valid choice
    character — case-sensitive, no word-boundary check.
    """
    if not text:
        return None
    last_pos = -1
    last_char: Optional[str] = None
    for c in choices:
        pos = text.rfind(c)
        if pos > last_pos:
            last_pos = pos
            last_char = c
    return last_char


class MCQEnv(BaseTextEnv):
    """Single-turn multiple-choice env with verifiable letter-match reward."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = str(extras["reward_spec"]["ground_truth"]).strip().upper()

    def step(self, action: str) -> BaseTextEnvStepOutput:
        pred = extract_choice(action)
        correct = pred is not None and pred == self.ground_truth
        reward = 1.0 if correct else 0.0
        metadata = {"acc": float(correct), "pred": pred or ""}
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata=metadata)
