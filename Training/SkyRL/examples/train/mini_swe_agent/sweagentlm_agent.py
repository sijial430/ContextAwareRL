"""
Custom agent and model for SWE-agent-LM that supports XML function calling format.

In mini-swe-agent v2, action parsing is handled by the Model class and execution
by the Environment. This module provides:
  - XMLFunctionCallingModel: Parses XML function calls and converts them to bash commands
  - SWEAgentLMAgent: Agent with turn reminders, history truncation, and submit handling
"""

import base64
import re

import litellm
from jinja2 import StrictUndefined, Template

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_text import format_observation_messages


# ---------------------------------------------------------------------------
# XML function call parsing
# ---------------------------------------------------------------------------

_FUNCTION_CALL_RE = re.compile(
    r"<function=(\w+)>\s*(.*?)\s*</function>",
    re.DOTALL,
)

_PARAMETER_RE = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>",
    re.DOTALL,
)


def parse_xml_function_call(content: str) -> dict | None:
    """Parse an XML function call from model output.

    Returns:
        {"function": str, "parameters": dict} or None if no valid call found
    """
    matches = _FUNCTION_CALL_RE.findall(content)
    if len(matches) != 1:
        return None
    func_name, param_block = matches[0]
    params = {}
    for param_match in _PARAMETER_RE.finditer(param_block):
        params[param_match.group(1)] = param_match.group(2)
    return {"function": func_name, "parameters": params}


# ---------------------------------------------------------------------------
# str_replace_editor -> bash conversion helpers
# ---------------------------------------------------------------------------


def _b64_encode(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _build_str_replace_editor_command(params: dict) -> str:
    """Convert str_replace_editor parameters into a bash command to execute in the sandbox."""
    command = params.get("command", "")
    path = params.get("path", "")

    if command == "view":
        view_range = params.get("view_range", "")
        if view_range:
            range_match = re.match(r"\[?\s*(\d+)\s*,\s*(-?\d+)\s*\]?", view_range)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                if end == -1:
                    return f'cat -n "{path}" | tail -n +{start}'
                else:
                    count = end - start + 1
                    return f'cat -n "{path}" | tail -n +{start} | head -n {count}'
        return (
            f'if [ -d "{path}" ]; then find "{path}" -maxdepth 2 -not -path "*/\\.*" | head -100; '
            f'else cat -n "{path}"; fi'
        )

    elif command == "create":
        file_text = params.get("file_text", "")
        file_b64 = _b64_encode(file_text)
        path_b64 = _b64_encode(path)
        py_script = (
            "import sys, base64, os\n"
            f"path = base64.b64decode('{path_b64}').decode()\n"
            f"file_text = base64.b64decode('{file_b64}').decode()\n"
            "if os.path.isfile(path):\n"
            "    print(f'ERROR: File already exists: {path}', file=sys.stderr)\n"
            "    sys.exit(1)\n"
            "os.makedirs(os.path.dirname(path), exist_ok=True)\n"
            "with open(path, 'w') as f:\n"
            "    f.write(file_text)\n"
            "print(f'File created successfully at: {path}')\n"
        )
        return f"python3 -c {_shell_quote(py_script)}"

    elif command == "str_replace":
        old_str = params.get("old_str", "")
        new_str = params.get("new_str", "")
        path_b64 = _b64_encode(path)
        old_b64 = _b64_encode(old_str)
        new_b64 = _b64_encode(new_str)
        py_script = (
            "import sys, base64\n"
            f"path = base64.b64decode('{path_b64}').decode()\n"
            f"old_str = base64.b64decode('{old_b64}').decode()\n"
            f"new_str = base64.b64decode('{new_b64}').decode()\n"
            "with open(path, 'r') as f:\n"
            "    content = f.read()\n"
            "count = content.count(old_str)\n"
            "if count == 0:\n"
            "    print(f'ERROR: old_str not found in {path}. No replacement made.', file=sys.stderr)\n"
            "    sys.exit(1)\n"
            "if count > 1:\n"
            "    print(f'ERROR: old_str found {count} times in {path}. Include more context to make it unique.', file=sys.stderr)\n"
            "    sys.exit(1)\n"
            "content = content.replace(old_str, new_str, 1)\n"
            "with open(path, 'w') as f:\n"
            "    f.write(content)\n"
            "print(f'Successfully replaced text in {path}')\n"
        )
        return f"python3 -c {_shell_quote(py_script)}"

    elif command == "insert":
        new_str = params.get("new_str", "")
        insert_line = params.get("insert_line", "0")
        try:
            line_num = int(insert_line)
        except (ValueError, TypeError):
            line_num = 0
        path_b64 = _b64_encode(path)
        new_b64 = _b64_encode(new_str)
        py_script = (
            "import sys, base64\n"
            f"path = base64.b64decode('{path_b64}').decode()\n"
            f"new_str = base64.b64decode('{new_b64}').decode()\n"
            f"insert_line = {line_num}\n"
            "with open(path, 'r') as f:\n"
            "    lines = f.readlines()\n"
            "if insert_line < 0 or insert_line > len(lines):\n"
            "    print(f'ERROR: insert_line {insert_line} out of range (file has {len(lines)} lines)', file=sys.stderr)\n"
            "    sys.exit(1)\n"
            "if not new_str.endswith('\\n'):\n"
            "    new_str += '\\n'\n"
            "lines.insert(insert_line, new_str)\n"
            "with open(path, 'w') as f:\n"
            "    f.writelines(lines)\n"
            "print(f'Successfully inserted text after line {insert_line} in {path}')\n"
        )
        return f"python3 -c {_shell_quote(py_script)}"

    elif command == "undo_edit":
        return "echo 'undo_edit is not supported in this environment. Please use str_replace to revert changes manually.'"

    else:
        return f"echo 'Unknown str_replace_editor command: {command}'"


# ---------------------------------------------------------------------------
# Custom Model: XML function calling
# ---------------------------------------------------------------------------


class XMLFunctionCallingModelConfig(LitellmModelConfig):
    """Config for the XML function calling model used by SWE-agent-LM."""

    format_error_template: str = (
        "ERROR: Invalid function call format. Expected exactly 1 XML function call, "
        "found {{actions|length}}.\n\n"
        "You must use the XML function calling format:\n"
        "<function=function_name>\n"
        "<parameter=parameter_name>value</parameter>\n"
        "</function>\n\n"
        "Available functions: bash, submit, str_replace_editor"
    )


class XMLFunctionCallingModel(LitellmModel):
    """Model that parses SWE-agent-LM's XML function calling format.

    Converts XML function calls to bash commands so the environment can execute them:
    - <function=bash> -> direct bash command
    - <function=str_replace_editor> -> converted to equivalent bash command
    - <function=submit> -> COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT sentinel command
      (handled by the environment's _check_finished)
    """

    def __init__(self, **kwargs):
        super().__init__(config_class=XMLFunctionCallingModelConfig, **kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        """Query without tool calls (text-based)."""
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _parse_actions(self, response) -> list[dict]:
        """Parse XML function call from model response and convert to bash command."""
        content = response.choices[0].message.content or ""
        parsed = parse_xml_function_call(content)
        if parsed is None:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(
                        self.config.format_error_template, undefined=StrictUndefined
                    ).render(actions=[], error="No valid XML function call found."),
                    "extra": {"interrupt_type": "FormatError", "model_response": content},
                }
            )

        func_name = parsed["function"]
        params = parsed["parameters"]

        if func_name == "submit":
            command = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached"
        elif func_name == "bash":
            command = params.get("command", "echo 'No command provided'")
        elif func_name == "str_replace_editor":
            command = _build_str_replace_editor_command(params)
        else:
            command = (
                f"echo 'Unknown function: {func_name}. "
                f"Available functions: bash, submit, str_replace_editor'"
            )

        return [{"command": command}]

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs as user observation messages."""
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )


# ---------------------------------------------------------------------------
# Custom Agent: turn reminders + history truncation
# ---------------------------------------------------------------------------


class SWEAgentLMAgent(DefaultAgent):
    """Agent with turn reminders and history truncation for SWE-agent-LM.

    Implements:
    - Turn countdown reminders appended to each observation
    - History truncation: keeps only the last N observations in full;
      older ones are replaced with a short placeholder to save context window
    """

    LAST_N_OBSERVATIONS = 5

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions, then add turn reminder to the last observation."""
        observation_messages = super().execute_actions(message)

        remaining = self.config.step_limit - self.n_calls

        # Add turn reminder to last observation
        if observation_messages:
            if remaining == 1:
                observation_messages[-1]["content"] += (
                    "\nREMINDER: You only have 1 turn left. Please submit your changes "
                    "using <function=submit>\n</function>."
                )
            elif remaining > 1:
                observation_messages[-1]["content"] += (
                    f"\nREMINDER: You have {remaining} turns left."
                )

        return observation_messages

    def step(self) -> list[dict]:
        """Run one step then truncate old observations."""
        result = super().step()
        self._truncate_old_observations()
        return result

    def _truncate_old_observations(self):
        """Truncate old user observation messages to save context window.

        Mimics SWE-agent's `last_n_observations` history processor.
        """
        obs_indices = [
            i for i, m in enumerate(self.messages)
            if i >= 2 and m.get("role") == "user"
        ]
        if len(obs_indices) <= self.LAST_N_OBSERVATIONS:
            return
        for idx in obs_indices[:-self.LAST_N_OBSERVATIONS]:
            content = self.messages[idx].get("content", "")
            if len(content) > 200:
                self.messages[idx]["content"] = (
                    "[Observation truncated for context window]\n"
                    + content[:100]
                    + "\n..."
                )
