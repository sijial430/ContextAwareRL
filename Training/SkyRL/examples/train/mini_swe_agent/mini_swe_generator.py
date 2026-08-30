import asyncio
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import yaml
import traceback
import ray
from pathlib import Path

from minisweagent.models import get_model
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_path
from .mini_swe_utils import evaluate_trajectory, get_sb_environment
from .sweagentlm_agent import SWEAgentLMAgent, XMLFunctionCallingModel

from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator, GeneratorOutput, GeneratorInput
from skyrl.train.generators.base import TrajectoryID, TrainingPhase, BatchMetadata
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl.train.generators.utils import (
    get_rollout_metrics,
    get_response_ids_and_loss_mask_from_messages,
)

# Hard wall-clock timeout per trajectory. If a single Ray rollout task (one
# Apptainer sandbox + agent loop) does not finish within this many seconds, it
# is forcibly cancelled so it cannot block the rest of the batch indefinitely.
# 90 minutes — long enough that normal trajectories (~observed step time) can
# finish, short enough that one hung sandbox can't stall the batch indefinitely.
TRAJECTORY_TIMEOUT_SECONDS = int(os.getenv("MINISWE_TRAJECTORY_TIMEOUT_SECONDS", 90 * 60))


@dataclass
class MiniSWEGeneratorConfig(GeneratorConfig):
    """Extended generator config with Mini-SWE-Agent-specific fields."""

    miniswe_config_path: str = ""
    miniswe_traj_dir: str = ""


def _clean_messages_for_tokenizer(messages: list[dict]) -> list[dict]:
    """Strip non-standard keys from messages for tokenization.

    mini-swe-agent v2 messages contain 'extra', 'tool_calls', etc. that
    tokenizers' apply_chat_template does not expect.
    """
    cleaned = []
    for m in messages:
        role = m.get("role", "")
        # Skip exit messages (added by v2 when agent finishes)
        if role == "exit":
            continue
        cleaned.append({"role": role, "content": m.get("content") or ""})
    return cleaned


class DefaultAgentWithReminder(DefaultAgent):
    """DefaultAgent subclass that appends turn-count reminders to observations."""

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions and append a turn reminder to the last observation."""
        observation_messages = super().execute_actions(message)

        remaining = self.config.step_limit - self.n_calls
        if observation_messages:
            if remaining == 1:
                observation_messages[-1]["content"] += (
                    "\nREMINDER: You only have 1 turn left. Please provide the final answer"
                )
            elif remaining > 1:
                observation_messages[-1]["content"] += (
                    f"\nREMINDER: You have {remaining} turns left to arrive at the solution."
                )

        return observation_messages


@ray.remote(num_cpus=0.5)
def init_and_run(
    instance: dict,
    litellm_model_name: str,
    sweagent_config: dict,
    generator_cfg: GeneratorConfig,
    data_source: str,
    sampling_params: dict,
    trajectory_id: TrajectoryID,
    global_step: int,
    training_phase: TrainingPhase,
):
    from loguru import logger

    model_config = sweagent_config.get("model", {})
    # Use new sampling parameters
    # Can also have custom sampling parameters per trajectory (ex: custom max tokens)
    model_config.setdefault("model_kwargs", {}).update(sampling_params)

    # Select agent class and model class based on config
    agent_config = sweagent_config.get("agent", {})
    use_xml_agent = "str_replace_editor" in agent_config.get("system_template", "")

    if use_xml_agent:
        # XML function calling mode: use custom model that parses XML
        model_config["model_name"] = litellm_model_name
        model_config.pop("model_class", None)  # not needed, we instantiate directly
        model = XMLFunctionCallingModel(**model_config)
    else:
        # Text-based bash mode: let get_model handle model_class selection
        model = get_model(litellm_model_name, model_config)

    agent = None
    env = None
    extra_info = None
    result = None
    reward = 0
    error = None
    exit_status = ""
    try:
        env = get_sb_environment(sweagent_config, instance, data_source)

        if use_xml_agent:
            agent = SWEAgentLMAgent(model, env, **agent_config)
        else:
            agent = DefaultAgentWithReminder(model, env, **agent_config)

        run_result = agent.run(instance["problem_statement"])
        exit_status = run_result.get("exit_status", "")
        result = run_result.get("submission", "")
    except Exception as e:
        error_msg = str(e).replace("{", "{{").replace("}", "}}")
        logger.error(f"Error processing instance {instance['instance_id']}: {error_msg}", exc_info=True)
        exit_status = type(e).__name__
        result = str(e)
        error = str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        # Explicitly clean up the rollout environment to avoid container/sandbox leaks
        if env is not None:
            try:
                env.cleanup()
            except Exception:
                logger.debug(f"Error cleaning up rollout environment for {instance['instance_id']}", exc_info=True)

        # Create trajectory directory with proper structure: step_{global_step}/{train/eval}
        path = Path(generator_cfg.miniswe_traj_dir) / f"step_{global_step}" / training_phase
        path.mkdir(parents=True, exist_ok=True)
        # Use instance_id and repetition_id for meaningful filename: {instance_id}_{repetition_id}.json
        instance_id = instance["instance_id"]
        filename = f"{instance_id}_{trajectory_id.repetition_id}.json"
        traj_path = path / filename
        if agent is not None:
            eval_error = None
            try:
                result = evaluate_trajectory(instance, result, sweagent_config, data_source)
                reward = int(result["resolved"])
                eval_error = result["eval_error"]
                if eval_error:
                    error = eval_error
                    safe_eval_error = str(eval_error).replace("{", "{{").replace("}", "}}")
                    logger.debug(f"Error during evaluation {safe_eval_error}")
            except Exception as e:
                safe_e = str(e).replace("{", "{{").replace("}", "}}")
                logger.debug(f"Error during evaluation {safe_e}")
                logger.debug(f"traceback: {traceback.format_exc()}")
                eval_error = str(e)
                error = str(e)

            # Use agent.save() (v2 API) instead of save_traj()
            agent.save(
                traj_path,
                {
                    "result": result,
                    "reward": reward,
                    "eval_error": eval_error,
                    "extra_info": extra_info,
                },
            )

    return (agent.messages if agent is not None else [], reward, error)


class MiniSweAgentGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        skyrl_gym_cfg: SkyRLGymConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):

        # Call parent constructor first
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer)

        self.http_server_inference_engine_client_host = generator_cfg.inference_engine.http_endpoint_host

        self.http_server_inference_engine_client_port = generator_cfg.inference_engine.http_endpoint_port

        self.base_url = (
            f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        )
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.litellm_model_name = "openai/" + self.model_name

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError("MiniSWEAgentGenerator doesn't support custom chat template")

    async def minisweagent_agent_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]]]:

        sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())
        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        instance = env_extras["instance"]
        if isinstance(instance, str):
            instance = json.loads(instance)

        obj_ref = init_and_run.remote(
            instance,
            self.litellm_model_name,
            sweagent_config,
            self.generator_cfg,
            env_extras["data_source"],
            sampling_params,
            trajectory_id,
            batch_metadata.global_step,
            batch_metadata.training_phase,
        )

        # Enforce a hard wall-clock timeout per trajectory so a single hung
        # sandbox cannot stall the whole batch (seen on step 22 of
        # run_mini_swe_agentforge8B: 127/128 trajectories finished in ~33min but
        # one sandbox hung for 3h+ until SLURM killed the job).
        async def _await_ref():
            return await obj_ref

        try:
            messages, reward, error = await asyncio.wait_for(
                _await_ref(), timeout=TRAJECTORY_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            from loguru import logger

            instance_id = instance.get("instance_id", "?") if isinstance(instance, dict) else "?"
            logger.warning(
                f"Trajectory for {instance_id} exceeded {TRAJECTORY_TIMEOUT_SECONDS}s hard timeout; "
                f"cancelling Ray task and dropping this sample."
            )
            try:
                # force=True sends SIGKILL to the worker process, which also
                # kills the Apptainer sandbox subprocess it spawned.
                ray.cancel(obj_ref, force=True)
            except Exception as cancel_err:
                logger.debug(f"Error cancelling ray task for {instance_id}: {cancel_err}")
            return None, None, None, None, None, None

        if not len(messages):
            return None, None, None, None, None, None

        # Clean messages for tokenization: strip v2 extra keys and exit messages
        clean_messages = _clean_messages_for_tokenizer(messages)

        # TODO (sumanthrh): This is currently hardcoded for SWEBench with 2 initial messages (system and user).
        response_messages = clean_messages[2:]

        for message in clean_messages[:2]:
            assert message["role"] in (
                "system",
                "user",
            ), "Expected the first two messages to be system and user messages"

        initial_input_ids = self.tokenizer.apply_chat_template(
            clean_messages[:2], add_generation_prompt=False, return_dict=False, tokenize=True
        )
        initial_prompt_length = len(initial_input_ids)

        # We remove trailing `user` messages - this is added by Mini-SWE-Agent to capture the final git diff for the trajectory
        last_idx = len(response_messages) - 1
        while last_idx >= 0 and response_messages[last_idx]["role"] == "user":
            last_idx -= 1
        if last_idx < 0:
            raise ValueError(
                "Found no assistant messages. Please ensure that your environment is configured correctly and the `OPENAI_BASE_URL` points to the HTTP server from the inference engine client"
            )
        response_messages = response_messages[: last_idx + 1]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
            response_messages,
            self.tokenizer,
            assistant_logprobs=None,
        )

        # Extract prompt ids
        prompt_ids = initial_input_ids

        # Calculate maximum response tokens allowed
        max_response_tokens = max_tokens + max_input_length - initial_prompt_length

        # Determine stop reason
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return (response_ids, reward, stop_reason, loss_mask, prompt_ids, None)

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch["batch_metadata"]
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length
        sampling_params = get_sampling_params_for_backend(
            self.generator_cfg.inference_engine.backend, self.generator_cfg.sampling_params
        )

        tasks = []

        for i in range(len(prompts)):
            tasks.append(
                self.minisweagent_agent_loop(
                    prompts[i],
                    env_extras[i],
                    max_tokens=max_tokens,
                    max_input_length=max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i],
                    batch_metadata=batch_metadata,
                )
            )

        # `return_exceptions=True` ensures a single trajectory failure
        # (timeout, sandbox crash, tokenization error, etc.) only drops that
        # one sample instead of killing the whole training step. Exceptions
        # raised by `minisweagent_agent_loop` are returned as objects and we
        # convert them into dropped-trajectory sentinels below.
        from loguru import logger

        all_outputs = await asyncio.gather(*tasks, return_exceptions=True)

        # Track outputs by original batch index so we can resample dropped
        # slots from sibling trajectories in the same GRPO group. The trainer's
        # `validate_generator_output` requires len(responses) == len(prompts),
        # so dropping without padding crashes the step (seen at step 41 of
        # run_mini_swe_agentforge8B when one trajectory hit the 5400s hard
        # timeout and left 127/128 responses).
        per_idx_output: List[Optional[Tuple]] = [None] * len(all_outputs)
        dropped_indices: List[int] = []
        for i, output in enumerate(all_outputs):
            if isinstance(output, BaseException):
                instance_id = "?"
                try:
                    inst = env_extras[i].get("instance")
                    if isinstance(inst, str):
                        inst = json.loads(inst)
                    if isinstance(inst, dict):
                        instance_id = inst.get("instance_id", "?")
                except Exception:
                    pass
                logger.warning(
                    f"Dropping trajectory for instance {instance_id} due to "
                    f"{type(output).__name__}: {output}"
                )
                dropped_indices.append(i)
                continue
            # Successful return but the agent loop chose to drop this sample
            # (e.g. timeout sentinel `(None, None, None, None, None, None)`).
            if output[0] is None:
                dropped_indices.append(i)
                continue
            per_idx_output[i] = output

        num_dropped = len(dropped_indices)
        if num_dropped:
            # Resample each dropped slot from a sibling in the same
            # instance_id group (i.e. another GRPO repetition of the same
            # prompt) so the group's advantage normalization stays coherent.
            # Fall back to any valid output if the entire group failed.
            from collections import defaultdict

            valid_by_instance: Dict[str, List[int]] = defaultdict(list)
            for i, out in enumerate(per_idx_output):
                if out is not None:
                    valid_by_instance[trajectory_ids[i].instance_id].append(i)

            any_valid_idx = next((i for i, o in enumerate(per_idx_output) if o is not None), None)
            num_resampled = 0
            num_unfilled = 0
            for idx in dropped_indices:
                siblings = valid_by_instance.get(trajectory_ids[idx].instance_id)
                if siblings:
                    src = siblings[idx % len(siblings)]
                elif any_valid_idx is not None:
                    src = any_valid_idx
                else:
                    num_unfilled += 1
                    continue
                per_idx_output[idx] = per_idx_output[src]
                num_resampled += 1

            logger.warning(
                f"Dropped {num_dropped}/{len(all_outputs)} trajectories this step; "
                f"resampled {num_resampled} from sibling trajectories"
                + (f", {num_unfilled} unfilled (whole batch failed)" if num_unfilled else "")
                + "."
            )

        valid_outputs = [o for o in per_idx_output if o is not None]

        responses = [output[0] for output in valid_outputs]
        rewards = [output[1] for output in valid_outputs]
        stop_reasons = [output[2] for output in valid_outputs]
        loss_masks = [output[3] for output in valid_outputs]
        prompt_token_ids = [output[4] for output in valid_outputs]
        if not len(responses):
            raise ValueError(
                "Found no valid responses for this step. This means that generation failed for all trajectories, likely due to errors in environment setup."
            )
        rollout_metrics = get_rollout_metrics(responses, rewards)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }

        return generator_output
