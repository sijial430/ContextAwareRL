"""
Tinker-based RL training for Mini-SWE-Agent.

Replaces the local FSDP training + vLLM inference with Tinker's remote API,
while keeping the existing minisweagent agent loop and Apptainer sandbox unchanged.

Architecture:
  1. Local OpenAI-compatible proxy server → Tinker SamplingClient (replaces vLLM)
  2. Tinker TrainingClient forward_backward + optim_step (replaces FSDP)
  3. Existing init_and_run() agent loop + Apptainer sandbox (unchanged)

Session strategy:
  - TrainingClient is only alive during the short training phase (~2 min).
  - During the long rollout phase (~80 min), only a lightweight SamplingClient
    exists, so there is no idle session that can time out.
  - State (weights + optimizer) is persisted to Tinker storage between phases
    and restored via create_training_client_from_state_with_optimizer.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import ray
import torch
import wandb
import yaml
from aiohttp import web
from datasets import load_dataset
from transformers import AutoTokenizer

import tinker
from tinker import types
from tinker.types.tensor_data import TensorData

from examples.train.mini_swe_agent.mini_swe_generator import (
    init_and_run,
    _clean_messages_for_tokenizer,
    MiniSWEGeneratorConfig,
    TRAJECTORY_TIMEOUT_SECONDS,
)
from skyrl.train.generators.utils import get_response_ids_and_loss_mask_from_messages
from skyrl.train.generators.base import TrajectoryID, TrainingPhase, BatchMetadata

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TinkerMiniSWEConfig:
    # Model
    model_name: str = "Qwen/Qwen3.5-35B-A3B"
    lora_rank: int = 32
    learning_rate: float = 1.0e-6

    # Data
    train_data: str = ""
    val_data: str = ""

    # Training loop
    max_epochs: int = 6
    train_batch_size: int = 32
    eval_batch_size: int = 100
    n_samples_per_prompt: int = 8
    save_every: int = 10
    eval_every: int = 20
    eval_before_train: bool = False

    # Generation
    max_prompt_length: int = 4096
    max_generate_length: int = 4096
    max_input_length: int = 28672
    max_turns: int = 35

    # Loss function: "ppo" or "importance_sampling"
    loss_fn: Literal["ppo", "importance_sampling"] = "ppo"
    normalize_advantages: bool = True

    # DAPO: asymmetric clip (Clip-Higher). Set clip_epsilon_high > clip_epsilon_low
    # to encourage exploration on low-probability tokens.
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.2
    # DAPO: dynamic sampling — drop groups whose rewards are all identical
    # (zero variance → zero advantage → no gradient signal).
    dynamic_sampling: bool = False
    # Number of PPO epochs per batch. Must be > 1 for the clip to actually
    # trigger — with a single optim step, π_θ == π_old and ratio ≡ 1.
    ppo_epochs: int = 1

    # Paths
    miniswe_config_path: str = "examples/train/mini_swe_agent/swebench.yaml"
    traj_dir: str = ""
    ckpt_path: str = ""
    log_path: str = ""

    # Proxy server
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 8001

    # Logging
    wandb_project: str = "mini_swe"
    wandb_run_name: str = "tinker_swe_qwen35b"

    # Resume
    resume_from: str = ""  # Path to Tinker state checkpoint


# ---------------------------------------------------------------------------
# Orphan sandbox cleanup (memory leak mitigation)
# ---------------------------------------------------------------------------
def kill_orphan_sandboxes() -> None:
    """Kill stray Apptainer/Singularity sandbox processes and purge their overlay dirs.

    When a trajectory hits the hard timeout we call ``ray.cancel(force=True)``
    which SIGKILLs the Ray worker, but the Apptainer/Singularity subprocess it
    spawned (plus its pytest grandchild) is not necessarily reaped. Across
    steps these orphans accumulate and push the host toward OOM — sacct shows
    MaxRSS hitting the ~750GB cap regardless of whether the job ran 1h or 9h.

    This helper is invoked at step boundaries, right alongside the existing
    ``ray.shutdown()``/``ray.init()`` reset. Safe to call even if nothing is
    orphaned: pkill returns 1 when there are no matches, which we ignore.
    """
    user = os.environ.get("USER", "")
    if user:
        for pattern in ("apptainer", "singularity", "starter-suid"):
            subprocess.run(
                ["pkill", "-KILL", "-u", user, "-f", pattern],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    tmpdir = os.environ.get("TMPDIR") or "/tmp"
    for prefix in ("rootfs-", "apptainer-", "singularity-"):
        try:
            for entry in os.listdir(tmpdir):
                if entry.startswith(prefix):
                    path = os.path.join(tmpdir, entry)
                    try:
                        if os.path.isdir(path):
                            shutil.rmtree(path, ignore_errors=True)
                        else:
                            os.unlink(path)
                    except OSError:
                        pass
        except OSError:
            pass


# ---------------------------------------------------------------------------
# OpenAI-compatible proxy backed by Tinker SamplingClient
# ---------------------------------------------------------------------------
class TinkerOpenAIProxy:
    """Minimal OpenAI-compatible HTTP server that forwards to Tinker SamplingClient."""

    def __init__(self, tokenizer, host: str = "127.0.0.1", port: int = 8001, max_context_length: int = 65536):
        self.tokenizer = tokenizer
        self.host = host
        self.port = port
        self.max_context_length = max_context_length
        self.sampling_client = None  # Updated after each training step
        self._app = web.Application()
        self._app.router.add_post("/v1/chat/completions", self._chat_completions)
        self._app.router.add_get("/v1/models", self._list_models)
        self._runner = None
        self._model_name = ""
        # Cache: decoded response text → FIFO queue of (tokens, logprobs) tuples.
        # Populated in _chat_completions, consumed by collect_trajectories to
        # recover π_old for each sampled assistant message. FIFO-by-text is an
        # approximation (collisions on identical assistant output are possible
        # in principle but vanishingly rare for long agent responses).
        self.logprob_cache: "defaultdict[str, deque]" = defaultdict(deque)

    def update_sampling_client(self, client, model_name: str = ""):
        """Swap in a new SamplingClient after weight updates."""
        self.sampling_client = client
        self._model_name = model_name

    def reset_logprob_cache(self):
        """Clear the sampling-logprob cache before a new rollout phase."""
        self.logprob_cache.clear()

    def pop_logprobs(self, text: str) -> Optional[List[float]]:
        """FIFO-pop the sampling logprobs for an assistant response text."""
        q = self.logprob_cache.get(text)
        if q:
            return q.popleft()
        return None

    async def _chat_completions(self, request):
        if self.sampling_client is None:
            return web.json_response({"error": "Sampling client not ready"}, status=503)

        data = await request.json()
        messages = data.get("messages", [])
        temperature = data.get("temperature", 1.0)
        top_p = data.get("top_p", 1.0)
        max_tokens = data.get("max_tokens", 4096)

        # Apply chat template to convert messages to token IDs
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        original_prompt_len = len(input_ids)

        # Reserve room for the generation, then enforce a head-preserving
        # truncation: always keep the first system+user messages (the task
        # instructions and PR description) and drop from the middle by tail-
        # keeping the most recent turns. Dropping the head would lose the
        # system prompt and the agent stops following the response format.
        budget = self.max_context_length - max_tokens
        if budget > 0 and original_prompt_len > budget:
            head_msgs = messages[:2] if len(messages) >= 2 else messages[:1]
            head_ids = self.tokenizer.apply_chat_template(
                head_msgs, add_generation_prompt=False, tokenize=True
            )
            head_len = len(head_ids)
            tail_budget = budget - head_len
            if tail_budget > 0:
                input_ids = list(head_ids) + list(input_ids[-tail_budget:])
            else:
                # Head alone already exceeds the budget — keep just the head,
                # truncated from its tail. Generation will be clamped below.
                input_ids = list(head_ids[:budget])
            logger.warning(
                f"Prompt ({original_prompt_len} tokens) exceeded budget {budget}; "
                f"head-preserving truncation to {len(input_ids)} tokens "
                f"(head={head_len}, tail_kept={max(0, len(input_ids) - head_len)})"
            )

        # Clamp max_tokens so that prompt + generation fits in the context window
        prompt_len = len(input_ids)
        headroom = self.max_context_length - prompt_len
        if max_tokens > headroom:
            logger.warning(
                f"Clamping max_tokens from {max_tokens} to {headroom} "
                f"(prompt={prompt_len}, context={self.max_context_length})"
            )
            max_tokens = headroom

        # Call Tinker SamplingClient
        prompt = types.ModelInput.from_ints(tokens=input_ids)
        sampling_params = types.SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        try:
            output = await self.sampling_client.sample_async(
                prompt=prompt, num_samples=1, sampling_params=sampling_params
            )
            seq = output.sequences[0]
            response_tokens = seq.tokens
            response_logprobs = seq.logprobs  # π_old, per-token; may be None
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

            # Stash π_old so collect_trajectories can recover it later.
            if response_logprobs is not None:
                self.logprob_cache[response_text].append(list(response_logprobs))

            finish_reason = "stop" if (
                len(response_tokens) > 0 and response_tokens[-1] == self.tokenizer.eos_token_id
            ) else "length"
        except Exception as e:
            logger.error(f"Tinker sampling error: {e}")
            return web.json_response({"error": str(e)}, status=500)

        return web.json_response({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "model": self._model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(response_tokens),
                "total_tokens": len(input_ids) + len(response_tokens),
            },
        })

    async def _list_models(self, request):
        return web.json_response({
            "data": [{"id": self._model_name, "object": "model"}]
        })

    async def start(self):
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"Tinker OpenAI proxy started at http://{self.host}:{self.port}/v1")

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()


# ---------------------------------------------------------------------------
# GRPO advantage computation (same as tinker_train.py)
# ---------------------------------------------------------------------------
def compute_advantages_grpo(
    rewards: List[float],
    group_size: int,
    normalize: bool = True,
    dynamic_sampling: bool = False,
) -> tuple[List[float], List[bool]]:
    """Compute GRPO advantages. Returns (advantages, keep_mask).

    When dynamic_sampling is True (DAPO), samples in groups with zero reward
    variance are marked keep=False so the caller can drop them.
    """
    rewards_arr = np.array(rewards)
    n = len(rewards_arr)
    advantages = [0.0] * n
    keep_mask = [True] * n

    def _process_group(start: int, end: int):
        group = rewards_arr[start:end]
        if dynamic_sampling and group.std() < 1e-8:
            for j in range(start, end):
                keep_mask[j] = False
            return
        group_mean = group.mean()
        for j, val in enumerate(group):
            advantages[start + j] = float(val - group_mean)

    n_groups = n // group_size
    for i in range(n_groups):
        _process_group(i * group_size, (i + 1) * group_size)
    if n % group_size > 0:
        _process_group(n_groups * group_size, n)

    if normalize:
        kept = [advantages[j] for j in range(n) if keep_mask[j]]
        if len(kept) > 1:
            mean = float(np.mean(kept))
            std = float(np.std(kept))
            if std > 1e-8:
                advantages = [(a - mean) / (std + 1e-8) for a in advantages]
            else:
                advantages = [0.0] * n

    return advantages, keep_mask


# ---------------------------------------------------------------------------
# Trajectory collection using existing init_and_run
# ---------------------------------------------------------------------------
async def collect_trajectories(
    instances: list,
    litellm_model_name: str,
    sweagent_config: dict,
    generator_cfg: MiniSWEGeneratorConfig,
    n_samples: int,
    tokenizer,
    global_step: int,
    training_phase: TrainingPhase,
    proxy: Optional["TinkerOpenAIProxy"] = None,
):
    """Run the agent loop for all instances, return trajectories.

    Returns (prompt_ids, response_ids, rewards, loss_masks, rollout_logprobs).
    rollout_logprobs[i] is a list aligned with response_ids[i], or None if
    π_old could not be recovered for that trajectory (fallback to zeros).
    """
    # Reset the proxy's π_old cache so this rollout phase starts clean — no
    # stale entries from a previous phase can leak into FIFO pops.
    if proxy is not None:
        proxy.reset_logprob_cache()

    futures = []
    trajectory_ids = []

    for inst_idx, instance in enumerate(instances):
        if isinstance(instance, str):
            instance = json.loads(instance)
        for rep_id in range(n_samples):
            instance_id = instance.get("instance_id", str(inst_idx))
            tid = TrajectoryID(instance_id=instance_id, repetition_id=rep_id)
            batch_metadata = BatchMetadata(global_step=global_step, training_phase=training_phase)
            data_source = instance.get("data_source", "swe-gym")

            fut = init_and_run.remote(
                instance,
                litellm_model_name,
                sweagent_config,
                generator_cfg,
                data_source,
                {},  # sampling_params (handled by proxy)
                tid,
                global_step,
                training_phase,
            )
            futures.append(fut)
            trajectory_ids.append(tid)

    # Per-trajectory hard wall-clock timeout. A single hung Apptainer sandbox
    # must not be allowed to stall the whole step. On timeout we ray.cancel
    # (force=True → SIGKILL the worker, which kills its sandbox subprocess)
    # and drop the trajectory.
    async def _await_with_timeout(obj_ref, tid):
        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(obj_ref.future()),
                timeout=TRAJECTORY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Trajectory {tid.instance_id} (rep {tid.repetition_id}) exceeded "
                f"{TRAJECTORY_TIMEOUT_SECONDS}s hard timeout; cancelling Ray task."
            )
            try:
                ray.cancel(obj_ref, force=True)
            except Exception as cancel_err:
                logger.debug(f"Error cancelling ray task for {tid.instance_id}: {cancel_err}")
            return None

    # Gather results — tolerate individual task failures and per-trajectory timeouts
    results = await asyncio.gather(
        *[_await_with_timeout(f, tid) for f, tid in zip(futures, trajectory_ids)],
        return_exceptions=True,
    )

    prompt_token_ids_list = []
    response_ids_list = []
    rewards_list = []
    loss_masks_list = []
    rollout_logprobs_list: List[Optional[List[float]]] = []

    for r in results:
        if isinstance(r, BaseException):
            logger.warning(f"Ray task failed: {r}")
            continue
        if r is None:
            # Trajectory was cancelled by the per-trajectory hard timeout above.
            continue
        messages, reward, error = r
        if not messages:
            continue

        clean_messages = _clean_messages_for_tokenizer(messages)
        if len(clean_messages) < 3:
            continue

        # First 2 messages are system + user (prompt)
        initial_ids = tokenizer.apply_chat_template(
            clean_messages[:2], add_generation_prompt=False, tokenize=True
        )

        # Response messages
        response_messages = clean_messages[2:]
        # Remove trailing user messages
        last_idx = len(response_messages) - 1
        while last_idx >= 0 and response_messages[last_idx]["role"] == "user":
            last_idx -= 1
        if last_idx < 0:
            continue
        response_messages = response_messages[: last_idx + 1]

        # Recover π_old for each assistant message from the proxy cache.
        assistant_logprobs_for_traj: Optional[List[List[float]]] = None
        if proxy is not None:
            collected: List[List[float]] = []
            any_missing = False
            for msg in response_messages:
                if msg["role"] != "assistant":
                    continue
                lp = proxy.pop_logprobs(msg["content"])
                if lp is None:
                    any_missing = True
                    break
                collected.append(lp)
            if not any_missing and collected:
                assistant_logprobs_for_traj = collected

        # Try with logprobs first; fall back to None on any alignment failure
        # (e.g. retokenized length != Tinker's sampled length).
        rollout_logprobs: Optional[List[float]] = None
        try:
            response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
                response_messages, tokenizer, assistant_logprobs=assistant_logprobs_for_traj
            )
        except (ValueError, AssertionError) as e:
            logger.warning(
                f"π_old alignment failed, falling back to zero logprobs for this trajectory: {e}"
            )
            response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
                response_messages, tokenizer, assistant_logprobs=None
            )
            rollout_logprobs = None

        # Truncate if needed
        max_response_tokens = (
            generator_cfg.sampling_params.max_generate_length
            + generator_cfg.max_input_length
            - len(initial_ids)
        )
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        if rollout_logprobs is not None:
            rollout_logprobs = rollout_logprobs[:max_response_tokens]

        prompt_token_ids_list.append(initial_ids)
        response_ids_list.append(response_ids)
        rewards_list.append(float(reward))
        loss_masks_list.append(loss_mask)
        rollout_logprobs_list.append(rollout_logprobs)

    return prompt_token_ids_list, response_ids_list, rewards_list, loss_masks_list, rollout_logprobs_list


# ---------------------------------------------------------------------------
# Convert trajectories to Tinker Datums
# ---------------------------------------------------------------------------
def make_tinker_datums(
    prompt_token_ids_list,
    response_ids_list,
    rewards_list,
    loss_masks_list,
    advantages_list,
    rollout_logprobs_list: Optional[List[Optional[List[float]]]] = None,
):
    datums = []
    for idx in range(len(response_ids_list)):
        full_seq = prompt_token_ids_list[idx] + response_ids_list[idx]
        prompt_len = len(prompt_token_ids_list[idx])
        response_len = len(response_ids_list[idx])

        target_tokens = full_seq[1:]
        # target_tokens[i] = full_seq[i+1]. The response portion starts at
        # target index prompt_len - 1 (where target = full_seq[prompt_len] =
        # response_ids[0]). Prompt positions get 0.0 because they are masked
        # out of the loss anyway.
        logprobs = [0.0] * (len(full_seq) - 1)
        if rollout_logprobs_list is not None and rollout_logprobs_list[idx] is not None:
            rl = rollout_logprobs_list[idx]
            n = min(len(rl), response_len, len(logprobs) - (prompt_len - 1))
            start = prompt_len - 1
            for i in range(n):
                logprobs[start + i] = float(rl[i])

        # Mask: 0 for prompt, loss_mask for response
        mask = [0] * prompt_len + loss_masks_list[idx]

        # Advantages
        adv_value = advantages_list[idx]
        advantages = torch.zeros(len(full_seq))
        for i in range(prompt_len, len(full_seq)):
            if i < len(mask) and mask[i] > 0:
                advantages[i] = adv_value
        advantages = advantages[1:]
        mask = mask[1:]

        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=full_seq[:-1]),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": TensorData.from_torch(torch.tensor(logprobs)),
                "advantages": TensorData.from_torch(advantages),
            },
        )
        datums.append(datum)

    return datums


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
def save_checkpoint_record(ckpt_path: str, step: int, state_path: str, sampler_path: str = ""):
    record = {"step": step, "state_path": state_path, "sampler_path": sampler_path}
    ckpt_file = os.path.join(ckpt_path, "tinker_checkpoints.jsonl")
    with open(ckpt_file, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(f"Saved checkpoint record: step={step}, state_path={state_path}")


def load_latest_checkpoint(ckpt_path: str):
    ckpt_file = os.path.join(ckpt_path, "tinker_checkpoints.jsonl")
    if not os.path.exists(ckpt_file):
        return None
    with open(ckpt_file) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        return None
    return max(records, key=lambda r: r["step"])


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
async def train(config: TinkerMiniSWEConfig):
    # Load dataset
    print("[TRACE] Loading datasets...", flush=True)
    train_dataset = load_dataset("parquet", data_files=config.train_data)["train"]
    val_dataset = load_dataset("parquet", data_files=config.val_data)["train"]
    print(f"[TRACE] Datasets loaded: train={len(train_dataset)}, val={len(val_dataset)}", flush=True)

    # Tokenizer
    print(f"[TRACE] Loading tokenizer for {config.model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    print("[TRACE] Tokenizer loaded.", flush=True)

    # Model name for LiteLLM (used by minisweagent via OPENAI_BASE_URL)
    litellm_model_name = f"openai/{config.model_name}"

    # SWE-agent config (same yaml as original)
    sweagent_config = yaml.safe_load(Path(config.miniswe_config_path).read_text())

    # Generator config (reuse dataclass for compatibility with init_and_run)
    from skyrl.train.config import SamplingParams, InferenceEngineConfig
    generator_cfg = MiniSWEGeneratorConfig(
        miniswe_config_path=config.miniswe_config_path,
        miniswe_traj_dir=config.traj_dir,
    )
    generator_cfg.sampling_params = SamplingParams(max_generate_length=config.max_generate_length)
    generator_cfg.max_input_length = config.max_input_length
    generator_cfg.inference_engine = InferenceEngineConfig()

    # WandB
    print("[TRACE] Initializing WandB...", flush=True)
    wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config))
    print("[TRACE] WandB initialized.", flush=True)

    # --- Tinker setup ---
    TINKER_CONNECT_TIMEOUT = 300  # seconds
    service_client = tinker.ServiceClient()
    adam_params = types.AdamParams(learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    # --- Determine resume state ---
    resume_step = 0
    # last_state_path holds the tinker:// path to restore weights+optimizer from.
    # None means start fresh (create_lora_training_client).
    last_state_path = None

    if config.resume_from:
        last_state_path = config.resume_from
    elif config.ckpt_path:
        ckpt = load_latest_checkpoint(config.ckpt_path)
        if ckpt:
            last_state_path = ckpt["state_path"]
            resume_step = ckpt["step"] + 1
            logger.info(f"Will resume from step {ckpt['step']}, state={last_state_path}")

    # --- Start OpenAI proxy ---
    # Cap proxy context at the training-time budget so rollouts can't generate
    # sequences longer than what `make_tinker_datums` will accept. The model's
    # `model_max_length` (e.g. 131072) is irrelevant here.
    max_ctx = config.max_input_length + config.max_generate_length
    proxy = TinkerOpenAIProxy(tokenizer, config.proxy_host, config.proxy_port, max_context_length=max_ctx)
    await proxy.start()

    # --- Training loop ---
    os.makedirs(config.ckpt_path, exist_ok=True)
    batch_size = config.train_batch_size
    n_train = len(train_dataset)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size  # ceil division
    total_steps = config.max_epochs * steps_per_epoch
    logger.info(
        f"Epoch-based training: {n_train} instances, batch_size={batch_size}, "
        f"steps_per_epoch={steps_per_epoch}, max_epochs={config.max_epochs}, total_steps={total_steps}"
    )

    # Epoch-based sampling: shuffle at start of each epoch, iterate sequentially
    dataset_indices = list(range(n_train))
    step = 0

    if resume_step > 0:
        logger.info(f"Resuming: will skip steps 1-{resume_step - 1}, start from step {resume_step}")

    for epoch in range(1, config.max_epochs + 1):
        # Deterministic shuffle per epoch so resume reproduces the same data order
        np.random.seed(epoch)
        np.random.shuffle(dataset_indices)
        logger.info(f"{'=' * 20} Epoch {epoch} {'=' * 20}")

        for epoch_step in range(steps_per_epoch):
            step += 1

            # Skip already-completed steps on resume
            if step < resume_step:
                continue

            logger.info(f"{'=' * 20} Epoch {epoch} Step {step} ({epoch_step+1}/{steps_per_epoch}) {'=' * 20}")
            metrics = {"step": step, "epoch": epoch}
            step_start = time.time()

            # ---- Phase 1: Prepare sampler weights (short-lived training client) ----
            logger.info("Creating Tinker TrainingClient for weight export...")
            try:
                if last_state_path:
                    training_client = await asyncio.wait_for(
                        service_client.create_training_client_from_state_with_optimizer_async(
                            path=last_state_path
                        ),
                        timeout=TINKER_CONNECT_TIMEOUT,
                    )
                    logger.info(f"TrainingClient restored from {last_state_path}")
                else:
                    training_client = await asyncio.wait_for(
                        service_client.create_lora_training_client_async(
                            base_model=config.model_name, rank=config.lora_rank
                        ),
                        timeout=TINKER_CONNECT_TIMEOUT,
                    )
                    logger.info("TrainingClient created (fresh).")
            except asyncio.TimeoutError:
                logger.error(f"Tinker connection timed out after {TINKER_CONNECT_TIMEOUT}s.")
                wandb.finish(exit_code=1)
                raise SystemExit(1)

            # Save weights for sampler + save full state (so we can release the client)
            sampler_result = await training_client.save_weights_for_sampler_async(name=f"step{step:06d}_sampler")
            sampler_path = (await sampler_result.result_async()).path

            state_result = await training_client.save_state_async(f"step{step:06d}_pre")
            pre_state_path = (await state_result.result_async()).path
            logger.info(f"Saved pre-rollout state: {pre_state_path}")

            # Release the training client — no idle session during rollout
            del training_client
            logger.info("TrainingClient released before rollout.")

            # ---- Phase 2: Rollout (only SamplingClient alive, no timeout risk) ----
            sampling_client = await service_client.create_sampling_client_async(model_path=sampler_path)
            proxy.update_sampling_client(sampling_client, config.model_name)
            logger.info("SamplingClient ready. Starting rollout...")

            # Optional eval
            if config.eval_every > 0 and (step % config.eval_every == 0) and (step > 0 or config.eval_before_train):
                logger.info(f"Running evaluation at step {step}")
                eval_instances = [val_dataset[i] for i in range(min(config.eval_batch_size, len(val_dataset)))]
                _, _, eval_rewards, _, _ = await collect_trajectories(
                    eval_instances, litellm_model_name, sweagent_config, generator_cfg,
                    n_samples=1, tokenizer=tokenizer, global_step=step, training_phase="eval",
                    proxy=proxy,
                )
                # Kill all Ray workers to free memory before next phase
                logger.info("Resetting Ray workers after eval rollout...")
                ray.shutdown()
                # Kill orphan sandbox subprocesses left behind by cancelled
                # Ray workers before they can accumulate across steps.
                kill_orphan_sandboxes()
                ray.init(ignore_reinit_error=True)
                if eval_rewards:
                    metrics["eval/reward_mean"] = np.mean(eval_rewards)
                    metrics["eval/resolve_rate"] = np.mean([1 if r > 0 else 0 for r in eval_rewards])
                    logger.info(f"Eval reward: {metrics['eval/reward_mean']:.4f}, resolve rate: {metrics['eval/resolve_rate']:.4f}")

            # Sample training batch — sequential slice within the shuffled epoch
            start_idx = epoch_step * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            batch_indices = dataset_indices[start_idx:end_idx]
            train_instances = [train_dataset[int(i)] for i in batch_indices]

            # Collect trajectories
            actual_batch = len(batch_indices)
            logger.info(f"Collecting {actual_batch * config.n_samples_per_prompt} trajectories...")
            t0 = time.time()
            prompt_ids, response_ids, rewards, loss_masks, rollout_logprobs = await collect_trajectories(
                train_instances, litellm_model_name, sweagent_config, generator_cfg,
                n_samples=config.n_samples_per_prompt, tokenizer=tokenizer,
                global_step=step, training_phase="train",
                proxy=proxy,
            )
            metrics["time/rollout"] = time.time() - t0
            metrics["rollout/logprobs_captured_frac"] = (
                sum(1 for lp in rollout_logprobs if lp is not None) / max(1, len(rollout_logprobs))
            )

            # Kill all Ray workers to free memory before training phase
            logger.info("Resetting Ray workers after train rollout...")
            ray.shutdown()
            # Kill orphan sandbox subprocesses left behind by cancelled
            # Ray workers before they can accumulate across steps.
            kill_orphan_sandboxes()
            ray.init(ignore_reinit_error=True)

            if not rewards:
                logger.warning(f"No valid trajectories at step {step}, skipping training")
                last_state_path = pre_state_path  # keep using the same state next step
                wandb.log(metrics, step=step, commit=True)
                continue

            metrics["reward/mean"] = np.mean(rewards)
            metrics["reward/max"] = np.max(rewards)
            metrics["reward/min"] = np.min(rewards)
            metrics["reward/resolve_rate"] = np.mean([1 if r > 0 else 0 for r in rewards])
            metrics["rollout/num_trajectories"] = len(rewards)

            # Compute GRPO advantages (+ DAPO dynamic sampling mask)
            advantages, keep_mask = compute_advantages_grpo(
                rewards,
                group_size=config.n_samples_per_prompt,
                normalize=config.normalize_advantages,
                dynamic_sampling=config.dynamic_sampling,
            )
            metrics["advantage/mean"] = float(np.mean(advantages))
            metrics["advantage/std"] = float(np.std(advantages))
            metrics["rollout/num_dropped_dynamic_sampling"] = int(sum(1 for k in keep_mask if not k))

            # Drop flat-reward groups (DAPO dynamic sampling)
            if not all(keep_mask):
                prompt_ids = [p for p, k in zip(prompt_ids, keep_mask) if k]
                response_ids = [r for r, k in zip(response_ids, keep_mask) if k]
                rewards = [r for r, k in zip(rewards, keep_mask) if k]
                loss_masks = [m for m, k in zip(loss_masks, keep_mask) if k]
                advantages = [a for a, k in zip(advantages, keep_mask) if k]
                rollout_logprobs = [lp for lp, k in zip(rollout_logprobs, keep_mask) if k]

            if not advantages:
                logger.warning(f"All groups filtered by dynamic sampling at step {step}, skipping training")
                last_state_path = pre_state_path
                wandb.log(metrics, step=step, commit=True)
                continue

            # Convert to Tinker datums (with π_old from the rollout sampler)
            datums = make_tinker_datums(
                prompt_ids, response_ids, rewards, loss_masks, advantages,
                rollout_logprobs_list=rollout_logprobs,
            )

            # ---- Phase 3: Training (new short-lived training client) ----
            logger.info("Recreating TrainingClient for training phase...")
            try:
                training_client = await asyncio.wait_for(
                    service_client.create_training_client_from_state_with_optimizer_async(
                        path=pre_state_path
                    ),
                    timeout=TINKER_CONNECT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.error(f"Tinker connection timed out after {TINKER_CONNECT_TIMEOUT}s.")
                wandb.finish(exit_code=1)
                raise SystemExit(1)
            logger.info("TrainingClient restored for training.")

            logger.info(
                f"Training on {len(datums)} datums for {config.ppo_epochs} PPO epoch(s)..."
            )
            t0 = time.time()
            loss_fn_config = {}
            if config.loss_fn == "ppo":
                # DAPO Clip-Higher: asymmetric PPO clip bounds on the ratio π_θ/π_old
                loss_fn_config = {
                    "clip_low_threshold": 1.0 - config.clip_epsilon_low,
                    "clip_high_threshold": 1.0 + config.clip_epsilon_high,
                }
            # Multi-epoch PPO on the same batch. π_old in the datums stays
            # frozen (captured during rollout); π_θ drifts as we take more
            # optim steps, so ratio = π_θ/π_old diverges from 1 and clip
            # (including DAPO Clip-Higher) actually starts to bite.
            for ppo_epoch in range(config.ppo_epochs):
                fwd_bwd_future = await training_client.forward_backward_async(
                    datums, loss_fn=config.loss_fn, loss_fn_config=loss_fn_config
                )
                optim_future = await training_client.optim_step_async(adam_params)
                await fwd_bwd_future.result_async()
                await optim_future.result_async()
            metrics["time/train"] = time.time() - t0

            # Save state after training (weights + optimizer for next step)
            state_result = await training_client.save_state_async(f"step{step:06d}_post")
            last_state_path = (await state_result.result_async()).path
            logger.info(f"Saved post-training state: {last_state_path}")

            # Persist checkpoint record for resume across job restarts
            save_checkpoint_record(config.ckpt_path, step, last_state_path, sampler_path)

            # Release training client
            del training_client
            logger.info("TrainingClient released after training.")

            metrics["time/step_total"] = time.time() - step_start
            logger.info(f"Step {step} metrics: reward={metrics['reward/mean']:.4f}, time={metrics['time/step_total']:.1f}s")
            wandb.log(metrics, step=step, commit=True)

    await proxy.stop()
    wandb.finish()
    logger.info("Training completed!")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def parse_config_from_argv() -> TinkerMiniSWEConfig:
    """Parse key=value arguments from command line."""
    config = TinkerMiniSWEConfig()
    for arg in sys.argv[1:]:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        # Strip quotes
        value = value.strip("'\"")
        if hasattr(config, key):
            field_type = type(getattr(config, key))
            if field_type == bool:
                setattr(config, key, value.lower() in ("true", "1", "yes"))
            elif field_type == int:
                setattr(config, key, int(value))
            elif field_type == float:
                setattr(config, key, float(value))
            else:
                setattr(config, key, value)
    return config


def main():
    config = parse_config_from_argv()

    # Initialize Ray for init_and_run remote tasks
    ray.init(ignore_reinit_error=True)

    asyncio.run(train(config))


if __name__ == "__main__":
    main()
