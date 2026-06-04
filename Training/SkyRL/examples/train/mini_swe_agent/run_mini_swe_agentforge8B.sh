#!/bin/bash
#SBATCH --job-name=mini_swe_agentforge8B_scale_v2
#SBATCH --output=logs/mini_swe_agentforge8B_scale_v2.out
#SBATCH --error=logs/mini_swe_agentforge8B_scale_v2.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --partition=ailab
set -x

# Colocated GRPO + choose-trajectory DPO aux loss for Klear-AgentForge-8B-SFT
# on the SWE-Bench task. Uses 1 node with 4 GPUs.
#
# RL data: /path/to/swe_gym_data/train_valid_best.parquet
#   (multi-turn, sandbox-evaluated, reward = resolved)
# CT data: /path/to/ct_data/eval_prompts.jsonl
#   (single-turn A/B MCQ, DPO-sigmoid loss on " A" / " B" next-token logits)
#
# sbatch examples/train/mini_swe_agent/run_mini_swe_agentforge8B_ct.sh

# Needed for outbound connectivity (e.g., WandB) on this cluster
module load proxy/default
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export MSWEA_COST_TRACKING="ignore_errors"
# Sandbox images must be OUTSIDE the working dir, otherwise Ray packages them (can be 60GB+)
# Ray uses /tmp/ray (short path for Unix sockets)
export TMPDIR="/tmp/${USER}_miniswe_sandboxes"
mkdir -p "$TMPDIR"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export UV_OFFLINE=true

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

export WANDB_DIR="${SLURM_TMPDIR:-$PWD}/wandb"
export WANDB_CACHE_DIR="${SLURM_TMPDIR:-$PWD}/.cache/wandb"
export WANDB_CONFIG_DIR="${SLURM_TMPDIR:-$PWD}/.config/wandb"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# Load env vars BEFORE ray start so Ray workers inherit them
set -a
source examples/train/mini_swe_agent/.env.miniswe
set +a

# Clean up stale Ray session from previous runs and start fresh
uv run ray stop --force 2>/dev/null || true
uv run ray start --head --num-gpus=4 --temp-dir=/tmp/ray

GYM_DIR="/path/to/swe_gym_data"
SMITH_DIR="/path/to/swe_smith_data"
CT_DATA="/path/to/ct_data/eval_prompts.jsonl"
CKPT_PATH="ckpts/llm_mini_swe_agentforge8B_scale_v2"
EXPORT_PATH="exports/mini_swe_agentforge8B_scale_v2"
MINISWE_TRAJ_DIR="traj/mini_swe_agentforge8B_scale_v2"
mkdir -p "$EXPORT_PATH" "$MINISWE_TRAJ_DIR"

NUM_GPUS=4
NNODES=1
NUM_INFERENCE_ENGINES=2
TP_SIZE=2
LOGGER=wandb

# Use .venv directly instead of `uv run --isolated` to avoid:
# 1. Creating .tmp* dirs in uv builds cache (millions of files on GPFS)
# 2. Per-worker 231-package reinstalls that slow down Ray worker startup
.venv/bin/python -m examples.train.mini_swe_agent.main_mini_swe_ct \
  data.train_data="['$GYM_DIR/train_valid_best.parquet','$SMITH_DIR/train_valid_best.parquet']" \
  data.val_data="['$GYM_DIR/validation_valid.parquet']" \
  data.choose_trajectory_data="['$CT_DATA']" \
  data.choose_trajectory_batch_size=8 \
  data.choose_trajectory_max_prompt_length=32768 \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.clip_ratio_c=10.0 \
  trainer.algorithm.loss_reduction="token_mean" \
  trainer.algorithm.zero_variance_filter=false \
  trainer.algorithm.choose_trajectory_coef=0.01 \
  trainer.algorithm.choose_trajectory_clip=5.0 \
  trainer.algorithm.choose_trajectory_prefill='"The correct option is"' \
  trainer.algorithm.choose_trajectory_enable_thinking=false \
  trainer.policy.model.path="/path/to/Klear-AgentForge-8B-SFT" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  trainer.policy.sequence_parallel_size=1 \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  trainer.epochs=2 \
  trainer.eval_batch_size=100 \
  trainer.eval_before_train=false \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.dump_data_batch=true \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=8192 \
  generator.sampling_params.max_generate_length=4096 \
  generator.max_input_length=28672 \
  generator.max_turns=35 \
  trainer.policy.optimizer_config.lr=2.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.0005 \
  trainer.algorithm.kl_estimator_type=k3 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=True \
  generator.inference_engine.enable_http_endpoint=True \
  generator.inference_engine.http_endpoint_host='127.0.0.1' \
  generator.inference_engine.http_endpoint_port=8001 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=8 \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  trainer.logger="$LOGGER" \
  trainer.project_name="mini_swe" \
  trainer.run_name="mini_swe_agentforge8B_scale_v2" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.log_path="logs/skyrl-logs" \
  generator.miniswe_config_path="examples/train/mini_swe_agent/swebench.yaml" \
  generator.miniswe_traj_dir=$MINISWE_TRAJ_DIR \
  $@

echo "End time: $(date)"
