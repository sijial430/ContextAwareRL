#!/bin/bash
#SBATCH --job-name=tinker_swe_qwen35b
#SBATCH --output=logs/tinker_swe_qwen35b.out
#SBATCH --error=logs/tinker_swe_qwen35b.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --partition=ailab
set -x

# =============================================================================
# Tinker RL Training for SWE-Bench with Qwen3.5-35B-A3B
# =============================================================================
# Uses Tinker for remote training + inference (no local GPU needed for model).
# Settings aligned with run_mini_swe_agentforge8B.sh where applicable.
# =============================================================================

# Cluster connectivity
module load proxy/default
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

# API Keys — set these before running
export TINKER_API_KEY="YOUR_TINKER_API_KEY"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Data paths — same data as run_mini_swe_agentforge8B.sh
DATA_DIR="/path/to/swe_gym_data"
DATASET_FILE="${DATA_DIR}/train_valid_light.parquet"
EVAL_DATASET_FILE="${DATA_DIR}/validation_valid.parquet"

# Output directory
OUTPUT_DIR="logs/tinker_swe_qwen35b"
mkdir -p "$OUTPUT_DIR"

# Model configuration
MODEL_NAME="Qwen/Qwen3.5-35B-A3B"
LORA_RANK=32

# Training hyperparameters — aligned with run_mini_swe_agentforge8B.sh
BATCH_SIZE=32           # Same: trainer.train_batch_size=32
EVAL_BATCH_SIZE=100     # Same: trainer.eval_batch_size=100
LEARNING_RATE=1.0e-6    # Same: trainer.policy.optimizer_config.lr=1.0e-6
MAX_STEPS=200           # ~2 epochs over the dataset
SAVE_EVERY=10           # Same: trainer.ckpt_interval=10
EVAL_EVERY=20           # Same: trainer.eval_interval=20

# RL configuration — aligned with GRPO settings
LOSS_FN="ppo"
GROUP_SIZE=8            # Same: generator.n_samples_per_prompt=8
NORMALIZE_ADVANTAGES="true"

# Logging
WANDB_PROJECT="mini_swe"
WANDB_NAME="tinker_swe_qwen35b_a3b_swe_gym"

# WandB temp dirs
export WANDB_DIR="${SLURM_TMPDIR:-$PWD}/wandb"
export WANDB_CACHE_DIR="${SLURM_TMPDIR:-$PWD}/.cache/wandb"
export WANDB_CONFIG_DIR="${SLURM_TMPDIR:-$PWD}/.config/wandb"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# Task configuration
TASK_YAML="./examples/run_tinker/tinker_swe_qwen35b.yaml"

echo "================================================"
echo "Tinker RL Training - SWE-Bench"
echo "================================================"
echo "Model: $MODEL_NAME"
echo "LoRA Rank: $LORA_RANK"
echo "Dataset: $DATASET_FILE"
echo "Task YAML: $TASK_YAML"
echo "Batch Size: $BATCH_SIZE"
echo "Group Size (GRPO): $GROUP_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Steps: $MAX_STEPS"
echo "Output: $OUTPUT_DIR"
echo "================================================"

# Run training via skyrl-agent's Tinker integration
cd skyrl-agent

python -m skyrl_agent.integrations.tinker.tinker_train \
    model_name="$MODEL_NAME" \
    skyrl_agent_task_yaml="$TASK_YAML" \
    dataset_file="$DATASET_FILE" \
    eval_dataset_file="$EVAL_DATASET_FILE" \
    batch_size="$BATCH_SIZE" \
    eval_batch_size="$EVAL_BATCH_SIZE" \
    learning_rate="$LEARNING_RATE" \
    lora_rank="$LORA_RANK" \
    max_steps="$MAX_STEPS" \
    save_every="$SAVE_EVERY" \
    eval_every="$EVAL_EVERY" \
    loss_fn="$LOSS_FN" \
    group_size="$GROUP_SIZE" \
    normalize_advantages="$NORMALIZE_ADVANTAGES" \
    wandb_project="$WANDB_PROJECT" \
    wandb_name="$WANDB_NAME" \
    log_dir="$OUTPUT_DIR" \
    "$@"

echo "================================================"
echo "Training completed!"
echo "End time: $(date)"
echo "================================================"
