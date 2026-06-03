#!/bin/bash
#SBATCH --job-name=qwen2_5_logits_v3
#SBATCH --output=logs/qwen2_5_logits_v3.out
#SBATCH --error=logs/qwen2_5_logits_v3.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=ailab
set -x

# Needed for outbound connectivity (e.g., WandB) on this cluster
module load proxy/default
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

export WANDB_DIR="${SLURM_TMPDIR:-$PWD}/wandb"
export WANDB_CACHE_DIR="${SLURM_TMPDIR:-$PWD}/.cache/wandb"
export WANDB_CONFIG_DIR="${SLURM_TMPDIR:-$PWD}/.config/wandb"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

MODEL_PATH=<PATH_TO_Qwen2.5-VL-7B-Instruct>
TRAIN_FILES=<PATH_TO_TRAIN_DATA>.jsonl
VAL_FILES=<PATH_TO_VAL_DATA>.jsonl

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_logits_v3 \
    trainer.n_gpus_per_node=4 \
    trainer.logger=[file,wandb] \
    data.rollout_batch_size=256 \
    data.max_prompt_length=16384 \
    data.max_response_length=4096 \
    worker.rollout.max_num_batched_tokens=24576 \
    worker.actor.global_batch_size=64 \
    data.choose_image_batch_size=32 \
    algorithm.choose_image_coef=0.005 \
    algorithm.choose_image_clip=5.0 \

echo "End time: $(date)"
