#!/bin/bash
#SBATCH --job-name=qwen3_8B_instruct_aug_logits
#SBATCH --output=logs/qwen3_8B_instruct_aug_logits.out
#SBATCH --error=logs/qwen3_8B_instruct_aug_logits.err
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
export WANDB_API_KEY="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"

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

export WANDB_DIR="${SLURM_TMPDIR:-$PWD}/wandb"
export WANDB_CACHE_DIR="${SLURM_TMPDIR:-$PWD}/.cache/wandb"
export WANDB_CONFIG_DIR="${SLURM_TMPDIR:-$PWD}/.config/wandb"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

MODEL_PATH=/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/Qwen3-VL-8B-Instruct
# Use Thinking model's chat template so outputs follow the <think>...</think> format
THINKING_CHAT_TEMPLATE=/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/Qwen3-VL-8B-Thinking/chat_template.json

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/DataGen/DataSource/rl_gt_qwen3_aug.jsonl\
    data.val_files=/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/DataGen/DataSource/val_gt_v3.jsonl \
    data.override_chat_template=${THINKING_CHAT_TEMPLATE} \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_8B_instruct_aug_logits \
    trainer.n_gpus_per_node=4 \
    trainer.logger=[console,file,wandb] \
    data.rollout_batch_size=256 \
    data.max_prompt_length=16384 \
    data.max_response_length=4096 \
    worker.rollout.max_num_batched_tokens=24576 \
    worker.actor.global_batch_size=64 \
    worker.rollout.limit_images=10 \
    worker.rollout.val_override_config.temperature=0.6 \
    worker.rollout.val_override_config.top_p=0.95 \
    worker.rollout.val_override_config.n=1 \
    data.choose_image_batch_size=32 \
    algorithm.choose_image_coef=0.001 \
    algorithm.choose_image_clip=5.0 \

echo "End time: $(date)"
