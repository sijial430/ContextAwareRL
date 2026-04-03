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
export WANDB_API_KEY="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"
export OPENAI_API_KEY="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"
# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

export WANDB_DIR="${SLURM_TMPDIR:-$PWD}/wandb"
export WANDB_CACHE_DIR="${SLURM_TMPDIR:-$PWD}/.cache/wandb"
export WANDB_CONFIG_DIR="${SLURM_TMPDIR:-$PWD}/.config/wandb"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

MODEL_PATH=/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/DataGen/DataSource/rl_gt_aug_v3.jsonl\
    data.val_files=/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/DataGen/DataSource/val_gt_v3.jsonl \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/origin.py:compute_score \
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
