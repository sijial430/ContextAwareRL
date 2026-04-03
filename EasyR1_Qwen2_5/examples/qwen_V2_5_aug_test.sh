#!/bin/bash
#SBATCH --job-name=ci_test_small
#SBATCH --output=logs/ci_test_small.out
#SBATCH --error=logs/ci_test_small.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --constraint=gpu80
set -x

module load proxy/default
export WANDB_API_KEY="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"
export OPENAI_API_KEY="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"

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
    data.train_files=/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/DataGen/DataSource/test_small_200r_40ci.jsonl \
    data.val_files=/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/DataGen/DataSource/val_gt.jsonl \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/origin.py:compute_score \
    trainer.experiment_name=ci_test_small \
    trainer.project_name=easy_r1_debug \
    trainer.n_gpus_per_node=4 \
    trainer.logger=[file,wandb] \
    trainer.max_steps=30 \
    trainer.val_freq=-1 \
    trainer.save_freq=-1 \
    trainer.val_before_train=false \
    data.rollout_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    worker.rollout.n=4 \
    worker.rollout.max_num_batched_tokens=24576 \
    worker.rollout.max_model_len=10240 \
    worker.actor.global_batch_size=32 \
    data.choose_image_batch_size=16 \
    algorithm.choose_image_coef=0.01 \
    algorithm.choose_image_clip=5.0 \

echo "End time: $(date)"
