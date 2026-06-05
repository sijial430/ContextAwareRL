#!/bin/bash
#SBATCH --job-name=lcb-v6-ctv2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=ailab
#SBATCH --output=logs/lcb_v6_ctv2_%j.out
#SBATCH --error=logs/lcb_v6_ctv2_%j.err

set -euo pipefail

module load proxy/default
export no_proxy=localhost,127.0.0.1

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH=<PATH_TO_YOUR_MODEL>
VERSION=v6
SEED=${SEED:-381}
TP=1
PYBIN=${PYBIN:-python}

cd "$PROJECT_DIR"
mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPUs:      $(nvidia-smi -L)"
echo "Model:     $MODEL_PATH"
echo "Version:   $VERSION"
echo "Seed:      $SEED"
echo "TP size:   $TP"
echo "=========================================="

# 1) Generate completions with vLLM
"$PYBIN" eval.py \
  --model "$MODEL_PATH" \
  --tensor_parallel_size "$TP" \
  --seed "$SEED" \
  --version "$VERSION" \
  --max_model_len 40960 \
  --gpu_memory_utilization 0.90 \
  --enforce_eager

# 2) Score completions
"$PYBIN" lcb_score.py \
  --model "$MODEL_PATH" \
  --tensor_parallel_size "$TP" \
  --seed "$SEED" \
  --version "$VERSION"

echo "[$(date)] Done."
echo "Results in: $PROJECT_DIR/outputs_tp${TP}_top_p0.95_seed${SEED}_${VERSION}/"
