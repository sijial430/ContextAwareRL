#!/usr/bin/env bash
#SBATCH --job-name=lmms_eval_qwen2_5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lmms_eval_qwen2_5_%j.out
#SBATCH --error=logs/lmms_eval_qwen2_5_%j.err
set -euo pipefail

# ── args: MODEL_PATH and RUN_NAME passed via --export ──
if [[ -z "${MODEL_PATH:-}" || -z "${RUN_NAME:-}" ]]; then
    echo "ERROR: MODEL_PATH and RUN_NAME must be set via --export"
    exit 1
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY not set}"

CONDA_ENV="${CONDA_ENV:-lmms-eval}"
WORK_DIR="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT_DIR=${WORK_DIR}/eval_results/${RUN_NAME}
# ── activate env ──
source activate "${CONDA_ENV}"
cd "${WORK_DIR}"

# ── use cached datasets ──
export HF_DATASETS_OFFLINE=1

# ── cache model outputs (responses) to skip re-running on same doc_id ──
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="${WORK_DIR}/cache"
mkdir -p "${LMMS_EVAL_HOME}"
echo "LMMS_EVAL_USE_CACHE=${LMMS_EVAL_USE_CACHE} LMMS_EVAL_HOME=${LMMS_EVAL_HOME}"

# ── print job info ──
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time: $(date)"
echo "Model: ${MODEL_PATH}"
echo "Run name: ${RUN_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Thinking: enabled"
echo "==========================================="


# ── benchmarks ──
TASKS="scienceqa_img,\
mmerealworld_lite,\
olympiadbench_OE_MM_physics_en_COMP,\
phyx_mc"



# ── run evaluation with thinking enabled (pass cache env so sbatch --export cannot strip it) ──
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained="${MODEL_PATH}",max_pixels=1003520,attn_implementation=sdpa,thinking=true \
    --tasks "${TASKS}" \
    --gen_kwargs max_new_tokens=8192 \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUTPUT_DIR}" \
    --verbosity DEBUG

echo "==========================================="
echo "End time: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
