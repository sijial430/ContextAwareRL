#!/bin/bash
#SBATCH --job-name=eval-swe-lite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_swe_lite_%j.out
#SBATCH --error=logs/eval_swe_lite_%j.err

set -eo pipefail

module load proxy/default
export no_proxy=localhost,127.0.0.1

# ============================================================
# SWE-bench Lite Evaluation
# ============================================================

PREDICTIONS_PATH="${PREDICTIONS_PATH:?PREDICTIONS_PATH not set}"
WORK_DIR="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${RUN_ID:-swe-lite-eval}"
SANDBOX_DIR="${SANDBOX_DIR:?SANDBOX_DIR not set}"
SANDBOX_WORK_DIR="${SANDBOX_WORK_DIR:-${WORK_DIR}/logs/sandbox_workdirs_lite_eval}"

conda activate "${CONDA_ENV:-swebench}"

cd "$WORK_DIR"
mkdir -p logs
mkdir -p "$SANDBOX_DIR" "$SANDBOX_WORK_DIR"
export SWEBENCH_SANDBOX_DIR="$SANDBOX_DIR"
export SWEBENCH_SANDBOX_WORK_DIR="$SANDBOX_WORK_DIR"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Predictions: $PREDICTIONS_PATH"
echo "Run ID: $RUN_ID"
echo "Sandbox dir: $SWEBENCH_SANDBOX_DIR"
echo "Dataset:    SWE-bench/SWE-bench_Lite (test)"
echo "=========================================="

if [ ! -f "$PREDICTIONS_PATH" ]; then
    echo "ERROR: Predictions file not found: $PREDICTIONS_PATH"
    exit 1
fi

python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --predictions_path "$PREDICTIONS_PATH" \
    --max_workers 8 \
    --run_id "$RUN_ID" \
    --namespace none \
    --timeout 1800 \
    --cache_level instance

echo "Evaluation complete!"
echo "Results in: logs/run_evaluation/${RUN_ID}/"
