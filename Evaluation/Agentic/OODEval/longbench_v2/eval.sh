#!/bin/bash
#SBATCH --job-name=lbv2-klear8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=ailab
#SBATCH --output=logs/lbv2_klear8b_%j.out
#SBATCH --error=logs/lbv2_klear8b_%j.err

set -euo pipefail

module load proxy/default
export no_proxy=localhost,127.0.0.1

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/LongBench" && pwd)"
MODEL_PATH=<PATH_TO_YOUR_MODEL>
MODEL_NAME=<YOUR_MODEL_NAME>
PORT=${PORT:-21513}
PYBIN=${PYBIN:-python}
VLLM_BIN=${VLLM_BIN:-vllm}

cd "$PROJECT_DIR"
mkdir -p logs results

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# pred.py reads these:
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi -L | head -1)"
echo "Model:     $MODEL_PATH (alias: $MODEL_NAME)"
echo "Port:      $PORT"
echo "=========================================="

# ---- 1. Launch vLLM OpenAI-compatible server in the background -----------------
SERVER_LOG=logs/vllm_server_${SLURM_JOB_ID}.log
"$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len 131072 \
    --hf-overrides '{"max_position_embeddings":131072,"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":65536}}' \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap "echo '[trap] killing vllm pid=$SERVER_PID'; kill -TERM $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true" EXIT

echo "[$(date)] vllm server pid=$SERVER_PID, log=$SERVER_LOG"
echo "[$(date)] waiting for server health..."
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "[$(date)] server ready after ${i}x10s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[ERROR] vllm process died before becoming ready. tail of log:"
        tail -50 "$SERVER_LOG"
        exit 1
    fi
    sleep 10
done

if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "[ERROR] server did not come up in 20 minutes"
    tail -80 "$SERVER_LOG"
    exit 1
fi

# ---- 2. Run prediction (CoT + answer-extraction follow-up) ---------------------
echo "[$(date)] running pred.py (0-shot CoT)..."
"$PYBIN" pred.py --model "$MODEL_NAME" --cot --n_proc 8

# ---- 3. Score ------------------------------------------------------------------
echo "[$(date)] running result.py..."
"$PYBIN" result.py
echo "------ result.txt ------"
cat result.txt
echo "------------------------"

echo "[$(date)] Done. Predictions in $PROJECT_DIR/results/, scores in $PROJECT_DIR/result.txt"
