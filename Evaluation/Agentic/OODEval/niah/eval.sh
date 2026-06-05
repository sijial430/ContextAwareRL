#!/bin/bash
#SBATCH --job-name=niah-ctv2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/niah_ctv2_%j.out
#SBATCH --error=logs/niah_ctv2_%j.err

set -euo pipefail

module load proxy/default
export no_proxy=localhost,127.0.0.1

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH=<PATH_TO_YOUR_MODEL>
MODEL_NAME=<YOUR_MODEL_NAME>

JUDGE_NAME=${JUDGE_NAME:-gpt-4o}        # OpenAI model used as the judge
PORT=${PORT:-21516}

export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY not set}"
CTX_MIN=${CTX_MIN:-1000}
CTX_MAX=${CTX_MAX:-32000}
CTX_INTERVALS=${CTX_INTERVALS:-15}
DEPTH_INTERVALS=${DEPTH_INTERVALS:-10}
PYBIN=${PYBIN:-python}
VLLM_BIN=${VLLM_BIN:-vllm}

cd "$PROJECT_DIR"
mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NIAH_MODEL_BASE_URL="http://127.0.0.1:${PORT}/v1"
export NIAH_MODEL_API_KEY=token-abc123
export NIAH_MODEL_TOKENIZER_PATH="$MODEL_PATH"
export NIAH_EVALUATOR_API_KEY="$OPENAI_API_KEY"

echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $(nvidia-smi -L | head -1)"
echo "Model:         $MODEL_PATH (alias: $MODEL_NAME)"
echo "Judge:         $JUDGE_NAME"
echo "Port:          $PORT"
echo "Context range: ${CTX_MIN}-${CTX_MAX} (${CTX_INTERVALS} intervals, ${DEPTH_INTERVALS} depths)"
echo "=========================================="

# ---- 1. Launch vLLM OpenAI-compatible server -----------------------------------
SERVER_LOG=logs/vllm_server_${SLURM_JOB_ID}.log
"$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len 65536 \
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

# ---- 2. Run NIAH ---------------------------------------------------------------
echo "[$(date)] running NIAH single-needle..."
"$PYBIN" -m needlehaystack.run \
    --provider openai \
    --model_name "$MODEL_NAME" \
    --evaluator openai \
    --evaluator_model_name "$JUDGE_NAME" \
    --context_lengths_min "$CTX_MIN" \
    --context_lengths_max "$CTX_MAX" \
    --context_lengths_num_intervals "$CTX_INTERVALS" \
    --document_depth_percent_intervals "$DEPTH_INTERVALS" \
    --num_concurrent_requests 4 \
    --save_contexts False \
    --save_results True

echo "[$(date)] Done. Results saved under $PROJECT_DIR/results/"
