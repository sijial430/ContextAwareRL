#!/bin/bash
#SBATCH --job-name=swe-lite-infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --time=20:00:00
#SBATCH --output=logs/swe_lite_infer_%j.out
#SBATCH --error=logs/swe_lite_infer_%j.err

# ============================================================
# mini-swe-agent Inference on SWE-bench Lite (test, 300 instances)
# (vLLM + Apptainer sandboxes)
# ============================================================

module load proxy/default
export no_proxy=localhost,127.0.0.1

MODEL_PATH="${MODEL_PATH:?MODEL_PATH not set}"
MINI_SWE_AGENT_DIR="${MINI_SWE_AGENT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../mini-swe-agent" && pwd)}"
SANDBOX_DIR="${SANDBOX_DIR:?SANDBOX_DIR not set}"
WORK_DIR="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

VLLM_PORT=$(shuf -i 29500-39500 -n 1)
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-$(basename "$MODEL_PATH")}"

NUM_WORKERS="${NUM_WORKERS:-1}"
SUBSET="lite"
SPLIT="test"
INSTANCE_SLICE=""

conda activate "${CONDA_ENV:-mini-swe-agent}"

export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY not set}"

cd "$WORK_DIR"
mkdir -p logs

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "Model: $MODEL_PATH"
echo "Sandbox dir: $SANDBOX_DIR"
echo "Subset: $SUBSET ($SPLIT)"
echo "=========================================="

echo "[$(date)] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$VLLM_PORT" \
    --served-model-name "$VLLM_MODEL_NAME" \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.9 \
    &> logs/vllm_server_${SLURM_JOB_ID}.log &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

echo "[$(date)] Waiting for vLLM server to start..."
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${VLLM_PORT}/health" &>/dev/null; then
        echo "[$(date)] vLLM server is ready!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[$(date)] ERROR: vLLM server process died!"
        cat logs/vllm_server_${SLURM_JOB_ID}.log
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "[$(date)] ERROR: vLLM server did not start within ${MAX_WAIT}s"
    cat logs/vllm_server_${SLURM_JOB_ID}.log
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo "[$(date)] Starting mini-swe-agent batch run..."

# Move overlay images off /tmp (tmpfs/RAM) to avoid EIO when
# 10GB ext3 overlays accumulate. Cleaned at job end.
export TMPDIR="${TMPDIR:-/tmp}/minisweagent_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
echo "[$(date)] TMPDIR=$TMPDIR"

OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/output_swe_lite}"

SLICE_ARG=""
if [ -n "$INSTANCE_SLICE" ]; then
    SLICE_ARG="--slice $INSTANCE_SLICE"
fi

mini-extra swebench \
    -c "${MINI_SWE_AGENT_DIR}/src/minisweagent/config/benchmarks/swebench_klear.yaml" \
    -c "model.model_name=openai/${VLLM_MODEL_NAME}" \
    -c "model.model_kwargs.api_base=http://localhost:${VLLM_PORT}/v1" \
    --subset "$SUBSET" \
    --split "$SPLIT" \
    --sandbox-dir "$SANDBOX_DIR" \
    --environment-class singularity \
    --workers "$NUM_WORKERS" \
    -o "$OUTPUT_DIR" \
    $SLICE_ARG

MINI_EXIT=$?

echo "[$(date)] Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "[$(date)] Done! Exit code: $MINI_EXIT"
echo "Results saved to: $OUTPUT_DIR"
