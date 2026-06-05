#!/bin/bash
# ============================================================
# Build all 300 SWE-bench Lite instance sandboxes
# Run directly on login node: bash scripts/build_instance_sandboxes_swe_lite.sh
# (compute nodes have no internet; instance build does git clone)
# ============================================================

WORK_DIR="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SANDBOX_DIR="${SANDBOX_DIR:-${WORK_DIR}/logs/sandboxes}"

conda activate "${CONDA_ENV:-swebench}"

cd "$WORK_DIR"
mkdir -p logs
export SWEBENCH_SANDBOX_DIR="$SANDBOX_DIR"

echo "=========================================="
echo "Sandbox dir: $SWEBENCH_SANDBOX_DIR"
echo "Dataset:    SWE-bench/SWE-bench_Lite (test, 300 instances)"
echo "=========================================="

python -m swebench.harness.prepare_images \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --split test \
    --max_workers 1 \
    --open_file_limit 1024 \
    --tag latest \
    --env_image_tag latest

echo "Build complete!"
echo "Total instance sandboxes: $(ls -d ${SANDBOX_DIR}/sweb.eval.* 2>/dev/null | wc -l)"

echo "=========================================="
echo "Installing swe-rex into all eval sandboxes..."
echo "=========================================="
bash "$WORK_DIR/scripts/install_swerex_sandboxes.sh"
