#!/usr/bin/env bash
set -euo pipefail

weka_workspace=/weka/nora-default/sijial/workspace
export CONTEXTAWARE_OUTPUT_ROOT="$weka_workspace/contextaware-rl/qwen3-8b-released-370-one-epoch"
export CONTEXTAWARE_ASSET_ROOT="$weka_workspace/contextaware-rl/assets/qwen3-8b-released-370"
export CONTEXTAWARE_TRAIN_ROWS=0
export CONTEXTAWARE_TRAINING_EPOCHS=1
export CONTEXTAWARE_CKPT_INTERVAL=10
export CONTEXTAWARE_RESUME_MODE=latest
export CONTEXTAWARE_EXPECTED_TRAJECTORIES=2944
export CONTEXTAWARE_SUCCESS_MARKER=RELEASED_370_ROW_EPOCH_VERIFIED
export CONTEXTAWARE_RUN_LABEL=qwen3-8b-released-370-one-epoch

exec /bin/bash "$(dirname "$0")/run_qwen3_fullcheck.sh"
