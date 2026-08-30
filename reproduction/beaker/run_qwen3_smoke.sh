#!/usr/bin/env bash
set -euxo pipefail

mkdir -p /results/ckpts /results/exports /results/logs /results/traj /workspace/assets/sifs
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl fuse2fs git libnuma1 proot squashfs-tools uidmap

curl -fL -o /tmp/apptainer.deb \
  https://github.com/apptainer/apptainer/releases/download/v1.4.5/apptainer_1.4.5_amd64.deb
apt-get install -y /tmp/apptainer.deb
curl -LsSf https://astral.sh/uv/0.11.2/install.sh | sh
export PATH="/root/.local/bin:$PATH"

tar -xzf /input/source.tar.gz -C /workspace
cd /workspace/Training/SkyRL
uvx --from huggingface-hub==0.36.2 hf download Qwen/Qwen3-8B \
  --local-dir /workspace/assets/Qwen3-8B
apptainer pull \
  /workspace/assets/sifs/xingyaoww_sweb.eval.x86_64.getmoto_s_moto-7365-latest.sif \
  docker://docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest
sha256sum /workspace/assets/sifs/*.sif | tee /results/SIF_SHA256SUMS
uv sync --python 3.12 --extra fsdp --extra miniswe

.venv/bin/python - <<'PY'
import json
import pandas as pd

source = '/workspace/data/ContextRL_Agentic/swe_gym_train.parquet'
row = pd.read_parquet(source).iloc[[0]]
row.to_parquet('/workspace/data/train.parquet', index=False)
row.to_parquet('/workspace/data/validation.parquet', index=False)

with open('/workspace/data/ContextRL_Agentic/contrastive_pairs.jsonl') as src:
    records = [json.loads(next(src)) for _ in range(8)]
with open('/workspace/data/contrastive_pairs.jsonl', 'w') as dst:
    for record in records:
        dst.write(json.dumps(record) + '\n')
PY

cp examples/train/mini_swe_agent/swebench.yaml /workspace/data/swebench-proot-smoke.yaml
sed -i 's/step_limit: 75/step_limit: 3/' /workspace/data/swebench-proot-smoke.yaml
sed -i 's/environment_class: singularity/environment_class: proot_sif/' /workspace/data/swebench-proot-smoke.yaml
sed -i 's/executable: apptainer/executable: proot/' /workspace/data/swebench-proot-smoke.yaml
sed -i 's|/path/to/swe_gym_images|/workspace/assets/sifs|' /workspace/data/swebench-proot-smoke.yaml
sed -i 's|/path/to/swe_smith_images|/workspace/assets/sifs|' /workspace/data/swebench-proot-smoke.yaml
# Qwen3's native thinking can consume the full short smoke-test allowance
# before it emits the bash action expected by mini-swe-agent.  Pass vLLM's
# chat-template switch through LiteLLM so the rollout uses direct answers.
sed -i '/    drop_params: true/a\    extra_body:\n      chat_template_kwargs:\n        enable_thinking: false' \
  /workspace/data/swebench-proot-smoke.yaml

export PYTHONPATH="$PWD"
.venv/bin/python - <<'PY'
import json
import pandas as pd
import yaml

from examples.train.mini_swe_agent.mini_swe_utils import get_sb_environment

row = pd.read_parquet('/workspace/data/train.parquet').iloc[0]
instance = json.loads(row['instance']) if isinstance(row['instance'], str) else row['instance']
with open('/workspace/data/swebench-proot-smoke.yaml') as stream:
    config = yaml.safe_load(stream)
env = get_sb_environment(config, instance, row['data_source'])
try:
    result = env.execute({'command': 'test -d /testbed && git rev-parse --is-inside-work-tree && git status --short'})
    print('PROOT_PREFLIGHT', result, flush=True)
    if result['returncode'] != 0 or 'true' not in result['output']:
        raise RuntimeError(f'PRoot SIF preflight failed: {result}')
finally:
    env.cleanup()
PY

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export MSWEA_COST_TRACKING="ignore_errors"
export TMPDIR="/tmp/miniswe-sandboxes"
export WANDB_DIR="/results/wandb"
export WANDB_CACHE_DIR="/tmp/wandb-cache"
export WANDB_CONFIG_DIR="/tmp/wandb-config"
mkdir -p "$TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

set -a
source examples/train/mini_swe_agent/.env.miniswe
set +a

.venv/bin/ray stop --force 2>/dev/null || true
.venv/bin/ray start --head --num-gpus=4 --temp-dir=/tmp/ray
export UV_OFFLINE=true

.venv/bin/python -m examples.train.mini_swe_agent.main_mini_swe_ct \
  data.train_data="['/workspace/data/train.parquet']" \
  data.val_data="['/workspace/data/validation.parquet']" \
  data.choose_trajectory_data="['/workspace/data/contrastive_pairs.jsonl']" \
  data.choose_trajectory_batch_size=4 \
  data.choose_trajectory_max_prompt_length=32768 \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.clip_ratio_c=10.0 \
  trainer.algorithm.loss_reduction="token_mean" \
  trainer.algorithm.zero_variance_filter=false \
  trainer.algorithm.choose_trajectory_coef=0.001 \
  trainer.algorithm.choose_trajectory_clip=5.0 \
  trainer.algorithm.choose_trajectory_prefill='""' \
  trainer.algorithm.choose_trajectory_enable_thinking=false \
  trainer.policy.model.path="/workspace/assets/Qwen3-8B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=4 \
  trainer.placement.ref_num_gpus_per_node=4 \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.policy.sequence_parallel_size=1 \
  generator.inference_engine.num_engines=2 \
  generator.inference_engine.tensor_parallel_size=2 \
  generator.inference_engine.served_model_name="Qwen/Qwen3-8B" \
  trainer.epochs=1 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1 \
  trainer.policy_mini_batch_size=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.dump_data_batch=true \
  trainer.ckpt_interval=1 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=1024 \
  generator.max_input_length=8192 \
  generator.max_turns=3 \
  trainer.policy.optimizer_config.lr=2.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.0005 \
  trainer.algorithm.kl_estimator_type=k3 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host='127.0.0.1' \
  generator.inference_engine.http_endpoint_port=8001 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=4 \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  trainer.logger="wandb" \
  trainer.project_name="ContextAwareRL-reproduction" \
  trainer.run_name="qwen3-8b-beaker-smoke" \
  trainer.resume_mode=none \
  trainer.ckpt_path="/results/ckpts" \
  trainer.export_path="/results/exports" \
  trainer.log_path="/results/logs" \
  generator.miniswe_config_path="/workspace/data/swebench-proot-smoke.yaml" \
  generator.miniswe_traj_dir="/results/traj"

test -d /results/ckpts/global_step_1
find /results/ckpts/global_step_1 -maxdepth 3 -type f -print | sort | tee /results/checkpoint-files.txt
printf 'TRAINING_STEP_VERIFIED\n' | tee /results/SUCCESS
