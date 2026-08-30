#!/usr/bin/env bash
set -euxo pipefail

weka_workspace="/weka/nora-default/sijial/workspace"
output_parent="${CONTEXTAWARE_OUTPUT_ROOT:-$weka_workspace/contextaware-rl/qwen3-8b-full-config-1-step}"
run_id="${BEAKER_EXPERIMENT_ID:-${BEAKER_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}"
output_dir="$output_parent/$run_id"
checkpoint_root="$output_dir/checkpoints"

test -d "$weka_workspace"
test -w "$weka_workspace"
mkdir -p \
  "$checkpoint_root" \
  "$output_dir/exports" \
  "$output_dir/logs" \
  "$output_dir/traj" \
  "$output_dir/wandb" \
  /workspace/assets/sifs
exec > >(tee -a "$output_dir/run.log") 2>&1
printf 'WEKA_OUTPUT_DIR=%s\n' "$output_dir" | tee "$output_dir/OUTPUT_DIR.txt"
cp /input/run_qwen3_fullcheck.sh "$output_dir/run_qwen3_fullcheck.sh"

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
uv sync --python 3.12 --extra fsdp --extra miniswe

.venv/bin/python - <<'PY'
import json
from pathlib import Path

import pandas as pd

source_path = Path('/workspace/data/ContextRL_Agentic/swe_gym_train.parquet')
source = pd.read_parquet(source_path)
train = source.iloc[:16].copy()
validation = source.iloc[16:32].copy()
train.to_parquet('/workspace/data/train_fullcheck.parquet', index=False)
validation.to_parquet('/workspace/data/validation_fullcheck.parquet', index=False)

manifest = []
for row in train.itertuples(index=False):
    instance = json.loads(row.instance) if isinstance(row.instance, str) else row.instance
    image_name = instance.get('image_name')
    if image_name is None:
        instance_id = instance['instance_id'].replace('__', '_s_').lower()
        image_name = f'docker.io/xingyaoww/sweb.eval.x86_64.{instance_id}:latest'
    sif_name = image_name.removeprefix('docker.io/').replace('/', '_').replace(':', '-') + '.sif'
    manifest.append((sif_name, image_name, instance['instance_id']))

with open('/workspace/data/sif_manifest.tsv', 'w') as stream:
    for sif_name, image_name, instance_id in dict.fromkeys(manifest):
        stream.write(f'{sif_name}\t{image_name}\t{instance_id}\n')

print('FULLCHECK_TRAIN_IDS', [
    (json.loads(value) if isinstance(value, str) else value)['instance_id']
    for value in train['instance']
])
PY

while IFS=$'\t' read -r sif_name image_name instance_id; do
  printf 'PULLING_SIF %s %s\n' "$instance_id" "$image_name"
  apptainer pull \
    "/workspace/assets/sifs/$sif_name" \
    "docker://$image_name"
done < /workspace/data/sif_manifest.tsv
sha256sum /workspace/assets/sifs/*.sif | tee "$output_dir/SIF_SHA256SUMS"

cp examples/train/mini_swe_agent/swebench.yaml /workspace/data/swebench-proot-fullcheck.yaml
sed -i 's/environment_class: singularity/environment_class: proot_sif/' /workspace/data/swebench-proot-fullcheck.yaml
sed -i 's/executable: apptainer/executable: proot/' /workspace/data/swebench-proot-fullcheck.yaml
sed -i 's|/path/to/swe_gym_images|/workspace/assets/sifs|' /workspace/data/swebench-proot-fullcheck.yaml
sed -i 's|/path/to/swe_smith_images|/workspace/assets/sifs|' /workspace/data/swebench-proot-fullcheck.yaml

export PYTHONPATH="$PWD"
.venv/bin/python - <<'PY'
import json
import pandas as pd
import yaml

from examples.train.mini_swe_agent.mini_swe_utils import get_sb_environment

row = pd.read_parquet('/workspace/data/train_fullcheck.parquet').iloc[0]
instance = json.loads(row['instance']) if isinstance(row['instance'], str) else row['instance']
with open('/workspace/data/swebench-proot-fullcheck.yaml') as stream:
    config = yaml.safe_load(stream)
assert config['agent']['step_limit'] == 75
env = get_sb_environment(config, instance, row['data_source'])
try:
    result = env.execute({'command': 'test -d /testbed && git rev-parse --is-inside-work-tree && git status --short'})
    print('PROOT_FULLCHECK_PREFLIGHT', result, flush=True)
    if result['returncode'] != 0 or 'true' not in result['output']:
        raise RuntimeError(f'PRoot SIF preflight failed: {result}')
finally:
    env.cleanup()
PY

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export MSWEA_COST_TRACKING="ignore_errors"
export TMPDIR="/tmp/miniswe-sandboxes"
export WANDB_DIR="$output_dir/wandb"
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
  data.train_data="['/workspace/data/train_fullcheck.parquet']" \
  data.val_data="['/workspace/data/validation_fullcheck.parquet']" \
  data.choose_trajectory_data="['/workspace/data/ContextRL_Agentic/contrastive_pairs.jsonl']" \
  data.choose_trajectory_batch_size=8 \
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
  trainer.eval_batch_size=100 \
  trainer.eval_before_train=false \
  trainer.eval_interval=0 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.dump_data_batch=true \
  trainer.ckpt_interval=1 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=4096 \
  generator.max_input_length=28672 \
  generator.max_turns=35 \
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
  generator.n_samples_per_prompt=8 \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  trainer.logger="wandb" \
  trainer.project_name="ContextAwareRL-reproduction" \
  trainer.run_name="qwen3-8b-full-config-1-step-$run_id" \
  trainer.resume_mode=none \
  trainer.ckpt_path="$checkpoint_root" \
  trainer.export_path="$output_dir/exports" \
  trainer.log_path="$output_dir/logs" \
  generator.miniswe_config_path="/workspace/data/swebench-proot-fullcheck.yaml" \
  generator.miniswe_traj_dir="$output_dir/traj"

checkpoint_dir="$(find "$checkpoint_root" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -1)"
test -n "$checkpoint_dir"
test "$(find "$checkpoint_dir/policy" -maxdepth 1 -name 'model_world_size_4_rank_*.pt' | wc -l)" -eq 4
test "$(find "$checkpoint_dir/policy" -maxdepth 1 -name 'optim_world_size_4_rank_*.pt' | wc -l)" -eq 4
find "$checkpoint_dir" -maxdepth 3 -type f -print | sort | tee "$output_dir/checkpoint-files.txt"
du -sh "$checkpoint_dir" | tee "$output_dir/checkpoint-size.txt"
printf 'FULL_CONFIG_TRAINING_STEP_VERIFIED %s\n' "$checkpoint_dir" | tee "$output_dir/SUCCESS"
