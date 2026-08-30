#!/usr/bin/env bash
set -euxo pipefail

weka_workspace="/weka/nora-default/sijial/workspace"
run_id="${BEAKER_EXPERIMENT_ID:-${BEAKER_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}"
output_dir="$weka_workspace/contextaware-rl/assets/$run_id"
test -d "$weka_workspace"
test -w "$weka_workspace"
mkdir -p "$output_dir/Qwen3-8B" "$output_dir/sifs"
exec > >(tee -a "$output_dir/prepare.log") 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl fuse2fs squashfs-tools uidmap

curl -LsSf https://astral.sh/uv/0.11.2/install.sh | sh
export PATH="/root/.local/bin:$PATH"
uv --version

uvx --from huggingface-hub==0.36.2 hf download \
  Qwen/Qwen3-8B \
  --local-dir "$output_dir/Qwen3-8B"

curl -fL -o /tmp/apptainer.deb \
  https://github.com/apptainer/apptainer/releases/download/v1.4.5/apptainer_1.4.5_amd64.deb
apt-get install -y /tmp/apptainer.deb

image_name="$(uv run --no-project --with pandas --with pyarrow python - <<'PY'
import json
import pandas as pd

row = pd.read_parquet('/input/data/ContextRL_Agentic/swe_gym_train.parquet').iloc[0]
instance = json.loads(row['instance']) if isinstance(row['instance'], str) else row['instance']
instance_id = instance['instance_id'].replace('__', '_s_')
print(f'docker.io/xingyaoww/sweb.eval.x86_64.{instance_id}:latest'.lower())
PY
)"
sif_name="$(printf '%s' "$image_name" | sed 's|docker.io/||; s|/|_|g; s|:|-|g').sif"
printf '%s\n' "$image_name" | tee "$output_dir/sandbox-image.txt"
apptainer pull "$output_dir/sifs/$sif_name" "docker://$image_name"

sha256sum "$output_dir"/sifs/*.sif | tee "$output_dir/SHA256SUMS"
du -sh "$output_dir/Qwen3-8B" "$output_dir/sifs" | tee "$output_dir/SIZES"
