#!/usr/bin/env bash
set -euxo pipefail

mkdir -p /results/Qwen3-8B /results/sifs
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl fuse2fs squashfs-tools uidmap

curl -LsSf https://astral.sh/uv/0.11.2/install.sh | sh
export PATH="/root/.local/bin:$PATH"
uv --version

uvx --from huggingface-hub==0.36.2 hf download \
  Qwen/Qwen3-8B \
  --local-dir /results/Qwen3-8B

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
printf '%s\n' "$image_name" | tee /results/sandbox-image.txt
apptainer pull "/results/sifs/$sif_name" "docker://$image_name"

sha256sum /results/sifs/*.sif | tee /results/SHA256SUMS
du -sh /results/Qwen3-8B /results/sifs | tee /results/SIZES
