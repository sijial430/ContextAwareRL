#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
output_dir="${1:?usage: package_input.sh OUTPUT_DIR [LAUNCHER]}"
launcher="${2:-run_qwen3_smoke.sh}"

mkdir -p "$output_dir"
tar \
  --exclude='Training/SkyRL/.venv' \
  --exclude='**/__pycache__' \
  --exclude='*.pyc' \
  -czf "$output_dir/source.tar.gz" \
  -C "$repo_root" \
  Training/SkyRL data/ContextRL_Agentic
cp "$script_dir/$launcher" "$output_dir/$launcher"

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$output_dir/source.tar.gz"
else
  shasum -a 256 "$output_dir/source.tar.gz"
fi
