"""Merge a SkyRL FSDP2 sharded checkpoint into HuggingFace safetensors format.

Assumes every tensor in the sharded checkpoint is a DTensor with Shard(dim=0)
placement — the default for SkyRL's FSDP2 policy saver.

Usage:
  srun --partition=cpu --mem=200G --cpus-per-task=8 --time=00:30:00 --pty \
    .venv/bin/python merge_fsdp_ckpt_to_hf.py \
    --ckpt <CKPT_PATH>/global_step_<N>/policy \
    --out  exports/<experiment_name>_step<N>_hf \
    --dtype bfloat16
"""
import argparse
import shutil
from pathlib import Path

import torch
from torch.distributed.tensor import Shard
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path,
                   help="Path to policy/ dir containing model_world_size_*_rank_*.pt")
    p.add_argument("--out", required=True, type=Path, help="Output HF dir")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def load_shards(ckpt_dir: Path, world_size: int):
    shards = []
    for r in range(world_size):
        path = ckpt_dir / f"model_world_size_{world_size}_rank_{r}.pt"
        print(f"  loading {path.name}")
        shards.append(torch.load(path, map_location="cpu", weights_only=False))
    return shards


def merge_key(shards, key):
    dt = shards[0][key]
    global_shape = tuple(dt.shape)
    placements = dt.placements
    assert len(placements) == 1 and isinstance(placements[0], Shard) \
        and placements[0].dim == 0, \
        f"{key}: expected Shard(dim=0), got {placements}"

    locals_ = []
    for r in range(len(shards)):
        v = shards[r][key]
        lt = v.to_local() if hasattr(v, "to_local") else v._local_tensor
        locals_.append(lt)
    full = torch.cat(locals_, dim=0)
    if full.shape[0] > global_shape[0]:
        full = full[: global_shape[0]].contiguous()
    assert tuple(full.shape) == global_shape, \
        f"{key}: merged {tuple(full.shape)} != global {global_shape}"
    return full


def main():
    args = parse_args()
    ckpt_dir: Path = args.ckpt
    out_dir: Path = args.out
    target_dtype = getattr(torch, args.dtype)

    fsdp_cfg_path = ckpt_dir / "fsdp_config.json"
    assert fsdp_cfg_path.exists(), f"missing {fsdp_cfg_path}"
    import json
    fsdp_cfg = json.loads(fsdp_cfg_path.read_text())
    world_size = fsdp_cfg["world_size"]
    print(f"fsdp_strategy={fsdp_cfg['fsdp_strategy']} world_size={world_size}")

    print("Loading shards...")
    shards = load_shards(ckpt_dir, world_size)

    keys = list(shards[0].keys())
    print(f"Merging {len(keys)} tensors...")
    merged = {}
    for i, k in enumerate(keys):
        full = merge_key(shards, k).to(target_dtype)
        merged[k] = full
        # Free per-rank entries as we go to cap memory.
        for r in range(world_size):
            shards[r].pop(k, None)
        if (i + 1) % 50 == 0 or i + 1 == len(keys):
            print(f"  merged {i+1}/{len(keys)}")
    del shards

    hf_src = ckpt_dir / "huggingface"
    print(f"Building empty model from {hf_src}")
    config = AutoConfig.from_pretrained(hf_src)
    config.torch_dtype = args.dtype
    model = AutoModelForCausalLM.from_config(config, torch_dtype=target_dtype)

    missing, unexpected = model.load_state_dict(merged, strict=False)
    if missing:
        print(f"WARN missing keys ({len(missing)}): {missing[:8]}")
    if unexpected:
        print(f"WARN unexpected keys ({len(unexpected)}): {unexpected[:8]}")
    del merged

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True)

    print("Copying tokenizer / chat template")
    tok = AutoTokenizer.from_pretrained(hf_src)
    tok.save_pretrained(out_dir)
    chat_tpl = hf_src / "chat_template.jinja"
    if chat_tpl.exists():
        shutil.copy2(chat_tpl, out_dir / "chat_template.jinja")

    print(f"Done. HF model at {out_dir}")


if __name__ == "__main__":
    main()
