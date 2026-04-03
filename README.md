# ContextAwareRL

Context-aware reinforcement learning training for vision-language models. Based on [EasyR1/verl](https://github.com/), with GRPO algorithm and image-selection-aware reward shaping (`choose_image_coef` / `choose_image_clip`).

Two sub-projects targeting different base models:

---

## EasyR1_Qwen2_5 (Qwen2.5-VL-7B-Instruct)

### Setup

```bash
cd EasyR1_Qwen2_5
pip install -e .
```

### Training

```bash
# Full training
sbatch examples/qwen_V2_5_aug.sh

# Quick sanity check (30 steps, ~1h)
sbatch examples/qwen_V2_5_aug_test.sh
```

### Post-training: Merge Model Shards

Training saves FSDP-sharded checkpoints (`model_world_size_*_rank_*.pt`). You must merge them into a single HuggingFace model before inference:

```bash
python scripts/model_merger.py --local_dir <checkpoint_step_dir>
```


## EasyR1_Qwen3 (Qwen3-VL-8B-Instruct)

### Setup

```bash
cd EasyR1_Qwen3
pip install -e .
```

### Training

```bash
# Full training
sbatch examples/qwen_V3_aug.sh

# Quick sanity check (30 steps, ~1h)
sbatch examples/qwen_V3_aug_test.sh
```

### Post-training Step 1: Merge Model Shards

Same as Qwen2.5 — merge FSDP shards into a HuggingFace model:

```bash
python scripts/model_merger.py --local_dir <checkpoint_step_dir>
```

### Post-training Step 2: Replace Chat Template (Qwen3 only)

After merging, the checkpoint still contains the original Instruct chat template. You **must** replace it with the Thinking model's template before deployment/inference:

```bash
python scripts/replace_chat_template.py <merged_huggingface_dir>
```

This updates `chat_template.jinja` and the `chat_template` field in `tokenizer_config.json` so inference uses the `<think>...</think>` format.

