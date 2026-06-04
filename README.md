# Official Repo for "Context-Aware RL for Agentic and Multimodal LLMs"

This repository contains the official code for the paper **"Context-Aware RL for Agentic and Multimodal LLMs"**. We provide a context-aware reinforcement learning (RL) recipe and apply it to two settings:

- **Multimodal LLMs** — vision-language reasoning where the model must learn *which image(s)* to attend to (image-selection-aware reward, `choose_image_coef` / `choose_image_clip`). Built on [EasyR1 / verl](https://github.com/hiyouga/EasyR1).
- **Agentic LLMs** — long-horizon, tool-using agents (e.g. SWE-bench-style code repair) trained with contrastive context pairs. Built on [SkyRL](https://github.com/NovaSky-AI/SkyRL).

```
Code/
├── DataPreparation/   # Prepare RL training data
├── Training/          # Training code and scripts for multimodal and agentic models
└── Evaluation/        # benchmark evaluation for both tracks
```

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Preparation](#2-data-preparation)
3. [Training](#3-training)
4. [Evaluation](#4-evaluation)
5. [Citation](#5-citation)

---

## 1. Environment Setup

We recommend Python 3.10+ and CUDA 12.x. Create a separate environment per track.

### Multimodal (EasyR1 / verl)

```bash

```

### Agentic (SkyRL)

```bash

```

> **Hardware.** Experiments were run on multi-GPU nodes (A100/H100 80GB) via Slurm. The provided launch scripts (`sbatch ...`) assume a Slurm cluster; adapt the resource directives at the top of each script to your environment.

---

## 2. Data Preparation

### 2.1 Multimodal

Download the pre-built RL training data directly from the Hugging Face Hub.

For **Qwen2.5 VL 7B**:

```bash
huggingface-cli download xupy21/ContextRL_Multimodal_Qwen2.5_VL --repo-type dataset --local-dir ./data/ContextRL_Multimodal_Qwen2.5_VL
```

Dataset: https://huggingface.co/datasets/xupy21/ContextRL_Multimodal_Qwen2.5_VL

For **Qwen3 VL 8B**:

```bash
huggingface-cli download xupy21/ContextRL_multimodal_Qwen3_VL --repo-type dataset --local-dir ./data/ContextRL_multimodal_Qwen3_VL
```

Dataset: https://huggingface.co/datasets/xupy21/ContextRL_multimodal_Qwen3_VL

### 2.2 Agentic

Download the pre-built RL training data directly from the Hugging Face Hub.

```bash
huggingface-cli download xupy21/ContextRL_Agentic --repo-type dataset --local-dir ./data/ContextRL_Agentic
```

Dataset: https://huggingface.co/datasets/xupy21/ContextRL_Agentic

#### Build the execution sandboxes

The agentic track runs each task inside an isolated, per-instance execution environment. After downloading the data above, build the sandboxes by pulling every required image as an Apptainer/Singularity `.sif`. We provide two pull scripts in `DataPreparation/`, one per image source:

- `pull_swe_gym_images.sh` — pulls the **SWE-Gym** images
- `pull_swe_smith_images.sh` — pulls the **SWE-smith** images

Both scripts require `Apptainer` on your `PATH`. They are resume-safe — already-downloaded `.sif` files are skipped, so they can be re-run after an interruption. If you need Docker, please use your own script.

```bash
# SWE-Gym images
bash DataPreparation/pull_swe_gym_images.sh

# SWE-smith images
bash DataPreparation/pull_swe_smith_images.sh
```

**Parameters you must set**:

- pull_swe_gym_images.sh

| Variable       | Default                      | Meaning                                      |
| -------------- | ---------------------------- | -------------------------------------------- |
| `PARQUET_FILE` | `data/swe_gym_train.parquet` | Input parquet the image list is derived from |
| `IMAGE_DIR`    | `data/swe_gym_images`        | Output directory for the `.sif` files        |

- pull_swe_smith_images.sh

| Variable       | Default                        | Meaning                                      |
| -------------- | ------------------------------ | -------------------------------------------- |
| `PARQUET_FILE` | `data/swe_smith_train.parquet` | Input parquet the image list is derived from |
| `IMAGE_DIR`    | `data/swe_smith_images`        | Output directory for the `.sif` files        |

> **Tip.** The pulls are large and may take many hours. 

Once all images are pulled, the sandboxes are ready, and you can proceed to [Training](#3-training).



---

## 3. Training

### 3.1 Multimodal

The multimodal track is trained with our customized [EasyR1 / verl](https://github.com/hiyouga/EasyR1) under `Training/EasyR1`. In the paper we use two backbones, **Qwen2.5-VL-7B-Instruct** and **Qwen3-VL-8B-Instruct**. 

#### Step 1 — Download the base model

Download the backbone(s) from the Hugging Face Hub. The paths below match the `MODEL_PATH` used in the training scripts; adjust them to your own layout if needed.

```bash
# Qwen2.5-VL-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./Qwen2.5-VL-7B-Instruct

# Qwen3-VL-8B-Instruct
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir ./Qwen3-VL-8B-Instruct
```

#### Step 2 — Prepare the data

Follow [Section 2.1](#21-multimodal) to download the RL training data for the corresponding backbone.

#### Step 3 — Launch RL training

We provide one launch script per backbone under `Training/EasyR1/examples/`:

- `qwen_V2_5.sh` — Qwen2.5-VL-7B-Instruct
- `qwen_V3.sh` — Qwen3-VL-8B-Instruct

Before submitting, set the placeholder variables at the top of the script to match your environment:

- `MODEL_PATH` — the base model directory from Step 1.
- `TRAIN_FILES` / `VAL_FILES` — the data from Step 2 (passed to `data.train_files` / `data.val_files`).
- `WANDB_API_KEY` (and `OPENAI_API_KEY` for `qwen_V2_5.sh`).

Submit on a Slurm cluster:

```bash
cd Training/EasyR1

# Qwen2.5-VL-7B-Instruct
sbatch examples/qwen_V2_5.sh

# Qwen3-VL-8B-Instruct
sbatch examples/qwen_V3.sh
```

Checkpoints are written to `checkpoints/<project_name>/<experiment_name>/global_step_<N>/` 

#### Step 4 — Merge the checkpoint into a Hugging Face model

Training saves sharded FSDP weights (`model_world_size_*_rank_*.pt`) under each `global_step_<N>/actor/`. Use `scripts/model_merger.py` to consolidate them into a standard Hugging Face model directory (written to a `huggingface/` subfolder) that can be loaded directly for inference:

```bash
python scripts/model_merger.py \
    --local_dir checkpoints/easy_r1/<experiment_name>/global_step_<N>/actor
```

#### Step 5 — Enable thinking mode (Qwen3 only)

The Qwen3 backbone is trained with the thinking-mode template, so the merged checkpoint should carry the same template for inference. Apply it with `scripts/replace_chat_template.py`:

```bash
python scripts/replace_chat_template.py \
    checkpoints/easy_r1/<experiment_name>/global_step_<N>/actor/huggingface \
    --template examples/chat_template.json
```

The resulting model is ready for [Evaluation](#4-evaluation).

### 3.2 Agentic

The agentic track is trained with our customized [SkyRL](https://github.com/NovaSky-AI/SkyRL) under `Training/SkyRL`. In the paper we use two backbones, **Klear-AgentForge-8B-SFT** and **Qwen3-8B**. Both are trained with colocated GRPO + a choose-trajectory (CT) auxiliary loss on SWE-Bench-style code-repair tasks, executed inside per-instance Apptainer sandboxes.

#### Step 1 — Download the base model

Download the backbone(s) from the Hugging Face Hub. The paths below match the `trainer.policy.model.path` used in the training scripts; adjust them to your own layout if needed.

```bash
# Klear-AgentForge-8B-SFT
huggingface-cli download Kwai-Klear/Klear-AgentForge-8B-SFT --local-dir ./Klear-AgentForge-8B-SFT

# Qwen3-8B
huggingface-cli download Qwen/Qwen3-8B --local-dir ./Qwen3-8B
```

#### Step 2 — Prepare the data

Follow [Section 2.2](#22-agentic) to download the RL training data and build the Apptainer execution sandboxes.

#### Step 3 — Configure the environment file and sandbox paths

The training scripts source `examples/train/mini_swe_agent/.env.miniswe` before starting Ray so that all workers inherit the LiteLLM and OpenAI-proxy settings. A ready-to-use template is already in place; the defaults (`OPENAI_BASE_URL`, `LITELLM_MODEL_REGISTRY_PATH`) work without modification as long as you follow the standard setup.

The Apptainer image directories for the sandboxes are configured separately in `examples/train/mini_swe_agent/swebench.yaml`. Set `local_sif_dir` to the directory where you pulled the `.sif` files in Section 2.2:

```yaml
environment:
  local_sif_dir:
    - /path/to/swe_gym_images    # directory containing SWE-Gym .sif files
    - /path/to/swe_smith_images  # directory containing SWE-Smith .sif files
```

#### Step 4 — Launch RL training

We provide one launch script per backbone under `Training/SkyRL/examples/train/mini_swe_agent/`:

- `run_mini_swe_agentforge8B.sh` — Klear-AgentForge-8B-SFT (trained on SWE-Gym + SWE-Smith)
- `run_mini_swe_qwen3_8b.sh` — Qwen3-8B (trained on SWE-Gym)

Before submitting, set the following variables at the top of each script to match your environment:

| Variable                                                   | Description                                     |
| ---------------------------------------------------------- | ----------------------------------------------- |
| `trainer.policy.model.path`                                | Base model directory from Step 1                |
| `GYM_DIR` / `SMITH_DIR` (agentforge) or `DATA_DIR` (qwen3) | RL training data directories from Step 2        |
| `CT_DATA`                                                  | Path to the CT `eval_prompts.jsonl` from Step 2 |
| `CKPT_PATH` / `EXPORT_PATH` / `MINISWE_TRAJ_DIR`           | Output directories (relative defaults provided) |
| `WANDB_API_KEY`                                            | Your Weights & Biases API key                   |

Submit on a Slurm cluster:

```bash
cd Training/SkyRL

# Klear-AgentForge-8B-SFT
sbatch examples/train/mini_swe_agent/run_mini_swe_agentforge8B.sh

# Qwen3-8B
sbatch examples/train/mini_swe_agent/run_mini_swe_qwen3_8b.sh
```

Checkpoints are written to `CKPT_PATH/global_step_<N>/`.

#### Step 5 — Merge the checkpoint into a Hugging Face model

Training saves sharded FSDP2 weights (`model_world_size_*_rank_*.pt`) under each `global_step_<N>/policy/`. Use `merge_fsdp_ckpt_to_hf.py` at the root of `Training/SkyRL/` to consolidate them into a standard Hugging Face model directory that can be loaded directly for inference:

```bash
cd Training/SkyRL

python merge_fsdp_ckpt_to_hf.py \
  --ckpt <CKPT_PATH>/global_step_<N>/policy \
  --out  exports/<experiment_name>_step<N>_hf \
  --dtype bfloat16
```

The merged model is written to the `--out` directory and is ready for [Evaluation](#4-evaluation).

---

## 4. Evaluation

TODO

---

## 5. Citation

If you find this work useful, please cite:

```bibtex

```