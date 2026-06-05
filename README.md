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

### 4.1 Multimodal

We use two evaluation toolkits: **VLMEvalKit** and **lmms-eval**. The subsections below cover each in turn.

#### 4.1.1 VLMEvalKit

Evaluation code lives under `Evaluation/Multimodal/VLMEval/`.

**Step 1 — Environment setup and dataset download**

Follow the official VLMEvalKit repository for installation and dataset preparation:
[https://github.com/open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit)

Set the `LMUData` environment variable to the directory where you have downloaded the benchmark data.

**Step 2 — Register your model path**

Open `Evaluation/Multimodal/VLMEval/vlmeval/config.py` and add an entry for your merged checkpoint inside the appropriate model series dict (e.g. `qwen2vl_series` for Qwen2.5-VL, `qwen3vl_series` for Qwen3-VL). The key must match the model name used in the eval scripts.

For example, for Qwen2.5-VL-7B add to `qwen2vl_series`:

```python
"Qwen2.5-VL-7B-Best": partial(
    Qwen2VLChat,
    model_path="/path/to/your/merged/qwen2.5vl-7b-checkpoint",
    min_pixels=1280 * 28 * 28,
    max_pixels=16384 * 28 * 28,
    use_custom_prompt=False,
),
```

For Qwen3-VL-8B add to `qwen3vl_series`:

```python
"Qwen3-VL-8B-Best": partial(
    Qwen3VLChat,
    model_path="/path/to/your/merged/qwen3vl-8b-checkpoint",
    use_custom_prompt=False,
    use_vllm=True,
    temperature=0.0,
    max_new_tokens=32768,
),
```

**Step 3 — Run evaluation**

We provide two ready-to-submit Slurm scripts. Before submitting, set `OPENAI_API_KEY` (required for judge-based metrics) and `LMUData` at the top of each script.

```bash
cd Evaluation/Multimodal/VLMEval

# Qwen2.5-VL-7B-Instruct backbone
# Benchmarks: MathVista_MINI, MathVerse_MINI, MathVision, MMMU_DEV_VAL,
#             MMMU_Pro_10c, VStarBench, MMStar, BLINK
bash eval_qwen25.sh

# Qwen3-VL-8B-Instruct backbone
# Benchmarks: MathVerse_MINI, MathVision, MMMU_DEV_VAL,
#             MMMU_Pro_10c, MMStar, BLINK
bash eval_qwen3.sh
```

Each script submits one Slurm job per model. Results are written by VLMEvalKit to the working directory (`.xlsx` / `.csv` files).

#### 4.1.2 lmms-eval

Evaluation code lives under `Evaluation/Multimodal/lmms-eval/`.

**Step 1 — Environment setup and dataset download**

Follow the official lmms-eval repository for installation and dataset preparation:
[https://github.com/EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)

Download the required benchmark datasets and point `LMMS_EVAL_HOME` to their cache directory (the scripts default it to `<WORK_DIR>/cache`).

**Step 2 — Run evaluation**

Both scripts are written as Slurm batch scripts and accept `MODEL_PATH` and `RUN_NAME` via `--export`. Set `OPENAI_API_KEY` in your environment before submitting (required for judge-based metrics).

```bash
cd Evaluation/Multimodal/lmms-eval

# Qwen2.5-VL-7B-Instruct backbone
# Benchmarks: ScienceQA-IMG, MME-RealWorld-Lite, OlympiadBench (physics),
#             PhyX-MC
sbatch --export=ALL,MODEL_PATH=/path/to/merged/qwen2.5vl-7b,RUN_NAME=qwen25_best \
    eval_qwen25.sh

# Qwen3-VL-8B-Instruct backbone
# Benchmarks: ScienceQA-IMG, MME-RealWorld-Lite, OlympiadBench (physics),
#             PhyX-MC, MathVista-testmini, VStarBench
sbatch --export=ALL,MODEL_PATH=/path/to/merged/qwen3vl-8b,RUN_NAME=qwen3_best \
    eval_qwen3.sh
```

Results are written to `eval_results/<RUN_NAME>/` inside the `lmms-eval` directory.

### 4.2 Agentic

#### 4.2.1 SWE-Bench Evaluation

Evaluation code lives under `Evaluation/Agentic/SWE-Bench/`. We evaluate on two splits: **SWE-bench Verified** (500 instances) and **SWE-bench Lite** (300 instances).

**Environment setup and benchmark download**

Follow the official SWE-bench repository:
[https://github.com/swe-bench/SWE-bench](https://github.com/swe-bench/SWE-bench)

---

**Step 1 — Build instance sandboxes**

Each instance runs inside an isolated Apptainer sandbox. Build all sandboxes:

```bash
cd Evaluation/Agentic/SWE-Bench

# SWE-bench Lite (300 instances)
SANDBOX_DIR=/path/to/sandboxes bash scripts/build_instance_sandboxes_swe_lite.sh

# SWE-bench Verified (500 instances)
SANDBOX_DIR=/path/to/sandboxes bash scripts/build_instance_sandboxes_swe_verified.sh
```

Sandboxes are written to `$SANDBOX_DIR/sweb.eval.x86_64.<instance_id>__latest/`.

---

**Step 2 — Run inference**

Inference uses **mini-swe-agent** backed by a local vLLM server. The agent config (`swebench_klear.yaml`) applies the prompt template used during training. Required environment variables:

| Variable          | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `MODEL_PATH`      | Path to your merged HF model checkpoint                      |
| `SANDBOX_DIR`     | Directory containing the built instance sandboxes from Step 1 |
| `OUTPUT_DIR`      | Where predictions (`preds.json`) are written (default: `<WORK_DIR>/output_swe_<split>`) |
| `OPENAI_API_KEY`  | API key (passed to mini-swe-agent; a dummy key is fine for local vLLM) |
| `VLLM_MODEL_NAME` | Served model alias (default: basename of `MODEL_PATH`)       |
| `NUM_WORKERS`     | Parallel agent workers (default: `1`)                        |

```bash
cd Evaluation/Agentic/SWE-Bench

# SWE-bench Lite
sbatch --export=ALL,MODEL_PATH=/path/to/model,SANDBOX_DIR=/path/to/sandboxes \
    scripts/inference_swe_lite.slurm

# SWE-bench Verified
sbatch --export=ALL,MODEL_PATH=/path/to/model,SANDBOX_DIR=/path/to/sandboxes \
    scripts/inference_swe_verified.slurm
```

Predictions are saved to `$OUTPUT_DIR/preds.json` and per-instance trajectories to `$OUTPUT_DIR/<instance_id>/<instance_id>.traj.json`.

---

**Step 3 — Score predictions**

Pass the `preds.json` produced in Step 2 to the SWE-bench harness:

| Variable           | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `PREDICTIONS_PATH` | Path to `preds.json` from Step 2                             |
| `SANDBOX_DIR`      | Same sandbox directory as Step 1                             |
| `RUN_ID`           | Identifier for this evaluation run (used to name the results directory) |

```bash
cd Evaluation/Agentic/SWE-Bench

# SWE-bench Lite
sbatch --export=ALL,PREDICTIONS_PATH=/path/to/output_swe_lite/preds.json,SANDBOX_DIR=/path/to/sandboxes,RUN_ID=my_run \
    scripts/eval_swe_lite.slurm

# SWE-bench Verified
sbatch --export=ALL,PREDICTIONS_PATH=/path/to/output_swe_verified/preds.json,SANDBOX_DIR=/path/to/sandboxes,RUN_ID=my_run \
    scripts/eval_swe_verified.slurm
```

Results are written to `logs/run_evaluation/<RUN_ID>/`.

---

#### 4.2.2 OOD Evaluation

We evaluate our agentic models on three out-of-distribution benchmarks. All evaluation scripts are under `Evaluation/Agentic/OODEval/` .

---

##### LiveCodeBench

Evaluation code lives under `Evaluation/Agentic/OODEval/livecode/`.

The pipeline (driven by `eval.sh`) has two steps: (1) generate completions with vLLM via `eval.py`, then (2) score them with `lcb_score.py`. Before submitting, set the following variables at the top of `eval.sh`:

| Variable     | Description                               |
| ------------ | ----------------------------------------- |
| `MODEL_PATH` | Path to your merged HF model checkpoint   |
| `VERSION`    | LiveCodeBench release tag (default: `v6`) |
| `SEED`       | Random seed (default: `381`)              |
| `PYBIN`      | Python binary to use (default: `python`)  |

```bash
cd Evaluation/Agentic/OODEval/livecode
mkdir -p logs
sbatch eval.sh
```

Results are written to `outputs_tp1_top_p0.95_seed<SEED>_<VERSION>/` inside the `livecode` directory.

---

##### LongBench v2

Evaluation code lives under `Evaluation/Agentic/OODEval/longbench_v2/LongBench/`.

The pipeline (driven by `eval.sh` one level up) has three steps: (1) launch a vLLM OpenAI-compatible server, (2) generate predictions with 0-shot CoT via `pred.py`, then (3) score with `result.py`. Before submitting, set:

| Variable     | Description                                 |
| ------------ | ------------------------------------------- |
| `MODEL_PATH` | Path to your merged HF model checkpoint     |
| `MODEL_NAME` | Short name used as the served model alias   |
| `PORT`       | Port for the vLLM server (default: `21513`) |
| `PYBIN`      | Python binary (default: `python`)           |
| `VLLM_BIN`   | vllm binary (default: `vllm`)               |

```bash
cd Evaluation/Agentic/OODEval/longbench_v2
mkdir -p logs
sbatch eval.sh
```

Predictions are written to `LongBench/results/` and the final score summary to `LongBench/result.txt`.

---

##### Needle-in-a-Haystack (NIAH)

Evaluation code lives under `Evaluation/Agentic/OODEval/niah/`.

The pipeline (driven by `eval.sh`) has two steps: (1) launch a vLLM server, then (2) run the single-needle NIAH test via `needlehaystack.run`. A GPT-4o judge scores each retrieval. Before submitting, set:

| Variable                            | Description                                                  |
| ----------------------------------- | ------------------------------------------------------------ |
| `MODEL_PATH`                        | Path to your merged HF model checkpoint                      |
| `MODEL_NAME`                        | Short name used as the served model alias                    |
| `OPENAI_API_KEY`                    | OpenAI key for the GPT-4o judge (must be set in environment) |
| `PORT`                              | Port for the vLLM server (default: `21516`)                  |
| `JUDGE_NAME`                        | Judge model (default: `gpt-4o`)                              |
| `CTX_MIN` / `CTX_MAX`               | Context length sweep range (default: `1000`–`32000`)         |
| `CTX_INTERVALS` / `DEPTH_INTERVALS` | Grid resolution (default: `15` / `10`)                       |
| `PYBIN` / `VLLM_BIN`                | Python and vllm binaries (default: `python` / `vllm`)        |

```bash
cd Evaluation/Agentic/OODEval/niah
mkdir -p logs
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
sbatch eval.sh
```

Results are saved under `niah/results/`.

---

## 5. Citation

If you find this work useful, please cite:

```bibtex

```