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



### 2.2 Agentic



---

## 3. Training

TODO

---

## 4. Evaluation

TODO

---

## 5. Citation

If you find this work useful, please cite:

```bibtex

```