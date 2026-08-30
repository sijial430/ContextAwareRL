# Agentic RL reproduction

This directory contains a verified, one-step reproduction of the Qwen3-8B
agentic training path described in the repository README. It runs Mini-SWE
rollouts, GRPO with the dual-clip policy objective, the choose-trajectory (CT)
auxiliary loss, a four-rank FSDP2 optimizer step, and checkpointing.

## Full upstream-sized one-step run

The full training configuration was reproduced through one optimizer step with
the upstream batch and sampling shape (16 prompts x 8 samples), rather than the
compact smoke-test shape. Evaluation was disabled for this one-step check, and
checkpointing was set to every step so the update is durably captured.

- Upstream revision: `0683df934b7c2698fb97df5a478664e1b310db2f`
- Beaker experiment: [01M183DM2R5H8P561SRRBZT6FX](https://beaker.org/ex/01M183DM2R5H8P561SRRBZT6FX)
- Beaker job: `01M183DM6DDR3GTJF93HB2FJ2W`
- W&B: [f71wkgqa](https://wandb.ai/sijial-ai2/ContextAwareRL-reproduction/runs/f71wkgqa)
- Hardware: four H100 GPUs; two colocated vLLM engines with TP=2; four-rank FSDP2
- Process result: exit code 0 after 1h14m20s
- Independent Weka verifier: `01M183TEQSRPM27V9VS5Q20QSR`, exit code 0
- Verification marker: `FULL_CONFIG_TRAINING_STEP_VERIFIED`

All persistent outputs are under:

```text
/weka/nora-default/sijial/workspace/contextaware-rl/qwen3-8b-full-config-1-step/01M183DM2R5H8P561SRRBZT6FX/
```

That directory contains the complete `run.log`, local W&B state, all 128
trajectory JSON files, dumped training data, checkpoint manifests, and the
`SUCCESS` marker. The verified final checkpoint is `checkpoints/global_step_2`
(92 GiB), containing four policy model shards, four optimizer shards, per-rank
extra state, trainer/RL/CT dataloader state, FSDP metadata, and Hugging Face
config/tokenizer files. An interval checkpoint was also saved at
`checkpoints/global_step_1` immediately after the update.

Observed step evidence:

| Signal | Value |
| --- | ---: |
| Training prompts x samples | 16 x 8 |
| Training sequences | 128 |
| Persisted trajectory files | 128 |
| Padded sequence length | 32,768 tokens |
| Average response length | 30,910.0625 tokens |
| Average task reward | 0.0 |
| Final loss | 5.4681828487e-7 |
| Policy loss | 0.0 |
| Policy entropy | 0.2034340845 |
| Policy KL | 0.0010936365 |
| Choose-trajectory loss | 0.9578492418 |
| Choose-trajectory accuracy | 0.5625 |
| Choose-trajectory coefficient | 0.001 |
| Gradient norm | 0.5257123709 |
| Policy training time | 304.25 s |
| Complete training-batch time | 56m59s |

All task rewards were zero, so the GRPO policy-loss term was zero. The
choose-trajectory auxiliary objective produced a finite nonzero loss and
gradient, exercising and updating the combined GRPO+CT policy path. Some
rollouts reached the repository's existing Qwen3 context boundary (36,865
input tokens plus a 4,096-token request exceeds the 40,960-token model limit
by one); the upstream generator catches and persists these episodes instead of
aborting the batch.

## Compact pipeline smoke run

- Upstream revision: `0683df934b7c2698fb97df5a478664e1b310db2f`
- Beaker experiment: [01M16N5RDS1DTCJJKD16804S74](https://beaker.org/ex/01M16N5RDS1DTCJJKD16804S74)
- Beaker job: `01M16N5RHBA6T0CGM460XTN6RS`
- W&B: [pmg508ct](https://wandb.ai/sijial-ai2/ContextAwareRL-reproduction/runs/pmg508ct)
- Hardware: four H100 GPUs
- Process result: exit code 0
- Verification marker: `TRAINING_STEP_VERIFIED`

The run used the first official SWE-Gym record (`getmoto__moto-7365`) and the
first eight official contrastive pairs. Four rollout samples were generated
with a three-turn agent limit. This is intentionally a pipeline smoke run, not
the full 370-row Qwen3 training schedule.

Observed step evidence:

| Signal | Value |
| --- | ---: |
| Valid rollout samples | 4 |
| Rollout generation time | 25.39 s |
| Average rollout response length | 1666.75 tokens |
| Average task reward | 0.0 |
| Policy training time | 14.59 s |
| CT loss | 1.9484333992 |
| CT accuracy | 0.5 |
| CT coefficient | 0.001 |
| Gradient norm | 0.4777443707 |
| Complete step time | 96.25 s |

The short agents did not submit passing patches, so the GRPO reward component
was zero. The CT loss and nonzero gradient nevertheless exercised and updated
the policy through the repository's combined GRPO+CT training path.

`global_step_1` was verified to contain all four policy model shards, all four
optimizer shards, per-rank extra state, trainer and RL dataloader state, CT
dataloader state, the FSDP config, and Hugging Face tokenizer/config files.
SkyRL also wrote its normal epoch-end `global_step_2` checkpoint.

## Why PRoot is used

Beaker's unprivileged OCI jobs cannot create the namespaces required by a
nested Apptainer execution. The sandbox image is still pulled as the exact
official `.sif`; it is not converted into a Docker runtime image.
`ProotSIFEnvironment` dumps the SIF filesystem partition with Apptainer,
extracts it with `unsquashfs`, and runs the resulting filesystem with PRoot.
The live preflight verified `/testbed` and its Git worktree before training.

The SIF produced by the verified run had SHA-256
`43b017a1195e84a49aee48c70d1a2b91ea03c12f41a4d1de4690ad5ff1dd3997`.
Apptainer's OCI-to-SIF packaging is not byte-deterministic, so a later pull can
have a different SIF hash while containing the same source OCI image.

## Run it again

The checked-in full-check spec points to the exact packaged source snapshot
used by the successful run and mounts `nora-default` at `/weka/nora-default`.
With Beaker authenticated for the `ai2/autodiscovery` workspace and the
`HF_TOKEN` and `SIJIAL_WANDB_API_KEY` secrets available, submit it with:

```bash
beaker experiment create reproduction/beaker/run_qwen3_fullcheck.yaml \
  --workspace ai2/autodiscovery \
  --name contextaware-qwen3-fullcheck
```

The compact smoke run remains available with:

```bash
beaker experiment create reproduction/beaker/run_qwen3_smoke.yaml \
  --workspace ai2/autodiscovery \
  --name contextaware-qwen3-smoke
```

To package a new code/data snapshot, create and upload a replacement input
dataset, then replace the dataset ID in the selected Beaker YAML:

```bash
input_dir="$(mktemp -d /tmp/contextaware-input.XXXXXX)"
bash reproduction/beaker/package_input.sh "$input_dir"
beaker dataset create "$input_dir" \
  --name contextaware-rl-qwen3-smoke-src \
  --workspace ai2/autodiscovery
```

The full-check launcher downloads `Qwen/Qwen3-8B`, pulls all 16 selected
official SWE-Gym images as SIFs, installs the pinned `uv` environment, builds
the official data subsets, verifies PRoot execution, and launches the
four-GPU training step. Outputs are placed below
`/weka/nora-default/sijial/workspace/contextaware-rl/` in an
experiment-specific directory.

The Beaker specs intentionally do not declare a result-dataset path. Training
checkpoints, trajectories, logs, W&B local state, prepared assets, manifests,
and verification markers are durable only under the requested Weka workspace.
Container-local paths such as `/workspace` and `/tmp` are used only for
ephemeral installation, caches, extracted sandboxes, and model inputs. The two
`migrate_*_to_weka.yaml` specs copy and byte-verify result datasets created by
older versions of the launchers before those source datasets are removed.

## Compatibility changes

- `mini_swe_utils.py` resolves local SIFs for the `proot_sif` environment and
  constructs `ProotSIFEnvironment`.
- `proot_sif_environment.py` provides the SIF extraction and PRoot execution
  backend required by unprivileged Beaker jobs.
- `main_mini_swe.py` passes the configured served-model alias to Mini-SWE, so
  LiteLLM calls the Qwen model exposed by SkyRL's vLLM endpoint even though the
  policy weights live at a local filesystem path.
- The Beaker launcher installs `libnuma1`, required by SkyRL's CPU offload
  affinity code.
- Qwen3 thinking is disabled for the short rollout smoke test through vLLM's
  chat-template kwargs. CT thinking is disabled independently by the upstream
  training option. Without this smoke-only setting, Qwen3 consumes the
  1,024-token allowance inside `<think>` before emitting a bash action.

The complete 370-row schedule can be restored from this verified one-step
configuration by using the full official parquet input and epoch count from
`run_mini_swe_qwen3_8b.sh`, after provisioning the remaining SIFs.
