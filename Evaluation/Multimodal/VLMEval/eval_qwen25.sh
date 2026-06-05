#!/bin/bash
set -x
GPUS=${GPUS:-1}  
GPUS_PER_TASK=${GPUS_PER_TASK:-1}

declare -a models=( \
"Qwen2.5-VL-7B-Best"
)


datasets="MathVista_MINI MathVerse_MINI MathVision MMMU_DEV_VAL MMMU_Pro_10c VStarBench MMStar BLINK"

LOG_DIR="logs_eval" 

module load proxy/default
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
export LMUData='<PATH_TO_LMUDATA>'

export THINK='true'
export TORCH_DTYPE='bfloat16'
for ((i=0; i<${#models[@]}; i++)); do

  model=${models[i]}

  if [[ "$model" =~ 38B|78B ]]; then
      GPUS_PER_TASK=8
  else
      GPUS_PER_TASK=1
  fi

  mkdir -p "${LOG_DIR}/${model}"

  sbatch --gres=gpu:${GPUS} \
    --ntasks=1 \
    --cpus-per-task=16 \
    --job-name="eval_${model}" \
    --time=24:00:00 \
    --output="${LOG_DIR}/${model}/slurm_%j.out" \
    --wrap="module load proxy/default && export OPENAI_API_KEY=${OPENAI_API_KEY} && export LMUData=${LMUData} && export SKIP_MD5_CHECK=1 && export THINK=${THINK} && export TORCH_DTYPE=${TORCH_DTYPE} && python -u run.py --data ${datasets} --model ${model} --verbose"

done