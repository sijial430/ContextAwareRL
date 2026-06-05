Inference:
1. sbatch scripts/inference_swe_lite.slurm / scripts/inference_swe_verified.slurm
2. For each sandbox, use --writable-tmpfs to write only to in-memory tmpfs
3. All inference results (preds.json + trajectories) are saved to $OUTPUT_DIR

Eval:
1. Pass $OUTPUT_DIR/preds.json as PREDICTIONS_PATH to scripts/eval_swe_lite.slurm / eval_swe_verified.slurm
2. For each sandbox, a fresh copy is made for evaluation; the original sandbox is not modified
