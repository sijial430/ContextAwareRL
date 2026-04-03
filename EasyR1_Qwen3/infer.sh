
srun --job-name=infer \
     --nodes=1 \
     --ntasks-per-node=1 \
     --gres=gpu:1 \
     --time=00:10:00 \
     --pty \
     python test_qwen3_infer.py