"""Replace the chat template in a saved checkpoint with the thinking model's template.

Usage:
    python scripts/replace_chat_template.py <checkpoint_dir> [--template <chat_template.json>]

Example:
    python scripts/replace_chat_template.py /scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/EasyR1_qwen3_Modify/checkpoints/easy_r1/qwen3_8B_instruct_aug_logits_v2/global_step_400/actor/huggingface_400
"""

import argparse
import json
import os

DEFAULT_TEMPLATE = "/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/Qwen3-VL-8B-Thinking/chat_template.json"


def main():
    parser = argparse.ArgumentParser(description="Replace chat template in a checkpoint with thinking template")
    parser.add_argument("checkpoint_dir", help="Path to the checkpoint directory")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="Path to chat_template.json (default: Qwen3-VL-8B-Thinking)")
    args = parser.parse_args()

    ckpt = args.checkpoint_dir

    # Load thinking template
    with open(args.template) as f:
        template = json.load(f)["chat_template"]

    # 1. Replace chat_template.jinja
    jinja_path = os.path.join(ckpt, "chat_template.jinja")
    with open(jinja_path, "w") as f:
        f.write(template)
    print(f"Updated {jinja_path}")

    # 2. Update tokenizer_config.json if it has chat_template field
    tc_path = os.path.join(ckpt, "tokenizer_config.json")
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            tc = json.load(f)
        if "chat_template" in tc:
            tc["chat_template"] = template
            with open(tc_path, "w") as f:
                json.dump(tc, f, indent=2, ensure_ascii=False)
            print(f"Updated {tc_path}")

    print("Done!")


if __name__ == "__main__":
    main()
