import json
import argparse
import re
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
import torch

# Parse arguments
parser = argparse.ArgumentParser(description='Process ChooseImg dataset with pass@8 evaluation')
parser.add_argument('--start_index', type=int, default=None, help='Start index (inclusive, 0-based)')
parser.add_argument('--end_index', type=int, default=None, help='End index (exclusive, 0-based)')
args = parser.parse_args()

# Load model
model_path = "/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/Qwen3-VL-8B-Instruct"

print("Loading model...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)
print("Model loaded successfully!")

# Read input jsonl file
input_file = "/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/DataGen/DataSource/ChooseImgSFT/ChooseImg_rl_gt.jsonl"

# Read all data
print("Reading input file...")
data_list = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data_list.append(json.loads(line))

# Slice data based on start_index and end_index
if args.start_index is not None or args.end_index is not None:
    start_idx = args.start_index if args.start_index is not None else 0
    end_idx = args.end_index if args.end_index is not None else len(data_list)
    data_list = data_list[start_idx:end_idx]
    print(f"Processing items from index {start_idx} to {end_idx-1} (total {len(data_list)} items)")
    output_file = f"/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/DataGen/DataSource/ChooseImgSFT/pass8_qwen3_results/pass8_results_{start_idx}_{end_idx}.jsonl"
else:
    print(f"Total {len(data_list)} items to process")
    output_file = "/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/DataGen/DataSource/ChooseImgSFT/pass8_qwen3_results/pass8_results.jsonl"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Helper function to check if the response contains the ground truth answer (A/B/C)
def check_answer_match(response, gt):
    """
    Check if the response matches the ground truth answer (A, B, or C).
    Looks for patterns like 'A', 'A.', 'Option A', 'Answer: A', etc.
    """
    gt = gt.strip().upper()
    response = response.strip()

    # Pattern: match the answer letter in common response formats
    # e.g., "A", "A.", "A. The first image", "The answer is A", "Option A", etc.
    pattern = r'(?:^|\b)' + re.escape(gt) + r'(?:\b|\.|\s|$)'
    if re.search(pattern, response, re.IGNORECASE):
        return True

    return False

# Helper function to compute pass8 for a given item with two images
def compute_pass8(images, problem, gt):
    # Split problem text by <image> placeholders and interleave with actual images
    parts = problem.split("<image>")
    content = []
    img_idx = 0
    for i, part in enumerate(parts):
        if i > 0 and img_idx < len(images):
            content.append({"type": "image", "image": images[img_idx]})
            img_idx += 1
        text = part.strip("\n")
        if text:
            content.append({"type": "text", "text": text})

    messages = [{"role": "user", "content": content}]

    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Run inference 8 times
    success_count = 0
    for _ in range(8):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=1.0,
            do_sample=True
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if check_answer_match(response, gt):
            success_count += 1

    return success_count

# Process each item and write results immediately
print(f"Writing results to {output_file}...")
total_pass8_sum = 0
total_items = 0

with open(output_file, 'w', encoding='utf-8') as f:
    for item in tqdm(data_list, desc="Processing items"):
        pass8_count = compute_pass8(
            item["images"],
            item["problem"] + " Directly provide the final answer without any explanation.",
            item["answer"]
        )

        pass8_rate = pass8_count / 8.0
        total_pass8_sum += pass8_rate
        total_items += 1

        result = {
            "images": item["images"],
            "problem": item["problem"],
            "answer": item["answer"],
            "pass8_count": pass8_count,
            "pass8_rate": pass8_rate
        }
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        f.flush()

        # Print running average every 100 items
        if total_items % 100 == 0:
            avg_pass8_rate = total_pass8_sum / total_items
            print(f"[{total_items}/{len(data_list)}] Running avg pass@8 rate: {avg_pass8_rate:.4f}")

# Print final statistics
avg_pass8_rate = total_pass8_sum / total_items if total_items > 0 else 0
print(f"\n===== Final Results =====")
print(f"Total items: {total_items}")
print(f"Average pass@8 rate: {avg_pass8_rate:.4f}")
print(f"Results saved to {output_file}")
