import json
from pathlib import Path

def merge_and_filter_jsonl():
    input_dir = Path("/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/DataGen/DataSource/ChooseImgSFT/pass8_qwen3_results")
    output_file = Path("/scratch/gpfs/PMITTAL/peiyang/px4668/ImageRL/DataGen/DataSource/ChooseImgSFT/ChooseImg_rl_gt_pass8_qwen3.jsonl")

    jsonl_files = sorted(input_dir.glob("*.jsonl"))    
    total_lines = 0
    filtered_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for jsonl_file in jsonl_files:
            file_lines = 0
            file_filtered = 0
            
            with open(jsonl_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        total_lines += 1
                        file_lines += 1
                        
                        pass8_rate = data.get("pass8_rate")
                        
                        if pass8_rate is not None and 0.25 <= pass8_rate <= 0.63:
                            data.pop("pass8_count", None)
                            data.pop("pass8_rate", None)
                            json.dump(data, outfile, ensure_ascii=False)
                            outfile.write('\n')
                            filtered_lines += 1
                            file_filtered += 1
                    except json.JSONDecodeError as e:
                        print(f"Warning: skip invalid JSON line: {e}")
                        continue
            
            print(f"  File {jsonl_file.name}: Total lines {file_lines}, filtered {file_filtered}")
    
    print(f"\nProcessing completed!")
    print(f"Total lines: {total_lines}")
    print(f"Filtered lines: {filtered_lines}")
    print(f"Filtered ratio: {filtered_lines/total_lines*100:.2f}%" if total_lines > 0 else "No data")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    merge_and_filter_jsonl()

