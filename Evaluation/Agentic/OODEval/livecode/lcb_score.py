# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Borrowed from: https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/utils.py

import multiprocessing
from typing import Dict, Optional
from datasets import load_dataset
from lcb_util import run_test
import traceback
import os, zlib, pickle, argparse
import concurrent.futures
import copy
import json, math
import base64
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
def translate_private_test_cases(encoded_data):
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    return json.loads(original_data)

def check_correctness(problem_to_check: Optional[dict], timeout, debug=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    def _temp_run(problem_to_check, debug, result, metadata_list, timeout):
        try:
            res, metadata = run_test(problem_to_check, debug=debug, timeout=timeout)
            result.append(res)
            metadata_list.append(metadata)
        except Exception as e:
            #traceback.print_exc(10)
            result.append([-1 for i in range(len(problem_to_check['input_output']))])
            metadata_list.append(e)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()

    total_timeout = (timeout + 1) * len(problem_to_check['input_output']) + 10
    p = multiprocessing.Process(target=_temp_run, args=(problem_to_check, debug, result, metadata_list, timeout))
    p.start()
    p.join(timeout=total_timeout + 1)
    if p.is_alive():
        p.kill()

    judge_value = bool(result and np.all(np.array(result[0]) > 0))
    if False:
    #if judge_value == False:
        if result:
            print(result[0], metadata_list[0])
        else:
            print("Time Limit Exceeded.")
    return judge_value

def get_starter_code(header_str):
    if "def " in header_str:
        starter_code = header_str.split("def")[1].split("(")[0].strip()
    else:
        starter_code = header_str

    return starter_code

def combine(id_to_results, version, identifier="question_id"):
    dataset = load_dataset("livecodebench/code_generation_lite", version_tag=version, trust_remote_code=True)
    dataset = dataset.map(
        lambda example: {
            "private_test_cases": translate_private_test_cases(
                example["private_test_cases"]
            )
        },
        writer_batch_size=100,
    )
    questions = dataset['test'].to_list()

    total_questions = len(questions)

    combined_results = {}
    for problem in questions:
        testcases = []
        for k, v in problem.items():
            if "cases" in k:
                if isinstance(v, str):
                    v = json.loads(v)
                testcases.extend(v)
        #print(testcases)
        #return combined_results, total_questions
        starter_code = get_starter_code(problem['starter_code'])

        combined_results[problem[identifier]] = {
            'input_output': testcases,
            'starter_code': starter_code,
            'question_id': problem[identifier],
            'generation': id_to_results[problem[identifier]],
            'difficulty': problem['difficulty']
        }

    print(f"Matched {len(combined_results)} question-generation pairs.")
    return combined_results, total_questions

def update_results(result, timeout=5):

    response_entry = {
            "correctness": None,
            "reason": None,
    }

    problem_to_check = copy.deepcopy(result)
    curr_res = check_correctness(problem_to_check, timeout=timeout)

    if curr_res:
        response_entry["correctness"] = True
        response_entry["reason"] = ""
    else:
        response_entry["correctness"] = False
        response_entry["reason"] = "Code is incorrect."

    return response_entry

def compute_accuracy(output_folder, output_name, id_to_results, version, timeout=5):

    results, total_questions = combine(id_to_results, version)
    print(f"Loaded {total_questions} questions for evaluation.")

    total_correct = 0
    total_finish = 0
    records = {}

    with ProcessPoolExecutor(max_workers=32) as executor:

        future_to_task = {}
        token_usages = {}
        correctness = {'easy':0, 'medium': 0, 'hard': 0}
        total = {'easy':0, 'medium': 0, 'hard': 0}
        for idx, (q_id, result) in enumerate(results.items()):
            future_to_task[
                executor.submit(
                    update_results, result
                )
            ] = q_id

        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc="Processing Generations",
        ):
            q_id = future_to_task[future]
            response_entry = future.result()
            total_correct += response_entry["correctness"]
            total_finish += 1
            total[results[q_id]["difficulty"]] += 1
            correctness[results[q_id]["difficulty"]] += response_entry["correctness"]
            records[q_id] = response_entry
    
    result_str = f"{output_name},{total_correct * 100. / total_questions:.3f},"
    print(f"Final acc: {total_correct}/{total_questions} = {total_correct * 100. / total_questions:.3f}")
    for key in total:
        print(f"{key} acc: {correctness[key]}/{total[key]} = {correctness[key] * 100. / total[key]:.3f}")
        result_str += f"{correctness[key] * 100. / total[key]:.3f},"
    result_str += f"{total_questions}\n"
    with open(os.path.join(output_folder, "results.csv"), 'a') as f:
        f.write(result_str)
    return records

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified verifier for different datasets/tasks."
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        help="Name of the model",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="trust_remote_code option used in huggingface models",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v4_v5",
        help="whether to use full set of tests (slower and more memory intensive evaluation)",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--seed", type=int, default=111, help="seed"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Temperature for sampling"
    )
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for sampling")
    parser.add_argument(
        "--max_tokens", type=int, default=32768, help="Max tokens for sampling"
    )
    parser.add_argument(
        "--multiprocess",
        default=0,
        type=int,
        help="Number of processes to use for generation (vllm runs do not use this)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="Tensor parallel size for vllm",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Tensor parallel size for vllm",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Dtype for vllm")
    # Added to avoid running extra generations (it's slow for reasoning models)
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./",
        help="result",
    )
    args = parser.parse_args()
    if args.top_p < 1:
        if args.tensor_parallel_size != 1 and args.tensor_parallel_size !=8:
            foldername = "outputs_tp{}_top_p{}_seed{}".format(args.tensor_parallel_size, args.top_p, args.seed)
        else:
            foldername = "outputs_top_p{}_seed{}".format(args.top_p, args.seed)
    if args.version != 'v4_v5': foldername += f'_{args.version}'
    output_folder = os.path.join(args.result_dir, foldername)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_name = "livecodebench_" + os.path.basename(args.model)
    output_name = output_name + ".jsonl"
    score_name = output_name + "_score.json"

    output_datapath = os.path.join(output_folder, output_name)
    score_datapath = os.path.join(output_folder, score_name)
    print("reading from %s" % output_datapath)
    df = pd.read_json(output_datapath, lines=True)
    id_to_results = dict(zip(df['task_id'], df['output']))
    output_list = compute_accuracy(output_folder, output_name, id_to_results, args.version, timeout=args.timeout)

    with open(score_datapath, "w", encoding='utf-8') as f:
        json.dump(output_list, f, indent=4)

    

