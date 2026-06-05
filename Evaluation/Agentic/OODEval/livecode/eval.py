from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import os
import json
import argparse
from datasets import load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_vllm_model(args):
    torch_dtype = torch.bfloat16
    tensor_parallel_size = args.tensor_parallel_size

    print("load model from %s" % args.model)
    print("torch_dtype:", torch_dtype)
    print("tensor_parallel_size:", tensor_parallel_size)

    model_vllm = LLM(
        args.model,
        tokenizer=args.model,
        dtype=torch_dtype,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )

    return model_vllm

def generate_prompt_old(prompt, starter_code=""):
    instruction = "<|im_start|>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    templating_parameters = {
        "stdin": "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition. {prompt}", 
        "with_starter_code": "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution. {prompt}\n Starter code:\n{starter_code}"
    }
    if starter_code != "":
        _input = templating_parameters["with_starter_code"].format(
            prompt=prompt, starter_code=starter_code
        )
    else:
        _input = templating_parameters["stdin"].format(prompt=prompt)
    final_prompt = instruction + "<|im_start|>user\n" + _input + "<|im_end|>\n<|im_start|>assistant\nthink\n"
    return final_prompt
def generate_prompt(prompt, starter_code, is_math):
    templating_parameters = {
        "stdin": "Problem:\n{prompt}\n\n Generate executable Python code to solve the problem. The code should take stdin as input and print the output. Please write the code in the following format:\n```python\n# Your solution code here\n```", 
        "with_starter_code": "Problem:\n{prompt}\n\n Generate an executable Python function to solve the problem. Return the function body without invoking it at the final solution. Please write the function starting with the following header:\n```\n{starter_code}\n```"
    }
    if is_math: return "Problem:\n" + prompt
    if starter_code is not None:
        _input = templating_parameters["with_starter_code"].format(
            prompt=prompt, starter_code=starter_code
        )
    else:
        _input = templating_parameters["stdin"].format(prompt=prompt)
    return _input
def preprocess_livecodebench_chatml_template_old(version):
    data_list = load_dataset("livecodebench/code_generation_lite", version_tag=version, trust_remote_code=True)['test'].to_list()
    prompt_list = []
    qid_list = []
    for item in data_list:
        question = item['question_content'].strip()
        final_prompt = generate_prompt(question, item['starter_code'])
        prompt_list.append(final_prompt)
        qid_list.append(item['question_id'])
    
    return prompt_list, qid_list

def _is_non_thinking_model(model_path):
    # Qwen3-Coder-* (and any *-Instruct variant explicitly tagged non-thinking)
    # do not have <think> in their chat template. Forcing <think> prefill makes
    # them emit <|im_end|> immediately, producing empty outputs.
    name = os.path.basename(os.path.normpath(model_path)).lower()
    return "coder" in name


def preprocess_livecodebench_chatml_template(version, model_path):
    non_thinking = _is_non_thinking_model(model_path)
    if non_thinking:
        system_prompt = "<|im_start|>system\nYou are Qwen, a helpful AI assistant.<|im_end|>\n"
    else:
        system_prompt = "<|im_start|>system\nYou are a helpful and harmless assistant. You should think step-by-step.<|im_end|>\n"

    data_list = load_dataset("livecodebench/code_generation_lite", version_tag=version)['test'].to_list()
    prompt_list = []
    qid_list = []
    for item in data_list:
        question = item['question_content'].strip()
        starter_code = None
        if item['starter_code'] != "":
            starter_code = item['starter_code']

        user_prompt = generate_prompt(question, starter_code, False)
        text = system_prompt
        text += "<|im_start|>user\n" + user_prompt + "<|im_end|>\n"
        if non_thinking:
            text += "<|im_start|>assistant\n"
        else:
            text += "<|im_start|>assistant\n<think>\n"

        prompt_list.append(text)
        qid_list.append(item['question_id'])

    return prompt_list, qid_list
def preprocess_livecodebench_chatml_template_original(version):
    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant. You should think step-by-step.<|im_end|>\n"

    code_instruction_nostartercode = """Write Python code to solve the problem. Please place the solution code in the following format:\n```python\n# Your solution code here\n```"""
    code_instruction_hasstartercode = """Please place the solution code in the following format:\n```python\n# Your solution code here\n```"""


    data_list = load_dataset("livecodebench/code_generation_lite", version_tag=version)['test'].to_list()
    prompt_list = []
    qid_list = []
    for item in data_list:
        question = item['question_content'].strip()
        if item['starter_code'] != "":
            question += "\n\n" + "Solve the problem starting with the provided function header.\n\nFunction header:\n" + "```\n" + item['starter_code'] + "\n```"
            question += "\n\n" + code_instruction_hasstartercode
        else:
            question += "\n\n" + code_instruction_nostartercode

        final_prompt = instruction + "<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"

        prompt_list.append(final_prompt)
        qid_list.append(item['question_id'])
    
    return prompt_list, qid_list
def get_prompt_list(args):

    prompt_list, qid_list = preprocess_livecodebench_chatml_template(args.version, args.model)


    print("number of test samples in the dataset:", len(prompt_list))
    print("non-thinking prompt:", _is_non_thinking_model(args.model))

    return prompt_list, qid_list


def main():
    parser = argparse.ArgumentParser(description="lcb")
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
        "--release_version",
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
    parser.add_argument(
        "--version",
        type=str,
        default="v4_v5",
        help="version",
    )
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--enforce_eager", action="store_true")

    args = parser.parse_args()
    model_vllm = load_vllm_model(args)

    prompt_list, qid_list = get_prompt_list(args)

    ## run inference
    print("args.max_tokens:", args.max_tokens)
   
    print("Start!!!!")
    if args.top_p < 1:
        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)
        print("args.seed:", args.seed)
        print("args.top_p:", args.top_p)
        print("args.temperature:", args.temperature)

    output_list = []
    # for i, prompt in enumerate(tqdm(prompt_list)):
    for i in tqdm(range(0, len(prompt_list), args.batch_size)):
        print(i)
        batch_prompts = prompt_list[i:i+args.batch_size]
        if qid_list:
            batch_qids = qid_list[i:i+args.batch_size]

        outputs = model_vllm.generate(batch_prompts, sampling_params)
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text

            if "<|im_end|>" in generated_text:
                idx = generated_text.index("<|im_end|>")
                generated_text = generated_text[:idx]
            if "<|end_of_text|>" in generated_text:
                idx = generated_text.index("<|end_of_text|>")
                generated_text = generated_text[:idx]
            if "<|eot_id|>" in generated_text:
                idx = generated_text.index("<|eot_id|>")
                generated_text = generated_text[:idx]
            
            if qid_list:
                qid = batch_qids[j]
                output_dict = {"task_id": qid, "output": generated_text}
                output_list.append(output_dict)
            else:
                output_dict = {"output": generated_text}
                output_list.append(output_dict)

    ## write to output_datapath
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
    
    output_datapath = os.path.join(output_folder, output_name)

    print("writing to %s" % output_datapath)
    with open(output_datapath, "w", encoding='utf-8') as f:
        for output in output_list:
            if type(output) == dict:
                f.write(json.dumps(output) + "\n")
            else:
                f.write(output + "\n")


if __name__ == "__main__":
    main()


