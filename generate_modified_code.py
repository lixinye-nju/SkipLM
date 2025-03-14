import logging
import argparse
import time
import ast
from tqdm import tqdm
import os
import json
import openai


from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from collections import defaultdict

from task_processing import TaskProcessingEngine
from utils import write_jsonl, extract_code_fences


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with vLLM on Qwen model using a dataset")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct", help="Base model to use")
    parser.add_argument("--base_url", type=str, required=True, help="IP address for openai-compatible server.")
    parser.add_argument("--num_of_sequences", type=int, default=1, help="Number of sequences to generate")
    parser.add_argument("--frequency_penalty", type=float, default=0.0, help="Frequency penalty")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty")
    parser.add_argument("--dataset_path", type=str, required=True, help="Input dataset file (huggingface.datasets compatible format).")
    parser.add_argument("--split", type=str, default="train", help="Input data split.")
    parser.add_argument("--output_path", type=str, required=True, help="Output file")
    parser.add_argument("--num_proc", type=int, default=128, help="Maximum number of processes for processing")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Maximum length for LLM.")
    return parser.parse_args()


def load_source_dataset(args):
    """Loading dataset in huggingface.datasets compatible format."""
    src_data = load_dataset(args.dataset_path, split=args.split)
    
    dataset_name = args.dataset_path.split("/")[-1]
    src_data = src_data.map(
        lambda x, idx: {"task_id": f"{dataset_name}/{idx}"},
        with_indices=True,
        num_proc=args.num_proc
    )
    
    if "code" in src_data.column_names:
        return src_data
    
    def extract_code(example):
        code_fences = extract_code_fences(example["output"])
        if len(code_fences) > 0:
            return {"code": code_fences[0]}
        else:
            return {"code": ""}
    
    src_data = src_data.map(extract_code, num_proc=args.num_proc)
    src_data = src_data.filter(lambda x: len(x["code"]) > 0, num_proc=args.num_proc)
    
    return src_data


def load_cached_data(args):
    """Load previously cached results and filter src_data accordingly."""
    num_generated_samples = defaultdict(int)
    total_num = 0
    if os.path.isfile(args.output_path):
        try:
            # Try to load as dataset first
            dst_data = load_dataset("json", data_files=[args.output_path], split="train")
        except:
            # Fall back to manual loading if schema exception
            try:
                with open(args.output_path, "r") as f:
                    lines = f.readlines()
                    
                objs = [json.loads(line) for line in lines if len(line.strip()) > 0]
                dst_data = Dataset.from_list(objs)
            except Exception as e:
                print(f"Warning: Couldn't load cache file properly: {e}")
                return num_generated_samples, total_num
                
        print(f"Loaded {len(dst_data)} cached examples from {args.output_path}.")
    
        for item in dst_data:
            num_generated_samples[item["task_id"]] += 1
        total_num = len(dst_data)
        
        return num_generated_samples, total_num
    else:
        return num_generated_samples, total_num
    

def get_sampling_params(args):
    """Configure and return sampling parameters based on arguments."""
    if args.num_of_sequences == 1:        
        return dict(
            n=args.num_of_sequences, 
            temperature=0.0, 
            max_tokens=4096,
            frequency_penalty=args.frequency_penalty, 
            presence_penalty=args.presence_penalty,
        )
    else:
        return dict(
            n=args.num_of_sequences, 
            temperature=0.4, 
            top_p=0.95, 
            max_tokens=4096,
            frequency_penalty=args.frequency_penalty, 
            presence_penalty=args.presence_penalty,
        )



PROMPT_TEMPLATE = r"""
```
{code}
```
Please generate `modified code` and `instruction` (in one sentence).

Format:
instruction: 
```text
# YourInstruction
```

modified_code:
```
# Your Code
```
"""

def process_single_example(example: Dict[str, Any], 
                           client: openai.Client,
                           model_name_or_path: str,
                           sampling_params: Dict[str, Any]):
    code = example["code"]
    query = PROMPT_TEMPLATE.format(code=code)
    messages = [{"role": "user", "content": query}]
    response = client.chat.completions.create(
        messages=messages,
        model=model_name_or_path,
        **sampling_params
    )
    
    text = response.choices[0].message.content
    code_fences = extract_code_fences(text)
    try:
        instruction = code_fences[0]
        modified_code = code_fences[1]
        return {
            "task_id": example["task_id"],
            "instruction": instruction,
            "code": code,
            "modified_code": modified_code
        }

    except Exception as e:
        logging.warning(f"{type(e)}: {e}\nCause:\n{text}")
        # return None to skip this generation
        # return failed result if keep
        return {
            "task_id": example["task_id"],
            "instruction": "",
            "code": code,
            "modified_code": modified_code
        }



if __name__ == "__main__":
    # Set up basic logging configuration
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(process)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    args = parse_args()

    # Init openai client
    client = openai.Client(
        base_url=args.base_url,
        api_key="EMPTY"
    )
    model_name_or_path = args.model
    
    # Get sampling parameters
    sampling_params = get_sampling_params(args)

    # Load source dataset
    src_data = load_source_dataset(args)
    
    
    # Init output path
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    
    # Load cached data
    num_generated_samples, total_num = load_cached_data(args)
    logging.info(f"Load {total_num} from {args.output_path}.")
    
    # Filter dataset to include only tasks that need more samples
    src_data = src_data.filter(
        lambda x: num_generated_samples[x["task_id"]] < args.num_of_sequences
    )
    print(src_data)
    
    total_generation_num = len(src_data)
    logging.info(f"Need to generate {total_generation_num} new samples across {len(num_generated_samples)} tasks")
    
    if total_generation_num == 0:
        logging.info("All samples already generated. Exiting.")
    
    all_tasks = [
        x
        for x in src_data
        for _ in range(args.num_of_sequences - num_generated_samples[x["task_id"]])
    ]
    
    def init_processor(worker_id=None):
        return {
            "worker_id": worker_id
        }
    
    def process_task(processor, task):
        return process_single_example(task, client, model_name_or_path, sampling_params)
    
    def handle_results(results):
        write_jsonl(args.output_path, results, append=True)
        
    engine = TaskProcessingEngine(
        processor_initializer=init_processor,
        processor_args={},
        process_func=process_task,
        result_handler=handle_results,
        num_workers=args.num_proc
    )
    
    logging.info("Creating sample tasks...")
    
    logging.info("Starting task processing...")
    engine.process_tasks(all_tasks)
    
    logging.info(f"Processing completed")

