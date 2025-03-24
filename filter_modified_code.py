import os
import re
import json
import logging
import argparse
import openai
import difflib

from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from collections import defaultdict


from task_processing import TaskProcessingEngine
from utils import write_jsonl, extract_code_fences



def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with vLLM to filter a dataset.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct", help="Base model to use")
    parser.add_argument("--base_url", type=str, required=True, help="IP address for openai-compatible server.")
    parser.add_argument("--json_path", type=str, required=True, help="Input json file.")
    parser.add_argument("--output_path", type=str, required=True, help="Output file")
    parser.add_argument("--use-cot_prompt", type=str, default=False, help="Whether to use CoT prompt.")
    parser.add_argument("--num_proc", type=int, default=128, help="Maximum number of processes for processing")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Maximum length for LLM.")
    return parser.parse_args()


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
    


def process_single_example(example: Dict[str, str],
                           client: openai.Client,
                           model_name_or_path: str,
                           sampling_params: Dict[str, Any],
                           use_cot_prompt: bool = False,
                           mininum_lines: int = 8,
                           diff_frac_threshold: float=0.2):
    task_id = example["task_id"]
    instruction = example["instruction"]
    original = example["code"] # type: str
    modified = example["modified_code"] # type: str
    
    original = "\n".join([x.strip() for x in original.splitlines()])
    modified = "\n".join([x.strip() for x in modified.splitlines()])
    
    original, modified = original.strip(), modified.strip()
    
    # First rule: original code cannot be identical to modification
    if original == modified:
        return {
            **example,
            "high_quality": False,
            "filter_reason": "identical_code"
        }
    
    # Second rule: code cannot be shorter than `mininum_lines`
    original_lines = original.splitlines()
    modified_lines = modified.splitlines()
    if len(original_lines) < mininum_lines:
        return {
            **example,
            "high_quality": False,
            "filter_reason": "original_code_too_short"
        }
    
    if len(modified_lines) < mininum_lines:
        return {
            **example,
            "high_quality": False,
            "filter_reason": "modified_code_too_short"
        }
        
    
    # Third rule: fraction of modified part in original code
    #              should be no greater than `threshold`
    seq_matcher = difflib.SequenceMatcher(a=original_lines, b=modified_lines)
    match_blocks = seq_matcher.get_matching_blocks()
    
    num_matched_lines = sum([x.size for x in match_blocks])
    diff_frac = 1 - (num_matched_lines / len(original_lines))
    
    if diff_frac > diff_frac_threshold:
        return {
            **example,
            "high_quality": False,
            "filter_reason": "diff_frac_less_than_threshold"
        }
        
    # Fourth rule: 
    PROMPT_FORMPT = r"""### Instruction
{instruction}

### OLD CODE
```
{old_code}
```

### NEW CODE
```
{new_code}

Your task is to judge if the code modification is aligned with the instruction.
1. ANSWER `YES` or `NO`
2. DO NOT include other content.
"""
    
    COT_PROMPT_FORMPT = r"""### Instruction
{instruction}

### OLD CODE
```
{old_code}
```

### NEW CODE
```
{new_code}

Your task is to judge if the code modification is aligned with the instruction.

Let think step by step and answer `YES` or `NO` in the end.

### FORMAT
Thoughts: write your thoughts Here.
ANSWER: YES / NO
"""

    if use_cot_prompt:
        query = COT_PROMPT_FORMPT.format(instruction=example["instruction"],
                                         old_code=example["code"],
                                         new_code=example["modified_code"])
    else:
        query = PROMPT_FORMPT.format(instruction=example["instruction"],
                                     old_code=example["code"],
                                     new_code=example["modified_code"])

    messages = [{"role": "user", "content": query}]
    response = client.chat.completions.create(
        messages=messages,
        model=model_name_or_path,
        **sampling_params
    )
    
    text = response.choices[0].message.content # type: str
    text = text.lower()
    
    if use_cot_prompt:
        cond = re.search(r"answer\s*[:]\s*(no|n)\b", text)
    else:
        cond = re.search(r"no", text)
    if cond:
        return {
            **example,
            "high_quality": False,
            "filter_reason": "misaligned_with_instruction"
        }
    
    return {
        **example,
        "high_quality": True,
        "filter_reason": ""
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
    sampling_params = dict(
        n=1,
        temperature=0.0,
        max_tokens=args.max_model_len,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    # Load source dataset from json file
    src_data = load_dataset("json", data_files=[args.json_path], split="train")
    
    # Init output path
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
        
    # Load cached data
    num_generated_samples, total_num = load_cached_data(args)
    logging.info(f"Load {total_num} from {args.output_path}.")
    
    src_data = src_data.filter(
        lambda x: num_generated_samples[x["task_id"]] < 1
    )
    print(src_data)
    
    total_generation_num = len(src_data)
    logging.info(f"Need to filter {total_generation_num} new samples across {len(num_generated_samples)} tasks")
    
    if total_generation_num == 0:
        logging.info("All samples ready filtered. Exiting.")

    def init_processor(worker_id=None):
        return {"worker_id": worker_id}
    
    def process_task(processor, task):
        return process_single_example(task, client, model_name_or_path, sampling_params, use_cot_prompt=args.use_cot_prompt)
    
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
    engine.process_tasks(src_data)
    
    logging.info(f"Processing completed")
    