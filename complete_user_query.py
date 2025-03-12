import logging
import argparse
import time
from tqdm import tqdm
import os
import json

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from vllm import LLM, SamplingParams

from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from collections import defaultdict

from task_processing import TaskProcessingEngine
from utils import write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with vLLM on Qwen model using a dataset")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-72B-Instruct", help="Base model to use")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--num_of_sequences", type=int, default=1, help="Number of sequences to generate")
    parser.add_argument("--frequency_penalty", type=float, default=0.0, help="Frequency penalty")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty")
    parser.add_argument("--swap_space", type=int, default=4, help="Swap space in GB")
    parser.add_argument("--dataset_path", type=str, required=True, help="Input dataset file (huggingface.datasets compatible format).")
    parser.add_argument("--split", type=str, default="train", help="Input data split.")
    parser.add_argument("--output_path", type=str, required=True, help="Output file")
    parser.add_argument("--batch_size", type=int, default=128, help="Maximum number of processes for processing")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Maximum length for LLM.")
    return parser.parse_args()


def format_chat_messages(query1: str, response1: str, query2: str) -> List[Dict[str, str]]:
    """Format chat messages for the model."""
    messages = [
        {"role": "user", "content": query1},
        {"role": "assistant", "content": response1},
        {"role": "user", "content": query2}
    ]
    return messages



def initialize_model(args):
    """Initialize and return the LLM model."""
    max_model_len = 8192  # Adjust based on your model's capabilities
    
    print("Loading model...")
    llm = LLM(
        model=args.base_model, 
        trust_remote_code=True, 
        tensor_parallel_size=args.num_gpus, 
        max_model_len=max_model_len, 
        swap_space=args.swap_space
    )
    print("Model loaded.")
    return llm


def get_sampling_params(args):
    """Configure and return sampling parameters based on arguments."""
    if args.num_of_sequences == 1:        
        return SamplingParams(
            n=args.num_of_sequences, 
            temperature=0.0, 
            max_tokens=2048,
            frequency_penalty=args.frequency_penalty, 
            presence_penalty=args.presence_penalty,
            best_of=1,
        )
    else:
        return SamplingParams(
            n=args.num_of_sequences, 
            temperature=0.4, 
            top_p=0.95, 
            max_tokens=2048,
            frequency_penalty=args.frequency_penalty, 
            presence_penalty=args.presence_penalty,
            best_of=1,
        )


def load_source_dataset(args):
    """Loading dataset in huggingface.datasets compatible format."""
    src_data = load_dataset(args.dataset_path, split=args.split)
    
    dataset_name = args.dataset_path.split("/")[-1]
    src_data = src_data.map(
        lambda x, idx: {"task_id": f"{dataset_name}/{idx}"},
        with_indices=True
    )
    
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


def process_single_example(example: Dict[str, Any],
                           llm: LLM,
                           sampling_params: SamplingParams,
                           prompt: str):
    query1 = example["instruction"]
    response1 = example["output"]
    query2 = prompt
    
    messages = format_chat_messages(query1, response1, query2)
    
    # First, let the LLM complete the user query
    output = llm.chat(
        messages,
        sampling_params,
        continue_final_message=True,
        add_generation_prompt=False
    )
    
    query2 = output.outputs[0].text
    messages = format_chat_messages(query1, response1, query2)
    
    # Second, let the LLM modify the code
    output = llm.chat(
        messages,
        sampling_params
    )
    response2 = output.outputs[0].text 
    
    result = {
        "task_id": example["task_id"],
        "query1": query1,
        "response1": response1,
        "query2": query2,
        "response2": response2
    }
    return result
    

def process_batch_examples(examples: List[Dict[str, Any]],
                           llm: LLM,
                           sampling_params: SamplingParams,
                           prompt: str):
    batch_messages_1st = [
        format_chat_messages(
            example["instruction"],
            example["output"],
            prompt
        )
        for example in examples
    ]
    
    batch_outputs_1st = llm.chat(
        batch_messages_1st,
        sampling_params,
        continue_final_message=True,
        add_generation_prompt=False,
    )
    
    batch_messages_2nd = [
        format_chat_messages(
            example["instruction"],
            example["output"],
            prompt + batch_outputs_1st[i].outputs[0].text
        )
        for i, example in enumerate(examples)
    ]
    batch_outputs_2nd = llm.chat(
        batch_messages_2nd,
        sampling_params,
    )
    
    batch_results = [
        {
            "task_id": example["task_id"],
            "query1": example["instruction"],
            "response1": example["output"],
            "query2": prompt + batch_outputs_1st[i].outputs[0].text,
            "response2": batch_outputs_2nd[i].outputs[0].text
        }
        for i, example in enumerate(examples)
    ]
    return batch_results


if __name__ == "__main__":
    # Set up basic logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    args = parse_args()
    
    # Initialize model
    model = initialize_model(args)
    
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
    # Sort tasks 
    all_tasks.sort(key=lambda x: len(x["instruction"]) + len(x["output"]))
    
    
    PROMPT = r""""You are doing great but there are a few minor issues. 
Please modify the above code according to the following requirements:
"""

    for idx in tqdm(range(0, len(src_data), args.batch_size)):
        batch_examples = all_tasks[idx:idx+args.batch_size]

        batch_results = process_batch_examples(
            batch_examples,
            model,
            sampling_params,
            PROMPT
        )
        
        write_jsonl(args.output_path, batch_results, append=True)
        
    # engine = TaskProcessingEngine(
    #     processor_initializer=init_processor,
    #     processor_args={},
    #     process_func=lambda processor, task: process_single_example(task, model, sampling_params, PROMPT),
    #     result_handler=lambda results: handle_results(results, args.output_path),
    #     num_workers=args.num_proc
    # )
    
    # engine.process_tasks(all_tasks)
    logging.info("Processing completed")
    