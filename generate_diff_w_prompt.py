import os
import re
import json
import openai
import logging
import argparse
import functools

from typing import Dict, Any, List, Literal
from utils import write_jsonl
from task_processing import TaskProcessingEngine


def generate_prompt_debug(data: Dict[str, Any], type='zero'):
    error_code = data['incorrect_solutions']
    code_language = data['code_language']
    public_tests_input = data['public_tests_input']
    public_tests_output = data['public_tests_output']
    if type == 'zero':
        prompt = f'''
### Instruction:
Please correct the errors in the buggy code snippet below, ensuring that your corrected code adheres to the specified programming language syntax and logic requirements. Validate your solution against the provided test cases to ensure its accuracy. Note that your solution should strictly consist of the corrected code only.
'''
    elif type == 'three':
        with open(f'CodeEditorBench/few_shot_prompt/close/prompt_debug.txt', 'r') as f:
            prompt = f.read()
    else:
        raise NotImplementedError
    prompt += f'''
### Question:
Below is the {code_language} buggy code:
{error_code}

Correct the code and ensure it passes the following test case:
Input: {public_tests_input}
Output: {public_tests_output}

        '''
    return prompt


def generate_prompt_polishment(data, type='zero'):
    source_code = data['source_code']
    public_tests_input = data['public_tests_input']
    public_tests_output = data['public_tests_output']
    if type == 'zero':
        prompt = f'''
### Instruction:
Please optimize the given code snippet to enhance its execution efficiency and reduce memory usage, while ensuring the accuracy of the code remains unaffected. Validate your solution against the provided test cases to ensure its accuracy. Note that your submission should strictly consist of the optimized code only.
'''
    elif type == 'three':
        with open(f'CodeEditorBench/few_shot_prompt/close/prompt_polishment.txt', 'r') as f:
            prompt = f.read()
    else:
        raise NotImplementedError
    prompt += f'''
### Question:
Below is the source code snippet that needs optimization:
{source_code}

Optimize the code and ensure it passes the following test case:
Input: {public_tests_input}
Output: {public_tests_output}

    '''
    return prompt



def generate_prompt_switch(data, type='zero'):
    similar_source_code = data['similar_source_code']
    public_similar_tests_input = data['public_similar_tests_input']
    public_similar_tests_output = data['public_similar_tests_output']
    public_target_tests_input = data['public_target_tests_input']
    public_target_tests_output = data['public_target_tests_output']
    if type == 'zero':
        prompt = f'''
### Instruction:
Please modify the given code snippet to implement a new function that is related to the original function implemented by the code. Ensure your modified code adheres to the programming language's syntax and logic requirements. Validate your solution against the provided test cases to ensure its accuracy. Your submission should strictly consist of the target code only.
'''
    elif type == 'three':
        with open(f'few_shot_prompt/close/prompt_switch.txt', 'r') as f:
            prompt = f.read()
    else:
        raise NotImplementedError
    prompt += f'''
### Question:
Below is the code snippet that implements a specific function:
{similar_source_code}

It currently performs the operation:
Input: {public_similar_tests_input}
Output: {public_similar_tests_output}

You are required to modify this code to implement a new function that is related to the original one, as detailed below:
Input: {public_target_tests_input}
Output: {public_target_tests_output}

Ensure your modified code passes the provided test case.

    '''
    return prompt


def generate_cot_prompt_debug(data):
    error_code = data['incorrect_solutions']
    code_language = data['code_language']
    public_tests_input = data['public_tests_input']
    public_tests_output = data['public_tests_output']
    prompt = f'''
### Instruction:
Please correct the buggy {code_language} code snippet below by using a step-by-step reasoning process, similar to how a large language model thinks. Follow these steps:

1. **Error Identification**: First, let's identify the errors in the code. Briefly describe what each error is and why it's a problem.

2. **Logical Reasoning**: For each identified error, use logical reasoning to explain how it impacts the code's functionality. This helps in understanding the nature of the problem.

3. **Solution Strategy**: Outline a strategy to correct the errors. This includes the steps you will take and the programming concepts you will apply to ensure the code meets the logic and syntax requirements of {code_language}.

4. **Implementation**: Provide the corrected code snippet based on your strategy. Make sure the code is syntactically correct and logically sound.

5. **Validation**: Validate your corrected code against the provided test case. Explain how the input leads to the expected output with your corrections.

### Question:
Below is the {code_language} buggy code snippet to correct:
{error_code}

Ensure your corrected code works correctly by testing it with this input and verifying it produces the expected output.
- Input: {public_tests_input}
- Output: {public_tests_output}

### Answer:

1. **Error Identification**: [Your identification and description of the errors] (Limit your response to 150 words)

2. **Logical Reasoning**: [Your reasoning on how each error affects the code] (Limit your response to 150 words)

3. **Solution Strategy**: [Outline of your strategy to correct the errors] (Limit your response to 150 words)

4. **Implementation**: [Your corrected code snippet]

5. **Validation**: [Explanation on how the input leads to the expected output with your corrections]
'''
    return prompt


def generate_cot_prompt_polishment(data):
    source_code = data['source_code']
    public_tests_input = data['public_tests_input']
    public_tests_output = data['public_tests_output']
    prompt = f'''
### Instruction:
Optimize the given code snippet for better execution efficiency and reduced memory usage by employing a step-by-step reasoning process. Follow these steps:

1. **Performance and Memory Analysis**: Start by thoroughly analyzing the source code to identify any performance bottlenecks and areas with excessive memory usage. Consider both the complexity of algorithms and the data structures used.

2. **Identify Optimization Opportunities**: Highlight specific lines of code or constructs that are inefficient. Explain why they are problematic and how they contribute to lower performance or higher memory usage.

3. **Optimization Strategies**: Develop a detailed plan for optimizing the code. This should include algorithmic improvements, data structure optimizations, and any coding practices that can enhance efficiency. Explain how each change is expected to improve performance or reduce memory usage.

4. **Implementation**: Apply your optimization strategies to modify the source code. Ensure that these changes maintain the code's accuracy and functionality.

5. **Validation**: Validate the optimized code using the provided test case. Detail how the optimizations impact the execution efficiency and memory usage in the context of the test case.

### Question:
Below is the code snippet that needs optimization:
{source_code}

Use this input to test your optimized code and confirm it produces the expected output.
- Input: {public_tests_input}
- Output: {public_tests_output}

### Answer:

1. **Performance and Memory Analysis**: [Your analysis of the code's performance and memory usage] (Limit your response to 150 words)

2. **Identify Optimization Opportunities**: [Specific inefficiencies and their impacts] (Limit your response to 150 words)

3. **Optimization Strategies**: [Your detailed plan for optimizing the code] (Limit your response to 150 words)

4. **Implementation**: [Your optimized code snippet]

5. **Validation**: [Explanation of how the optimizations improved efficiency or reduced memory usage]
'''
    return prompt


def generate_cot_prompt_switch(data):
    similar_source_code = data['similar_source_code']
    public_similar_tests_input = data['public_similar_tests_input']
    public_similar_tests_output = data['public_similar_tests_output']
    public_target_tests_input = data['public_target_tests_input']
    public_target_tests_output = data['public_target_tests_output']
    prompt = f'''
### Instruction:
Modify the following code snippet to implement a new specified function, using a step-by-step reasoning process. The process involves analyzing the current code's functionality, identifying necessary changes for the new requirements, and implementing these changes effectively. Follow these steps:

1. **Current Functionality Analysis**: Begin by thoroughly understanding and analyzing the functionality and logic of the current code. Identify the key operations and how they contribute to producing the given output.

2. **Identify New Requirements**: Compare the current functionality with the new function requirements. Highlight the differences in input and output specifications and any additional functionalities that need to be incorporated.

3. **Determine Required Changes**: Based on the analysis, specify the changes needed to adapt the existing code to the new requirements. This includes altering existing logic, adding new logic or functions, and removing unnecessary parts.

4. **Modification Strategy**: Develop a detailed plan for making these changes. This should outline any new algorithms, data structures, or coding constructs required to implement the new functionality while maintaining code efficiency and readability.

5. **Implementation**: Modify the source code according to your strategy. Ensure that the new code accurately implements the desired function and adheres to good coding practices.

6. **Validation**: Validate your modified code using the provided test case. Explain how the new input leads to the expected output, demonstrating the effectiveness of your modifications.

### Question:
Below is the code snippet that needs modification:
{similar_source_code}

It currently performs this operation:
- Input: {public_similar_tests_input}
- Output: {public_similar_tests_output}

Modify the code to achieve the following new functionality:
- Input: {public_target_tests_input}
- Output: {public_target_tests_output}

### Answer:

1. **Current Functionality Analysis**: [Your analysis of the source code's current functionality] (Limit your response to 150 words)

2. **Identify New Requirements**: [The differences between current and new requirements] (Limit your response to 150 words)

3. **Determine Required Changes**: [Specific changes needed for the new functionality] (Limit your response to 150 words)

4. **Modification Strategy**: [Your plan for implementing these changes] (Limit your response to 150 words)

5. **Implementation**: [The modified code snippet]

6. **Validation**: [Explanation of how the modifications meet the new requirements]
'''
    return prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the specified model and dataset.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct", help="Base model to use")
    parser.add_argument("--base_url", type=str, required=True, help="IP address for openai-compatible server.")
    parser.add_argument("--use_diff", action="store_true",
                        help="Whether to use diff-format when generating code edition.")
    
    parser.add_argument("--num_of_sequences", type=int, default=1, help="Number of sequences to generate")
    parser.add_argument("--frequency_penalty", type=float, default=0.0, help="Frequency penalty")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty")
    parser.add_argument("--prompt_type", default="zero", type=str, help="Type of Prompts, zero or three.")
        
    parser.add_argument("--dataset", default="debug", type=str, 
                         choices=["debug", "polishment", "switch"], help="Name of dataset.")
    parser.add_argument("--input_data_dir", default="Input.jsonl", type=str, help="Path to input data.")
    parser.add_argument("--output_data_dir", default="Output.jsonl", type=str, help="Path to output data.")
    parser.add_argument("--start_idx", default=0, type=int, help="Start index for data processing.")
    parser.add_argument("--end_idx", default=-1, type=int, help="End index for data processing.")

    
    parser.add_argument("--num_proc", type=int, default=128, help="Maximum number of processes for processing")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Maximum length for LLM.")

    args = parser.parse_args()
    return args


def get_sampling_params(args):
    """Configure and return sampling parameters based on arguments."""
    if args.num_of_sequences == 1:        
        return dict(
            n=args.num_of_sequences, 
            temperature=0.0, 
            max_tokens=8192,
            frequency_penalty=args.frequency_penalty, 
            presence_penalty=args.presence_penalty,
        )
    else:
        return dict(
            n=args.num_of_sequences, 
            temperature=0.4, 
            top_p=0.95, 
            max_tokens=8192,
            frequency_penalty=args.frequency_penalty, 
            presence_penalty=args.presence_penalty,
        )


GENERATE_DIFF_FORMAT = r"""
### Format
To save tokens, generate a compact diff if appropriate, otherwise provide complete code.
The diff should be compact, include the insertion / deletion / replacement.

### Example
Instruction: Modify the function to return both the longest word and its length.
Old code:
```
def find_longest_word(sentence):
    words = sentence.split()
    longest_word = ""
    max_length = 0
    for word in words:
        if len(word) > max_length:
            longest_word = word
            max_length = len(word)
    return longest_word
```
Diff:
```diff
-    return longest_word
+    return longest_word, max_length
```
"""


def output_example(example: Dict[str, Any],
                   dataset_name: Literal["debug", "polishment", "switch"],
                   output: str, 
                   prompt_tokens: int,
                   output_tokens: int):
    new_data = {
        "problem_id": example["idx"],
        "completion_id": 0,
        "prompt": example["prompt"],
        "code": output,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens
    }
    if dataset_name == "debug":
        new_data.update({
            "language": example["code_language"],
            "error_type": example["type"],
            "difficulty": example["difficulty"]
        })
    elif dataset_name == "polishment":
        new_data.update({
            "language": example["source_lang"],
            "difficulty": example["difficulty"]
        })
    elif dataset_name == "switch":
        new_data.update({
            "language": example["language"],
            "pair": example["pair_id"]
        })
    
    return new_data


def process_single_example(example: Dict[str, Any],
                           client: openai.Client,
                           model_name_or_path: str, 
                           dataset_name: Literal["debug", "polishment", "switch"],
                           sampling_params: Dict[str, Any]):
    messages = [{"role": "user", "content": example["prompt"]}]
    response = client.chat.completions.create(
        messages=messages,
        model=model_name_or_path,
        **sampling_params
    )
    
    # We only support num_generation=1 for now.
    output = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    # diff_code_pattern = r'```\s*diff\s*\n(.*?)\n\s*```'
    # diff_matches = re.findall(diff_code_pattern, text, re.DOTALL)
    return output_example(example, dataset_name, output, prompt_tokens, output_tokens)



if __name__ == "__main__":
    # Set up basic logging configuration
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(process)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    args = parse_args()
    
    client = openai.Client(
        base_url=args.base_url,
        api_key="EMPTY"
    )    

    model_name_or_path = args.model 

    # Set hyperparameters
    sampling_params = get_sampling_params(args) # type: dict

    # Input file name
    input_data_path = args.input_data_dir + f"code_{args.dataset}_primary.jsonl"
    model_name = args.model.split('/')[-1].replace('-', '_').replace('.', '_')\
    
    # Output file name
    # Make sure the directory exists
    end = args.end_idx if args.end_idx != -1 else "end"
    if args.prompt_type == "zero":
        output_data_path = args.output_data_dir + f"code_{args.dataset}/{model_name}_{args.start_idx}_{end}.jsonl"
    elif args.prompt_type == "three":
        output_data_path = args.output_data_dir + f"code_{args.dataset}/Few_Shot_{model_name}_{args.start_idx}_{end}.jsonl"
    elif args.prompt_type == "cot":
        output_data_path = args.output_data_dir + f"code_{args.dataset}/Cot_{model_name}_{args.start_idx}_{end}.jsonl"
    else:
        raise ValueError("Invalid prompt type.")

    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

    print(f"Input file: {input_data_path}")
    print(f"Output file: {output_data_path}")
    

    meta_data_flag = False
    if os.path.exists(output_data_path):
        with open(output_data_path, "r") as f:
            objs = [json.loads(line) for line in f.readlines()]
            line_count = len(objs)
            print(f"Output file exists. Appending to {output_data_path}. Line count: {line_count}")
        if line_count >= 1:
            meta_data_flag = True
            objs = objs[1:]
        seen_task_idx = set([x["problem_id"] for x in objs])
    else:
        seen_task_idx = set()
        
    
    if args.end_idx == -1:
        args.end_idx = None
        

    with open(input_data_path, "r") as f:
        dataset = [json.loads(line) for line in f.readlines()]

        
    # filtering
    dataset = [x for x in dataset if x["idx"] not in seen_task_idx]
    
    # Generate prompt
    if args.prompt_type in ["zero", "three"]:
        prompt_func = globals()[f"generate_prompt_{args.dataset}"]
        prompt_func = functools.partial(prompt_func, type=args.prompt_type)
    else:
        prompt_func = globals()[f"generate_cot_prompt_{args.dataset}"]

    dataset = [{**x, "prompt": prompt_func(x)} for x in dataset]
    # dataset = dataset.map(lambda x: {"prompt": prompt_func(x)}, num_proc=args.num_proc)
    if args.use_diff:
        # dataset = dataset.map(lambda x: {"prompt": x["prompt"] + GENERATE_DIFF_FORMAT},
        #                       num_proc=args.num_proc)
        dataset = [{**x, "prompt": x["prompt"] + GENERATE_DIFF_FORMAT} for x in dataset]


    if not meta_data_flag:
        meta_data_flag = True
        meta_data = {
            "model": args.model,
            # "model_size": model.num_parameters(),
            "model_url": f"https://huggingface.co/{args.model}",
            # "do_sample": generation_config["do_sample"],
            "num_output": sampling_params.get("n", 1),
            "temperature": sampling_params["temperature"],
            "top_p": sampling_params.get("top_p", 0.0),
            "top_k": sampling_params.get("top_k", 0.0)
        }
        write_jsonl(output_data_path, [meta_data], append=True)
    
    

    
    def init_processor(worker_id=None):
        return {
            "worker_id": worker_id
        }
    
    def process_task(processor, task):
        return process_single_example(task, 
                                      client, 
                                      model_name_or_path, 
                                      args.dataset,
                                      sampling_params)
    
    def handle_results(results):
        write_jsonl(output_data_path, results, append=True)
        
    
    
    engine = TaskProcessingEngine(
        processor_initializer=init_processor,
        processor_args={},
        process_func=process_task,
        result_handler=handle_results,
        num_workers=args.num_proc
    )
    
    logging.info("Creating sample tasks...")
    
    logging.info("Starting task processing...")
    engine.process_tasks(dataset)
    
    logging.info(f"Processing completed")
