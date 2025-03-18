#%%
# Load dataset and tokenizer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


subset = "package"


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")# dataset = load_dataset("data/opc-sft-stage2/evol_instruct", split="train")"train")
dataset = load_dataset(f"data/opc-sft-stage2/{subset}_instruct", split="train")
# %%
def get_num_tokens(text: str, tokenizer: AutoTokenizer):
    return len(tokenizer(text).input_ids)

dataset = dataset.map(
    lambda x: {
        "num_instruction_tokens": get_num_tokens(x["instruction"], tokenizer),
        "num_output_tokens": get_num_tokens(x["output"], tokenizer)
    },
    num_proc=128
)
# %%
instruction_tokens = dataset["num_instruction_tokens"]
output_tokens = dataset["num_output_tokens"]
#%%
import numpy as np 

print(f"[{subset}] mean tokens of instruction: {np.mean(instruction_tokens)}")
print(f"[{subset}] mean tokens of output: {np.mean(output_tokens)}")
# %%
print(f"[{subset}] p99 tokens of instruction: {np.percentile(instruction_tokens, 99)}")
print(f"[{subset}] p99 tokens of output: {np.percentile(output_tokens, 99)}")