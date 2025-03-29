base_url=http://210.28.135.36:8930/v1
base_models=(
    # "deepseek-ai/deepseek-coder-33b-instruct"
    # "Qwen/Qwen2.5-Coder-7B-Instruct"
    # "Qwen/Qwen2.5-Coder-32B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
)
# datasets=("debug" "translate" "polishment" "switch")
datasets=("debug" "polishment" "switch")

for base_model in "${base_models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Running inference with base_model: $base_model and dataset: $dataset"
        HF_HUB_OFFLINE=1 python generate_diff_w_prompt.py \
            --use_diff \
            --model "$base_model" \
            --base_url "$base_url" \
            --dataset "$dataset" \
            --input_data_dir "CodeEditorBench/data/" \
            --output_data_dir "CodeEditorBench/greedy_result_w_diff/" \
            --num_of_sequences 1 \
            --prompt_type "zero" \
            --num_proc 128 \
            --start_idx 0 \
            --end_idx -1
    done
done