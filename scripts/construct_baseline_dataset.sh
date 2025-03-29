
python construct_dataset.py \
    --data_files \
    outputs/opc-sft-stage2/education_qwen72b_high_quality.json \
    outputs/opc-sft-stage2/evol_qwen72b_high_quality.json \
    outputs/opc-sft-stage2/mceval_qwen72b_high_quality.json \
    outputs/opc-sft-stage2/package_qwen72b_high_quality.json \
    --output_path outputs/opc-sft-stage2/baseline_data.json \
    --mode baseline \
    --num_proc 128 