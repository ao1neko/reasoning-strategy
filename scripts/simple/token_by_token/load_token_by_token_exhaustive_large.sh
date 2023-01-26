#!/bin/bash

project_dir=$1
load_model_dir=$2
save_model_dir=$3

for depth in `seq 1 12`
do
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python "${project_dir}/src/main.py" \
        --device=cuda \
        --architecture_name=token_by_token \
        --train_dataset_name="${project_dir}/numerical_datasets/data/depth_1_5_distractor_3/train_token_by_token_exhaustive.jsonl" \
        --valid_dataset_name="${project_dir}/numerical_datasets/data/depth_1_5_distractor_3/valid_token_by_token_exhaustive.jsonl" \
        --test_dataset_name="${project_dir}/numerical_datasets/data/depth_${depth}_distractor_3/test_token_by_token_exhaustive.jsonl" \
        --model_name=t5 \
        --load_model_dir="${load_model_dir}" \
        --train=false \
        --predict=true \
        --train_epochs=1 \
        --eval_steps=100 \
        --output_dir="${save_model_dir}/reasoning_model/42/depth_${depth}_distractor_3_by_depth_1_5_distractor_3/t5_large/token_by_token_exhaustive" \
        --run_dir="42/depth_${depth}_distractor_3_by_depth_1_5_distractor_3/t5_large/token_by_token_exhaustive" \
        --batch_size=8 \
        --load_tokenizer="t5-large"

done
0
