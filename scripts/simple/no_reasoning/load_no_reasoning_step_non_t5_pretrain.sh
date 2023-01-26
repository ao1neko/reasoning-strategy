#!/bin/bash

project_dir=$1
load_model_dir=$2
save_model_dir=$3

for depth in `seq 1 12`
do
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python "${project_dir}/src/main.py" \
        --device=cuda \
        --architecture_name=no_reasoning_step \
        --train_dataset_name="${project_dir}/numerical_datasets/data/depth_1_5_distractor_3/train_no_reasoning_step.jsonl" \
        --valid_dataset_name="${project_dir}/numerical_datasets/data/depth_1_5_distractor_3/valid_no_reasoning_step.jsonl" \
        --test_dataset_name="${project_dir}/numerical_datasets/data/depth_${depth}_distractor_3/test_no_reasoning_step.jsonl" \
        --model_name=t5 \
        --load_model_dir="${load_model_dir}" \
        --train=false \
        --predict=true \
        --train_epochs=1 \
        --eval_steps=100  \
        --output_dir="${save_model_dir}/reasoning_model/42/depth_${depth}_distractor_3_by_depth_1_5_distractor_3/no_reasoning_step_non_t5_pretrain" \
        --run_dir="42/depth_${depth}_distractor_3_by_depth_1_5_distractor_3/no_reasoning_step_non_t5_pretrain" \
        --batch_size=16 \
        --load_tokenizer="t5-base"

done

