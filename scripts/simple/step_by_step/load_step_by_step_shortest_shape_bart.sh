#!/bin/bash

project_dir=$1
load_model_dir=$2
save_model_dir=$3

for depth in `seq 1 12`
do
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python "${project_dir}/src/main.py" \
        --device=cuda \
        --architecture_name=step_by_step \
        --train_dataset_name="${project_dir}/numerical_datasets/data/depth_1_5_distractor_3/train_step_by_step_shortest.jsonl" \
        --valid_dataset_name="${project_dir}/numerical_datasets/data/depth_1_5_distractor_3/valid_step_by_step_shortest.jsonl" \
        --test_dataset_name="${project_dir}/numerical_datasets/data/depth_${depth}_distractor_3/test_step_by_step_shortest.jsonl" \
        --model_name=shape_bart \
        --load_model_dir="${load_model_dir}" \
        --train=false \
        --predict=true \
        --train_epochs=1 \
        --eval_steps=100  \
        --output_dir="${save_model_dir}/reasoning_model/42/depth_${depth}_distractor_3_by_depth_1_5_distractor_3/shape_bart/step_by_step_shortest" \
        --run_dir="42/depth_${depth}_distractor_3_by_depth_1_5_distractor_3/shape_bart/step_by_step_shortest" \
        --batch_size=8 \
        --load_tokenizer="facebook/bart-base"

done

