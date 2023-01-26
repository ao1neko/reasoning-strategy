#!/bin/bash

project_dir=$1
load_model_dir=$2
save_model_dir=$3


CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python "${project_dir}/src/main.py" \
        --device=cuda \
        --architecture_name=no_reasoning_step \
        --train_dataset_name="${project_dir}/numerical_datasets/data/depth_1_5_distractor_3/train_no_reasoning_step.jsonl" \
        --valid_dataset_name="${project_dir}/numerical_datasets/data/depth_1_5_distractor_3/valid_no_reasoning_step.jsonl" \
        --test_dataset_name="${project_dir}/numerical_datasets/data/depth_1_5_distractor_3/test_no_reasoning_step.jsonl" \
        --model_name=t5 \
        --load_model_dir="${load_model_dir}" \
        --train=true \
        --seed="42" \
        --predict=false \
        --train_epochs=30 \
        --eval_steps=1000  \
        --save_steps=5000 \
        --output_dir="${save_model_dir}/reasoning_model/42/depth_1_5_distractor_3/no_reasoning_step_caliculum" \
        --run_dir="42/depth_1_5_distractor_3/no_reasoning_step_caliculum" \
        --batch_size=16 \
        --learning_rate=0.0001 \
        --load_tokenizer="t5-base"




