#!/bin/bash

project_dir=$1
load_model_dir=$2
save_model_dir=$3


CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python "${project_dir}/src/main.py" \
        --device=cuda \
        --architecture_name=step_by_step \
        --train_dataset_name="${project_dir}/numerical_datasets/data/depth_1_5_distractor_3/train_NL_step_by_step_exhaustive.jsonl" \
        --valid_dataset_name="${project_dir}/numerical_datasets/data/depth_1_5_distractor_3/valid_NL_step_by_step_exhaustive.jsonl" \
        --test_dataset_name="${project_dir}/numerical_datasets/data/depth_1_5_distractor_3/test_NL_step_by_step_exhaustive.jsonl" \
        --model_name=t5 \
        --load_model_dir="${load_model_dir}" \
        --train=true \
        --seed="42" \
        --predict=false \
        --train_epochs=2000 \
        --eval_steps=10000  \
        --save_steps=10000 \
        --output_dir="${save_model_dir}/reasoning_model/42/depth_1_5_distractor_3/t5_large/NL_step_by_step_exhaustive" \
        --run_dir="42/depth_1_5_distractor_3/t5_large/NL_step_by_step_exhaustive" \
        --batch_size=2 \
        --learning_rate=0.0001 \
        --load_tokenizer="t5-large"


